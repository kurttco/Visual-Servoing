#!/usr/bin/env python3
"""
safe_vs_controller.py  — IBVS-MPC edition
------------------------------------------
Visual servoing controller with finite-horizon MPC and square-bypass
obstacle avoidance.

The servoing law is a full IBVS-MPC: at each tick the QP is solved over a
horizon of N steps, the optimal first command is applied, and the problem is
re-solved at the next step with a fresh measurement (receding horizon).

All FSM logic, obstacle avoidance, and CSV logging are unchanged.

Requirements:
  pip install qpsolvers[osqp] --break-system-packages
"""

import csv
import os
import time as time_mod
from enum import Enum

import numpy as np
import scipy.sparse as sp

try:
    from qpsolvers import solve_qp
except ImportError as exc:
    raise SystemExit(
        'qpsolvers not found. Install with:\n'
        '  pip install qpsolvers[osqp] --break-system-packages'
    ) from exc

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from puzzlebot_mc2.msg import TargetFeatures, ObstacleFeatures


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# ═══════════════════════════════════════════════════════════════════════════
# IBVS-MPC solver
# ═══════════════════════════════════════════════════════════════════════════

class IBVS_MPC:
    """
    Finite-horizon Image-Based Visual Servoing MPC.

    State:   s = [u_c, sqrt_A]   (image features)
    Control: u = [v, omega]       (diff-drive velocities)

    Cost (Eq. 2 in the report):
        J = sum_{k=1}^{N}  ds_k' Q ds_k  +  u_k' R u_k  +  Du_k' S Du_k

    Constraints:
        |v|     <= v_max
        |omega| <= omega_max
        |v|/k_lin + |omega|/k_ang <= max_cmd_couple   (bridge coupling)

    The QP is solved with OSQP via qpsolvers at every control tick.
    The first element of the optimal sequence is applied (receding horizon).
    """

    def __init__(self, N, dt, f, cx, sqrt_A_real,
                 v_max, omega_max, k_lin, k_ang, max_cmd_couple,
                 Q, R, S):
        self.N            = int(N)
        self.dt           = float(dt)
        self.f            = float(f)
        self.cx           = float(cx)
        self.sqrt_A_real  = float(sqrt_A_real)
        self.v_max        = float(v_max)
        self.omega_max    = float(omega_max)
        self.k_lin        = float(k_lin)
        self.k_ang        = float(k_ang)
        self.max_couple   = float(max_cmd_couple)
        self.Q            = np.asarray(Q, dtype=float).reshape(2, 2)
        self.R            = np.asarray(R, dtype=float).reshape(2, 2)
        self.S            = np.asarray(S, dtype=float).reshape(2, 2)

        n = 2 * self.N

        # ── slew penalty matrix  D  ───────────────────────────────────────
        D = np.zeros((n, n))
        I2 = np.eye(2)
        for k in range(self.N):
            D[2*k:2*k+2, 2*k:2*k+2] = I2
            if k > 0:
                D[2*k:2*k+2, 2*(k-1):2*(k-1)+2] = -I2
        I_S = np.kron(np.eye(self.N), self.S)
        I_R = np.kron(np.eye(self.N), self.R)
        # base Hessian (only R and S parts; Q part added per-step)
        self._H_base = 2.0 * (I_R + D.T @ I_S @ D)
        self._D      = D
        self._I_S    = I_S

        # ── coupled actuator constraints  G u <= h ────────────────────────
        a = 1.0 / self.k_lin
        b = 1.0 / self.k_ang
        rows, rhs = [], []
        for k in range(self.N):
            for sv, so in [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]:
                row = np.zeros(n)
                row[2*k]   = sv * a
                row[2*k+1] = so * b
                rows.append(row)
                rhs.append(self.max_couple)
        self._G = np.array(rows)
        self._h = np.array(rhs)

        # ── box constraints ───────────────────────────────────────────────
        self._lb = np.tile([-self.v_max,    -self.omega_max], self.N)
        self._ub = np.tile([ self.v_max,     self.omega_max], self.N)

    # ── helpers ──────────────────────────────────────────────────────────

    def _jacobian(self, u_c, sqrt_A):
        """
        2×2 image Jacobian  L  for a differential-drive robot.
        Maps [v, omega] -> d/dt [u_c, sqrt_A].

            L = [ e_u/Z ,  f + e_u^2/f  ]
                [ sA/Z  ,  0            ]

        Depth Z estimated from apparent area using the pinhole model (Eq. 2).
        """
        e_u = u_c - self.cx
        # depth estimate; clamped to avoid singularity
        Z = max((self.sqrt_A_real * self.f) / max(float(sqrt_A), 1.0), 0.20)
        return np.array([
            [e_u / Z,       self.f + e_u**2 / self.f],
            [sqrt_A / Z,    0.0                      ],
        ], dtype=float)

    # ── main solver ──────────────────────────────────────────────────────

    def solve(self, s_now, s_star, u_prev):
        """
        Solve the IBVS-MPC QP and return the first optimal command.

        Parameters
        ----------
        s_now  : [u_c, sqrt_A]        current image features
        s_star : [cx,  sqrt_A_star]   desired features
        u_prev : [v_prev, w_prev]     previous command (for slew term)

        Returns
        -------
        (v_cmd, w_cmd)  — first element of the optimal sequence
        """
        s_now  = np.asarray(s_now,  dtype=float).reshape(2)
        s_star = np.asarray(s_star, dtype=float).reshape(2)
        u_prev = np.asarray(u_prev, dtype=float).reshape(2)

        L    = self._jacobian(s_now[0], s_now[1])
        L_dt = self.dt * L
        ds0  = s_now - s_star
        n    = 2 * self.N

        # ── build Hessian and gradient ────────────────────────────────────
        H_s = np.zeros((n, n))
        g_s = np.zeros(n)
        for k in range(1, self.N + 1):
            # M_k maps the full control sequence U to ds at step k
            M_k = np.zeros((2, n))
            for i in range(k):
                M_k[:, 2*i:2*i+2] = L_dt
            H_s += 2.0 * M_k.T @ self.Q @ M_k
            g_s += 2.0 * M_k.T @ self.Q @ ds0

        H = self._H_base + H_s

        # slew gradient term
        c      = np.zeros(n)
        c[0:2] = u_prev
        g = g_s - 2.0 * self._D.T @ self._I_S @ c

        # symmetrise and regularise for numerical safety
        H = 0.5 * (H + H.T) + 1e-6 * np.eye(n)

        try:
            U = solve_qp(
                sp.csc_matrix(H), g,
                self._G, self._h,
                lb=self._lb, ub=self._ub,
                solver='osqp',
                verbose=False,
            )
        except Exception:
            return 0.0, 0.0

        if U is None:
            return 0.0, 0.0

        return float(U[0]), float(U[1])


# ═══════════════════════════════════════════════════════════════════════════
# FSM
# ═══════════════════════════════════════════════════════════════════════════

class State(Enum):
    ACQUIRE = 0
    SERVO = 1
    PRE_AVOID_STOP = 2
    APPROACH_OBSTACLE = 3
    SQ_TURN_1 = 4
    SQ_LEG_1 = 5
    SQ_TURN_2 = 6
    SQ_LEG_2 = 7
    REACQUIRE = 8
    HOLD = 9
    LOST = 10


# ═══════════════════════════════════════════════════════════════════════════
# Main node
# ═══════════════════════════════════════════════════════════════════════════

class SafeVSController(Node):

    def __init__(self):
        super().__init__('safe_vs_controller')

        # ── image geometry ────────────────────────────────────────────────
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_center_u', -1.0)

        # ── camera / target model (needed by MPC Jacobian) ────────────────
        self.declare_parameter('focal_length_px', 600.0)
        self.declare_parameter('target_real_side_m', 0.15)

        # ── target servoing setpoint ──────────────────────────────────────
        self.declare_parameter('sqrt_area_star', 850.0)
        self.declare_parameter('target_area_deadband', 8.0)
        self.declare_parameter('target_center_deadband_px', 25.0)

        # ── MPC weights and horizon ───────────────────────────────────────
        self.declare_parameter('mpc_horizon_N', 8)
        self.declare_parameter('mpc_dt_s', 0.10)
        self.declare_parameter('Q_u',  1.0)       # lateral error weight
        self.declare_parameter('Q_A',  5.0)       # area (depth) error weight
        self.declare_parameter('R_v',  30000.0)   # forward velocity cost
        self.declare_parameter('R_w',  5000.0)    # angular velocity cost
        self.declare_parameter('S_v',  50.0)      # forward slew cost
        self.declare_parameter('S_w',  50.0)      # angular slew cost

        # ── actuator limits ───────────────────────────────────────────────
        self.declare_parameter('v_max', 0.18)
        self.declare_parameter('w_max', 0.45)
        self.declare_parameter('k_lin', 0.074)
        self.declare_parameter('k_ang', 0.561)
        self.declare_parameter('max_cmd_couple', 2.8)

        # ── acquisition / timeout ─────────────────────────────────────────
        self.declare_parameter('acquire_w', 0.30)
        self.declare_parameter('feature_timeout_s', 0.8)

        # ── blue obstacle trigger ─────────────────────────────────────────
        self.declare_parameter('obstacle_enable', True)
        self.declare_parameter('obstacle_min_sqrt_area', 150.0)
        self.declare_parameter('obstacle_center_band_frac', 0.90)
        self.declare_parameter('obstacle_priority_over_lost', True)
        self.declare_parameter('avoid_cooldown_s', 2.0)

        # ── approach phase ────────────────────────────────────────────────
        self.declare_parameter('approach_obs_v', 0.07)
        self.declare_parameter('approach_obs_kp_w', 0.0025)
        self.declare_parameter('approach_obs_sqrt_area', 350.0)
        self.declare_parameter('approach_obs_max_duration_s', 6.0)

        # ── square bypass ─────────────────────────────────────────────────
        self.declare_parameter('square_turn_w', 0.45)
        self.declare_parameter('square_turn_duration_s', 3.5)
        self.declare_parameter('square_leg_v', 0.10)
        self.declare_parameter('square_leg_1_duration_s', 2.5)
        self.declare_parameter('square_leg_2_duration_s', 3.0)

        # ── goal / hold ───────────────────────────────────────────────────
        self.declare_parameter('hold_required_frames', 10)

        # ── slew rate ─────────────────────────────────────────────────────
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('v_slew_rate', 0.18)
        self.declare_parameter('w_slew_rate', 0.80)

        # ── CSV logging ───────────────────────────────────────────────────
        self.declare_parameter('log_csv', True)
        self.declare_parameter('log_dir', '/tmp/puzzlebot_logs')

        # ── read parameters ───────────────────────────────────────────────
        self.image_width = int(self.get_parameter('image_width').value)
        cx_param = float(self.get_parameter('image_center_u').value)
        self.cx = self.image_width / 2.0 if cx_param < 0 else cx_param

        f_px        = float(self.get_parameter('focal_length_px').value)
        side_m      = float(self.get_parameter('target_real_side_m').value)
        sqrt_A_real = side_m          # physical side -> same units as sqrt(px^2)

        self.sqrt_area_star = float(self.get_parameter('sqrt_area_star').value)
        self.area_db        = float(self.get_parameter('target_area_deadband').value)
        self.center_db      = float(self.get_parameter('target_center_deadband_px').value)

        N    = int(self.get_parameter('mpc_horizon_N').value)
        dt   = float(self.get_parameter('mpc_dt_s').value)
        Q_u  = float(self.get_parameter('Q_u').value)
        Q_A  = float(self.get_parameter('Q_A').value)
        R_v  = float(self.get_parameter('R_v').value)
        R_w  = float(self.get_parameter('R_w').value)
        S_v  = float(self.get_parameter('S_v').value)
        S_w  = float(self.get_parameter('S_w').value)

        self.v_max      = float(self.get_parameter('v_max').value)
        self.w_max      = float(self.get_parameter('w_max').value)
        k_lin           = float(self.get_parameter('k_lin').value)
        k_ang           = float(self.get_parameter('k_ang').value)
        max_couple      = float(self.get_parameter('max_cmd_couple').value)

        self.acquire_w      = float(self.get_parameter('acquire_w').value)
        self.feature_timeout_s = float(self.get_parameter('feature_timeout_s').value)

        self.obstacle_enable         = bool(self.get_parameter('obstacle_enable').value)
        self.obstacle_min_sqrt_area  = float(self.get_parameter('obstacle_min_sqrt_area').value)
        self.obstacle_center_band_frac = float(self.get_parameter('obstacle_center_band_frac').value)
        self.obstacle_priority_over_lost = bool(self.get_parameter('obstacle_priority_over_lost').value)
        self.avoid_cooldown_s        = float(self.get_parameter('avoid_cooldown_s').value)

        self.approach_obs_v           = float(self.get_parameter('approach_obs_v').value)
        self.approach_obs_kp_w        = float(self.get_parameter('approach_obs_kp_w').value)
        self.approach_obs_sqrt_area   = float(self.get_parameter('approach_obs_sqrt_area').value)
        self.approach_obs_max_duration_s = float(self.get_parameter('approach_obs_max_duration_s').value)

        self.square_turn_w            = float(self.get_parameter('square_turn_w').value)
        self.square_turn_duration_s   = float(self.get_parameter('square_turn_duration_s').value)
        self.square_leg_v             = float(self.get_parameter('square_leg_v').value)
        self.square_leg_1_duration_s  = float(self.get_parameter('square_leg_1_duration_s').value)
        self.square_leg_2_duration_s  = float(self.get_parameter('square_leg_2_duration_s').value)

        self.hold_required_frames = int(self.get_parameter('hold_required_frames').value)
        self.v_slew_rate = float(self.get_parameter('v_slew_rate').value)
        self.w_slew_rate = float(self.get_parameter('w_slew_rate').value)
        rate = float(self.get_parameter('control_rate_hz').value)

        self.log_csv = bool(self.get_parameter('log_csv').value)
        self.log_dir = self.get_parameter('log_dir').value

        # ── build MPC solver ──────────────────────────────────────────────
        self.mpc = IBVS_MPC(
            N=N, dt=dt, f=f_px, cx=self.cx, sqrt_A_real=sqrt_A_real,
            v_max=self.v_max, omega_max=self.w_max,
            k_lin=k_lin, k_ang=k_ang, max_cmd_couple=max_couple,
            Q=np.diag([Q_u, Q_A]),
            R=np.diag([R_v, R_w]),
            S=np.diag([S_v, S_w]),
        )

        # ── internal state ────────────────────────────────────────────────
        self.state              = State.ACQUIRE
        self.target             = None
        self.obstacle           = None
        self.last_target_time   = None
        self.last_obstacle_time = None
        self.last_loop_time     = None
        self.last_v             = 0.0
        self.last_w             = 0.0
        self.state_start_time   = self.get_clock().now()
        self.avoid_sign         = 1.0
        self.hold_counter       = 0
        self.last_debug         = ''
        self.last_avoid_end_time = None

        # ── CSV logging setup ─────────────────────────────────────────────
        self._log_rows  = []
        self._run_start = time_mod.time()
        self._log_path  = None

        if self.log_csv:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = int(self._run_start)
            self._log_path = os.path.join(self.log_dir, f'run_{ts}.csv')
            with open(self._log_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    't_rel', 'state',
                    'target_det', 'u_c', 'sqrt_area', 'e_u', 'e_A',
                    'obs_det', 'obs_sqrt_area', 'obs_u_c',
                    'danger', 'v_cmd', 'w_cmd'
                ])
            self.get_logger().info(f'Logging to: {self._log_path}')

        # ── ROS I/O ───────────────────────────────────────────────────────
        self.create_subscription(
            TargetFeatures,  '/target_features',   self.cb_target,   10)
        self.create_subscription(
            ObstacleFeatures,'/obstacle_features', self.cb_obstacle, 10)

        self.pub_cmd   = self.create_publisher(Twist,  '/cmd_vel',    10)
        self.pub_state = self.create_publisher(String, '/vs_state',   10)
        self.pub_debug = self.create_publisher(String, '/avoid_debug',10)

        self.create_timer(1.0 / rate, self.step)

        self.get_logger().info(
            f'Safe VS controller (IBVS-MPC) | N={N} dt={dt}s | '
            f'cx={self.cx:.1f} f={f_px:.1f}px | '
            f'sqrt_area_star={self.sqrt_area_star:.1f} | '
            f'obs_trigger={self.obstacle_min_sqrt_area:.1f}'
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def cb_target(self, msg):
        self.target = msg
        self.last_target_time = self.get_clock().now()

    def cb_obstacle(self, msg):
        self.obstacle = msg
        self.last_obstacle_time = self.get_clock().now()

    # ── time helpers ──────────────────────────────────────────────────────

    def age_s(self, stamp):
        if stamp is None:
            return None
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def state_elapsed_s(self):
        return (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9

    def set_state(self, new_state):
        if new_state != self.state:
            self.get_logger().info(f'{self.state.name} -> {new_state.name}')
            self.state = new_state
            self.state_start_time = self.get_clock().now()
            self.hold_counter = 0

    def cooldown_active(self):
        if self.last_avoid_end_time is None:
            return False
        return (
            self.get_clock().now() - self.last_avoid_end_time
        ).nanoseconds * 1e-9 < self.avoid_cooldown_s

    # ── feature checks ────────────────────────────────────────────────────

    def target_fresh(self):
        if self.target is None:
            return False
        age = self.age_s(self.last_target_time)
        if age is None or age > self.feature_timeout_s:
            return False
        return bool(self.target.detected)

    def obstacle_fresh(self):
        if self.obstacle is None:
            return False
        age = self.age_s(self.last_obstacle_time)
        if age is None or age > self.feature_timeout_s:
            return False
        return bool(self.obstacle.detected)

    def obstacle_danger(self):
        if not self.obstacle_enable:
            return False, 'obstacle disabled'
        if self.cooldown_active():
            return False, 'cooldown active'
        if not self.obstacle_fresh():
            return False, 'blue not visible'

        sqrt_area = float(self.obstacle.sqrt_area)
        obs_u     = float(self.obstacle.u_c)

        if sqrt_area < self.obstacle_min_sqrt_area:
            return False, f'blue too small {sqrt_area:.1f} < {self.obstacle_min_sqrt_area:.1f}'

        half_band = 0.5 * self.obstacle_center_band_frac * self.image_width
        if abs(obs_u - self.cx) > half_band:
            return False, f'blue outside band u={obs_u:.1f}'

        return True, f'DANGER sqrt_A={sqrt_area:.1f}'

    def compute_avoid_sign(self):
        if not self.obstacle_fresh():
            return 1.0
        return 1.0 if float(self.obstacle.u_c) > self.cx else -1.0

    def at_goal(self):
        if not self.target_fresh():
            return False
        return (
            abs(float(self.target.u_c) - self.cx) < self.center_db
            and abs(float(self.target.sqrt_area) - self.sqrt_area_star) < self.area_db
        )

    # ── control laws ──────────────────────────────────────────────────────

    def mpc_command(self):
        """
        Compute the optimal velocity command using the IBVS-MPC solver.

        At each call the QP is solved over the N-step horizon. Only the
        first command of the optimal sequence is returned and applied
        (receding horizon principle). The previous published command is
        passed as u_prev so the slew term S in the cost function is
        evaluated correctly.
        """
        s_now  = [float(self.target.u_c),     float(self.target.sqrt_area)]
        s_star = [self.cx,                     self.sqrt_area_star]
        u_prev = [self.last_v,                 self.last_w]
        v_cmd, w_cmd = self.mpc.solve(s_now, s_star, u_prev)
        return v_cmd, w_cmd

    def approach_command(self):
        """Slow approach toward the blue obstacle while centering it."""
        obs_u   = float(self.obstacle.u_c)
        e_obs_u = obs_u - self.cx
        w = -self.approach_obs_kp_w * e_obs_u
        v = self.approach_obs_v
        v = clip(v, 0.0, self.approach_obs_v)
        w = clip(w, -self.w_max, self.w_max)
        return v, w

    # ── publish ───────────────────────────────────────────────────────────

    def publish_cmd(self, v_des, w_des, dt, force_zero_v=False):
        if force_zero_v:
            v = 0.0
            self.last_v = 0.0
        else:
            max_dv = self.v_slew_rate * dt
            v = clip(v_des, self.last_v - max_dv, self.last_v + max_dv)
            self.last_v = v
        max_dw = self.w_slew_rate * dt
        w = clip(w_des, self.last_w - max_dw, self.last_w + max_dw)
        self.last_w = w
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def publish_state(self):
        msg = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)

    def publish_debug(self, text):
        msg = String()
        msg.data = text
        self.pub_debug.publish(msg)
        if text != self.last_debug:
            self.get_logger().info(f'[debug] {text}')
            self.last_debug = text

    # ── CSV logging ───────────────────────────────────────────────────────

    def _log_tick(self, danger):
        if not self.log_csv:
            return
        t_rel = time_mod.time() - self._run_start
        if self.target is not None and bool(self.target.detected):
            tdet, tu, tsa = 1, float(self.target.u_c), float(self.target.sqrt_area)
        else:
            tdet, tu, tsa = 0, 0.0, 0.0
        e_u = tu - self.cx if tdet else 0.0
        e_A = self.sqrt_area_star - tsa if tdet else 0.0
        if self.obstacle is not None and bool(self.obstacle.detected):
            odet, osa, ou = 1, float(self.obstacle.sqrt_area), float(self.obstacle.u_c)
        else:
            odet, osa, ou = 0, 0.0, 0.0
        self._log_rows.append([
            f'{t_rel:.3f}', self.state.name,
            tdet, f'{tu:.1f}', f'{tsa:.1f}',
            f'{e_u:.1f}', f'{e_A:.1f}',
            odet, f'{osa:.1f}', f'{ou:.1f}',
            int(danger), f'{self.last_v:.4f}', f'{self.last_w:.4f}'
        ])
        if len(self._log_rows) >= 50:
            self._flush_log()

    def _flush_log(self):
        if not self._log_rows or self._log_path is None:
            return
        with open(self._log_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerows(self._log_rows)
        self._log_rows.clear()

    # ── main FSM loop ─────────────────────────────────────────────────────

    def step(self):
        now = self.get_clock().now()
        if self.last_loop_time is None:
            self.last_loop_time = now
            return

        dt = (now - self.last_loop_time).nanoseconds * 1e-9
        self.last_loop_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        v_des = 0.0
        w_des = 0.0
        force_zero_v = False
        elapsed = self.state_elapsed_s()

        danger, danger_reason = self.obstacle_danger()
        target_ok = self.target_fresh()

        # ── ACQUIRE ───────────────────────────────────────────────────────
        if self.state == State.ACQUIRE:
            if target_ok:
                self.set_state(State.SERVO)
            else:
                w_des = self.acquire_w

        # ── SERVO — MPC command ───────────────────────────────────────────
        elif self.state == State.SERVO:
            if danger:
                self.last_v = 0.0
                self.set_state(State.PRE_AVOID_STOP)
            elif not target_ok:
                self.set_state(State.LOST)
            elif self.at_goal():
                self.hold_counter += 1
                if self.hold_counter >= self.hold_required_frames:
                    self.set_state(State.HOLD)
            else:
                self.hold_counter = 0
                v_des, w_des = self.mpc_command()    # ← IBVS-MPC

        # ── PRE_AVOID_STOP ────────────────────────────────────────────────
        elif self.state == State.PRE_AVOID_STOP:
            force_zero_v = True
            if elapsed >= 0.45:
                self.set_state(State.APPROACH_OBSTACLE)

        # ── APPROACH_OBSTACLE ─────────────────────────────────────────────
        elif self.state == State.APPROACH_OBSTACLE:
            if not self.obstacle_fresh():
                self.get_logger().warn('APPROACH: obstacle lost, starting bypass.')
                self.avoid_sign = self.compute_avoid_sign()
                self.set_state(State.SQ_TURN_1)
            elif elapsed >= self.approach_obs_max_duration_s:
                self.get_logger().warn('APPROACH: timeout, starting bypass.')
                self.avoid_sign = self.compute_avoid_sign()
                self.set_state(State.SQ_TURN_1)
            elif float(self.obstacle.sqrt_area) >= self.approach_obs_sqrt_area:
                self.avoid_sign = self.compute_avoid_sign()
                self.get_logger().info(
                    f'APPROACH: reached target sqrt_area={self.obstacle.sqrt_area:.1f} '
                    f'avoid_sign={self.avoid_sign:+.0f}')
                self.set_state(State.SQ_TURN_1)
            else:
                v_des, w_des = self.approach_command()
                danger_reason = (
                    f'approaching | '
                    f'sqrt_area={self.obstacle.sqrt_area:.1f}'
                    f'/{self.approach_obs_sqrt_area:.1f}'
                )

        # ── SQ_TURN_1 ─────────────────────────────────────────────────────
        elif self.state == State.SQ_TURN_1:
            w_des = self.avoid_sign * self.square_turn_w
            force_zero_v = True
            if elapsed >= self.square_turn_duration_s:
                self.set_state(State.SQ_LEG_1)

        # ── SQ_LEG_1 ──────────────────────────────────────────────────────
        elif self.state == State.SQ_LEG_1:
            v_des = self.square_leg_v
            w_des = 0.0
            if elapsed >= self.square_leg_1_duration_s:
                self.set_state(State.SQ_TURN_2)

        # ── SQ_TURN_2 ─────────────────────────────────────────────────────
        elif self.state == State.SQ_TURN_2:
            w_des = -self.avoid_sign * self.square_turn_w
            force_zero_v = True
            if elapsed >= self.square_turn_duration_s:
                self.set_state(State.SQ_LEG_2)

        # ── SQ_LEG_2 ──────────────────────────────────────────────────────
        elif self.state == State.SQ_LEG_2:
            v_des = self.square_leg_v
            w_des = 0.0
            if elapsed >= self.square_leg_2_duration_s:
                self.last_avoid_end_time = self.get_clock().now()
                self.set_state(State.REACQUIRE)

        # ── REACQUIRE ─────────────────────────────────────────────────────
        elif self.state == State.REACQUIRE:
            if target_ok:
                self.set_state(State.SERVO)
            else:
                w_des = self.acquire_w

        # ── LOST ──────────────────────────────────────────────────────────
        elif self.state == State.LOST:
            if self.obstacle_priority_over_lost and danger:
                self.last_v = 0.0
                self.set_state(State.PRE_AVOID_STOP)
            elif target_ok:
                self.set_state(State.SERVO)
            else:
                w_des = self.acquire_w

        # ── HOLD ──────────────────────────────────────────────────────────
        elif self.state == State.HOLD:
            force_zero_v = True

        # ── publish + log ─────────────────────────────────────────────────
        self.publish_cmd(v_des, w_des, dt, force_zero_v=force_zero_v)
        self.publish_state()

        obs_str = f'{float(self.obstacle.sqrt_area):.1f}' if self.obstacle_fresh() else 'none'
        self.publish_debug(
            f'state={self.state.name} | target={target_ok} | '
            f'danger={danger} | obs={obs_str} | t={elapsed:.1f}s | {danger_reason}'
        )

        self._log_tick(danger)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SafeVSController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        try:
            for _ in range(10):
                node.pub_cmd.publish(stop)
        except Exception:
            pass
        node._flush_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
