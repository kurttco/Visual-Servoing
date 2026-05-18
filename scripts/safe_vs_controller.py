#!/usr/bin/env python3
"""
safe_vs_controller.py
======================
Visual servoing controller for the PuzzleBot (Manchester Robotics).
Part of: TE3002B – Mobile Robotics and Computer Vision
         Tecnológico de Monterrey

OVERVIEW
--------
Implements a closed-loop Image-Based Visual Servoing (IBVS) pipeline:

  1. Receives image features from /target_features and /obstacle_features
  2. Runs a proportional IBVS control law (derived from an MPC formulation)
  3. Applies a centering-scale coupling to eliminate oscillatory approach
  4. Executes an approach-then-arc obstacle avoidance maneuver when needed
  5. Publishes velocity commands to /cmd_vel

COLLISION AVOIDANCE MODEL
--------------------------
When a blue obstacle is detected ahead, the robot:
  (a) Stops completely            [PRE_AVOID_STOP]
  (b) Slowly approaches obstacle  [APPROACH_OBSTACLE]
      - Centers it in the camera using visual feedback
      - Advances until sqrt(A_obs) >= approach_obs_sqrt_area
      - This establishes a consistent, known starting distance
  (c) Executes a smooth arc       [AVOID_ARC]
      - Direction decided from obstacle position at arc start
  (d) Counter-rotates to realign  [AVOID_RECENTER]
  (e) Searches for target again   [REACQUIRE]

FSM STATES
----------
  ACQUIRE -> SERVO -> PRE_AVOID_STOP -> APPROACH_OBSTACLE
          -> AVOID_ARC -> AVOID_RECENTER -> REACQUIRE -> SERVO
  SERVO   -> HOLD  (goal reached)
  SERVO   -> LOST  -> ACQUIRE

SUBSCRIPTIONS
-------------
  /target_features    puzzlebot_mc2/TargetFeatures
  /obstacle_features  puzzlebot_mc2/ObstacleFeatures

PUBLICATIONS
------------
  /cmd_vel            geometry_msgs/Twist
  /vs_state           std_msgs/String
  /avoid_debug        std_msgs/String    (human-readable debug info)

PARAMETERS  (see config/safe_vs_params.yaml for full list and comments)
----------
  See Parameter sections below.
"""

from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from puzzlebot_mc2.msg import TargetFeatures, ObstacleFeatures


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clip(x, lo, hi):
    """Clamp x to the interval [lo, hi]."""
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# FSM State Enum
# ---------------------------------------------------------------------------

class State(Enum):
    ACQUIRE           = 0
    SERVO             = 1
    PRE_AVOID_STOP    = 2
    APPROACH_OBSTACLE = 3
    AVOID_ARC         = 4
    AVOID_RECENTER    = 5
    REACQUIRE         = 6
    HOLD              = 7
    LOST              = 8


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------

class SafeVSController(Node):

    def __init__(self):
        super().__init__('safe_vs_controller')

        # ── Image geometry ────────────────────────────────────────────────
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_center_u', -1.0)  # <0 → use width/2

        # ── Target servoing ───────────────────────────────────────────────
        self.declare_parameter('sqrt_area_star', 850.0)
        self.declare_parameter('target_area_deadband', 8.0)
        self.declare_parameter('target_center_deadband_px', 25.0)
        self.declare_parameter('kp_w', 0.0025)
        self.declare_parameter('kp_v', 0.0020)
        self.declare_parameter('v_max', 0.18)
        self.declare_parameter('w_max', 0.45)
        self.declare_parameter('v_min_when_far', 0.035)

        # Centering-scale band.  When lateral error |e_u| > v_stop_band_px,
        # forward speed is set to zero and the robot only rotates.
        # This eliminates the oscillatory wave motion in the servo phase.
        self.declare_parameter('v_stop_band_px', 60.0)

        # ── Acquisition / timeout ─────────────────────────────────────────
        self.declare_parameter('acquire_w', 0.30)
        self.declare_parameter('feature_timeout_s', 0.8)

        # ── Blue obstacle trigger ─────────────────────────────────────────
        self.declare_parameter('obstacle_enable', True)
        self.declare_parameter('obstacle_min_sqrt_area', 150.0)
        self.declare_parameter('obstacle_center_band_frac', 0.75)
        self.declare_parameter('obstacle_priority_over_lost', True)
        self.declare_parameter('avoid_cooldown_s', 1.0)

        # ── APPROACH_OBSTACLE ─────────────────────────────────────────────
        self.declare_parameter('approach_obs_v', 0.07)
        self.declare_parameter('approach_obs_kp_w', 0.0025)
        self.declare_parameter('approach_obs_sqrt_area', 350.0)
        self.declare_parameter('approach_obs_max_duration_s', 6.0)

        # ── AVOID_ARC ─────────────────────────────────────────────────────
        # arc radius r = v/w.  Higher w → tighter radius → more lateral
        # displacement.  Tune arc_w first to achieve physical clearance,
        # then adjust arc_duration for how far past the obstacle to travel.
        self.declare_parameter('avoid_arc_duration_s', 4.0)
        self.declare_parameter('avoid_arc_v', 0.07)
        self.declare_parameter('avoid_arc_w', 0.35)

        # ── AVOID_RECENTER ────────────────────────────────────────────────
        self.declare_parameter('avoid_recenter_duration_s', 1.80)
        self.declare_parameter('avoid_recenter_w', 0.67)

        # ── Hold ─────────────────────────────────────────────────────────
        self.declare_parameter('hold_required_frames', 10)

        # ── Slew rate / control rate ──────────────────────────────────────
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('v_slew_rate', 0.20)
        self.declare_parameter('w_slew_rate', 0.90)

        # ── Read parameters ───────────────────────────────────────────────
        self.image_width  = int(self.get_parameter('image_width').value)
        center_p          = float(self.get_parameter('image_center_u').value)
        self.cx           = self.image_width / 2.0 if center_p < 0 else center_p

        self.sqrt_area_star   = float(self.get_parameter('sqrt_area_star').value)
        self.area_db          = float(self.get_parameter('target_area_deadband').value)
        self.center_db        = float(self.get_parameter('target_center_deadband_px').value)
        self.kp_w             = float(self.get_parameter('kp_w').value)
        self.kp_v             = float(self.get_parameter('kp_v').value)
        self.v_max            = float(self.get_parameter('v_max').value)
        self.w_max            = float(self.get_parameter('w_max').value)
        self.v_min_when_far   = float(self.get_parameter('v_min_when_far').value)
        self.v_stop_band_px   = float(self.get_parameter('v_stop_band_px').value)

        self.acquire_w        = float(self.get_parameter('acquire_w').value)
        self.feature_timeout  = float(self.get_parameter('feature_timeout_s').value)

        self.obstacle_enable       = bool(self.get_parameter('obstacle_enable').value)
        self.obstacle_min_sqrt_area= float(self.get_parameter('obstacle_min_sqrt_area').value)
        self.obs_center_band_frac  = float(self.get_parameter('obstacle_center_band_frac').value)
        self.obs_priority_lost     = bool(self.get_parameter('obstacle_priority_over_lost').value)
        self.avoid_cooldown_s      = float(self.get_parameter('avoid_cooldown_s').value)

        self.approach_obs_v            = float(self.get_parameter('approach_obs_v').value)
        self.approach_obs_kp_w         = float(self.get_parameter('approach_obs_kp_w').value)
        self.approach_obs_sqrt_area    = float(self.get_parameter('approach_obs_sqrt_area').value)
        self.approach_obs_max_duration = float(self.get_parameter('approach_obs_max_duration_s').value)

        self.avoid_arc_duration  = float(self.get_parameter('avoid_arc_duration_s').value)
        self.avoid_arc_v         = float(self.get_parameter('avoid_arc_v').value)
        self.avoid_arc_w         = float(self.get_parameter('avoid_arc_w').value)

        self.avoid_recenter_duration = float(self.get_parameter('avoid_recenter_duration_s').value)
        self.avoid_recenter_w        = float(self.get_parameter('avoid_recenter_w').value)

        self.hold_required_frames = int(self.get_parameter('hold_required_frames').value)
        self.v_slew_rate          = float(self.get_parameter('v_slew_rate').value)
        self.w_slew_rate          = float(self.get_parameter('w_slew_rate').value)
        rate                      = float(self.get_parameter('control_rate_hz').value)

        # ── Internal state ────────────────────────────────────────────────
        self.state            = State.ACQUIRE
        self.target           = None
        self.obstacle         = None
        self.last_target_time = None
        self.last_obs_time    = None
        self.last_loop_time   = None
        self.last_v           = 0.0
        self.last_w           = 0.0
        self.state_start_time = self.get_clock().now()
        self.avoid_sign       = 1.0   # +1 arc left, -1 arc right
        self.hold_counter     = 0
        self.last_debug       = ''
        self.last_avoid_end   = None

        # ── ROS I/O ───────────────────────────────────────────────────────
        self.create_subscription(TargetFeatures,   '/target_features',
                                 self.cb_target,   10)
        self.create_subscription(ObstacleFeatures, '/obstacle_features',
                                 self.cb_obstacle, 10)

        self.pub_cmd   = self.create_publisher(Twist,  '/cmd_vel',     10)
        self.pub_state = self.create_publisher(String, '/vs_state',    10)
        self.pub_debug = self.create_publisher(String, '/avoid_debug', 10)

        self.create_timer(1.0 / rate, self.step)

        self.get_logger().info(
            f'SafeVSController ready | '
            f'cx={self.cx:.0f}px | sqrt_A*={self.sqrt_area_star:.0f} | '
            f'obs_trigger={self.obstacle_min_sqrt_area:.0f} | '
            f'approach_target={self.approach_obs_sqrt_area:.0f}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def cb_target(self, msg):
        self.target = msg
        self.last_target_time = self.get_clock().now()

    def cb_obstacle(self, msg):
        self.obstacle = msg
        self.last_obs_time = self.get_clock().now()

    # ── Time helpers ──────────────────────────────────────────────────────

    def _age(self, stamp):
        if stamp is None:
            return None
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _state_elapsed(self):
        return (self.get_clock().now() - self.state_start_time).nanoseconds * 1e-9

    def _set_state(self, new_state):
        if new_state != self.state:
            self.get_logger().info(f'FSM: {self.state.name} -> {new_state.name}')
            self.state            = new_state
            self.state_start_time = self.get_clock().now()
            self.hold_counter     = 0

    def _cooldown_active(self):
        if self.last_avoid_end is None:
            return False
        return (self.get_clock().now() - self.last_avoid_end
                ).nanoseconds * 1e-9 < self.avoid_cooldown_s

    # ── Feature checks ────────────────────────────────────────────────────

    def _target_fresh(self):
        """True when a green target is actively detected."""
        if self.target is None:
            return False
        age = self._age(self.last_target_time)
        return age is not None and age <= self.feature_timeout and bool(self.target.detected)

    def _obstacle_fresh(self):
        """True when a blue obstacle is actively detected."""
        if self.obstacle is None:
            return False
        age = self._age(self.last_obs_time)
        return age is not None and age <= self.feature_timeout and bool(self.obstacle.detected)

    def _obstacle_danger(self):
        """
        Return (danger: bool, reason: str).

        Danger requires:
          1. Obstacle detected and fresh.
          2. Apparent size >= obstacle_min_sqrt_area  (close enough).
          3. Centroid within the center band          (directly ahead).
          4. No cooldown active.
        """
        if not self.obstacle_enable:
            return False, 'disabled'
        if self._cooldown_active():
            return False, 'cooldown'
        if not self._obstacle_fresh():
            return False, 'blue not visible'

        sa  = float(self.obstacle.sqrt_area)
        u   = float(self.obstacle.u_c)

        if sa < self.obstacle_min_sqrt_area:
            return False, f'too small {sa:.0f}<{self.obstacle_min_sqrt_area:.0f}'

        half = 0.5 * self.obs_center_band_frac * self.image_width
        if abs(u - self.cx) > half:
            return False, f'outside band u={u:.0f}'

        return True, f'DANGER sa={sa:.0f} u={u:.0f}'

    def _compute_avoid_sign(self):
        """
        Decide arc direction from obstacle lateral position.
        Obstacle on the right -> arc left (+1).
        Obstacle on the left  -> arc right (-1).
        """
        if not self._obstacle_fresh():
            return 1.0
        return 1.0 if float(self.obstacle.u_c) > self.cx else -1.0

    def _at_goal(self):
        if not self._target_fresh():
            return False
        return (abs(float(self.target.u_c) - self.cx)    < self.center_db and
                abs(float(self.target.sqrt_area) - self.sqrt_area_star) < self.area_db)

    # ── Control laws ──────────────────────────────────────────────────────

    def _servo_command(self):
        """
        Proportional IBVS control law toward the green target.

        Angular command: w = -kp_w * e_u   (center target horizontally)
        Linear command:  v = kp_v * e_A    (approach until desired size)

        Centering-scale coupling (eliminates wave motion):
          When |e_u| > v_stop_band_px, v = 0 (pure rotation only).
          When |e_u| <= v_stop_band_px, v is scaled linearly from 0 to full.
          This prevents the robot from driving forward while still off-center.
        """
        e_u = float(self.target.u_c) - self.cx
        e_A = self.sqrt_area_star - float(self.target.sqrt_area)

        w = -self.kp_w * e_u

        v = self.kp_v * e_A if e_A > self.area_db else 0.0
        if e_A > self.area_db:
            v = max(v, self.v_min_when_far)

        # Centering-scale: zero v outside band, linear ramp inside
        if abs(e_u) > self.v_stop_band_px:
            v = 0.0
        else:
            v = v * (1.0 - abs(e_u) / self.v_stop_band_px)

        return clip(v, 0.0, self.v_max), clip(w, -self.w_max, self.w_max)

    def _approach_command(self):
        """
        Slow advance toward the blue obstacle with simultaneous centering.

        Forward speed is fixed (not scaled by centering error) so the robot
        always moves toward the obstacle even when it is not perfectly centered.
        The angular command handles centering in parallel.
        """
        e_obs_u = float(self.obstacle.u_c) - self.cx
        w = -self.approach_obs_kp_w * e_obs_u
        v = self.approach_obs_v
        return clip(v, 0.0, self.approach_obs_v), clip(w, -self.w_max, self.w_max)

    # ── Publish helpers ───────────────────────────────────────────────────

    def _publish_cmd(self, v_des, w_des, dt, force_zero_v=False):
        """Apply slew-rate limiting and publish /cmd_vel."""
        if force_zero_v:
            v = 0.0
            self.last_v = 0.0
        else:
            dv = self.v_slew_rate * dt
            v  = clip(v_des, self.last_v - dv, self.last_v + dv)
            self.last_v = v

        dw = self.w_slew_rate * dt
        w  = clip(w_des, self.last_w - dw, self.last_w + dw)
        self.last_w = w

        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _publish_state(self):
        msg = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)

    def _publish_debug(self, text):
        msg = String()
        msg.data = text
        self.pub_debug.publish(msg)
        if text != self.last_debug:
            self.get_logger().info(f'[debug] {text}')
            self.last_debug = text

    # ── Main control loop ─────────────────────────────────────────────────

    def step(self):
        now = self.get_clock().now()
        if self.last_loop_time is None:
            self.last_loop_time = now
            return

        dt = (now - self.last_loop_time).nanoseconds * 1e-9
        self.last_loop_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        v_des       = 0.0
        w_des       = 0.0
        force_zero  = False

        danger, dreason = self._obstacle_danger()
        target_ok       = self._target_fresh()

        # ── ACQUIRE ───────────────────────────────────────────────────────
        if self.state == State.ACQUIRE:
            if target_ok:
                self._set_state(State.SERVO)
            else:
                w_des = self.acquire_w

        # ── SERVO ─────────────────────────────────────────────────────────
        elif self.state == State.SERVO:
            if danger:
                self.last_v = 0.0
                self._set_state(State.PRE_AVOID_STOP)

            elif not target_ok:
                self._set_state(State.LOST)

            elif self._at_goal():
                self.hold_counter += 1
                if self.hold_counter >= self.hold_required_frames:
                    self._set_state(State.HOLD)

            else:
                self.hold_counter = 0
                v_des, w_des = self._servo_command()

        # ── PRE_AVOID_STOP ────────────────────────────────────────────────
        elif self.state == State.PRE_AVOID_STOP:
            force_zero = True
            if self._state_elapsed() >= 0.45:
                self._set_state(State.APPROACH_OBSTACLE)

        # ── APPROACH_OBSTACLE ─────────────────────────────────────────────
        # Robot centers the obstacle and approaches slowly until the obstacle
        # apparent size reaches approach_obs_sqrt_area, establishing a known,
        # consistent starting distance for the arc.
        elif self.state == State.APPROACH_OBSTACLE:

            timed_out = self._state_elapsed() >= self.approach_obs_max_duration

            if not self._obstacle_fresh():
                # Obstacle left FOV — proceed with last known direction
                self.get_logger().warn('APPROACH: obstacle lost, firing arc anyway.')
                self.avoid_sign = self._compute_avoid_sign()
                self._set_state(State.AVOID_ARC)

            elif timed_out:
                self.get_logger().warn('APPROACH: timeout, firing arc.')
                self.avoid_sign = self._compute_avoid_sign()
                self._set_state(State.AVOID_ARC)

            elif float(self.obstacle.sqrt_area) >= self.approach_obs_sqrt_area:
                # Good visual fix obtained — compute direction and fire arc
                self.avoid_sign = self._compute_avoid_sign()
                self.get_logger().info(
                    f'APPROACH: target reached sa={self.obstacle.sqrt_area:.0f} '
                    f'sign={self.avoid_sign:+.0f}'
                )
                self._set_state(State.AVOID_ARC)

            else:
                # Still approaching
                v_des, w_des = self._approach_command()
                dreason = (f'approaching obs | '
                           f'sa={float(self.obstacle.sqrt_area):.0f}'
                           f'/{self.approach_obs_sqrt_area:.0f}')

        # ── AVOID_ARC ─────────────────────────────────────────────────────
        elif self.state == State.AVOID_ARC:
            v_des = self.avoid_arc_v
            w_des = self.avoid_sign * self.avoid_arc_w
            if self._state_elapsed() >= self.avoid_arc_duration:
                self._set_state(State.AVOID_RECENTER)

        # ── AVOID_RECENTER ────────────────────────────────────────────────
        elif self.state == State.AVOID_RECENTER:
            w_des      = -self.avoid_sign * self.avoid_recenter_w
            force_zero = True
            if self._state_elapsed() >= self.avoid_recenter_duration:
                self.last_avoid_end = self.get_clock().now()
                self._set_state(State.REACQUIRE)

        # ── REACQUIRE ─────────────────────────────────────────────────────
        elif self.state == State.REACQUIRE:
            if target_ok:
                self._set_state(State.SERVO)
            else:
                w_des = self.acquire_w

        # ── LOST ──────────────────────────────────────────────────────────
        elif self.state == State.LOST:
            if self.obs_priority_lost and danger:
                self.last_v = 0.0
                self._set_state(State.PRE_AVOID_STOP)
            elif target_ok:
                self._set_state(State.SERVO)
            else:
                w_des = self.acquire_w

        # ── HOLD ──────────────────────────────────────────────────────────
        elif self.state == State.HOLD:
            force_zero = True

        # ── Publish ───────────────────────────────────────────────────────
        self._publish_cmd(v_des, w_des, dt, force_zero_v=force_zero)
        self._publish_state()

        obs_sa = (f'{float(self.obstacle.sqrt_area):.0f}'
                  if self._obstacle_fresh() else 'none')

        self._publish_debug(
            f'state={self.state.name} | target={target_ok} | '
            f'danger={danger} | obs_sa={obs_sa} | {dreason}'
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
