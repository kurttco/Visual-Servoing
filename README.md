# Optimal Control for Visual Servoing — PuzzleBot

**Course:** TE3002B — Mobile Robotics and Computer Vision  
**Institution:** Tecnológico de Monterrey  
**Team:** Ferro M., Banda F., Ortega Z., Cárdenas A., Proal F.

---

## Overview

This repository contains the full implementation of a closed-loop visual servoing pipeline on the PuzzleBot differential drive robot. The robot uses only its forward-facing camera to navigate toward a colored target while detecting and avoiding a colored obstacle placed in its path.

**Key features:**
- Classical HSV color segmentation (no neural networks)
- Image-Based Visual Servoing (IBVS) formulated as a finite-horizon MPC and solved as a QP with OSQP at 10 Hz
- Visual-feedback obstacle avoidance: approach-then-square-bypass strategy
- Full FSM with logging, parameter-driven tuning, and result plot generation

**Demo video:** https://youtu.be/xGdBGcd0faU

---

## Repository Structure

```
puzzlebot_mc2/
│
├── msg/
│   ├── TargetFeatures.msg        # Image features of the green target
│   └── ObstacleFeatures.msg      # Image features of the blue obstacle
│
├── scripts/
│   │
│   │ 
│   ├── safe_vs_controller.py     # Main controller: IBVS-MPC + obstacle avoidance FSM
│   ├── color_features_detector.py # Vision node: HSV segmentation → feature messages
│   ├── cmd_vel_to_wheels.py      # Converts /cmd_vel → /VelocitySetL, /VelocitySetR
│   └── plot_results.py           # Post-run analysis: reads CSV log → report figures
│
├── launch/
│   └── safe_vs_demo.launch.py    # Launches the full VS demo (3 nodes)
│
├── config/
│   ├── safe_vs_params.yaml       # All parameters for the VS demo
│   └── robot_params.yaml         # Physical robot parameters (odometry)
│
├── CMakeLists.txt
└── package.xml
```
---

## Dependencies

**ROS 2:** Humble Hawksbill  
**OS:** Ubuntu 22.04 (tested on Jetson Nano)

Python packages:

```bash
pip install qpsolvers[osqp] numpy scipy --break-system-packages
pip install opencv-python --break-system-packages   # if not already present
```

ROS 2 packages (should already be installed on PuzzleBot image):

```
rclpy  std_msgs  geometry_msgs  sensor_msgs  cv_bridge
```

---

## Installation

```bash
# Clone into your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/kurttco/Visual-Servoing.git puzzlebot_mc2

# Build
cd ~/ros2_ws
colcon build --packages-select puzzlebot_mc2
source install/setup.bash
```

---

## Running the Demo

```bash
ros2 launch puzzlebot_mc2 safe_vs_demo.launch.py
```

This starts three nodes:

| Node | Script | Role |
|---|---|---|
| `color_features_detector` | `color_features_detector.py` | Camera → HSV → feature messages |
| `safe_vs_controller` | `safe_vs_controller.py` | MPC + FSM + avoidance |
| `cmd_vel_to_wheels` | `cmd_vel_to_wheels.py` | `/cmd_vel` → wheel commands |

**Monitor in real time:**

```bash
# FSM state
ros2 topic echo /vs_state

# Detailed debug per tick (state, danger flag, obstacle size, etc.)
ros2 topic echo /avoid_debug

# Target and obstacle detections
ros2 topic echo /target_features
ros2 topic echo /obstacle_features
```

**CSV log** is written automatically to `/tmp/puzzlebot_logs/run_<timestamp>.csv` at 10 Hz. Use this to generate the result plots after a run.

---

## Generating Result Plots

```bash
# On the Jetson — copy the log to your laptop
scp puzzlebot@<JETSON_IP>:/tmp/puzzlebot_logs/run_*.csv ./

# Generate the three report figures
python3 scripts/plot_results.py run_<timestamp>.csv --out ./figures/
```

This produces:

| File | Contents |
|---|---|
| `fig_errors.png` | Feature errors eᵤ and e_A during SERVO phases |
| `fig_avoidance.png` | Obstacle apparent size and FSM state trace |
| `fig_cmds.png` | Commanded v and ω over the full run |

Optional arguments:

```bash
python3 plot_results.py run.csv \
  --obs-trigger  150   \   # obstacle_min_sqrt_area in your yaml
  --obs-approach 350   \   # approach_obs_sqrt_area in your yaml
  --out ./figures/
```

---

## Key Parameters (`safe_vs_params.yaml`)

### Calibration — do these first

| Parameter | Where to calibrate | What it controls |
|---|---|---|
| `sqrt_area_star` | Place robot at goal distance, read `sqrt_area` from `/target_features` | Desired standoff |
| `focal_length_px` | Place target at 1.0 m, read `sqrt_area`, compute `f = sqrt_area / target_real_side_m` | MPC depth model |
| `square_turn_duration_s` | Command the robot to spin in place and time 90° | Turn accuracy in bypass |

### MPC weights

| Parameter | Default | Effect of raising |
|---|---|---|
| `Q_u` | 1.0 | Faster lateral centering |
| `Q_A` | 5.0 | Faster depth convergence |
| `R_v` | 30000.0 | More conservative forward speed |
| `R_w` | 5000.0 | More conservative turning |
| `S_v / S_w` | 50.0 | Smoother command transitions |

### Obstacle avoidance

| Parameter | Default | Notes |
|---|---|---|
| `obstacle_min_sqrt_area` | 150 | Lower = avoidance triggers earlier (more room) |
| `obstacle_center_band_frac` | 0.90 | Fraction of image width that activates avoidance |
| `approach_obs_sqrt_area` | 350 | Target apparent size before bypass begins (calibrate at desired approach distance) |
| `square_turn_duration_s` | 3.5 | Calibrate to achieve exactly 90° on your robot |
| `square_leg_1_duration_s` | 2.5 | Lateral clearance: `Δy ≈ v_leg × t_leg1` |
| `square_leg_2_duration_s` | 3.0 | Forward clearance: `Δx ≈ v_leg × t_leg2` |

---

## System Architecture

```
Camera ──► color_features_detector ──► /target_features  ──► safe_vs_controller ──► /cmd_vel ──► cmd_vel_to_wheels
                                   └──► /obstacle_features ─►                                          │
                                                                                                       ▼
                                   ◄────────────────── visual feedback ──────────── /VelocitySetL/R → PuzzleBot
```

### FSM states

```
ACQUIRE ──► SERVO ──► PRE_AVOID_STOP ──► APPROACH_OBSTACLE
              │                                  │
              ├──► HOLD                     SQ_TURN_1
              │                                  │
              └──► LOST                     SQ_LEG_1
                                                 │
                                           SQ_TURN_2
                                                 │
                                           SQ_LEG_2
                                                 │
                                           REACQUIRE ──► SERVO
```

---

## How the Controller Works

### Vision pipeline

Each camera frame is converted to HSV. Binary masks are generated separately for the green target and blue obstacle using configurable HSV bounds. After morphological filtering, the largest contour in each mask is selected and three features are extracted: horizontal centroid `u_c`, apparent size `√A`, and solidity. These are published as `TargetFeatures` and `ObstacleFeatures` messages at the camera frame rate.

### IBVS-MPC

The controller minimizes a quadratic cost over image feature errors across a prediction horizon of N=8 steps, subject to actuator limits. The image Jacobian **L** maps velocity commands `[v, ω]` to predicted feature changes `[u̇_c, √Ȧ]` and is re-evaluated at every tick from the current measurement. The resulting QP is solved with OSQP in under 10 ms.

### Obstacle avoidance

When the obstacle appears large and centered in the image, the robot stops, slowly approaches the obstacle while centering it in the frame, and then executes a two-turn, two-leg rectangular detour. The arc direction is determined from the obstacle's lateral position at the moment the approach threshold is reached, giving the controller the most reliable visual fix possible before committing to the maneuver.

---

## Troubleshooting

**`color_features_detector` not publishing:**  
Check the image topic. The detector defaults to `/video_source/raw`. Verify with:
```bash
ros2 node info /color_features_detector   # look at Subscribers
ros2 topic hz /video_source/raw           # check camera is running
```

**Build fails with `Goal.msg doesn't exist`:**  
You are running `colcon build` from the wrong directory. Run it from `~/ros2_ws`, not from your home folder:
```bash
cd ~/ros2_ws && colcon build --packages-select puzzlebot_mc2
```

**MPC import error (`qpsolvers not found`):**  
```bash
pip install qpsolvers[osqp] --break-system-packages
```

**Robot spins in place during SERVO:**  
`sqrt_area_star` is set too high — the target never appears large enough to satisfy the goal condition, so the robot keeps trying to approach but the centering error prevents forward motion. Place the robot at the desired stopping distance and read `√A` from `/target_features`.

**Avoidance triggers too late (robot too close to obstacle):**  
Lower `obstacle_min_sqrt_area`. The current value requires the obstacle to appear at a minimum size before avoidance fires; reducing it causes earlier triggering with more physical clearance room.

---

## License

MIT
