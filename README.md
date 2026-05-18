# Optimal Control for Visual Servoing — PuzzleBot
**TE3002B · Tecnológico de Monterrey**

> Closed-loop Image-Based Visual Servoing (IBVS) with obstacle avoidance on a PuzzleBot differential drive robot, using only a forward-facing camera.

---

## 📹 Demo Video
**[Watch on YouTube →](https://www.youtube.com/PLACEHOLDER)**

---

## Overview

The robot uses classical computer vision (HSV segmentation) to detect a **green target** and a **blue obstacle**. It servo-controls toward the target using an IBVS proportional controller derived from a Model Predictive Control formulation. When an obstacle is detected ahead, it executes an **approach-then-arc** avoidance maneuver:

1. **Stop** completely
2. **Approach** the obstacle slowly while centering it — establishes a known distance
3. **Arc** smoothly around it
4. **Reacquire** the target and continue

```
Camera → [Detector] → features → [Controller] → /cmd_vel → [Wheels] → Robot
                                      ↑___________________feedback___________|
```

---

## Repository Structure

```
puzzlebot_mc2/
├── scripts/
│   ├── safe_vs_controller.py       # Main controller (IBVS + avoidance FSM)
│   ├── color_features_detector.py  # HSV segmentation → image features
│   └── cmd_vel_to_wheels.py        # /cmd_vel → wheel velocity setpoints
├── config/
│   └── safe_vs_params.yaml         # All tunable parameters (documented)
├── launch/
│   └── safe_vs_demo.launch.py      # Launches all three nodes
├── report/
│   └── technical_report.tex        # IEEE-format technical report (LaTeX)
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 |
| ROS 2 | Humble Hawksbill |
| Python | 3.10+ |
| OpenCV | 4.x (`cv_bridge`) |

The `puzzlebot_mc2` package must already be set up in your ROS 2 workspace with the custom messages (`TargetFeatures`, `ObstacleFeatures`).

---

## Installation

```bash
# 1. Clone into your ROS 2 workspace source directory
cd ~/ros2_ws/src
git clone https://github.com/PLACEHOLDER/puzzlebot-vs.git puzzlebot_mc2

# 2. Build
cd ~/ros2_ws
colcon build --packages-select puzzlebot_mc2
source install/setup.bash
```

---

## Running

```bash
# Source your workspace (if not in .bashrc already)
source ~/ros2_ws/install/setup.bash

# Launch everything (detector + controller + wheel driver)
ros2 launch puzzlebot_mc2 safe_vs_demo.launch.py
```

That single command starts all three nodes with the parameters from `config/safe_vs_params.yaml`.

---

## Monitoring in Real Time

Open separate terminals for each:

```bash
# FSM state (ACQUIRE / SERVO / APPROACH_OBSTACLE / AVOID_ARC / ...)
ros2 topic echo /vs_state

# Human-readable debug: state, danger flag, obstacle size, reason
ros2 topic echo /avoid_debug

# Raw target features (u_c, sqrt_area, detected, solidity)
ros2 topic echo /target_features

# Raw obstacle features
ros2 topic echo /obstacle_features

# Velocity commands being sent
ros2 topic echo /cmd_vel
```

---

## Tuning Guide

All parameters live in `config/safe_vs_params.yaml`. No code changes are needed for tuning.

### Step 1 — Calibrate the target setpoint

Place the robot at the exact desired stopping distance from the green target. Run the stack and read `sqrt_area` from `/target_features`. Set that value as `sqrt_area_star`.

### Step 2 — Tune HSV bounds

Lighting changes will shift hue/saturation values. Run with `rqt_image_view` to visualize the segmentation mask, or echo `/target_features` and `/obstacle_features` while moving the target/obstacle into frame. Adjust `green_h_min/max` and `blue_h_min/max` until detection is clean.

### Step 3 — Tune servo gains

Watch `/avoid_debug` with the robot approaching the target:
- **Wavy path** → lower `kp_w` or lower `v_stop_band_px`
- **Too slow to center** → raise `kp_w`
- **Oscillates around goal** → lower `kp_v` or raise `target_area_deadband`

### Step 4 — Tune obstacle trigger

Watch `/avoid_debug` as the robot approaches the obstacle:
- `"too small"` always showing → lower `obstacle_min_sqrt_area`
- Avoidance fires too far away → raise `obstacle_min_sqrt_area`

### Step 5 — Calibrate approach endpoint

Place the obstacle at the distance where you want the arc to start. Read `sqrt_area` from `/obstacle_features`. Set that as `approach_obs_sqrt_area`.

### Step 6 — Tune the arc

The arc radius is `r = avoid_arc_v / avoid_arc_w`. Lateral displacement is approximately `r · (1 − cos(avoid_arc_w · avoid_arc_duration_s))`.

- **Robot clips obstacle** → raise `avoid_arc_w` (tighter radius) or increase `avoid_arc_duration_s`
- **Arc too sharp / aggressive** → lower `avoid_arc_w` (wider radius)
- **Robot doesn't reacquire target** → increase `avoid_recenter_duration_s`

---

## System Architecture

### Nodes and Topics

```
/video_source/raw  (sensor_msgs/Image)
        │
        ▼
[color_features_detector]
        │
        ├─── /target_features   (puzzlebot_mc2/TargetFeatures)
        └─── /obstacle_features (puzzlebot_mc2/ObstacleFeatures)
                        │
                        ▼
              [safe_vs_controller]
                        │
                        └─── /cmd_vel (geometry_msgs/Twist)
                                  │
                                  ▼
                       [cmd_vel_to_wheels]
                                  │
                        ┌─────────┴──────────┐
                        ▼                    ▼
               /VelocitySetL        /VelocitySetR
               (std_msgs/Float32)   (std_msgs/Float32)
```

### FSM States

| State | Description |
|---|---|
| `ACQUIRE` | Rotate in place until green target is found |
| `SERVO` | IBVS proportional control toward green target |
| `PRE_AVOID_STOP` | Full stop before avoidance maneuver |
| `APPROACH_OBSTACLE` | Center and advance toward blue obstacle to get visual fix |
| `AVOID_ARC` | Smooth arc around the obstacle |
| `AVOID_RECENTER` | Counter-rotation to realign with target area |
| `REACQUIRE` | Search for target after avoidance |
| `HOLD` | Goal reached — stop and hold |
| `LOST` | Target disappeared — search again |

### Control Law

The proportional IBVS law with centering-scale coupling:

```
e_u = u_c - c_x                          (lateral error, pixels)
e_A = sqrt_area_star - sqrt_area          (area error, pixels)

w = -kp_w * e_u                           (angular command)
v = kp_v * e_A                            (forward command, before scaling)

# Centering-scale: eliminate wave motion
if |e_u| > v_stop_band_px:
    v = 0                                 (pure rotation when off-center)
else:
    v = v * (1 - |e_u| / v_stop_band_px) (linear ramp near center)
```

---

## Key Parameters Reference

| Parameter | Default | Effect |
|---|---|---|
| `sqrt_area_star` | 850 px | Target stopping distance |
| `kp_w` | 0.0025 | Angular servo gain |
| `kp_v` | 0.0020 | Linear servo gain |
| `v_stop_band_px` | 60 px | Centering band width |
| `obstacle_min_sqrt_area` | 150 px | Obstacle trigger threshold |
| `approach_obs_sqrt_area` | 350 px | Approach endpoint |
| `approach_obs_v` | 0.07 m/s | Approach speed |
| `avoid_arc_v` | 0.07 m/s | Arc forward speed |
| `avoid_arc_w` | 0.35 rad/s | Arc turn rate |
| `avoid_arc_duration_s` | 4.0 s | Arc duration |

---

## Troubleshooting

**Robot doesn't detect the target**
→ Check `green_h_min/max` under your lighting. Echo `/target_features` and look at the `detected` field.

**Robot detects target but doesn't move**
→ Check `sqrt_area_star`. If `sqrt_area` is already >= `sqrt_area_star`, the robot thinks it's at the goal.

**Avoidance never triggers**
→ Echo `/avoid_debug`. If it says `"too small"`, lower `obstacle_min_sqrt_area`. If it says `"outside band"`, the obstacle may be approaching at an angle — check `obstacle_center_band_frac`.

**Robot clips the obstacle during the arc**
→ Raise `avoid_arc_w` to tighten the arc radius, or raise `avoid_arc_duration_s` to extend the arc further past the obstacle.

**Robot doesn't reacquire target after avoidance**
→ Increase `avoid_recenter_duration_s` so the robot turns further back toward the target area.

---

## License

Academic project — Tecnológico de Monterrey, 2025.
