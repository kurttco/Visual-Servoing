# TargetFeatures.msg
# Image-space features of the visual servoing target.
# Published by target_detector.py, consumed by vs_controller.py.

std_msgs/Header header

# True if a valid blob was found in this frame.
bool detected

# Horizontal centroid of the visible target portion, in pixels.
# Range: [0, image_width]. Undefined when detected == false.
float32 u_c

# Square root of contour area, in sqrt(pixels^2).
# Used instead of plain area because it is linear in 1/Z, which keeps the
# image Jacobian well-conditioned. Zero when detected == false.
float32 sqrt_area

# Contour area divided by convex hull area. Range [0, 1].
# Solidity ~= 1.0 -> blob is convex (clean target view).
# Solidity < 0.85 -> visible region has notches (likely partial occlusion).
# This is the controller's occlusion signal. Zero when detected == false.
float32 solidity

# Raw contour area in pixels, kept for diagnostics and CSV logging.
float32 area_px

float32 v_c
