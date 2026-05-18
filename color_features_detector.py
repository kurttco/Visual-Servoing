#!/usr/bin/env python3

import cv2
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from puzzlebot_mc2.msg import TargetFeatures, ObstacleFeatures


class ColorFeaturesDetector(Node):
    """
    Detects:
      - Green target  -> /target_features
      - Blue obstacle -> /obstacle_features
    """

    def __init__(self):
        super().__init__('color_features_detector')

        # ============================================================
        # Camera topic
        # ============================================================
        self.declare_parameter('image_topic', '/image_raw')

        # ============================================================
        # Green target HSV
        # ============================================================
        self.declare_parameter('green_h_min', 40)
        self.declare_parameter('green_s_min', 50)
        self.declare_parameter('green_v_min', 40)

        self.declare_parameter('green_h_max', 90)
        self.declare_parameter('green_s_max', 255)
        self.declare_parameter('green_v_max', 255)

        # ============================================================
        # Blue obstacle HSV
        # ============================================================
        self.declare_parameter('blue_h_min', 90)
        self.declare_parameter('blue_s_min', 50)
        self.declare_parameter('blue_v_min', 35)

        self.declare_parameter('blue_h_max', 135)
        self.declare_parameter('blue_s_max', 255)
        self.declare_parameter('blue_v_max', 255)

        # ============================================================
        # Filtering
        # ============================================================
        self.declare_parameter('min_target_area_px', 250.0)
        self.declare_parameter('min_obstacle_area_px', 200.0)

        self.declare_parameter('morph_kernel_size', 5)

        self.declare_parameter('roi_y_min_frac', 0.0)
        self.declare_parameter('roi_y_max_frac', 1.0)

        image_topic = self.get_parameter('image_topic').value

        self.bridge = CvBridge()

        self.sub_img = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.pub_target = self.create_publisher(
            TargetFeatures,
            '/target_features',
            10
        )

        self.pub_obstacle = self.create_publisher(
            ObstacleFeatures,
            '/obstacle_features',
            10
        )

        self.get_logger().info(
            f'Color features detector running | image_topic={image_topic}'
        )


    def get_hsv_range(self, prefix):
        h_min = int(self.get_parameter(f'{prefix}_h_min').value)
        s_min = int(self.get_parameter(f'{prefix}_s_min').value)
        v_min = int(self.get_parameter(f'{prefix}_v_min').value)

        h_max = int(self.get_parameter(f'{prefix}_h_max').value)
        s_max = int(self.get_parameter(f'{prefix}_s_max').value)
        v_max = int(self.get_parameter(f'{prefix}_v_max').value)

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        return lower, upper

    def detect_largest_blob(self, hsv_roi, lower, upper, min_area_px, y_offset):
        mask = cv2.inRange(hsv_roi, lower, upper)

        k_size = int(self.get_parameter('morph_kernel_size').value)
        k_size = max(3, k_size)

        kernel = np.ones((k_size, k_size), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        sqrt_area = math.sqrt(max(area, 0.0))

        if area < min_area_px:
            return False, 0.0, 0.0, area, sqrt_area, 0.0

        M = cv2.moments(contour)

        if abs(M['m00']) < 1e-6:
            return False, 0.0, 0.0, area, sqrt_area, 0.0

        u_c = float(M['m10'] / M['m00'])
        v_c = float(M['m01'] / M['m00']) + float(y_offset)

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))

        solidity = area / hull_area if hull_area > 1e-6 else 0.0

        return True, u_c, v_c, area, sqrt_area, solidity

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        H, W = frame.shape[:2]

        y_min_frac = float(self.get_parameter('roi_y_min_frac').value)
        y_max_frac = float(self.get_parameter('roi_y_max_frac').value)

        y_min_frac = max(0.0, min(1.0, y_min_frac))
        y_max_frac = max(0.0, min(1.0, y_max_frac))

        y0 = int(y_min_frac * H)
        y1 = int(y_max_frac * H)

        if y1 <= y0:
            y0, y1 = 0, H

        roi = frame[y0:y1, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        green_lower, green_upper = self.get_hsv_range('green')
        blue_lower, blue_upper = self.get_hsv_range('blue')

        min_target_area = float(self.get_parameter('min_target_area_px').value)
        min_obstacle_area = float(self.get_parameter('min_obstacle_area_px').value)

        target_detected, tu, tv, ta, tsa, tsol = self.detect_largest_blob(
            hsv,
            green_lower,
            green_upper,
            min_target_area,
            y0
        )

        obstacle_detected, ou, ov, oa, osa, osol = self.detect_largest_blob(
            hsv,
            blue_lower,
            blue_upper,
            min_obstacle_area,
            y0
        )

        target_msg = TargetFeatures()
        target_msg.header = msg.header
        target_msg.detected = bool(target_detected)
        target_msg.u_c = float(tu)
        target_msg.v_c = float(tv)
        target_msg.area_px = float(ta)
        target_msg.sqrt_area = float(tsa)
        target_msg.solidity = float(tsol)

        obstacle_msg = ObstacleFeatures()
        obstacle_msg.header = msg.header
        obstacle_msg.detected = bool(obstacle_detected)
        obstacle_msg.u_c = float(ou)
        obstacle_msg.v_c = float(ov)
        obstacle_msg.area_px = float(oa)
        obstacle_msg.sqrt_area = float(osa)
        obstacle_msg.solidity = float(osol)

        self.pub_target.publish(target_msg)
        self.pub_obstacle.publish(obstacle_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ColorFeaturesDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
