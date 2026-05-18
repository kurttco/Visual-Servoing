#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class CmdVelToWheels(Node):
    """
    Converts /cmd_vel into /VelocitySetL and /VelocitySetR.
    """

    def __init__(self):
        super().__init__('cmd_vel_to_wheels')

        self.declare_parameter('k_lin', 0.074)
        self.declare_parameter('k_ang', 0.561)
        self.declare_parameter('max_wheel_cmd', 2.8)

        self.k_lin = float(self.get_parameter('k_lin').value)
        self.k_ang = float(self.get_parameter('k_ang').value)
        self.max_wheel_cmd = float(self.get_parameter('max_wheel_cmd').value)

        self.pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)

        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cb_cmd,
            10
        )

        self.get_logger().info(
            f'cmd_vel_to_wheels running | '
            f'k_lin={self.k_lin:.3f}, k_ang={self.k_ang:.3f}, '
            f'max_cmd={self.max_wheel_cmd:.2f}'
        )

    def cb_cmd(self, msg):
        v = float(msg.linear.x)
        w = float(msg.angular.z)

        base = v / self.k_lin if abs(self.k_lin) > 1e-6 else 0.0
        turn = w / self.k_ang if abs(self.k_ang) > 1e-6 else 0.0

        cmd_l = base - turn
        cmd_r = base + turn

        cmd_l = clip(cmd_l, -self.max_wheel_cmd, self.max_wheel_cmd)
        cmd_r = clip(cmd_r, -self.max_wheel_cmd, self.max_wheel_cmd)

        ml = Float32()
        mr = Float32()

        ml.data = float(cmd_l)
        mr.data = float(cmd_r)

        self.pub_l.publish(ml)
        self.pub_r.publish(mr)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToWheels()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Float32()
        stop.data = 0.0

        try:
            for _ in range(10):
                node.pub_l.publish(stop)
                node.pub_r.publish(stop)
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
