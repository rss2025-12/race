import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
from cv.race_cv import RaceCV
from cv.homography import Homography
import numpy as np
import csv, os

class Race(Node):
    def __init__(self):
        super().__init__("race")
        self.declare_parameter('drive_topic', "/vesc/high_level/input/nav_0")
        self.declare_parameter('drive_speed', 4.0)

        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.drive_speed = self.get_parameter('drive_speed').get_parameter_value().double_value

        # Subscribe to ZED camera RGB frames
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

        self.race = True
        self.race_cv = RaceCV()
        self.homography = Homography()

        self.prev_wp_angle = 0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        self.write_data = False
        if self.write_data is True:
            output_path = os.path.join(os.path.dirname(__file__), '../data/crosstrack_two.csv') # File name
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.csv_file = open(output_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'cross_track_error'])

        if self.race is True:
            self.get_logger().info(f"Race node initialized with speed {self.drive_speed}")
        else:
            self.get_logger().info("Race node initialized")

    def image_callback(self, image_msg):
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        if self.race is False:
            # self.race_cv.show_video(img)
            # self.race_cv.record_video(img)
            # self.homography.test_homography(img)
            return

        x_target, y_target  = self.race_cv.lane_following(img)

        if x_target is None:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_msg)
            return

        if self.write_data is True:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.csv_writer.writerow([timestamp, y_target])

        steering_angle = self.steering_angle(x_target, y_target)
        # self.get_logger().info(steering_angle)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.drive_speed
        drive_msg.drive.steering_angle = steering_angle

        self.drive_pub.publish(drive_msg)

    def steering_angle(self, x_target, y_target):
        """
        x is forward
        y is right
        """
        dx = x_target
        dy = y_target
        angle_to_wp = np.arctan2(dy, dx)

        current_time = self.get_clock().now().nanoseconds / 1e9
        angle_derivative = (angle_to_wp - self.prev_wp_angle) / (current_time - self.prev_time)

        self.prev_time = current_time
        self.prev_wp_angle = angle_to_wp

        wheelbase = 0.3
        lookahead = 2.7
        angle = 1 * angle_to_wp + 0.1 * angle_derivative
        alpha = np.arctan2(np.sin(angle), np.cos(angle))
        steering_angle = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead)

        return steering_angle


def main(args=None):
    rclpy.init(args=args)
    node = Race()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
