import cv2
import numpy as np

class RaceCV():
    def __init__(self):
        self.out = None
        self.prev_lx = []
        self.prev_rx = []
        self.prev_left_fit = []
        self.prev_right_fit = []

        PTS_IMAGE_PLANE = [[33, 277], [358, 235], [543, 240],
                           [129, 203], [350, 210], [476, 213],
                           [176, 181], [341, 187], [435, 189],
                           [201, 175], [335, 177], [411, 177],
                           [230, 170], [342, 173], [405, 174],
                           [234, 155], [328, 158], [382, 158],
                           [252, 151], [336, 156], [382, 156]]

        PTS_GROUND_PLANE = [[24, 23.5], [24, 0], [24, -13.25],
                            [36, 23.5], [36, 0], [36, -13.25],
                            [48, 23.5], [48, 0], [48, -13.25],
                            [60, 23.5], [60, 0], [60, -13.25],
                            [72, 23.5], [72, 0], [72, -13.25],
                            [84, 23.5], [84, 0], [84, -13.25],
                            [96, 23.5], [96, 0], [96, -13.25]]

        METERS_PER_INCH = 0.0254

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, _ = cv2.findHomography(np_pts_image, np_pts_ground)

    def show_video(self, img):
        """
        Show incoming images.
        """
        cv2.imshow("image", img)

        ### Continuous streaming ###
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def record_video(self, img):
        """
        Record incoming images.
        """
        if img is None:
            if self.out is not None:
                self.out.release()
            return

        height, width, _ = img.shape
        fps = 30
        frame_size = (width, height)

        if self.out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter('src/race/vids/homography.mp4', fourcc, fps, frame_size)

        self.out.write(img)

    def lane_following(self, frame):
        """
        Returns center offset given image of lane.
        """
        def get_centroid_x(contour):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                return int(M["m10"] / M["m00"])
            return None

        frame_height, frame_width, _ = frame.shape # (360, 640, 3)

        ### Area of focus ###
        tl = (frame_width // 2 - 105 + 20, 150)
        tr = (frame_width // 2 + 105 - 10, 150)
        bl = (20, 215)
        br = (640 - 10, 215)

        # Old: l: 35, r: 15

        # Draw area of focus
        # cv2.circle(frame, tl, 5, (0, 0, 255), -1)
        # cv2.circle(frame, tr, 5, (0, 0, 255), -1)
        # cv2.circle(frame, bl, 5, (0, 0, 255), -1)
        # cv2.circle(frame, br, 5, (0, 0, 255), -1)

        pts1 = np.float32([tl, tr, bl, br])
        pts2 = np.float32([
            (0, 0),       # top-left
            (640, 0),     # top-right
            (0, 360),     # bottom-left
            (640, 360)    # bottom-right
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(frame, matrix, (640, 360))

        ### Lane Detection ###
        hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

        lh = 0
        ls = 0
        lv = 150
        uh = 255
        us = 50
        uv = 255

        lower = np.array([lh, ls, lv])
        upper = np.array([uh, us, uv])
        mask = cv2.inRange(hsv, lower, upper)
        original_mask = mask.copy()

        ### Mask processing ###
        vertical_kernel = np.ones((1, 5), np.uint8)
        horizontal_kernel = np.ones((10, 1), np.uint8)

        mask = cv2.dilate(mask, vertical_kernel, iterations=3)
        mask = cv2.erode(mask, horizontal_kernel, iterations=8)

        ### Histogram ###
        hist_offset = 110
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        left_base = np.argmax(histogram[:midpoint - hist_offset])
        right_base = np.argmax(histogram[midpoint + hist_offset:]) + midpoint + hist_offset

        ### Sliding window ###
        win_width = 50 # Half of width
        win_height = 40
        y = mask.shape[0] # Start at bottom

        lx, rx = [], []
        left_bases, right_bases = [left_base], [right_base]

        window_mask = mask.copy()
        while y > 0:
            y_top = max(0, y - win_height)

            # Left window
            left_window = mask[y_top:y, max(0, left_base - win_width):left_base + win_width]
            contours, _ = cv2.findContours(left_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cx = get_centroid_x(contour)
                if cx is not None:
                    left_base = max(0, left_base - win_width) + cx
                    lx.append(left_base)
                    left_bases.append(left_base)
                    break  # Use only the first valid contour

            # Right
            right_window = mask[y_top:y, max(0, right_base - win_width):right_base + win_width]
            contours, _ = cv2.findContours(right_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cx = get_centroid_x(contour)
                if cx is not None:
                    right_base = max(0, right_base - win_width) + cx
                    rx.append(right_base)
                    right_bases.append(right_base)
                    break

            cv2.rectangle(window_mask, (left_base - win_width, y), (left_base + win_width, y - win_height), (255, 255, 255), 2)
            cv2.rectangle(window_mask, (right_base - win_width, y), (right_base + win_width, y - win_height), (255, 255, 255), 2)

            y -= win_height

        ### Lane coordinates ###
        if len(lx) == 0:
            lx = self.prev_lx
        else:
            self.prev_lx = lx
        if len(rx) == 0:
            rx = self.prev_rx
        else:
            self.prev_rx = rx
        min_length = min(len(lx), len(rx))

        if min_length == 0:
            print("No lanes detected")
            return None, None

        # Offset
        window_index = 9
        available_index = min(window_index, min_length - 1)

        left_base = lx[available_index]
        right_base = rx[available_index]
        x_center = (left_base + right_base) // 2
        y_center = mask.shape[0] - win_height * available_index

        # Transform to original image
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        lane_center_bev = np.array([[[x_center, y_center]]], dtype=np.float32)
        lane_center_img = cv2.perspectiveTransform(lane_center_bev, inv_matrix)
        x_img, y_img = lane_center_img[0][0]

        # Project to world coordinates
        img_point = np.array([[x_img, y_img, 1]]).T  # Shape (3, 1)
        world_point = np.dot(self.h, img_point)
        world_point /= world_point[2, 0]  # Normalize homogeneous coordinate

        x_world = world_point[0, 0]
        y_world = world_point[1, 0]

        cv2.circle(frame, (int(x_img), int(y_img)), 5, (0, 255, 255), -1)

        ### Overlay ###
        # top_left = (lx[min_length - 1], win_height * (9 - min_length))
        # top_right = (rx[min_length - 1], win_height * (9 - min_length))
        # bottom_left = (lx[0], frame_height)
        # bottom_right = (rx[0], frame_height)

        # quad_points = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.int32)
        # quad_points = quad_points.reshape((-1, 1, 2))
        # overlay = transformed_frame.copy()
        # cv2.fillPoly(overlay, [quad_points], (0, 255, 0))
        # alpha = 0.5
        # cv2.addWeighted(transformed_frame, alpha, overlay, 1 - alpha, 0, overlay)
        # inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        # overlay_original = cv2.warpPerspective(overlay, inv_matrix, (640, 360))
        # result = cv2.addWeighted(frame, 1, overlay_original, 0.5, 0)

        ### Visualization ###
        # cv2.imshow("Original", frame)
        # cv2.imshow("Bird's Eye View", transformed_frame)
        # cv2.imshow("Original Mask", original_mask)
        # cv2.imshow("Lane Detection", mask)
        # cv2.imshow("Sliding Windows", window_mask)
        # cv2.imshow("Overlay", overlay)
        # cv2.imshow("Overlay Original", overlay_original)
        # cv2.imshow("Result", result)

        ### Continuous streaming ###
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        return x_world, y_world
