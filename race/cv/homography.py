import numpy as np
import cv2

class Homography():
    def __init__(self):
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

    def build_homography(self):
        """
        Upon clicking a known real world point, the pixel coordinate is printed.
        """
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
                pixel_value = frame[y, x]  # Note: indexing is (row, column)
                print(f'Pixel at ({x}, {y}): {pixel_value}')

        video_path = "../../vids/homography.mp4"
        paused = False
        frame = None
        cap = cv2.VideoCapture(video_path)
        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', mouse_callback)

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or can't read the frame.")
                    break
                cv2.imshow('Video', frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Press space to toggle pause
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()

    def test_homography(self, frame):
        """
        Test homography matrix.
        """
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
                pixel = np.array([[x, y, 1]], dtype='float32').T
                world = self.h @ pixel
                world /= world[2, 0]
                print(world[0, 0], world[1, 0])

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", mouse_callback)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
