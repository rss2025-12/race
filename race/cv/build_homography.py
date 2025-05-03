import cv2

video_path = "../../vids/homography.mp4"
paused = False
frame = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        pixel_value = frame[y, x]  # Note: indexing is (row, column)
        print(f'Pixel at ({x}, {y}): {pixel_value}')

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
