import face_recognition
import cv2
import numpy as np

# Initialize two video capture objects for two cameras
video_capture_1 = cv2.VideoCapture(0)
video_capture_2 = cv2.VideoCapture(1)

# Rest of your code...

while True:
    # Grab a single frame from each video source
    ret_1, frame_1 = video_capture_1.read()
    ret_2, frame_2 = video_capture_2.read()

    # Combine frames horizontally
    combined_frame = np.concatenate((frame_1, frame_2), axis=1)

    # Perform face recognition for the combined frame

    # Rest of your code...

    # Display the resulting combined image
    cv2.imshow('Combined Video', combined_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handles to the webcams
video_capture_1.release()
video_capture_2.release()
cv2.destroyAllWindows()
