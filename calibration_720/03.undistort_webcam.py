import numpy as np
import cv2 as cv
import pickle

# Load the saved calibration data
with open('calibration_720/calibration_720.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Open the webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Process webcam frames
while True:
    # Capture the frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Get the frame size
    h, w = frame.shape[:2]

    # Undistort the image
    undistorted_frame = cv.undistort(frame, cameraMatrix, dist, None)

    # Display the original frame
    cv.imshow('Original Frame', frame)

    # Display the undistorted frame
    cv.imshow('Undistorted Frame', undistorted_frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv.destroyAllWindows()

