import numpy as np
import cv2
import pickle

T = 70  # Distance between cameras (mm)

# Load calibration results
with open('calibration_720/calibration_720.pkl', 'rb') as f:
    cameraMatrix_720, dist_720 = pickle.load(f)
with open('calibration_HD720/calibration_HD720.pkl', 'rb') as f:
    cameraMatrix_HD720, dist_HD720 = pickle.load(f)

# Get focal lengths
F1 = cameraMatrix_720[0, 0]  # Focal length in the x-direction for the 720 camera (px)
F2 = cameraMatrix_HD720[0, 0]  # Focal length in the x-direction for the HD720 camera (px)

# Open webcams
cap_HD720 = cv2.VideoCapture(0)
cap_720 = cv2.VideoCapture(1)

if not cap_720.isOpened() or not cap_HD720.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Main process
while True:
    # Read frames
    ret_720, frame_720 = cap_720.read()
    ret_HD720, frame_HD720 = cap_HD720.read()

    if not ret_720 or not ret_HD720:
        print("Failed to capture frame. Exiting...")
        break

    # Undistort frames
    undistorted_720 = cv2.undistort(frame_720, cameraMatrix_720, dist_720, None)
    undistorted_HD720 = cv2.undistort(frame_HD720, cameraMatrix_HD720, dist_HD720, None)

    # Convert images to HSV color space
    hsv1 = cv2.cvtColor(undistorted_720, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(undistorted_HD720, cv2.COLOR_BGR2HSV)

    # Specify lower and upper bounds for [Hue (H), Saturation (S), Value (V)]
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv1, lower_color, upper_color)
    mask2 = cv2.inRange(hsv2, lower_color, upper_color)

    # Calculate centroids
    M1 = cv2.moments(mask1)
    M2 = cv2.moments(mask2)
    if M1["m00"] != 0 and M2["m00"] != 0:
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])

        # Calculate pixel difference and distance
        D = abs(cx1 - cx2)
        if D != 0:
            z1 = (F1 * T) / D
            z2 = (F2 * T) / D
            z = (z1 + z2) / 2
            print(f"Estimated distance: {z:.2f} mm")

            # Draw centroids and distance on the frames
            cv2.circle(undistorted_720, (cx1, cy1), 5, (0, 255, 0), -1)
            cv2.putText(undistorted_720, f"Distance: {z:.2f} mm", (cx1 + 10, cy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(undistorted_HD720, (cx2, cy2), 5, (0, 255, 0), -1)
            cv2.putText(undistorted_HD720, f"Distance: {z:.2f} mm", (cx2 + 10, cy2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            print("The difference in centroid positions is zero. Cannot calculate distance.")
    else:
        print("Centroids not detected. Distance calculation failed.")

    # Display undistorted frames
    cv2.imshow('720 Frame', undistorted_720)
    cv2.imshow('HD720 Frame', undistorted_HD720)

    # Press "q" to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program")
        break

# Release resources
cap_720.release()
cap_HD720.release()
cv2.destroyAllWindows()
