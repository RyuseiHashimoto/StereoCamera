import numpy as np
import cv2
import pickle

T = 70  # Distance between cameras (mm)

# Load calibration results (calibration.pkl)
with open('calibration_720/calibration_720.pkl', 'rb') as f:
    cameraMatrix_720, dist_720 = pickle.load(f)
with open('calibration_HD720/calibration_HD720.pkl', 'rb') as f:
    cameraMatrix_HD720, dist_HD720 = pickle.load(f)

# Get focal lengths
F1 = cameraMatrix_720[0, 0]  # Focal length in x direction for 720 camera (px)
F2 = cameraMatrix_HD720[0, 0]  # Focal length in x direction for HD720 camera (px)

# Open webcams
cap_HD720 = cv2.VideoCapture(0)
cap_720 = cv2.VideoCapture(1)

# Set frame rate
cap_HD720.set(cv2.CAP_PROP_FPS, 5)
cap_720.set(cv2.CAP_PROP_FPS, 5)

# Check if cameras are successfully opened
if not cap_720.isOpened() or not cap_HD720.isOpened():
    print("Failed to open cameras")
    exit()

# Main process
while True:
    # Read frames
    ret_720, frame_720 = cap_720.read()
    ret_HD720, frame_HD720 = cap_HD720.read()

    # Verify frames are successfully captured
    if not ret_720 or not ret_HD720:
        print("Failed to capture frame. Exiting...")
        break

    # Get frame dimensions
    h, w = frame_720.shape[:2]

    # Undistort frames
    undistorted_720 = cv2.undistort(frame_720, cameraMatrix_720, dist_720, None)
    undistorted_HD720 = cv2.undistort(frame_HD720, cameraMatrix_HD720, dist_HD720, None)

    # Display original frames (commented out)
    # cv2.imshow('Original 720 Frame', frame_720)
    # cv2.imshow('Original HD720 Frame', frame_HD720)

    # Display undistorted frames
    cv2.imshow('Undistorted 720 Frame', undistorted_720)
    cv2.imshow('Undistorted HD720 Frame', undistorted_HD720)

    # Handle key inputs
    key = cv2.waitKey(1) & 0xFF

    # Press "c" to save images from both cameras
    if key == ord('c'):
        cv2.imwrite('720_image.jpg', undistorted_720)
        cv2.imwrite('HD720_image.jpg', undistorted_HD720)
        print("Saved images from both cameras")

    # Press "e" to calculate distance
    if key == ord('e'):
        # Load images captured by both cameras
        image = cv2.imread('720_image.jpg')
        HD_image = cv2.imread('HD720_image.jpg')

        # Verify images are loaded successfully
        if image is None or HD_image is None:
            print("Failed to load images.")
            continue

        # Convert images to HSV color space
        hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(HD_image, cv2.COLOR_BGR2HSV)

        # Filter frames and generate masks (extract red color)
        lower_colore = np.array([0, 100, 100])
        upper_colore = np.array([15, 255, 255])
        mask1 = cv2.inRange(hsv1, lower_colore, upper_colore)
        mask2 = cv2.inRange(hsv2, lower_colore, upper_colore)

        # Find centroid in image 1
        M1 = cv2.moments(mask1)
        if M1["m00"] != 0:
            cx1 = int(M1["m10"] / M1["m00"])
            cy1 = int(M1["m01"] / M1["m00"])
            pt1 = (cx1, cy1)
            cv2.circle(mask1, pt1, 10, (100), 3)
        else:
            pt1 = None

        # Find centroid in image 2
        M2 = cv2.moments(mask2)
        if M2["m00"] != 0:
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])
            pt2 = (cx2, cy2)
            cv2.circle(mask2, pt2, 10, (100), 3)
        else:
            pt2 = None

        # Display masks
        cv2.imshow("720_mask_image", mask1)
        cv2.imshow("light_mask_image", mask2)

        # Calculate distance
        if pt1 is not None and pt2 is not None:
            D = abs(pt1[0] - pt2[0])  # Pixel difference
            if D != 0:
                z1 = (F1 * T) / D  # Estimated distance 1
                z2 = (F2 * T) / D  # Estimated distance 2
                z = (z1 + z2) / 2  # Average estimated distance
                print(f"Estimated distance: {z:.2f} mm")

            else:
                print("The difference in centroid positions is zero. Cannot calculate distance.")
        else:
            print("Centroids not detected, distance calculation failed.")
        
    # Press "q" to exit
    if key == ord('q'):
        print("Exiting program")
        break

# Release cameras and close windows
cap_720.release()
cap_HD720.release()
cv2.destroyAllWindows()
