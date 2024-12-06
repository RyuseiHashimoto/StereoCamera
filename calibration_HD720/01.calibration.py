import numpy as np
import cv2 as cv
import glob
import pickle

# Chessboard settings
chessboardSize = (6, 9)

# Load frame size from file
with open('calibration_HD720/frameSize_HD720.pkl', 'rb') as f:
    frameSize = pickle.load(f)
    print(f"Loaded frame size: {frameSize}")

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real world space)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Size of chessboard squares in mm
size_of_chessboard_squares_mm = 23
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load chessboard images
images = glob.glob('calibration_HD720/images/*.png')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If corners are found, refine them and store the points
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

# Camera calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the calibration results for later use
pickle.dump((cameraMatrix, dist), open("calibration_HD720/calibration_HD720.pkl", "wb"))
pickle.dump(cameraMatrix, open("calibration_HD720/cameraMatrix_HD720.pkl", "wb"))
pickle.dump(dist, open("calibration_HD720/dist.pkl", "wb"))

# Save object points and image points
pickle.dump(objpoints, open("calibration_HD720/objpoints_HD720.pkl", "wb"))
pickle.dump(imgpoints, open("calibration_HD720/imgpoints_HD720.pkl", "wb"))

# Save rvecs and tvecs for reprojection error calculation
pickle.dump(rvecs, open("calibration_HD720/rvecs_HD720.pkl", "wb"))
pickle.dump(tvecs, open("calibration_HD720/tvecs_HD720.pkl", "wb"))

print("Camera calibration completed and saved.")

# Load the calibration data
with open('calibration_HD720/calibration_HD720.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Print the focal lengths
f_x = cameraMatrix[0, 0]
f_y = cameraMatrix[1, 1]

print(f"焦点距離 (f_x): {f_x} px")
print(f"焦点距離 (f_y): {f_y} px")