import cv2 as cv
import numpy as np
import os

# Configuration
CHESS_BOARD_DIM = (7, 7)
image_dir_path = "images"
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create images directory if not exists
if not os.path.isdir(image_dir_path):
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already Exists.')

# Capture and save checkerboard images
cap = cv.VideoCapture(0)
n = 0  # Image counter

def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)
    return image, ret

while True:
    ret, frame = cap.read()
    if not ret:
        break
    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(frame, gray, criteria, CHESS_BOARD_DIM)

    cv.putText(frame, f"saved_img : {n}", (30, 40), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow("frame", frame)
    cv.imshow("copyFrame", copyFrame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s") and board_detected:
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)
        print(f"saved image number {n}")
        n += 1

cap.release()
cv.destroyAllWindows()
print("Total saved Images:", n)

# -------------------------------------
# Calibration Phase
# -------------------------------------
print("\nStarting Calibration...")

# Prepare object points
objp = np.zeros((CHESS_BOARD_DIM[0]*CHESS_BOARD_DIM[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0], 0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load saved images and detect corners
for i in range(n):
    img = cv.imread(f"{image_dir_path}/image{i}.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHESS_BOARD_DIM, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Perform calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# -------------------------------------
# Undistort one image and crop it
# -------------------------------------
img = cv.imread(f"{image_dir_path}/image0.png")
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Crop and save
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
print("Undistorted and cropped image saved as 'calibresult.png'")

# -------------------------------------
# Re-projection error
# -------------------------------------
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total re-projection error: {}".format(mean_error / len(objpoints)))
