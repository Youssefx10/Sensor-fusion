import cv2
import numpy as np
import os

print("loading data stored using numpy savez function\n \n \n")

calib_data_path = "calib_data"
CHECK_DIR = os.path.isdir(calib_data_path)

data = np.load(f"{calib_data_path}/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("loaded calibration data successfully")

print("Camera Matrix (camMatrix):\n", camMatrix)
print("\nDistortion Coefficients (distCof):\n", distCof)
print("\nRotation Vectors (rVector):\n", rVector)
print("\nTranslation Vectors (tVector):\n", tVector)