import serial
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----- Camera Intrinsics -----
camMatrix = np.array([
    [616.63066325, 0.0, 329.98716933],
    [0.0, 614.63960663, 253.85548567],
    [0.0, 0.0, 1.0]
])

# ----- Transformation: Ultrasonic → Camera -----
rotation = R.from_euler('x', 90, degrees=True)
R_matrix = rotation.as_matrix()

t = np.array([[0.0],    # x translation
              [-0.10],  # y translation
              [0.10]])  # z translation

# ----- Open Serial Port -----
ser = serial.Serial('COM4', 9600, timeout=1)  # Adjust the port as needed
time.sleep(2)

def parse_line(line):
    try:
        x_str, y_str, r_str = line.strip().split(',')
        x = float(x_str) / 100  # Convert cm to meters
        y = float(y_str) / 100
        z = 0.0  # Flat plane assumption
        return np.array([[x], [y], [z]])
    except:
        return None

while True:
    try:
        line = ser.readline().decode().strip()
        if not line or "Warning" in line:
            continue

        P_ultra = parse_line(line)
        if P_ultra is None:
            print(f"Skipping malformed line: {line}")
            continue

        # Transform to camera frame
        P_cam = R_matrix @ P_ultra + t
        X, Y, Z = P_cam.flatten()

        # Project to 2D image coordinates
        u = camMatrix[0, 0] * X / Z + camMatrix[0, 2]
        v = camMatrix[1, 1] * Y / Z + camMatrix[1, 2]

        print(f"3D (Camera): [{X:.2f}, {Y:.2f}, {Z:.2f}] → Pixel: (u={u:.2f}, v={v:.2f})")

    except Exception as e:
        print(f"Error: {e}")
