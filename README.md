# 🧠 Ultrasonic-Camera Sensor Fusion

This project projects a 3D position of an object detected by a servo-mounted ultrasonic sensor onto a live camera feed. It combines basic sensor fusion, camera calibration, and 3D-to-2D projection using OpenCV and Arduino.

---

## 📷 What It Does

- Collects object distances using an HC-SR04 ultrasonic sensor mounted on a servo.
- Calibrates a camera to get intrinsic parameters.
- Calculates 3D coordinates of detected objects using distance and angle.
- Projects 3D positions onto a 2D live camera feed using OpenCV.

---

## 🛠️ Hardware Used

- Arduino Uno
- HC-SR04 Ultrasonic Sensor
- SG90 Servo Motor
- USB Webcam
- Breadboard + Jumper Wires

---

## 💻 Software Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
