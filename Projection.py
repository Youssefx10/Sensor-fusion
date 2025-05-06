import cv2
import numpy as np
import math
import serial
import time

class SensorFusionSystem:
    def __init__(self, camera_matrix, dist_coeffs, rvec, tvec, serial_port, baud_rate=11111):
        
        # camera_matrix: 3x3 intrinsic matrix
        # dist_coeffs: Distortion coefficients
        # rvec: Rotation vector - extrinsic
        # tvec: Translation vector - extrinsic
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = rvec
        self.tvec = tvec
        
        # camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")
        
        # ESP32 serial connection
        try:
            self.sensor = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"Connected to ESP32 on {serial_port}")
            time.sleep(2)  # time for ESP32 to initialize
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to connect to ESP32: {e}")
        
        # sensor data
        self.distance = 0
        self.angle = 0
        self.last_valid_distance = 0
    
    def read_sensor_data(self):
        
        # Read data from ESP32 - Returns distance and angle.
        try:
            if self.sensor.in_waiting > 0:
                data = self.sensor.readline().decode('utf-8').strip()
                parts = data.split(',')
                
                if len(parts) == 2:
                    try:
                        distance = float(parts[0])
                        angle = float(parts[1])
                        self.distance = distance   
                        self.angle = angle
                        return self.distance, self.angle
                    except ValueError:
                        print("Invalid data")
        except Exception as e:
            print(f"Error reading from ESP32: {e}")
        return self.distance, self.angle
    
    def polar_to_cartesian(self, r, theta):
        # Returns (x, y, z) coordinates in world coordinate system
        # angle to radians
        theta_rad = math.radians(theta)
        # Calculate x, y coordinates
        x = r * math.cos(theta_rad)
        y = r * math.sin(theta_rad)
        z = 0  # Assuming the sensor and object are on the same plane
        return np.array([x, y, z])
    
    def project_point_to_camera(self, world_point):
        #   world point (3D) to (u, v) pixel coordinates on the image
        #   Reshape point for Opencv projection
        world_point = world_point.reshape((1, 1, 3))
        
        # Project point using camera parameters
        image_point, _ = cv2.projectPoints(
            world_point, self.rvec, self.tvec, 
            self.camera_matrix, self.dist_coeffs
        )
        
        # Extract pixel coordinates
        u, v = image_point[0][0]
        return int(u), int(v)
    
    def run(self):
        # Run the sensor fusion system.
        try:
            while True:
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed")
                    break
                
                # Read sensor data
                distance, angle = self.read_sensor_data()
                
                # Convert to world coordinates
                world_point = self.polar_to_cartesian(distance, angle)
                
                # Project to image
                image_point = self.project_point_to_camera(world_point)
                
                # Draw the projected point on the image
                cv2.circle(frame, image_point, 10, (0, 0, 255), -1)
                
                # Display distance and angle
                cv2.putText(frame, f"Distance: {distance:.1f} cm", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Angle: {angle:.1f} deg", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display world coordinates
                cv2.putText(frame, f"X: {world_point[0]:.1f}, Y: {world_point[1]:.1f}, Z: {world_point[2]:.1f}", 
                          (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Sensor Fusion', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("terminated")
        finally:
            self.cap.release()
            self.sensor.close()
            cv2.destroyAllWindows()


def main():
    # Camera calibration parameters 
    camera_matrix = np.array([
        [1000, 0, 320],  # fx, 0, cx
        [0, 1000, 240],  # 0, fy, cy
        [0, 0, 1]
    ])
    
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3
    
    # Extrinsic parameters (world 2 camera)
    rvec = np.array([0.0, 0.0, 0.0])  
    tvec = np.array([0.0, 0.0, 50.0])  # camera is 50cm from origin
    
    serial_port = 'COM3'  
    
    # Create and run the system
    system = SensorFusionSystem(camera_matrix, dist_coeffs, rvec, tvec, serial_port)
    system.run()


if __name__ == "__main__":
    main()