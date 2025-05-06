import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from ultralytics import YOLO

class SensorFusionSystem:
    def __init__(self, camera_matrix, dist_coeffs, rvec, tvec):
            # camera_matrix: 3x3 camera intrinsic matrix
            # dist_coeffs: Distortion coefficients
            # rvec: Rotation vector (extrinsic)
            # tvec: Translation vector (extrinsic)
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = rvec
        self.tvec = tvec
        
        # Define the zero position offset of ultrasonic sensor
        # These are the coordinates where the servo and ultrasonic are positioned
        # relative to the world origin (chessboard center)
        self.sensor_position = np.array([0.0, 0.0, 0.0])  # !!!! modify this with the real position of ultrsonic relative to the origin !!!
        
    def ultrasonic_servo_to_xyz(self, distance, servo_angle):

        # Convert ultrasonic distance and servo angles to 3D coordinates.
        # Returns 3D coordinates (x, y, z) in the world coordinate system
        
        # angles to radians
        theta_h = math.radians(servo_angle)
        
        # Calculate 3D coordinates 
        x = distance *  math.sin(theta_h)  # distance in meter
        y = 0
        z = distance *  math.cos(theta_h)
        
        # Add sensor position offset
        point_3d = np.array([x, y, z]) + self.sensor_position
        
        return point_3d
    
    def project_point_to_camera(self, point_3d):

        # Project a 3D point onto the camera image plane.
        # Returns 2D coordinates (u, v) in the image plane
        
        # Reshape point for projectPoints
        point_3d = np.array([point_3d], dtype=np.float32)
        
        # Project 3D point to image plane
        point_2d, _ = cv2.projectPoints(point_3d, self.rvec, self.tvec, 
                                        self.camera_matrix, self.dist_coeffs)
        
        # Extract and return pixel coordinates
        return tuple(map(int, point_2d[0][0]))
    
    
    def load_yolo_model(self):
            self.yolo_model = YOLO('yolov8n.pt')

    def detect_objects(self, frame):
        """
        Detect objects in the frame using YOLOv8.
        Returns list of detected objects with their bounding boxes and class.
        """
        if self.yolo_model is None:
            return []
        
        # Run YOLOv8 detection with confidence threshold
        results = self.yolo_model.predict(frame, conf=0.4, iou=0.45)
        
        detected_objects = []
        if results and len(results) > 0:
            # Process the first result (current frame)
            result = results[0]
            
            # Extract boxes, classes and confidence scores
            for box in result.boxes:
                # Get box coordinates (xmin, ymin, xmax, ymax)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Calculate center
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Get class name
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                obj = {
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (int(x_center), int(y_center))
                }
                detected_objects.append(obj)
            
        return detected_objects
    
    def run_live_detection(self):
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # !!!! modify with readings from esp32 !!!!
                
                distance = 0.5  # 50cm distance
                servo_angle_h = 20  # 20 degrees horizontal 
                
                # Convert to 3D coordinates
                object_3d = self.ultrasonic_servo_to_xyz(distance, servo_angle_h)
                
                # Project to camera view
                object_2d = self.project_point_to_camera(object_3d)
                
                # Draw detected point on frame
                cv2.circle(frame, object_2d, 10, (0, 255, 0), -1)
                cv2.putText(frame, f"3D: ({object_3d[0]:.2f}, {object_3d[1]:.2f}, {object_3d[2]:.2f})")
                
                # Display
                cv2.imshow('Sensor Fusion', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    # system.run_live_detection()