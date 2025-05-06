import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Projection import SensorFusionSystem

def visualize_setup(self, img_size=(640, 480)):
        
        # 3D plot showing camera, world origin and ultrasonic sensor position.

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot world coordinate system
        ax.scatter([0], [0], [0], color='red', s=100, label='World Origin (Chessboard Center)')
        
        # Plot camera position
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(self.rvec)
        
        # Camera position in world coordinates
        cam_pos = -np.dot(R.T, self.tvec).squeeze()
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color='blue', s=100, label='Camera')
        
        # Plot ultrasonic sensor position
        ax.scatter(self.sensor_position[0], self.sensor_position[1], self.sensor_position[2], 
                  color='green', s=100, label='Ultrasonic Sensor')
        
        # Plot coordinate axes
        length = 0.2 
        # X-axis
        ax.plot([0, length], [0, 0], [0, 0], 'r-', linewidth=2)
        # Y-axis
        ax.plot([0, 0], [0, length], [0, 0], 'g-', linewidth=2)
        # Z-axis
        ax.plot([0, 0], [0, 0], [0, length], 'b-', linewidth=2)
        
        # Add camera coordinate system
        cam_x = cam_pos + R.T[:, 0] * length/2
        cam_y = cam_pos + R.T[:, 1] * length/2
        cam_z = cam_pos + R.T[:, 2] * length/2
        
        ax.plot([cam_pos[0], cam_x[0]], [cam_pos[1], cam_x[1]], [cam_pos[2], cam_x[2]], 'r-', linewidth=2)
        ax.plot([cam_pos[0], cam_y[0]], [cam_pos[1], cam_y[1]], [cam_pos[2], cam_y[2]], 'g-', linewidth=2)
        ax.plot([cam_pos[0], cam_z[0]], [cam_pos[1], cam_z[1]], [cam_pos[2], cam_z[2]], 'b-', linewidth=2)
        
        # Plot field of view of ultrasonic sensor 
        # Create some sample points at different angles
        angles_h = np.linspace(-30, 30, 10)  # Horizontal field of view
        max_distance = 3.0  # Maximum sensing distance
        
        for angle_h in angles_h:
                point = self.ultrasonic_servo_to_xyz(max_distance, angle_h)
                ax.scatter(point[0], point[1], point[2], color='cyan', alpha=0.2, s=10)
        
        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Sensor Fusion Setup')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([
            np.max([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]),
            np.min([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        ]).max()
        mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
        mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
        mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        
        plt.show()

SensorFusionSystem.visualize_setup = visualize_setup

# Example usage
if __name__ == "__main__":
    camera_matrix = np.array([
        [800, 0, 320],  
        [0, 800, 240],   
        [0, 0, 1]        
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros(5, dtype=np.float32)
    rvec = np.array([[0.1], [0.1], [0.0]], dtype=np.float32)  
    tvec = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)  
    
    system = SensorFusionSystem(camera_matrix, dist_coeffs, rvec, tvec)
    
    system.visualize_setup()
    
    # Object at 60cm with servo at 15° horizontal 
    object_3d = system.ultrasonic_servo_to_xyz(0.6, 15)
    object_2d = system.project_point_to_camera(object_3d)
    
    print(f"Object 3D coordinates: {object_3d}")
    print(f"Projected to camera at pixel: {object_2d}")