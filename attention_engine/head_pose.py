"""
3D Head Pose Estimation using OpenCV solvePnP.
Calculates Yaw, Pitch, and Roll angles from facial landmarks.
"""
import cv2
import numpy as np
import config


class HeadPoseEstimator:
    """
    Estimates 3D head pose (Yaw, Pitch, Roll) using OpenCV solvePnP.
    Provides numerical angles as well as qualitative directional descriptions.
    """
    def __init__(
        self,
        image_size=(640, 480),
        yaw_threshold: float = config.YAW_DISTRACTED_THRESHOLD,
        pitch_threshold: float = config.PITCH_DISTRACTED_THRESHOLD,
        roll_threshold: float = config.ROLL_DISTRACTED_THRESHOLD
    ):
        h, w = image_size
        focal_length = w
        center = (w / 2, h / 2)
        
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold

        # Camera intrinsic matrix approximation
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Standard 3D facial model points (in mm / world units)
        # 1: Nose tip, 152: Chin, 33: Left eye outer corner, 263: Right eye outer corner, 61: Left mouth corner, 291: Right mouth corner
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # Landmark indices in MediaPipe FaceMesh
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def get_orientation_description(self, yaw: float, pitch: float, roll: float) -> str:
        """Provide a human-understandable directional status based on thresholds."""
        directions = []
        if yaw > self.yaw_threshold:
            directions.append("Looking Right")
        elif yaw < -self.yaw_threshold:
            directions.append("Looking Left")
            
        if pitch > self.pitch_threshold:
            directions.append("Looking Down")
        elif pitch < -self.pitch_threshold:
            directions.append("Looking Up")
            
        if not directions:
            if abs(roll) > self.roll_threshold:
                directions.append("Head Tilted")
            else:
                directions.append("Facing Screen")
                
        return ", ".join(directions)

    def estimate_pose(self, landmarks_px, frame_shape):
        """
        Estimate Head Pose (Yaw, Pitch, Roll) from pixel facial landmarks.
        
        Returns:
            dict containing:
            - yaw: float (-90 to +90 degrees)
            - pitch: float (-90 to +90 degrees)
            - roll: float (-90 to +90 degrees)
            - pose_score: float normalized [0.0 to 1.0] representing frontality
            - orientation: str description (e.g. "Facing Screen", "Looking Left")
            - rvec: rotation vector
            - tvec: translation vector
            - nose_start: tuple (x, y)
            - nose_end: tuple (x, y)
        """
        h, w = frame_shape[:2]
        
        # Dynamically adjust camera matrix if frame dimensions change
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        # Extract the 6 key 2D landmark points
        image_points = []
        for idx in self.landmark_indices:
            if idx < len(landmarks_px):
                image_points.append(landmarks_px[idx])
            else:
                image_points.append([w // 2, h // 2])
        
        image_points = np.array(image_points, dtype=np.float64)
        nose_center = (int(image_points[0][0]), int(image_points[0][1]))

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'pose_score': 1.0,
                'orientation': 'Facing Screen (Fallback)',
                'rvec': np.zeros((3, 1)),
                'tvec': np.zeros((3, 1)),
                'nose_start': nose_center,
                'nose_end': nose_center
            }

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Compute Euler Angles (Yaw, Pitch, Roll)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        # Normalize angles to [-90, 90]
        yaw = float(np.clip(yaw, -90.0, 90.0))
        pitch = float(np.clip(pitch, -90.0, 90.0))
        roll = float(np.clip(roll, -90.0, 90.0))

        # Calculate pose frontality score: 1.0 = looking straight at camera, 0.0 = turned far away
        angular_deviation = np.sqrt(yaw**2 + pitch**2 + (roll * 0.5)**2)
        pose_score = float(np.clip(1.0 - (angular_deviation / 75.0), 0.0, 1.0))
        if angular_deviation <= 20.0:
            pose_score = float(np.clip(0.85 + (1.0 - angular_deviation / 20.0) * 0.15, 0.85, 1.0))
        orientation = self.get_orientation_description(yaw, pitch, roll)

        # Project 3D Coordinate Axes (X-Red, Y-Green, Z-Blue)
        axis_len = 150.0
        axis_points_3D = np.array([
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, -axis_len, 0.0],
            [0.0, 0.0, axis_len]
        ], dtype=np.float64)

        axis_points_2D, _ = cv2.projectPoints(
            axis_points_3D, rvec, tvec, camera_matrix, self.dist_coeffs
        )

        origin_p = nose_center
        x_axis_p = (int(axis_points_2D[1][0][0]), int(axis_points_2D[1][0][1]))
        y_axis_p = (int(axis_points_2D[2][0][0]), int(axis_points_2D[2][0][1]))
        z_axis_p = (int(axis_points_2D[3][0][0]), int(axis_points_2D[3][0][1]))

        nose_p1 = origin_p
        nose_p2 = z_axis_p

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'pose_score': pose_score,
            'orientation': orientation,
            'rvec': rvec,
            'tvec': tvec,
            'nose_start': nose_p1,
            'nose_end': nose_p2,
            'axes_3d': {
                'origin': origin_p,
                'x_axis': x_axis_p,
                'y_axis': y_axis_p,
                'z_axis': z_axis_p
            }
        }

    @staticmethod
    def draw_3d_axes(frame, pose_dict):
        """Draw 3D RGB orientation coordinate axes on image frame."""
        axes = pose_dict.get('axes_3d')
        if not axes:
            return frame

        o = axes['origin']
        x = axes['x_axis']
        y = axes['y_axis']
        z = axes['z_axis']

        # Red for X-axis (Pitch/Sides), Green for Y-axis (Yaw/Up), Blue for Z-axis (Forward)
        cv2.line(frame, o, x, (0, 0, 255), 2)
        cv2.line(frame, o, y, (0, 255, 0), 2)
        cv2.line(frame, o, z, (255, 100, 0), 2)
        return frame
