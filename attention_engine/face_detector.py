"""
Face Detection and Facial Landmark Extraction using MediaPipe / OpenCV.
"""
import os
import cv2
import numpy as np
import logging

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh')
except ImportError:
    HAS_MEDIAPIPE = False

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in video frames and extracts 3D facial landmarks (468/478 points).
    Provides OpenCV Haar Cascade fallback if MediaPipe is unavailable.
    """
    def __init__(self, max_num_faces=10, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_face_mesh = None
        self.face_mesh = None
        
        if HAS_MEDIAPIPE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=True,  # Includes iris landmarks
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                logger.info("MediaPipe FaceMesh initialized with iris refinement.")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe FaceMesh: {e}. Falling back to OpenCV.")
                self.face_mesh = None

        # Fallback Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.haar_cascade = cv2.CascadeClassifier()

    def process(self, frame_bgr, static_image=False):
        """
        Process a BGR image frame.
        Returns a list of dictionaries, each containing:
        - bbox: (x, y, w, h)
        - landmarks_3d: np.ndarray shape (N, 3) normalized [0..1]
        - landmarks_px: np.ndarray shape (N, 2) pixel coords (x, y)
        - confidence: float score
        """
        h, w = frame_bgr.shape[:2]
        results = []

        if self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_results = self.face_mesh.process(rgb_frame)

                if mp_results.multi_face_landmarks:
                    for face_landmarks in mp_results.multi_face_landmarks:
                        pts_norm = []
                        pts_px = []
                        
                        for lm in face_landmarks.landmark:
                            pts_norm.append([lm.x, lm.y, lm.z])
                            pts_px.append([int(lm.x * w), int(lm.y * h)])

                        pts_norm = np.array(pts_norm, dtype=np.float32)
                        pts_px = np.array(pts_px, dtype=np.int32)

                        # Calculate bounding box from landmarks
                        x_min, y_min = np.min(pts_px[:, :2], axis=0)
                        x_max, y_max = np.max(pts_px[:, :2], axis=0)
                        
                        # Pad box slightly
                        pad_x = int((x_max - x_min) * 0.1)
                        pad_y = int((y_max - y_min) * 0.15)
                        
                        bx = max(0, x_min - pad_x)
                        by = max(0, y_min - pad_y)
                        bw = min(w - bx, (x_max - x_min) + 2 * pad_x)
                        bh = min(h - by, (y_max - y_min) + 2 * pad_y)

                        # Filter spurious tiny background noise detections
                        eye_dist = np.linalg.norm(pts_px[33] - pts_px[263]) if len(pts_px) > 263 else 50.0
                        if bw < 45 or bh < 45 or eye_dist < 15.0:
                            continue

                        results.append({
                            'bbox': (bx, by, bw, bh),
                            'landmarks_3d': pts_norm,
                            'landmarks_px': pts_px,
                            'confidence': 0.95
                        })
                    
                    results = self._apply_nms(results)
                    return results
            except Exception as e:
                logger.warning(f"MediaPipe FaceMesh processing error: {e}")

        # Fallback to Haar cascade if available
        if self.haar_cascade is not None and not self.haar_cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )

                for (fx, fy, fw, fh) in faces:
                    synthetic_px = self._create_synthetic_landmarks(fx, fy, fw, fh)
                    synthetic_norm = synthetic_px.astype(np.float32) / np.array([w, h], dtype=np.float32)
                    synthetic_norm_3d = np.hstack([synthetic_norm, np.zeros((len(synthetic_norm), 1), dtype=np.float32)])

                    results.append({
                        'bbox': (int(fx), int(fy), int(fw), int(fh)),
                        'landmarks_3d': synthetic_norm_3d,
                        'landmarks_px': synthetic_px,
                        'confidence': 0.75
                    })
                results = self._apply_nms(results)
            except Exception as e:
                logger.warning(f"Haar cascade detection failed: {e}")

        return results

    def _apply_nms(self, detections, iou_thresh=0.30):
        """
        Non-Maximum Suppression & Overlap Suppression to eliminate duplicate bounding boxes for the same student face.
        """
        if len(detections) <= 1:
            return detections

        # Sort detections by confidence (descending) then area (descending)
        detections = sorted(
            detections,
            key=lambda d: (d.get('confidence', 0.5), d['bbox'][2] * d['bbox'][3]),
            reverse=True
        )

        keep = []
        for det in detections:
            x1, y1, w1, h1 = det['bbox']
            area1 = w1 * h1
            is_duplicate = False
            
            for k in keep:
                kx, ky, kw, kh = k['bbox']
                area2 = kw * kh

                # Compute intersection box
                ix1 = max(x1, kx)
                iy1 = max(y1, ky)
                ix2 = min(x1 + w1, kx + kw)
                iy2 = min(y1 + h1, ky + kh)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih

                union = area1 + area2 - inter
                iou = inter / float(union) if union > 0 else 0.0
                min_overlap = inter / float(min(area1, area2)) if min(area1, area2) > 0 else 0.0

                if iou > iou_thresh or min_overlap > 0.40:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(det)

        return keep

    def _create_synthetic_landmarks(self, x, y, w, h):
        """Create basic 468 landmark approximation array when only BBox is available."""
        pts = np.zeros((468, 2), dtype=np.int32)
        cx, cy = x + w // 2, y + h // 2
        
        # Key landmark points
        pts[1] = [cx, cy]                          # Nose tip
        pts[152] = [cx, y + int(h * 0.95)]          # Chin
        pts[33] = [x + int(w * 0.3), y + int(h * 0.4)]  # Left eye outer
        pts[263] = [x + int(w * 0.7), y + int(h * 0.4)] # Right eye outer
        pts[61] = [x + int(w * 0.35), y + int(h * 0.75)]# Left mouth
        pts[291] = [x + int(w * 0.65), y + int(h * 0.75)]# Right mouth
        
        for i in range(468):
            if np.all(pts[i] == 0):
                pts[i] = [cx, cy]
        return pts

    def release(self):
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception:
                pass

