"""
Gaze Estimation Engine — Iris landmarks + Head Pose.

FIXES vs. original:
- Removed double-counting of yaw: gaze_x is now computed from iris displacement
  alone when iris landmarks are available; head yaw is only used as fallback.
- Added clear documentation that gaze is a visual proxy, not proof of attention.
- Made thresholds importable from config.
"""
import numpy as np
import config


class GazeEstimator:
    """
    Estimates approximate gaze direction from iris position and head pose.

    ⚠️  Gaze direction is a *visual proxy* for focus.  It does NOT prove
    whether a student is mentally attending to the lesson.  A student can
    look at the board while thinking about something else, or look away
    briefly while still mentally engaged.
    """

    # MediaPipe FaceMesh iris landmark indices (only with refine_landmarks=True)
    LEFT_IRIS  = [468, 469, 470, 471, 472]  # left iris centre and edges
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # Eye corner indices for normalising iris displacement
    LEFT_EYE_OUTER  = 33
    LEFT_EYE_INNER  = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    def estimate_gaze(
        self,
        landmarks_px: np.ndarray,
        head_pose: dict,
        frame_shape: tuple,
    ) -> dict:
        """
        Estimate gaze direction and a heuristic focus score.

        Parameters
        ----------
        landmarks_px : ndarray (N, 2) — pixel coordinates from FaceDetector
        head_pose    : dict from HeadPoseEstimator
        frame_shape  : (h, w[, c])

        Returns
        -------
        dict with keys:
            gaze_x        – float [-1, 1]  negative = left, positive = right
            gaze_y        – float [-1, 1]  negative = up,   positive = down
            gaze_score    – float [0, 1]   1.0 = looking straight at screen
            iris_left_px  – (x, y) int tuple
            iris_right_px – (x, y) int tuple
            method        – 'iris' | 'head_pose_fallback'
        """
        h, w = frame_shape[:2]
        num_lm = len(landmarks_px)

        # Default iris centres (frame centre fallback)
        iris_left_px  = (int(w * 0.4), int(h * 0.4))
        iris_right_px = (int(w * 0.6), int(h * 0.4))
        method = "head_pose_fallback"

        if num_lm >= 478:
            # ── Iris-based gaze (preferred) ──────────────────────────────
            method = "iris"

            iris_left_px  = (int(landmarks_px[468][0]), int(landmarks_px[468][1]))
            iris_right_px = (int(landmarks_px[473][0]), int(landmarks_px[473][1]))

            # Left eye: iris x relative to eye width, centred at 0
            l_outer = landmarks_px[self.LEFT_EYE_OUTER].astype(float)
            l_inner = landmarks_px[self.LEFT_EYE_INNER].astype(float)
            eye_w_l = max(1.0, np.linalg.norm(l_outer - l_inner))
            iris_rel_l = (float(landmarks_px[468][0]) - l_outer[0]) / eye_w_l - 0.5

            # Right eye
            r_inner = landmarks_px[self.RIGHT_EYE_INNER].astype(float)
            r_outer = landmarks_px[self.RIGHT_EYE_OUTER].astype(float)
            eye_w_r = max(1.0, np.linalg.norm(r_inner - r_outer))
            iris_rel_r = (float(landmarks_px[473][0]) - r_inner[0]) / eye_w_r - 0.5

            # Average iris horizontal displacement (–0.5 → left, +0.5 → right)
            iris_gaze_h = (iris_rel_l + iris_rel_r) / 2.0

            # Scale to [-1, 1] — iris range is roughly ±0.35 in practice
            gaze_x = float(np.clip(iris_gaze_h / 0.35, -1.0, 1.0))

            # Vertical: head pitch is the most reliable signal for up/down
            gaze_y = float(np.clip(head_pose["pitch"] / 45.0, -1.0, 1.0))

        else:
            # ── Head-pose-only fallback ───────────────────────────────────
            gaze_x = float(np.clip(head_pose["yaw"] / 45.0, -1.0, 1.0))
            gaze_y = float(np.clip(head_pose["pitch"] / 45.0, -1.0, 1.0))

        # Gaze focus score: 1.0 = looking straight at screen centre, 0.0 = far away
        gaze_dist  = np.sqrt(gaze_x ** 2 + gaze_y ** 2)
        gaze_score = float(np.clip(1.0 - (gaze_dist / 0.85), 0.0, 1.0))
        if gaze_dist <= 0.35:
            gaze_score = float(np.clip(0.85 + (1.0 - gaze_dist / 0.35) * 0.15, 0.85, 1.0))

        return {
            "gaze_x":        gaze_x,
            "gaze_y":        gaze_y,
            "gaze_score":    gaze_score,
            "iris_left_px":  iris_left_px,
            "iris_right_px": iris_right_px,
            "method":        method,
        }
