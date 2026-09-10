"""
Drowsiness and Fatigue Detector — Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

FIXES vs. original:
- Added per-student consecutive-frame counter so a single blink (1–2 frames)
  does NOT trigger is_drowsy.  Prolonged closure (≥ CONSECUTIVE_DROWSY_FRAMES)
  is required.
- Added prolonged_closure flag for closures longer than PROLONGED_CLOSURE_FRAMES.
- Made all thresholds configurable via constructor.
- Separated blink_detected from is_drowsy.
"""
import numpy as np
import config


class DrowsinessDetector:
    """
    Measures facial fatigue via Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

    EAR is a proxy for eye-openness; it does NOT conclusively determine whether
    a student is mentally alert.  Brief eye closures (blinks) are distinguished
    from prolonged closures using a configurable consecutive-frame threshold.
    """

    # MediaPipe FaceMesh landmark indices for left and right eyes
    # Order: outer_corner, top1, top2, inner_corner, bot2, bot1
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks for MAR
    MOUTH_CORNER   = (61, 291)
    MOUTH_TOP_BOT  = [(13, 14), (82, 87), (312, 317)]

    def __init__(
        self,
        ear_threshold: float = config.EAR_THRESHOLD,
        mar_threshold: float = config.MAR_THRESHOLD,
        consecutive_frames_threshold: int = config.CONSECUTIVE_DROWSY_FRAMES,
        prolonged_closure_frames: int = config.PROLONGED_CLOSURE_FRAMES,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.prolonged_closure_frames = prolonged_closure_frames

        # Per-detector frame counters (one detector instance per tracked student)
        self._low_ear_counter: int = 0
        self._blink_count: int = 0

    def reset(self) -> None:
        """Reset frame counters (call when a new student track starts)."""
        self._low_ear_counter = 0
        self._blink_count = 0

    # ── EAR ───────────────────────────────────────────────────────────────────

    def calculate_ear(self, landmarks_px: np.ndarray, eye_indices: list) -> float:
        """
        Eye Aspect Ratio  =  (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)

        Returns 0.30 (open-eye default) when landmarks are missing.
        """
        if len(landmarks_px) <= max(eye_indices):
            return 0.30

        p1 = landmarks_px[eye_indices[0]]  # outer corner
        p2 = landmarks_px[eye_indices[1]]  # top-1
        p3 = landmarks_px[eye_indices[2]]  # top-2
        p4 = landmarks_px[eye_indices[3]]  # inner corner
        p5 = landmarks_px[eye_indices[4]]  # bot-2
        p6 = landmarks_px[eye_indices[5]]  # bot-1

        v1 = np.linalg.norm(p2.astype(float) - p6.astype(float))
        v2 = np.linalg.norm(p3.astype(float) - p5.astype(float))
        h  = np.linalg.norm(p1.astype(float) - p4.astype(float))

        if h < 1e-6:
            return 0.30

        return float((v1 + v2) / (2.0 * h))

    # ── MAR ───────────────────────────────────────────────────────────────────

    def calculate_mar(self, landmarks_px: np.ndarray) -> float:
        """
        Mouth Aspect Ratio — proxy for yawn / mouth opening.
        Returns 0.10 (closed mouth) when landmarks are missing.
        """
        if len(landmarks_px) <= 317:
            return 0.10

        vertical = 0.0
        for (top_idx, bot_idx) in self.MOUTH_TOP_BOT:
            vertical += np.linalg.norm(
                landmarks_px[top_idx].astype(float) - landmarks_px[bot_idx].astype(float)
            )
        vertical /= len(self.MOUTH_TOP_BOT)

        h = np.linalg.norm(
            landmarks_px[self.MOUTH_CORNER[0]].astype(float)
            - landmarks_px[self.MOUTH_CORNER[1]].astype(float)
        )
        if h < 1e-6:
            return 0.10

        return float(vertical / h)

    # ── Main process ──────────────────────────────────────────────────────────

    def process(self, landmarks_px: np.ndarray) -> dict:
        """
        Process landmarks for one frame and return drowsiness metrics.

        Returns
        -------
        dict with keys:
            left_ear          – float
            right_ear         – float
            avg_ear           – float
            mar               – float
            blink_detected    – bool  (brief closure, < consecutive threshold)
            is_drowsy         – bool  (prolonged closure ≥ consecutive threshold)
            prolonged_closure – bool  (very long closure ≥ prolonged_closure_frames)
            is_yawning        – bool
            drowsiness_score  – float [0.0 closed → 1.0 open]
            consecutive_low_ear_frames – int
        """
        left_ear  = self.calculate_ear(landmarks_px, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks_px, self.RIGHT_EYE)
        avg_ear   = (left_ear + right_ear) / 2.0
        mar       = self.calculate_mar(landmarks_px)

        # ── Consecutive-frame counter ──────────────────────────────────────
        if avg_ear < self.ear_threshold:
            self._low_ear_counter += 1
        else:
            # Eye re-opened — if it was a short closure it was a blink
            if 0 < self._low_ear_counter < self.consecutive_frames_threshold:
                self._blink_count += 1
            self._low_ear_counter = 0

        blink_detected    = (self._low_ear_counter == 0 and
                             0 < self._low_ear_counter < self.consecutive_frames_threshold)
        is_drowsy         = self._low_ear_counter >= self.consecutive_frames_threshold
        prolonged_closure = self._low_ear_counter >= self.prolonged_closure_frames
        is_yawning        = mar > self.mar_threshold

        # ── Drowsiness score: 1.0 = eyes wide open, 0.0 = fully closed ───
        drowsiness_score = float(np.clip(avg_ear / 0.28, 0.0, 1.0))
        if avg_ear >= 0.20 and not is_drowsy:
            drowsiness_score = max(drowsiness_score, 0.85)
        if prolonged_closure:
            drowsiness_score = min(drowsiness_score, 0.15)
        elif is_drowsy:
            drowsiness_score = min(drowsiness_score, 0.30)
        if is_yawning:
            drowsiness_score = min(drowsiness_score, 0.45)

        return {
            "left_ear":  left_ear,
            "right_ear": right_ear,
            "avg_ear":   avg_ear,
            "mar":       mar,
            "blink_detected":    blink_detected,
            "is_drowsy":         is_drowsy,
            "prolonged_closure": prolonged_closure,
            "is_yawning":        is_yawning,
            "drowsiness_score":  drowsiness_score,
            "consecutive_low_ear_frames": self._low_ear_counter,
        }
