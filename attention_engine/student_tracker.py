"""
Multi-Student Persistent ID Tracker with EMA Smoothing.

FIXES vs. original:
- CRITICAL: dist_matrix.values crash fixed (numpy ndarray has no .values attribute)
- Fixed index alignment between detections and attention_results by tracking
  latest matched visual features inside TrackedStudent
- Each TrackedStudent owns its own DrowsinessDetector so consecutive-frame
  counters are per-student, not shared
- Added Unknown state support
- Made all parameters configurable
- Added wall-clock time tracking for accurate alert duration
"""
import time
import numpy as np
import config
from attention_engine.drowsiness_detector import DrowsinessDetector


class TrackedStudent:
    """Maintains temporal state for one student face track."""

    def __init__(self, student_id: int, initial_bbox: tuple):
        self.student_id   = student_id
        self.bbox         = initial_bbox
        self.centroid     = self._centroid(initial_bbox)

        # EMA attention score
        self.attention_score_ema: float = 75.0
        self.alpha: float = config.EMA_ALPHA

        # Frame & time tracking
        self.disappeared_frames:   int   = 0
        self.total_frames_tracked: int   = 0
        self.first_seen_time:      float = time.time()
        self.last_seen_time:       float = time.time()

        # Status history (rolling window for timeline)
        self.status_history: list = []
        self.latest_attention_data: dict = {}

        # Sustained low-attention tracking (wall-clock seconds, not frame count)
        self.low_attention_start_time: float | None = None
        self.sustained_low_seconds: float = 0.0

        # Per-student drowsiness detector (keeps consecutive-frame counters)
        self.drowsiness_detector = DrowsinessDetector()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _centroid(bbox: tuple) -> tuple:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    # ── update ────────────────────────────────────────────────────────────────

    def update(
        self,
        bbox: tuple,
        current_attention_score: float,
        status: str,
        attention_data: dict | None = None
    ) -> None:
        """Apply new frame detection to this track."""
        self.bbox         = bbox
        self.centroid     = self._centroid(bbox)
        self.disappeared_frames = 0
        self.total_frames_tracked += 1
        self.last_seen_time = time.time()

        if attention_data is not None:
            self.latest_attention_data = attention_data

        # EMA smoothing
        self.attention_score_ema = (
            self.alpha * current_attention_score
            + (1.0 - self.alpha) * self.attention_score_ema
        )

        # Rolling status history (last 60 frames)
        self.status_history.append(status)
        if len(self.status_history) > 60:
            self.status_history.pop(0)

        # Wall-clock sustained low-attention timer
        now = time.time()
        if status in ("Distracted", "Drowsy"):
            if self.low_attention_start_time is None:
                self.low_attention_start_time = now
            self.sustained_low_seconds = now - self.low_attention_start_time
        else:
            self.low_attention_start_time = None
            self.sustained_low_seconds    = 0.0


class StudentTracker:
    """
    Tracks multiple student faces across frames using centroid matching.

    Each detected face is assigned a persistent anonymous ID (Student #1, #2, …).
    No facial identity or biometric data is stored.
    """

    def __init__(
        self,
        max_disappeared: int   = config.TRACKER_MAX_DISAPPEARED,
        max_distance:    float = config.TRACKER_MAX_CENTROID_DISTANCE,
    ):
        self.next_student_id = 1
        self.students: dict[int, TrackedStudent] = {}
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def reset(self) -> None:
        """Reset all tracked students (call on new session start)."""
        self.students.clear()
        self.next_student_id = 1

    def update(self, detections: list, attention_results: list) -> list:
        """
        Match incoming detections to existing tracks.

        Parameters
        ----------
        detections       : list of dicts with keys 'bbox', 'landmarks_px', …
        attention_results: list of dicts with keys 'attention_score', 'status', …
                           Must be the same length and order as detections.

        Returns
        -------
        List of dicts with tracked student metadata for this frame.
        """
        # Degrade all tracks when no detections
        if len(detections) == 0:
            for sid in list(self.students.keys()):
                self.students[sid].disappeared_frames += 1
                if self.students[sid].disappeared_frames > self.max_disappeared:
                    del self.students[sid]
            return []

        # Compute input centroids
        input_centroids = []
        for det in detections:
            x, y, w, h = det["bbox"]
            input_centroids.append((x + w // 2, y + h // 2))

        # No existing tracks → register all
        if len(self.students) == 0:
            self.next_student_id = 1
            for i, det in enumerate(detections):
                att = attention_results[i] if i < len(attention_results) else {"attention_score": 75.0, "status": "Unknown"}
                self._register(det["bbox"], att["attention_score"], att["status"], att)
        else:
            student_ids        = list(self.students.keys())
            existing_centroids = [self.students[sid].centroid for sid in student_ids]
            n_existing = len(existing_centroids)
            n_input    = len(input_centroids)
            used_rows: set = set()
            used_cols: set = set()

            # ── Special case: Single active student on webcam ────────────────────
            if n_existing == 1 and n_input == 1:
                sid = student_ids[0]
                att = attention_results[0] if len(attention_results) > 0 else {"attention_score": 75.0, "status": "Unknown"}
                self.students[sid].update(
                    detections[0]["bbox"],
                    att["attention_score"],
                    att["status"],
                    att
                )
                used_rows.add(0)
                used_cols.add(0)
            else:
                # ── Euclidean distance matrix ──────────────────────────────────
                dist_matrix = np.full((n_existing, n_input), np.inf, dtype=np.float32)

                for i, ec in enumerate(existing_centroids):
                    for j, ic in enumerate(input_centroids):
                        dist_matrix[i, j] = np.linalg.norm(
                            np.array(ec, dtype=float) - np.array(ic, dtype=float)
                        )

                # ── Greedy min-distance matching ───────────────────────────────
                flat_order = np.argsort(dist_matrix, axis=None)

                for flat_idx in flat_order:
                    row = int(flat_idx) // n_input
                    col = int(flat_idx) % n_input

                    if row in used_rows or col in used_cols:
                        continue
                    if dist_matrix[row, col] > self.max_distance:
                        break   # remaining distances are all larger (sorted)

                    sid = student_ids[row]
                    att = attention_results[col] if col < len(attention_results) else {"attention_score": 75.0, "status": "Unknown"}
                    self.students[sid].update(
                        detections[col]["bbox"],
                        att["attention_score"],
                        att["status"],
                        att
                    )
                    used_rows.add(row)
                    used_cols.add(col)

            # Register unmatched new detections as new students
            for j in range(n_input):
                if j not in used_cols:
                    att = attention_results[j] if j < len(attention_results) else {"attention_score": 75.0, "status": "Unknown"}
                    self._register(detections[j]["bbox"], att["attention_score"], att["status"], att)

            # Mark unmatched existing tracks as disappeared
            for i, sid in enumerate(student_ids):
                if i not in used_rows:
                    self.students[sid].disappeared_frames += 1
                    if self.students[sid].disappeared_frames > self.max_disappeared:
                        del self.students[sid]

        # ── Compile output for current frame ──────────────────────────────
        output = []
        for sid, student in self.students.items():
            if student.disappeared_frames == 0:
                output.append({
                    "student_id":            student.student_id,
                    "bbox":                  student.bbox,
                    "smoothed_attention":    student.attention_score_ema,
                    "sustained_low_seconds": student.sustained_low_seconds,
                    "status_history":        list(student.status_history),
                    "total_frames":          student.total_frames_tracked,
                    "drowsiness_detector":   student.drowsiness_detector,
                    "latest_attention_data": student.latest_attention_data,
                })
        return output

    def _register(self, bbox: tuple, score: float, status: str, attention_data: dict | None = None) -> None:
        # Reset student ID to 1 whenever there are no active students (e.g., solo webcam mode)
        if len(self.students) == 0:
            self.next_student_id = 1

        student = TrackedStudent(self.next_student_id, bbox)
        student.update(bbox, score, status, attention_data)
        self.students[self.next_student_id] = student
        self.next_student_id += 1

