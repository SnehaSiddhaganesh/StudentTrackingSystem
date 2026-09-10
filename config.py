"""
config.py — Centralised configurable thresholds and parameters.

All tuneable values live here so they can be adjusted without touching
processing logic.  UI sliders in app.py override these defaults at runtime.
"""

# ── Face Detection ─────────────────────────────────────────────────────────────
MAX_NUM_FACES: int = 10
MIN_DETECTION_CONFIDENCE: float = 0.5
MIN_TRACKING_CONFIDENCE: float = 0.5

# ── Head Pose ──────────────────────────────────────────────────────────────────
# Angles beyond which a student is considered "turned away"
YAW_DISTRACTED_THRESHOLD: float = 30.0    # degrees left/right
PITCH_DISTRACTED_THRESHOLD: float = 25.0  # degrees up/down
ROLL_DISTRACTED_THRESHOLD: float = 30.0   # degrees tilt

# ── Gaze ───────────────────────────────────────────────────────────────────────
GAZE_DISTRACTED_DISTANCE: float = 0.5     # normalised gaze distance from centre

# ── Drowsiness / Eye Closure ───────────────────────────────────────────────────
EAR_THRESHOLD: float = 0.17               # below → eye considered genuinely closed
MAR_THRESHOLD: float = 0.60              # above → yawn detected
# Number of *consecutive* frames with EAR < threshold before flagging drowsy.
# At 15 fps: 8 frames ≈ 0.5 s  (avoids blinks and downward gaze false positives)
CONSECUTIVE_DROWSY_FRAMES: int = 8
# Frames of low EAR that indicate prolonged closure vs. a blink
PROLONGED_CLOSURE_FRAMES: int = 20       # ≈ 1.3 s at 15 fps

# ── Attention Scoring Weights ──────────────────────────────────────────────────
WEIGHT_POSE: float = 0.35
WEIGHT_GAZE: float = 0.35
WEIGHT_DROWSINESS: float = 0.30

# Thresholds for final status labels
ATTENTIVE_SCORE_THRESHOLD: float = 65.0
DISTRACTED_SCORE_THRESHOLD: float = 40.0

# Minimum landmark count required to produce a real result (vs. "Unknown")
MIN_LANDMARKS_FOR_CLASSIFICATION: int = 50

# ── Student Tracker ────────────────────────────────────────────────────────────
EMA_ALPHA: float = 0.25                   # EMA smoothing factor
TRACKER_MAX_DISAPPEARED: int = 60         # 60 frames (4s) before a track is dropped
TRACKER_MAX_CENTROID_DISTANCE: float = 180.0  # pixels

# ── Alerts ─────────────────────────────────────────────────────────────────────
ALERT_SUSTAINED_SECONDS: float = 4.0     # default sustained-low-attention window
ALERT_DEDUP_WINDOW: int = 5              # skip alert if same student alerted in last N alerts
ALERT_COOLDOWN_SECONDS: float = 15.0     # minimum seconds between database alert logs per student

# ── Performance ────────────────────────────────────────────────────────────────
TARGET_FPS: int = 15                      # target processing rate
FRAME_RESIZE_WIDTH: int = 640            # resize input frame for faster processing
ATTENTION_HISTORY_SECONDS: int = 120     # keep last N seconds of class timeline
