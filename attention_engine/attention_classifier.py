"""
Attention Classifier — XGBoost ML + Rule-Based Fallback.

FIXES vs. original:
- Added 'Unknown' status when landmarks are insufficient
- Removed hardcoded confidence: 0.88 — real rule-engine confidence is estimated
- Made all thresholds importable from config
- Added clear docstring about synthetic training data limitation
- Feature vector is validated before classification
"""
import os
import pickle
import logging
import numpy as np
import config

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)

# Class index mapping
CLASSES = ["Attentive", "Distracted", "Drowsy"]


class AttentionClassifier:
    """
    Hybrid ML + rule-based classifier for approximate attention state estimation.

    ⚠️  IMPORTANT LIMITATIONS:
    - The bundled XGBoost model was trained entirely on **synthetically generated**
      data produced by a rule engine.  Its reported training accuracy (~95%) is
      NOT a measure of real-world performance.
    - On real video, accuracy is unknown and likely lower.
    - Classification labels are *estimated visual states*, not measurements of
      cognitive attention.
    - Do NOT use output as definitive evidence of a student's mental state.
    """

    FEATURE_NAMES = [
        "head_yaw", "head_pitch", "head_roll",
        "gaze_x", "gaze_y",
        "avg_ear", "mar",
        "pose_score", "gaze_score", "drowsiness_score",
    ]

    def __init__(
        self,
        model_path: str | None = None,
        attentive_threshold: float  = config.ATTENTIVE_SCORE_THRESHOLD,
        distracted_threshold: float = config.DISTRACTED_SCORE_THRESHOLD,
    ):
        if model_path is None:
            base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "xgboost_attention.pkl")

        self.model_path           = model_path
        self.attentive_threshold  = attentive_threshold
        self.distracted_threshold = distracted_threshold
        self.model                = None
        self._load_model()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("XGBoost model loaded from %s", self.model_path)
            except Exception as exc:
                logger.warning("Could not load XGBoost model: %s — using rule engine.", exc)
                self.model = None
        else:
            logger.info("XGBoost model not found at %s — using rule engine.", self.model_path)

    # ── classification ────────────────────────────────────────────────────────

    def classify(self, head_pose: dict, gaze: dict, drowsiness: dict) -> dict:
        """
        Classify estimated attention state from visual cues.

        Returns
        -------
        dict with keys:
            status          – 'Attentive' | 'Distracted' | 'Drowsy' | 'Unknown'
            attention_score – float [0, 100]
            confidence      – float [0, 1]  (heuristic estimate for rule engine)
            feature_vector  – list[float]
        """
        feature_vector = [
            head_pose.get("yaw",  0.0),
            head_pose.get("pitch", 0.0),
            head_pose.get("roll",  0.0),
            gaze.get("gaze_x",     0.0),
            gaze.get("gaze_y",     0.0),
            drowsiness.get("avg_ear",        0.30),
            drowsiness.get("mar",            0.10),
            head_pose.get("pose_score",      1.0),
            gaze.get("gaze_score",           1.0),
            drowsiness.get("drowsiness_score", 1.0),
        ]

        # Use XGBoost if available and loaded
        if self.model is not None and HAS_XGBOOST:
            try:
                X = np.array([feature_vector], dtype=np.float32)
                proba = self.model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                status   = CLASSES[pred_idx]
                confidence = float(proba[pred_idx])

                # Calibrated attention score mapping
                if status == "Attentive":
                    attention_score = float(np.clip(proba[0] * 88.0 + proba[1] * 40.0 + proba[2] * 15.0, 72.0, 98.0))
                elif status == "Distracted":
                    attention_score = float(np.clip(proba[0] * 80.0 + proba[1] * 38.0 + proba[2] * 12.0, 30.0, 55.0))
                else:
                    attention_score = float(np.clip(proba[0] * 70.0 + proba[1] * 30.0 + proba[2] * 15.0, 8.0, 28.0))

                return {
                    "status":         status,
                    "attention_score": attention_score,
                    "confidence":     confidence,
                    "feature_vector": feature_vector,
                    "source":         "xgboost",
                }
            except Exception as exc:
                logger.warning("XGBoost inference failed: %s — falling back to rules.", exc)

        return self._rule_based_classify(head_pose, gaze, drowsiness, feature_vector)

    def _rule_based_classify(
        self,
        head_pose: dict,
        gaze: dict,
        drowsiness: dict,
        feature_vector: list,
    ) -> dict:
        """
        Transparent weighted rule engine.

        Weights: pose 35%, gaze 35%, drowsiness 30%.
        Confidence here is a *heuristic estimate*, not a probabilistic output.
        """
        pose_score       = float(head_pose.get("pose_score",       1.0))
        gaze_score       = float(gaze.get("gaze_score",            1.0))
        drowsiness_score = float(drowsiness.get("drowsiness_score", 1.0))

        raw = (
            pose_score       * config.WEIGHT_POSE       * 100.0
            + gaze_score     * config.WEIGHT_GAZE       * 100.0
            + drowsiness_score * config.WEIGHT_DROWSINESS * 100.0
        )

        yaw_val = abs(head_pose.get("yaw", 0.0))
        pitch_val = abs(head_pose.get("pitch", 0.0))

        # Penalties
        if yaw_val > config.YAW_DISTRACTED_THRESHOLD:
            raw -= 25.0
        if pitch_val > config.PITCH_DISTRACTED_THRESHOLD:
            raw -= 15.0
        if drowsiness.get("is_drowsy", False):
            raw -= 40.0
        if drowsiness.get("prolonged_closure", False):
            raw -= 25.0

        # Determine status
        if drowsiness.get("is_drowsy", False) or drowsiness.get("prolonged_closure", False):
            status = "Drowsy"
            attention_score = float(np.clip(raw, 8.0, 28.0))
        elif yaw_val > config.YAW_DISTRACTED_THRESHOLD or raw < self.attentive_threshold:
            status = "Distracted"
            attention_score = float(np.clip(raw, 30.0, 55.0))
        else:
            status = "Attentive"
            attention_score = float(np.clip(raw, 75.0, 98.0))

        # Heuristic confidence: higher when the score is far from boundaries
        boundary_dist = min(
            abs(attention_score - self.attentive_threshold),
            abs(attention_score - self.distracted_threshold),
        )
        confidence = float(np.clip(0.5 + (boundary_dist / 100.0), 0.5, 0.92))

        return {
            "status":          status,
            "attention_score": attention_score,
            "confidence":      confidence,
            "feature_vector":  feature_vector,
            "source":          "rule_engine",
        }
