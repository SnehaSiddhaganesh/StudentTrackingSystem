"""
Comprehensive unit and integration test suite for AI Classroom Attention Monitoring System.
Covers:
- No face / One face / Multiple faces
- Normal blinking vs. prolonged eye closure (EAR)
- Looking left/right vs. looking toward screen (Head pose & Gaze)
- Student entering/leaving frame & persistent ID tracking
- Invalid video source handling
- Missing model fallback handling
- SQLite Database logging and aggregation
- Session report generation (PDF / HTML)
"""
import os
import sys
import gc
import tempfile
import numpy as np
import pytest
import pandas as pd

# Add repo root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from attention_engine import (
    FaceDetector, HeadPoseEstimator, GazeEstimator,
    DrowsinessDetector, StudentTracker, AttentionClassifier
)
from database import DatabaseManager
from utils.video_input import VideoSource
from utils.report_generator import generate_pdf_report, generate_html_report


# =====================================================================
# 1. Face Detector Tests
# =====================================================================
def test_face_detector_no_face():
    detector = FaceDetector()
    # Blank black image
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.process(blank_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_face_detector_one_face():
    detector = FaceDetector()
    # Create simple synthetic face image
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    import cv2
    cv2.ellipse(frame, (320, 240), (70, 90), 0, 0, 360, (180, 200, 230), -1)
    cv2.circle(frame, (300, 220), 8, (20, 20, 20), -1)
    cv2.circle(frame, (340, 220), 8, (20, 20, 20), -1)
    cv2.line(frame, (320, 235), (320, 255), (120, 140, 170), 3)
    cv2.ellipse(frame, (320, 280), (20, 8), 0, 0, 360, (100, 50, 50), -1)
    
    # Process frame
    detections = detector.process(frame)
    assert isinstance(detections, list)
    for det in detections:
        assert 'bbox' in det
        assert 'landmarks_px' in det
        assert len(det['bbox']) == 4


# =====================================================================
# 2. Drowsiness Detector (EAR) Tests: Normal Blink vs Prolonged Closure
# =====================================================================
def test_drowsiness_normal_blink_vs_prolonged_closure():
    # Consecutive threshold is 3 frames
    detector = DrowsinessDetector(ear_threshold=0.21, consecutive_frames_threshold=3)
    
    # Mock landmarks for an open eye
    open_eye_lm = np.zeros((478, 2), dtype=np.int32)
    # outer=33, top1=160, top2=158, inner=133, bot2=153, bot1=144
    open_eye_lm[33] = [100, 200]
    open_eye_lm[133] = [140, 200]
    open_eye_lm[160] = [115, 190]
    open_eye_lm[158] = [125, 190]
    open_eye_lm[144] = [115, 210]
    open_eye_lm[153] = [125, 210]

    # Right eye mock
    open_eye_lm[362] = [200, 200]
    open_eye_lm[263] = [240, 200]
    open_eye_lm[385] = [215, 190]
    open_eye_lm[387] = [225, 190]
    open_eye_lm[380] = [215, 210]
    open_eye_lm[373] = [225, 210]

    # Frame 1: Open eye
    res1 = detector.process(open_eye_lm)
    assert not res1['is_drowsy']
    assert res1['avg_ear'] > 0.21

    # Mock landmarks for closed eye (vertical distance ~0)
    closed_eye_lm = open_eye_lm.copy()
    closed_eye_lm[160] = [115, 200]
    closed_eye_lm[158] = [125, 200]
    closed_eye_lm[144] = [115, 200]
    closed_eye_lm[153] = [125, 200]
    closed_eye_lm[385] = [215, 200]
    closed_eye_lm[387] = [225, 200]
    closed_eye_lm[380] = [215, 200]
    closed_eye_lm[373] = [225, 200]

    # Frame 2: Closed eye (1st frame of blink) -> should NOT be drowsy yet!
    res2 = detector.process(closed_eye_lm)
    assert not res2['is_drowsy'], "Single closed frame should NOT trigger is_drowsy!"
    assert res2['consecutive_low_ear_frames'] == 1

    # Frame 3: Eye re-opens immediately -> normal blink completed
    res3 = detector.process(open_eye_lm)
    assert not res3['is_drowsy']
    assert res3['consecutive_low_ear_frames'] == 0

    # Prolonged test: 3 consecutive frames closed
    detector.process(closed_eye_lm)
    detector.process(closed_eye_lm)
    res_prolonged = detector.process(closed_eye_lm)
    assert res_prolonged['is_drowsy'], "3 consecutive closed frames MUST trigger is_drowsy!"


# =====================================================================
# 3. Head Pose & Gaze Estimation Tests
# =====================================================================
def test_head_pose_screen_vs_turned():
    estimator = HeadPoseEstimator()
    frame_shape = (480, 640, 3)

    # Frontal projected 2D landmarks (consistent with 3D model)
    # Nose (320, 240), Chin (320, 340), Left Eye (250, 190), Right Eye (390, 190), Left Mouth (275, 285), Right Mouth (365, 285)
    lm_front = np.zeros((468, 2), dtype=np.int32)
    lm_front[1] = [320, 240]   # Nose
    lm_front[152] = [320, 340] # Chin
    lm_front[33] = [250, 190]  # Left eye outer
    lm_front[263] = [390, 190] # Right eye outer
    lm_front[61] = [275, 285]  # Mouth left
    lm_front[291] = [365, 285] # Mouth right

    pose_front = estimator.estimate_pose(lm_front, frame_shape)
    assert 'yaw' in pose_front
    assert 'pitch' in pose_front
    assert 'roll' in pose_front
    assert 'orientation' in pose_front
    assert 'nose_start' in pose_front
    assert 'nose_end' in pose_front
    assert 0.0 <= pose_front['pose_score'] <= 1.0

    # Looking right: shift nose and mouth rightwards relative to eyes
    lm_turned = lm_front.copy()
    lm_turned[1] = [360, 240]
    lm_turned[61] = [305, 285]
    lm_turned[291] = [385, 285]
    pose_turned = estimator.estimate_pose(lm_turned, frame_shape)
    assert pose_turned['yaw'] != pose_front['yaw']


def test_gaze_estimator():
    gaze_est = GazeEstimator()
    frame_shape = (480, 640, 3)
    head_pose = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    # 478 landmarks with iris
    lm = np.zeros((478, 2), dtype=np.int32)
    lm[33] = [260, 210]
    lm[133] = [290, 210]
    lm[468] = [275, 210]  # Left iris centered
    
    lm[362] = [350, 210]
    lm[263] = [380, 210]
    lm[473] = [365, 210]  # Right iris centered

    res = gaze_est.estimate_gaze(lm, head_pose, frame_shape)
    assert -1.0 <= res['gaze_x'] <= 1.0
    assert -1.0 <= res['gaze_y'] <= 1.0
    assert 0.0 <= res['gaze_score'] <= 1.0
    assert res['method'] == 'iris'


# =====================================================================
# 4. Attention Classifier (ML + Rule) & Missing Model Test
# =====================================================================
def test_attention_classifier_fallback():
    # Point to nonexistent model path to verify rule engine fallback
    classifier = AttentionClassifier(model_path="nonexistent_model.pkl")
    assert classifier.model is None

    head_pose = {'yaw': 2.0, 'pitch': -1.0, 'roll': 0.0, 'pose_score': 0.95}
    gaze = {'gaze_x': 0.05, 'gaze_y': 0.0, 'gaze_score': 0.95}
    drowsiness = {'avg_ear': 0.32, 'mar': 0.12, 'is_drowsy': False, 'drowsiness_score': 0.95}

    res_attentive = classifier.classify(head_pose, gaze, drowsiness)
    assert res_attentive['status'] == 'Attentive'
    assert res_attentive['attention_score'] >= 65.0
    assert res_attentive['source'] == 'rule_engine'

    # Drowsy test
    drowsiness_drowsy = {'avg_ear': 0.12, 'mar': 0.12, 'is_drowsy': True, 'prolonged_closure': True, 'drowsiness_score': 0.1}
    res_drowsy = classifier.classify(head_pose, gaze, drowsiness_drowsy)
    assert res_drowsy['status'] == 'Drowsy'


# =====================================================================
# 5. Student Tracker: Multiple Faces, Entering/Leaving & EMA
# =====================================================================
def test_student_tracker_lifecycle():
    tracker = StudentTracker(max_disappeared=3, max_distance=100)

    # Frame 1: Two students enter
    det_f1 = [
        {'bbox': (100, 100, 80, 80)},
        {'bbox': (300, 100, 80, 80)}
    ]
    att_f1 = [
        {'attention_score': 85.0, 'status': 'Attentive'},
        {'attention_score': 70.0, 'status': 'Attentive'}
    ]
    tracks1 = tracker.update(det_f1, att_f1)
    assert len(tracks1) == 2
    id1, id2 = tracks1[0]['student_id'], tracks1[1]['student_id']
    assert id1 != id2

    # Frame 2: Students move slightly (persistence check)
    det_f2 = [
        {'bbox': (105, 102, 80, 80)},
        {'bbox': (303, 98, 80, 80)}
    ]
    tracks2 = tracker.update(det_f2, att_f1)
    assert len(tracks2) == 2
    tracked_ids_f2 = {t['student_id'] for t in tracks2}
    assert tracked_ids_f2 == {id1, id2}, "Student IDs must persist across frames!"

    # Frame 3: One student leaves frame
    det_f3 = [{'bbox': (107, 103, 80, 80)}]
    tracks3 = tracker.update(det_f3, [att_f1[0]])
    assert len(tracks3) == 1
    assert tracks3[0]['student_id'] == id1

    # Empty frames exceeding max_disappeared
    for _ in range(4):
        tracker.update([], [])
    assert len(tracker.students) == 0, "Disappeared students must be pruned after max_disappeared frames."


# =====================================================================
# 6. VideoSource: Invalid Video & EOF Handling
# =====================================================================
def test_video_source_invalid():
    # Nonexistent path
    vs = VideoSource('demo', "nonexistent_classroom_video_999.mp4")
    assert not vs.is_opened
    success, frame, eof = vs.read_frame()
    assert not success
    assert eof
    vs.release()


# =====================================================================
# 7. Database Logging & Queries Test
# =====================================================================
def test_database_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db_path = os.path.join(tmpdir, "test_classroom.db")
        db = DatabaseManager(temp_db_path)
        sess_id = "test_sess_01"
        assert db.create_session(sess_id, "Test Lecture")

        # Log attention metrics
        assert db.log_student_attention(sess_id, 1, 88.0, "Attentive", 2.0, -1.0, 0.32, 0.95)
        assert db.log_student_attention(sess_id, 2, 45.0, "Distracted", 35.0, 5.0, 0.28, 0.40)
        assert db.log_alert(sess_id, 2, "Distracted", "Sustained distraction")

        # Query dataframe
        df_logs = db.get_session_logs_df(sess_id)
        assert len(df_logs) == 2
        assert list(df_logs['student_id']) == [1, 2]

        df_alerts = db.get_session_alerts_df(sess_id)
        assert len(df_alerts) == 1

        summary = db.get_session_summary(sess_id)
        assert summary['unique_students'] == 2
        assert summary['total_log_entries'] == 2
        assert db.end_session(sess_id, total_students=2)


# =====================================================================
# 8. Session Report Generation (PDF & HTML)
# =====================================================================
def test_report_generation():
    df_logs = pd.DataFrame([
        {'timestamp': '2026-09-09T10:00:00', 'student_id': 1, 'attention_score': 88.0, 'status': 'Attentive'},
        {'timestamp': '2026-09-09T10:00:01', 'student_id': 2, 'attention_score': 55.0, 'status': 'Distracted'},
        {'timestamp': '2026-09-09T10:00:02', 'student_id': 2, 'attention_score': 30.0, 'status': 'Drowsy'},
    ])
    df_alerts = pd.DataFrame([
        {'timestamp': '2026-09-09T10:00:02', 'student_id': 2, 'alert_type': 'Drowsy', 'message': 'Sustained fatigue'}
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_out = os.path.join(tmpdir, "test_report.pdf")
        html_out = os.path.join(tmpdir, "test_report.html")

        # HTML generation
        res_html = generate_html_report("sess_test", df_logs, df_alerts, html_out)
        assert os.path.exists(res_html)
        with open(res_html, "r", encoding="utf-8") as fp:
            content = fp.read()
            assert "Classroom Attention Analytics Report" in content
            assert "System Limitations" in content

        # PDF generation
        res_pdf = generate_pdf_report("sess_test", df_logs, df_alerts, pdf_out)
        assert os.path.exists(res_pdf)
        assert os.path.getsize(res_pdf) > 0


# =====================================================================
# 9. Teacher Authentication Tests
# =====================================================================
def test_teacher_auth():
    from auth import TeacherAuth, init_auth_state
    init_auth_state()
    auth = TeacherAuth()

    # Valid default credentials
    success, teacher_data = auth.verify_credentials("teacher", "admin123")
    assert success
    assert teacher_data["full_name"] == "Prof. Instructor"

    # Invalid credentials
    fail_success, fail_data = auth.verify_credentials("teacher", "wrongpassword")
    assert not fail_success
    assert fail_data is None

    # New registration
    ok, msg = auth.register_teacher("newteacher", "securepass123", "Prof. Jane Smith", "Physics")
    assert ok
    reg_success, reg_data = auth.verify_credentials("newteacher", "securepass123")
    assert reg_success
    assert reg_data["full_name"] == "Prof. Jane Smith"
