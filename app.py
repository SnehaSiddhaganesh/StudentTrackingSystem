"""
AI-Based Classroom Attention Monitoring System - Main Application Dashboard.
Developed as a lightweight, ethical on-device visual analytics teaching aid.
"""
import os
import sys
import time
import uuid
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add base directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from attention_engine import (
    FaceDetector, HeadPoseEstimator, GazeEstimator,
    DrowsinessDetector, StudentTracker, AttentionClassifier
)
from database import DatabaseManager
try:
    from utils.sample_generator import generate_sample_video, generate_sample_image
except ImportError:
    from utils.sample_generator import generate_sample_video
    def generate_sample_image(output_path=None):
        if output_path is None:
            output_path = os.path.join(BASE_DIR, 'assets', 'sample_classroom.jpg')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = np.full((720, 1280, 3), (30, 35, 50), dtype=np.uint8)
        cv2.imwrite(output_path, img)
        return output_path
from utils.report_generator import generate_pdf_report, generate_html_report
from utils.video_input import VideoSource, get_available_cameras
from auth import init_auth_state, render_login_portal

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="AI Classroom Attention Monitoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS if available
css_path = os.path.join(BASE_DIR, 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Resource Initialization ---
@st.cache_resource
def initialize_system_resources():
    model_path = os.path.join(BASE_DIR, 'models', 'xgboost_attention.pkl')
    if not os.path.exists(model_path):
        train_xgboost_model(model_path)
        
    video_path = os.path.join(BASE_DIR, 'assets', 'sample_classroom.mp4')
    if not os.path.exists(video_path):
        generate_sample_video(video_path)

initialize_system_resources()

# Initialize Database Manager
db_manager = DatabaseManager()

# --- Teacher Authentication Check ---
init_auth_state()
if not st.session_state.get('authenticated', False):
    render_login_portal()
    st.stop()

teacher_info = st.session_state.get('teacher_info') or {}
teacher_name = teacher_info.get('full_name', 'Prof. Instructor')
teacher_username = teacher_info.get('username', 'teacher')

# --- Session State Defaults ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if 'is_running' not in st.session_state:
    st.session_state.is_running = False  # Controlled strictly by "Start Session" button
if 'attention_history' not in st.session_state:
    st.session_state.attention_history = []
if 'alerts_list' not in st.session_state:
    st.session_state.alerts_list = []
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'session_ended' not in st.session_state:
    st.session_state.session_ended = False
if 'last_session_id' not in st.session_state:
    st.session_state.last_session_id = None
if 'last_alert_times' not in st.session_state:
    st.session_state.last_alert_times = {}

# --- Sidebar Controls ---
st.sidebar.markdown(f"### 👨‍🏫 {teacher_name}")
st.sidebar.caption(f"Dept: {teacher_info.get('department', 'Academics')}")

if st.sidebar.button("🚪 Logout Teacher Portal", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.is_running = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎓 Session Controls")

course_title = st.sidebar.text_input("Course / Lecture Title", value="Mathematics 101")

# System Status Indicator
if st.session_state.is_running:
    st.sidebar.success("🟢 Monitoring Active")
else:
    st.sidebar.warning("⏸️ Session Stopped")

input_mode = st.sidebar.selectbox(
    "📹 Video Source",
    ["📷 Live Webcam Feed", "🖼️ Demo Classroom Image", "🎥 Demo Classroom Video", "📹 Upload Video File/Image"],
    index=0
)

uploaded_file = None
webcam_id = 0

if input_mode == "📹 Upload Video File/Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Classroom Video or Photo",
        type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"],
        disabled=st.session_state.is_running
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Detection Parameters")

min_attention_score = st.sidebar.slider(
    "Attentive Threshold Score (%)", 40, 90, int(config.ATTENTIVE_SCORE_THRESHOLD)
)
ear_thresh = st.sidebar.slider(
    "Drowsiness EAR Threshold", 0.15, 0.35, float(config.EAR_THRESHOLD), step=0.01
)
sustained_sec = st.sidebar.slider(
    "Alert Sustained Threshold (Sec)", 2, 10, int(config.ALERT_SUSTAINED_SECONDS)
)
max_faces = st.sidebar.slider("Max Detected Faces", 1, 15, int(config.MAX_NUM_FACES))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 Privacy Safeguards")
blur_faces = st.sidebar.checkbox("Real-time Face Blurring (Anonymization)", value=False)
mask_student_ids = st.sidebar.checkbox("Mask Student Display IDs", value=False)

st.sidebar.markdown("---")

col_start, col_stop = st.sidebar.columns(2)
with col_start:
    start_clicked = st.button("▶️ Start Session", use_container_width=True, disabled=st.session_state.is_running)
with col_stop:
    stop_clicked = st.button("⏹️ Stop & Save", use_container_width=True, disabled=not st.session_state.is_running)

if start_clicked:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.is_running = True
    st.session_state.session_start_time = time.time()
    st.session_state.attention_history = []
    st.session_state.alerts_list = []
    st.session_state.last_alert_times = {}
    st.session_state.session_ended = False
    st.session_state.last_session_id = st.session_state.session_id
    db_manager.create_session(st.session_state.session_id, f"{course_title}", teacher_username)
    st.rerun()

if stop_clicked:
    st.session_state.is_running = False
    st.session_state.session_ended = True
    db_manager.end_session(st.session_state.session_id)
    st.rerun()

# --- Main App Header ---
st.markdown(f"""
    <div style='background: #ffffff; padding: 1.25rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h1 style='margin:0; padding:0; font-size: 1.9rem;'>AI Classroom Attention Monitoring System</h1>
                <p style='color: #475569; font-weight: 500; margin-top: 0.2rem; margin-bottom: 0;'>
                    🏫 <strong>Lecture:</strong> {course_title} &nbsp;|&nbsp; 👨‍🏫 <strong>Instructor:</strong> {teacher_name} &nbsp;|&nbsp; 🔑 <strong>Session ID:</strong> <code>{st.session_state.session_id}</code>
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Privacy & Disclaimer Banner
st.info(
    "⚠️ **Ethical & Methodological Notice**: This system calculates **visual proxies** for attention "
    "(head orientation, gaze vector, eye openness). It does NOT measure cognitive focus, comprehension, "
    "or mental state. All processing is strictly local and on-device without cloud transmission. "
    "Data collection must be accompanied by institutional ethics clearance and participant consent."
)

# --- Top Real-time Class Metrics ---
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
with metric_col1:
    m_attention = st.empty()
    m_attention.metric("Avg Attention", "-- %")
with metric_col2:
    m_attentive = st.empty()
    m_attentive.metric("Attentive", "--")
with metric_col3:
    m_distracted = st.empty()
    m_distracted.metric("Distracted", "--")
with metric_col4:
    m_drowsy = st.empty()
    m_drowsy.metric("Drowsy", "--")
with metric_col5:
    m_unknown = st.empty()
    m_unknown.metric("Unknown / Insuff.", "--")
with metric_col6:
    m_fps = st.empty()
    m_fps.metric("Processing FPS", "--")

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid: Video Stream + Engagement Timeline
canvas_col, chart_col = st.columns([1.4, 1.0])

with canvas_col:
    st.markdown("### 📹 Live Feed & Visual Overlays")
    video_placeholder = st.empty()
    video_status_placeholder = st.empty()
    if not st.session_state.is_running:
        video_status_placeholder.info("⏸️ **Monitoring Stopped**. Select your settings in the sidebar and click **▶️ Start Session** to begin live camera/video tracking.")

with chart_col:
    st.markdown("### 📈 Attention Score Timeline")
    chart_placeholder = st.empty()
    if not st.session_state.is_running:
        chart_placeholder.info("Timeline will display attention score progression once session starts.")

st.markdown("---")

# Bottom Grid: Live Student Attention Board + Sustained Alerts Feed
student_col, alert_col = st.columns([1.3, 1.0])

with student_col:
    st.markdown("### 👥 Live Student Attention Tracking Board")
    student_cards_placeholder = st.empty()
    if not st.session_state.is_running:
        student_cards_placeholder.info("▶️ Click **Start Session** to launch real-time student tracking cards.")

with alert_col:
    st.markdown("### 🚨 Sustained Low-Attention & Fatigue Alerts")
    alerts_placeholder = st.empty()
    if not st.session_state.is_running:
        alerts_placeholder.info("No active alerts. Click **▶️ Start Session** to log real-time disengagement alerts.")

# --- Live Session Report & Summary ---
st.markdown("---")
st.markdown("### 📊 Live Session Report & Analytics Summary")

target_session = st.session_state.last_session_id or st.session_state.session_id
sess_logs = db_manager.get_session_logs_df(target_session)
sess_alerts = db_manager.get_session_alerts_df(target_session)

if not sess_logs.empty:
    unique_st_cnt = sess_logs['student_id'].nunique()
    avg_sess_score = sess_logs['attention_score'].mean()
    
    rep_m1, rep_m2, rep_m3, rep_m4 = st.columns(4)
    with rep_m1:
        st.metric("Total Students Tracked", f"{unique_st_cnt}")
    with rep_m2:
        st.metric("Overall Session Attention", f"{avg_sess_score:.1f} %")
    with rep_m3:
        st.metric("Total Logged Samples", f"{len(sess_logs)}")
    with rep_m4:
        st.metric("Sustained Fatigue Alerts", f"{len(sess_alerts)}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Per-Student Live Summary Cards (Displayed vertically one by one)
    st.markdown("#### 👨‍🎓 Per-Student Attention Summary")
    st_summary = sess_logs.groupby('student_id').agg(
        avg_score=('attention_score', 'mean'),
        last_status=('status', 'last')
    ).reset_index()
    
    for idx, row in st_summary.iterrows():
        sid_val = int(row['student_id'])
        score_val = row['avg_score']
        status_val = row['last_status']
        
        if status_val == "Attentive":
            badge_c, icon_c = "#047857", "🟢"
        elif status_val == "Distracted":
            badge_c, icon_c = "#b45309", "🟡"
        else:
            badge_c, icon_c = "#b91c1c", "🔴"
            
        disp_name = f"Student #{sid_val}" if not mask_student_ids else f"Student #{sid_val * 13 % 97}"
        st.markdown(f"""
            <div style='background: #ffffff; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 0.75rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <strong style='color:#0f172a; font-size: 1.05rem;'>{disp_name}</strong>
                    <span style='color:{badge_c}; font-weight:700; font-size: 1rem;'>{icon_c} {score_val:.1f}%</span>
                </div>
                <div style='font-size:0.88rem; color:#475569; margin-top:0.4rem;'>
                    Average Attention: <strong>{score_val:.1f}%</strong> &nbsp;|&nbsp; Dominant Status: <strong>{status_val}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Download Buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        pdf_path = generate_pdf_report(target_session, sess_logs, sess_alerts)
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as fp:
                st.download_button(
                    label="📥 Download PDF Summary Report",
                    data=fp,
                    file_name=f"Attention_Report_{target_session}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_btn_{target_session}"
                )
    with btn_col2:
        from utils.report_generator import generate_html_report
        html_out_path = os.path.join(BASE_DIR, 'assets', f"Session_Report_{target_session}.html")
        generate_html_report(target_session, sess_logs, sess_alerts, html_out_path)
        if os.path.exists(html_out_path):
            with open(html_out_path, "r", encoding="utf-8") as fp:
                st.download_button(
                    label="📥 Download HTML Summary Report",
                    data=fp.read(),
                    file_name=f"Attention_Report_{target_session}.html",
                    mime="text/html",
                    use_container_width=True,
                    key=f"html_btn_{target_session}"
                )
else:
    st.info("No tracking data logged for the current session yet. Click **▶️ Start Session** to record attention data.")


# --- Main Video Processing Loop ---
if st.session_state.is_running:
    # Resolve video source
    video_src = None
    if input_mode == "🖼️ Demo Classroom Image":
        sample_img_path = os.path.join(BASE_DIR, 'assets', 'sample_classroom.jpg')
        if not os.path.exists(sample_img_path):
            generate_sample_image(sample_img_path)
        video_src = VideoSource('image', sample_img_path, loop=True)
    elif input_mode == "🎥 Demo Classroom Video":
        sample_path = os.path.join(BASE_DIR, 'assets', 'sample_classroom.mp4')
        if not os.path.exists(sample_path):
            generate_sample_video(sample_path)
        video_src = VideoSource('demo', sample_path, loop=True)
    elif input_mode in ["📹 Upload Video File", "📹 Upload Video File/Image"]:
        if uploaded_file is not None:
            video_src = VideoSource('upload', uploaded_file, loop=False)
        else:
            video_status_placeholder.error("Please upload a valid classroom video file or photo image in the sidebar.")
            st.session_state.is_running = False
    elif input_mode == "📷 Live Webcam Feed":
        video_src = VideoSource('webcam', webcam_id)

    if video_src is None or not video_src.is_opened:
        video_status_placeholder.error(
            f"Unable to access video stream ({input_mode}). "
            "Please check device permissions or select Demo video."
        )
        st.session_state.is_running = False
    else:
        # Initialize pipeline modules
        detector = FaceDetector(max_num_faces=max_faces)
        head_pose_est = HeadPoseEstimator()
        gaze_est = GazeEstimator()
        drowsiness_det = DrowsinessDetector(ear_threshold=ear_thresh)
        student_tracker = StudentTracker()
        classifier = AttentionClassifier(attentive_threshold=min_attention_score)

        fps_timer = time.time()
        frame_counter = 0
        computed_fps = 0.0

        try:
            while st.session_state.is_running:
                success, frame, is_eof = video_src.read_frame()
                if not success:
                    if is_eof:
                        video_status_placeholder.info("Video playback completed.")
                    else:
                        video_status_placeholder.warning("Frame read warning or stream disconnected.")
                    break

                frame_counter += 1
                if frame_counter % 5 == 0:
                    now = time.time()
                    dt = now - fps_timer
                    if dt > 0:
                        computed_fps = 5.0 / dt
                    fps_timer = now

                h, w = frame.shape[:2]

                # 1. Face detection
                is_static = input_mode in ["🖼️ Demo Classroom Image", "📹 Upload Video File/Image"]
                detections = detector.process(frame, static_image=is_static)

                # 2. Extract cues & classify per detection
                raw_results = []
                for det in detections:
                    lm_px = det['landmarks_px']
                    confidence = det.get('confidence', 0.8)

                    if len(lm_px) < config.MIN_LANDMARKS_FOR_CLASSIFICATION or confidence < 0.5:
                        att_res = {
                            'status': 'Unknown',
                            'attention_score': 50.0,
                            'confidence': 0.0,
                            'feature_vector': [0.0] * 10
                        }
                        head_pose = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'pose_score': 0.5, 'orientation': 'Unknown'}
                        gaze = {'gaze_x': 0.0, 'gaze_y': 0.0, 'gaze_score': 0.5}
                        drowsiness = {'avg_ear': 0.30, 'mar': 0.10, 'is_drowsy': False, 'drowsiness_score': 0.5}
                    else:
                        head_pose = head_pose_est.estimate_pose(lm_px, frame.shape)
                        gaze = gaze_est.estimate_gaze(lm_px, head_pose, frame.shape)
                        drowsiness = drowsiness_det.process(lm_px)
                        att_res = classifier.classify(head_pose, gaze, drowsiness)

                    att_res['head_pose'] = head_pose
                    att_res['gaze'] = gaze
                    att_res['drowsiness'] = drowsiness
                    raw_results.append(att_res)

                # 3. Multi-student persistent tracking & EMA
                tracked_students = student_tracker.update(detections, raw_results)

                # 4. Aggregations & Visual Overlays
                attentive_cnt = 0
                distracted_cnt = 0
                drowsy_cnt = 0
                unknown_cnt = 0
                total_attention_score = 0.0

                for student in tracked_students:
                    sid = student['student_id']
                    bx, by, bw, bh = student['bbox']
                    smoothed_score = student['smoothed_attention']
                    att_data = student.get('latest_attention_data', {})
                    status = att_data.get('status', 'Attentive' if smoothed_score >= min_attention_score else 'Distracted')

                    if status == 'Attentive':
                        attentive_cnt += 1
                    elif status == 'Distracted':
                        distracted_cnt += 1
                    elif status == 'Drowsy':
                        drowsy_cnt += 1
                    else:
                        unknown_cnt += 1

                    total_attention_score += smoothed_score

                    # Face Blurring (Privacy Anonymization)
                    if blur_faces:
                        roi = frame[by:by+bh, bx:bx+bw]
                        if roi.size > 0:
                            frame[by:by+bh, bx:bx+bw] = cv2.GaussianBlur(roi, (51, 51), 30)

                    # Bounding Box Color
                    if status == 'Attentive':
                        box_color = (50, 205, 50)     # Green
                    elif status == 'Distracted':
                        box_color = (0, 215, 255)     # Yellow
                    elif status == 'Drowsy':
                        box_color = (50, 50, 255)     # Red
                    else:
                        box_color = (180, 180, 180)   # Gray

                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)

                    # Draw 3D RGB Coordinate Axes (X-Red, Y-Green, Z-Blue) & Head Pose Vector
                    hp = att_data.get('head_pose', {})
                    HeadPoseEstimator.draw_3d_axes(frame, hp)
                    if 'nose_start' in hp and 'nose_end' in hp:
                        cv2.arrowedLine(frame, hp['nose_start'], hp['nose_end'], (255, 255, 0), 2, tipLength=0.3)

                    # Label card
                    disp_id = f"Student #{sid}" if not mask_student_ids else f"Student #{sid * 13 % 97}"
                    label_str = f"{disp_id} | {status} ({smoothed_score:.0f}%)"
                    cv2.rectangle(frame, (bx, by - 24), (bx + len(label_str) * 9, by), box_color, -1)
                    cv2.putText(frame, label_str, (bx + 4, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    # Database Logging
                    db_manager.log_student_attention(
                        st.session_state.session_id, sid, smoothed_score, status,
                        hp.get('yaw', 0.0), hp.get('pitch', 0.0),
                        att_data.get('drowsiness', {}).get('avg_ear', 0.3),
                        att_data.get('gaze', {}).get('gaze_score', 1.0)
                    )

                    # Sustained Low-Attention Alert Check with Cooldown
                    if student['sustained_low_seconds'] >= sustained_sec:
                        now_ts = time.time()
                        last_log_t = st.session_state.last_alert_times.get(sid, 0.0)
                        if now_ts - last_log_t >= config.ALERT_COOLDOWN_SECONDS:
                            st.session_state.last_alert_times[sid] = now_ts
                            alert_msg = f"Sustained {status} state for {disp_id} (>= {sustained_sec}s)"
                            db_manager.log_alert(st.session_state.session_id, sid, status, alert_msg)
                            
                            st.session_state.alerts_list.append({
                                'time': time.strftime('%H:%M:%S'),
                                'sid': sid,
                                'disp_id': disp_id,
                                'status': status,
                                'msg': alert_msg
                            })

                # Compute Class Average Score
                n_tracked = len(tracked_students)
                avg_class_attention = (total_attention_score / n_tracked) if n_tracked > 0 else 0.0

                # Append to timeline history (limit to last 60 entries)
                st.session_state.attention_history.append({
                    'time': time.strftime('%H:%M:%S'),
                    'attention': avg_class_attention,
                    'attentive': attentive_cnt,
                    'distracted': distracted_cnt,
                    'drowsy': drowsy_cnt,
                    'unknown': unknown_cnt
                })
                if len(st.session_state.attention_history) > 60:
                    st.session_state.attention_history.pop(0)

                # --- Update Dashboard UI Elements ---
                # Update top metric counters
                m_attention.metric("Avg Attention", f"{avg_class_attention:.1f} %")
                m_attentive.metric("Attentive", f"{attentive_cnt}")
                m_distracted.metric("Distracted", f"{distracted_cnt}")
                m_drowsy.metric("Drowsy", f"{drowsy_cnt}")
                m_unknown.metric("Unknown / Insuff.", f"{unknown_cnt}")
                m_fps.metric("Processing FPS", f"{computed_fps:.1f}")

                # Update Video Canvas
                rgb_canvas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_canvas, channels="RGB", use_container_width=True)

                # Update Real-time Trend Chart
                df_hist = pd.DataFrame(st.session_state.attention_history)
                if not df_hist.empty:
                    if HAS_PLOTLY:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_hist['time'], y=df_hist['attention'],
                            mode='lines+markers', name='Attention %',
                            line=dict(color='#2563eb', width=2.5),
                            fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.08)'
                        ))
                        fig.add_hline(
                            y=min_attention_score, line_dash="dash", line_color="#d97706",
                            annotation_text="Target Threshold", annotation_font_color="#d97706"
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='#ffffff',
                            font=dict(color='#475569', family='Inter'),
                            margin=dict(l=10, r=10, t=20, b=10),
                            height=260,
                            xaxis=dict(gridcolor='#e2e8f0', title="Time"),
                            yaxis=dict(range=[0, 100], title="Score %", gridcolor='#e2e8f0')
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"trend_{frame_counter}")
                    else:
                        chart_placeholder.line_chart(df_hist.set_index('time')['attention'], height=240)

                # Update Live Student Attention Board Cards
                if tracked_students:
                    with student_cards_placeholder.container():
                        card_cols = st.columns(min(max(len(tracked_students), 1), 3))
                        for idx, st_item in enumerate(tracked_students):
                            col_target = card_cols[idx % len(card_cols)]
                            sid = st_item['student_id']
                            sc = st_item['smoothed_attention']
                            st_data = st_item.get('latest_attention_data', {})
                            st_status = st_data.get('status', 'Attentive' if sc >= min_attention_score else 'Distracted')
                            hp_desc = st_data.get('head_pose', {}).get('orientation', 'Facing Screen')
                            ear_val = st_data.get('drowsiness', {}).get('avg_ear', 0.30)
                            disp_sid = f"Student #{sid}" if not mask_student_ids else f"Student #{sid * 13 % 97}"

                            if st_status == "Attentive":
                                badge_bg, badge_fg, icon = "#d1fae5", "#047857", "🟢"
                            elif st_status == "Distracted":
                                badge_bg, badge_fg, icon = "#fef3c7", "#b45309", "🟡"
                            elif st_status == "Drowsy":
                                badge_bg, badge_fg, icon = "#fee2e2", "#b91c1c", "🔴"
                            else:
                                badge_bg, badge_fg, icon = "#f1f5f9", "#475569", "⚪"

                            with col_target:
                                with st.container(border=True):
                                    st.markdown(f"**{disp_sid}** &nbsp; <span style='background:{badge_bg}; color:{badge_fg}; font-weight:700; padding:2px 7px; border-radius:10px; font-size:0.75rem;'>{icon} {sc:.0f}%</span>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='font-size:0.8rem; color:#475569; margin-top:4px;'>Status: <strong>{st_status}</strong></div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='font-size:0.75rem; color:#64748b;'>📐 {hp_desc} | 👁️ EAR: {ear_val:.2f}</div>", unsafe_allow_html=True)
                                    st.progress(min(1.0, max(0.0, float(sc) / 100.0)))
                else:
                    student_cards_placeholder.info("No student faces currently detected in frame.")

                # Update Alerts Log
                if st.session_state.alerts_list:
                    with alerts_placeholder.container():
                        for al in list(reversed(st.session_state.alerts_list))[:4]:
                            st.warning(f"⚠️ [{al['time']}] {al['disp_id']} ({al['status']}): {al['msg']}")
                else:
                    alerts_placeholder.info("No sustained alerts triggered.")

                time.sleep(0.01)

        finally:
            video_src.release()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748B; font-size: 0.8rem;'>"
    "AI-Based Classroom Attention Monitoring System • Research & Teaching Aid Demonstration"
    "</p>",
    unsafe_allow_html=True
)
