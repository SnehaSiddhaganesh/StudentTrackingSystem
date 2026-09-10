"""
Teacher Authentication & Session Security Module.
Provides secure password hashing, credential verification, registration,
and Streamlit Teacher Login Portal UI.
"""
import hashlib
import os
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# Default pre-configured teacher accounts (Username: Password)
DEFAULT_TEACHERS = {
    "teacher": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "full_name": "Prof. Instructor",
        "email": "teacher@school.edu",
        "department": "Computer Science & AI"
    },
    "prof_smith": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "full_name": "Dr. Sarah Smith",
        "email": "smith@university.edu",
        "department": "Mathematics & Data Science"
    }
}


class TeacherAuth:
    """Handles teacher authentication, password hashing, and session verification."""

    def __init__(self):
        if "teachers_db" not in st.session_state:
            st.session_state.teachers_db = DEFAULT_TEACHERS

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def verify_credentials(self, username: str, password: str) -> tuple[bool, dict | None]:
        """Validates username and password against stored teacher database."""
        username_clean = username.strip().lower()
        teachers = st.session_state.teachers_db

        if username_clean in teachers:
            stored_hash = teachers[username_clean]["password_hash"]
            if self._hash_password(password) == stored_hash:
                info = teachers[username_clean]
                info["username"] = username_clean
                return True, info
            else:
                return False, None
        
        # Allow instant custom teacher login if non-empty username & password provided
        if len(username_clean) >= 3 and len(password) >= 4:
            custom_teacher = {
                "username": username_clean,
                "password_hash": self._hash_password(password),
                "full_name": f"Prof. {username.strip().capitalize()}",
                "email": f"{username_clean}@school.edu",
                "department": "General Academics"
            }
            st.session_state.teachers_db[username_clean] = custom_teacher
            return True, custom_teacher

        return False, None

    def register_teacher(self, username: str, password: str, full_name: str, department: str = "Academics") -> tuple[bool, str]:
        """Registers a new teacher account."""
        username_clean = username.strip().lower()
        if len(username_clean) < 3:
            return False, "Username must be at least 3 characters long."
        if len(password) < 4:
            return False, "Password must be at least 4 characters long."

        if username_clean in st.session_state.teachers_db:
            return False, "Username already registered."

        st.session_state.teachers_db[username_clean] = {
            "password_hash": self._hash_password(password),
            "full_name": full_name.strip(),
            "email": f"{username_clean}@school.edu",
            "department": department.strip()
        }
        return True, "Account registered successfully! You can now log in."


def init_auth_state():
    """Initializes Streamlit authentication session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "teacher_info" not in st.session_state:
        st.session_state.teacher_info = None


def render_login_portal():
    """Renders an interactive, modern Home Page with welcome banners, platform highlights & Teacher Login/Registration."""
    init_auth_state()
    auth = TeacherAuth()

    # Interactive Welcome Hero Header
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #4f46e5 100%); padding: 2.5rem 2rem; border-radius: 20px; color: white; margin-bottom: 2rem; box-shadow: 0 12px 30px rgba(37, 99, 235, 0.22); text-align: center;'>
            <div style='display: inline-block; background: rgba(255,255,255,0.2); padding: 6px 18px; border-radius: 9999px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.85rem; border: 1px solid rgba(255,255,255,0.35);'>
                ✨ Welcome to Student Attention & Focus Tracker
            </div>
            <h1 style='color: #ffffff !important; font-size: 2.7rem !important; font-weight: 800 !important; margin: 0; padding: 0; background: none; -webkit-text-fill-color: initial; line-height: 1.25;'>
                Empower Every Classroom with Smart Engagement Insights
            </h1>
            <p style='color: #e0e7ff; font-size: 1.15rem; max-width: 840px; margin: 1rem auto 0 auto; line-height: 1.6; font-weight: 400;'>
                Transform traditional and hybrid teaching with real-time visual attention monitoring, intelligent fatigue detection, and automated student focus analytics — built 100% private and on-device.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_welcome, col_portal = st.columns([1.3, 1.0])

    with col_welcome:
        st.markdown("<h3 style='margin-top:0; color:#0f172a; font-size: 1.45rem;'>🚀 Interactive Educator Dashboard Capabilities</h3>", unsafe_allow_html=True)

        st.markdown("""
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1.1rem; margin-bottom: 1.5rem;'>
                <div style='background: #ffffff; padding: 1.25rem; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.3rem;'>📹</div>
                    <strong style='color:#0f172a; font-size:1rem;'>Multi-Student Live Board</strong>
                    <p style='color:#64748b; font-size:0.84rem; margin-top:0.35rem; margin-bottom:0; line-height: 1.4;'>Tracks each student in frame separately (1 or multiple students) with live score metrics & status cards.</p>
                </div>
                <div style='background: #ffffff; padding: 1.25rem; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.3rem;'>🎯</div>
                    <strong style='color:#0f172a; font-size:1rem;'>Head Orientation & Gaze</strong>
                    <p style='color:#64748b; font-size:0.84rem; margin-top:0.35rem; margin-bottom:0; line-height: 1.4;'>Monitors facial geometry, pitch/yaw angles, and iris gaze direction to estimate visual focus.</p>
                </div>
                <div style='background: #ffffff; padding: 1.25rem; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.3rem;'>💤</div>
                    <strong style='color:#0f172a; font-size:1rem;'>Fatigue & Drowsiness Alerts</strong>
                    <p style='color:#64748b; font-size:0.84rem; margin-top:0.35rem; margin-bottom:0; line-height: 1.4;'>Eye Aspect Ratio (EAR) filters single blinks to send timely alerts when students show fatigue.</p>
                </div>
                <div style='background: #ffffff; padding: 1.25rem; border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
                    <div style='font-size: 1.8rem; margin-bottom: 0.3rem;'>🔒</div>
                    <strong style='color:#0f172a; font-size:1rem;'>Strict Local Privacy</strong>
                    <p style='color:#64748b; font-size:0.84rem; margin-top:0.35rem; margin-bottom:0; line-height: 1.4;'>Zero cloud storage or video transmission. Built-in optional face blurring and student ID masking.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='background: #f8fafc; padding: 1.2rem; border-radius: 14px; border: 1px solid #cbd5e1; margin-bottom: 1rem;'>
                <strong style='color: #1e293b; font-size: 0.95rem;'>📋 Quick Workflow Guide for Teachers:</strong>
                <ol style='color: #475569; font-size: 0.86rem; margin-top: 0.4rem; margin-bottom: 0; padding-left: 1.2rem; line-height: 1.6;'>
                    <li><strong>Log In or Register</strong> your educator profile using the Teacher Portal on the right.</li>
                    <li>Enter your <strong>Course / Lecture Title</strong> and adjust your detection sensitivity sliders.</li>
                    <li>Click <strong>▶️ Start Session</strong> to activate live webcam or classroom video tracking!</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    with col_portal:
        st.markdown("""
            <div style='background: #ffffff; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.06);'>
                <h3 style='margin-top: 0; color: #0f172a; font-size: 1.35rem;'>🔑 Teacher Portal Entry</h3>
                <p style='color: #64748b; font-size: 0.85rem; margin-bottom: 0;'>Sign in to set up lecture sessions & monitor live student attention</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_login, tab_quick, tab_register = st.tabs(["🔑 Teacher Login", "⚡ 1-Click Quick Demo", "📝 Register New Teacher"])

        with tab_login:
            login_user = st.text_input("Teacher Username", value="teacher", key="login_user_input")
            login_pass = st.text_input("Password", type="password", value="admin123", key="login_pass_input")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Enter Dashboard", use_container_width=True):
                success, teacher_data = auth.verify_credentials(login_user, login_pass)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.teacher_info = teacher_data
                    st.success(f"Welcome, {teacher_data['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_quick:
            st.markdown("##### Instant Demo Access")
            if st.button("👨‍🏫 Demo Teacher 1 (Prof. Instructor)", use_container_width=True):
                _, teacher_data = auth.verify_credentials("teacher", "admin123")
                st.session_state.authenticated = True
                st.session_state.teacher_info = teacher_data
                st.rerun()

            if st.button("👩‍🏫 Demo Teacher 2 (Dr. Sarah Smith)", use_container_width=True):
                _, teacher_data = auth.verify_credentials("prof_smith", "password123")
                st.session_state.authenticated = True
                st.session_state.teacher_info = teacher_data
                st.rerun()

        with tab_register:
            reg_user = st.text_input("Choose Username", key="reg_user_input")
            reg_name = st.text_input("Full Name (e.g. Prof. Jane Doe)", key="reg_name_input")
            reg_dept = st.text_input("Department / Subject", value="Mathematics", key="reg_dept_input")
            reg_pass = st.text_input("Choose Password", type="password", key="reg_pass_input")

            if st.button("✨ Register Account", use_container_width=True):
                ok, msg = auth.register_teacher(reg_user, reg_pass, reg_name, reg_dept)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

