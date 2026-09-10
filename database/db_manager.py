"""
SQLite Database Manager — session, attention-log, and alert storage.

FIXES vs. original:
- All DB operations wrapped in try/except — no crash on DB failure
- Clean context manager ensures sqlite3 connection is closed immediately
  preventing file locks on Windows
- duration_sec computed properly in end_session
- get_session_summary added for analytics
- No raw video stored; only numeric/text metrics
"""
import sqlite3
import os
import logging
from contextlib import contextmanager
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite storage for monitoring sessions, per-student attention logs,
    and low-attention alerts.

    Privacy note: only numeric scores, anonymous student IDs, and timestamps
    are stored.  No biometric data, face images, or personal identifiers.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path  = os.path.join(base_dir, "database", "classroom_attention.db")

        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    # ── internals ─────────────────────────────────────────────────────────────

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.executescript("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id    TEXT PRIMARY KEY,
                        session_name  TEXT,
                        start_time    TEXT,
                        end_time      TEXT,
                        duration_sec  INTEGER DEFAULT 0,
                        avg_attention REAL    DEFAULT 0.0,
                        total_students INTEGER DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS attention_logs (
                        id             INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id     TEXT,
                        timestamp      TEXT,
                        student_id     INTEGER,
                        attention_score REAL,
                        status         TEXT,
                        head_yaw       REAL,
                        head_pitch     REAL,
                        avg_ear        REAL,
                        gaze_score     REAL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    );

                    CREATE TABLE IF NOT EXISTS alerts (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id  TEXT,
                        timestamp   TEXT,
                        student_id  INTEGER,
                        alert_type  TEXT,
                        message     TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    );
                """)
                # Migration check for existing databases
                try:
                    cur.execute("ALTER TABLE sessions ADD COLUMN teacher_username TEXT DEFAULT 'teacher'")
                except Exception:
                    pass

        except Exception as exc:
            logger.error("DB init failed: %s", exc)

    # ── session management ────────────────────────────────────────────────────

    def create_session(self, session_id: str, session_name: str = "Classroom Lecture", teacher_username: str = "teacher") -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, session_name, teacher_username, start_time) VALUES (?,?,?,?)",
                    (session_id, session_name, teacher_username, datetime.now().isoformat()),
                )
            return True
        except Exception as exc:
            logger.error("create_session failed: %s", exc)
            return False

    def end_session(self, session_id: str, total_students: int = 0) -> bool:
        try:
            end_time = datetime.now().isoformat()
            with self._get_connection() as conn:
                cur = conn.cursor()

                # Compute average attention
                cur.execute(
                    "SELECT AVG(attention_score), MIN(timestamp) FROM attention_logs WHERE session_id=?",
                    (session_id,),
                )
                row = cur.fetchone()
                avg_attention = row[0] if row and row[0] is not None else 0.0
                start_ts      = row[1] if row and row[1] is not None else end_time

                # Duration in seconds
                try:
                    start_dt = datetime.fromisoformat(start_ts)
                    end_dt   = datetime.fromisoformat(end_time)
                    duration = int((end_dt - start_dt).total_seconds())
                except Exception:
                    duration = 0

                cur.execute(
                    """UPDATE sessions
                       SET end_time=?, avg_attention=?, total_students=?, duration_sec=?
                       WHERE session_id=?""",
                    (end_time, avg_attention, total_students, duration, session_id),
                )
            return True
        except Exception as exc:
            logger.error("end_session failed: %s", exc)
            return False

    # ── logging ───────────────────────────────────────────────────────────────

    def log_student_attention(
        self,
        session_id: str,
        student_id: int,
        attention_score: float,
        status: str,
        head_yaw: float,
        head_pitch: float,
        avg_ear: float,
        gaze_score: float,
    ) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT INTO attention_logs
                       (session_id, timestamp, student_id, attention_score,
                        status, head_yaw, head_pitch, avg_ear, gaze_score)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (session_id, datetime.now().isoformat(), student_id,
                     attention_score, status, head_yaw, head_pitch, avg_ear, gaze_score),
                )
            return True
        except Exception as exc:
            logger.error("log_student_attention failed: %s", exc)
            return False

    def log_alert(self, session_id: str, student_id: int, alert_type: str, message: str) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO alerts (session_id, timestamp, student_id, alert_type, message) VALUES (?,?,?,?,?)",
                    (session_id, datetime.now().isoformat(), student_id, alert_type, message),
                )
            return True
        except Exception as exc:
            logger.error("log_alert failed: %s", exc)
            return False

    # ── queries ───────────────────────────────────────────────────────────────

    def get_session_logs_df(self, session_id: str) -> pd.DataFrame:
        try:
            with self._get_connection() as conn:
                return pd.read_sql_query(
                    """SELECT timestamp, student_id, attention_score, status,
                              head_yaw, head_pitch, avg_ear, gaze_score
                       FROM attention_logs WHERE session_id=? ORDER BY id ASC""",
                    conn, params=(session_id,),
                )
        except Exception as exc:
            logger.error("get_session_logs_df failed: %s", exc)
            return pd.DataFrame()

    def get_session_alerts_df(self, session_id: str) -> pd.DataFrame:
        try:
            with self._get_connection() as conn:
                return pd.read_sql_query(
                    "SELECT timestamp, student_id, alert_type, message FROM alerts WHERE session_id=? ORDER BY id DESC",
                    conn, params=(session_id,),
                )
        except Exception as exc:
            logger.error("get_session_alerts_df failed: %s", exc)
            return pd.DataFrame()

    def get_all_sessions(self, teacher_username: str | None = None) -> pd.DataFrame:
        try:
            with self._get_connection() as conn:
                if teacher_username:
                    return pd.read_sql_query(
                        "SELECT session_id, session_name, start_time, end_time, duration_sec, avg_attention, total_students FROM sessions WHERE teacher_username=? ORDER BY start_time DESC",
                        conn, params=(teacher_username,),
                    )
                return pd.read_sql_query(
                    "SELECT session_id, session_name, start_time, end_time, duration_sec, avg_attention, total_students FROM sessions ORDER BY start_time DESC",
                    conn,
                )
        except Exception as exc:
            logger.error("get_all_sessions failed: %s", exc)
            return pd.DataFrame()

    def get_session_summary(self, session_id: str) -> dict:
        """Return aggregate stats dict for a session (used in report generation)."""
        try:
            df = self.get_session_logs_df(session_id)
            if df.empty:
                return {}
            total = len(df)
            return {
                "avg_attention":     float(df["attention_score"].mean()),
                "max_attention":     float(df["attention_score"].max()),
                "min_attention":     float(df["attention_score"].min()),
                "attentive_pct":     float((df["status"] == "Attentive").sum())  / total * 100,
                "distracted_pct":    float((df["status"] == "Distracted").sum()) / total * 100,
                "drowsy_pct":        float((df["status"] == "Drowsy").sum())     / total * 100,
                "unknown_pct":       float((df["status"] == "Unknown").sum())    / total * 100,
                "unique_students":   int(df["student_id"].nunique()),
                "total_log_entries": int(total),
            }
        except Exception as exc:
            logger.error("get_session_summary failed: %s", exc)
            return {}
