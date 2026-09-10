"""
Video Input Handler for AI-Based Classroom Attention Monitoring System.
Handles Webcam, Uploaded Video Files, and Demo/Synthetic Video Streams.
Provides graceful error handling for missing devices, corrupted files, and EOF.
"""
import os
import cv2
import tempfile
import logging
from typing import Tuple, Optional, Generator

logger = logging.getLogger(__name__)


def get_available_cameras(max_tested: int = 4) -> list[int]:
    """Scans and returns indices of available webcam devices."""
    available = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
        else:
            # Try default backend
            cap_any = cv2.VideoCapture(idx)
            if cap_any.isOpened():
                available.append(idx)
                cap_any.release()
    return available if available else [0]


class VideoSource:
    """
    Robust video capture wrapper that handles camera sources, video files,
    and automatic cleanup of temporary uploaded files.
    """
    def __init__(self, source_type: str, source_value=None, loop: bool = False):
        """
        source_type: 'demo' | 'webcam' | 'upload'
        source_value: file path (str), device index (int), or uploaded file buffer / bytes
        loop: whether to loop video at EOF (typically True for demo video)
        """
        self.source_type = source_type
        self.source_value = source_value
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self._temp_file_path: Optional[str] = None
        self.is_opened = False
        self.fps = 30.0
        self.width = 640
        self.height = 480
        self.frame_count = 0
        self._open()

    def _open(self):
        try:
            if self.source_type == 'webcam':
                device_idx = int(self.source_value) if self.source_value is not None else 0
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
                
                opened = False
                for backend in backends:
                    try:
                        self.cap = cv2.VideoCapture(device_idx, backend)
                        if self.cap and self.cap.isOpened():
                            # Warm up check
                            for _ in range(5):
                                ret, frame = self.cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    opened = True
                                    break
                                cv2.waitKey(20)
                            if opened:
                                logger.info(f"Webcam {device_idx} successfully opened with backend {backend}.")
                                break
                            else:
                                self.cap.release()
                    except Exception as be_err:
                        logger.warning(f"Backend {backend} failed for camera {device_idx}: {be_err}")

                if not opened:
                    # Final fallback: standard index call without backend flag
                    self.cap = cv2.VideoCapture(device_idx)
                    if self.cap and self.cap.isOpened():
                        opened = True

            elif self.source_type in ['demo', 'image']:
                if isinstance(self.source_value, str) and os.path.exists(self.source_value):
                    if self.source_value.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_frame = cv2.imread(self.source_value)
                        if self.image_frame is not None:
                            self.is_opened = True
                            self.height, self.width = self.image_frame.shape[:2]
                        else:
                            self.is_opened = False
                    else:
                        self.cap = cv2.VideoCapture(self.source_value)
                else:
                    logger.error(f"Demo file path not found: {self.source_value}")
                    self.cap = None

            elif self.source_type == 'upload':
                # Handle Streamlit UploadedFile or file-like buffer
                if hasattr(self.source_value, 'read'):
                    suffix = '.mp4'
                    if hasattr(self.source_value, 'name'):
                        _, ext = os.path.splitext(self.source_value.name)
                        if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png']:
                            suffix = ext.lower()

                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    self._temp_file_path = tf.name
                    self.source_value.seek(0)
                    tf.write(self.source_value.read())
                    tf.flush()
                    tf.close()

                    if suffix in ['.jpg', '.jpeg', '.png']:
                        self.image_frame = cv2.imread(self._temp_file_path)
                        if self.image_frame is not None:
                            self.is_opened = True
                            self.height, self.width = self.image_frame.shape[:2]
                    else:
                        self.cap = cv2.VideoCapture(self._temp_file_path)
                elif isinstance(self.source_value, str) and os.path.exists(self.source_value):
                    if self.source_value.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_frame = cv2.imread(self.source_value)
                        if self.image_frame is not None:
                            self.is_opened = True
                            self.height, self.width = self.image_frame.shape[:2]
                    else:
                        self.cap = cv2.VideoCapture(self.source_value)
                else:
                    logger.error("Invalid uploaded file buffer or path provided.")
                    self.cap = None

            if (self.cap and self.cap.isOpened()) or getattr(self, 'image_frame', None) is not None:
                self.is_opened = True
                if self.cap and self.cap.isOpened():
                    f_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.fps = f_fps if f_fps > 0 and f_fps < 120 else 30.0
                    self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                else:
                    self.fps = 15.0
            else:
                self.is_opened = False
                logger.warning(f"Failed to open video source ({self.source_type}).")

        except Exception as e:
            logger.error(f"Error opening video source ({self.source_type}): {e}")
            self.is_opened = False

    def read_frame(self) -> Tuple[bool, Optional[object], bool]:
        """
        Reads next frame from stream.
        Returns:
            (success: bool, frame: np.ndarray, is_eof: bool)
        """
        if getattr(self, 'image_frame', None) is not None:
            return True, self.image_frame.copy(), False

        if not self.is_opened or self.cap is None:
            return False, None, True

        ret, frame = self.cap.read()
        
        # Retry up to 5 times for transient camera read drops
        if not ret or frame is None:
            if self.source_type == 'webcam':
                for _ in range(5):
                    cv2.waitKey(10)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        return True, frame, False
            elif self.loop and self.source_type in ['demo', 'upload']:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame, False
            return False, None, True

        return True, frame, False

    def release(self):
        """Releases capture and cleans up any temp files."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.is_opened = False

        if self._temp_file_path and os.path.exists(self._temp_file_path):
            try:
                os.remove(self._temp_file_path)
            except Exception as e:
                logger.warning(f"Could not remove temp video file {self._temp_file_path}: {e}")
            self._temp_file_path = None

    def __del__(self):
        self.release()
