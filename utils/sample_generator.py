"""
Generates a realistic simulated classroom video file (MP4) for testing and demo purposes.
"""
import os
import cv2
import numpy as np

def generate_sample_video(output_path=None, duration_sec=15, fps=20):
    """
    Generates a synthetic classroom MP4 video with 3 animated student faces displaying various behaviors.
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, 'assets', 'sample_classroom.mp4')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    width, height = 1280, 720
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Base background (dark classroom setup)
    bg_color = (25, 30, 45) # dark blue-gray

    # Student base positions in frame
    students = [
        {'id': 1, 'center': (320, 360), 'scale': 1.0, 'skin': (180, 210, 240)}, # Student 1
        {'id': 2, 'center': (640, 360), 'scale': 1.0, 'skin': (170, 200, 235)}, # Student 2
        {'id': 3, 'center': (960, 360), 'scale': 1.0, 'skin': (190, 220, 245)}, # Student 3
    ]

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Draw Classroom Desk & Board background elements
        cv2.rectangle(frame, (50, 40), (1230, 140), (40, 50, 70), -1)
        cv2.putText(frame, "CLASSROOM ATTENTION MONITORING DEMO FEED", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 220, 255), 2)

        # Draw Desk rail
        cv2.rectangle(frame, (0, 550), (1280, 720), (35, 45, 60), -1)

        for s in students:
            cx, cy = s['center']
            skin_color = s['skin']
            
            # Default state parameters
            head_shift_x, head_shift_y = 0, 0
            eye_close_factor = 0.0 # 0.0 = open, 1.0 = closed
            yawn_factor = 0.0

            # Dynamic behaviors over time
            if s['id'] == 1:
                # Student 1: Highly attentive, slight breathing movement
                head_shift_y = int(np.sin(t * 2) * 3)
                if int(t) % 4 == 0 and (t - int(t)) < 0.2:
                    eye_close_factor = 0.9 # blink
            
            elif s['id'] == 2:
                # Student 2: Turns head away (Distracted) at t=4s to t=9s
                if 4.0 <= t <= 9.0:
                    head_shift_x = int((t - 4.0) * 15) if t < 6.0 else 30
                    if t > 7.5:
                        head_shift_x = int((9.0 - t) * 20)
                else:
                    head_shift_x = int(np.sin(t * 1.5) * 5)
            
            elif s['id'] == 3:
                # Student 3: Gets Drowsy & Yawns at t=5s to t=12s
                if 5.0 <= t <= 12.0:
                    eye_close_factor = min(1.0, (t - 5.0) * 0.3)
                    head_shift_y = int((t - 5.0) * 4)
                    if 8.0 <= t <= 10.5:
                        yawn_factor = np.sin((t - 8.0) / 2.5 * np.pi)

            # Draw Head Ellipse
            fx, fy = cx + head_shift_x, cy + head_shift_y
            cv2.ellipse(frame, (fx, fy), (90, 120), 0, 0, 360, skin_color, -1)
            cv2.ellipse(frame, (fx, fy), (90, 120), 0, 0, 360, (120, 140, 160), 3)

            # Hair cap
            cv2.ellipse(frame, (fx, fy - 50), (92, 70), 0, 180, 360, (50, 40, 30), -1)

            # Eyes
            eye_off_x = 35
            eye_y = fy - 20
            eye_h = int(14 * (1.0 - eye_close_factor))
            
            # Left Eye
            cv2.ellipse(frame, (fx - eye_off_x, eye_y), (22, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (fx - eye_off_x + int(head_shift_x * 0.2), eye_y), max(1, eye_h - 4), (40, 30, 20), -1)
            
            # Right Eye
            cv2.ellipse(frame, (fx + eye_off_x, eye_y), (22, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (fx + eye_off_x + int(head_shift_x * 0.2), eye_y), max(1, eye_h - 4), (40, 30, 20), -1)

            # Eyebrows
            cv2.line(frame, (fx - eye_off_x - 18, eye_y - 18), (fx - eye_off_x + 18, eye_y - 18), (60, 50, 40), 3)
            cv2.line(frame, (fx + eye_off_x - 18, eye_y - 18), (fx + eye_off_x + 18, eye_y - 18), (60, 50, 40), 3)

            # Nose
            cv2.line(frame, (fx, eye_y), (fx + int(head_shift_x * 0.3), fy + 20), (140, 160, 190), 3)

            # Mouth / Yawn
            mouth_y = fy + 55
            mouth_h = int(8 + (yawn_factor * 35))
            cv2.ellipse(frame, (fx, mouth_y), (28, max(4, mouth_h)), 0, 0, 360, (180, 70, 70), -1)

            # Label student ID tag
            cv2.putText(frame, f"Student #{s['id']}", (fx - 45, fy + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 255), 2)

        out.write(frame)

    out.release()
    print(f"Sample classroom video successfully generated at {output_path}")
    return output_path


def generate_sample_image(output_path=None, force=False):
    """
    Generates a realistic classroom demo image containing 3 students displaying key attention states:
    - Student 1: Attentively listening to lecture (facing forward, eyes open)
    - Student 2: Sleeping / head resting down (eyes closed, low EAR)
    - Student 3: Looking outside window (head turned sideways, looking away)
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, 'assets', 'sample_classroom.jpg')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    width, height = 1280, 720
    bg_color = (25, 30, 45)
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Classroom Blackboard & Title Banner
    cv2.rectangle(frame, (50, 30), (1230, 130), (35, 45, 65), -1)
    cv2.putText(frame, "AI CLASSROOM ATTENTION TRACKING - DEMO SESSION", (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 235, 255), 2)
    cv2.putText(frame, "3 Students: 1 Attentive | 1 Sleeping | 1 Looking Outside Window", (80, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 190, 220), 1)

    # Classroom Desk
    cv2.rectangle(frame, (0, 560), (1280, 720), (30, 40, 55), -1)
    cv2.line(frame, (0, 560), (1280, 560), (70, 85, 110), 3)

    # 3 Students configuration
    students = [
        # Student 1: Attentive (facing screen)
        {'id': 1, 'center': (280, 370), 'state': 'attentive', 'skin': (180, 210, 240), 'label': 'Attentive'},
        # Student 2: Sleeping (head down, eyes closed)
        {'id': 2, 'center': (640, 410), 'state': 'sleeping', 'skin': (170, 200, 235), 'label': 'Sleeping'},
        # Student 3: Looking outside window (turned sideways)
        {'id': 3, 'center': (1000, 370), 'state': 'looking_outside', 'skin': (190, 220, 245), 'label': 'Looking Window'},
    ]

    for s in students:
        cx, cy = s['center']
        skin_color = s['skin']
        state = s['state']

        if state == 'attentive':
            # Attentive face: straight, eyes wide open
            cv2.ellipse(frame, (cx, cy), (85, 115), 0, 0, 360, skin_color, -1)
            cv2.ellipse(frame, (cx, cy), (85, 115), 0, 0, 360, (110, 130, 150), 3)
            # Hair
            cv2.ellipse(frame, (cx, cy - 45), (87, 65), 0, 180, 360, (40, 30, 25), -1)
            # Open eyes
            eye_y = cy - 15
            cv2.ellipse(frame, (cx - 32, eye_y), (20, 12), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (cx - 32, eye_y), 7, (30, 20, 15), -1)
            cv2.ellipse(frame, (cx + 32, eye_y), (20, 12), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (cx + 32, eye_y), 7, (30, 20, 15), -1)
            # Eyebrows
            cv2.line(frame, (cx - 48, eye_y - 16), (cx - 16, eye_y - 16), (50, 40, 30), 3)
            cv2.line(frame, (cx + 16, eye_y - 16), (cx + 48, eye_y - 16), (50, 40, 30), 3)
            # Nose
            cv2.line(frame, (cx, eye_y), (cx, cy + 22), (130, 150, 180), 3)
            # Normal mouth
            cv2.ellipse(frame, (cx, cy + 52), (24, 6), 0, 0, 360, (160, 60, 60), -1)
            cv2.putText(frame, "Student #1 (Attentive)", (cx - 75, cy + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 235, 100), 2)

        elif state == 'sleeping':
            # Sleeping face: head down, tilted
            cv2.ellipse(frame, (cx, cy + 20), (90, 105), 15, 0, 360, skin_color, -1)
            cv2.ellipse(frame, (cx, cy + 20), (90, 105), 15, 0, 360, (110, 130, 150), 3)
            # Hair cap
            cv2.ellipse(frame, (cx - 10, cy - 30), (92, 60), 15, 180, 360, (40, 30, 25), -1)
            # Closed eyes (horizontal arc lines / EAR = 0)
            eye_y = cy + 10
            cv2.ellipse(frame, (cx - 35, eye_y), (18, 4), 10, 0, 180, (40, 30, 20), 3)
            cv2.ellipse(frame, (cx + 25, eye_y + 5), (18, 4), 10, 0, 180, (40, 30, 20), 3)
            # Eyebrows slightly low
            cv2.line(frame, (cx - 50, eye_y - 12), (cx - 20, eye_y - 8), (50, 40, 30), 3)
            cv2.line(frame, (cx + 10, eye_y - 8), (cx + 40, eye_y - 4), (50, 40, 30), 3)
            # Nose downward
            cv2.line(frame, (cx - 5, eye_y + 5), (cx - 10, cy + 40), (130, 150, 180), 3)
            # Yawn or closed sleeping mouth
            cv2.ellipse(frame, (cx - 10, cy + 60), (20, 12), 10, 0, 360, (140, 50, 50), -1)
            cv2.putText(frame, "Student #2 (Sleeping)", (cx - 75, cy + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 245), 2)

        elif state == 'looking_outside':
            # Turned head looking outside window (yaw = 40 deg right)
            shift_x = 42
            cv2.ellipse(frame, (cx, cy), (85, 115), 0, 0, 360, skin_color, -1)
            cv2.ellipse(frame, (cx, cy), (85, 115), 0, 0, 360, (110, 130, 150), 3)
            # Hair shifted
            cv2.ellipse(frame, (cx - 10, cy - 45), (87, 65), 0, 180, 360, (40, 30, 25), -1)
            # Eyes shifted far right
            eye_y = cy - 15
            cv2.ellipse(frame, (cx - 10, eye_y), (16, 11), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (cx - 5, eye_y), 6, (30, 20, 15), -1)
            cv2.ellipse(frame, (cx + 45, eye_y), (18, 11), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (cx + 50, eye_y), 6, (30, 20, 15), -1)
            # Eyebrows shifted
            cv2.line(frame, (cx - 25, eye_y - 15), (cx + 5, eye_y - 15), (50, 40, 30), 3)
            cv2.line(frame, (cx + 30, eye_y - 15), (cx + 60, eye_y - 15), (50, 40, 30), 3)
            # Nose far right
            cv2.line(frame, (cx + 20, eye_y), (cx + shift_x, cy + 22), (130, 150, 180), 3)
            # Mouth shifted
            cv2.ellipse(frame, (cx + 25, cy + 52), (20, 5), 0, 0, 360, (160, 60, 60), -1)
            cv2.putText(frame, "Student #3 (Looking Window)", (cx - 95, cy + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 245), 2)

    cv2.imwrite(output_path, frame)
    print(f"3-Student demo classroom image successfully saved to {output_path}")
    return output_path


if __name__ == '__main__':
    generate_sample_video()
    generate_sample_image(force=True)

