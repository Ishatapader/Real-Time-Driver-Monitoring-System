import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import csv
import os
import platform
import json
from datetime import datetime
from collections import deque

class DrowsinessDetector:
    def __init__(self, config_path="config.json"):

        #  LOAD CONFIGURATION
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except FileNotFoundError:
            print(f"[ERROR] Configuration file '{config_path}' missing!")
            exit(1)

        # Dynamically pulling values from config.json
        self.EAR_THRESH = config['thresholds']['ear_drowsiness_limit']
        self.EAR_CONSEC_FRAMES = config['thresholds']['ear_consec_frames']

        self.MAR_THRESH = config['thresholds']['mar_yawn_limit']
        self.MAR_CONSEC_FRAMES = config['thresholds']['mar_consec_frames']

        self.PITCH_DELTA_THRESH = config['thresholds']['pitch_nod_limit']
        self.PITCH_CONSEC_FRAMES = config['thresholds']['pitch_consec_frames']

        self.ALARM_COOLDOWN_FRAMES = config['system']['alarm_cooldown_frames']
        self.CALIBRATION_FRAMES = config['system']['calibration_frames']
        self.CSV_FILE = config['system']['csv_log_file']
        
        # State Tracking Variables
        self.calibration_pitches = []
        self.pitch_baseline = None
        self.ear_counter = 0
        self.mar_counter = 0
        self.pitch_counter = 0
        self.cooldown_counter = 0
        
        # Audio & Threading Variables
        self.alarm_on = False
        self.is_muted = False
        self.alarm_lock = threading.Lock()
        self.audio_queue = queue.Queue()
        
        # Start the background audio worker
        threading.Thread(target=self._audio_worker, daemon=True).start()
        
        self.PITCH_SMOOTH_N = 5
        self.pitch_buffer = deque(maxlen=self.PITCH_SMOOTH_N)
        
        # Initialize the CSV File
        self._init_csv()
        
       
        #  LANDMARK INDICES & 3D MODEL

        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH     = [78, 81, 13, 311, 308, 402, 14, 178]
        self.POSE_LANDMARKS = [1, 152, 33, 263, 61, 291] 
        
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0,-150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float32)
        
    def _init_csv(self):
        if not os.path.exists(self.CSV_FILE):
            with open(self.CSV_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Warning_Type", "EAR", "MAR", "Pitch_Delta"])

    def log_telemetry(self, warning_type, current_ear, current_mar, current_pitch_delta):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, warning_type, round(current_ear, 3), round(current_mar, 3), round(current_pitch_delta, 3)])
        print(f"[LOG] {warning_type} recorded at {timestamp}")


    # AUDIO ENGINE
 
    def _audio_worker(self):
        """A completely crash-proof audio player using native OS sounds."""
        while True:
            message = self.audio_queue.get()
            if message is None: 
                break
            
            try:
                sys_os = platform.system()
                if sys_os == "Windows":
                    import winsound
                    winsound.Beep(1000, 1500) 
                elif sys_os == "Darwin": # macOS
                    os.system('say "Wake up! Danger!"')
                else: # Linux
                    print('\a')
                    time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Audio failed: {e}")
            finally:
                with self.alarm_lock:
                    self.alarm_on = False

    def trigger_alarm(self, warning_type):
        if self.is_muted: 
            return
        
        with self.alarm_lock:
            if not self.alarm_on:
                self.alarm_on = True
                self.audio_queue.put("TRIGGER_ALARM")

  
    #  MATH & GEOMETRY FUNCTIONS
   
    @staticmethod
    def calculate_ear(eye_landmarks, landmarks_array):
        coords = np.array([landmarks_array[idx] for idx in eye_landmarks])
        p2_p6 = np.linalg.norm(coords[1] - coords[5])
        p3_p5 = np.linalg.norm(coords[2] - coords[4])
        p1_p4 = np.linalg.norm(coords[0] - coords[3])
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    @staticmethod
    def calculate_mar(mouth_landmarks, landmarks_array):
        coords = np.array([landmarks_array[idx] for idx in mouth_landmarks])
        v1 = np.linalg.norm(coords[1] - coords[7])
        v2 = np.linalg.norm(coords[2] - coords[6])
        v3 = np.linalg.norm(coords[3] - coords[5])
        h  = np.linalg.norm(coords[0] - coords[4])
        return (v1 + v2 + v3) / (3.0 * h)

    def estimate_head_pose(self, landmarks_array, img_size):
        image_points = np.array([landmarks_array[idx] for idx in self.POSE_LANDMARKS], dtype=np.float32)
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        return euler_angles[0, 0], euler_angles[1, 0], euler_angles[2, 0]


    #  MAIN PIPELINE

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("\n[SYSTEM] Telemetry Engine Started.")
        print("[CONTROLS] Press 'm' to mute/unmute audio.")
        print("[CONTROLS] Press 'q' to quit.")

        prev_time = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[WARNING] Failed to read frame.")
                    break

                frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                status_color = (0, 255, 0)
                warning_text = "SAFE"
                avg_ear, mar, pitch_raw, pitch_delta, pitch_smooth = 0, 0, 0, 0, 0

                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1

                # CALIBRATION OVERLAY 
                if self.pitch_baseline is None:
                    progress = len(self.calibration_pitches)
                    bar_w = int((progress / self.CALIBRATION_FRAMES) * (w - 40))
                    cv2.rectangle(frame, (20, h//2 - 40), (w - 20, h//2 + 40), (0, 0, 0), -1)
                    cv2.putText(frame, "Sit normally & look at screen...", (30, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Calibrating: {progress}/{self.CALIBRATION_FRAMES}", (30, h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(frame, (20, h//2 + 50), (20 + bar_w, h//2 + 65), (0, 255, 0), -1)
                    cv2.imshow("Driver Telemetry Dashboard", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"): break
                    
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            landmarks_array = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                            pitch_raw, _, _ = self.estimate_head_pose(landmarks_array, (h, w))
                            self.calibration_pitches.append(pitch_raw)
                            
                            if len(self.calibration_pitches) >= self.CALIBRATION_FRAMES:
                                self.pitch_baseline = float(np.median(self.calibration_pitches))
                                print(f"[INFO] Pitch baseline set to {self.pitch_baseline:.1f} degrees")
                    continue

                #  CORE DETECTION LOGIC 
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_array = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                        avg_ear = (self.calculate_ear(self.LEFT_EYE, landmarks_array) + self.calculate_ear(self.RIGHT_EYE, landmarks_array)) / 2.0
                        mar = self.calculate_mar(self.MOUTH, landmarks_array)
                        pitch_raw, _, _ = self.estimate_head_pose(landmarks_array, (h, w))

                        self.pitch_buffer.append(pitch_raw)
                        pitch_smooth = float(np.mean(self.pitch_buffer))
                        pitch_delta = pitch_smooth - self.pitch_baseline

                        eye_warn, yawn_warn, pitch_warn = False, False, False

                        # Metrics Counters
                        self.ear_counter = self.ear_counter + 1 if avg_ear < self.EAR_THRESH else max(0, self.ear_counter - 2)
                        self.mar_counter = self.mar_counter + 1 if mar > self.MAR_THRESH else max(0, self.mar_counter - 2)
                        self.pitch_counter = self.pitch_counter + 1 if pitch_delta < -self.PITCH_DELTA_THRESH else max(0, self.pitch_counter - 2)

                        if self.ear_counter >= self.EAR_CONSEC_FRAMES: eye_warn = True
                        if self.mar_counter >= self.MAR_CONSEC_FRAMES: yawn_warn = True
                        if self.pitch_counter >= self.PITCH_CONSEC_FRAMES: pitch_warn = True

                        # Priority Hierarchy
                        active_threat = None
                        if pitch_warn:
                            warning_text, status_color, active_threat = "WARNING: HEAD NOD", (0, 0, 255), "nod"
                        elif eye_warn:
                            warning_text, status_color, active_threat = "WARNING: DROWSINESS", (0, 0, 255), "eyes"
                        elif yawn_warn:
                            warning_text, status_color, active_threat = "WARNING: YAWNING", (0, 165, 255), "yawn"
                        else:
                            warning_text, status_color = "SAFE", (0, 255, 0)

                        # Alarm & Logging Trigger
                        if active_threat and self.cooldown_counter == 0:
                            self.trigger_alarm(active_threat)
                            self.log_telemetry(active_threat, avg_ear, mar, pitch_delta)
                            self.cooldown_counter = self.ALARM_COOLDOWN_FRAMES

                        # Draw Landmarks
                        for idx in self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH:
                            cv2.circle(frame, landmarks_array[idx], 1, status_color, -1)

                # DASHBOARD DISPLAY 
                if self.pitch_baseline is not None:
                    cv2.rectangle(frame, (10, 10), (370, 180), (0, 0, 0), -1)
                    cv2.putText(frame, warning_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}  (thresh {self.EAR_THRESH})", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    cv2.putText(frame, f"MAR: {mar:.2f}  (thresh {self.MAR_THRESH})", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    cv2.putText(frame, f"Pitch: {pitch_smooth:.1f}  base: {self.pitch_baseline:.1f}  delta: {pitch_delta:.1f}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    cv2.putText(frame, f"Nod thresh: delta < -{self.PITCH_DELTA_THRESH}", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
                    cv2.putText(frame, f"Ctrs  EAR:{self.ear_counter}  MAR:{self.mar_counter}  P:{self.pitch_counter}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

                cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if fps > 15 else (0, 0, 255), 2)

                cv2.imshow("Driver Telemetry Dashboard", frame)

                #  KEYBOARD CONTROLS 
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("m"):
                    self.is_muted = not self.is_muted
                    state = "MUTED" if self.is_muted else "ON"
                    print(f"[SYSTEM] Audio is now {state}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio_queue.put(None)
            print("[INFO] Shutdown complete.")

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()