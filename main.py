import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
import argparse
import subprocess
import threading
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration ---
CAMERA_INDEX = 2
DATA_DIR = "data"
BRIGHT_LEVEL = "100%"
DIM_LEVEL = "1%"
DIM_DELAY = 1.0

# --- MediaPipe Setup ---
# Using standard import after fixing dependency
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FaceSystem:
    def __init__(self):
        self.os_makers = os.makedirs(DATA_DIR, exist_ok=True)
        self.known_faces = []
        self.known_labels = []
        self.model = None
        self.load_data()
        
        self.mp_face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Dimmer State
        self.max_brightness = self.get_max_brightness()
        self.current_brightness = 100
        self.target_brightness = 100
        self.last_face_time = time.time()
        self.is_dimmed = False
        self.last_set_val = -1

    def get_max_brightness(self):
        try:
            res = subprocess.run(["brightnessctl", "m"], capture_output=True, text=True)
            return int(res.stdout.strip())
        except:
            return 255

    def set_brightness(self, pct):
        val = int((pct/100.0) * self.max_brightness)
        if val != self.last_set_val:
            try:
                subprocess.run(["brightnessctl", "s", str(val)], check=False, stdout=subprocess.DEVNULL)
                self.last_set_val = val
            except:
                pass

    def load_data(self):
        try:
            with open(os.path.join(DATA_DIR, "faces.pkl"), "rb") as f:
                self.known_faces = pickle.load(f)
            with open(os.path.join(DATA_DIR, "labels.pkl"), "rb") as f:
                self.known_labels = pickle.load(f)
            if len(self.known_faces) > 0:
                self.train_model()
        except:
            pass

    def save_data(self):
        with open(os.path.join(DATA_DIR, "faces.pkl"), "wb") as f:
            pickle.dump(self.known_faces, f)
        with open(os.path.join(DATA_DIR, "labels.pkl"), "wb") as f:
            pickle.dump(self.known_labels, f)

    def train_model(self):
        if len(self.known_faces) < 5:
            return False
        X = np.array([f.flatten() for f in self.known_faces])
        y = np.array(self.known_labels)
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X, y)
        return True

    def get_embedding(self, frame, landmarks):
        # Extract eye/nose region for recognition
        h, w, _ = frame.shape
        xs = [l.x for l in landmarks.landmark]
        ys = [l.y for l in landmarks.landmark]
        x_min, x_max = int(min(xs)*w), int(max(xs)*w)
        y_min, y_max = int(min(ys)*h), int(max(ys)*h)
        
        # Padding
        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        face = frame[y_min:y_max, x_min:x_max]
        if face.size == 0: return None
        
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.resize(face_gray, (64, 64))

    def get_head_pose(self, landmarks, img_w, img_h):
        # 3D Head Pose Estimation
        face_3d = []
        face_2d = []
        
        keypoints = [1, 152, 33, 263, 61, 291]
        
        for idx in keypoints:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
            
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            x = angles[0] * 360
            y = angles[1] * 360
            return x, y
        except:
            return 0, 0

    def is_looking_at_screen(self, landmarks, img_w, img_h):
        try:
            x, y = self.get_head_pose(landmarks, img_w, img_h)
            
            # Allow for wider range (calibration issue?)
            # Yaw (Y): Left/Right
            # Pitch (X): Up/Down
            
            looking_up_down = -20 < x < 20
            looking_left_right = -20 < y < 20
            
            return looking_up_down and looking_left_right, (x, y)
        except:
             return False, (0,0)

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        # Visualizing the head pose direction
        # pitch = X, yaw = Y
        
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        
        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
        y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
        y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

        # Z-Axis (out of screen) drawn in blue
        x3 = size * (np.sin(yaw)) + tdx
        y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
        return img

    def run_training(self, name):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # Buckets for angles: Center, Left, Right, Up, Down
        needed = {
            "Center": 10,
            "Left": 10,
            "Right": 10,
            "Up": 10,
            "Down": 10
        }
        
        print(f"Training for user: {name}")
        print("Follow the on-screen instructions to capture all angles.")
        
        last_capture = time.time()
        
        while any(v > 0 for v in needed.values()):
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face_mesh.process(rgb)
            
            status_text = "Looking for face..."
            
            if res.multi_face_landmarks:
                for face_landmarks in res.multi_face_landmarks:
                    # Draw Mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    x, y = self.get_head_pose(face_landmarks, w, h)
                    
                    # Determine current Pose
                    pose = "Unknown"
                    # Remove dead zones. Check Center first (stricter), then others.
                    if abs(x) < 10 and abs(y) < 10: 
                        pose = "Center"
                    elif y < -10: 
                        pose = "Right"
                    elif y > 10: 
                        pose = "Left"
                    elif x < -10: 
                        pose = "Down"
                    elif x > 10: 
                        pose = "Up"
                    
                    # Debug Info
                    cv2.putText(frame, f"Angles: X={x:.0f} Y={y:.0f}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(frame, f"Detected: {pose}", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    if pose in needed and needed[pose] > 0:
                        status_text = f"Capturing {pose}!"
                        if time.time() - last_capture > 0.2:
                            feat = self.get_embedding(frame, face_landmarks)
                            if feat is not None:
                                self.known_faces.append(feat)
                                self.known_labels.append(name)
                                needed[pose] -= 1
                                last_capture = time.time()
                    else:
                        # Find what is still needed
                        missing = [k for k, v in needed.items() if v > 0]
                        if len(missing) > 0:
                            status_text = f"Please Look: {missing[0]}"
                        else:
                            status_text = "Done!"

            # HUD
            y_off = 40
            cv2.putText(frame, f"Training: {name}", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_off += 30
            cv2.putText(frame, status_text, (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            y_off += 40
            for k, v in needed.items():
                col = (0, 255, 0) if v == 0 else (0, 0, 255)
                cv2.putText(frame, f"{k}: {10-v}/10", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
                y_off += 25
            
            cv2.imshow("Training (Press 'q' to abort)", frame)
            if cv2.waitKey(1) == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()
        self.save_data()
        self.train_model()
        print("Training complete.")

    def run_main(self, use_dimmer=False):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print("Starting Face ID System...")
        
        # Ensure bright
        self.set_brightness(100)
        
        # FPS Calculation
        prev_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_face_mesh.process(rgb)
                
                looking = False
                
                if res.multi_face_landmarks:
                    for face_landmarks in res.multi_face_landmarks:
                        # Attention Check
                        looking, angles = self.is_looking_at_screen(face_landmarks, w, h)
                        
                        # Draw Axis at nose tip
                        nose = face_landmarks.landmark[1]
                        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                        self.draw_axis(frame, angles[1], angles[0], 0, nose_x, nose_y, size=50)
                        
                        # Draw Mesh based on attention
                        color_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        if not looking:
                            # Red tint for mesh if not looking 
                            # (Note: styles are complex to mutate, we'll just use text/box)
                            pass
                            
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=color_spec)
                        
                        # Visuals
                        color = (0, 255, 0) if looking else (0, 0, 255)
                        status = "LOOKING" if looking else "AWAY"
                        cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(frame, f"Pose: X:{angles[0]:.0f} Y:{angles[1]:.0f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # Recognition
                        if looking and self.model:
                             feat = self.get_embedding(frame, face_landmarks)
                             if feat is not None:
                                 try:
                                     pred = self.model.predict([feat.flatten()])
                                     # Confidence? KNN doesn't give probability easily without predict_proba
                                     # Just show label
                                     cv2.putText(frame, f"Hello {pred[0]}", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                                 except: pass

                # Dimmer Logic
                if use_dimmer:
                    if looking:
                        self.last_face_time = time.time()
                        self.target_brightness = 100
                    else:
                        if time.time() - self.last_face_time > DIM_DELAY:
                            self.target_brightness = 1
                        else:
                            self.target_brightness = 100
                    
                    # Smooth Transition
                    if self.current_brightness != self.target_brightness:
                        step = 5 # Speed
                        if self.current_brightness > self.target_brightness: # Dimming
                            step = 2 # Slower dim
                            self.current_brightness = max(self.current_brightness - step, self.target_brightness)
                        else: # Brightening
                            self.current_brightness = min(self.current_brightness + step, self.target_brightness)
                        
                        self.set_brightness(self.current_brightness)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.imshow("FaceOS", frame)
                if cv2.waitKey(1) == ord('q'): break
                
        except KeyboardInterrupt:
            pass
        finally:
            self.set_brightness(100)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    train_parser = subparsers.add_parser('train', help='Train new user')
    train_parser.add_argument('--name', required=True, help='Name of the user')
    
    run_parser = subparsers.add_parser('run', help='Run detection')
    run_parser.add_argument('--dim', action='store_true', help='Enable screen dimming')
    
    args = parser.parse_args()
    
    sys = FaceSystem()
    if args.command == 'train':
        sys.run_training(args.name)
    elif args.command == 'run':
        sys.run_main(args.dim)
    else:
        parser.print_help()
