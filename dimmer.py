import cv2
import time
import subprocess
import threading

# Configuration
CAMERA_INDEX = 2 # IR Camera preferred
DIM_DELAY = 1.5 # Seconds before dimming
TARGET_BRIGHT = 100 # Percent
TARGET_DIM = 0 # Percent (Minimum)
TRANSITION_STEP = 5 # Percent change per step
TRANSITION_DELAY = 0.05 # Seconds between steps

def get_max_brightness():
    try:
        res = subprocess.run(["brightnessctl", "m"], capture_output=True, text=True)
        return int(res.stdout.strip())
    except:
        return 255 # Fallback

MAX_BRIGHTNESS = get_max_brightness()
current_brightness_pct = TARGET_BRIGHT
last_set_val = -1

def set_brightness(percent):
    global last_set_val
    val = int((percent / 100.0) * MAX_BRIGHTNESS)
    
    # Avoid redundant calls to hardware (prevents flashing/lag)
    if val != last_set_val:
        try:
            subprocess.run(["brightnessctl", "s", str(val)], check=False, stdout=subprocess.DEVNULL)
            last_set_val = val
        except Exception as e:
            pass

def face_detector():
    # Setup Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    # Increase buffer to prevent detection flicker
    # Use a detection buffer: need N consecutive "no faces" to dim
    consecutive_no_face = 0
    REQUIRED_NO_FACE_FRAMES = 5 
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    last_seen_time = time.time()
    
    print(f"Dimmer Active. Max Brightness: {MAX_BRIGHTNESS}")
    set_brightness(TARGET_BRIGHT)
    
    global current_brightness_pct
    target_state = TARGET_BRIGHT
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                last_seen_time = time.time()
                consecutive_no_face = 0
                target_state = TARGET_BRIGHT
            else:
                consecutive_no_face += 1
                # Only dim if we haven't seen a face for a while AND we have consecutive misses
                if (time.time() - last_seen_time > DIM_DELAY) and (consecutive_no_face > REQUIRED_NO_FACE_FRAMES):
                    target_state = TARGET_DIM
                else:
                    target_state = TARGET_BRIGHT
            
            # Non-blocking smooth transition step
            if current_brightness_pct != target_state:
                if current_brightness_pct < target_state:
                    current_brightness_pct = min(current_brightness_pct + TRANSITION_STEP, target_state)
                else:
                    # Slow down dimming slightly for better effect
                    step = max(1, TRANSITION_STEP // 2)
                    current_brightness_pct = max(current_brightness_pct - step, target_state)
                
                set_brightness(current_brightness_pct)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nExiting... Restoring brightness.")
        set_brightness(100)
    finally:
        cap.release()

if __name__ == "__main__":
    face_detector()
