import cv2
import time

def list_cameras():
    # We saw /dev/video0 to /dev/video3
    # Often 0/2 are video and 1/3 are metadata, or 0 is RGB and 2 is IR
    available_cameras = []
    
    print("Scanning for cameras...")
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"Camera {index}: Found - Resolution {w}x{h}")
                available_cameras.append(index)
                
                # Save a snapshot to identify it
                filename = f"camera_{index}_snapshot.jpg"
                cv2.imwrite(filename, frame)
                print(f"  Saved snapshot to {filename}")
            else:
                print(f"Camera {index}: Opened but failed to read frame")
            cap.release()
        else:
            pass # Index not available
            
    return available_cameras

if __name__ == "__main__":
    cams = list_cameras()
    print(f"Available camera indices: {cams}")
