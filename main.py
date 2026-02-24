import cv2
import os
import pickle
import argparse
import yaml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import signal

# Load configuration
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

CAMERA_INDEX = config["camera_device"]
DATA_DIR = config["data_dir"]
NUM_SAMPLES = config["num_samples"]

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class FaceSystem:
    def __init__(self):
        self.known_faces = []
        self.known_labels = []
        self.model = None
        self.load_data()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_data(self):
        try:
            with open(os.path.join(DATA_DIR, "faces.pkl"), "rb") as f:
                self.known_faces = pickle.load(f)
            with open(os.path.join(DATA_DIR, "labels.pkl"), "rb") as f:
                self.known_labels = pickle.load(f)
            if self.known_faces:
                self.train_model()
        except FileNotFoundError:
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

    def get_embedding(self, face):
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.resize(face_gray, (64, 64))

    def train(self, name, camera_index):
        cap = cv2.VideoCapture(camera_index)
        print(f"Training for user: {name}")
        print("Look directly at the camera.")

        samples_collected = 0
        while samples_collected < NUM_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            for x, y, w, h in faces:
                face = frame[y : y + h, x : x + w]
                embedding = self.get_embedding(face)
                self.known_faces.append(embedding)
                self.known_labels.append(name)
                samples_collected += 1
                print(f"Samples collected: {samples_collected}/{NUM_SAMPLES}")
                if samples_collected >= NUM_SAMPLES:
                    break

            cv2.imshow("Training", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()
        self.train_model()
        print("Training complete.")

    def test(self, camera_index):
        if not self.model:
            print("No trained model available. Train first.")
            return

        cap = cv2.VideoCapture(camera_index)
        print("Starting face recognition...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            for x, y, w, h in faces:
                face = frame[y : y + h, x : x + w]
                embedding = self.get_embedding(face).flatten()
                label = self.model.predict([embedding])
                confidence = self.model.kneighbors([embedding], n_neighbors=1)[0][0][0]
                if confidence < 0.5:
                    label = "Unknown"
                print(f"Label: {label}, Confidence: {confidence:.2f}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    label[0],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def test_devices(self):
        config_device = CAMERA_INDEX  # Start with the camera from config

        print(f"Testing configured camera device {config_device}...")
        cap = cv2.VideoCapture(config_device)
        if cap.read()[0]:
            print(f"Displaying configured camera {config_device}.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow(f"Configured Camera {config_device}", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("n"):
                    break
                elif k == ord("q"):
                    print("Exiting test devices mode.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Configured camera device {config_device} is not available.")
            cap.release()

        print("Testing remaining available camera devices...")
        print("Testing available camera devices...")
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                print(f"Camera device {index} is not available.")
                cap.release()
                break

            print(f"Displaying camera {index}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow(f"Test Camera {index}", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("n"):
                    break
                elif k == ord("q"):
                    print("Exiting test devices mode.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()
            cv2.destroyAllWindows()
            index += 1


if __name__ == "__main__":

    def signal_handler(sig, frame):
        print("\nInterrupt received. Cleaning up...")
        cv2.destroyAllWindows()
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Index of the camera to use (default: IR camera)",
    )
    parser.add_argument("--name", type=str, help="Name for training")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "test_devices"],
        help="Mode to run: train, test, or test_devices",
    )
    args = parser.parse_args()

    face_system = FaceSystem()

    if args.mode == "train" and args.name:
        face_system.train(args.name, args.camera)
    elif args.mode == "test":
        face_system.test(args.camera)
    elif args.mode == "test_devices":
        face_system.test_devices()
    else:
        print("Invalid arguments. Use --help for usage details.")
