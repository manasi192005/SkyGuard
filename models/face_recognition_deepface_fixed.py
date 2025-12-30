"""
FIXED: ULTRA HIGH-ACCURACY Face Recognition using DeepFace
Maximum accuracy with proper database search and debugging
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageEnhance

# ----------------------------
# DeepFace Import
# ----------------------------
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    raise ImportError("Install dependencies: pip install deepface tf-keras")


# ----------------------------
# ADMIN ALERT SYSTEM
# ----------------------------
class AdminAlertSystem:
    def __init__(self, config_path="data/admin_config.json"):
        self.config_path = config_path
        self.load_admin_config()

    def load_admin_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {"admins": []}
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

    def send_alert(self, suspect_name, latitude, longitude, confidence, frame_id=0):
        print("\n" + "=" * 70)
        print("🚨 SUSPECT DETECTED")
        print("=" * 70)
        print(f"Name       : {suspect_name}")
        print(f"Confidence : {confidence:.1%}")
        print(f"Location   : ({latitude}, {longitude})")
        print(f"Frame      : {frame_id}")
        print(f"Time       : {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70 + "\n")
        return True


# ----------------------------
# FACE RECOGNITION CLASS
# ----------------------------
class FaceRecognitionDeepFaceFixed:
    def __init__(
        self,
        suspects_db_path="data/suspects",
        confidence_threshold=0.50,
        debug_mode=True,
    ):
        self.suspects_db_path = suspects_db_path
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode

        self.model_name = "Facenet512"
        self.detector_backend = "opencv"
        self.distance_metric = "cosine"

        self.alert_system = AdminAlertSystem()
        self.recent_detections = {}
        self.alert_cooldown = 3

        self.total_frames = 0
        self.faces_detected = 0
        self.matches_found = 0
        self.match_attempts = []

        os.makedirs(self.suspects_db_path, exist_ok=True)

        print("\n" + "=" * 70)
        print("🚀 DEEPFACE ULTRA – FIXED VERSION")
        print("=" * 70)
        print(f"Confidence threshold: {confidence_threshold:.0%}")
        print(f"Debug mode          : {'ON' if debug_mode else 'OFF'}")
        print("=" * 70)

        self.load_suspects_database()
        self.prepare_database_structure()

    # ----------------------------
    # LOAD SUSPECTS
    # ----------------------------
    def load_suspects_database(self):
        self.suspects = []
        metadata = os.path.join(self.suspects_db_path, "metadata.json")

        if os.path.exists(metadata):
            with open(metadata, "r") as f:
                self.suspects = json.load(f)

        if self.debug_mode:
            print("\n📋 Loaded suspects:")
            for s in self.suspects:
                print(f" • {s['name']} → {s['image_path']}")

    # ----------------------------
    # DATABASE STRUCTURE
    # ----------------------------
    def prepare_database_structure(self):
        self.db_path = os.path.join(self.suspects_db_path, "deepface_db")
        os.makedirs(self.db_path, exist_ok=True)

        print("\n🔄 Preparing DeepFace database...\n")

        for suspect in self.suspects:
            name = suspect["name"]
            img_path = suspect["image_path"]

            if not os.path.exists(img_path):
                print(f"⚠️ Missing image: {img_path}")
                continue

            person_dir = os.path.join(self.db_path, name)
            os.makedirs(person_dir, exist_ok=True)

            img = cv2.imread(img_path)
            enhanced = self.enhance_face_mild(img)

            dest = os.path.join(person_dir, f"{name}.jpg")
            cv2.imwrite(dest, enhanced)

            print(f"✓ {name} added")

        print(f"\n✅ Database ready at {self.db_path}\n")

    # ----------------------------
    # IMAGE ENHANCEMENT
    # ----------------------------
    def enhance_face_mild(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image)

        pil = ImageEnhance.Contrast(pil).enhance(1.2)
        pil = ImageEnhance.Sharpness(pil).enhance(1.3)
        pil = ImageEnhance.Brightness(pil).enhance(1.05)

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ----------------------------
    # FACE DETECTION
    # ----------------------------
    def detect_faces(self, frame):
        try:
            faces = []
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
            )

            for d in detections:
                if d["confidence"] > 0.7:
                    fa = d["facial_area"]
                    faces.append(
                        {
                            "bbox": (fa["x"], fa["y"], fa["w"], fa["h"]),
                            "face": d["face"],
                        }
                    )
            return faces
        except Exception as e:
            if self.debug_mode:
                print("⚠️ Face detection error:", e)
            return []

    # ----------------------------
    # FACE RECOGNITION
    # ----------------------------
    def recognize_face(self, face_img, frame_id=0):
        try:
            temp = f"/tmp/query_{frame_id}.jpg"
            cv2.imwrite(temp, face_img)

            dfs = DeepFace.find(
                img_path=temp,
                db_path=self.db_path,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True,
            )

            os.remove(temp)

            if dfs and len(dfs[0]) > 0:
                best = dfs[0].iloc[0]
                distance = best["distance"]
                name = best["identity"].split(os.sep)[-2]

                similarity = 1 - distance
                similarity = np.clip(similarity, 0, 1)

                self.match_attempts.append(
                    {
                        "frame": frame_id,
                        "suspect": name,
                        "similarity": similarity,
                        "matched": similarity >= self.confidence_threshold,
                    }
                )

                if similarity >= self.confidence_threshold:
                    return {
                        "name": name,
                        "confidence": similarity,
                        "distance": distance,
                    }
            return None
        except Exception as e:
            if self.debug_mode:
                print("⚠️ Recognition error:", e)
            return None

    # ----------------------------
    # FRAME PROCESSING
    # ----------------------------
    def process_frame(self, frame, latitude=0, longitude=0, draw_boxes=True, frame_id=0):
        self.total_frames += 1
        output = frame.copy()
        faces = self.detect_faces(frame)
        self.faces_detected += len(faces)

        recognized = []

        for face in faces:
            x, y, w, h = face["bbox"]
            match = self.recognize_face(face["face"], frame_id)

            if match:
                self.matches_found += 1
                recognized.append(match)

                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(
                    output,
                    f"{match['name']} ({match['confidence']:.1%})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return output, faces, recognized

    # ----------------------------
    # STATS
    # ----------------------------
    def get_stats(self):
        return {
            "total_frames": self.total_frames,
            "faces_detected": self.faces_detected,
            "matches_found": self.matches_found,
        }


# ----------------------------
# VIDEO PROCESSING
# ----------------------------
def process_video_with_recognition(video_path, recognizer, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            int(cap.get(cv2.CAP_PROP_FPS)),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        output, _, _ = recognizer.process_frame(frame, frame_id=frame_id)

        if writer:
            writer.write(output)

    cap.release()
    if writer:
        writer.release()

    print("\n📊 FINAL STATS:", recognizer.get_stats())


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    print("🚀 DeepFace Ultra – Fixed Version Ready")
