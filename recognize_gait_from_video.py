"""
SkyGuard – Video Gait Recognition (Stable, Hackathon-Ready)
Authoritative fixed version – NO syntax errors
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from models.gait_recognition_emergency import GaitRecognitionEmergency


# =========================================================
# VIDEO GAIT PROCESSOR
# =========================================================
class VideoGaitProcessor:

    def __init__(self, gait_engine, min_frames=60):
        self.gait_engine = gait_engine
        self.min_frames = min_frames

        self.mp_pose = mp.solutions.pose
        self.drawer = mp.solutions.drawing_utils

        print("⏳ Initializing MediaPipe Pose...")
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe ready")

        self.track_features = []

    # -----------------------------------------------------
    def extract_features(self, landmarks):
        lm = landmarks.landmark

        try:
            lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            lk = lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            rk = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            la = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ra = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            features = {
                "stride_length": abs(la.x - ra.x),
                "hip_sway": abs(lh.x - rh.x),
                "knee_lift_left": abs(lk.y - lh.y),
                "knee_lift_right": abs(rk.y - rh.y),
                "leg_extension_left": np.linalg.norm([la.x - lk.x, la.y - lk.y]),
                "leg_extension_right": np.linalg.norm([ra.x - rk.x, ra.y - rk.y]),
            }
            return features
        except:
            return None

    # -----------------------------------------------------
    def build_signature(self, features):
        signature = {}

        for key in features[0]:
            values = [f[key] for f in features if f and key in f]
            if len(values) < 10:
                continue

            arr = np.array(values)
            signature[f"{key}_mean"] = float(np.mean(arr))
            signature[f"{key}_std"] = float(np.std(arr))

        if "stride_length_mean" in signature:
            peaks, _ = find_peaks(
                np.array([f["stride_length"] for f in features]), distance=8
            )
            if len(peaks) > 2:
                signature["stride_frequency"] = float(len(peaks))

        return signature

    # -----------------------------------------------------
    def match_signature(self, signature, threshold):
        best_match = None
        best_score = 0

        for name, profile in self.gait_engine.profiles.items():
            prof_sig = profile["signature"]
            common = set(signature.keys()) & set(prof_sig.keys())

            if not common:
                continue

            scores = []
            for k in common:
                a = signature[k]
                b = prof_sig[k]
                score = 1 - abs(a - b) / (abs(a) + abs(b) + 1e-6)
                scores.append(score)

            overall = sum(scores) / len(scores)

            if overall > best_score:
                best_score = overall
                best_match = name

        if best_score >= threshold:
            return best_match, best_score

        return None, best_score

    # -----------------------------------------------------
    def process_video(self, video_path, output_path=None, threshold=0.65):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Cannot open video")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        frame_count = 0
        print("🔍 Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            if result.pose_landmarks:
                self.drawer.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                features = self.extract_features(result.pose_landmarks)
                if features:
                    self.track_features.append(features)

                if len(self.track_features) >= self.min_frames:
                    signature = self.build_signature(self.track_features)
                    if signature:
                        name, confidence = self.match_signature(signature, threshold)

                        if name:
                            cv2.putText(
                                frame,
                                f"SUSPECT: {name} ({confidence:.1%})",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                3
                            )
                            print(f"🚨 MATCH FOUND: {name} ({confidence:.1%})")

            if writer:
                writer.write(frame)

            cv2.imshow("SkyGuard – Gait Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print("✅ Processing complete")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 70)
    print("🎥 SkyGuard – Video Gait Recognition")
    print("=" * 70)

    gait_engine = GaitRecognitionEmergency()

    if not gait_engine.profiles:
        print("❌ No gait profiles found in gait_profiles/")
        return

    print(f"✅ Loaded {len(gait_engine.profiles)} gait profile(s):")
    for name in gait_engine.profiles:
        print(f"   • {name}")

    video_path = input("\n📂 Enter video path: ").strip()
    if not os.path.exists(video_path):
        print("❌ Video file not found")
        return

    save = input("💾 Save annotated video? (y/n): ").lower()
    output_path = None

    if save == "y":
        os.makedirs("processed_videos", exist_ok=True)
        output_path = "processed_videos/gait_result.mp4"

    threshold = 0.65
    processor = VideoGaitProcessor(gait_engine)
    processor.process_video(video_path, output_path, threshold)


if __name__ == "__main__":
    main()
