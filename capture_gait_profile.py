"""
SkyGuard – Real Gait Profile Capture (MediaPipe Based)
"""

import cv2
import mediapipe as mp
import numpy as np
from models.gait_recognition_emergency import GaitRecognitionEmergency


def main():
    print("\n" + "=" * 70)
    print("🚶 SkyGuard - Gait Profile Capture System")
    print("=" * 70)
    input("\nPress ENTER to start capture...")

    # ---------------- Camera Initialization ----------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Unable to access camera")
        return

    print("📷 Camera initialized")

    # ---------------- MediaPipe Pose ----------------
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---------------- Gait Engine ----------------
    gait_engine = GaitRecognitionEmergency()

    frame_count = 0

    # ---------------- Capture Loop ----------------
    print("\n📹 Camera active - Start walking!")
    print("   Press 's' to SAVE when ready")
    print("   Press 'q' to QUIT\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            lm = results.pose_landmarks.landmark

            # Key joints
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            lk = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Simple gait features
            stride_length = abs(la.x - ra.x)
            hip_sway = abs(lh.x - rh.x)
            knee_lift = abs(lk.y - lh.y)

            gait_features = {
                "stride_length": stride_length,
                "hip_sway": hip_sway,
                "knee_lift": knee_lift
            }

            gait_engine.add_frame(gait_features)

            # Approximate walk count
            if frame_count % 40 == 0:
                gait_engine.increment_walk()

        quality = gait_engine.estimate_quality()
        status = "READY TO SAVE" if gait_engine.ready_to_save() else "CAPTURING"

        # ---------------- UI Overlay ----------------
        cv2.putText(frame, f"Frames: {len(gait_engine.frame_buffer)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Walks: {gait_engine.walk_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Quality: {quality}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, status,
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if status == "READY TO SAVE" else (0, 165, 255), 2)

        cv2.imshow("SkyGuard Gait Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n❌ Capture cancelled")
            break
        if key == ord('s') and gait_engine.ready_to_save():
            name = input("\n👤 Enter suspect/profile name: ")
            if name.strip():
                desc = input("📝 Description (optional): ")
                success = gait_engine.save_profile(name.strip(), desc.strip())
                if success:
                    print(f"\n✅ SUCCESS! Profile '{name}' saved to gait_profiles/")
                    print(f"   File: gait_profiles/{name}.json")
                break
            else:
                print("\n⚠️  Name cannot be empty")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Capture finished")


if __name__ == "__main__":
    main()
