"""
SkyGuard - Real-Time Gait Recognition
Matches live video against saved gait profiles
"""

import cv2
import mediapipe as mp
import numpy as np
from models.gait_recognition_emergency import GaitRecognitionEmergency
from datetime import datetime


def main():
    print("\n" + "=" * 70)
    print("🚶 SkyGuard - Real-Time Gait Recognition")
    print("=" * 70)
    
    # Initialize gait engine
    gait_engine = GaitRecognitionEmergency()
    
    if len(gait_engine.profiles) == 0:
        print("\n⚠️  No gait profiles found!")
        print("   Please capture profiles first: python3 capture_gait_profile.py")
        return
    
    print(f"\n✅ Ready to recognize {len(gait_engine.profiles)} profile(s):")
    for name in gait_engine.profiles.keys():
        print(f"   • {name}")
    
    input("\nPress ENTER to start recognition...")

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to access camera")
        return

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Tracking
    frame_buffer = []
    walk_count = 0
    last_ankle_y = None
    frame_count = 0
    
    min_frames_for_match = 60  # Need 60 frames (2 seconds) for matching

    print("\n📹 Camera active - Walk in front of camera")
    print("Press 'q' to quit\n")

    # Recognition loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        matched_name = None
        confidence = 0

        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            lm = results.pose_landmarks.landmark

            # Extract key joints
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            lk = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Calculate gait features
            stride_length = abs(la.x - ra.x)
            hip_sway = abs(lh.x - rh.x)
            knee_lift = abs(lk.y - lh.y)

            gait_features = {
                "stride_length": stride_length,
                "hip_sway": hip_sway,
                "knee_lift": knee_lift
            }

            frame_buffer.append(gait_features)

            # Detect walk (simple step detection)
            current_ankle_y = (la.y + ra.y) / 2
            if last_ankle_y is not None:
                if abs(current_ankle_y - last_ankle_y) > 0.05:
                    walk_count += 1
            last_ankle_y = current_ankle_y

            # Try matching if enough frames collected
            if len(frame_buffer) >= min_frames_for_match:
                # Generate signature from current buffer
                current_signature = {}
                for key in frame_buffer[0]:
                    values = [f[key] for f in frame_buffer if key in f]
                    if values:
                        current_signature[f"{key}_mean"] = float(np.mean(values))
                        current_signature[f"{key}_std"] = float(np.std(values))
                        current_signature[f"{key}_min"] = float(np.min(values))
                        current_signature[f"{key}_max"] = float(np.max(values))

                # Match against profiles
                matched_name, confidence = gait_engine.match_gait(
                    current_signature,
                    threshold=0.70  # 70% confidence required
                )

                # If match found, send alert
                if matched_name and confidence >= 0.70:
                    if gait_engine.should_alert(matched_name):
                        print(f"\n{'='*70}")
                        print(f"🚨 GAIT MATCH DETECTED!")
                        print(f"{'='*70}")
                        print(f"Suspect: {matched_name}")
                        print(f"Confidence: {confidence:.1%}")
                        print(f"Frames analyzed: {len(frame_buffer)}")
                        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                        print(f"{'='*70}\n")
                        
                        gait_engine.log_match(matched_name, confidence)

                # Keep buffer size manageable
                if len(frame_buffer) > 120:
                    frame_buffer = frame_buffer[-60:]

        # UI Overlay
        cv2.putText(frame, f"Frames: {len(frame_buffer)}/{min_frames_for_match}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Walk count: {walk_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show match status
        if matched_name and confidence >= 0.70:
            # RED for match
            cv2.rectangle(frame, (10, 90), (500, 180), (0, 0, 255), -1)
            cv2.putText(frame, f"MATCH: {matched_name}", 
                       (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.0%}", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif len(frame_buffer) >= min_frames_for_match:
            # YELLOW for analyzing
            cv2.putText(frame, "ANALYZING...", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # BLUE for collecting
            cv2.putText(frame, "COLLECTING DATA...", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("SkyGuard Gait Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Recognition session ended")


if __name__ == "__main__":
    main()
