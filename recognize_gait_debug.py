"""
Debug version - Shows why detections are rejected
"""

import cv2
import mediapipe as mp
import numpy as np
from models.gait_recognition_emergency import GaitRecognitionEmergency
from datetime import datetime
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Initialize
gait_engine = GaitRecognitionEmergency()

if len(gait_engine.profiles) == 0:
    print("⚠️  No profiles found!")
    exit()

print(f"✅ Loaded profiles: {list(gait_engine.profiles.keys())}")

# Video path
video_path = input("\n📂 Enter video path: ").strip('\'"')

if not os.path.exists(video_path):
    print(f"❌ File not found: {video_path}")
    exit()

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,  # Lowered
    min_tracking_confidence=0.5    # Lowered
)

cap = cv2.VideoCapture(video_path)

frame_number = 0
rejection_reasons = {}

print("\n🔍 Analyzing video...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_number += 1
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        # Check 1: Visibility
        key_landmarks = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
        visibilities = [lm[idx.value].visibility for idx in key_landmarks]
        min_vis = min(visibilities)
        avg_vis = np.mean(visibilities)
        
        if min_vis < 0.6:
            reason = f"Low visibility: min={min_vis:.2f}, avg={avg_vis:.2f}"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        
        # Check 2: Body size
        xs = [lm[i].x for i in range(len(lm))]
        ys = [lm[i].y for i in range(len(lm))]
        
        body_width = (max(xs) - min(xs))
        body_height = (max(ys) - min(ys))
        body_area = body_width * body_height
        
        if body_area < 0.15:
            reason = f"Body too small: {body_area:.3f}"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        
        if body_area > 0.8:
            reason = f"Body too large: {body_area:.3f}"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        
        # Check 3: Proportions
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        leg_length_ratio = abs(left_ankle.y - left_hip.y) / body_height
        
        if leg_length_ratio < 0.3 or leg_length_ratio > 0.7:
            reason = f"Invalid proportions: {leg_length_ratio:.2f}"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        
        # Check 4: Upright
        hip_ankle_angle = abs(left_hip.y - left_ankle.y)
        
        if hip_ankle_angle < 0.15:
            reason = f"Not upright: {hip_ankle_angle:.2f}"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        
        # Passed all checks!
        print(f"✅ Frame {frame_number}: PASSED all checks")
        print(f"   Visibility: min={min_vis:.2f}, avg={avg_vis:.2f}")
        print(f"   Body area: {body_area:.3f}")
        print(f"   Leg ratio: {leg_length_ratio:.2f}")
        print(f"   Upright: {hip_ankle_angle:.2f}\n")

cap.release()

print("\n" + "="*70)
print("📊 REJECTION ANALYSIS")
print("="*70)
print(f"Total frames: {frame_number}")
print(f"\nRejection reasons:")
for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"  • {reason}: {count} frames ({count/frame_number*100:.1f}%)")
print("="*70)
