"""
Test Gait Recognition from Video
"""

import cv2
from models.gait_recognition_video import GaitRecognitionVideo

print("\n" + "="*70)
print("🚶 Testing Gait Recognition from Video")
print("="*70 + "\n")

gait_rec = GaitRecognitionVideo()

print(f"✓ Gait profiles loaded: {len(gait_rec.gait_profiles)}")
for name, profile in gait_rec.gait_profiles.items():
    print(f"   • {name} ({profile['video_frames']} frames)")

print("\n📹 Opening camera...")
print("Walk in front of camera to test recognition")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)
frame_number = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        output, matches, tracking = gait_rec.process_frame(frame, frame_number, draw_visualization=True)
        
        # Show matches
        for match in matches:
            print(f"\n🚨 GAIT MATCH: {match['name']} ({match['confidence']:.0%})")
            print(f"   Person ID: {match['person_id']}")
            print(f"   Frames analyzed: {match['frames_analyzed']}")
        
        cv2.imshow('Gait Recognition Test', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏸ Stopped")

cap.release()
cv2.destroyAllWindows()
