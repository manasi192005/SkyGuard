"""
Capture Gait Profile from Video
Records walking pattern for suspect identification
"""

import cv2
from models.gait_recognition_video import GaitRecognitionVideo
import sys

print("\n" + "="*70)
print("�� Gait Profile Capture from Video")
print("="*70)

print("\n📋 Instructions:")
print("  1. Person should walk naturally in camera view")
print("  2. Walk back and forth for at least 10-15 seconds")
print("  3. System will track and analyze gait automatically")
print("  4. Press 's' to save profile when ready")
print("  5. Press 'q' to quit")
print("\n" + "="*70 + "\n")

gait_rec = GaitRecognitionVideo(min_sequence_length=60)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit(1)

print("📹 Camera opened. Start walking...\n")

frame_number = 0
target_person_id = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        # Process frame
        output, matches, tracking = gait_rec.process_frame(frame, frame_number, draw_visualization=True)
        
        # Auto-select first tracked person
        if target_person_id is None and tracking:
            target_person_id = tracking[0]['person_id']
            print(f"✓ Tracking person ID: {target_person_id}")
        
        # Show progress for target person
        if target_person_id is not None:
            target_info = next((t for t in tracking if t['person_id'] == target_person_id), None)
            
            if target_info:
                frames = target_info['frames_captured']
                cv2.putText(output, f"Target ID {target_person_id}: {frames} frames captured", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if frames >= 60:
                    cv2.putText(output, "READY TO SAVE - Press 's'", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Gait Capture', output)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save profile
            if target_person_id is not None:
                print("\n" + "="*70)
                print("💾 Saving Gait Profile")
                print("="*70 + "\n")
                
                name = input("Suspect Name: ").strip()
                if not name:
                    print("❌ Name required")
                    continue
                
                description = input("Description (optional): ").strip()
                
                success, message = gait_rec.save_gait_profile(target_person_id, name, description)
                
                if success:
                    print(f"\n✅ GAIT PROFILE SAVED!")
                    print(f"   Name: {name}")
                    print(f"   {message}")
                    print(f"   Profile ready for recognition")
                    break
                else:
                    print(f"\n❌ Failed to save: {message}")
            else:
                print("\n⚠ No person tracked yet. Keep walking!")
        
        elif key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏸ Interrupted")

cap.release()
cv2.destroyAllWindows()

print("\n✓ Capture session ended")
