"""
Add Suspect to Database
Capture photo and add to face recognition system
"""

import cv2
from models.face_recognition_enhanced import FaceRecognitionEnhanced
import os

print("\n" + "="*60)
print("🛡️  SkyGuard - Add Suspect to Database")
print("="*60)

# Initialize face recognition system
face_rec = FaceRecognitionEnhanced()

print("\n📸 Instructions:")
print("  1. Position the suspect's face in the camera")
print("  2. Press SPACE to capture photo")
print("  3. Press Q to quit without capturing")
print("\n" + "="*60 + "\n")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open camera")
    exit()

captured = False
photo_path = None

print("📹 Camera opened. Position face and press SPACE...\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Error: Cannot read frame")
        break
    
    # Show instructions on frame
    cv2.putText(frame, "Press SPACE to capture, Q to quit", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Add Suspect - SkyGuard', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # SPACE pressed
        # Save the photo
        os.makedirs('data/suspects', exist_ok=True)
        timestamp = cv2.getTickCount()
        photo_path = f'data/suspects/temp_{timestamp}.jpg'
        cv2.imwrite(photo_path, frame)
        
        print("✅ Photo captured!")
        captured = True
        break
    
    elif key == ord('q'):  # Q pressed
        print("❌ Cancelled by user")
        break

cap.release()
cv2.destroyAllWindows()

if captured and photo_path:
    print("\n" + "="*60)
    print("📝 Enter Suspect Details")
    print("="*60 + "\n")
    
    # Get suspect details
    name = input("Suspect Name: ").strip()
    
    if not name:
        print("❌ Name is required!")
        os.remove(photo_path)
        exit()
    
    description = input("Description (optional): ").strip()
    uploaded_by = input("Your Name (uploader): ").strip() or "Admin"
    
    # Add to database
    try:
        face_rec.add_suspect(
            name=name,
            image_path=photo_path,
            description=description,
            uploaded_by=uploaded_by
        )
        
        print("\n" + "="*60)
        print("✅ SUSPECT ADDED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📋 Details:")
        print(f"   Name: {name}")
        print(f"   Description: {description or 'None'}")
        print(f"   Uploaded by: {uploaded_by}")
        print(f"   Photo: {photo_path}")
        print(f"\n✓ Suspect is now in the recognition database")
        print("✓ System will alert admins when detected")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error adding suspect: {e}")
        if os.path.exists(photo_path):
            os.remove(photo_path)

else:
    print("\n❌ No photo captured. Exiting.")
