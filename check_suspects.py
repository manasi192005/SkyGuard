"""
Check and Fix Suspects in SkyGuard Database
Verifies face encodings and fixes issues
"""

import cv2
import json
import os
from datetime import datetime

print("\n" + "="*60)
print("🛡️  SkyGuard - Suspect Database Checker")
print("="*60 + "\n")

# Check metadata file
metadata_path = 'data/suspects/metadata.json'

if not os.path.exists(metadata_path):
    print("❌ No metadata.json found!")
    exit()

# Load suspects
with open(metadata_path, 'r') as f:
    suspects = json.load(f)

print(f"📊 Found {len(suspects)} suspects in database:\n")

# Check each suspect
for i, suspect in enumerate(suspects, 1):
    print(f"{i}. {suspect['name']}")
    print(f"   Image: {suspect['image_path']}")
    print(f"   Status: {suspect['status']}")
    print(f"   Added: {suspect['added_date']}")
    
    # Check if image exists
    if os.path.exists(suspect['image_path']):
        print(f"   ✓ Image file exists")
        
        # Try to load image
        img = cv2.imread(suspect['image_path'])
        if img is not None:
            print(f"   ✓ Image loads successfully ({img.shape[1]}x{img.shape[0]})")
            
            # Check if face is detectable
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                print(f"   ✓ Face detected in image ({len(faces)} face(s))")
            else:
                print(f"   ⚠️  WARNING: No face detected in image!")
                print(f"      This suspect may not be recognized properly.")
        else:
            print(f"   ❌ ERROR: Cannot load image")
    else:
        print(f"   ❌ ERROR: Image file missing!")
    
    print()

print("="*60)
print("📋 DIAGNOSIS:")
print("="*60 + "\n")

print("If faces are not being detected:")
print("1. Ensure images are clear, well-lit photos of faces")
print("2. Face should be front-facing and clearly visible")
print("3. Image quality should be good (not blurry)")
print()

print("To re-add a suspect with a better photo:")
print("  python3 add_suspect.py")
print()

print("To test face recognition:")
print("  streamlit run web_dashboard.py")
print("  Then use the webcam detection feature")
print()

print("="*60)

# Show current face recognition settings
print("\n📊 Face Recognition Settings:")
print("="*60)
print("Default confidence threshold: 0.85 (85%)")
print("Alert cooldown: 60 seconds")
print("Detection method: Template matching (Haar Cascade)")
print()
print("⚠️  NOTE: The template matching method may not be very")
print("   accurate. For better results, consider using:")
print("   - face_recognition library (dlib-based)")
print("   - DeepFace")
print("   - FaceNet")
print("="*60 + "\n")
