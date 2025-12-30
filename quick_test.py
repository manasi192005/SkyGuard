"""
QUICK TEST - Run this first to verify everything works
"""

import cv2
import os
import sys

print("="*70)
print("⚡ QUICK VERIFICATION TEST")
print("="*70)

# Step 1: Check file structure
print("\n1️⃣  Checking file structure...")

required_paths = {
    'suspects_folder': 'data/suspects',
    'suspects_metadata': 'data/suspects/metadata.json',
    'models_folder': 'models',
    'face_recognition_file': 'models/face_recognition_deepface_fixed.py'
}

all_ok = True
for name, path in required_paths.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {name}: {path}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n❌ Missing required files/folders!")
    print("\n📝 Setup instructions:")
    print("   1. Create data/suspects folder")
    print("   2. Add suspect images to data/suspects/")
    print("   3. Create metadata.json (or use add_suspect)")
    print("   4. Place face_recognition_deepface_fixed.py in models/")
    sys.exit(1)

# Step 2: Check DeepFace installation
print("\n2️⃣  Checking DeepFace installation...")
try:
    from deepface import DeepFace
    print("   ✅ DeepFace installed")
except ImportError:
    print("   ❌ DeepFace not installed")
    print("\n📦 Install with: pip install deepface tf-keras")
    sys.exit(1)

# Step 3: Load recognizer
print("\n3️⃣  Loading face recognizer...")
try:
    from models.face_recognition_deepface_fixed import FaceRecognitionDeepFaceFixed
    
    recognizer = FaceRecognitionDeepFaceFixed(
        suspects_db_path='data/suspects',
        confidence_threshold=0.50,
        debug_mode=True
    )
    print("   ✅ Recognizer loaded successfully")
except Exception as e:
    print(f"   ❌ Failed to load recognizer: {e}")
    sys.exit(1)

# Step 4: Check suspects
print("\n4️⃣  Checking suspect database...")
if len(recognizer.suspects) == 0:
    print("   ⚠️  No suspects in database!")
    print("\n📝 To add suspects:")
    print("   1. Place suspect photos in data/suspects/")
    print("   2. Name them: suspect_name.jpg")
    print("   3. Run add_suspect script")
else:
    print(f"   ✅ Found {len(recognizer.suspects)} suspect(s):")
    for suspect in recognizer.suspects:
        img_exists = os.path.exists(suspect['image_path'])
        status = "✅" if img_exists else "❌"
        print(f"      {status} {suspect['name']}: {suspect['image_path']}")

# Step 5: Test with sample
print("\n5️⃣  Testing recognition (if test image available)...")

test_image_paths = [
    'data/test_image.jpg',
    'data/suspects/test.jpg',
    'test.jpg'
]

test_image = None
for path in test_image_paths:
    if os.path.exists(path):
        test_image = path
        break

if test_image:
    print(f"   Using test image: {test_image}")
    
    frame = cv2.imread(test_image)
    if frame is not None:
        print("   Processing...")
        
        output, faces, matches = recognizer.process_frame(
            frame,
            draw_boxes=True,
            frame_id=1
        )
        
        print(f"\n   📊 Results:")
        print(f"      Faces detected: {len(faces)}")
        print(f"      Matches found: {len(matches)}")
        
        if matches:
            for match in matches:
                print(f"\n      🎯 Match:")
                print(f"         Name: {match['name']}")
                print(f"         Confidence: {match['confidence']:.1%}")
                print(f"         Distance: {match['distance']:.4f}")
        
        # Save result
        output_path = 'quick_test_result.jpg'
        cv2.imwrite(output_path, output)
        print(f"\n   💾 Saved result: {output_path}")
        
        # Display
        cv2.imshow('Quick Test Result', output)
        print(f"\n   👁️  Showing result (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("   ❌ Failed to load test image")
else:
    print("   ⚠️  No test image found")
    print("   Create test.jpg or data/test_image.jpg to test")

# Step 6: Summary
print("\n" + "="*70)
print("✅ VERIFICATION COMPLETE")
print("="*70)
print("\n📝 Next steps:")
print("   1. If suspects are missing, add them using add_suspect.py")
print("   2. Run full test: python test_deepface_recognition.py")
print("   3. Test with your video in the dashboard")
print("\n💡 Tips:")
print("   - Start with threshold 0.40-0.50 for testing")
print("   - Use clear, front-facing suspect photos")
print("   - Ensure good lighting in both suspect photo and test video")
print("="*70)