"""SkyGuard Quick Demo"""
import cv2
import sys

print("\n" + "="*60)
print("SkyGuard Demo - Testing System")
print("="*60)

# Test OpenCV
try:
    import cv2
    print("\n✓ OpenCV installed:", cv2.__version__)
except:
    print("\n✗ OpenCV not installed")

# Test other dependencies
deps = ['numpy', 'torch', 'deepface', 'mediapipe', 'fastapi', 'streamlit']
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep} installed")
    except:
        print(f"✗ {dep} not installed")

# Test database
try:
    from models.database import init_database
    engine = init_database()
    print("\n✓ Database initialized successfully!")
except Exception as e:
    print(f"\n✗ Database error: {e}")

# Test webcam
print("\nTesting webcam...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✓ Webcam working!")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    else:
        print("✗ Could not open webcam")
except Exception as e:
    print(f"✗ Webcam error: {e}")

print("\n" + "="*60)
print("Setup Status: Ready to build main modules!")
print("="*60 + "\n")
