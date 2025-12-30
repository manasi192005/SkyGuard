"""
Test Face Recognition System
"""
import cv2
from models.face_recognition_enhanced import FaceRecognitionEnhanced

print("\n" + "="*60)
print("🛡️  Testing Face Recognition System")
print("="*60 + "\n")

# Initialize system
face_rec = FaceRecognitionEnhanced()

print(f"✓ Loaded {len(face_rec.suspects)} suspects")
print(f"✓ Loaded {len(face_rec.suspect_encodings)} face encodings\n")

# List suspects
for suspect in face_rec.suspects:
    print(f"  - {suspect['name']}")

print("\n" + "="*60)
print("📹 Starting webcam test...")
print("Press 'q' to quit")
print("="*60 + "\n")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame, detections, suspects = face_rec.process_frame(
        frame, 
        latitude=19.0760, 
        longitude=72.8777,
        draw_boxes=True
    )
    
    # Show stats
    cv2.putText(processed_frame, f"Faces: {len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(processed_frame, f"Suspects: {len(suspects)}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow('SkyGuard - Face Recognition Test', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✓ Test complete")
