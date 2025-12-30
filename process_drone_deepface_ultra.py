import cv2
import os
from models.face_recognition_deepface_fixed import FaceRecognitionDeepFaceUltra

video_path = input("\n📂 Video path: ").strip('\'"')

if not os.path.exists(video_path):
    print(f"❌ Not found")
    exit()

print("\n🔄 Processing...")

face_rec = FaceRecognitionDeepFaceUltra(confidence_threshold=0.70)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("processed_videos", exist_ok=True)
out_path = f"processed_videos/ultra_{os.path.basename(video_path)}"

writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_id = 0
detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    if frame_id % 10 == 0:
        print(f"   {frame_id}/{total} ({frame_id/total*100:.0f}%)")
    
    output, faces, recognized = face_rec.process_frame(frame, frame_id=frame_id, draw_boxes=True)
    
    for match in recognized:
        detections.append({'frame': frame_id, 'name': match['name'], 'conf': match['confidence']})
    
    cv2.putText(output, f"Frame: {frame_id}/{total}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    writer.write(output)

cap.release()
writer.release()

stats = face_rec.get_stats()
print(f"\n✅ Complete!")
print(f"Faces: {stats['faces_detected']}")
print(f"Matches: {stats['matches_found']}")
print(f"Saved: {out_path}\n")
