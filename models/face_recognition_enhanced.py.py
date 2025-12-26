
"""
Enhanced Face Recognition for Suspect Detection in Crowds
"""

import cv2
import numpy as np
import os
from datetime import datetime
import json

class FaceRecognitionEnhanced:
    """Enhanced Face Recognition with Better Crowd Handling"""
    
    def __init__(self, suspects_db_path='data/suspects', confidence_threshold=0.85):
        """Initialize Enhanced Face Recognition System"""
        self.suspects_db_path = suspects_db_path
        self.confidence_threshold = confidence_threshold
        
        # Create suspects directory
        os.makedirs(suspects_db_path, exist_ok=True)
        
        # Initialize face detector (Haar Cascade - faster)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load suspects
        self.load_suspects_database()
        self.load_suspect_encodings()
        
        # Detection history (to avoid duplicate alerts)
        self.recent_detections = {}
        
    def load_suspects_database(self):
        """Load suspect metadata"""
        self.suspects = []
        metadata_path = os.path.join(self.suspects_db_path, 'metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.suspects = json.load(f)
        
        print(f"Loaded {len(self.suspects)} suspects from database")
    
    def load_suspect_encodings(self):
        """Pre-load and encode suspect faces for faster matching"""
        self.suspect_encodings = {}
        
        for suspect in self.suspects:
            try:
                img_path = suspect['image_path']
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Store the template image
                        self.suspect_encodings[suspect['name']] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error loading suspect {suspect['name']}: {e}")
    
    def add_suspect(self, name, image_path, description=''):
        """Add a new suspect to the database"""
        # Copy image to suspects folder
        import shutil
        dest_path = os.path.join(self.suspects_db_path, f"{name}.jpg")
        
        if os.path.exists(image_path):
            shutil.copy(image_path, dest_path)
        else:
            dest_path = image_path
        
        suspect_info = {
            'name': name,
            'image_path': dest_path,
            'description': description,
            'added_date': datetime.now().isoformat()
        }
        
        self.suspects.append(suspect_info)
        
        # Save metadata
        metadata_path = os.path.join(self.suspects_db_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.suspects, f, indent=2)
        
        # Load encoding
        img = cv2.imread(dest_path)
        if img is not None:
            self.suspect_encodings[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"✓ Added suspect: {name}")
    
    def detect_faces(self, frame):
        """Detect all faces in frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': 1.0
            })
        
        return detections
    
    def match_face(self, frame, face_bbox):
        """
        Match detected face against suspect database
        Uses template matching for speed
        """
        x, y, w, h = face_bbox
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        if face_region.size == 0:
            return None
        
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))
        
        # Compare with each suspect
        best_match = None
        best_score = 0
        
        for suspect_name, suspect_template in self.suspect_encodings.items():
            # Resize suspect template to match
            template_resized = cv2.resize(suspect_template, (100, 100))
            
            # Calculate similarity using template matching
            result = cv2.matchTemplate(face_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]
            
            # Normalize to 0-1 range
            similarity = (similarity + 1) / 2
            
            if similarity > best_score and similarity > self.confidence_threshold:
                best_score = similarity
                best_match = suspect_name
        
        if best_match:
            # Find full suspect info
            suspect_info = next((s for s in self.suspects if s['name'] == best_match), None)
            return {
                'name': best_match,
                'confidence': best_score,
                'description': suspect_info['description'] if suspect_info else '',
                'bbox': face_bbox
            }
        
        return None
    
    def process_frame(self, frame, draw_boxes=True):
        """
        Process frame: detect faces and identify suspects
        
        Returns:
            processed_frame, detections, recognized_suspects
        """
        detections = self.detect_faces(frame)
        recognized_suspects = []
        
        output_frame = frame.copy()
        current_time = datetime.now()
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Try to match face
            match = self.match_face(frame, bbox)
            
            if match:
                # Check if this is a recent detection (avoid spam)
                last_detection = self.recent_detections.get(match['name'])
                is_new_detection = (
                    last_detection is None or 
                    (current_time - last_detection).seconds > 5
                )
                
                if is_new_detection:
                    recognized_suspects.append(match)
                    self.recent_detections[match['name']] = current_time
                
                if draw_boxes:
                    # Draw RED box for suspects
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    
                    # Draw alert background
                    alert_h = 60
                    cv2.rectangle(output_frame, (x, y-alert_h), (x+w+200, y), (0, 0, 255), -1)
                    
                    # Suspect name
                    cv2.putText(output_frame, f"SUSPECT: {match['name']}", 
                               (x+5, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Confidence
                    cv2.putText(output_frame, f"Confidence: {match['confidence']:.0%}", 
                               (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Blinking alert
                    if int(current_time.timestamp() * 2) % 2 == 0:
                        cv2.putText(output_frame, "!!! ALERT !!!", 
                                   (x+w+10, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                if draw_boxes:
                    # Draw GREEN box for unknown faces
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return output_frame, detections, recognized_suspects


# Test module
if __name__ == '__main__':
    print("Enhanced Face Recognition Module - Test")
    face_rec = FaceRecognitionEnhanced()
    print(f"✓ Suspects in database: {len(face_rec.suspects)}")
    print("✓ Face detector ready")
    print("✓ Module loaded successfully!")
