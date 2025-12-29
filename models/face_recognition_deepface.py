"""
High-Accuracy Face Recognition using DeepFace
No dlib/cmake required - easier installation
Uses state-of-the-art deep learning models
"""

import cv2
import numpy as np
import os
from datetime import datetime
import json
import pickle

try:
    from deepface import DeepFace
    USE_DEEPFACE = True
    print("✓ Using DeepFace (state-of-the-art)")
except ImportError:
    USE_DEEPFACE = False
    print("⚠ DeepFace not available")

class FaceRecognitionDeepFace:
    """
    High-accuracy face recognition using DeepFace
    Supports multiple models: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace
    """
    
    def __init__(self, suspects_db_path='data/suspects', model_name='Facenet', distance_threshold=0.4):
        """
        Initialize DeepFace recognition
        
        Args:
            suspects_db_path: Path to suspect photos
            model_name: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'ArcFace'
            distance_threshold: Lower = more strict (0.3-0.5 recommended)
        """
        self.suspects_db_path = suspects_db_path
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        
        os.makedirs(suspects_db_path, exist_ok=True)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load suspects
        self.load_suspects()
        
        # Alert tracking
        self.recent_detections = {}
        self.alert_cooldown = 60
        
        print(f"✓ DeepFace recognition initialized")
        print(f"✓ Model: {model_name}")
        print(f"✓ Suspects loaded: {len(self.suspect_embeddings)}")
    
    def load_suspects(self):
        """Load suspect photos and generate embeddings"""
        self.suspect_embeddings = {}
        self.suspect_metadata = {}
        self.suspect_images = {}
        
        metadata_path = os.path.join(self.suspects_db_path, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            print("⚠ No suspects found. Add suspects first: python3 add_suspect.py")
            return
        
        with open(metadata_path, 'r') as f:
            suspects_list = json.load(f)
        
        print(f"\n📸 Loading {len(suspects_list)} suspect(s)...")
        
        for suspect in suspects_list:
            name = suspect['name']
            img_path = suspect['image_path']
            
            if not os.path.exists(img_path):
                print(f"  ✗ {name}: Photo not found")
                continue
            
            try:
                if USE_DEEPFACE:
                    # Generate embedding using DeepFace
                    # This will download the model on first run
                    embedding_obj = DeepFace.represent(
                        img_path=img_path,
                        model_name=self.model_name,
                        enforce_detection=False
                    )
                    
                    if embedding_obj and len(embedding_obj) > 0:
                        embedding = embedding_obj[0]['embedding']
                        
                        self.suspect_embeddings[name] = np.array(embedding)
                        self.suspect_images[name] = img_path
                        self.suspect_metadata[name] = {
                            'description': suspect.get('description', ''),
                            'added_by': suspect.get('uploaded_by', 'Unknown'),
                            'added_date': suspect.get('added_date', '')
                        }
                        
                        print(f"  ✓ {name}: Embedding generated ({len(embedding)}-D vector)")
                    else:
                        print(f"  ✗ {name}: No face detected in photo")
                
                else:
                    # Fallback: Store image for OpenCV matching
                    image = cv2.imread(img_path)
                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                        )
                        
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_img = gray[y:y+h, x:x+w]
                            face_img = cv2.resize(face_img, (128, 128))
                            face_img = cv2.equalizeHist(face_img)
                            
                            self.suspect_embeddings[name] = face_img
                            self.suspect_metadata[name] = {
                                'description': suspect.get('description', ''),
                                'added_by': suspect.get('uploaded_by', 'Unknown')
                            }
                            
                            print(f"  ✓ {name}: Template stored (fallback mode)")
                        else:
                            print(f"  ✗ {name}: No face detected")
            
            except Exception as e:
                print(f"  ✗ {name}: Error - {e}")
        
        print(f"\n✓ Successfully loaded {len(self.suspect_embeddings)} suspect(s)")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        return [[x, y, w, h] for (x, y, w, h) in faces]
    
    def identify_face(self, frame, face_location):
        """
        Identify if detected face matches any suspect
        
        Returns:
            (suspect_name, confidence) or (None, 0)
        """
        x, y, w, h = face_location
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return None, 0
        
        # Add padding to improve detection
        padding = 20
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        
        face_padded = frame[y1:y2, x1:x2]
        
        if USE_DEEPFACE:
            try:
                # Generate embedding for detected face
                face_embedding_obj = DeepFace.represent(
                    img_path=face_padded,
                    model_name=self.model_name,
                    enforce_detection=False
                )
                
                if not face_embedding_obj or len(face_embedding_obj) == 0:
                    return None, 0
                
                face_embedding = np.array(face_embedding_obj[0]['embedding'])
                
                # Compare with all suspects
                best_match = None
                best_distance = float('inf')
                
                for suspect_name, suspect_embedding in self.suspect_embeddings.items():
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(face_embedding - suspect_embedding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = suspect_name
                
                # Convert distance to confidence (0-1)
                # Lower distance = higher confidence
                if best_distance < self.distance_threshold * 10:  # Scale threshold
                    confidence = 1.0 - (best_distance / (self.distance_threshold * 10))
                    confidence = max(0, min(1, confidence))
                    
                    # Apply threshold
                    threshold = 0.6  # Minimum confidence
                    if confidence > threshold:
                        return best_match, confidence
                
                return None, 0
            
            except Exception as e:
                # Silently fail and try next frame
                return None, 0
        
        else:
            # Fallback: Template matching
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (128, 128))
            gray_face = cv2.equalizeHist(gray_face)
            
            best_match = None
            best_score = 0
            
            for suspect_name, suspect_template in self.suspect_embeddings.items():
                result = cv2.matchTemplate(gray_face, suspect_template, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]
                score = (score + 1) / 2
                
                if score > best_score:
                    best_score = score
                    best_match = suspect_name
            
            threshold = 0.75
            if best_score > threshold:
                return best_match, best_score
            
            return None, 0
    
    def should_send_alert(self, suspect_name):
        """Check alert cooldown"""
        current_time = datetime.now()
        
        if suspect_name in self.recent_detections:
            last_alert = self.recent_detections[suspect_name]
            time_diff = (current_time - last_alert).seconds
            
            if time_diff < self.alert_cooldown:
                return False
        
        return True
    
    def process_frame(self, frame, latitude=19.0760, longitude=72.8777, draw_boxes=True):
        """
        Process frame: detect and identify suspects with high accuracy
        
        Returns:
            output_frame, face_detections, identified_suspects
        """
        output_frame = frame.copy()
        
        # Detect all faces
        face_locations = self.detect_faces(frame)
        
        face_detections = []
        identified_suspects = []
        
        for face_location in face_locations:
            x, y, w, h = face_location
            
            # Identify face
            suspect_name, confidence = self.identify_face(frame, face_location)
            
            if suspect_name and confidence > 0.6:
                # SUSPECT IDENTIFIED
                
                # Check alert cooldown
                if self.should_send_alert(suspect_name):
                    identified_suspects.append({
                        'name': suspect_name,
                        'confidence': confidence,
                        'bbox': face_location,
                        'latitude': latitude,
                        'longitude': longitude,
                        'timestamp': datetime.now()
                    })
                    
                    self.recent_detections[suspect_name] = datetime.now()
                    
                    # Console alert
                    print(f"\n{'='*70}")
                    print(f"🚨 SUSPECT IDENTIFIED (HIGH ACCURACY)")
                    print(f"{'='*70}")
                    print(f"Name: {suspect_name}")
                    print(f"Confidence: {confidence:.1%}")
                    print(f"Model: {self.model_name}")
                    print(f"Location: {latitude}, {longitude}")
                    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"{'='*70}\n")
                
                if draw_boxes:
                    # RED box for suspect
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
                    
                    # Red background for label
                    label_h = 120
                    cv2.rectangle(output_frame, (x, y-label_h), (x+w+300, y), (0, 0, 255), -1)
                    
                    # Text
                    cv2.putText(output_frame, f"SUSPECT: {suspect_name}", 
                               (x+5, y-95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"Confidence: {confidence:.1%}", 
                               (x+5, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"GPS: {latitude:.4f}, {longitude:.4f}", 
                               (x+5, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(output_frame, f"Model: {self.model_name}", 
                               (x+5, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(output_frame, "DEEPFACE AI", 
                               (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Blinking alert
                    if int(datetime.now().timestamp() * 2) % 2 == 0:
                        cv2.putText(output_frame, "!!! MATCH !!!", 
                                   (x+w+10, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                
                face_detections.append({
                    'bbox': face_location,
                    'identified': True,
                    'name': suspect_name,
                    'confidence': confidence
                })
            
            else:
                # Unknown face
                if draw_boxes:
                    # GREEN box for unknown (thinner line)
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output_frame, "Unknown", 
                               (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                face_detections.append({
                    'bbox': face_location,
                    'identified': False
                })
        
        return output_frame, face_detections, identified_suspects


if __name__ == '__main__':
    print("="*70)
    print("🎯 DeepFace High-Accuracy Recognition - Test")
    print("="*70)
    
    face_rec = FaceRecognitionDeepFace(model_name='Facenet')
    
    print(f"\n✓ System ready")
    print(f"✓ Suspects loaded: {len(face_rec.suspect_embeddings)}")
    
    for name in face_rec.suspect_embeddings.keys():
        print(f"   • {name}")
    
    print("\n📹 Ready for video processing")
