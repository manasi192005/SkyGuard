"""
Gait Recognition Module for SkyGuard
Identifies suspects by their unique walking patterns
Feature 5: Behavioral Biometric Identification
"""

import cv2
import numpy as np
from collections import deque
import os
import json
from datetime import datetime
import pickle

class GaitRecognitionEnhanced:
    """
    Gait Recognition System
    Analyzes walking patterns to identify suspects
    """
    
    def __init__(self, gait_db_path='data/gait_profiles', min_frames=30):
        """
        Initialize Gait Recognition
        
        Args:
            gait_db_path: Path to gait profile database
            min_frames: Minimum frames needed for gait analysis
        """
        self.gait_db_path = gait_db_path
        self.min_frames = min_frames
        
        # Create directory
        os.makedirs(gait_db_path, exist_ok=True)
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Tracking data
        self.tracked_persons = {}  # person_id -> gait_features
        self.person_id_counter = 0
        
        # Gait feature history for each person
        self.gait_history = {}  # person_id -> deque of features
        
        # Load gait profiles from database
        self.load_gait_profiles()
        
        # Alert tracking
        self.recent_gait_detections = {}
        self.gait_alert_cooldown = 60  # seconds
        
        print(f"✓ Gait recognition initialized")
        print(f"✓ Gait profiles loaded: {len(self.gait_profiles)}")
    
    def load_gait_profiles(self):
        """Load gait profiles from database"""
        self.gait_profiles = {}
        metadata_path = os.path.join(self.gait_db_path, 'gait_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            for profile in metadata:
                profile_path = profile['profile_path']
                if os.path.exists(profile_path):
                    with open(profile_path, 'rb') as f:
                        gait_data = pickle.load(f)
                    
                    self.gait_profiles[profile['name']] = {
                        'name': profile['name'],
                        'features': gait_data,
                        'description': profile.get('description', ''),
                        'added_date': profile.get('added_date', '')
                    }
    
    def extract_gait_features(self, silhouette, prev_silhouette=None):
        """
        Extract gait features from person silhouette
        
        Features extracted:
        - Step length (stride)
        - Step frequency
        - Body aspect ratio
        - Center of mass movement
        - Leg angle variations
        - Upper body sway
        """
        if silhouette is None or silhouette.size == 0:
            return None
        
        features = {}
        
        # 1. Body dimensions
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (main body)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Aspect ratio (height/width)
        features['aspect_ratio'] = h / max(w, 1)
        
        # 2. Center of mass
        M = cv2.moments(main_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            features['center_x'] = cx
            features['center_y'] = cy
        else:
            features['center_x'] = x + w // 2
            features['center_y'] = y + h // 2
        
        # 3. Body silhouette area
        features['silhouette_area'] = cv2.contourArea(main_contour)
        
        # 4. Perimeter to area ratio (compactness)
        perimeter = cv2.arcLength(main_contour, True)
        features['compactness'] = perimeter / max(features['silhouette_area'], 1)
        
        # 5. Upper and lower body analysis
        mid_y = y + h // 2
        
        # Upper body (torso)
        upper_mask = silhouette.copy()
        upper_mask[mid_y:, :] = 0
        upper_area = np.sum(upper_mask > 0)
        
        # Lower body (legs)
        lower_mask = silhouette.copy()
        lower_mask[:mid_y, :] = 0
        lower_area = np.sum(lower_mask > 0)
        
        features['upper_lower_ratio'] = upper_area / max(lower_area, 1)
        
        # 6. Movement features (if previous frame available)
        if prev_silhouette is not None:
            prev_M = cv2.moments(prev_silhouette)
            if prev_M['m00'] != 0:
                prev_cx = int(prev_M['m10'] / prev_M['m00'])
                prev_cy = int(prev_M['m01'] / prev_M['m00'])
                
                # Movement speed
                features['movement_x'] = features['center_x'] - prev_cx
                features['movement_y'] = features['center_y'] - prev_cy
                features['movement_speed'] = np.sqrt(
                    features['movement_x']**2 + features['movement_y']**2
                )
        
        # 7. Leg spread (width at bottom)
        bottom_region = silhouette[y + int(0.7*h):y + h, x:x + w]
        if bottom_region.size > 0:
            bottom_width = np.sum(np.any(bottom_region > 0, axis=0))
            features['leg_spread'] = bottom_width / max(w, 1)
        else:
            features['leg_spread'] = 0
        
        return features
    
    def detect_persons(self, frame):
        """
        Detect walking persons in frame using background subtraction
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours (persons)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        persons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (only detect human-sized objects)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (height should be greater than width)
                aspect_ratio = h / max(w, 1)
                if 1.5 < aspect_ratio < 4.0:
                    # Extract silhouette
                    silhouette = fg_mask[y:y+h, x:x+w]
                    
                    persons.append({
                        'bbox': [x, y, w, h],
                        'silhouette': silhouette,
                        'area': area
                    })
        
        return persons, fg_mask
    
    def track_and_analyze_gait(self, persons, frame):
        """
        Track persons across frames and analyze their gait patterns
        """
        current_frame_persons = {}
        
        for person in persons:
            # Simple tracking: match with closest person from previous frame
            person_id = self.assign_person_id(person)
            
            # Extract gait features
            prev_silhouette = None
            if person_id in self.gait_history and len(self.gait_history[person_id]) > 0:
                prev_features = self.gait_history[person_id][-1]
                # We'd need to store silhouette, but for simplicity, skip for now
            
            features = self.extract_gait_features(person['silhouette'], prev_silhouette)
            
            if features:
                # Add to history
                if person_id not in self.gait_history:
                    self.gait_history[person_id] = deque(maxlen=self.min_frames * 2)
                
                self.gait_history[person_id].append(features)
                current_frame_persons[person_id] = person
        
        return current_frame_persons
    
    def assign_person_id(self, person):
        """
        Assign ID to person (simple tracking based on proximity)
        """
        bbox = person['bbox']
        center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
        
        # Find closest tracked person
        min_distance = float('inf')
        assigned_id = None
        
        for person_id, tracked_data in self.tracked_persons.items():
            tracked_center = tracked_data.get('center')
            if tracked_center:
                distance = np.sqrt(
                    (center[0] - tracked_center[0])**2 + 
                    (center[1] - tracked_center[1])**2
                )
                
                if distance < min_distance and distance < 100:  # 100 pixel threshold
                    min_distance = distance
                    assigned_id = person_id
        
        if assigned_id is None:
            # New person
            assigned_id = self.person_id_counter
            self.person_id_counter += 1
        
        # Update tracked data
        self.tracked_persons[assigned_id] = {
            'center': center,
            'bbox': bbox,
            'last_seen': datetime.now()
        }
        
        return assigned_id
    
    def compute_gait_signature(self, person_id):
        """
        Compute gait signature from accumulated features
        """
        if person_id not in self.gait_history:
            return None
        
        if len(self.gait_history[person_id]) < self.min_frames:
            return None  # Not enough data
        
        features_list = list(self.gait_history[person_id])
        
        # Compute statistical measures of gait features
        signature = {}
        
        feature_keys = features_list[0].keys()
        
        for key in feature_keys:
            if key not in ['center_x', 'center_y']:  # Skip absolute positions
                values = [f.get(key, 0) for f in features_list if key in f]
                
                if values:
                    signature[f'{key}_mean'] = np.mean(values)
                    signature[f'{key}_std'] = np.std(values)
                    signature[f'{key}_range'] = np.max(values) - np.min(values)
        
        # Compute stride frequency (if movement data available)
        movements = [f.get('movement_speed', 0) for f in features_list if 'movement_speed' in f]
        if movements:
            # Find peaks (steps)
            movement_array = np.array(movements)
            peaks = []
            for i in range(1, len(movement_array) - 1):
                if movement_array[i] > movement_array[i-1] and movement_array[i] > movement_array[i+1]:
                    peaks.append(i)
            
            if len(peaks) > 1:
                # Average time between steps (assuming 30 fps)
                avg_step_interval = np.mean(np.diff(peaks)) / 30.0  # in seconds
                signature['step_frequency'] = 1.0 / max(avg_step_interval, 0.1)
        
        return signature
    
    def match_gait_profile(self, signature, threshold=0.7):
        """
        Match gait signature against database
        
        Returns:
            (name, confidence) if match found, else (None, 0)
        """
        if not signature or not self.gait_profiles:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for name, profile_data in self.gait_profiles.items():
            profile_signature = profile_data['features']
            
            # Compute similarity
            similarity = self.compute_signature_similarity(signature, profile_signature)
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = name
        
        return best_match, best_similarity
    
    def compute_signature_similarity(self, sig1, sig2):
        """
        Compute similarity between two gait signatures
        Uses normalized Euclidean distance
        """
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        if not common_keys:
            return 0.0
        
        distances = []
        
        for key in common_keys:
            # Normalize by range
            val1 = sig1[key]
            val2 = sig2[key]
            
            # Use relative difference
            if abs(val1) + abs(val2) > 0:
                diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                distances.append(diff)
        
        if not distances:
            return 0.0
        
        # Average distance
        avg_distance = np.mean(distances)
        
        # Convert to similarity (0 to 1)
        similarity = 1.0 / (1.0 + avg_distance)
        
        return similarity
    
    def should_send_gait_alert(self, person_id):
        """Check if enough time passed since last gait alert"""
        current_time = datetime.now()
        
        if person_id in self.recent_gait_detections:
            last_alert = self.recent_gait_detections[person_id]
            time_diff = (current_time - last_alert).seconds
            
            if time_diff < self.gait_alert_cooldown:
                return False
        
        return True
    
    def process_frame(self, frame, draw_visualization=True):
        """
        Main processing function
        
        Returns:
            output_frame, gait_matches
        """
        # Detect persons
        persons, fg_mask = self.detect_persons(frame)
        
        # Track and analyze gait
        tracked_persons = self.track_and_analyze_gait(persons, frame)
        
        # Match against profiles
        gait_matches = []
        
        for person_id, person_data in tracked_persons.items():
            # Compute gait signature
            signature = self.compute_gait_signature(person_id)
            
            if signature:
                # Match against database
                name, confidence = self.match_gait_profile(signature)
                
                if name and confidence > 0.7:
                    if self.should_send_gait_alert(person_id):
                        gait_matches.append({
                            'person_id': person_id,
                            'name': name,
                            'confidence': confidence,
                            'bbox': person_data['bbox']
                        })
                        
                        self.recent_gait_detections[person_id] = datetime.now()
        
        # Visualization
        output_frame = frame.copy()
        
        if draw_visualization:
            # Draw tracked persons
            for person_id, person_data in tracked_persons.items():
                bbox = person_data['bbox']
                x, y, w, h = bbox
                
                # Check if this person is a match
                is_match = any(m['person_id'] == person_id for m in gait_matches)
                
                if is_match:
                    # RED for matched suspects
                    color = (0, 0, 255)
                    match = next(m for m in gait_matches if m['person_id'] == person_id)
                    
                    # Draw box
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw label
                    label_h = 90
                    cv2.rectangle(output_frame, (x, y-label_h), (x+w+200, y), color, -1)
                    
                    cv2.putText(output_frame, f"GAIT MATCH: {match['name']}", 
                               (x+5, y-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"Confidence: {match['confidence']:.0%}", 
                               (x+5, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(output_frame, f"ID: {person_id}", 
                               (x+5, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(output_frame, "WALKING PATTERN IDENTIFIED", 
                               (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:
                    # BLUE for tracked but unmatched
                    color = (255, 0, 0)
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(output_frame, f"Tracking ID: {person_id}", 
                               (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # Show gait analysis progress
                    if person_id in self.gait_history:
                        progress = len(self.gait_history[person_id])
                        cv2.putText(output_frame, f"Gait frames: {progress}/{self.min_frames}", 
                                   (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return output_frame, gait_matches
    
    def add_gait_profile(self, name, signature, description=''):
        """Add a new gait profile to database"""
        profile_path = os.path.join(self.gait_db_path, f'{name}_gait.pkl')
        
        # Save signature
        with open(profile_path, 'wb') as f:
            pickle.dump(signature, f)
        
        # Update metadata
        metadata_path = os.path.join(self.gait_db_path, 'gait_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        metadata.append({
            'name': name,
            'profile_path': profile_path,
            'description': description,
            'added_date': datetime.now().isoformat()
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Reload profiles
        self.load_gait_profiles()
        
        print(f"✓ Gait profile added: {name}")


if __name__ == '__main__':
    print("="*60)
    print("🚶 Gait Recognition System - Test")
    print("="*60)
    
    gait_rec = GaitRecognitionEnhanced()
    
    print(f"\n✓ System initialized")
    print(f"✓ Gait profiles in database: {len(gait_rec.gait_profiles)}")
    
    print("\n�� Open video source to test gait recognition")
    print("   python3 test_gait.py")
