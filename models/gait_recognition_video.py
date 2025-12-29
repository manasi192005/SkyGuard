"""
Enhanced Gait Recognition from Video - High Accuracy Version
Fixes: False detections, poor tracking, low accuracy
"""

import cv2
import numpy as np
from collections import deque
import os
import json
from datetime import datetime
import pickle
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean

class GaitRecognitionVideo:
    """
    Enhanced Gait Recognition with improved accuracy
    - Better person detection (filters non-living objects)
    - Robust tracking (same person keeps same ID)
    - Advanced feature extraction
    """
    
    def __init__(self, gait_db_path='data/gait_profiles', min_sequence_length=90):
        """
        Initialize enhanced gait recognition
        
        Args:
            gait_db_path: Path to gait profile database
            min_sequence_length: Minimum frames (90 = 3 seconds at 30fps)
        """
        self.gait_db_path = gait_db_path
        self.min_sequence_length = min_sequence_length
        
        os.makedirs(gait_db_path, exist_ok=True)
        
        # Enhanced background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=True
        )
        
        # Tracking improvements
        self.tracked_persons = {}
        self.person_id_counter = 0
        self.gait_sequences = {}
        self.person_velocities = {}  # Track movement velocity
        self.track_history = {}  # Store tracking history
        
        # Load gait profiles
        self.load_gait_profiles()
        
        # Alert tracking
        self.recent_identifications = {}
        self.alert_cooldown = 60
        
        # Quality thresholds
        self.min_person_area = 2000  # Increased minimum
        self.max_person_area = 80000
        self.min_aspect_ratio = 1.8  # Stricter
        self.max_aspect_ratio = 4.2
        self.min_movement_speed = 0.5  # Pixels per frame
        self.max_tracking_distance = 100  # Reduced for better tracking
        self.track_timeout = 1.5  # Seconds
        
        print(f"✓ Enhanced gait recognition initialized")
        print(f"✓ Gait profiles: {len(self.gait_profiles)}")
    
    def load_gait_profiles(self):
        """Load saved gait profiles"""
        self.gait_profiles = {}
        metadata_path = os.path.join(self.gait_db_path, 'gait_metadata.json')
        
        if not os.path.exists(metadata_path):
            print("  ⚠ No gait profiles found. Capture profiles first.")
            return
        
        with open(metadata_path, 'r') as f:
            profiles_list = json.load(f)
        
        print(f"\n🚶 Loading gait profiles...")
        
        for profile in profiles_list:
            name = profile['name']
            profile_path = profile['profile_path']
            
            if os.path.exists(profile_path):
                with open(profile_path, 'rb') as f:
                    gait_signature = pickle.load(f)
                
                self.gait_profiles[name] = {
                    'signature': gait_signature,
                    'description': profile.get('description', ''),
                    'added_date': profile.get('added_date', ''),
                    'video_frames': profile.get('video_frames', 0)
                }
                
                print(f"  ✓ {name}: {profile.get('video_frames', 0)} frames")
            else:
                print(f"  ✗ {name}: Profile file not found")
        
        print(f"✓ Loaded {len(self.gait_profiles)} profile(s)\n")
    
    def detect_walking_persons(self, frame):
        """
        Enhanced person detection with better filtering
        """
        # Preprocessing for better detection
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame_blur, learningRate=0.001)
        
        # Remove shadows (value 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Advanced morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill gaps
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        # Final cleanup
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        persons = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Strict area filter
            if not (self.min_person_area < area < self.max_person_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)
            
            # Strict aspect ratio (standing person)
            if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                continue
            
            # Calculate shape metrics
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Filter out circular objects (likely not persons)
            if circularity > 0.8:
                continue
            
            # Convexity check
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            
            solidity = area / hull_area
            
            # Filter based on solidity (persons have specific solidity range)
            if not (0.5 < solidity < 0.95):
                continue
            
            # Extract silhouette
            silhouette = fg_mask[y:y+h, x:x+w]
            
            # Additional validation: check if silhouette has person-like structure
            if not self.validate_person_silhouette(silhouette, aspect_ratio):
                continue
            
            persons.append({
                'bbox': [x, y, w, h],
                'silhouette': silhouette,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'contour': contour
            })
        
        return persons, fg_mask
    
    def validate_person_silhouette(self, silhouette, aspect_ratio):
        """
        Validate that silhouette looks like a person
        """
        if silhouette is None or silhouette.size == 0:
            return False
        
        h, w = silhouette.shape
        
        # Divide into thirds
        third = h // 3
        if third == 0:
            return False
        
        upper = silhouette[0:third, :]
        middle = silhouette[third:2*third, :]
        lower = silhouette[2*third:, :]
        
        upper_pixels = np.sum(upper > 0)
        middle_pixels = np.sum(middle > 0)
        lower_pixels = np.sum(lower > 0)
        
        total_pixels = upper_pixels + middle_pixels + lower_pixels
        
        if total_pixels < 100:
            return False
        
        # Person should have more pixels in middle/lower (body and legs)
        if middle_pixels < upper_pixels * 0.5:
            return False
        
        # Check for vertical continuity (person should be continuous)
        middle_rows = np.any(middle > 0, axis=1)
        continuity = np.sum(middle_rows) / max(len(middle_rows), 1)
        
        if continuity < 0.6:  # At least 60% vertical continuity
            return False
        
        return True
    
    def assign_tracking_id(self, person, frame):
        """
        Robust tracking with Kalman-like prediction
        """
        bbox = person['bbox']
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        current_time = datetime.now()
        
        # Clean up old tracks
        ids_to_remove = []
        for person_id, tracked_data in self.tracked_persons.items():
            time_diff = (current_time - tracked_data['last_seen']).total_seconds()
            if time_diff > self.track_timeout:
                ids_to_remove.append(person_id)
        
        for pid in ids_to_remove:
            del self.tracked_persons[pid]
            if pid in self.gait_sequences:
                del self.gait_sequences[pid]
            if pid in self.person_velocities:
                del self.person_velocities[pid]
            if pid in self.track_history:
                del self.track_history[pid]
        
        # Find best matching track
        best_match_id = None
        best_score = float('inf')
        
        for person_id, tracked_data in self.tracked_persons.items():
            tracked_center = tracked_data['center']
            tracked_bbox = tracked_data['bbox']
            
            # Distance score
            distance = euclidean(center, tracked_center)
            
            if distance > self.max_tracking_distance:
                continue
            
            # Size similarity score
            size_diff = abs(w - tracked_bbox[2]) + abs(h - tracked_bbox[3])
            size_score = size_diff / max(w + h, 1)
            
            # Velocity prediction (if available)
            velocity_score = 0
            if person_id in self.person_velocities:
                velocity = self.person_velocities[person_id]
                predicted_center = (
                    tracked_center[0] + velocity[0],
                    tracked_center[1] + velocity[1]
                )
                velocity_distance = euclidean(center, predicted_center)
                velocity_score = velocity_distance / max(distance, 1)
            
            # Combined score (lower is better)
            combined_score = distance + size_score * 20 + velocity_score * 10
            
            if combined_score < best_score:
                best_score = combined_score
                best_match_id = person_id
        
        if best_match_id is None:
            # Create new track
            best_match_id = self.person_id_counter
            self.person_id_counter += 1
            self.track_history[best_match_id] = deque(maxlen=30)
            print(f"  📍 New person: ID {best_match_id}")
        
        # Update tracking
        if best_match_id in self.tracked_persons:
            old_center = self.tracked_persons[best_match_id]['center']
            velocity = (center[0] - old_center[0], center[1] - old_center[1])
            self.person_velocities[best_match_id] = velocity
        
        self.tracked_persons[best_match_id] = {
            'center': center,
            'bbox': bbox,
            'last_seen': current_time,
            'area': person['area']
        }
        
        # Update history
        self.track_history[best_match_id].append({
            'center': center,
            'bbox': bbox,
            'timestamp': current_time
        })
        
        return best_match_id
    
    def extract_gait_features(self, person_data, frame_number):
        """
        Enhanced gait feature extraction
        """
        bbox = person_data['bbox']
        silhouette = person_data['silhouette']
        x, y, w, h = bbox
        
        if silhouette is None or silhouette.size == 0:
            return None
        
        features = {}
        
        # Basic spatial features
        features['height'] = h
        features['width'] = w
        features['aspect_ratio'] = person_data['aspect_ratio']
        features['area'] = person_data['area']
        features['solidity'] = person_data['solidity']
        
        # Center of mass
        M = cv2.moments(silhouette)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            features['center_x'] = cx / max(w, 1)  # Normalized
            features['center_y'] = cy / max(h, 1)
        else:
            features['center_x'] = 0.5
            features['center_y'] = 0.5
        
        # Region analysis (5 segments for better resolution)
        segments = 5
        seg_height = h // segments
        
        segment_areas = []
        segment_widths = []
        
        for i in range(segments):
            start = i * seg_height
            end = (i + 1) * seg_height if i < segments - 1 else h
            segment = silhouette[start:end, :]
            
            seg_area = np.sum(segment > 0)
            segment_areas.append(seg_area)
            
            # Width at segment
            seg_projection = np.any(segment > 0, axis=0)
            seg_width = np.sum(seg_projection)
            segment_widths.append(seg_width)
        
        # Normalize
        total_area = sum(segment_areas) + 1
        features['seg_areas'] = [a / total_area for a in segment_areas]
        features['seg_widths'] = [sw / max(w, 1) for sw in segment_widths]
        
        # Leg analysis (bottom 40%)
        leg_start = int(h * 0.6)
        legs = silhouette[leg_start:, :]
        
        if legs.shape[0] > 0:
            # Leg spread (horizontal extent)
            leg_projection = np.any(legs > 0, axis=0)
            leg_spread = np.sum(leg_projection)
            features['leg_spread'] = leg_spread / max(w, 1)
            
            # Leg separation (detect gap between legs)
            col_sums = np.sum(legs > 0, axis=0)
            
            if len(col_sums) > 0:
                # Normalize
                col_sums_norm = col_sums / (max(col_sums) + 1)
                
                # Find local minima (gap between legs)
                threshold = 0.3
                gaps = col_sums_norm < threshold
                features['leg_gap_ratio'] = np.sum(gaps) / len(gaps)
        
        # Contour-based features
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            perimeter = cv2.arcLength(main_contour, True)
            features['perimeter'] = perimeter
            
            # Compactness
            if features['area'] > 0:
                features['compactness'] = (perimeter ** 2) / features['area']
        
        # Temporal
        features['frame_number'] = frame_number
        features['timestamp'] = datetime.now().timestamp()
        
        return features
    
    def build_gait_signature(self, person_id):
        """
        Build comprehensive gait signature with temporal analysis
        """
        if person_id not in self.gait_sequences:
            return None
        
        sequence = list(self.gait_sequences[person_id])
        
        if len(sequence) < self.min_sequence_length:
            return None
        
        signature = {}
        
        # Statistical features for each measurement
        feature_keys = [k for k in sequence[0].keys() 
                       if k not in ['frame_number', 'timestamp', 'center_x', 'center_y', 'seg_areas', 'seg_widths']]
        
        for key in feature_keys:
            values = [f[key] for f in sequence if key in f and f[key] is not None]
            
            if len(values) > 0:
                values = np.array(values)
                
                signature[f'{key}_mean'] = np.mean(values)
                signature[f'{key}_std'] = np.std(values)
                signature[f'{key}_median'] = np.median(values)
                signature[f'{key}_range'] = np.ptp(values)
                
                # Percentiles for robustness
                signature[f'{key}_q25'] = np.percentile(values, 25)
                signature[f'{key}_q75'] = np.percentile(values, 75)
        
        # Segment area ratios (averaged)
        seg_areas_all = [f['seg_areas'] for f in sequence if 'seg_areas' in f]
        if seg_areas_all:
            seg_areas_array = np.array(seg_areas_all)
            for i in range(len(seg_areas_all[0])):
                signature[f'seg{i}_mean'] = np.mean(seg_areas_array[:, i])
                signature[f'seg{i}_std'] = np.std(seg_areas_array[:, i])
        
        # Temporal gait cycle analysis
        leg_spreads = [f.get('leg_spread', 0) for f in sequence]
        
        if len(leg_spreads) > 30:
            # Find peaks (stride points)
            leg_spreads_array = np.array(leg_spreads)
            
            # Smooth signal
            leg_spreads_smooth = gaussian_filter1d(leg_spreads_array, sigma=2)
            
            # Find peaks
            peaks, properties = find_peaks(leg_spreads_smooth, 
                                          distance=10,  # Min 10 frames between steps
                                          prominence=0.1)
            
            if len(peaks) > 2:
                # Step frequency
                peak_intervals = np.diff(peaks)
                avg_interval = np.mean(peak_intervals)
                signature['step_frequency'] = 30.0 / max(avg_interval, 1)  # Assuming 30fps
                signature['step_regularity'] = np.std(peak_intervals)
                signature['stride_count'] = len(peaks)
                
                # Gait symmetry
                if len(peak_intervals) > 1:
                    signature['gait_symmetry'] = 1.0 - (np.std(peak_intervals) / (np.mean(peak_intervals) + 1))
        
        # Movement smoothness
        centers_x = [f.get('center_x', 0.5) for f in sequence]
        if len(centers_x) > 10:
            # Calculate jerk (derivative of acceleration)
            velocity = np.diff(centers_x)
            acceleration = np.diff(velocity)
            jerk = np.diff(acceleration)
            signature['movement_smoothness'] = -np.std(jerk)  # Lower jerk = smoother
        
        return signature
    
    def match_gait_signature(self, signature, threshold=0.72):
        """
        Enhanced matching with weighted features
        """
        if not signature or not self.gait_profiles:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for name, profile_data in self.gait_profiles.items():
            profile_signature = profile_data['signature']
            
            similarity = self.compute_signature_similarity(signature, profile_signature)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity > threshold:
            return best_match, best_similarity
        
        return None, 0
    
    def compute_signature_similarity(self, sig1, sig2):
        """
        Enhanced similarity computation with feature weighting
        """
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        if not common_keys:
            return 0.0
        
        # Feature importance weights
        feature_weights = {
            'step_frequency': 3.0,  # Most distinctive
            'stride_count': 2.5,
            'gait_symmetry': 2.5,
            'step_regularity': 2.0,
            'aspect_ratio_mean': 2.0,
            'leg_spread_std': 2.5,
            'movement_smoothness': 1.5,
            'compactness_mean': 1.5,
            'solidity_mean': 1.2,
        }
        
        distances = []
        weights = []
        
        for key in common_keys:
            val1 = sig1[key]
            val2 = sig2[key]
            
            # Handle None/NaN
            if val1 is None or val2 is None or np.isnan(val1) or np.isnan(val2):
                continue
            
            # Normalized difference (0 to 1)
            abs_sum = abs(val1) + abs(val2)
            if abs_sum > 0:
                diff = abs(val1 - val2) / abs_sum
            else:
                diff = 0
            
            # Apply weight
            weight = 1.0
            for feature_name, feature_weight in feature_weights.items():
                if feature_name in key:
                    weight = feature_weight
                    break
            
            distances.append(diff)
            weights.append(weight)
        
        if not distances:
            return 0.0
        
        # Weighted average distance
        weighted_distance = np.average(distances, weights=weights)
        
        # Convert to similarity (exponential decay)
        similarity = np.exp(-weighted_distance * 3)
        
        return similarity
    
    def should_send_alert(self, person_id):
        """Check alert cooldown"""
        current_time = datetime.now()
        
        if person_id in self.recent_identifications:
            last_alert = self.recent_identifications[person_id]
            time_diff = (current_time - last_alert).total_seconds()
            
            if time_diff < self.alert_cooldown:
                return False
        
        return True
    
    def process_frame(self, frame, frame_number, draw_visualization=True):
        """
        Process frame with enhanced detection and tracking
        """
        output_frame = frame.copy()
        
        # Detect persons
        persons, fg_mask = self.detect_walking_persons(frame)
        
        gait_matches = []
        tracking_info = []
        
        for person in persons:
            # Assign tracking ID
            person_id = self.assign_tracking_id(person, frame)
            
            # Validate movement (filter static objects)
            if person_id in self.person_velocities:
                velocity = self.person_velocities[person_id]
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                
                # Skip if too slow (likely static object)
                if speed < self.min_movement_speed:
                    continue
            
            # Extract features
            features = self.extract_gait_features(person, frame_number)
            
            if features:
                # Add to sequence
                if person_id not in self.gait_sequences:
                    self.gait_sequences[person_id] = deque(maxlen=150)
                
                self.gait_sequences[person_id].append(features)
                
                # Attempt matching
                if len(self.gait_sequences[person_id]) >= self.min_sequence_length:
                    signature = self.build_gait_signature(person_id)
                    
                    if signature:
                        name, confidence = self.match_gait_signature(signature)
                        
                        if name and confidence > 0.72:
                            if self.should_send_alert(person_id):
                                gait_matches.append({
                                    'person_id': person_id,
                                    'name': name,
                                    'confidence': confidence,
                                    'bbox': person['bbox'],
                                    'frames_analyzed': len(self.gait_sequences[person_id])
                                })
                                
                                self.recent_identifications[person_id] = datetime.now()
                
                tracking_info.append({
                    'person_id': person_id,
                    'bbox': person['bbox'],
                    'frames_captured': len(self.gait_sequences[person_id]) if person_id in self.gait_sequences else 0,
                    'speed': np.sqrt(self.person_velocities.get(person_id, (0,0))[0]**2 + 
                                    self.person_velocities.get(person_id, (0,0))[1]**2)
                })
        
        # Visualization
        if draw_visualization:
            for match in gait_matches:
                bbox = match['bbox']
                x, y, w, h = bbox
                
                # RED alert box
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
                
                label_h = 140
                cv2.rectangle(output_frame, (x, y-label_h), (x+w+350, y), (0, 0, 200), -1)
                
                cv2.putText(output_frame, f"MATCH: {match['name']}", 
                           (x+5, y-115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Confidence: {match['confidence']:.1%}", 
                           (x+5, y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Frames: {match['frames_analyzed']}", 
                           (x+5, y-65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output_frame, "GAIT PATTERN MATCHED", 
                           (x+5, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(output_frame, f"Track ID: {match['person_id']}", 
                           (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Blinking
                if int(datetime.now().timestamp() * 3) % 2 == 0:
                    cv2.putText(output_frame, "!!! IDENTIFIED !!!", 
                               (x+w+15, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            
            # Tracking visualization
            for info in tracking_info:
                if any(m['person_id'] == info['person_id'] for m in gait_matches):
                    continue
                
                bbox = info['bbox']
                x, y, w, h = bbox
                frames = info['frames_captured']
                progress = min(frames / self.min_sequence_length, 1.0)
                
                if frames < self.min_sequence_length:
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 150, 0), 2)
                    cv2.putText(output_frame, f"Track ID:{info['person_id']} [{frames}/{self.min_sequence_length}]", 
                               (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
                    
                    # Progress bar
                    bar_width = int(w * progress)
                    cv2.rectangle(output_frame, (x, y+h+5), (x+bar_width, y+h+15), (255, 150, 0), -1)
                else:
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(output_frame, f"Analyzing ID:{info['person_id']}", 
                               (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return output_frame, gait_matches, tracking_info
    
    def save_gait_profile(self, person_id, name, description=''):
        """Save gait profile"""
        if person_id not in self.gait_sequences:
            return False, "Person ID not found"
        
        sequence = self.gait_sequences[person_id]
        
        if len(sequence) < self.min_sequence_length:
            return False, f"Need {self.min_sequence_length} frames (have {len(sequence)})"
        
        signature = self.build_gait_signature(person_id)
        
        if not signature:
            return False, "Failed to build signature"
        
        profile_path = os.path.join(self.gait_db_path, f'{name}_gait.pkl')
        
        with open(profile_path, 'wb') as f:
            pickle.dump(signature, f)
        
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
            'added_date': datetime.now().isoformat(),
            'video_frames': len(sequence)
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Reload profiles
        self.load_gait_profiles()
        
        return True, f"Profile saved ({len(sequence)} frames)"


if __name__ == '__main__':
    print("="*70)
    print("🚶 Enhanced Gait Recognition from Video - Test")
    print("="*70)
    
    gait_rec = GaitRecognitionVideo()
    
    print(f"\n✓ System ready")
    print(f"✓ Profiles loaded: {len(gait_rec.gait_profiles)}")