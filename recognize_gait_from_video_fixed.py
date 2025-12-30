"""
SkyGuard - Video Gait Recognition (Fixed Dependencies)
Compatible version without MediaPipe issues
"""

import cv2
import numpy as np
from models.gait_recognition_emergency import GaitRecognitionEmergency
from datetime import datetime
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"⚠️  MediaPipe import warning: {e}")
    MEDIAPIPE_AVAILABLE = False

from scipy.signal import find_peaks
from scipy.stats import pearsonr


class VideoGaitProcessor:
    """Enhanced video processor with fixed dependencies"""
    
    def __init__(self, gait_engine, debug_mode=False, show_match_scores=True):
        self.gait_engine = gait_engine
        self.debug_mode = debug_mode
        self.show_match_scores = show_match_scores
        
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available. Please fix installation.")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("⏳ Initializing MediaPipe...")
        
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Use 1 for compatibility
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe ready")
        except Exception as e:
            print(f"❌ MediaPipe initialization failed: {e}")
            print("\n💡 Try: pip install --upgrade mediapipe protobuf==3.20.3")
            raise
        
        # Tracking
        self.person_tracks = {}
        self.next_track_id = 0
        self.min_frames_for_match = 60
        self.max_track_gap = 30
        
        # Quality filters
        self.min_visibility_threshold = 0.5
        self.min_body_size = 0.05
        self.max_body_size = 0.85
        
        # Debug tracking
        self.rejection_reasons = {}
        self.match_attempts = []
        
    def validate_pose_quality(self, landmarks, frame_shape):
        """Validate pose quality"""
        lm = landmarks.landmark
        
        if len(lm) < 33:
            return False, "Insufficient landmarks"
        
        key_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
        visible_count = sum(1 for idx in key_landmarks 
                          if lm[idx.value].visibility >= self.min_visibility_threshold)
        
        if visible_count < 4:
            return False, f"Low visibility: {visible_count}/6"
        
        xs = [lm[i].x for i in range(len(lm)) if lm[i].visibility > self.min_visibility_threshold]
        ys = [lm[i].y for i in range(len(lm)) if lm[i].visibility > self.min_visibility_threshold]
        
        if not xs or not ys:
            return False, "No visible landmarks"
        
        body_width = max(xs) - min(xs)
        body_height = max(ys) - min(ys)
        body_area = body_width * body_height
        
        if body_area < self.min_body_size or body_area > self.max_body_size:
            return False, f"Body size: {body_area:.4f}"
        
        aspect_ratio = body_height / max(body_width, 0.001)
        if aspect_ratio < 1.3:
            return False, f"Aspect ratio: {aspect_ratio:.2f}"
        
        return True, "Valid"
    
    def extract_features(self, landmarks):
        """Extract comprehensive gait features"""
        lm = landmarks.landmark
        
        try:
            # Get landmarks
            lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            lk = lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            rk = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            la = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            ra = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            features = {
                "stride_length": abs(la.x - ra.x),
                "stride_height_diff": abs(la.y - ra.y),
                "hip_sway": abs(lh.x - rh.x),
                "hip_width": np.sqrt((lh.x - rh.x)**2 + (lh.y - rh.y)**2),
                "knee_lift_left": abs(lk.y - lh.y),
                "knee_lift_right": abs(rk.y - rh.y),
                "knee_spread": abs(lk.x - rk.x),
                "leg_angle_left": abs(la.y - lk.y),
                "leg_angle_right": abs(ra.y - rk.y),
                "leg_extension_left": np.sqrt((la.x - lk.x)**2 + (la.y - lk.y)**2),
                "leg_extension_right": np.sqrt((ra.x - rk.x)**2 + (ra.y - rk.y)**2),
                "shoulder_sway": abs(ls.x - rs.x),
                "body_lean": abs((lh.x + rh.x)/2 - (ls.x + rs.x)/2),
                "knee_asymmetry": abs(abs(lk.y - lh.y) - abs(rk.y - rh.y)),
                "stance_width": abs((lh.x + rh.x)/2 - (la.x + ra.x)/2),
            }
            
            return features
        except Exception as e:
            return {}
    
    def is_valid_motion(self, track_features):
        """Check for valid walking motion"""
        if len(track_features) < 10:
            return False
        
        strides = [f['stride_length'] for f in track_features[-30:] if 'stride_length' in f]
        
        if len(strides) < 10:
            return False
        
        stride_std = np.std(strides)
        
        if stride_std < 0.01:
            return False
        
        # Check for periodic pattern
        if len(strides) >= 20:
            try:
                stride_array = np.array(strides)
                normalized = stride_array - np.mean(stride_array)
                peaks, _ = find_peaks(normalized, distance=5)
                
                if len(peaks) >= 2:
                    return True
            except:
                pass
        
        return stride_std > 0.008
    
    def build_enhanced_signature(self, track_features):
        """Build comprehensive gait signature"""
        if len(track_features) < 30:
            return {}
        
        signature = {}
        
        # Statistical features
        for key in track_features[0].keys():
            values = [f[key] for f in track_features if key in f and f[key] is not None]
            
            if len(values) < 10:
                continue
            
            values = np.array(values)
            
            signature[f"{key}_mean"] = float(np.mean(values))
            signature[f"{key}_std"] = float(np.std(values))
            signature[f"{key}_median"] = float(np.median(values))
            signature[f"{key}_q25"] = float(np.percentile(values, 25))
            signature[f"{key}_q75"] = float(np.percentile(values, 75))
            signature[f"{key}_range"] = float(np.ptp(values))
        
        # Temporal features
        stride_values = [f['stride_length'] for f in track_features if 'stride_length' in f]
        
        if len(stride_values) >= 30:
            try:
                stride_array = np.array(stride_values)
                normalized = stride_array - np.mean(stride_array)
                
                peaks, _ = find_peaks(normalized, distance=8, prominence=0.01)
                
                if len(peaks) >= 3:
                    intervals = np.diff(peaks)
                    signature['stride_frequency'] = float(30.0 / np.mean(intervals))
                    signature['stride_regularity'] = float(np.std(intervals))
                    signature['stride_count'] = len(peaks)
            except:
                pass
        
        # Gait symmetry
        left_knee = [f['knee_lift_left'] for f in track_features if 'knee_lift_left' in f]
        right_knee = [f['knee_lift_right'] for f in track_features if 'knee_lift_right' in f]
        
        if len(left_knee) >= 20 and len(right_knee) >= 20:
            try:
                min_len = min(len(left_knee), len(right_knee))
                correlation, _ = pearsonr(left_knee[:min_len], right_knee[:min_len])
                signature['gait_symmetry'] = float(abs(correlation))
            except:
                pass
        
        return signature
    
    def match_with_improved_algorithm(self, current_signature, threshold=0.65):
        """Enhanced matching with weighted scoring"""
        if not self.gait_engine.profiles:
            return None, 0, {}
        
        best_match = None
        best_score = 0
        all_scores = {}
        
        # Feature weights
        feature_weights = {
            'stride_length': 3.0,
            'stride_frequency': 3.5,
            'stride_regularity': 2.5,
            'gait_symmetry': 2.8,
            'hip_sway': 2.2,
            'knee_lift': 2.0,
            'leg_extension': 2.0,
            'body_lean': 1.5,
            'shoulder_sway': 1.3,
            'stance_width': 1.8,
            'knee_asymmetry': 2.2,
        }
        
        for name, profile_data in self.gait_engine.profiles.items():
            profile_sig = profile_data['signature']
            
            common_keys = set(current_signature.keys()) & set(profile_sig.keys())
            
            if not common_keys:
                all_scores[name] = {'overall': 0, 'features': {}}
                continue
            
            weighted_scores = []
            weights_used = []
            feature_scores = {}
            
            for key in common_keys:
                curr_val = current_signature[key]
                prof_val = profile_sig[key]
                
                if abs(curr_val) + abs(prof_val) > 0:
                    normalized_diff = abs(curr_val - prof_val) / (abs(curr_val) + abs(prof_val))
                else:
                    normalized_diff = 0
                
                feature_score = max(0, 1 - normalized_diff)
                feature_scores[key] = feature_score
                
                # Apply weight
                weight = 1.0
                for feature_prefix, feature_weight in feature_weights.items():
                    if feature_prefix in key:
                        weight = feature_weight
                        break
                
                weighted_scores.append(feature_score * weight)
                weights_used.append(weight)
            
            if weighted_scores:
                overall_score = sum(weighted_scores) / sum(weights_used)
                
                # Bonus for key features
                key_features = ['stride_frequency', 'gait_symmetry', 'stride_regularity']
                key_count = sum(1 for kf in key_features if any(kf in k for k in common_keys))
                
                if key_count >= 2:
                    overall_score *= 1.1
                
                overall_score = min(1.0, overall_score)
                
                all_scores[name] = {
                    'overall': overall_score,
                    'features': feature_scores,
                    'common_features': len(common_keys)
                }
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = name
        
        if best_score >= threshold:
            return best_match, best_score, all_scores
        else:
            return None, best_score, all_scores
    
    def assign_track_id(self, bbox, frame_number):
        """Track persons across frames"""
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        best_match_id = None
        best_score = float('inf')
        
        for track_id, track_data in list(self.person_tracks.items()):
            gap = frame_number - track_data['last_frame']
            if gap > self.max_track_gap:
                del self.person_tracks[track_id]
                continue
            
            last_x, last_y = track_data['last_center']
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            
            last_bbox = track_data.get('bbox', bbox)
            size_diff = abs(bbox[2] - last_bbox[2]) + abs(bbox[3] - last_bbox[3])
            size_score = size_diff / max(bbox[2] + bbox[3], 1)
            
            combined_score = distance + size_score * 30
            
            if combined_score < best_score and distance < 200:
                best_score = combined_score
                best_match_id = track_id
        
        if best_match_id is None:
            best_match_id = self.next_track_id
            self.next_track_id += 1
            self.person_tracks[best_match_id] = {
                'features': [],
                'last_center': (center_x, center_y),
                'last_frame': frame_number,
                'first_frame': frame_number,
                'bbox': bbox
            }
        else:
            self.person_tracks[best_match_id]['last_center'] = (center_x, center_y)
            self.person_tracks[best_match_id]['last_frame'] = frame_number
            self.person_tracks[best_match_id]['bbox'] = bbox
        
        return best_match_id
    
    def process_video(self, video_path, output_path=None, matching_threshold=0.65):
        """Process video with enhanced matching"""
        print(f"\n{'='*70}")
        print(f"🎥 Processing: {os.path.basename(video_path)}")
        print(f"{'='*70}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Cannot open video")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📊 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f}s")
        print(f"\n🔧 Settings:")
        print(f"   Matching threshold: {matching_threshold:.0%}")
        print(f"   Min frames: {self.min_frames_for_match}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        matches_found = []
        rejected_count = 0
        accepted_count = 0
        
        print(f"\n🔍 Processing...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                if frame_number % 30 == 0:
                    print(f"   Frame {frame_number}/{total_frames} ({frame_number/total_frames*100:.0f}%)")
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    results = self.pose.process(rgb)
                except Exception as e:
                    print(f"⚠️  Frame {frame_number} processing error: {e}")
                    if writer:
                        writer.write(frame)
                    continue
                
                if not results.pose_landmarks:
                    if writer:
                        writer.write(frame)
                    continue
                
                is_valid, reason = self.validate_pose_quality(results.pose_landmarks, frame.shape)
                
                if not is_valid:
                    rejected_count += 1
                    if writer:
                        writer.write(frame)
                    continue
                
                accepted_count += 1
                
                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    )
                )
                
                lm = results.pose_landmarks.landmark
                xs = [lm[i].x * width for i in range(len(lm)) 
                      if lm[i].visibility > self.min_visibility_threshold]
                ys = [lm[i].y * height for i in range(len(lm)) 
                      if lm[i].visibility > self.min_visibility_threshold]
                
                if xs and ys:
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    
                    padding = 10
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(width, x_max + padding)
                    y_max = min(height, y_max + padding)
                    
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    track_id = self.assign_track_id(bbox, frame_number)
                    features = self.extract_features(results.pose_landmarks)
                    
                    if features:
                        self.person_tracks[track_id]['features'].append(features)
                    
                    track_features = self.person_tracks[track_id]['features']
                    
                    if len(track_features) >= self.min_frames_for_match:
                        if not self.is_valid_motion(track_features):
                            continue
                        
                        current_signature = self.build_enhanced_signature(track_features)
                        
                        if not current_signature or len(current_signature) < 10:
                            continue
                        
                        # Match
                        matched_name, confidence, all_scores = self.match_with_improved_algorithm(
                            current_signature, threshold=matching_threshold
                        )
                        
                        # Store for debugging
                        if self.show_match_scores and len(track_features) % 30 == 0:
                            self.match_attempts.append({
                                'frame': frame_number,
                                'track_id': track_id,
                                'frames_collected': len(track_features),
                                'scores': all_scores
                            })
                        
                        if matched_name and confidence >= matching_threshold:
                            # Avoid duplicates
                            recent = next((m for m in matches_found[-5:] 
                                         if m['track_id'] == track_id and 
                                         frame_number - m['frame'] < 30), None)
                            
                            if not recent:
                                match_info = {
                                    'frame': frame_number,
                                    'time': frame_number / fps,
                                    'name': matched_name,
                                    'confidence': confidence,
                                    'track_id': track_id,
                                    'frames_analyzed': len(track_features)
                                }
                                
                                matches_found.append(match_info)
                                
                                print(f"\n{'='*70}")
                                print(f"🚨 SUSPECT DETECTED!")
                                print(f"{'='*70}")
                                print(f"Frame: {frame_number}")
                                print(f"Time: {frame_number/fps:.1f}s")
                                print(f"Name: {matched_name}")
                                print(f"Confidence: {confidence:.1%}")
                                print(f"{'='*70}\n")
                            
                            # RED box
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
                            
                            cv2.rectangle(frame, (x, y-120), (x+w+300, y), (0, 0, 200), -1)
                            
                            cv2.putText(frame, f"SUSPECT: {matched_name}", 
                                       (x+5, y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                                       (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, f"Time: {frame_number/fps:.1f}s", 
                                       (x+5, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            if int(frame_number / 10) % 2 == 0:
                                cv2.putText(frame, "!!! MATCHED !!!", 
                                           (x+w+15, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        else:
                            # Tracking box
                            x, y, w, h = bbox
                            
                            if confidence > 0.5:
                                color = (0, 165, 255)
                            elif confidence > 0.4:
                                color = (0, 255, 255)
                            else:
                                color = (255, 0, 0)
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            status = f"ID:{track_id} ({len(track_features)}f)"
                            if confidence > 0.3:
                                status += f" {confidence:.0%}"
                            
                            cv2.putText(frame, status, 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if writer:
                    writer.write(frame)
        
        except KeyboardInterrupt:
            print("\n⏸  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            cap.release()
            if writer:
                writer.release()
        
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Accepted: {accepted_count} ({accepted_count/max(frame_number,1)*100:.1f}%)")
        print(f"Rejected: {rejected_count} ({rejected_count/max(frame_number,1)*100:.1f}%)")
        print(f"Tracks: {len(self.person_tracks)}")
        print(f"Matches: {len(matches_found)}")
        
        # Show match analysis
        if self.show_match_scores and self.match_attempts:
            print(f"\n{'='*70}")
            print(f"🔍 MATCH SCORE ANALYSIS")
            print(f"{'='*70}")
            
            for attempt in self.match_attempts[-3:]:
                print(f"\nFrame {attempt['frame']} (Track {attempt['track_id']}, {attempt['frames_collected']} frames):")
                for name, score_data in attempt['scores'].items():
                    if isinstance(score_data, dict):
                        overall = score_data['overall']
                        print(f"  {name}: {overall:.1%}")
                        if overall > 0.40:
                            sorted_feats = sorted(score_data['features'].items(), 
                                                key=lambda x: x[1], reverse=True)
                            for feat, score in sorted_feats[:5]:
                                print(f"    • {feat}: {score:.1%}")
        
        if matches_found:
            print(f"\n🚨 DETECTIONS:")
            for m in matches_found[:10]:
                print(f"   • {m['name']} at {m['time']:.1f}s ({m['confidence']:.1%})")
        
        if output_path:
            print(f"\n💾 Saved: {output_path}")
        
        print(f"{'='*70}\n")
        
        return matches_found


def main():
    print("\n" + "=" * 70)
    print("🎥 SkyGuard - Enhanced Gait Recognition (Fixed)")
    print("=" * 70)
    
    try:
        gait_engine = GaitRecognitionEmergency()
    except Exception as e:
        print(f"❌ Failed to load gait engine: {e}")
        return
    
    if not gait_engine.profiles:
        print("\n⚠️  No profiles loaded!")
        return
    
    print(f"\n✅ Loaded {len(gait_engine.profiles)} profile(s):")
    for name in gait_engine.profiles.keys():
        print(f"   • {name}")
    
    video_path = input("\n📂 Video path: ").strip('\'"')
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return
    
    save = input("💾 Save output? (y/n): ").lower()
    output_path = None
    
    if save == 'y':
        os.makedirs("processed_videos", exist_ok=True)
        output_path = f"processed_videos/enhanced_{os.path.basename(video_path)}"
    
    print("\n🎯 Threshold:")
    print("   1. Strict (70%)")
    print("   2. Moderate (65%)")
    print("   3. Lenient (60%)")
    
    choice = input("Choose (1-3): ").strip()
    thresholds = {'1': 0.70, '2': 0.65, '3': 0.60}
    threshold = thresholds.get(choice, 0.65)
    
    try:
        processor = VideoGaitProcessor(gait_engine, show_match_scores=True)
        matches = processor.process_video(video_path, output_path, matching_threshold=threshold)
        
        if matches:
            print(f"\n✅ Found {len(matches)} match(es)!")
        else:
            print("\n⚠️  No matches found")
            print("💡 Try lowering threshold or recapturing profile")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()