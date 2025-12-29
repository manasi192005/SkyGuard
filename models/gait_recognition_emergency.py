"""
Emergency Gait Recognition Engine
Captures, stores, and recognizes gait profiles
"""

import os
import json
import numpy as np
from datetime import datetime


class GaitRecognitionEmergency:
    def __init__(self):
        self.gait_db_path = "gait_profiles"
        os.makedirs(self.gait_db_path, exist_ok=True)

        self.min_capture_frames = 150
        self.min_walks = 3
        self.min_quality_score = 0.5

        self.frame_buffer = []
        self.walk_count = 0
        self.current_quality = 0.0
        
        # Load existing profiles
        self.profiles = {}
        self.load_all_profiles()
        
        # Alert tracking
        self.recent_alerts = {}
        self.alert_cooldown = 60  # seconds

        print("✅ GaitRecognitionEmergency initialized")
        if len(self.profiles) > 0:
            print(f"✅ Loaded {len(self.profiles)} existing profile(s)")

    def load_all_profiles(self):
        """Load all saved gait profiles from database"""
        if not os.path.exists(self.gait_db_path):
            return
        
        files = [f for f in os.listdir(self.gait_db_path) if f.endswith('.json')]
        
        for filename in files:
            filepath = os.path.join(self.gait_db_path, filename)
            try:
                with open(filepath, 'r') as f:
                    profile = json.load(f)
                    name = profile.get('name', filename.replace('.json', ''))
                    self.profiles[name] = profile
                    print(f"  ✓ Loaded: {name} ({profile.get('num_frames', 0)} frames)")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")

    def add_frame(self, features):
        """Add gait features from current frame"""
        if isinstance(features, dict):
            self.frame_buffer.append(features)

    def increment_walk(self):
        """Increment walk counter"""
        self.walk_count += 1

    def estimate_quality(self):
        """Estimate quality of captured gait data"""
        frame_ratio = min(len(self.frame_buffer) / self.min_capture_frames, 1.0)
        walk_ratio = min(self.walk_count / self.min_walks, 1.0)
        self.current_quality = round((frame_ratio * 0.6 + walk_ratio * 0.4), 2)
        return self.current_quality

    def ready_to_save(self):
        """Check if enough data collected to save profile"""
        return (
            len(self.frame_buffer) >= self.min_capture_frames
            and self.walk_count >= self.min_walks
            and self.current_quality >= self.min_quality_score
        )

    def generate_signature(self):
        """Generate gait signature from captured frames"""
        if not self.frame_buffer:
            return {}

        signature = {}
        for key in self.frame_buffer[0]:
            values = [f[key] for f in self.frame_buffer if key in f]
            if values:
                signature[f"{key}_mean"] = float(np.mean(values))
                signature[f"{key}_std"] = float(np.std(values))
                signature[f"{key}_min"] = float(np.min(values))
                signature[f"{key}_max"] = float(np.max(values))
        return signature

    def save_profile(self, name, description=""):
        """Save captured gait profile to database"""
        profile = {
            "name": name,
            "description": description,
            "created_date": datetime.now().isoformat(),
            "num_frames": len(self.frame_buffer),
            "num_walks": self.walk_count,
            "quality_score": self.current_quality,
            "signature": self.generate_signature()
        }

        path = os.path.join(self.gait_db_path, f"{name}.json")
        with open(path, "w") as f:
            json.dump(profile, f, indent=4)

        print(f"✅ Profile saved → {path}")
        
        # Reload profiles
        self.load_all_profiles()
        
        return True

    def match_gait(self, current_signature, threshold=0.70):
        """
        Match current gait signature against all profiles
        
        Args:
            current_signature: Gait signature from live video
            threshold: Minimum confidence (0.70 = 70%)
        
        Returns:
            (matched_name, confidence) or (None, 0)
        """
        if not current_signature or not self.profiles:
            return None, 0
        
        best_match = None
        best_confidence = 0
        
        for profile_name, profile_data in self.profiles.items():
            stored_signature = profile_data.get('signature', {})
            
            if not stored_signature:
                continue
            
            # Calculate similarity
            similarity = self.compute_similarity(current_signature, stored_signature)
            
            if similarity > best_confidence:
                best_confidence = similarity
                best_match = profile_name
        
        # Apply threshold
        if best_confidence >= threshold:
            return best_match, best_confidence
        
        return None, 0

    def compute_similarity(self, sig1, sig2):
        """
        Compute similarity between two gait signatures
        
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        # Find common features
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        if not common_keys:
            return 0.0
        
        distances = []
        weights = []
        
        # Feature importance weights
        feature_weights = {
            'stride_length': 2.0,
            'hip_sway': 1.5,
            'knee_lift': 1.8,
        }
        
        for key in common_keys:
            val1 = sig1[key]
            val2 = sig2[key]
            
            # Normalized distance
            if abs(val1) + abs(val2) > 0:
                diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                
                # Apply weights
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
        avg_distance = np.average(distances, weights=weights)
        
        # Convert to similarity (0 to 1)
        similarity = 1.0 / (1.0 + avg_distance * 3)
        
        return similarity

    def should_alert(self, person_name):
        """Check if alert cooldown has passed"""
        current_time = datetime.now()
        
        if person_name in self.recent_alerts:
            last_alert_time = self.recent_alerts[person_name]
            time_diff = (current_time - last_alert_time).total_seconds()
            
            if time_diff < self.alert_cooldown:
                return False
        
        return True

    def log_match(self, person_name, confidence):
        """Log successful match for alert cooldown"""
        self.recent_alerts[person_name] = datetime.now()

    def reset_capture(self):
        """Reset capture buffer for new profile"""
        self.frame_buffer = []
        self.walk_count = 0
        self.current_quality = 0.0
