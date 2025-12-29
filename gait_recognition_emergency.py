"""
Emergency Gait Recognition Engine
Hackathon-stable, zero-dependency
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

        print("✅ GaitRecognitionEmergency initialized")

    def add_frame(self, features):
        if isinstance(features, dict):
            self.frame_buffer.append(features)

    def increment_walk(self):
        self.walk_count += 1

    def estimate_quality(self):
        frame_ratio = min(len(self.frame_buffer) / self.min_capture_frames, 1.0)
        walk_ratio = min(self.walk_count / self.min_walks, 1.0)
        self.current_quality = round((frame_ratio * 0.6 + walk_ratio * 0.4), 2)
        return self.current_quality

    def ready_to_save(self):
        return (
            len(self.frame_buffer) >= self.min_capture_frames
            and self.walk_count >= self.min_walks
            and self.current_quality >= self.min_quality_score
        )

    def generate_signature(self):
        if not self.frame_buffer:
            return {}

        signature = {}
        for key in self.frame_buffer[0]:
            values = [f[key] for f in self.frame_buffer if key in f]
            signature[f"{key}_mean"] = float(np.mean(values))
            signature[f"{key}_std"] = float(np.std(values))
        return signature

    def save_profile(self, name, description=""):
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
