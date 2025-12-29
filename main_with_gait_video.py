#!/usr/bin/env python3
"""
SkyGuard with Video-Based Gait Recognition
Identifies suspects by their walking patterns captured from video
"""

import cv2
from datetime import datetime
import sys

from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
from models.face_recognition_deepface import FaceRecognitionDeepFace
from models.gait_recognition_video import GaitRecognitionVideo
from models.database import init_database, get_session

from blockchain import BlockchainRBAC, Permission

class SkyGuardGaitVideo:
    """SkyGuard with Video-Based Gait Recognition"""
    
    def __init__(self, username, gps_lat=19.0760, gps_lon=72.8777):
        print("\n" + "="*70)
        print("🛡️  SKYGUARD - GAIT RECOGNITION FROM VIDEO")
        print("="*70)
        print("🚶 Identifies suspects by walking patterns")
        print("📹 Works with video surveillance footage")
        print("="*70)
        
        # Blockchain
        self.rbac = BlockchainRBAC()
        self.username = username
        
        user_role = self.rbac.get_user_role(username)
        if not user_role:
            print(f"❌ User '{username}' not found")
            sys.exit(1)
        
        print(f"\n✓ User: {username}")
        print(f"✓ Role: {user_role.name}")
        
        self.gps_lat = gps_lat
        self.gps_lon = gps_lon
        
        # Database
        self.db_engine = init_database('data/database/skyguard.db')
        self.db_session = get_session(self.db_engine)
        
        # Initialize features
        self.initialize_features()
        
        self.frame_count = 0
        
        print("\n" + "="*70)
        print("✅ GAIT RECOGNITION SYSTEM READY")
        print("="*70 + "\n")
    
    def initialize_features(self):
        """Initialize features"""
        
        # Feature 1: Heat Map
        if self.rbac.has_permission(self.username, Permission.VIEW_HEATMAP):
            print("\n[Feature 1] Crowd Heat Map - ENABLED")
            self.crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
            self.has_heatmap = True
        else:
            self.crowd_analyzer = None
            self.has_heatmap = False
        
        # Feature 2: Face Recognition (DeepFace)
        if self.rbac.has_permission(self.username, Permission.VIEW_SUSPECTS):
            print("\n[Feature 2] Face Recognition - ENABLED")
            self.face_recognition = FaceRecognitionDeepFace(model_name='Facenet')
            self.has_face = True
        else:
            self.face_recognition = None
            self.has_face = False
        
        # Feature 5: Gait Recognition from Video
        if self.rbac.has_permission(self.username, Permission.VIEW_SUSPECTS):
            print("\n[Feature 5] Gait Recognition (Video) - ENABLED")
            self.gait_recognition = GaitRecognitionVideo(min_sequence_length=60)
            print(f"✓ Gait profiles loaded: {len(self.gait_recognition.gait_profiles)}")
            self.has_gait = True
        else:
            self.gait_recognition = None
            self.has_gait = False
    
    def process_frame(self, frame):
        """Process frame with all features"""
        self.frame_count += 1
        
        output_frame = frame.copy()
        
        # Feature 1: Heat Map
        if self.has_heatmap:
            try:
                heatmap_frame, crowd_count, risk_level, heatmap, zones = \
                    self.crowd_analyzer.analyze_crowd(frame, draw_visualization=True)
                output_frame = heatmap_frame
            except:
                pass
        
        # Feature 2: Face Recognition
        if self.has_face:
            try:
                output_frame, detections, suspects = \
                    self.face_recognition.process_frame(
                        output_frame,
                        latitude=self.gps_lat,
                        longitude=self.gps_lon,
                        draw_boxes=True
                    )
            except:
                pass
        
        # Feature 5: Gait Recognition
        if self.has_gait:
            try:
                output_frame, gait_matches, tracking = \
                    self.gait_recognition.process_frame(
                        output_frame,
                        self.frame_count,
                        draw_visualization=True
                    )
                
                for match in gait_matches:
                    print(f"\n{'='*70}")
                    print(f"🚶 GAIT MATCH FROM VIDEO")
                    print(f"{'='*70}")
                    print(f"Name: {match['name']}")
                    print(f"Confidence: {match['confidence']:.0%}")
                    print(f"Frames analyzed: {match['frames_analyzed']}")
                    print(f"Location: {self.gps_lat}, {self.gps_lon}")
                    print(f"{'='*70}\n")
            except Exception as e:
                pass
        
        # Add system info
        cv2.putText(output_frame, f"User: {self.username} | Frame: {self.frame_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, "Gait Recognition: Active (Video Analysis)",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return output_frame
    
    def run(self, video_source=0):
        """Run system"""
        print(f"\n🎥 Opening video: {video_source}\n")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Cannot open video")
            return
        
        print(f"▶ Running as: {self.username}")
        print(f"🚶 Gait recognition active (video-based)")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = self.process_frame(frame)
                
                cv2.imshow('SkyGuard Gait (Video)', processed)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏸ Stopped")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db_session.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SkyGuard with Gait Recognition')
    parser.add_argument('--user', type=str, required=True, help='Username')
    parser.add_argument('--video', type=str, default='0', help='Video source')
    parser.add_argument('--lat', type=float, default=19.0760, help='GPS Latitude')
    parser.add_argument('--lon', type=float, default=72.8777, help='GPS Longitude')
    args = parser.parse_args()
    
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    system = SkyGuardGaitVideo(username=args.user, gps_lat=args.lat, gps_lon=args.lon)
    system.run(video_source)
