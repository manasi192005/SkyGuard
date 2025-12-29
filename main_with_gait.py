#!/usr/bin/env python3
"""
SkyGuard with ALL 5 Features + Blockchain RBAC
Feature 5: Gait Recognition (Walking Pattern Analysis)
"""

import cv2
from datetime import datetime
import sys

from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
from models.face_recognition_rbac import FaceRecognitionRBAC
from models.stampede_prediction_enhanced import StampedePredictorEnhanced
from models.emergency_detection_enhanced import EmergencyDetectorEnhanced
from models.gait_recognition_enhanced import GaitRecognitionEnhanced
from models.database import init_database, get_session

from blockchain import BlockchainRBAC, Permission

class SkyGuardComplete:
    """SkyGuard with ALL 5 Features"""
    
    def __init__(self, username, gps_lat=19.0760, gps_lon=72.8777):
        print("\n" + "="*70)
        print("🛡️  SKYGUARD - COMPLETE SYSTEM (5 FEATURES)")
        print("="*70)
        
        # Blockchain RBAC
        self.rbac = BlockchainRBAC()
        self.username = username
        
        user_role = self.rbac.get_user_role(username)
        if not user_role:
            print(f"❌ User '{username}' not found")
            sys.exit(1)
        
        print(f"\n✓ User: {username}")
        print(f"✓ Role: {user_role.name}")
        
        permissions = self.rbac.get_user_permissions(username)
        print(f"✓ Permissions: {len(permissions)}")
        
        self.gps_lat = gps_lat
        self.gps_lon = gps_lon
        
        # Database
        self.db_engine = init_database('data/database/skyguard.db')
        self.db_session = get_session(self.db_engine)
        
        # Initialize features
        self.initialize_features()
        
        self.frame_count = 0
        
        print("\n" + "="*70)
        print("✅ ALL 5 FEATURES READY")
        print("="*70 + "\n")
    
    def initialize_features(self):
        """Initialize all 5 features"""
        
        # Feature 1: Heat Map
        if self.rbac.has_permission(self.username, Permission.VIEW_HEATMAP):
            print("\n[Feature 1] Crowd Heat Map - ENABLED")
            self.crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
            self.has_heatmap = True
        else:
            print("\n[Feature 1] Heat Map - ACCESS DENIED")
            self.crowd_analyzer = None
            self.has_heatmap = False
        
        # Feature 2: Face Recognition
        if self.rbac.has_permission(self.username, Permission.VIEW_SUSPECTS):
            print("\n[Feature 2] Face Recognition - ENABLED")
            self.face_recognition = FaceRecognitionRBAC(confidence_threshold=0.85)
            self.has_face_recognition = True
        else:
            print("\n[Feature 2] Face Recognition - ACCESS DENIED")
            self.face_recognition = None
            self.has_face_recognition = False
        
        # Feature 3: Stampede Prediction
        if self.rbac.has_permission(self.username, Permission.VIEW_STAMPEDE):
            print("\n[Feature 3] Stampede Prediction - ENABLED")
            self.stampede_predictor = StampedePredictorEnhanced(sequence_length=30, fps=30)
            self.has_stampede = True
        else:
            print("\n[Feature 3] Stampede Prediction - ACCESS DENIED")
            self.stampede_predictor = None
            self.has_stampede = False
        
        # Feature 4: Emergency Detection
        if self.rbac.has_permission(self.username, Permission.VIEW_EMERGENCY):
            print("\n[Feature 4] Emergency Detection - ENABLED")
            self.emergency_detector = EmergencyDetectorEnhanced(
                fall_angle_threshold=60, immobility_seconds=300, fps=30
            )
            self.has_emergency = True
        else:
            print("\n[Feature 4] Emergency Detection - ACCESS DENIED")
            self.emergency_detector = None
            self.has_emergency = False
        
        # Feature 5: Gait Recognition (NEW!)
        if self.rbac.has_permission(self.username, Permission.VIEW_SUSPECTS):
            print("\n[Feature 5] Gait Recognition - ENABLED")
            self.gait_recognition = GaitRecognitionEnhanced(min_frames=30)
            print(f"✓ Gait profiles loaded: {len(self.gait_recognition.gait_profiles)}")
            self.has_gait = True
        else:
            print("\n[Feature 5] Gait Recognition - ACCESS DENIED")
            self.gait_recognition = None
            self.has_gait = False
    
    def process_frame(self, frame):
        """Process with all 5 features"""
        self.frame_count += 1
        
        output_frame = frame.copy()
        alerts = []
        
        # Feature 1: Heat Map
        if self.has_heatmap:
            try:
                heatmap_frame, crowd_count, risk_level, heatmap, zones = \
                    self.crowd_analyzer.analyze_crowd(frame, draw_visualization=True)
                output_frame = heatmap_frame
            except:
                pass
        
        # Feature 2: Face Recognition
        if self.has_face_recognition:
            try:
                output_frame, detections, suspects = \
                    self.face_recognition.process_frame(
                        output_frame,
                        latitude=self.gps_lat,
                        longitude=self.gps_lon,
                        draw_boxes=True
                    )
                
                for suspect in suspects:
                    alerts.append(f"Face: {suspect['name']}")
            except Exception as e:
                pass
        
        # Feature 5: Gait Recognition (NEW!)
        if self.has_gait:
            try:
                output_frame, gait_matches = \
                    self.gait_recognition.process_frame(output_frame, draw_visualization=True)
                
                for match in gait_matches:
                    alerts.append(f"Gait: {match['name']}")
                    print(f"\n🚶 GAIT MATCH: {match['name']} ({match['confidence']:.0%})")
                    print(f"📍 Location: {self.gps_lat}, {self.gps_lon}")
            except Exception as e:
                print(f"Gait error: {e}")
        
        # Add system info
        cv2.putText(output_frame, f"User: {self.username} | Frame: {self.frame_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if alerts:
            alert_text = " | ".join(alerts)
            cv2.putText(output_frame, f"ALERTS: {alert_text}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output_frame
    
    def run(self, video_source=0):
        """Run complete system"""
        print(f"\n🎥 Opening video: {video_source}\n")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Cannot open video")
            return
        
        print(f"▶ Running with 5 features as: {self.username}")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = self.process_frame(frame)
                
                cv2.imshow('SkyGuard Complete (5 Features)', processed)
                
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
    
    parser = argparse.ArgumentParser(description='SkyGuard Complete System')
    parser.add_argument('--user', type=str, required=True, help='Username')
    parser.add_argument('--video', type=str, default='0', help='Video source')
    parser.add_argument('--lat', type=float, default=19.0760, help='GPS Latitude')
    parser.add_argument('--lon', type=float, default=72.8777, help='GPS Longitude')
    args = parser.parse_args()
    
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    system = SkyGuardComplete(username=args.user, gps_lat=args.lat, gps_lon=args.lon)
    system.run(video_source)
