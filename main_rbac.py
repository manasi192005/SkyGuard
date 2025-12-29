#!/usr/bin/env python3
"""
SkyGuard with Blockchain RBAC Integration
Role-based access control using blockchain permissions
"""

import cv2
import numpy as np
from datetime import datetime
import sys

from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
from models.face_recognition_enhanced import FaceRecognitionEnhanced
from models.stampede_prediction_enhanced import StampedePredictorEnhanced
from models.emergency_detection_enhanced import EmergencyDetectorEnhanced
from models.database import init_database, get_session, add_crowd_analytics, add_detected_suspect, add_emergency_event

# Import blockchain RBAC
from blockchain import BlockchainRBAC, Permission

class SkyGuardRBAC:
    """SkyGuard System with Blockchain Role-Based Access Control"""
    
    def __init__(self, username, gps_lat=19.0760, gps_lon=72.8777):
        print("\n" + "="*70)
        print("🛡️  SKYGUARD WITH BLOCKCHAIN RBAC")
        print("="*70)
        
        # Initialize blockchain RBAC
        print("\n[Blockchain] Initializing RBAC...")
        self.rbac = BlockchainRBAC()
        self.username = username
        
        # Verify user
        user_role = self.rbac.get_user_role(username)
        if not user_role:
            print(f"❌ User '{username}' not found in blockchain")
            print("   Please register user first: python3 blockchain/manage_users.py")
            sys.exit(1)
        
        print(f"✓ User authenticated: {username}")
        print(f"✓ Role: {user_role.name}")
        
        # Display user permissions
        permissions = self.rbac.get_user_permissions(username)
        print(f"✓ Permissions granted: {len(permissions)}")
        for perm in permissions:
            print(f"   • {perm.value}")
        
        # GPS coordinates
        self.gps_lat = gps_lat
        self.gps_lon = gps_lon
        
        # Database
        print("\n[Database] Connecting...")
        self.db_engine = init_database('data/database/skyguard.db')
        self.db_session = get_session(self.db_engine)
        print("✓ Database connected")
        
        # Initialize features based on permissions
        self.initialize_features()
        
        self.frame_count = 0
        self.start_time = datetime.now()
        
        print("\n" + "="*70)
        print("✅ SYSTEM READY WITH RBAC ENABLED")
        print(f"📍 GPS Location: {self.gps_lat}, {self.gps_lon}")
        print("="*70 + "\n")
    
    def initialize_features(self):
        """Initialize features based on user permissions"""
        
        # Feature 1: Crowd Analysis / Heat Map
        if self.rbac.has_permission(self.username, Permission.VIEW_HEATMAP):
            print("\n[Feature 1] Crowd Density Heat Map...")
            self.crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
            print("✓ Heat map ENABLED")
            self.has_heatmap = True
        else:
            print("\n[Feature 1] Heat Map - ACCESS DENIED")
            self.crowd_analyzer = None
            self.has_heatmap = False
        
        # Feature 2: Face Recognition
        if self.rbac.has_permission(self.username, Permission.VIEW_SUSPECTS):
            print("\n[Feature 2] Face Recognition...")
            self.face_recognition = FaceRecognitionEnhanced(confidence_threshold=0.85)
            print(f"✓ Face recognition ENABLED ({len(self.face_recognition.suspects)} suspects)")
            self.has_face_recognition = True
        else:
            print("\n[Feature 2] Face Recognition - ACCESS DENIED")
            self.face_recognition = None
            self.has_face_recognition = False
        
        # Feature 3: Stampede Prediction
        if self.rbac.has_permission(self.username, Permission.VIEW_STAMPEDE):
            print("\n[Feature 3] Stampede Prediction...")
            self.stampede_predictor = StampedePredictorEnhanced(sequence_length=30, fps=30)
            print("✓ Stampede prediction ENABLED")
            self.has_stampede = True
        else:
            print("\n[Feature 3] Stampede Prediction - ACCESS DENIED")
            self.stampede_predictor = None
            self.has_stampede = False
        
        # Feature 4: Emergency Detection
        if self.rbac.has_permission(self.username, Permission.VIEW_EMERGENCY):
            print("\n[Feature 4] Emergency Detection...")
            self.emergency_detector = EmergencyDetectorEnhanced(
                fall_angle_threshold=60,
                immobility_seconds=300,
                fps=30
            )
            print("✓ Emergency detection ENABLED")
            self.has_emergency = True
        else:
            print("\n[Feature 4] Emergency Detection - ACCESS DENIED")
            self.emergency_detector = None
            self.has_emergency = False
    
    def process_frame(self, frame):
        """Process frame with RBAC-controlled features"""
        self.frame_count += 1
        
        results = {
            'frame_number': self.frame_count,
            'timestamp': datetime.now(),
            'user': self.username,
            'gps_location': {'lat': self.gps_lat, 'lon': self.gps_lon},
            'alerts': []
        }
        
        output_frame = frame.copy()
        crowd_count = 0
        zones = []
        
        # FEATURE 1: Heat Map (if permitted)
        if self.has_heatmap:
            try:
                heatmap_frame, crowd_count, risk_level, heatmap, zones = \
                    self.crowd_analyzer.analyze_crowd(frame, draw_visualization=True)
                output_frame = heatmap_frame
                results['crowd'] = {
                    'count': crowd_count,
                    'risk_level': risk_level,
                    'stampede_zones': len(zones)
                }
            except Exception as e:
                print(f"Heat map error: {e}")
        
        # FEATURE 2: Face Recognition (if permitted)
        if self.has_face_recognition:
            try:
                output, detections, suspects = \
                    self.face_recognition.process_frame(
                        output_frame,
                        latitude=self.gps_lat,
                        longitude=self.gps_lon,
                        draw_boxes=True
                    )
                output_frame = output
                
                for suspect in suspects:
                    results['alerts'].append({
                        'type': 'SUSPECT_DETECTED',
                        'severity': 'critical',
                        'message': f"SUSPECT: {suspect['name']}",
                        'feature': 'Feature 2',
                        'suspect_name': suspect['name'],
                        'confidence': suspect['confidence']
                    })
                    print(f"\n🚨 SUSPECT: {suspect['name']} ({suspect['confidence']:.0%})")
            except Exception as e:
                print(f"Face recognition error: {e}")
        
        # FEATURE 3: Stampede Prediction (if permitted)
        if self.has_stampede and crowd_count > 0:
            try:
                warning = self.stampede_predictor.generate_early_warning(crowd_count, zones)
                if warning and warning['risk_level'] in ['high', 'critical']:
                    results['stampede_warning'] = warning
                    if warning['risk_level'] == 'critical':
                        cv2.rectangle(output_frame, (10, 200), (650, 320), (0, 0, 255), -1)
                        cv2.putText(output_frame, "STAMPEDE IN 90 SECONDS!",
                                   (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            except Exception as e:
                print(f"Stampede prediction error: {e}")
        
        # FEATURE 4: Emergency Detection (if permitted)
        if self.has_emergency:
            try:
                emergency_frame, emergency_detected, emergency_info = \
                    self.emergency_detector.detect_emergency(output_frame, self.frame_count, draw_visualization=True)
                output_frame = emergency_frame
                
                if emergency_detected:
                    results['alerts'].append({
                        'type': 'MEDICAL_EMERGENCY',
                        'severity': 'critical',
                        'message': f"Medical emergency! {emergency_info['immobile_duration_seconds']/60:.1f} min"
                    })
            except Exception as e:
                print(f"Emergency detection error: {e}")
        
        # Add user info and GPS to frame
        cv2.putText(output_frame, f"User: {self.username}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, f"GPS: {self.gps_lat:.4f}, {self.gps_lon:.4f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Frame: {self.frame_count}",
                   (10, output_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame, results
    
    def run(self, video_source=0):
        """Run the system"""
        print(f"\n🎥 Opening video source: {video_source}\n")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Cannot open video source")
            return
        
        print(f"▶ System running as: {self.username}")
        print(f"📍 GPS: {self.gps_lat}, {self.gps_lon}")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed, results = self.process_frame(frame)
                
                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count} | Alerts: {len(results['alerts'])}")
                
                cv2.imshow('SkyGuard RBAC', processed)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏸ Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db_session.close()
            print(f"\n✓ Processed {self.frame_count} frames")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SkyGuard with Blockchain RBAC')
    parser.add_argument('--user', type=str, required=True, help='Username')
    parser.add_argument('--video', type=str, default='0', help='Video source')
    parser.add_argument('--lat', type=float, default=19.0760, help='GPS Latitude')
    parser.add_argument('--lon', type=float, default=72.8777, help='GPS Longitude')
    args = parser.parse_args()
    
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    system = SkyGuardRBAC(username=args.user, gps_lat=args.lat, gps_lon=args.lon)
    system.run(video_source)
