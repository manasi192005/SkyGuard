#!/usr/bin/env python3
"""
SkyGuard with Full Blockchain Integration
Face detection alerts sent only to blockchain-authorized users
"""

import cv2
from datetime import datetime
import sys

from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
from models.face_recognition_rbac import FaceRecognitionRBAC
from models.stampede_prediction_enhanced import StampedePredictorEnhanced
from models.emergency_detection_enhanced import EmergencyDetectorEnhanced
from models.database import init_database, get_session

from blockchain import BlockchainRBAC, Permission

class SkyGuardBlockchain:
    """SkyGuard with Full Blockchain Integration"""
    
    def __init__(self, username, gps_lat=19.0760, gps_lon=72.8777):
        print("\n" + "="*70)
        print("🛡️  SKYGUARD - FULL BLOCKCHAIN INTEGRATION")
        print("="*70)
        
        # Initialize blockchain
        self.rbac = BlockchainRBAC()
        self.username = username
        
        # Verify user
        user_role = self.rbac.get_user_role(username)
        if not user_role:
            print(f"❌ User '{username}' not found in blockchain")
            sys.exit(1)
        
        print(f"\n✓ User: {username}")
        print(f"✓ Role: {user_role.name}")
        
        permissions = self.rbac.get_user_permissions(username)
        print(f"✓ Permissions: {len(permissions)}")
        for perm in permissions:
            print(f"   • {perm.value}")
        
        self.gps_lat = gps_lat
        self.gps_lon = gps_lon
        
        # Database
        self.db_engine = init_database('data/database/skyguard.db')
        self.db_session = get_session(self.db_engine)
        
        # Initialize features
        self.initialize_features()
        
        self.frame_count = 0
        
        print("\n" + "="*70)
        print("✅ BLOCKCHAIN SECURITY ENABLED")
        print("="*70 + "\n")
    
    def initialize_features(self):
        """Initialize features based on permissions"""
        
        # Feature 1: Heat Map
        if self.rbac.has_permission(self.username, Permission.VIEW_HEATMAP):
            print("\n[Feature 1] Heat Map - ENABLED")
            self.crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
            self.has_heatmap = True
        else:
            print("\n[Feature 1] Heat Map - ACCESS DENIED")
            self.crowd_analyzer = None
            self.has_heatmap = False
        
        # Feature 2: Face Recognition with Blockchain Alerts
        if self.rbac.has_permission(self.username, Permission.VIEW_SUSPECTS):
            print("\n[Feature 2] Face Recognition - ENABLED")
            self.face_recognition = FaceRecognitionRBAC(confidence_threshold=0.85)
            print(f"✓ Blockchain-secured alerts active")
            print(f"✓ Suspects in database: {len(self.face_recognition.suspects)}")
            
            # Show who will receive alerts
            authorized = self.face_recognition.alert_system.get_authorized_users_for_suspects()
            print(f"✓ Authorized alert recipients: {len(authorized)}")
            for user in authorized:
                print(f"   • {user['username']} ({user['role']})")
            
            self.has_face_recognition = True
        else:
            print("\n[Feature 2] Face Recognition - ACCESS DENIED")
            self.face_recognition = None
            self.has_face_recognition = False
        
        # Feature 3: Stampede
        if self.rbac.has_permission(self.username, Permission.VIEW_STAMPEDE):
            print("\n[Feature 3] Stampede Prediction - ENABLED")
            self.stampede_predictor = StampedePredictorEnhanced(sequence_length=30, fps=30)
            self.has_stampede = True
        else:
            print("\n[Feature 3] Stampede Prediction - ACCESS DENIED")
            self.stampede_predictor = None
            self.has_stampede = False
        
        # Feature 4: Emergency
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
    
    def process_frame(self, frame):
        """Process frame with blockchain security"""
        self.frame_count += 1
        
        output_frame = frame.copy()
        crowd_count = 0
        
        # Heat map
        if self.has_heatmap:
            try:
                heatmap_frame, crowd_count, risk_level, heatmap, zones = \
                    self.crowd_analyzer.analyze_crowd(frame, draw_visualization=True)
                output_frame = heatmap_frame
            except Exception as e:
                pass
        
        # Face recognition with blockchain-secured alerts
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
                
                # Blockchain-secured alerts already sent in process_frame
            except Exception as e:
                print(f"Face recognition error: {e}")
        
        # Add blockchain status
        cv2.putText(output_frame, f"User: {self.username} (Blockchain-Secured)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, f"GPS: {self.gps_lat:.4f}, {self.gps_lon:.4f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output_frame
    
    def run(self, video_source=0):
        """Run the system"""
        print(f"\n🎥 Opening video: {video_source}\n")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Cannot open video")
            return
        
        print(f"▶ Running as: {self.username}")
        print(f"📍 GPS: {self.gps_lat}, {self.gps_lon}")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed = self.process_frame(frame)
                
                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count}")
                
                cv2.imshow('SkyGuard Blockchain', processed)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏸ Stopped")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db_session.close()
            print(f"\n✓ Session complete")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SkyGuard with Full Blockchain Integration')
    parser.add_argument('--user', type=str, required=True, help='Username')
    parser.add_argument('--video', type=str, default='0', help='Video source')
    parser.add_argument('--lat', type=float, default=19.0760, help='GPS Latitude')
    parser.add_argument('--lon', type=float, default=72.8777, help='GPS Longitude')
    args = parser.parse_args()
    
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    system = SkyGuardBlockchain(username=args.user, gps_lat=args.lat, gps_lon=args.lon)
    system.run(video_source)
