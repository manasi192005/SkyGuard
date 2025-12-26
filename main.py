
"""
SkyGuard Main Integration System
Coordinates all detection modules and provides unified interface
"""

import cv2
import numpy as np
from datetime import datetime
import sys
import os

# Import all modules
try:
    from models.face_recognition import FaceRecognitionSystem
    from models.crowd_analysis_enhanced import CrowdAnalyzer
    from models.violence_detection import ViolenceDetector, AnomalyDetector
    from models.fall_detection import FallDetector, EmergencyResponseSystem
    from models.stampede_prediction import StampedePredictor
    from models.database import init_database, get_session
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SkyGuardSystem:
    """Main SkyGuard surveillance system"""
    
    def __init__(self, config=None):
        """Initialize SkyGuard System"""
        print("\n" + "=" * 60)
        print("Initializing SkyGuard System...")
        print("=" * 60)
        
        self.config = config or {
            'face_recognition_enabled': True,
            'crowd_analysis_enabled': True,
            'violence_detection_enabled': True,
            'fall_detection_enabled': True,
            'stampede_prediction_enabled': True,
            'suspects_db_path': 'data/suspects',
            'database_path': 'data/database/skyguard.db',
            'default_gps': {'lat': 19.0760, 'lon': 72.8777}
        }
        
        # Initialize database
        print("\n[1/6] Initializing database...")
        try:
            self.db_engine = init_database(self.config['database_path'])
            self.db_session = get_session(self.db_engine)
            print("✓ Database initialized")
        except Exception as e:
            print(f"❌ Database error: {e}")
            raise
        
        # Initialize modules
        self.modules = {}
        
        if self.config['face_recognition_enabled']:
            print("\n[2/6] Loading Face Recognition module...")
            try:
                self.modules['face_recognition'] = FaceRecognitionSystem(
                    suspects_db_path=self.config['suspects_db_path']
                )
                print("✓ Face Recognition loaded")
            except Exception as e:
                print(f"⚠ Face Recognition failed: {e}")
        
        if self.config['crowd_analysis_enabled']:
            print("\n[3/6] Loading Crowd Analysis module...")
            try:
                self.modules['crowd_analysis'] = CrowdAnalyzer(use_simple_detection=True)
                print("✓ Crowd Analysis loaded")
            except Exception as e:
                print(f"⚠ Crowd Analysis failed: {e}")
        
        if self.config['violence_detection_enabled']:
            print("\n[4/6] Loading Violence Detection module...")
            try:
                self.modules['violence_detection'] = ViolenceDetector()
                self.modules['anomaly_detection'] = AnomalyDetector()
                print("✓ Violence Detection loaded")
            except Exception as e:
                print(f"⚠ Violence Detection failed: {e}")
        
        if self.config['fall_detection_enabled']:
            print("\n[5/6] Loading Fall Detection module...")
            try:
                self.modules['fall_detection'] = EmergencyResponseSystem()
                print("✓ Fall Detection loaded")
            except Exception as e:
                print(f"⚠ Fall Detection failed: {e}")
        
        if self.config['stampede_prediction_enabled']:
            print("\n[6/6] Loading Stampede Prediction module...")
            try:
                self.modules['stampede_prediction'] = StampedePredictor()
                print("✓ Stampede Prediction loaded")
            except Exception as e:
                print(f"⚠ Stampede Prediction failed: {e}")
        
        self.frame_count = 0
        self.start_time = datetime.now()
        
        print("\n" + "=" * 60)
        print("SkyGuard System Ready!")
        print("=" * 60 + "\n")
    
    def process_frame(self, frame, gps_location=None):
        """Process a single frame through all enabled modules"""
        if gps_location is None:
            gps_location = self.config['default_gps']
        
        self.frame_count += 1
        results = {
            'frame_number': self.frame_count,
            'timestamp': datetime.now(),
            'gps_location': gps_location,
            'alerts': []
        }
        
        output_frame = frame.copy()
        crowd_count = 0
        
        # 1. Crowd Analysis
        if 'crowd_analysis' in self.modules:
            try:
                crowd_frame, crowd_count, risk_level, density_map = \
                    self.modules['crowd_analysis'].analyze_crowd(frame, draw_visualization=True)
                
                results['crowd'] = {
                    'count': crowd_count,
                    'risk_level': risk_level
                }
                output_frame = crowd_frame
            except Exception as e:
                print(f"Crowd analysis error: {e}")
        
        # 2. Stampede Prediction
        if 'stampede_prediction' in self.modules and crowd_count > 0:
            try:
                warning = self.modules['stampede_prediction'].generate_early_warning(crowd_count)
                
                if warning and warning['risk_level'] in ['high', 'critical']:
                    results['alerts'].append({
                        'type': 'stampede_risk',
                        'severity': warning['risk_level'],
                        'message': f"Stampede risk! Predicted: {warning['predicted_density']}",
                    })
            except Exception as e:
                print(f"Stampede prediction error: {e}")
        
        # 3. Violence Detection
        if 'violence_detection' in self.modules:
            try:
                _, violence_detected, v_score, motion_type = \
                    self.modules['violence_detection'].detect_violence(frame, draw_visualization=False)
                
                results['violence'] = {
                    'detected': violence_detected,
                    'score': v_score
                }
                
                if violence_detected:
                    results['alerts'].append({
                        'type': 'violence',
                        'severity': 'high',
                        'message': f"Violence detected! Type: {motion_type}",
                    })
            except Exception as e:
                print(f"Violence detection error: {e}")
        
        # Draw alerts on frame
        y_pos = 30
        for alert in results.get('alerts', []):
            color = (0, 0, 255) if alert['severity'] == 'critical' else (0, 165, 255)
            cv2.putText(output_frame, f"ALERT: {alert['message']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30
        
        # Show frame number
        cv2.putText(output_frame, f"Frame: {self.frame_count}", 
                   (10, output_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame, results
    
    def process_video(self, video_source, display=True):
        """Process video from file or camera"""
        print(f"\n🎥 Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source: {video_source}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Video opened successfully")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print("\n▶ Processing started... Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ End of video or camera disconnected")
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Print alerts
                if results.get('alerts'):
                    for alert in results['alerts']:
                        print(f"🚨 [ALERT] {alert['type']}: {alert['message']}")
                
                # Display every 30 frames
                if self.frame_count % 30 == 0:
                    crowd_info = results.get('crowd', {})
                    print(f"Frame {self.frame_count}: Crowd={crowd_info.get('count', 0)}, "
                          f"Risk={crowd_info.get('risk_level', 'unknown')}")
                
                # Display frame
                if display:
                    cv2.imshow('SkyGuard System', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏸ User requested stop")
                        break
        
        except KeyboardInterrupt:
            print("\n⏸ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n✓ Processed {self.frame_count} frames")
    
    def shutdown(self):
        """Cleanup and shutdown system"""
        try:
            self.db_session.close()
            print("✓ Database connection closed")
        except:
            pass
        print("✓ SkyGuard System shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SkyGuard Surveillance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py --video 0              # Use webcam
  python3 main.py --video video.mp4      # Process video file
        """
    )
    parser.add_argument('--video', type=str, default='0', 
                       help='Video file path or camera index (default: 0 for webcam)')
    
    args = parser.parse_args()
    
    # Convert to int if it's a digit (camera index)
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    print("\n🛡️  SkyGuard Surveillance System")
    print("=" * 60)
    
    # Initialize system
    skyguard = SkyGuardSystem()
    
    try:
        # Process video
        skyguard.process_video(video_source=video_source, display=True)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        skyguard.shutdown()
    
    print("\n" + "=" * 60)
    print("Thank you for using SkyGuard!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
