#!/usr/bin/env python3
"""
SkyGuard Complete Integrated System
All 4 main features working together
"""

import cv2
import numpy as np
from datetime import datetime
import sys
import os

# Import all enhanced modules
try:
    from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
    from models.face_recognition_enhanced import FaceRecognitionEnhanced
    from models.stampede_prediction_enhanced import StampedePredictorEnhanced
    from models.emergency_detection_enhanced import EmergencyDetectorEnhanced
    from models.database import init_database, get_session
    print("✓ All enhanced modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SkyGuardEnhanced:
    """Complete SkyGuard System with All 4 Main Features"""
    
    def __init__(self, config=None):
        """Initialize Complete SkyGuard System"""
        print("\n" + "=" * 70)
        print("🛡️  SKYGUARD ENHANCED SURVEILLANCE SYSTEM")
        print("=" * 70)
        print("\nInitializing all modules...")
        
        self.config = config or {
            'database_path': 'data/database/skyguard.db',
            'suspects_db_path': 'data/suspects',
            'default_gps': {'lat': 19.0760, 'lon': 72.8777},  # Mumbai
            'fps': 30
        }
        
        # Initialize database
        print("\n[Database] Initializing...")
        try:
            self.db_engine = init_database(self.config['database_path'])
            self.db_session = get_session(self.db_engine)
            print("✓ Database ready")
        except Exception as e:
            print(f"⚠ Database warning: {e}")
            self.db_session = None
        
        # Feature 1: Density Heat Map with Stampede Zones
        print("\n[Feature 1] Loading Density Heat Map...")
        try:
            self.crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
            print("✓ Density heat map ready")
            print("  • Red areas = High crowd density (stampede risk)")
            print("  • Yellow/Orange = Medium density")
            print("  • Blue/Green = Low density")
        except Exception as e:
            print(f"❌ Heat map error: {e}")
            self.crowd_analyzer = None
        
        # Feature 2: Face Recognition with Suspect Alerts
        print("\n[Feature 2] Loading Face Recognition...")
        try:
            self.face_recognition = FaceRecognitionEnhanced(
                suspects_db_path=self.config['suspects_db_path']
            )
            print("✓ Face recognition ready")
            print(f"  • Suspects in database: {len(self.face_recognition.suspects)}")
            print("  • Alert system: Active")
        except Exception as e:
            print(f"❌ Face recognition error: {e}")
            self.face_recognition = None
        
        # Feature 3: Stampede Prediction (90-second early warning)
        print("\n[Feature 3] Loading Stampede Prediction...")
        try:
            self.stampede_predictor = StampedePredictorEnhanced(
                sequence_length=30,
                fps=self.config['fps']
            )
            print("✓ Stampede prediction ready")
            print(f"  • Early warning: {self.stampede_predictor.prediction_horizon_seconds} seconds")
            print("  • AI model: LSTM neural network")
        except Exception as e:
            print(f"❌ Stampede prediction error: {e}")
            self.stampede_predictor = None
        
        # Feature 4: Medical Emergency Detection (5-minute immobility)
        print("\n[Feature 4] Loading Emergency Detection...")
        try:
            self.emergency_detector = EmergencyDetectorEnhanced(
                fall_angle_threshold=60,
                immobility_seconds=300,  # 5 minutes
                fps=self.config['fps']
            )
            print("✓ Emergency detection ready")
            print(f"  • Body angle monitoring: Active")
            print(f"  • Emergency threshold: 5 minutes immobility")
        except Exception as e:
            print(f"❌ Emergency detection error: {e}")
            self.emergency_detector = None
        
        self.frame_count = 0
        self.start_time = datetime.now()
        self.alerts_history = []
        
        print("\n" + "=" * 70)
        print("✅ SKYGUARD SYSTEM READY!")
        print("=" * 70)
        print("\nActive Features:")
        print(f"  [1] Density Heat Map: {'✓' if self.crowd_analyzer else '✗'}")
        print(f"  [2] Face Recognition: {'✓' if self.face_recognition else '✗'}")
        print(f"  [3] Stampede Prediction: {'✓' if self.stampede_predictor else '✗'}")
        print(f"  [4] Emergency Detection: {'✓' if self.emergency_detector else '✗'}")
        print()
    
    def process_frame(self, frame, gps_location=None):
        """
        Process frame through all 4 features
        
        Returns:
            processed_frame, results
        """
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
        
        # FEATURE 1: Density Heat Map
        crowd_count = 0
        stampede_zones = []
        
        if self.crowd_analyzer:
            try:
                heatmap_frame, crowd_count, risk_level, heatmap, zones = \
                    self.crowd_analyzer.analyze_crowd(frame, draw_visualization=True)
                
                output_frame = heatmap_frame
                stampede_zones = zones
                
                results['crowd'] = {
                    'count': crowd_count,
                    'risk_level': risk_level,
                    'stampede_zones': len(zones)
                }
                
                # Alert for critical zones
                if zones:
                    for zone in zones:
                        results['alerts'].append({
                            'type': 'STAMPEDE_ZONE',
                            'severity': 'critical',
                            'message': f"High-risk stampede zone detected! Area: {zone['area']} px²",
                            'feature': 'Feature 1: Heat Map'
                        })
                
            except Exception as e:
                print(f"Crowd analysis error: {e}")
        
        # FEATURE 2: Face Recognition
        if self.face_recognition:
            try:
                face_frame, detections, suspects = \
                    self.face_recognition.process_frame(output_frame, draw_boxes=True)
                
                output_frame = face_frame
                
                results['faces'] = {
                    'detections': len(detections),
                    'suspects': len(suspects)
                }
                
                # Alert for detected suspects
                for suspect in suspects:
                    alert = {
                        'type': 'SUSPECT_DETECTED',
                        'severity': 'critical',
                        'message': f"⚠ SUSPECT: {suspect['name']} (Confidence: {suspect['confidence']:.0%})",
                        'feature': 'Feature 2: Face Recognition',
                        'suspect_name': suspect['name'],
                        'confidence': suspect['confidence']
                    }
                    results['alerts'].append(alert)
                    
                    # Log to console
                    print(f"\n🚨 ALERT: Suspect '{suspect['name']}' detected!")
                    print(f"   Confidence: {suspect['confidence']:.0%}")
                    print(f"   Location: Frame {self.frame_count}")
                
            except Exception as e:
                print(f"Face recognition error: {e}")
        
        # FEATURE 3: Stampede Prediction (90-second warning)
        if self.stampede_predictor and crowd_count > 0:
            try:
                warning = self.stampede_predictor.generate_early_warning(
                    crowd_count, stampede_zones
                )
                
                if warning:
                    results['stampede_warning'] = warning
                    
                    # Alert for high/critical risk
                    if warning['risk_level'] in ['high', 'critical']:
                        alert = {
                            'type': 'STAMPEDE_WARNING',
                            'severity': warning['risk_level'],
                            'message': f"Stampede prediction: {warning['predicted_density']} people (Risk: {warning['risk_level'].upper()})",
                            'feature': 'Feature 3: Stampede Prediction',
                            'time_to_critical': warning.get('time_to_critical'),
                            'actions': warning['recommended_actions']
                        }
                        results['alerts'].append(alert)
                        
                        # Draw warning on frame
                        if warning['risk_level'] == 'critical':
                            cv2.rectangle(output_frame, (10, 200), (650, 320), (0, 0, 255), -1)
                            cv2.putText(output_frame, "⚠ STAMPEDE RISK IN 90 SECONDS!", 
                                       (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            cv2.putText(output_frame, f"Predicted: {warning['predicted_density']} people", 
                                       (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(output_frame, "INITIATE EVACUATION PROCEDURES!", 
                                       (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Stampede prediction error: {e}")
        
        # FEATURE 4: Medical Emergency Detection (5-minute check)
        if self.emergency_detector:
            try:
                emergency_frame, emergency_detected, emergency_info = \
                    self.emergency_detector.detect_emergency(
                        output_frame, self.frame_count, draw_visualization=True
                    )
                
                output_frame = emergency_frame
                
                results['emergency'] = {
                    'detected': emergency_detected,
                    'immobile': emergency_info['immobile'],
                    'duration_seconds': emergency_info['immobile_duration_seconds']
                }
                
                # Alert for confirmed emergency (5 minutes immobile)
                if emergency_detected:
                    alert = {
                        'type': 'MEDICAL_EMERGENCY',
                        'severity': 'critical',
                        'message': f"🚨 MEDICAL EMERGENCY! Person immobile for {emergency_info['immobile_duration_seconds']/60:.1f} minutes",
                        'feature': 'Feature 4: Emergency Detection',
                        'duration_minutes': emergency_info['immobile_duration_seconds'] / 60
                    }
                    results['alerts'].append(alert)
                    
                    print(f"\n🚨 MEDICAL EMERGENCY DETECTED!")
                    print(f"   Duration: {emergency_info['immobile_duration_seconds']/60:.1f} minutes")
                    print(f"   Frame: {self.frame_count}")
                
                # Warning for potential emergency
                elif emergency_info['immobile'] and emergency_info['time_remaining_seconds']:
                    if emergency_info['time_remaining_seconds'] < 60:
                        alert = {
                            'type': 'POTENTIAL_EMERGENCY',
                            'severity': 'high',
                            'message': f"⚠ Person lying down for {emergency_info['immobile_duration_seconds']:.0f}s. Emergency in {emergency_info['time_remaining_seconds']:.0f}s",
                            'feature': 'Feature 4: Emergency Detection'
                        }
                        results['alerts'].append(alert)
                
            except Exception as e:
                print(f"Emergency detection error: {e}")
        
        # Add frame counter
        cv2.putText(output_frame, f"Frame: {self.frame_count}", 
                   (10, output_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame, results
    
    def process_video(self, video_source, display=True):
        """Process video from camera or file"""
        print(f"\n{'='*70}")
        print(f"🎥 Opening video source: {video_source}")
        print(f"{'='*70}")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Video opened successfully")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"\n▶ Processing started...")
        print(f"  Press 'q' to quit")
        print(f"  Press 's' to take screenshot")
        print()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ End of video")
                    break
                
                # Process frame through all 4 features
                processed_frame, results = self.process_frame(frame)
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0:
                    crowd_info = results.get('crowd', {})
                    face_info = results.get('faces', {})
                    
                    print(f"Frame {self.frame_count}: "
                          f"People={crowd_info.get('count', 0)}, "
                          f"Risk={crowd_info.get('risk_level', 'N/A')}, "
                          f"Faces={face_info.get('detections', 0)}, "
                          f"Alerts={len(results['alerts'])}")
                
                # Print all alerts
                for alert in results['alerts']:
                    print(f"\n{'='*70}")
                    print(f"🚨 {alert['type']} - {alert['severity'].upper()}")
                    print(f"   {alert['message']}")
                    print(f"   Source: {alert['feature']}")
                    
                    if alert['type'] == 'STAMPEDE_WARNING' and 'actions' in alert:
                        print(f"   Recommended Actions:")
                        for action in alert['actions'][:3]:
                            print(f"      • {action}")
                    
                    print(f"{'='*70}")
                
                # Display
                if display:
                    cv2.imshow('SkyGuard Enhanced - All Features Active', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏸ User requested stop")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        filename = f"skyguard_screenshot_{self.frame_count}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"📸 Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n⏸ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n{'='*70}")
            print(f"✓ Processing Complete")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Duration: {(datetime.now() - self.start_time).seconds}s")
            print(f"{'='*70}\n")
    
    def shutdown(self):
        """Cleanup"""
        try:
            if self.db_session:
                self.db_session.close()
        except:
            pass
        print("✓ SkyGuard system shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SkyGuard Enhanced Surveillance System - All 4 Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  [1] Density Heat Map - Red areas show stampede-prone zones
  [2] Face Recognition - Identifies suspects and alerts security
  [3] Stampede Prediction - 90-second early warning system
  [4] Emergency Detection - 5-minute immobility check

Examples:
  python3 main_enhanced.py --video 0              # Webcam
  python3 main_enhanced.py --video video.mp4      # Video file
        """
    )
    parser.add_argument('--video', type=str, default='0',
                       help='Video source: 0 for webcam or video file path')
    
    args = parser.parse_args()
    
    # Convert to int if it's a digit
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    # Initialize system
    skyguard = SkyGuardEnhanced()
    
    try:
        # Process video
        skyguard.process_video(video_source=video_source, display=True)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        skyguard.shutdown()


if __name__ == '__main__':
    main()
