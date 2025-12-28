
#!/usr/bin/env python3
"""
SkyGuard Optimized System - High Accuracy Version
All 4 features with enhanced accuracy, database logging, and GPS-based admin alerts
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

class SkyGuardOptimized:
    """Optimized SkyGuard System with High Accuracy and GPS Alerts"""
    
    def __init__(self, gps_lat=19.0760, gps_lon=72.8777):
        print("\n" + "="*70)
        print("🛡️  SKYGUARD OPTIMIZED SYSTEM")
        print("="*70)
        
        # GPS coordinates (Mumbai by default)
        self.gps_lat = gps_lat
        self.gps_lon = gps_lon
        
        # Database
        print("\n[Database] Connecting...")
        self.db_engine = init_database('data/database/skyguard.db')
        self.db_session = get_session(self.db_engine)
        print("✓ Database connected")
        
        # Feature 1: Optimized Crowd Analysis
        print("\n[Feature 1] Crowd Density Heat Map...")
        self.crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
        print("✓ Heat map ready (Optimized HOG)")
        
        # Feature 2: Optimized Face Recognition with GPS Alerts
        print("\n[Feature 2] Face Recognition with Admin Alerts...")
        self.face_recognition = FaceRecognitionEnhanced(confidence_threshold=0.85)
        print(f"✓ Face recognition ready ({len(self.face_recognition.suspects)} suspects)")
        print(f"✓ Admin alert system active ({len(self.face_recognition.alert_system.config.get('admins', []))} admins)")
        
        # Feature 3: Stampede Prediction
        print("\n[Feature 3] Stampede Prediction...")
        self.stampede_predictor = StampedePredictorEnhanced(sequence_length=30, fps=30)
        print("✓ Stampede prediction ready (90-sec warning)")
        
        # Feature 4: Emergency Detection
        print("\n[Feature 4] Emergency Detection...")
        self.emergency_detector = EmergencyDetectorEnhanced(
            fall_angle_threshold=60,
            immobility_seconds=300,
            fps=30
        )
        print("✓ Emergency detection ready (5-min threshold)")
        
        self.frame_count = 0
        self.start_time = datetime.now()
        
        print("\n" + "="*70)
        print("✅ ALL SYSTEMS READY - HIGH ACCURACY MODE")
        print(f"📍 GPS Location: {self.gps_lat}, {self.gps_lon}")
        print("="*70 + "\n")
    
    def update_gps(self, latitude, longitude):
        """Update GPS coordinates (for drone movement)"""
        self.gps_lat = latitude
        self.gps_lon = longitude
    
    def log_to_database(self, results):
        """Log results to database"""
        try:
            if self.frame_count % 30 == 0 and 'crowd' in results:
                crowd = results['crowd']
                add_crowd_analytics(
                    self.db_session,
                    crowd['count'],
                    crowd['risk_level'],
                    self.gps_lat,
                    self.gps_lon,
                    predicted=results.get('stampede_warning', {}).get('predicted_density'),
                    anomaly=crowd.get('stampede_zones', 0) > 0,
                    anomaly_type='stampede_zone' if crowd.get('stampede_zones', 0) > 0 else 'normal'
                )
            
            if 'faces' in results:
                for alert in results['alerts']:
                    if alert['type'] == 'SUSPECT_DETECTED':
                        add_detected_suspect(
                            self.db_session,
                            alert['suspect_name'],
                            alert['confidence'],
                            self.gps_lat,
                            self.gps_lon,
                            f"frame_{self.frame_count}.jpg"
                        )
            
            if results.get('emergency', {}).get('detected'):
                emergency_info = results['emergency']
                add_emergency_event(
                    self.db_session,
                    'medical_emergency',
                    self.gps_lat,
                    self.gps_lon,
                    'critical',
                    f"emergency_{self.frame_count}.jpg",
                    f"Person immobile for {emergency_info['duration_seconds']/60:.1f} minutes"
                )
        
        except Exception as e:
            print(f"Database logging error: {e}")
    
    def process_frame(self, frame):
        """Process frame with all 4 features and GPS tracking"""
        self.frame_count += 1
        
        results = {
            'frame_number': self.frame_count,
            'timestamp': datetime.now(),
            'gps_location': {'lat': self.gps_lat, 'lon': self.gps_lon},
            'alerts': []
        }
        
        output_frame = frame.copy()
        crowd_count = 0
        zones = []
        
        # FEATURE 1: Density Heat Map
        try:
            heatmap_frame, crowd_count, risk_level, heatmap, zones = \
                self.crowd_analyzer.analyze_crowd(frame, draw_visualization=True)
            
            output_frame = heatmap_frame
            
            results['crowd'] = {
                'count': crowd_count,
                'risk_level': risk_level,
                'stampede_zones': len(zones)
            }
            
            if zones:
                for zone in zones:
                    results['alerts'].append({
                        'type': 'STAMPEDE_ZONE',
                        'severity': 'critical',
                        'message': f"High-risk zone: {zone['area']} px²",
                        'feature': 'Feature 1'
                    })
        
        except Exception as e:
            print(f"Crowd analysis error: {e}")
        
        # FEATURE 2: Face Recognition with GPS-based Admin Alerts
        try:
            # Pass GPS coordinates to face recognition system
            output, detections, suspects = \
                self.face_recognition.process_frame(
                    output_frame, 
                    latitude=self.gps_lat,   # Current GPS latitude
                    longitude=self.gps_lon,  # Current GPS longitude
                    draw_boxes=True
                )
            
            output_frame = output
            
            results['faces'] = {
                'detections': len(detections),
                'suspects': len(suspects)
            }
            
            for suspect in suspects:
                results['alerts'].append({
                    'type': 'SUSPECT_DETECTED',
                    'severity': 'critical',
                    'message': f"SUSPECT: {suspect['name']}",
                    'feature': 'Feature 2',
                    'suspect_name': suspect['name'],
                    'confidence': suspect['confidence'],
                    'gps': {'lat': self.gps_lat, 'lon': self.gps_lon}
                })
                
                # Console alert with GPS
                print(f"\n{'='*70}")
                print(f"🚨 SUSPECT DETECTED: {suspect['name']} ({suspect['confidence']:.0%})")
                print(f"📍 GPS Location: {self.gps_lat}, {self.gps_lon}")
                print(f"🗺️  Google Maps: https://maps.google.com?q={self.gps_lat},{self.gps_lon}")
                print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"📧 Admin alerts sent automatically")
                print(f"{'='*70}\n")
        
        except Exception as e:
            print(f"Face recognition error: {e}")
        
        # FEATURE 3: Stampede Prediction
        if crowd_count > 0:
            try:
                warning = self.stampede_predictor.generate_early_warning(crowd_count, zones)
                
                if warning and warning['risk_level'] in ['high', 'critical']:
                    results['stampede_warning'] = warning
                    results['alerts'].append({
                        'type': 'STAMPEDE_WARNING',
                        'severity': warning['risk_level'],
                        'message': f"Stampede risk! Predicted: {warning['predicted_density']}",
                        'feature': 'Feature 3',
                        'gps': {'lat': self.gps_lat, 'lon': self.gps_lon}
                    })
                    
                    if warning['risk_level'] == 'critical':
                        cv2.rectangle(output_frame, (10, 200), (650, 320), (0, 0, 255), -1)
                        cv2.putText(output_frame, "STAMPEDE IN 90 SECONDS!", 
                                   (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                        cv2.putText(output_frame, f"Predicted: {warning['predicted_density']} people", 
                                   (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            except Exception as e:
                print(f"Stampede prediction error: {e}")
        
        # FEATURE 4: Emergency Detection
        try:
            emergency_frame, emergency_detected, emergency_info = \
                self.emergency_detector.detect_emergency(output_frame, self.frame_count, draw_visualization=True)
            
            output_frame = emergency_frame
            
            results['emergency'] = {
                'detected': emergency_detected,
                'duration_seconds': emergency_info.get('immobile_duration_seconds', 0)
            }
            
            if emergency_detected:
                results['alerts'].append({
                    'type': 'MEDICAL_EMERGENCY',
                    'severity': 'critical',
                    'message': f"Medical emergency! {emergency_info['immobile_duration_seconds']/60:.1f} min",
                    'feature': 'Feature 4',
                    'gps': {'lat': self.gps_lat, 'lon': self.gps_lon}
                })
                print(f"\n🚨 MEDICAL EMERGENCY: {emergency_info['immobile_duration_seconds']/60:.1f} min immobile")
                print(f"📍 Location: {self.gps_lat}, {self.gps_lon}")
        
        except Exception as e:
            print(f"Emergency detection error: {e}")
        
        # Log to database
        self.log_to_database(results)
        
        # Add GPS and frame info to video
        cv2.putText(output_frame, f"GPS: {self.gps_lat:.4f}, {self.gps_lon:.4f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Frame: {self.frame_count} | Alerts: {len(results['alerts'])}", 
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
        
        print("▶ System running... Press 'q' to quit")
        print(f"📍 Monitoring at GPS: {self.gps_lat}, {self.gps_lon}\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed, results = self.process_frame(frame)
                
                if self.frame_count % 30 == 0:
                    crowd = results.get('crowd', {})
                    print(f"Frame {self.frame_count}: People={crowd.get('count', 0)}, "
                          f"Risk={crowd.get('risk_level', 'N/A')}, Alerts={len(results['alerts'])}")
                
                cv2.imshow('SkyGuard Optimized - All Features', processed)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏸ Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db_session.close()
            print(f"\n✓ Processed {self.frame_count} frames")
            print("✓ System shutdown complete\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SkyGuard Optimized System')
    parser.add_argument('--video', type=str, default='0', help='Video source')
    parser.add_argument('--lat', type=float, default=19.0760, help='GPS Latitude')
    parser.add_argument('--lon', type=float, default=72.8777, help='GPS Longitude')
    args = parser.parse_args()
    
    video_source = int(args.video) if args.video.isdigit() else args.video
    
    # Initialize system with GPS coordinates
    system = SkyGuardOptimized(gps_lat=args.lat, gps_lon=args.lon)
    system.run(video_source)
