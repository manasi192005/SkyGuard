"""
SkyGuard Thermal Weapon Detection Module
Integrates thermal weapon detection into existing SkyGuard system
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json

class SkyGuardThermalModule:
    """
    Main thermal weapon detection module for SkyGuard
    Integrates with existing suspect tracking system
    """
    
    def __init__(self, model_path='best.pt', enable_alerts=True):
        """
        Initialize thermal detection module
        
        Args:
            model_path: Path to trained thermal weapon detection model
            enable_alerts: Whether to send alerts for detections
        """
        self.model = YOLO(model_path)
        self.enable_alerts = enable_alerts
        
        # Detection thresholds
        self.CONFIDENCE_THRESHOLD = 0.75  # 75% minimum for weapon detection
        self.CRITICAL_THRESHOLD = 0.85    # 85%+ triggers immediate alert
        
        # Thermal simulator (for demo without real thermal camera)
        from thermal_simulator import ThermalSimulator
        self.thermal_sim = ThermalSimulator()
        
        # Alert system
        self.alerts = []
        self.detection_log = []
        
    def process_suspect_frame(self, frame, suspect_id, suspect_bbox):
        """
        Analyzes a suspect's frame for concealed weapons
        
        Args:
            frame: RGB video frame
            suspect_id: ID of identified suspect
            suspect_bbox: (x, y, w, h) bounding box of suspect
            
        Returns:
            dict with detection results
        """
        # Step 1: Extract suspect ROI
        x, y, w, h = suspect_bbox
        suspect_roi = frame[y:y+h, x:x+w]
        
        # Step 2: Convert to thermal simulation
        thermal_frame = self.thermal_sim.create_thermal_effect(suspect_roi)
        
        # Step 3: Run weapon detection on thermal frame
        detections = self._detect_weapons(thermal_frame)
        
        # Step 4: Analyze and classify threat
        threat_assessment = self._assess_threat(detections, suspect_id)
        
        # Step 5: Generate alert if needed
        if threat_assessment['threat_level'] in ['CRITICAL', 'HIGH']:
            self._generate_alert(suspect_id, threat_assessment, thermal_frame)
        
        # Step 6: Log detection
        self._log_detection(suspect_id, threat_assessment)
        
        return {
            'suspect_id': suspect_id,
            'weapons_detected': len(detections),
            'detections': detections,
            'threat_assessment': threat_assessment,
            'thermal_frame': thermal_frame
        }
    
    def _detect_weapons(self, thermal_frame):
        """Runs YOLO detection on thermal frame"""
        results = self.model.predict(
            thermal_frame,
            conf=self.CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    'weapon_type': class_name,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'location': self._determine_body_location(
                        int(y1), thermal_frame.shape[0]
                    )
                })
        
        return detections
    
    def _determine_body_location(self, y_pos, frame_height):
        """Determines where on body the weapon is concealed"""
        relative_pos = y_pos / frame_height
        
        if relative_pos < 0.3:
            return "UPPER_BODY"  # Shoulder/chest holster
        elif relative_pos < 0.6:
            return "WAIST"       # Most common - waistband
        elif relative_pos < 0.8:
            return "THIGH"       # Thigh holster
        else:
            return "ANKLE"       # Ankle holster
    
    def _assess_threat(self, detections, suspect_id):
        """
        Assesses overall threat level based on detections
        
        Threat levels:
        - CRITICAL: High confidence weapon detection (>85%)
        - HIGH: Medium-high confidence (75-85%)
        - MEDIUM: Lower confidence or knife detected
        - LOW: No weapons or very low confidence
        - CLEAR: No detections
        """
        if not detections:
            return {
                'threat_level': 'CLEAR',
                'confidence': 0.0,
                'primary_weapon': None,
                'weapon_count': 0,
                'recommendation': 'Continue monitoring'
            }
        
        # Get highest confidence detection
        primary_detection = max(detections, key=lambda x: x['confidence'])
        
        # Calculate threat level
        conf = primary_detection['confidence']
        weapon_type = primary_detection['weapon_type']
        
        if conf >= self.CRITICAL_THRESHOLD:
            threat_level = 'CRITICAL'
            recommendation = '🚨 IMMEDIATE RESPONSE REQUIRED - Armed suspect detected'
        elif conf >= self.CONFIDENCE_THRESHOLD:
            threat_level = 'HIGH'
            recommendation = '⚠️ Deploy ground units for verification'
        elif weapon_type == 'knife':
            threat_level = 'MEDIUM'
            recommendation = 'Maintain visual surveillance, prepare response'
        else:
            threat_level = 'LOW'
            recommendation = 'Continue monitoring'
        
        return {
            'threat_level': threat_level,
            'confidence': conf,
            'primary_weapon': weapon_type,
            'weapon_count': len(detections),
            'body_location': primary_detection['location'],
            'recommendation': recommendation,
            'all_detections': detections
        }
    
    def _generate_alert(self, suspect_id, threat_assessment, thermal_frame):
        """Generates and stores critical alert"""
        alert = {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'suspect_id': suspect_id,
            'threat_level': threat_assessment['threat_level'],
            'weapon_type': threat_assessment['primary_weapon'],
            'confidence': threat_assessment['confidence'],
            'location': threat_assessment['body_location'],
            'recommendation': threat_assessment['recommendation'],
            'thermal_snapshot': thermal_frame  # Store for evidence
        }
        
        self.alerts.append(alert)
        
        # Print to console (for demo)
        print("\n" + "="*70)
        print(f"🚨 {alert['threat_level']} ALERT - {alert['alert_id']}")
        print(f"Suspect ID: {suspect_id}")
        print(f"Weapon Detected: {alert['weapon_type'].upper()}")
        print(f"Confidence: {alert['confidence']:.1%}")
        print(f"Location on Body: {alert['location']}")
        print(f"Action: {alert['recommendation']}")
        print("="*70 + "\n")
        
        # Save alert to file
        self._save_alert_to_file(alert)
        
        return alert
    
    def _save_alert_to_file(self, alert):
        """Saves alert data to JSON file"""
        alert_copy = alert.copy()
        alert_copy.pop('thermal_snapshot', None)  # Remove image data for JSON
        
        filename = f"alerts/{alert['alert_id']}.json"
        import os
        os.makedirs('alerts', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(alert_copy, f, indent=2)
        
        # Save thermal snapshot separately
        cv2.imwrite(f"alerts/{alert['alert_id']}_thermal.jpg", 
                    alert['thermal_snapshot'])
    
    def _log_detection(self, suspect_id, threat_assessment):
        """Logs all detections for analysis"""
        self.detection_log.append({
            'timestamp': datetime.now().isoformat(),
            'suspect_id': suspect_id,
            'threat_level': threat_assessment['threat_level'],
            'confidence': threat_assessment['confidence']
        })
    
    def visualize_detection(self, frame, detections, threat_assessment):
        """
        Creates visualization overlay for detection results
        
        Args:
            frame: Original frame
            detections: List of weapon detections
            threat_assessment: Threat assessment dict
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Color scheme based on threat level
        colors = {
            'CRITICAL': (0, 0, 255),    # Red
            'HIGH': (0, 140, 255),      # Orange
            'MEDIUM': (0, 255, 255),    # Yellow
            'LOW': (0, 255, 0),         # Green
            'CLEAR': (128, 128, 128)    # Gray
        }
        
        threat_level = threat_assessment['threat_level']
        color = colors.get(threat_level, (255, 255, 255))
        
        # Draw bounding boxes for each weapon
        for det in detections:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            weapon = det['weapon_type']
            
            # Draw box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)
            
            # Draw label with background
            label = f"{weapon.upper()} {conf:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(annotated, (x, y-label_h-10), 
                         (x+label_w, y), color, -1)
            cv2.putText(annotated, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw threat level banner
        banner_height = 60
        banner = np.zeros((banner_height, annotated.shape[1], 3), dtype=np.uint8)
        banner[:] = color
        
        # Add text to banner
        threat_text = f"THREAT: {threat_level}"
        cv2.putText(banner, threat_text, (10, 25),
                   cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)
        
        if threat_assessment['primary_weapon']:
            weapon_text = f"{threat_assessment['primary_weapon'].upper()} | " \
                         f"{threat_assessment['confidence']:.0%} confidence"
            cv2.putText(banner, weapon_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine banner and frame
        result = np.vstack([banner, annotated])
        
        return result
    
    def get_alert_summary(self):
        """Returns summary of all alerts generated"""
        return {
            'total_alerts': len(self.alerts),
            'critical_alerts': sum(1 for a in self.alerts if a['threat_level'] == 'CRITICAL'),
            'high_alerts': sum(1 for a in self.alerts if a['threat_level'] == 'HIGH'),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }


# INTEGRATION EXAMPLE - Add this to your main SkyGuard loop
def integrate_with_skyguard(video_source='demo_video.mp4'):
    """
    Example integration with SkyGuard main system
    """
    # Initialize thermal module
    thermal_module = SkyGuardThermalModule(
        model_path='runs/detect/thermal_weapon_demo/weights/best.pt'
    )
    
    # Open video
    cap = cv2.VideoCapture(video_source)
    
    # Simulate suspect tracking (replace with your actual tracking)
    suspect_id = "SUSPECT_001"
    frame_count = 0
    
    print("🎥 Starting SkyGuard with Thermal Weapon Detection...")
    print("Press 'q' to quit\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Simulate suspect detection (replace with your face recognition)
        # For demo, assume suspect is in center of frame
        h, w = frame.shape[:2]
        suspect_bbox = (w//4, h//4, w//2, h//2)  # Center region
        
        # Every 30 frames, check for weapons
        if frame_count % 30 == 0:
            print(f"\n📡 Scanning suspect {suspect_id} (Frame {frame_count})...")
            
            # Process frame for weapons
            result = thermal_module.process_suspect_frame(
                frame, suspect_id, suspect_bbox
            )
            
            # Visualize if weapons detected
            if result['weapons_detected'] > 0:
                # Show thermal view
                cv2.imshow('Thermal Analysis', result['thermal_frame'])
                
                # Show annotated result
                annotated = thermal_module.visualize_detection(
                    frame,
                    result['detections'],
                    result['threat_assessment']
                )
                cv2.imshow('SkyGuard - Weapon Detection', annotated)
            else:
                print("✓ No weapons detected")
                cv2.imshow('SkyGuard - Weapon Detection', frame)
        else:
            cv2.imshow('SkyGuard - Weapon Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    summary = thermal_module.get_alert_summary()
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Critical: {summary['critical_alerts']} | High: {summary['high_alerts']}")
    print(f"Alerts saved to: ./alerts/")
    print("="*70)


if __name__ == "__main__":
    # Run demo
    integrate_with_skyguard('test_video.mp4')