"""
Shape/Bulge Detection for Regular Webcam
Detects unusual bulges or shapes in clothing (NOT thermal)
WARNING: This cannot detect flat objects under clothing!
Only works for objects that create visible bulges
"""

import cv2
import numpy as np
from collections import deque

class WebcamBulgeDetector:
    """
    Detects shape anomalies that might indicate concealed objects
    Note: This is NOT thermal detection and has high false positive rate
    """
    
    def __init__(self):
        # Body region tracking
        self.baseline_body_shape = None
        self.baseline_frames = []
        self.baseline_samples = 90
        
        # Detection parameters
        self.contour_diff_threshold = 1000  # Pixels of difference
        self.min_bulge_area = 800
        self.max_bulge_area = 6000
        
        # Temporal consistency
        self.detection_history = deque(maxlen=30)
        self.min_persistence = 20
        
    def extract_body_contour(self, frame):
        """Extract body outline using background subtraction and edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(filtered, 30, 100)
        
        # Dilate edges to connect them
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (should be the person)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour
        
        return None
    
    def build_baseline_shape(self, contour):
        """Build baseline body shape without objects"""
        if len(self.baseline_frames) < self.baseline_samples:
            if contour is not None:
                self.baseline_frames.append(contour)
            return False
        
        if self.baseline_body_shape is None:
            # Use median contour as baseline
            # This is approximate - real implementation would need contour alignment
            self.baseline_body_shape = self.baseline_frames[len(self.baseline_frames)//2]
            print(f"✓ Baseline body shape established")
        
        return True
    
    def detect_shape_anomalies(self, current_contour, frame_shape):
        """Detect differences between current and baseline shape"""
        if self.baseline_body_shape is None or current_contour is None:
            return np.zeros(frame_shape[:2], dtype=np.uint8)
        
        # Create masks for baseline and current
        h, w = frame_shape[:2]
        baseline_mask = np.zeros((h, w), dtype=np.uint8)
        current_mask = np.zeros((h, w), dtype=np.uint8)
        
        cv2.drawContours(baseline_mask, [self.baseline_body_shape], -1, 255, -1)
        cv2.drawContours(current_mask, [current_contour], -1, 255, -1)
        
        # Find differences (areas that are NEW in current)
        difference = cv2.subtract(current_mask, baseline_mask)
        
        # Also check for unusual protrusions
        # Dilate baseline and see what extends beyond
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        baseline_dilated = cv2.dilate(baseline_mask, kernel, iterations=1)
        
        protrusions = cv2.subtract(current_mask, baseline_dilated)
        
        # Combine
        anomaly_mask = cv2.bitwise_or(difference, protrusions)
        
        return anomaly_mask
    
    def refine_bulge_detection(self, anomaly_mask):
        """Clean up detection"""
        # Remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Close gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        return mask
    
    def analyze_bulges(self, mask, frame_shape):
        """Analyze detected shape anomalies"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        h, w = frame_shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (self.min_bulge_area < area < self.max_bulge_area):
                continue
            
            x, y, width, height = cv2.boundingRect(contour)
            
            # Aspect ratio check
            aspect_ratio = float(width) / height if height > 0 else 0
            if aspect_ratio > 4.0 or aspect_ratio < 0.2:
                continue
            
            # Calculate shape features
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Confidence scoring
            size_score = min(area / 2000, 1.0)
            shape_score = 1.0 if 0.3 < circularity < 0.8 else 0.5
            solidity_score = solidity
            
            confidence = (size_score * 0.4 + shape_score * 0.3 + solidity_score * 0.3)
            
            if confidence < 0.50:
                continue
            
            # Location
            relative_y = (y + height/2) / h
            if relative_y < 0.35:
                location = "CHEST"
            elif relative_y < 0.55:
                location = "ABDOMEN"
            elif relative_y < 0.75:
                location = "WAIST"
            else:
                location = "LOWER"
            
            detections.append({
                'bbox': (x, y, width, height),
                'contour': contour,
                'confidence': confidence,
                'location': location,
                'area': area,
                'type': 'bulge'
            })
        
        return detections
    
    def temporal_filtering(self, detections):
        """Filter using temporal consistency"""
        self.detection_history.append(len(detections) > 0)
        recent = sum(self.detection_history)
        
        if recent < self.min_persistence:
            return []
        
        return detections
    
    def visualize(self, frame, mask, detections, baseline_ready):
        """Visualization"""
        h, w = frame.shape[:2]
        display = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Panel 1: Original with detections
        annotated = frame.copy()
        
        for det in detections:
            x, y, width, height = det['bbox']
            
            color = (0, 140, 255)  # Orange
            cv2.rectangle(annotated, (x, y), (x+width, y+height), color, 2)
            
            label = f"{det['location']}"
            cv2.putText(annotated, label, (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            conf_label = f"Bulge: {det['confidence']:.0%}"
            cv2.putText(annotated, conf_label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        display[:, 0:w] = annotated
        
        if baseline_ready:
            status = "ANOMALY DETECTED" if detections else "NORMAL"
            color = (0, 140, 255) if detections else (0, 255, 0)
        else:
            status = "CALIBRATING..."
            color = (255, 255, 0)
        
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Panel 2: Anomaly mask
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[mask > 0] = [0, 140, 255]
        display[:, w:w*2] = mask_colored
        
        cv2.putText(display, "Shape Anomalies", (w+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display


def main():
    """Main application"""
    print("="*70)
    print("⚠️  WEBCAM BULGE DETECTION (NOT THERMAL)")
    print("="*70)
    print("\n❗ LIMITATIONS:")
    print("   • This is NOT thermal imaging")
    print("   • Can only detect VISIBLE bulges in clothing")
    print("   • Flat objects (phone, thin metal) will NOT be detected")
    print("   • High false positive rate")
    print("   • Lighting and movement affect accuracy\n")
    print("💡 RECOMMENDATION:")
    print("   For detecting hidden objects under fabric, you MUST use:")
    print("   • Real thermal camera ($200-500)")
    print("   • Millimeter wave scanner (airport-style)")
    print("   • X-ray imaging (medical/security)\n")
    print("📋 CALIBRATION:")
    print("   1. First 90 frames: Stand naturally without objects")
    print("   2. Keep still and well-lit")
    print("   3. After calibration, add visible bulge under clothing\n")
    print("🎥 Starting webcam...\n")
    
    detector = WebcamBulgeDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam not available")
        return
    
    frame_count = 0
    
    print("🔴 CALIBRATION - Stand still without objects\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        
        # Extract body contour
        contour = detector.extract_body_contour(frame)
        
        # Build baseline
        baseline_ready = detector.build_baseline_shape(contour)
        
        if baseline_ready:
            # Detect anomalies
            anomaly_mask = detector.detect_shape_anomalies(contour, frame.shape)
            refined_mask = detector.refine_bulge_detection(anomaly_mask)
            detections = detector.analyze_bulges(refined_mask, frame.shape)
            confirmed = detector.temporal_filtering(detections)
        else:
            anomaly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            refined_mask = anomaly_mask
            confirmed = []
        
        # Visualize
        display = detector.visualize(frame, refined_mask, confirmed, baseline_ready)
        
        # Status
        if not baseline_ready:
            progress = len(detector.baseline_frames)
            remaining = detector.baseline_samples - progress
            print(f"\r🔴 Calibrating: {progress}/{detector.baseline_samples} ({remaining} left)   ",
                  end="", flush=True)
        else:
            if confirmed:
                info = " | ".join([f"{d['location']} ({d['area']:.0f}px)" for d in confirmed])
                print(f"\r⚠️  Frame {frame_count}: Bulge detected - {info}                    ",
                      end="", flush=True)
            else:
                print(f"\r✓ Frame {frame_count}: Normal                                        ",
                      end="", flush=True)
        
        cv2.imshow('Webcam Shape Detection (NOT Thermal)', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.baseline_body_shape = None
            detector.baseline_frames = []
            print("\n\n🔄 Baseline reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n\n" + "="*70)
    print("⚠️  Remember: This is shape detection, NOT thermal imaging")
    print("For real concealed object detection, use a thermal camera!")
    print("="*70)


if __name__ == "__main__":
    main()