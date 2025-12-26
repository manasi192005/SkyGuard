
"""
Violence & Anomaly Detection Module
Uses optical flow and motion analysis to detect violent behavior
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime

class ViolenceDetector:
    """
    Detect violent behavior and anomalies in crowd footage
    Uses optical flow analysis for motion patterns
    """
    def __init__(self, history_size=15):
        """
        Initialize Violence Detector
        
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = history_size
        self.motion_history = deque(maxlen=history_size)
        self.prev_gray = None
        
        # Thresholds for violence detection
        self.violence_threshold = 0.15
        self.chaos_threshold = 0.25
        
        # Dense optical flow (Farneback)
        self.use_dense_flow = True
    
    def calculate_optical_flow(self, frame):
        """
        Calculate optical flow between current and previous frame
        
        Returns:
            flow: Optical flow field
            magnitude: Motion magnitude
            angle: Motion angle
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0, None
        
        if self.use_dense_flow:
            # Dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
        else:
            magnitude = 0
            angle = None
            flow = None
        
        self.prev_gray = gray
        
        return flow, magnitude, angle
    
    def analyze_motion_patterns(self, magnitude, angle=None):
        """
        Analyze motion patterns to detect violence
        
        Returns:
            violence_score: 0-1 score indicating violence likelihood
            motion_type: Type of motion detected
        """
        # Add to history
        if isinstance(magnitude, np.ndarray):
            avg_magnitude = np.mean(magnitude)
        else:
            avg_magnitude = magnitude
            
        self.motion_history.append(avg_magnitude)
        
        if len(self.motion_history) < 5:
            return 0.0, 'insufficient_data'
        
        recent_motion = list(self.motion_history)[-10:]
        
        # Calculate statistics
        avg_motion = np.mean(recent_motion)
        motion_variance = np.var(recent_motion)
        max_motion = np.max(recent_motion)
        
        # Detect violence patterns
        violence_score = 0.0
        motion_type = 'normal'
        
        # High sustained motion (fighting, running)
        if avg_motion > self.violence_threshold:
            violence_score += 0.4
            motion_type = 'high_motion'
        
        # Chaotic motion (struggle, panic)
        if motion_variance > self.chaos_threshold:
            violence_score += 0.4
            motion_type = 'chaotic_motion'
        
        # Sudden spike (attack, fall)
        if max_motion > avg_motion * 2 and max_motion > 0.2:
            violence_score += 0.3
            motion_type = 'sudden_spike'
        
        # Combined patterns (likely violence)
        if avg_motion > self.violence_threshold and motion_variance > self.chaos_threshold:
            violence_score = 0.9
            motion_type = 'violence_detected'
        
        return min(violence_score, 1.0), motion_type
    
    def detect_violence(self, frame, draw_visualization=True):
        """
        Main violence detection function
        
        Args:
            frame: Input video frame
            draw_visualization: Whether to draw flow visualization
            
        Returns:
            processed_frame, violence_detected, violence_score, motion_type
        """
        output_frame = frame.copy()
        
        # Calculate optical flow
        flow, magnitude, angle = self.calculate_optical_flow(frame)
        
        if flow is None:
            return output_frame, False, 0.0, 'initializing'
        
        # Analyze motion patterns
        violence_score, motion_type = self.analyze_motion_patterns(magnitude, angle)
        
        # Determine if violence detected
        violence_detected = violence_score > 0.7
        
        # Visualization
        if draw_visualization and self.use_dense_flow and flow is not None:
            # Create flow visualization
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            
            if angle is not None:
                # Dense flow visualization
                mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                ang_normalized = angle * 180 / np.pi / 2
                
                hsv[..., 0] = ang_normalized
                hsv[..., 2] = mag_normalized
                
                flow_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                output_frame = cv2.addWeighted(output_frame, 0.7, flow_rgb, 0.3, 0)
        
        # Draw information
        if draw_visualization:
            # Violence indicator
            color = (0, 0, 255) if violence_detected else (0, 255, 0)
            status = "VIOLENCE DETECTED!" if violence_detected else "Normal"
            
            cv2.putText(output_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.putText(output_frame, f"Score: {violence_score:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(output_frame, f"Motion: {motion_type}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame, violence_detected, violence_score, motion_type
    
    def reset(self):
        """Reset detector state"""
        self.motion_history.clear()
        self.prev_gray = None


class AnomalyDetector:
    """
    Detect various anomalies in crowd behavior
    Combines multiple detection methods
    """
    def __init__(self):
        self.violence_detector = ViolenceDetector()
        self.anomaly_history = deque(maxlen=30)
    
    def detect_anomalies(self, frame, crowd_density=None):
        """
        Detect multiple types of anomalies
        
        Returns:
            Dictionary with anomaly information
        """
        anomalies = {
            'violence': False,
            'stampede_risk': False,
            'unusual_gathering': False,
            'rapid_dispersal': False,
            'score': 0.0,
            'type': 'normal'
        }
        
        # Violence detection
        _, violence, v_score, motion_type = self.violence_detector.detect_violence(frame, False)
        anomalies['violence'] = violence
        anomalies['score'] = max(anomalies['score'], v_score)
        
        if violence:
            anomalies['type'] = 'violence'
        
        # Crowd-based anomalies
        if crowd_density is not None:
            self.anomaly_history.append(crowd_density)
            
            if len(self.anomaly_history) >= 10:
                recent = list(self.anomaly_history)[-10:]
                avg = np.mean(recent)
                
                # Stampede risk (sudden density increase)
                if crowd_density > avg * 1.5 and crowd_density > 50:
                    anomalies['stampede_risk'] = True
                    anomalies['score'] = max(anomalies['score'], 0.8)
                    anomalies['type'] = 'stampede_risk'
                
                # Rapid dispersal (panic)
                if crowd_density < avg * 0.5 and avg > 30:
                    anomalies['rapid_dispersal'] = True
                    anomalies['score'] = max(anomalies['score'], 0.6)
                    anomalies['type'] = 'rapid_dispersal'
        
        return anomalies


# Test the module
if __name__ == '__main__':
    print("Violence Detection Module - Test")
    detector = ViolenceDetector()
    print("Module loaded successfully!")
    print("Optical flow engine: Farneback")
    print("Detection thresholds configured")
