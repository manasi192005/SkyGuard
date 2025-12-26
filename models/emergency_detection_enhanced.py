"""
Enhanced Emergency Detection with 5-Minute Immobility Check
Uses MediaPipe for body angle analysis
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import math

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available, using simplified detection")

class EmergencyDetectorEnhanced:
    """
    Enhanced Emergency Detection
    Detects people lying down for more than 5 minutes
    """
    def __init__(self, fall_angle_threshold=60, immobility_seconds=300, fps=30):
        """
        Initialize Emergency Detector
        
        Args:
            fall_angle_threshold: Angle threshold for horizontal position (degrees)
            immobility_seconds: Seconds of immobility to confirm emergency (default 300 = 5 minutes)
            fps: Frames per second of video
        """
        self.fall_angle_threshold = fall_angle_threshold
        self.immobility_seconds = immobility_seconds
        self.immobility_frames = immobility_seconds * fps  # 9000 frames at 30fps
        self.fps = fps
        
        # Initialize MediaPipe Pose if available
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
                print("✓ Using MediaPipe Pose for body angle detection")
            except Exception as e:
                print(f"MediaPipe initialization failed: {e}")
                self.use_mediapipe = False
        else:
            self.use_mediapipe = False
            print("✓ Using simplified detection (MediaPipe not available)")
        
        # Tracking state for multiple people
        self.person_states = {}
        self.emergency_history = deque(maxlen=100)
        
        # Fallback: use contour-based detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
    def calculate_body_angle(self, landmarks):
        """
        Calculate body angle from pose landmarks
        
        Returns:
            angle: Body angle in degrees (0=horizontal, 90=vertical)
            center: Body center point
        """
        if not landmarks:
            return None, None
        
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate body center
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        
        # Calculate angle
        diff = shoulder_center - hip_center
        angle = abs(math.degrees(math.atan2(diff[1], diff[0])))
        
        # Normalize angle (0-90 degrees, where 90 is vertical)
        if angle > 90:
            angle = 180 - angle
        
        body_angle = 90 - angle
        
        return body_angle, shoulder_center
    
    def detect_lying_simple(self, frame):
        """
        Simple detection of lying people using contours
        Fallback when MediaPipe is not available
        
        Returns:
            is_lying: Boolean
            angle_estimate: Estimated body angle
            center: Position
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 3000:  # Significant object
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # If width > height significantly, likely lying down
                if aspect_ratio > 1.5:
                    center = (x + w//2, y + h//2)
                    angle_estimate = 30  # Estimated horizontal angle
                    return True, angle_estimate, center
        
        return False, 90, None
    
    def detect_immobility(self, position, person_id, current_frame_idx):
        """
        Detect if person is immobile for 5 minutes
        
        Returns:
            is_immobile: Boolean indicating if immobile
            immobile_duration_seconds: Duration in seconds
            is_emergency: Boolean indicating confirmed emergency (5+ minutes)
        """
        if person_id not in self.person_states:
            self.person_states[person_id] = {
                'last_position': None,
                'immobile_since_frame': None,
                'emergency_confirmed': False,
                'alert_sent': False
            }
        
        state = self.person_states[person_id]
        
        if position is None:
            return False, 0, False
        
        # Check movement
        if state['last_position'] is not None:
            if isinstance(position, tuple):
                current_pos = np.array(position)
                last_pos = np.array(state['last_position'])
            else:
                current_pos = position
                last_pos = state['last_position']
            
            movement = np.linalg.norm(current_pos - last_pos)
            
            # Very small movement threshold
            if movement < 20:  # pixels for simple detection
                if state['immobile_since_frame'] is None:
                    state['immobile_since_frame'] = current_frame_idx
            else:
                # Person moved - reset
                state['immobile_since_frame'] = None
                state['emergency_confirmed'] = False
                state['alert_sent'] = False
        
        state['last_position'] = position
        
        # Calculate immobility duration
        if state['immobile_since_frame'] is not None:
            immobile_frames = current_frame_idx - state['immobile_since_frame']
            immobile_seconds = immobile_frames / self.fps
            
            # Check if reached 5-minute threshold
            is_emergency = immobile_frames >= self.immobility_frames
            
            if is_emergency and not state['emergency_confirmed']:
                state['emergency_confirmed'] = True
                print(f"\n🚨 MEDICAL EMERGENCY DETECTED for person {person_id}")
                print(f"   Immobile for {immobile_seconds/60:.1f} minutes")
            
            return True, immobile_seconds, is_emergency
        
        return False, 0, False
    
    def detect_emergency(self, frame, frame_idx=0, draw_visualization=True):
        """
        Main emergency detection function
        
        Returns:
            processed_frame, emergency_detected, emergency_info
        """
        output_frame = frame.copy()
        
        emergency_detected = False
        emergency_info = {
            'body_angle': None,
            'immobile': False,
            'immobile_duration_seconds': 0,
            'emergency_confirmed': False,
            'position': None,
            'time_remaining_seconds': None
        }
        
        if self.use_mediapipe:
            # Use MediaPipe detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw pose landmarks
                if draw_visualization:
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                
                # Calculate body angle
                body_angle, center = self.calculate_body_angle(landmarks)
                
                if body_angle is not None and body_angle < self.fall_angle_threshold:
                    emergency_info['body_angle'] = body_angle
                    
                    # Convert normalized center to pixel coordinates
                    h, w = frame.shape[:2]
                    center_px = (int(center[0] * w), int(center[1] * h))
                    
                    # Check immobility
                    is_immobile, duration_seconds, is_emergency = self.detect_immobility(
                        center_px, 0, frame_idx
                    )
                    
                    emergency_info['immobile'] = is_immobile
                    emergency_info['immobile_duration_seconds'] = duration_seconds
                    emergency_info['emergency_confirmed'] = is_emergency
                    emergency_info['position'] = center_px
                    
                    if is_immobile and not is_emergency:
                        time_remaining = self.immobility_seconds - duration_seconds
                        emergency_info['time_remaining_seconds'] = max(0, time_remaining)
                    
                    emergency_detected = is_emergency
        else:
            # Use simple detection
            is_lying, angle_estimate, center = self.detect_lying_simple(frame)
            
            if is_lying:
                emergency_info['body_angle'] = angle_estimate
                
                # Check immobility
                is_immobile, duration_seconds, is_emergency = self.detect_immobility(
                    center, 0, frame_idx
                )
                
                emergency_info['immobile'] = is_immobile
                emergency_info['immobile_duration_seconds'] = duration_seconds
                emergency_info['emergency_confirmed'] = is_emergency
                emergency_info['position'] = center
                
                if is_immobile and not is_emergency:
                    time_remaining = self.immobility_seconds - duration_seconds
                    emergency_info['time_remaining_seconds'] = max(0, time_remaining)
                
                emergency_detected = is_emergency
        
        # Draw visualization
        if draw_visualization and (emergency_info['immobile'] or emergency_detected):
            if emergency_detected:
                # CRITICAL EMERGENCY - RED ALERT
                alert_color = (0, 0, 255)
                cv2.rectangle(output_frame, (10, 60), (600, 180), alert_color, -1)
                
                cv2.putText(output_frame, "🚨 MEDICAL EMERGENCY! 🚨", (20, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
                duration_text = f"Person immobile: {emergency_info['immobile_duration_seconds']/60:.1f} min"
                cv2.putText(output_frame, duration_text, (20, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(output_frame, "DISPATCH MEDICAL TEAM NOW!", (20, 165),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw circle at location
                if emergency_info['position']:
                    radius = 50 + int(20 * abs(np.sin(frame_idx * 0.1)))
                    cv2.circle(output_frame, emergency_info['position'], radius, alert_color, 5)
            
            elif emergency_info['immobile']:
                # Potential emergency - YELLOW WARNING
                warning_color = (0, 255, 255)
                cv2.rectangle(output_frame, (10, 60), (550, 150), warning_color, -1)
                
                cv2.putText(output_frame, "⚠ Potential Emergency", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                
                duration_text = f"Lying down: {emergency_info['immobile_duration_seconds']:.0f}s"
                cv2.putText(output_frame, duration_text, (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                if emergency_info['time_remaining_seconds']:
                    time_text = f"Emergency in: {emergency_info['time_remaining_seconds']:.0f}s"
                    cv2.putText(output_frame, time_text, (20, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return output_frame, emergency_detected, emergency_info
    
    def reset_tracking(self):
        """Reset all tracking states"""
        self.person_states.clear()


# Test module
if __name__ == '__main__':
    print("Enhanced Emergency Detection Module - Test")
    detector = EmergencyDetectorEnhanced()
    print("✓ Module loaded successfully!")
    print(f"✓ Fall angle threshold: {detector.fall_angle_threshold}°")
    print(f"✓ Emergency threshold: {detector.immobility_seconds}s ({detector.immobility_seconds/60} min)")
    print(f"✓ Monitoring at {detector.fps} FPS")
