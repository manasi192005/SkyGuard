
"""
Fall Detection & Emergency Response Module
Uses MediaPipe Pose estimation to detect falls and medical emergencies
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime
import math

class FallDetector:
    """
    Detect falls using pose estimation and body angle analysis
    """
    def __init__(self, fall_angle_threshold=60, immobility_frames=30):
        """
        Initialize Fall Detector
        
        Args:
            fall_angle_threshold: Angle threshold for fall detection (degrees)
            immobility_frames: Frames of immobility to confirm fall
        """
        self.fall_angle_threshold = fall_angle_threshold
        self.immobility_frames = immobility_frames
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking state
        self.person_states = {}
        self.fall_history = deque(maxlen=100)
    
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
    
    def detect_immobility(self, landmarks, person_id, current_frame_idx):
        """
        Detect if person is immobile (confirming fall)
        
        Returns:
            is_immobile: Boolean indicating immobility
            immobile_duration: Number of frames immobile
        """
        if person_id not in self.person_states:
            self.person_states[person_id] = {
                'last_position': None,
                'immobile_since': None,
                'fall_confirmed': False
            }
        
        state = self.person_states[person_id]
        
        # Calculate body center
        if not landmarks:
            return False, 0
        
        center = np.array([
            sum([lm.x for lm in landmarks]) / len(landmarks),
            sum([lm.y for lm in landmarks]) / len(landmarks)
        ])
        
        # Check movement
        if state['last_position'] is not None:
            movement = np.linalg.norm(center - state['last_position'])
            
            if movement < 0.02:  # Very small movement threshold
                if state['immobile_since'] is None:
                    state['immobile_since'] = current_frame_idx
            else:
                state['immobile_since'] = None
                state['fall_confirmed'] = False
        
        state['last_position'] = center
        
        # Calculate immobility duration
        if state['immobile_since'] is not None:
            immobile_duration = current_frame_idx - state['immobile_since']
            is_immobile = immobile_duration >= self.immobility_frames
            
            if is_immobile and not state['fall_confirmed']:
                state['fall_confirmed'] = True
            
            return is_immobile, immobile_duration
        
        return False, 0
    
    def detect_fall(self, frame, frame_idx=0, draw_visualization=True):
        """
        Main fall detection function
        
        Args:
            frame: Input video frame
            frame_idx: Current frame index
            draw_visualization: Whether to draw pose and annotations
            
        Returns:
            processed_frame, fall_detected, fall_info
        """
        output_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_frame)
        
        fall_detected = False
        fall_info = {
            'body_angle': None,
            'immobile': False,
            'immobile_duration': 0,
            'confidence': 0.0,
            'position': None
        }
        
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
            
            if body_angle is not None:
                fall_info['body_angle'] = body_angle
                
                # Check if person is horizontal (potential fall)
                if body_angle < self.fall_angle_threshold:
                    # Check immobility to confirm fall
                    is_immobile, duration = self.detect_immobility(landmarks, 0, frame_idx)
                    
                    fall_info['immobile'] = is_immobile
                    fall_info['immobile_duration'] = duration
                    
                    if is_immobile:
                        fall_detected = True
                        fall_info['confidence'] = 1.0
                        fall_info['position'] = center
                        
                        # Log fall
                        self.fall_history.append({
                            'timestamp': datetime.now(),
                            'frame_idx': frame_idx,
                            'body_angle': body_angle,
                            'position': center
                        })
                    else:
                        fall_info['confidence'] = 0.5
                
                # Visualization
                if draw_visualization:
                    # Draw body angle
                    angle_text = f"Angle: {body_angle:.1f}°"
                    cv2.putText(output_frame, angle_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if fall_detected:
                        # Draw FALL DETECTED alert
                        cv2.putText(output_frame, "FALL DETECTED!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        
                        cv2.putText(output_frame, f"Immobile: {duration} frames", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Draw alert circle at person location
                        if center is not None:
                            h, w = frame.shape[:2]
                            cx, cy = int(center[0] * w), int(center[1] * h)
                            cv2.circle(output_frame, (cx, cy), 50, (0, 0, 255), 3)
                    
                    elif body_angle < self.fall_angle_threshold:
                        cv2.putText(output_frame, "Potential Fall", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        return output_frame, fall_detected, fall_info
    
    def reset_tracking(self):
        """Reset all tracking states"""
        self.person_states.clear()


class EmergencyResponseSystem:
    """
    Coordinate emergency responses to detected falls
    """
    def __init__(self, response_delay=2.0):
        """
        Initialize Emergency Response System
        
        Args:
            response_delay: Delay before triggering response (seconds)
        """
        self.response_delay = response_delay
        self.active_emergencies = {}
        self.fall_detector = FallDetector()
    
    def process_frame(self, frame, frame_idx, gps_location=None):
        """
        Process frame and coordinate emergency response
        
        Returns:
            processed_frame, emergency_triggered, emergency_info
        """
        # Detect falls
        processed_frame, fall_detected, fall_info = self.fall_detector.detect_fall(
            frame, frame_idx
        )
        
        emergency_triggered = False
        emergency_info = None
        
        if fall_detected:
            # Create emergency event
            emergency_info = {
                'type': 'fall_detected',
                'timestamp': datetime.now(),
                'location': gps_location,
                'fall_info': fall_info,
                'response_initiated': True
            }
            
            emergency_triggered = True
            
            # Draw emergency instructions on frame
            cv2.rectangle(processed_frame, (10, 150), (400, 250), (0, 0, 255), -1)
            cv2.putText(processed_frame, "EMERGENCY RESPONSE INITIATED", (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Medical Kit Delivery: ACTIVE", (20, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if gps_location:
                cv2.putText(processed_frame, f"Location: {gps_location}", (20, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return processed_frame, emergency_triggered, emergency_info
    
    def trigger_medical_delivery(self, location):
        """Trigger medical kit delivery (placeholder for drone coordination)"""
        print(f"[EMERGENCY] Medical kit delivery initiated to location: {location}")
        return {
            'delivery_initiated': True,
            'target_location': location,
            'estimated_arrival': '60-90 seconds'
        }


# Test the module
if __name__ == '__main__':
    print("Fall Detection Module - Test")
    detector = FallDetector()
    print("Module loaded successfully!")
    print("MediaPipe Pose initialized")
    print(f"Fall angle threshold: {detector.fall_angle_threshold}°")
    print(f"Immobility confirmation: {detector.immobility_frames} frames")
