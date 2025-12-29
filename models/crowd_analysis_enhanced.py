"""
Enhanced Crowd Density Analysis with Heat Map
Creates density heat maps with red indicating high crowd areas
"""

import cv2
import numpy as np
from collections import deque

class CrowdAnalyzerEnhanced:
    """Enhanced Crowd Density Analysis with Heat Map Visualization"""
    
    def __init__(self, use_simple_detection=True):
        """Initialize Enhanced Crowd Analyzer"""
        self.use_simple_detection = use_simple_detection
        
        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # History for temporal analysis
        self.density_history = deque(maxlen=30)
        self.risk_threshold = {
            'low': 5,
            'medium': 15,
            'high': 30,
            'critical': 50
        }
        
        # Heat map parameters
        self.heatmap_size = (640, 480)
        self.density_map = None
        
    def detect_people_hog(self, frame):
        """Detect people using HOG"""
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Scale back boxes
        boxes = [[int(x/scale), int(y/scale), int(w/scale), int(h/scale)] 
                 for (x, y, w, h) in boxes]
        
        return boxes, len(boxes), weights
    
    def create_density_heatmap(self, frame, boxes):
        """
        Create density heat map
        Red = High density (stampede prone)
        Yellow/Orange = Medium density
        Green/Blue = Low density
        """
        h, w = frame.shape[:2]
        
        # Create empty density map
        density_map = np.zeros((h, w), dtype=np.float32)
        
        # For each detected person, add a gaussian blob
        for (x, y, box_w, box_h) in boxes:
            # Center of person
            cx = x + box_w // 2
            cy = y + box_h // 2
            
            # Create gaussian influence around person
            for i in range(max(0, cy-50), min(h, cy+50)):
                for j in range(max(0, cx-50), min(w, cx+50)):
                    dist = np.sqrt((i-cy)**2 + (j-cx)**2)
                    if dist < 50:
                        # Gaussian falloff
                        intensity = np.exp(-(dist**2) / (2 * 30**2))
                        density_map[i, j] += intensity
        
        # Normalize density map
        if density_map.max() > 0:
            density_map = (density_map / density_map.max() * 255).astype(np.uint8)
        else:
            density_map = density_map.astype(np.uint8)
        
        # Apply color map (JET: Blue->Green->Yellow->Red)
        heatmap_colored = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
        
        return heatmap_colored, density_map
    
    def identify_stampede_zones(self, density_map, threshold=150):
        """
        Identify high-risk stampede zones (red areas)
        Returns coordinates of dangerous areas
        """
        # Find areas with density above threshold
        high_density_mask = density_map > threshold
        
        # Find contours of high-risk zones
        contours, _ = cv2.findContours(
            high_density_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        stampede_zones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                stampede_zones.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'severity': 'critical' if area > 5000 else 'high'
                })
        
        return stampede_zones
    
    def analyze_crowd(self, frame, draw_visualization=True):
        """
        Main crowd analysis with heat map
        
        Returns:
            processed_frame, crowd_count, risk_level, heatmap, stampede_zones
        """
        output_frame = frame.copy()
        
        # Detect people
        boxes, count, weights = self.detect_people_hog(frame)
        
        # Create heat map
        heatmap_colored, density_map = self.create_density_heatmap(frame, boxes)
        
        # Identify stampede zones
        stampede_zones = self.identify_stampede_zones(density_map, threshold=150)
        
        # Add to history
        self.density_history.append(count)
        
        # Determine risk level
        risk_level = self.get_risk_level(count)
        
        if draw_visualization:
            # Blend heat map with original frame
            # More transparent when less people, more visible when more people
            alpha = min(0.6, 0.3 + (count / 50) * 0.3)
            output_frame = cv2.addWeighted(output_frame, 1-alpha, heatmap_colored, alpha, 0)
            
            # Draw bounding boxes for detected people
            for (x, y, w, h) in boxes:
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw stampede zones with RED warning
            for zone in stampede_zones:
                x, y, w, h = zone['bbox']
                severity_color = (0, 0, 255) if zone['severity'] == 'critical' else (0, 165, 255)
                
                # Draw thick red border for stampede zones
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), severity_color, 4)
                
                # Add warning label
                label = f"STAMPEDE RISK! Area: {zone['area']}"
                cv2.putText(output_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, severity_color, 2)
            
            # Draw information panel
            info_panel_height = 120
            panel = np.zeros((info_panel_height, frame.shape[1], 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
            
            # Crowd count
            cv2.putText(panel, f"People Count: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Risk level with color
            risk_color = self.get_risk_color(risk_level)
            cv2.putText(panel, f"Risk Level: {risk_level.upper()}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
            
            # Stampede zones count
            if stampede_zones:
                cv2.putText(panel, f"Stampede Zones: {len(stampede_zones)}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add legend
            legend_x = frame.shape[1] - 200
            cv2.putText(panel, "Heat Map Legend:", (legend_x, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(panel, (legend_x, 35), (legend_x+30, 45), (0, 0, 255), -1)
            cv2.putText(panel, "High Risk", (legend_x+35, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(panel, (legend_x, 55), (legend_x+30, 65), (0, 255, 255), -1)
            cv2.putText(panel, "Medium", (legend_x+35, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(panel, (legend_x, 75), (legend_x+30, 85), (255, 0, 0), -1)
            cv2.putText(panel, "Low Risk", (legend_x+35, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Stack panel on top
            output_frame = np.vstack([panel, output_frame])
        
        return output_frame, count, risk_level, heatmap_colored, stampede_zones
    
    def get_risk_level(self, count):
        """Determine risk level based on crowd count"""
        if count < self.risk_threshold['low']:
            return 'low'
        elif count < self.risk_threshold['medium']:
            return 'medium'
        elif count < self.risk_threshold['high']:
            return 'high'
        else:
            return 'critical'
    
    def get_risk_color(self, risk_level):
        """Get color for risk level visualization"""
        colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'high': (0, 165, 255),    # Orange
            'critical': (0, 0, 255)   # Red
        }
        return colors.get(risk_level, (255, 255, 255))
    
    def detect_anomaly(self):
        """Detect crowd anomalies"""
        if len(self.density_history) < 10:
            return False, 'normal'
        
        recent = list(self.density_history)[-10:]
        avg_density = np.mean(recent)
        current_density = recent[-1]
        
        # Sudden increase (stampede risk)
        if current_density > avg_density * 1.5 and current_density > 20:
            return True, 'stampede_risk'
        
        return False, 'normal'


# Test the module
if __name__ == '__main__':
    print("Enhanced Crowd Analysis Module - Test")
    analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
    print("✓ Module loaded successfully!")
    print("✓ Heat map generation ready")
    print("✓ Stampede zone detection ready")