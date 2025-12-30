"""
Precise Cold Spot Detector
Sharp, accurate detection with minimal spread around objects
Focus: Tight boundaries around actual metal objects blocking heat
"""

import cv2
import numpy as np

class PreciseColdSpotDetector:
    def __init__(self):
        # Stricter parameters for precise detection
        self.min_cold_area = 600          # Even larger minimum
        self.max_cold_area = 5000         # Smaller maximum
        self.edge_detection_threshold = 100  # For sharp boundaries
        
    def create_sharp_thermal(self, frame):
        """
        Creates thermal image with sharp edges and minimal blur
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter (preserves edges while smoothing)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Invert for heat map
        heat_map = cv2.bitwise_not(bilateral)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        heat_map = clahe.apply(heat_map)
        
        # Minimal blur (less spread)
        heat_map = cv2.GaussianBlur(heat_map, (7, 7), 0)  # Much smaller kernel
        
        # Apply thermal colormap
        thermal = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        
        return thermal, heat_map
    
    def detect_precise_cold_spots(self, thermal, heat_map, original):
        """
        Detects cold spots with tight, precise boundaries
        """
        # Step 1: Edge detection on original image for object boundaries
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Step 2: Very strict HSV-based cold detection
        hsv = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV)
        
        # Extremely narrow blue range (only very cold regions)
        lower_cold = np.array([110, 150, 100])  # Very strict
        upper_cold = np.array([130, 255, 255])
        
        cold_mask_hsv = cv2.inRange(hsv, lower_cold, upper_cold)
        
        # Step 3: Direct temperature threshold (sharp cutoff)
        _, cold_mask_temp = cv2.threshold(heat_map, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Step 4: Combine both (intersection = very strict)
        cold_mask = cv2.bitwise_and(cold_mask_hsv, cold_mask_temp)
        
        # Step 5: Use edges to refine boundaries
        # Dilate edges slightly
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
        
        # Use edges to constrain cold spots
        cold_mask = cv2.bitwise_and(cold_mask, cv2.bitwise_not(edges_dilated))
        
        # Step 6: Minimal morphology (preserve sharp edges)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cold_mask = cv2.morphologyEx(cold_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Step 7: Remove very small noise
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cold_mask = cv2.morphologyEx(cold_mask, cv2.MORPH_CLOSE, kernel_tiny, iterations=1)
        
        # Step 8: Erode to tighten boundaries
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cold_mask = cv2.erode(cold_mask, kernel_erode, iterations=2)
        
        return cold_mask, edges
    
    def filter_valid_obstructions(self, cold_mask, frame_shape):
        """
        Filters to keep only body-region obstructions with tight boundaries
        """
        h, w = frame_shape[:2]
        
        # Body region mask (tighter)
        body_mask = np.zeros((h, w), dtype=np.uint8)
        x_start = int(w * 0.25)
        x_end = int(w * 0.75)
        y_start = int(h * 0.20)
        y_end = int(h * 0.80)
        
        cv2.rectangle(body_mask, (x_start, y_start), (x_end, y_end), 255, -1)
        
        # Apply body filter
        cold_mask = cv2.bitwise_and(cold_mask, body_mask)
        
        return cold_mask
    
    def analyze_tight_spots(self, cold_mask, frame_shape):
        """
        Analyzes cold spots with strict validation
        """
        contours, _ = cv2.findContours(
            cold_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        valid_spots = []
        h, w = frame_shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Strict size filter
            if not (self.min_cold_area < area < self.max_cold_area):
                continue
            
            # Bounding box
            x, y, width, height = cv2.boundingRect(contour)
            
            # Aspect ratio check
            aspect_ratio = float(width) / height if height > 0 else 0
            
            # Reject extreme shapes
            if aspect_ratio > 4.0 or aspect_ratio < 0.2:
                continue
            
            # Calculate solidity (how filled the contour is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Objects should be relatively solid (not scattered pixels)
            if solidity < 0.6:  # Too scattered
                continue
            
            # Calculate perimeter-to-area ratio (compactness)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            compactness = (perimeter * perimeter) / (4 * np.pi * area)
            
            # Should not be too irregular
            if compactness > 3.0:  # Too irregular
                continue
            
            # Circularity check (avoid circles)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.80:  # Too circular
                continue
            
            # Calculate tightness score (how well-defined the boundary is)
            rect_area = width * height
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            # Calculate confidence
            size_score = min(area / 2500, 1.0)
            solidity_score = solidity
            shape_score = 1.0 - abs(circularity - 0.5)  # Prefer medium circularity
            tightness_score = fill_ratio
            
            confidence = (
                size_score * 0.25 + 
                solidity_score * 0.25 + 
                shape_score * 0.25 +
                tightness_score * 0.25
            )
            
            # Very strict confidence threshold
            if confidence < 0.70:  # Higher threshold
                continue
            
            # Determine location
            relative_y = (y + height/2) / h
            if relative_y < 0.35:
                location = "CHEST"
            elif relative_y < 0.55:
                location = "WAIST"
            elif relative_y < 0.75:
                location = "HIP"
            else:
                location = "THIGH"
            
            # Use actual contour for tighter fitting polygon
            epsilon = 0.02 * perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            valid_spots.append({
                'bbox': (x, y, width, height),
                'contour': approx_contour,
                'confidence': confidence,
                'location': location,
                'area': area,
                'solidity': solidity
            })
        
        # Sort by confidence
        valid_spots.sort(key=lambda x: x['confidence'], reverse=True)
        
        return valid_spots
    
    def visualize_precise(self, original, thermal, cold_mask, spots, edges):
        """
        Clean visualization with tight boundaries
        """
        h, w = original.shape[:2]
        
        # Create display
        display = np.zeros((h, w*3, 3), dtype=np.uint8)
        
        # Panel 1: Original with tight contours
        original_annotated = original.copy()
        
        for spot in spots:
            contour = spot['contour']
            conf = spot['confidence']
            loc = spot['location']
            
            # Color based on confidence
            if conf > 0.85:
                color = (0, 0, 255)      # Red
                thickness = 3
            elif conf > 0.75:
                color = (0, 140, 255)    # Orange
                thickness = 2
            else:
                color = (0, 255, 255)    # Yellow
                thickness = 2
            
            # Draw tight contour (not bounding box)
            cv2.drawContours(original_annotated, [contour], -1, color, thickness)
            
            # Get bounding box just for label placement
            x, y, width, height = spot['bbox']
            
            # Label
            label = f"{conf:.0%}"
            cv2.putText(original_annotated, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Location
            cv2.putText(original_annotated, loc, (x, y+height+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        display[:, 0:w] = original_annotated
        cv2.putText(display, "Original (Tight Detection)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Panel 2: Thermal with tight boundaries
        thermal_annotated = thermal.copy()
        
        for spot in spots:
            # Draw filled contour with transparency
            overlay = thermal_annotated.copy()
            cv2.drawContours(overlay, [spot['contour']], -1, (0, 0, 255), -1)
            thermal_annotated = cv2.addWeighted(thermal_annotated, 0.7, overlay, 0.3, 0)
            
            # Draw outline
            cv2.drawContours(thermal_annotated, [spot['contour']], -1, (255, 255, 255), 2)
        
        display[:, w:w*2] = thermal_annotated
        
        status_color = (0, 0, 255) if len(spots) > 0 else (0, 255, 0)
        cv2.putText(display, f"Thermal: {len(spots)} Obstruction(s)", (w+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Panel 3: Precise cold mask
        cold_colored = cv2.cvtColor(cold_mask, cv2.COLOR_GRAY2BGR)
        cold_colored[cold_mask > 0] = [255, 0, 0]  # Pure blue for cold spots
        
        display[:, w*2:w*3] = cold_colored
        cv2.putText(display, "Precise Cold Spots", (w*2+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display

def main():
    """
    Main demo with precise cold spot detection
    """
    print("="*70)
    print("PRECISE COLD SPOT DETECTOR")
    print("Tight, accurate boundaries around metal objects")
    print("="*70)
    print("\n📍 Key Features:")
    print("   • Sharp edges (minimal blur)")
    print("   • Tight contours (no spread)")
    print("   • High confidence only (70%+)")
    print("   • Body region focus\n")
    print("🧪 Test Instructions:")
    print("   1. Hold metal bottle/phone FIRMLY against body")
    print("   2. Keep it pressed for 2-3 seconds")
    print("   3. Object should appear as TIGHT blue spot")
    print("   4. Move object around chest/waist area\n")
    print("⚙️  Detection Settings:")
    print("   • Min size: 600 pixels (medium objects)")
    print("   • Max size: 5000 pixels (large objects)")
    print("   • Confidence: 70%+ threshold\n")
    print("🎥 Starting webcam...")
    print("   Press 'q' to quit")
    print("   Press 's' to save screenshot\n")
    
    detector = PreciseColdSpotDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam not available")
        return
    
    print("✓ Detection active - Hold object against body\n")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to standard size
        frame = cv2.resize(frame, (640, 480))
        
        # Process with precise detection
        thermal, heat_map = detector.create_sharp_thermal(frame)
        cold_mask, edges = detector.detect_precise_cold_spots(thermal, heat_map, frame)
        cold_mask = detector.filter_valid_obstructions(cold_mask, frame.shape)
        spots = detector.analyze_tight_spots(cold_mask, frame.shape)
        
        # Visualize
        display = detector.visualize_precise(frame, thermal, cold_mask, spots, edges)
        
        # Console feedback
        if spots:
            detection_count += 1
            spot_info = " | ".join([
                f"{s['location']} ({s['confidence']:.0%}, {s['area']:.0f}px)"
                for s in spots[:2]
            ])
            print(f"\r🎯 Frame {frame_count}: DETECTED - {spot_info}                    ", 
                  end="", flush=True)
        else:
            print(f"\r⚪ Frame {frame_count}: Clear - No obstructions detected            ", 
                  end="", flush=True)
        
        # Display
        cv2.imshow('SkyGuard - Precise Cold Spot Detection', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'precise_detection_{frame_count}.jpg'
            cv2.imwrite(filename, display)
            print(f"\n✓ Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Detection rate: {(detection_count/frame_count*100) if frame_count > 0 else 0:.1f}%")
    print("\n💡 Tips for better detection:")
    print("   • Press object firmly against body")
    print("   • Hold steady for 2-3 seconds")
    print("   • Position in chest/waist area")
    print("   • Use larger metal objects (phone, bottle)")
    print("="*70)

if __name__ == "__main__":
    main()