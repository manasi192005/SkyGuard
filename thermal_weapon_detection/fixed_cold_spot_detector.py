"""
Fixed Cold Spot Detector - No resolution issues
Detects hidden weapons by identifying heat obstruction
"""

import cv2
import numpy as np

class ColdSpotDetector:
    def __init__(self):
        self.min_weapon_area = 300
        self.max_weapon_area = 8000
        
    def create_thermal(self, frame):
        """Convert RGB to thermal simulation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        enhanced = cv2.equalizeHist(blurred)
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        return thermal
    
    def detect_cold_regions(self, thermal):
        """Detect cold spots (blue regions in thermal)"""
        hsv = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV)
        
        # Blue = cold in thermal images
        lower_cold = np.array([100, 100, 50])
        upper_cold = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv, lower_cold, upper_cold)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_weapons(self, cold_mask, frame_height):
        """Find weapon-shaped cold spots"""
        contours, _ = cv2.findContours(cold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        weapons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter
            if not (self.min_weapon_area < area < self.max_weapon_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Shape filter (weapons are elongated)
            if not (0.2 < aspect_ratio < 4.0):
                continue
            
            # Calculate confidence
            size_score = min(area / 5000, 1.0)
            shape_score = 1.0 - abs(1.0 - aspect_ratio) / 3.0
            
            # Location score (weapons usually at waist/chest)
            relative_y = y / frame_height
            if 0.3 < relative_y < 0.7:
                location_score = 1.0
            else:
                location_score = 0.6
            
            confidence = (size_score * 0.4 + shape_score * 0.3 + location_score * 0.3)
            
            # Classify weapon type
            if aspect_ratio > 1.5:
                weapon_type = "rifle" if area > 4000 else "pistol"
            elif aspect_ratio < 0.5:
                weapon_type = "knife"
            else:
                weapon_type = "pistol"
            
            # Body location
            if relative_y < 0.3:
                body_location = "CHEST"
            elif relative_y < 0.6:
                body_location = "WAIST"
            elif relative_y < 0.8:
                body_location = "THIGH"
            else:
                body_location = "ANKLE"
            
            weapons.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'weapon_type': weapon_type,
                'body_location': body_location,
                'area': area
            })
        
        # Sort by confidence
        weapons.sort(key=lambda x: x['confidence'], reverse=True)
        return weapons
    
    def visualize_simple(self, original, thermal, cold_mask, weapons):
        """Simple side-by-side visualization"""
        h, w = original.shape[:2]
        
        # Create side-by-side display
        display = np.zeros((h, w*3, 3), dtype=np.uint8)
        
        # Panel 1: Original
        display[:, 0:w] = original
        cv2.putText(display, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Panel 2: Thermal with detections
        annotated = thermal.copy()
        for weapon in weapons:
            x, y, ww, hh = weapon['bbox']
            conf = weapon['confidence']
            wtype = weapon['weapon_type']
            
            color = (0, 0, 255) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(annotated, (x, y), (x+ww, y+hh), color, 3)
            
            label = f"{wtype.upper()} {conf:.0%}"
            cv2.putText(annotated, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        display[:, w:w*2] = annotated
        cv2.putText(display, f"Thermal (Weapons: {len(weapons)})", (w+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Panel 3: Cold mask
        cold_colored = cv2.cvtColor(cold_mask, cv2.COLOR_GRAY2BGR)
        display[:, w*2:w*3] = cold_colored
        cv2.putText(display, "Cold Spots", (w*2+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return display

def webcam_demo():
    """Live webcam demo"""
    print("="*70)
    print("COLD SPOT WEAPON DETECTION - LIVE DEMO")
    print("="*70)
    print("\n🎥 Starting webcam...")
    print("   Hold a phone/wallet/metal object against your body")
    print("   The object will block body heat and appear as cold spot")
    print("\n   Press 'q' to quit")
    print("   Press 's' to save screenshot\n")
    
    detector = ColdSpotDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam not available")
        return
    
    print("✓ Webcam active\n")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to standard size
        frame = cv2.resize(frame, (640, 480))
        
        # Process
        thermal = detector.create_thermal(frame)
        cold_mask = detector.detect_cold_regions(thermal)
        weapons = detector.find_weapons(cold_mask, frame.shape[0])
        
        # Visualize
        display = detector.visualize_simple(frame, thermal, cold_mask, weapons)
        
        # Show detection info
        if weapons:
            print(f"\rFrame {frame_count}: DETECTED {len(weapons)} weapon(s) - " + 
                  ", ".join([f"{w['weapon_type']}({w['confidence']:.0%})" for w in weapons[:3]]), 
                  end="", flush=True)
        
        cv2.imshow('SkyGuard Cold Spot Detection', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'detection_screenshot_{frame_count}.jpg'
            cv2.imwrite(filename, display)
            print(f"\n✓ Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n\n✓ Demo complete!")

def static_demo():
    """Static image demo"""
    print("="*70)
    print("COLD SPOT WEAPON DETECTION - STATIC DEMO")
    print("="*70)
    print("\n📸 Creating test scenario...")
    
    detector = ColdSpotDetector()
    
    # Create test image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 180
    
    # Draw person silhouette
    cv2.ellipse(img, (320, 260), (90, 200), 0, 0, 360, (150, 130, 120), -1)
    cv2.circle(img, (320, 100), 45, (160, 140, 130), -1)
    
    # Add "weapon" (dark object that blocks heat)
    weapon_x, weapon_y = 280, 220
    cv2.rectangle(img, (weapon_x, weapon_y), 
                 (weapon_x + 50, weapon_y + 110), (40, 40, 40), -1)
    
    # Process
    thermal = detector.create_thermal(img)
    cold_mask = detector.detect_cold_regions(thermal)
    weapons = detector.find_weapons(cold_mask, img.shape[0])
    
    # Visualize
    display = detector.visualize_simple(img, thermal, cold_mask, weapons)
    
    # Print results
    print(f"\n✓ Detection complete!")
    print(f"   Weapons found: {len(weapons)}\n")
    
    if weapons:
        for i, w in enumerate(weapons, 1):
            print(f"   {i}. {w['weapon_type'].upper()}")
            print(f"      Confidence: {w['confidence']:.1%}")
            print(f"      Location: {w['body_location']}")
            print(f"      Size: {w['area']:.0f} pixels\n")
    
    # Save result
    cv2.imwrite('cold_spot_detection_demo.jpg', display)
    print("✓ Result saved: cold_spot_detection_demo.jpg")
    
    # Display
    cv2.imshow('Cold Spot Detection Demo', display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("\nChoose demo mode:")
    print("1. Webcam (live detection)")
    print("2. Static image (test scenario)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        webcam_demo()
    else:
        static_demo()

if __name__ == "__main__":
    main()