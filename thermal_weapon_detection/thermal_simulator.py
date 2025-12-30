"""
Standalone thermal simulator test - No imports needed
Run this to verify thermal effect works
"""

import cv2
import numpy as np
import os

def create_thermal_effect(image):
    """Converts RGB image to simulated thermal image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (15, 15), 0)
    enhanced = cv2.equalizeHist(blurred)
    thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    return thermal

def add_weapon_cold_spot(thermal_image, bbox):
    """Adds a cold spot at weapon location"""
    x, y, w, h = bbox
    mask = np.zeros(thermal_image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    cold_overlay = thermal_image.copy()
    cold_overlay[mask > 0] = [255, 0, 0]  # Deep blue = cold
    result = cv2.addWeighted(thermal_image, 0.7, cold_overlay, 0.3, 0)
    return result

def create_sample_images():
    """Creates sample thermal images with weapons"""
    print("📸 Creating sample thermal images...")
    
    os.makedirs('test_images', exist_ok=True)
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    for i in range(60):  # 50 train + 10 val
        # Create person silhouette
        img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add person shape
        person_x, person_y = 200, 80
        person_w, person_h = 240, 400
        cv2.ellipse(img, (person_x + person_w//2, person_y + person_h//2), 
                   (person_w//2, person_h//2), 0, 0, 360, (150, 130, 120), -1)
        
        # Add head
        cv2.circle(img, (person_x + person_w//2, person_y - 20), 35, 
                  (180, 160, 140), -1)
        
        # Apply thermal effect
        thermal = create_thermal_effect(img)
        
        # Add weapon (pistol/knife/rifle)
        weapon_type = i % 3
        weapon_x = person_x + np.random.randint(40, 160)
        weapon_y = person_y + np.random.randint(150, 300)
        
        if weapon_type == 0:  # pistol
            weapon_w, weapon_h = 50, 80
        elif weapon_type == 1:  # knife
            weapon_w, weapon_h = 30, 100
        else:  # rifle
            weapon_w, weapon_h = 80, 60
        
        thermal = add_weapon_cold_spot(thermal, (weapon_x, weapon_y, weapon_w, weapon_h))
        
        # Determine split
        split = 'train' if i < 50 else 'val'
        
        # Save image
        img_path = f'dataset/images/{split}/thermal_{i:04d}.jpg'
        cv2.imwrite(img_path, thermal)
        
        # Create YOLO label
        center_x = (weapon_x + weapon_w/2) / 640
        center_y = (weapon_y + weapon_h/2) / 480
        width = weapon_w / 640
        height = weapon_h / 480
        
        label_path = f'dataset/labels/{split}/thermal_{i:04d}.txt'
        with open(label_path, 'w') as f:
            f.write(f"{weapon_type} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ Created {i+1}/60 images")
    
    print(f"\n✓ Dataset created!")
    print(f"  - Training images: 50 (dataset/images/train/)")
    print(f"  - Validation images: 10 (dataset/images/val/)")
    print(f"  - Labels created in YOLO format")

def test_webcam():
    """Test thermal effect with webcam"""
    print("\n🎥 Testing webcam (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠️  Webcam not available")
        return False
    
    print("✓ Webcam active - showing thermal effect")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        thermal = create_thermal_effect(frame)
        
        cv2.imshow('Original', frame)
        cv2.imshow('Thermal Effect', thermal)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    print("="*60)
    print("THERMAL WEAPON DETECTION - QUICK TEST")
    print("="*60)
    
    print("\n[1/2] Testing thermal effect...")
    
    # Try webcam
    webcam_ok = test_webcam()
    
    if not webcam_ok:
        print("Webcam test skipped - will create sample images instead")
    
    print("\n[2/2] Creating training dataset...")
    create_sample_images()
    
    print("\n" + "="*60)
    print("✓ READY FOR TRAINING!")
    print("="*60)
    print("\nNext step: Run training")
    print("  python3 train_thermal_weapon_detector.py")
    print("="*60)

if __name__ == "__main__":
    main()