"""
Helper script to add suspects to the database
Ensures proper face detection and quality checks
"""

import cv2
import os
import json
from datetime import datetime
import shutil

def detect_face_in_image(image_path):
    """Check if image contains a detectable face"""
    img = cv2.imread(image_path)
    if img is None:
        return False, "Cannot load image"
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    
    if len(faces) == 0:
        return False, "No face detected"
    elif len(faces) > 1:
        return False, f"Multiple faces detected ({len(faces)}). Use image with single face."
    else:
        x, y, w, h = faces[0]
        # Check face size
        img_area = img.shape[0] * img.shape[1]
        face_area = w * h
        face_ratio = face_area / img_area
        
        if face_ratio < 0.05:
            return False, "Face too small in image"
        elif w < 80 or h < 80:
            return False, f"Face resolution too low ({w}x{h}). Need at least 80x80."
        else:
            return True, f"Face detected: {w}x{h} pixels ({face_ratio*100:.1f}% of image)"


def add_suspect(name, image_path, description='', suspects_db_path='data/suspects'):
    """Add a suspect to the database with validation"""
    
    print(f"\n{'='*70}")
    print(f"➕ ADDING SUSPECT: {name}")
    print(f"{'='*70}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Validate face in image
    print(f"\n🔍 Validating image...")
    has_face, message = detect_face_in_image(image_path)
    
    if has_face:
        print(f"   ✅ {message}")
    else:
        print(f"   ❌ {message}")
        retry = input("\n   Continue anyway? (y/n): ").strip().lower()
        if retry != 'y':
            print("   Cancelled.")
            return False
    
    # Create suspects directory
    os.makedirs(suspects_db_path, exist_ok=True)
    
    # Copy image to suspects folder
    dest_filename = f"{name.replace(' ', '_').lower()}.jpg"
    dest_path = os.path.join(suspects_db_path, dest_filename)
    
    print(f"\n📋 Copying image...")
    print(f"   From: {image_path}")
    print(f"   To: {dest_path}")
    
    shutil.copy(image_path, dest_path)
    print(f"   ✅ Image copied")
    
    # Load or create metadata
    metadata_path = os.path.join(suspects_db_path, 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            suspects = json.load(f)
    else:
        suspects = []
    
    # Check if suspect already exists
    existing = next((s for s in suspects if s['name'].lower() == name.lower()), None)
    
    if existing:
        print(f"\n⚠️  Suspect '{name}' already exists!")
        overwrite = input("   Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("   Cancelled.")
            return False
        # Remove existing entry
        suspects = [s for s in suspects if s['name'].lower() != name.lower()]
    
    # Create suspect entry
    suspect_info = {
        'name': name,
        'image_path': dest_path,
        'description': description,
        'uploaded_by': 'Manual',
        'added_date': datetime.now().isoformat(),
        'status': 'active'
    }
    
    suspects.append(suspect_info)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(suspects, f, indent=2)
    
    print(f"\n✅ SUSPECT ADDED SUCCESSFULLY")
    print(f"   Name: {name}")
    print(f"   Image: {dest_path}")
    print(f"   Total suspects in DB: {len(suspects)}")
    
    # Show preview
    img = cv2.imread(dest_path)
    if img is not None:
        # Detect and draw face
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow(f'Added: {name}', img)
        print(f"\n👁️  Preview (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True


def list_suspects(suspects_db_path='data/suspects'):
    """List all suspects in database"""
    metadata_path = os.path.join(suspects_db_path, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        print("No suspects in database.")
        return
    
    with open(metadata_path, 'r') as f:
        suspects = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"📋 SUSPECTS DATABASE ({len(suspects)} total)")
    print(f"{'='*70}")
    
    for idx, suspect in enumerate(suspects, 1):
        img_exists = os.path.exists(suspect['image_path'])
        status_icon = "✅" if img_exists else "❌"
        
        print(f"\n{idx}. {suspect['name']} {status_icon}")
        print(f"   Image: {suspect['image_path']}")
        print(f"   Status: {suspect.get('status', 'active')}")
        print(f"   Added: {suspect.get('added_date', 'Unknown')}")
        if suspect.get('description'):
            print(f"   Description: {suspect['description']}")


def delete_suspect(name, suspects_db_path='data/suspects'):
    """Delete a suspect from database"""
    metadata_path = os.path.join(suspects_db_path, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        print("No suspects in database.")
        return
    
    with open(metadata_path, 'r') as f:
        suspects = json.load(f)
    
    # Find suspect
    suspect = next((s for s in suspects if s['name'].lower() == name.lower()), None)
    
    if not suspect:
        print(f"❌ Suspect '{name}' not found")
        return
    
    print(f"\n⚠️  Delete suspect: {suspect['name']}?")
    confirm = input("   Type 'DELETE' to confirm: ").strip()
    
    if confirm != 'DELETE':
        print("   Cancelled.")
        return
    
    # Remove from list
    suspects = [s for s in suspects if s['name'].lower() != name.lower()]
    
    # Delete image file
    if os.path.exists(suspect['image_path']):
        os.remove(suspect['image_path'])
        print(f"   ✅ Deleted image: {suspect['image_path']}")
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(suspects, f, indent=2)
    
    print(f"   ✅ Removed from database")
    print(f"   Total suspects remaining: {len(suspects)}")


def main():
    """Interactive menu"""
    while True:
        print(f"\n{'='*70}")
        print("SUSPECT DATABASE MANAGER")
        print(f"{'='*70}")
        print("1. Add suspect")
        print("2. List suspects")
        print("3. Delete suspect")
        print("0. Exit")
        print("="*70)
        
        choice = input("\nSelect option (0-3): ").strip()
        
        if choice == '1':
            name = input("\nSuspect name: ").strip()
            if not name:
                print("❌ Name required")
                continue
            
            image_path = input("Image path: ").strip()
            if not image_path:
                print("❌ Image path required")
                continue
            
            description = input("Description (optional): ").strip()
            
            add_suspect(name, image_path, description)
        
        elif choice == '2':
            list_suspects()
        
        elif choice == '3':
            name = input("\nSuspect name to delete: ").strip()
            if name:
                delete_suspect(name)
        
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()