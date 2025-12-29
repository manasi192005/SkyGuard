"""
Capture Gait Profile with Quality Controls
Ensures high-quality profiles for accurate recognition
"""

import cv2
import sys
import time
from models.gait_recognition_enhanced import GaitRecognitionEnhanced

def print_banner():
    print("\n" + "="*70)
    print("🚶 SkyGuard - Gait Profile Capture System")
    print("="*70 + "\n")

def print_instructions():
    print("📋 SETUP INSTRUCTIONS:")
    print("=" * 70)
    print("\n1. CAMERA SETUP:")
    print("   • Mount camera at 6-8 feet height")
    print("   • Angle camera 30-45° downward")
    print("   • Ensure person walks 10-15 feet from camera")
    print("   • Good lighting - avoid shadows and backlighting")
    
    print("\n2. CAPTURE REQUIREMENTS:")
    print("   • Person must walk naturally (not too fast/slow)")
    print("   • Walk in a STRAIGHT LINE back and forth")
    print("   • Need MINIMUM 3-4 complete walks (150+ frames)")
    print("   • Try to maintain SIDE VIEW for best results")
    print("   • Avoid crowded scenes - one person only")
    
    print("\n3. DURING CAPTURE:")
    print("   • System will analyze camera setup quality")
    print("   • Follow on-screen guidance for optimal capture")
    print("   • Wait for 'READY TO SAVE' before stopping")
    
    print("\n" + "="*70)
    print("Press ENTER to continue, or 'q' to quit")
    
    response = input().strip().lower()
    if response == 'q':
        sys.exit(0)

def get_quality_feedback(gait_rec, person_id):
    """Generate real-time feedback on capture quality"""
    if person_id not in gait_rec.gait_history:
        return None
    
    history = list(gait_rec.gait_history[person_id])
    frames_count = len(history)
    
    feedback = {
        'frames': frames_count,
        'required_frames': gait_rec.min_capture_frames,
        'messages': [],
        'is_ready': False,
        'side_view_count': 0,
        'quality_score': 0
    }
    
    if frames_count < 50:
        feedback['messages'].append("Keep walking... collecting data")
        return feedback
    
    # Count side views
    side_views = sum(1 for _, angle, _ in history if angle in ['left_side', 'right_side'])
    side_view_ratio = side_views / frames_count if frames_count > 0 else 0
    feedback['side_view_count'] = side_views
    
    # Check walks
    walks = gait_rec.segment_walks(history)
    num_walks = len(walks)
    
    # Calculate quality
    quality = gait_rec.calculate_quality_score(history)
    feedback['quality_score'] = quality
    
    # Generate messages
    if frames_count < gait_rec.min_capture_frames:
        remaining = gait_rec.min_capture_frames - frames_count
        feedback['messages'].append(f"Need {remaining} more frames - keep walking")
    
    if num_walks < 3:
        feedback['messages'].append(f"Complete {3 - num_walks} more walks (walk back & forth)")
    
    if side_view_ratio < 0.6:
        feedback['messages'].append("Turn to show SIDE VIEW (profile)")
    
    if quality < 0.5:
        feedback['messages'].append("⚠ Low quality - check lighting & position")
    
    # Check if ready
    if (frames_count >= gait_rec.min_capture_frames and 
        num_walks >= 3 and 
        quality >= 0.5 and
        side_view_ratio >= 0.5):
        feedback['is_ready'] = True
        feedback['messages'] = ["✓ EXCELLENT DATA - Press 'q' to save profile"]
    
    return feedback

def draw_progress_bar(frame, current, total, x, y, width, height):
    """Draw a progress bar on the frame"""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Progress
    progress = min(current / total, 1.0)
    progress_width = int(width * progress)
    
    color = (0, 255, 0) if progress >= 1.0 else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + progress_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    # Text
    text = f"{current}/{total}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    print_banner()
    print_instructions()
    
    print("\n🔧 Initializing gait recognition system...")
    gait_rec = GaitRecognitionEnhanced(
        min_capture_frames=150,
        min_recognition_frames=90
    )
    print("✓ System initialized\n")
    
    # Open camera
    print("📹 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera")
        print("   Check camera connection and permissions")
        sys.exit(1)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened successfully")
    print("\n" + "="*70)
    print("🎬 CAPTURE STARTED")
    print("="*70)
    print("\nStart walking now! Press 'q' when ready to save or 'ESC' to cancel\n")
    
    person_id_to_capture = None
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Process frame
            output, _ = gait_rec.process_frame(frame, draw_visualization=True)
            
            # Get tracked person
            if gait_rec.tracked_persons and person_id_to_capture is None:
                person_id_to_capture = list(gait_rec.tracked_persons.keys())[0]
                print(f"✓ Person detected - tracking ID: {person_id_to_capture}")
            
            # Get quality feedback
            if person_id_to_capture is not None:
                feedback = get_quality_feedback(gait_rec, person_id_to_capture)
                
                if feedback:
                    # Draw overlay panel
                    panel_height = 180
                    cv2.rectangle(output, (0, 0), (w, panel_height), (0, 0, 0), -1)
                    cv2.rectangle(output, (0, 0), (w, panel_height), (255, 255, 255), 2)
                    
                    # Title
                    cv2.putText(output, "GAIT PROFILE CAPTURE", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Frames progress bar
                    cv2.putText(output, "Frames Captured:", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    draw_progress_bar(output, feedback['frames'], 
                                    feedback['required_frames'], 200, 45, 300, 20)
                    
                    # Side view progress
                    side_view_needed = int(feedback['required_frames'] * 0.6)
                    cv2.putText(output, "Side Views:", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    draw_progress_bar(output, feedback['side_view_count'], 
                                    side_view_needed, 200, 75, 300, 20)
                    
                    # Quality score
                    quality_pct = int(feedback['quality_score'] * 100)
                    quality_color = (0, 255, 0) if quality_pct >= 70 else (0, 165, 255) if quality_pct >= 50 else (0, 0, 255)
                    cv2.putText(output, f"Quality Score: {quality_pct}%", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
                    
                    # Messages
                    y_pos = 145
                    for msg in feedback['messages'][:2]:
                        color = (0, 255, 0) if '✓' in msg else (255, 255, 255)
                        cv2.putText(output, msg, 
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_pos += 25
                    
                    # Ready indicator
                    if feedback['is_ready']:
                        # Flash ready message
                        if int(time.time() * 2) % 2 == 0:
                            cv2.rectangle(output, (w//2 - 200, h - 100), (w//2 + 200, h - 40), (0, 255, 0), -1)
                            cv2.putText(output, "READY TO SAVE!", 
                                       (w//2 - 120, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            else:
                # Waiting for person
                cv2.putText(output, "Waiting for person to enter frame...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show elapsed time
            elapsed = int(time.time() - start_time)
            cv2.putText(output, f"Time: {elapsed}s", 
                       (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Gait Profile Capture', output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC
                print("\n⏹ Capture cancelled")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏸ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Process and save profile
    print("\n" + "="*70)
    print("💾 PROCESSING CAPTURE")
    print("="*70 + "\n")
    
    if person_id_to_capture is None or person_id_to_capture not in gait_rec.gait_history:
        print("❌ ERROR: No person was tracked")
        print("   Make sure someone walks in front of the camera")
        sys.exit(1)
    
    frames_captured = len(gait_rec.gait_history[person_id_to_capture])
    
    if frames_captured < gait_rec.min_capture_frames:
        print(f"❌ ERROR: Not enough frames captured")
        print(f"   Captured: {frames_captured} frames")
        print(f"   Required: {gait_rec.min_capture_frames} frames")
        print("   Person needs to walk longer")
        sys.exit(1)
    
    # Compute signature with quality checks
    print("🔍 Computing gait signature...")
    signature = gait_rec.compute_gait_signature(person_id_to_capture, require_multiple_walks=True)
    
    if not signature:
        print("❌ ERROR: Could not compute gait signature")
        print("   Reasons could be:")
        print("   • Not enough walks detected (need 3-4 walks)")
        print("   • Poor quality frames")
        print("   • Person not walking naturally")
        sys.exit(1)
    
    # Display capture statistics
    print("\n✅ CAPTURE SUCCESSFUL!")
    print("\nStatistics:")
    print(f"   • Total frames: {signature['num_frames']}")
    print(f"   • Number of walks: {signature['num_walks']}")
    print(f"   • Quality score: {signature['quality_score']:.1%}")
    
    if signature['quality_score'] < 0.6:
        print("\n⚠ WARNING: Quality score is below optimal (60%)")
        print("   Profile will work but may be less accurate")
        proceed = input("\nProceed with saving? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Profile not saved")
            sys.exit(0)
    
    # Get profile information
    print("\n" + "="*70)
    print("👤 SUSPECT INFORMATION")
    print("="*70 + "\n")
    
    while True:
        name = input("Suspect Name: ").strip()
        if name:
            break
        print("❌ Name is required")
    
    description = input("Description (optional): ").strip()
    
    # Save profile
    print("\n💾 Saving profile...")
    success = gait_rec.add_gait_profile(name, signature, description)
    
    if success:
        print("\n" + "="*70)
        print("✅ GAIT PROFILE SAVED SUCCESSFULLY!")
        print("="*70)
        print(f"\n   Name: {name}")
        if description:
            print(f"   Description: {description}")
        print(f"   Frames: {signature['num_frames']}")
        print(f"   Walks: {signature['num_walks']}")
        print(f"   Quality: {signature['quality_score']:.1%}")
        print(f"\n   Profile valid for {gait_rec.profile_expiry_months} months")
        print("   Update profile every 3-6 months for best accuracy")
        print("\n" + "="*70 + "\n")
    else:
        print("\n❌ ERROR: Failed to save profile")
        sys.exit(1)

if __name__ == "__main__":
    main()