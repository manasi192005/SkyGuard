"""
Complete Testing Suite for DeepFace Face Recognition
Tests accuracy, suggests optimal thresholds, and provides detailed diagnostics
"""

import cv2
import os
import sys
import numpy as np
import json

# Import your face recognition module
try:
    from models.face_recognition_deepface_fixed import FaceRecognitionDeepFaceFixed
except ImportError:
    print("❌ Cannot import FaceRecognitionDeepFaceFixed")
    print("Make sure the file is in models/ folder")
    sys.exit(1)


class RecognitionTester:
    """Comprehensive testing for face recognition"""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.test_results = {
            "image_tests": [],
            "video_tests": [],
            "threshold_analysis": []
        }

    # ---------------------------------------------------------
    # IMAGE TEST
    # ---------------------------------------------------------
    def test_single_image(self, image_path, expected_name=None):
        print("\n" + "=" * 70)
        print(f"🖼️ TESTING IMAGE: {image_path}")
        print("=" * 70)

        if not os.path.isfile(image_path):
            print(f"❌ Image not found: {image_path}")
            return None

        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Failed to load image")
            return None

        print(f"✓ Image loaded: {frame.shape[1]}x{frame.shape[0]}")

        output, faces, matches = self.recognizer.process_frame(
            frame,
            draw_boxes=True,
            frame_id=1
        )

        print("\n📊 RESULTS:")
        print(f" Faces detected: {len(faces)}")
        print(f" Matches found: {len(matches)}")

        if matches:
            for i, match in enumerate(matches, 1):
                print(f"\n Match {i}:")
                print(f"  Name: {match['name']}")
                print(f"  Confidence: {match['confidence']:.1%}")
                print(f"  Distance: {match['distance']:.4f}")

                if expected_name:
                    correct = match["name"].lower() == expected_name.lower()
                    print(f"  Status: {'✅ CORRECT' if correct else '❌ WRONG'}")
        else:
            print(" ⚠️ No matches found")

        output_path = image_path.rsplit(".", 1)[0] + "_result.jpg"
        cv2.imwrite(output_path, output)
        print(f"\n💾 Saved result: {output_path}")

        cv2.imshow("Test Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.test_results["image_tests"].append({
            "image": image_path,
            "expected": expected_name,
            "faces_detected": len(faces),
            "matches_found": len(matches),
            "matches": matches
        })

    # ---------------------------------------------------------
    # VIDEO TEST
    # ---------------------------------------------------------
    def test_video(self, video_path, process_every_n_frames=5, max_frames=300):
        print("\n" + "=" * 70)
        print(f"🎬 TESTING VIDEO: {video_path}")
        print("=" * 70)

        if not os.path.isfile(video_path):
            print(f"❌ Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return None

        processed = 0
        frame_id = 0
        total_faces = 0
        total_matches = 0
        match_history = {}

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % process_every_n_frames != 0:
                continue

            processed += 1
            output, faces, matches = self.recognizer.process_frame(
                frame,
                draw_boxes=True,
                frame_id=frame_id
            )

            total_faces += len(faces)
            total_matches += len(matches)

            for m in matches:
                name = m["name"]
                match_history.setdefault(name, []).append(m["confidence"])

            if processed % 10 == 0:
                print(f"Frame {frame_id}: {len(faces)} faces, {len(matches)} matches")

            if processed % 30 == 0:
                display = cv2.resize(output, (960, 540))
                cv2.imshow("Video Test", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

        print("\n📊 VIDEO SUMMARY")
        print(f"Processed frames: {processed}")
        print(f"Total faces: {total_faces}")
        print(f"Total matches: {total_matches}")

        self.test_results["video_tests"].append({
            "video": video_path,
            "processed_frames": processed,
            "faces": total_faces,
            "matches": total_matches,
            "match_history": match_history
        })

    # ---------------------------------------------------------
    # THRESHOLD SWEEP
    # ---------------------------------------------------------
    def test_threshold_sweep(self, image_path, expected_name):
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Cannot load test image")
            return

        original = self.recognizer.confidence_threshold
        results = []

        for t in thresholds:
            self.recognizer.confidence_threshold = t
            _, _, matches = self.recognizer.process_frame(frame, False, 0)

            matched = bool(matches)
            correct = matched and matches[0]["name"].lower() == expected_name.lower()

            print(
                f"{'✅' if correct else '❌' if matched else '⚪'} "
                f"Threshold {t:.2f}: "
                f"{'MATCH' if matched else 'NO MATCH'}"
            )

            results.append({
                "threshold": t,
                "matched": matched,
                "correct": correct
            })

        self.recognizer.confidence_threshold = original
        self.test_results["threshold_analysis"].append(results)

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------
    def save_test_report(self, filename="test_report.json"):
        with open(filename, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Report saved to {filename}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("🧪 SKYGUARD FACE RECOGNITION TEST SUITE")
    print("=" * 70)

    recognizer = FaceRecognitionDeepFaceFixed(
        suspects_db_path="data/suspects",
        confidence_threshold=0.50,
        debug_mode=True
    )

    tester = RecognitionTester(recognizer)

    while True:
        print("\nMENU")
        print("1. Test single image")
        print("2. Test video")
        print("3. Threshold sweep")
        print("4. Save report")
        print("0. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            img = input("Image path: ").strip()
            name = input("Expected name (optional): ").strip() or None
            tester.test_single_image(img, name)

        elif choice == "2":
            vid = input("Video path: ").strip()
            tester.test_video(vid)

        elif choice == "3":
            img = input("Image path: ").strip()
            name = input("Expected name: ").strip()
            tester.test_threshold_sweep(img, name)

        elif choice == "4":
            tester.save_test_report()

        elif choice == "0":
            print("👋 Exiting")
            break

        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
