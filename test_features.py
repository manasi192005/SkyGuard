"""Test enhanced features"""
import cv2

print("Testing Feature 1: Density Heat Map")
from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
analyzer = CrowdAnalyzerEnhanced()
print("✓ Heat map ready\n")

print("Testing Feature 2: Face Recognition")
from models.face_recognition_enhanced import FaceRecognitionEnhanced
face_rec = FaceRecognitionEnhanced()
print(f"✓ Face recognition ready ({len(face_rec.suspects)} suspects)\n")

print("All features loaded successfully!")

# Quick test script

"""Test enhanced features"""
import cv2

print("Testing Feature 1: Density Heat Map")
from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
analyzer = CrowdAnalyzerEnhanced()
print("✓ Heat map ready\n")

print("Testing Feature 2: Face Recognition")
from models.face_recognition_enhanced import FaceRecognitionEnhanced
face_rec = FaceRecognitionEnhanced()
print(f"✓ Face recognition ready ({len(face_rec.suspects)} suspects)\n")

print("All features loaded successfully!")
