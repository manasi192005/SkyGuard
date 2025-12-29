"""
SkyGuard REST API Server
FastAPI backend for web/mobile apps
"""

from fastapi import FastAPI, WebSocket, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import json
from datetime import datetime
import asyncio

from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
from models.face_recognition_enhanced import FaceRecognitionEnhanced
from models.stampede_prediction_enhanced import StampedePredictorEnhanced
from models.emergency_detection_enhanced import EmergencyDetectorEnhanced
from models.database import init_database, get_session, get_recent_analytics

app = FastAPI(title="SkyGuard API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
db_engine = init_database('data/database/skyguard.db')
db_session = get_session(db_engine)

crowd_analyzer = CrowdAnalyzerEnhanced()
face_recognition = FaceRecognitionEnhanced()
stampede_predictor = StampedePredictorEnhanced()
emergency_detector = EmergencyDetectorEnhanced()

@app.get("/")
def root():
    return {
        "name": "SkyGuard API",
        "version": "1.0",
        "status": "operational",
        "features": ["heat_map", "face_recognition", "stampede_prediction", "emergency_detection"]
    }

@app.get("/api/status")
def get_status():
    return {
        "system": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "crowd_analysis": True,
            "face_recognition": True,
            "stampede_prediction": True,
            "emergency_detection": True
        }
    }

@app.get("/api/analytics")
def get_analytics(limit: int = 100):
    analytics = get_recent_analytics(db_session, limit)
    return {
        "count": len(analytics),
        "data": [
            {
                "timestamp": a.timestamp.isoformat(),
                "density": a.crowd_density,
                "risk_level": a.risk_level
            }
            for a in analytics
        ]
    }

@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output, count, risk, _, zones = crowd_analyzer.analyze_crowd(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', output)
            frame_data = buffer.tobytes()
            
            # Send data
            await websocket.send_json({
                "frame": frame_data.hex(),
                "crowd_count": count,
                "risk_level": risk,
                "zones": len(zones)
            })
            
            await asyncio.sleep(0.033)  # ~30 FPS
    
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
