"""
main.py

FastAPI application for Ishaara ISL system.
Exposes endpoints for:
- WebSocket live inference (Sign -> Text)
- Text to ISL video stitching (Text -> Sign)

Author: Ishaara System
"""

import sys
import os
import torch
import numpy as np
import base64
import cv2
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

# Adjust path to import from siblings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.live_stream_extractor import LiveExtractor
from recognition.models.bilstm_encoder import BiLSTMEncoder
# from generation.gloss_to_video_stitcher import stitch_video (To be added in Phase 6)

app = FastAPI(title="Ishaara API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CLASS_NAMES = []
CONFIDENCE_THRESHOLD = 0.0  # Temporarily 0.0 so untrained model still returns predictions for demo
MAX_SEQ_LEN = 200

def load_model():
    global MODEL, CLASS_NAMES
    # Mocking loading for now if actual checkpoint doesn't exist
    # In production, load the trained BiLSTM checkpoint and class map
    # Here we initialize a random model if weights aren't found
    MODEL = BiLSTMEncoder(num_classes=263).to(DEVICE)
    MODEL.eval()
    
    # Normally read from class_to_idx.json
    CLASS_NAMES = [f"Sign_{i}" for i in range(263)]
    
@app.on_event("startup")
async def startup_event():
    load_model()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

def compute_motion(buffer):
    """
    Computes heuristic motion magnitude over the last few frames to detect end-of-sign.
    """
    if len(buffer) < 2:
        return 1.0
    # Average L2 norm of landmark differences between consecutive frames
    diffs = np.linalg.norm(buffer[-1] - buffer[-2])
    return diffs

@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    extractor = LiveExtractor()
    
    landmark_buffer = []
    motion_history = []
    
    # Motion threshold to determine end of sign
    MOTION_THRESHOLD = 0.15
    K_FRAMES = 5
    
    async def trigger_inference(buffer):
        input_tensor = torch.tensor(np.array([buffer]), dtype=torch.float32).to(DEVICE)
        mask = torch.ones((1, len(buffer)), dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(input_tensor, mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        conf_val = conf.item()
        pred_idx = pred.item()
        
        if conf_val >= CONFIDENCE_THRESHOLD:
            detected_sign = CLASS_NAMES[pred_idx]
        else:
            detected_sign = "NOT_CONFIDENT"
            
        await websocket.send_json({
            "status": "success",
            "sign": detected_sign,
            "confidence": conf_val
        })
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "frame" in message:
                # Decode base64 image
                img_data = base64.b64decode(message["frame"].split(',')[1])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract landmarks
                features, is_valid, raw_landmarks = extractor.process_frame(frame)
                
                if is_valid:
                    landmark_buffer.append(features)
                    motion = compute_motion(landmark_buffer)
                    motion_history.append(motion)
                    
                    if len(motion_history) > K_FRAMES:
                        motion_history.pop(0)
                        
                    # Send ONLY hand landmarks (indices 33 to 75) to frontend
                    hand_landmarks = raw_landmarks[33:75]
                    await websocket.send_json({
                        "status": "tracking",
                        "landmarks": hand_landmarks
                    })
                    
                    # Trigger inference if motion stops or buffer fills
                    if len(landmark_buffer) >= 8 and (np.mean(motion_history) < MOTION_THRESHOLD or len(landmark_buffer) >= MAX_SEQ_LEN):
                        await trigger_inference(landmark_buffer)
                        # Reset buffer after sign prediction
                        landmark_buffer = []
                        motion_history = []
                else:
                    # Hands dropped - trigger inference if we have enough frames buffered
                    if len(landmark_buffer) >= 8:
                        await trigger_inference(landmark_buffer)
                    
                    # Reset buffer because hands dropped
                    landmark_buffer = []
                    motion_history = []
                    
                    await websocket.send_json({
                        "status": "error",
                        "message": "NO_HAND_DETECTED"
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        extractor.close()
    except Exception as e:
        print(f"WS Error: {e}")
        manager.disconnect(websocket)
        extractor.close()

from pydantic import BaseModel
class TranslationRequest(BaseModel):
    glosses: List[str]

@app.post("/api/translate")
async def translate_glosses(req: TranslationRequest):
    # Mocking T5 Gloss-to-Text inference since model isn't trained yet
    joined = " ".join(req.glosses).upper()
    
    mock_dict = {
        "I APPLE EAT": "I am eating an apple.",
        "YOUR NAME WHAT": "What is your name?",
        "TOMORROW I MARKET GO": "I will go to the market tomorrow.",
        "BOY RUN": "The boy is running.",
        "I KNOW NOT": "I do not know."
    }
    
    if joined in mock_dict:
        return {"english": mock_dict[joined]}
    else:
        # Fallback pseudo-translation if not in mock dict
        return {"english": f"Translated sentence for: {joined.lower()}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
