"""
FastAPI Application for Emotion AI
Provides REST API endpoints for emotion detection and keypoint prediction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import base64
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.emotion_model import EmotionClassifier
from src.models.keypoint_model import KeypointDetector

# Initialize FastAPI app
app = FastAPI(
    title="Emotion AI API",
    description="REST API for emotion detection and facial keypoint prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models (update paths to your saved models)
emotion_model = None
keypoint_model = None


# Pydantic models for request/response
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    model_version: str = "1"


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]


class KeypointResponse(BaseModel):
    keypoints: List[Dict[str, float]]
    num_keypoints: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global emotion_model, keypoint_model
    
    try:
        # Load emotion classifier
        emotion_model = EmotionClassifier()
        emotion_model.compile_model()
        print("✓ Emotion classifier loaded")
    except Exception as e:
        print(f"✗ Failed to load emotion classifier: {e}")
    
    try:
        # Load keypoint detector
        keypoint_model = KeypointDetector()
        keypoint_model.compile_model()
        print("✓ Keypoint detector loaded")
    except Exception as e:
        print(f"✗ Failed to load keypoint detector: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "models_loaded": {
            "emotion_classifier": emotion_model is not None,
            "keypoint_detector": keypoint_model is not None
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if (emotion_model and keypoint_model) else "degraded",
        "models_loaded": {
            "emotion_classifier": emotion_model is not None,
            "keypoint_detector": keypoint_model is not None
        }
    }


def decode_image(base64_string: str, target_size: tuple):
    """Decode base64 image and resize"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/api/v1/predict/emotion", response_model=EmotionResponse)
async def predict_emotion(request: ImageRequest):
    """
    Predict emotion from facial image
    
    Args:
        request: ImageRequest with base64 encoded image
        
    Returns:
        EmotionResponse with predicted emotion and probabilities
    """
    if emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded")
    
    try:
        # Decode and preprocess image
        image = decode_image(request.image, (48, 48))
        
        # Predict
        result = emotion_model.predict(image)
        
        return EmotionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/keypoints", response_model=KeypointResponse)
async def predict_keypoints(request: ImageRequest):
    """
    Predict facial keypoints from image
    
    Args:
        request: ImageRequest with base64 encoded image
        
    Returns:
        KeypointResponse with keypoint coordinates
    """
    if keypoint_model is None:
        raise HTTPException(status_code=503, detail="Keypoint model not loaded")
    
    try:
        # Decode and preprocess image
        image = decode_image(request.image, (96, 96))
        
        # Predict
        keypoints = keypoint_model.predict(image)
        
        # Format response
        keypoint_list = [
            {"x": float(kp[0]), "y": float(kp[1])}
            for kp in keypoints
        ]
        
        return KeypointResponse(
            keypoints=keypoint_list,
            num_keypoints=len(keypoint_list)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/both")
async def predict_both(request: ImageRequest):
    """
    Predict both emotion and keypoints from image
    
    Args:
        request: ImageRequest with base64 encoded image
        
    Returns:
        Combined response with emotion and keypoints
    """
    if emotion_model is None or keypoint_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Decode images
        emotion_img = decode_image(request.image, (48, 48))
        keypoint_img = decode_image(request.image, (96, 96))
        
        # Predict
        emotion_result = emotion_model.predict(emotion_img)
        keypoints = keypoint_model.predict(keypoint_img)
        
        # Format response
        keypoint_list = [
            {"x": float(kp[0]), "y": float(kp[1])}
            for kp in keypoints
        ]
        
        return {
            "emotion": emotion_result,
            "keypoints": {
                "keypoints": keypoint_list,
                "num_keypoints": len(keypoint_list)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded file
    
    Args:
        file: Uploaded image file
        
    Returns:
        Emotion prediction result
    """
    if emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded")
    
    try:
        # Read file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img = cv2.resize(img, (48, 48))
        
        # Predict
        result = emotion_model.predict(img)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def main():
    """Run the API server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
