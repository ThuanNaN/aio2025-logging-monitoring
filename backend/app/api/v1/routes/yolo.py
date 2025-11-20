import os
import io
import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from app.api.v1.controller.yolo import get_yolo_controller
from app.api.v1.detector.evidently_yolo_drift import get_yolo_drift_detector
from app.api.v1.routes.metrics import (
    image_brightness_metric, 
    brightness_histogram,
    gpu_allocated_metric,
    INFERENCE_COUNT,
    INFERENCE_LATENCY,
    DRIFT_EMBEDDING_DISTANCE,
    DRIFT_BRIGHTNESS,
    EVIDENTLY_DATASET_DRIFT,
    EVIDENTLY_DRIFT_SHARE,
    EVIDENTLY_NUM_DRIFTED_FEATURES,
    EVIDENTLY_BRIGHTNESS_DRIFT_SCORE,
    EVIDENTLY_CONFIDENCE_DRIFT_SCORE,
    EVIDENTLY_DETECTIONS_DRIFT_SCORE,
)


LOCAL_ARTIFACTS = Path("./DATA/artifacts")
LOCAL_CAPTURED = Path("./DATA/captured/yolo")

UPLOAD_DIR = LOCAL_CAPTURED / "uploads"
ANNOTATED_DIR = LOCAL_CAPTURED / "annotated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize controller and drift detector
yolo_controller = get_yolo_controller()
evidently_detector = get_yolo_drift_detector()

# Global variable for drift detection
reference_embedding = None


def load_model(model_path: str) -> YOLO:
    """Load YOLO model from file path"""
    try:
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model.to(DEVICE)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = LOCAL_ARTIFACTS / "yolo" / "yolo11x.pt"
    model = load_model(model_path)
    models["yolo"] = model
    # Set the model in the controller
    yolo_controller.set_model(model)
    yield
    models.clear()

router = APIRouter(lifespan=lifespan)


@router.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for object detection on uploaded image
    
    :param file: Uploaded image file
    :return: JSON response with detection results and file paths
    """
    global reference_embedding
    
    try:
        # Increment inference counter
        INFERENCE_COUNT.inc()
        
        # Start timing for latency
        start = time.time()
        
        # Generate unique filename
        original_filename = file.filename
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        
        # Full paths for saving
        original_filepath = os.path.join(UPLOAD_DIR, unique_filename)
        annotated_filepath = os.path.join(ANNOTATED_DIR, f"annotated_{unique_filename}")
        
        # Read and save original image
        contents = await file.read()
        with open(original_filepath, "wb") as f:
            f.write(contents)
        
        # Open image for detection
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get detection results from YOLO controller
        result = yolo_controller.detect_objects(
            image=pil_image,
            annotated_filepath=annotated_filepath
        )
        
        # Extract features
        features = result['features']
        brightness = features['brightness']
        avg_confidence = result['avg_confidence']
        embedding = features['embedding_features']
        
        # Update basic metrics
        gpu_allocated_metric.set(features['vram_allocated'])
        image_brightness_metric.set(brightness)
        brightness_histogram.observe(brightness)
        DRIFT_BRIGHTNESS.set(brightness)
        
        # Set baseline embedding on first run
        if reference_embedding is None:
            reference_embedding = embedding
        
        # Compute cosine similarity drift
        drift_score = 1 - cosine_similarity(reference_embedding, embedding)[0][0]
        DRIFT_EMBEDDING_DISTANCE.set(drift_score)
        
        # Add sample to Evidently detector
        evidently_detector.add_sample(
            brightness=brightness,
            num_detections=features['num_detections'],
            avg_confidence=avg_confidence,
            embedding_features=embedding
        )
        
        # Detect drift using Evidently
        drift_result = evidently_detector.detect_drift()
        
        # Update Prometheus metrics with Evidently results if drift was analyzed
        if drift_result.get('drift_share') is not None:
            EVIDENTLY_DATASET_DRIFT.set(1.0 if drift_result.get('dataset_drift', False) else 0.0)
            EVIDENTLY_DRIFT_SHARE.set(drift_result.get('drift_share', 0.0))
            EVIDENTLY_NUM_DRIFTED_FEATURES.set(drift_result.get('num_drifted_features', 0))
            
            # Update feature-specific drift scores
            feature_scores = drift_result.get('feature_drift_scores', {})
            if 'brightness' in feature_scores:
                EVIDENTLY_BRIGHTNESS_DRIFT_SCORE.set(feature_scores['brightness'].get('drift_score', 0.0))
            if 'avg_confidence' in feature_scores:
                EVIDENTLY_CONFIDENCE_DRIFT_SCORE.set(feature_scores['avg_confidence'].get('drift_score', 0.0))
            if 'num_detections' in feature_scores:
                EVIDENTLY_DETECTIONS_DRIFT_SCORE.set(feature_scores['num_detections'].get('drift_score', 0.0))
        
        # Record inference latency
        INFERENCE_LATENCY.observe(time.time() - start)
        
        # Prepare response
        response = {
            'detections': result['detections'],
            'total_objects': result['total_objects'],
            'device': result['device'],
            'inference_time': result['inference_time'],
            'brightness': brightness,
            'embedding_drift': float(drift_score),
            'avg_confidence': avg_confidence,
            'evidently_drift': drift_result
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@router.get("/drift/status")
async def get_drift_status():
    """
    Get current YOLO drift detection status
    
    Returns:
        JSON response with drift status
    """
    try:
        drift_summary = evidently_detector.get_drift_summary()
        return JSONResponse(content=drift_summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting drift status: {str(e)}")


@router.get("/drift/summary")
async def get_drift_summary():
    """
    Get detailed YOLO drift summary
    
    Returns:
        JSON response with detailed drift information
    """
    try:
        drift_summary = evidently_detector.get_drift_summary()
        stats = evidently_detector.get_stats()
        
        response = {
            "drift_summary": drift_summary,
            "detector_stats": stats
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting drift summary: {str(e)}")


@router.post("/drift/reset-reference")
async def reset_reference():
    """
    Reset reference dataset for YOLO drift detection
    
    Returns:
        JSON response confirming reset
    """
    try:
        evidently_detector.reset_reference()
        stats = evidently_detector.get_stats()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Reference dataset reset successfully",
            "detector_stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting reference: {str(e)}")


@router.get("/data-quality")
async def get_data_quality():
    """
    Get YOLO data quality report
    
    Returns:
        JSON response with data quality metrics
    """
    try:
        quality_report = evidently_detector.get_data_quality_report()
        return JSONResponse(content=quality_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quality report: {str(e)}")


@router.get("/model/info")
async def get_model_info():
    """
    Get YOLO model information
    
    Returns:
        JSON response with model details
    """
    try:
        model_info = yolo_controller.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")
    

@router.get("/health")
async def health_check():
    """
    Health check endpoint for YOLO service
    
    Returns:
        JSON response with service health status
    """
    try:
        model_info = yolo_controller.get_model_info()
        detector_stats = evidently_detector.get_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "yolo",
            "model": model_info,
            "drift_detector": {
                "reference_samples": detector_stats['reference_size'],
                "current_samples": detector_stats['current_size'],
                "drift_detected": detector_stats['drift_detected']
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "yolo",
                "error": str(e)
            }
        )