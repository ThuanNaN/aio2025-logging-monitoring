"""
VQA API Routes
Endpoints for Visual Question Answering with BLIP and Evidently drift detection
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.api.v1.controller.blip import get_blip_controller
from app.api.v1.detector.evidently_vqa_drift import get_vqa_drift_detector
from app.api.v1.routes.metrics import (
    VQA_INFERENCE_COUNT,
    VQA_INFERENCE_LATENCY,
    VQA_QUESTION_LENGTH,
    VQA_ANSWER_LENGTH,
    VQA_QUESTION_TYPE,
    VQA_EVIDENTLY_DATASET_DRIFT,
    VQA_EVIDENTLY_DRIFT_SHARE,
    VQA_EVIDENTLY_NUM_DRIFTED_FEATURES,
    VQA_EVIDENTLY_BRIGHTNESS_DRIFT_SCORE,
    VQA_EVIDENTLY_QUESTION_LENGTH_DRIFT_SCORE,
    VQA_EVIDENTLY_ANSWER_LENGTH_DRIFT_SCORE,
    VQA_EVIDENTLY_INFERENCE_TIME_DRIFT_SCORE
)

router = APIRouter()

# Initialize controller and drift detector
blip_controller = get_blip_controller()
vqa_drift_detector = get_vqa_drift_detector()


@router.post("/answer")
async def answer_question(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_length: int = Form(50),
    num_beams: int = Form(5)
):
    """
    Answer a question about an image using BLIP
    
    Args:
        image: Image file
        question: Question about the image
        max_length: Maximum answer length
        num_beams: Number of beams for beam search
        
    Returns:
        JSON response with answer and drift information
    """
    try:
        # Read and process image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get answer from BLIP
        result = blip_controller.answer_question(
            image=pil_image,
            question=question,
            max_length=max_length,
            num_beams=num_beams
        )
        
        # Extract features
        features = result['features']
        features['inference_time'] = result['inference_time']
        
        # Update Prometheus metrics
        VQA_INFERENCE_COUNT.inc()
        VQA_INFERENCE_LATENCY.observe(result['inference_time'])
        VQA_QUESTION_LENGTH.observe(features['question_length'])
        VQA_ANSWER_LENGTH.observe(features['answer_length'])
        VQA_QUESTION_TYPE.labels(question_type=features['question_type']).inc()
        
        # Add to drift detector
        vqa_drift_detector.add_sample(features)
        
        # Detect drift
        drift_result = vqa_drift_detector.detect_drift()
        
        # Update Evidently drift metrics if drift was analyzed
        if drift_result.get('drift_share') is not None:
            VQA_EVIDENTLY_DATASET_DRIFT.set(1 if drift_result.get('dataset_drift', False) else 0)
            VQA_EVIDENTLY_DRIFT_SHARE.set(drift_result.get('drift_share', 0.0))
            VQA_EVIDENTLY_NUM_DRIFTED_FEATURES.set(drift_result.get('num_drifted_features', 0))
            
            # Update feature-level drift scores
            feature_scores = drift_result.get('feature_drift_scores', {})
            if 'brightness' in feature_scores:
                VQA_EVIDENTLY_BRIGHTNESS_DRIFT_SCORE.set(feature_scores['brightness'].get('drift_score', 0.0))
            if 'question_length' in feature_scores:
                VQA_EVIDENTLY_QUESTION_LENGTH_DRIFT_SCORE.set(feature_scores['question_length'].get('drift_score', 0.0))
            if 'answer_length' in feature_scores:
                VQA_EVIDENTLY_ANSWER_LENGTH_DRIFT_SCORE.set(feature_scores['answer_length'].get('drift_score', 0.0))
            if 'inference_time' in feature_scores:
                VQA_EVIDENTLY_INFERENCE_TIME_DRIFT_SCORE.set(feature_scores['inference_time'].get('drift_score', 0.0))
        
        # Prepare response
        response = {
            "question": question,
            "answer": result['answer'],
            "inference_time": result['inference_time'],
            "model_name": result['model_name'],
            "device": result['device'],
            "features": features,
            "evidently_drift": drift_result
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing VQA request: {str(e)}")


@router.get("/drift/status")
async def get_drift_status():
    """
    Get current VQA drift detection status
    
    Returns:
        JSON response with drift status
    """
    try:
        drift_summary = vqa_drift_detector.get_drift_summary()
        return JSONResponse(content=drift_summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting drift status: {str(e)}")


@router.get("/drift/summary")
async def get_drift_summary():
    """
    Get detailed VQA drift summary
    
    Returns:
        JSON response with detailed drift information
    """
    try:
        drift_summary = vqa_drift_detector.get_drift_summary()
        stats = vqa_drift_detector.get_stats()
        
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
    Reset reference dataset for VQA drift detection
    
    Returns:
        JSON response confirming reset
    """
    try:
        vqa_drift_detector.reset_reference()
        stats = vqa_drift_detector.get_stats()
        
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
    Get VQA data quality report
    
    Returns:
        JSON response with data quality metrics
    """
    try:
        quality_report = vqa_drift_detector.get_data_quality_report()
        return JSONResponse(content=quality_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quality report: {str(e)}")


@router.get("/model/info")
async def get_model_info():
    """
    Get BLIP model information
    
    Returns:
        JSON response with model details
    """
    try:
        model_info = blip_controller.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for VQA service
    
    Returns:
        JSON response with service health status
    """
    try:
        model_info = blip_controller.get_model_info()
        detector_stats = vqa_drift_detector.get_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "vqa",
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
                "service": "vqa",
                "error": str(e)
            }
        )
