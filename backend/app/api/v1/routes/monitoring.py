"""
Monitoring and drift detection endpoints using Evidently
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.api.v1.routes.yolo import evidently_detector


router = APIRouter()


@router.get("/drift/status")
async def get_drift_status():
    """
    Get current drift detection status from Evidently
    
    Returns:
        JSON response with drift status and statistics
    """
    # Get drift detection result
    drift_result = evidently_detector.detect_drift(window_size=50)
    
    # Get summary statistics
    stats = evidently_detector.get_summary_statistics()
    
    if drift_result is None:
        return JSONResponse(content={
            'status': 'insufficient_data',
            'message': 'Not enough data collected for drift detection',
            'statistics': stats,
        })
    
    return JSONResponse(content={
        'status': 'success',
        'drift_detection': drift_result,
        'statistics': stats,
    })


@router.get("/drift/summary")
async def get_drift_summary():
    """
    Get summary statistics of the collected data
    
    Returns:
        JSON response with summary statistics
    """
    stats = evidently_detector.get_summary_statistics()
    
    return JSONResponse(content={
        'status': 'success',
        'statistics': stats,
    })


@router.post("/drift/reset-reference")
async def reset_reference_data():
    """
    Reset the reference data to use current data as new baseline
    
    Returns:
        JSON response confirming the reset
    """
    try:
        evidently_detector.reset_reference()
        return JSONResponse(content={
            'status': 'success',
            'message': 'Reference data has been reset to current data',
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': f'Failed to reset reference data: {str(e)}',
            }
        )


@router.get("/data-quality")
async def get_data_quality():
    """
    Get data quality report from Evidently
    
    Returns:
        JSON response with data quality metrics
    """
    quality_report = evidently_detector.get_data_quality_report(window_size=50)
    
    if quality_report is None:
        return JSONResponse(content={
            'status': 'insufficient_data',
            'message': 'Not enough data collected for quality report',
        })
    
    return JSONResponse(content={
        'status': 'success',
        'data_quality': quality_report,
    })
