from fastapi import APIRouter
from app.api.v1.routes.yolo import router as yolo_router
from app.api.v1.routes.monitoring import router as monitoring_router
from app.api.v1.routes.vqa import router as vqa_router

router = APIRouter()

# Health Check
@router.get("/health")
async def health_check():
    return {"status": "ok"}

# Include the v1 router
router.include_router(yolo_router, prefix="/yolo", tags=["Detect"])
router.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
router.include_router(vqa_router, prefix="/vqa", tags=["VQA"])
