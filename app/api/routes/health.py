from datetime import datetime
from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse, ModelStatus

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check(request: Request):
    state = request.app.state
    prediction_model = state.prediction_model
    anomaly_detector = state.anomaly_detector
    scheduler = state.scheduler
    aws = state.aws

    aws_ok = await aws.check_connectivity()

    model_status = ModelStatus(
        prediction_model_trained=prediction_model.is_trained,
        anomaly_model_trained=anomaly_detector.is_trained,
        last_trained=prediction_model.last_trained,
        training_samples=prediction_model.training_samples,
        prediction_accuracy=prediction_model.get_accuracy(),
        next_retrain=state.feedback_loop.get_next_retrain_time(),
    )

    return HealthResponse(
        status="healthy" if aws_ok else "degraded",
        version=state.settings.app_version,
        timestamp=datetime.utcnow(),
        aws_connected=aws_ok,
        model_status=model_status,
        uptime_seconds=scheduler.uptime_seconds,
    )


@router.get("/ping")
async def ping():
    return {"pong": True, "timestamp": datetime.utcnow().isoformat()}
