from datetime import datetime
from typing import List
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
import structlog

from app.models.schemas import (
    PredictionRequest, PredictionResult, AnomalyResult,
    TrainingDataRequest, ScalingDecision, ScalingDirection, ScalingTrigger,
    MetricDataPoint,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/prediction", tags=["AI Prediction"])


@router.post("/predict", response_model=PredictionResult)
async def predict_load(body: PredictionRequest, request: Request):
    """
    Run the ML model to forecast CPU/memory and optimal instance count
    for the next N hours.
    """
    prediction_model = request.app.state.prediction_model
    monitoring = request.app.state.monitoring
    settings = request.app.state.settings
    cost_service = request.app.state.cost_service
    aws = request.app.state.aws

    if not prediction_model.is_trained:
        raise HTTPException(
            status_code=422,
            detail="Prediction model not yet trained. POST /prediction/train first.",
        )

    recent = monitoring.get_recent_data_points(hours=4)
    if len(recent) < 5:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient history ({len(recent)} points). Need at least 5.",
        )

    metric_objs = [_dict_to_metric_obj(p) for p in recent[-48:]]
    try:
        result = prediction_model.predict(
            metric_objs,
            horizon_hours=body.horizon_hours,
            include_confidence=body.include_confidence,
        )
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Build recommendation
    try:
        current_capacity = aws.describe_asg()["current_capacity"]
    except Exception:
        current_capacity = 1

    peak_instances = max(result["predicted_instance_count"])
    cost_analysis = cost_service.analyze(
        current_capacity, peak_instances, result["predicted_cpu"]
    )
    direction = (
        ScalingDirection.OUT if peak_instances > current_capacity
        else ScalingDirection.IN if peak_instances < current_capacity
        else ScalingDirection.NONE
    )
    recommendation = ScalingDecision(
        direction=direction,
        current_capacity=current_capacity,
        target_capacity=peak_instances,
        trigger=ScalingTrigger.PREDICTION,
        confidence=result["model_accuracy"],
        reason=f"ML forecast: peak demand needs {peak_instances} instances over {body.horizon_hours}h",
        estimated_cost_delta=cost_analysis["delta_hourly"] * body.horizon_hours,
    )

    return PredictionResult(
        timestamp=datetime.utcnow(),
        horizon_hours=body.horizon_hours,
        predicted_cpu=result["predicted_cpu"],
        predicted_memory=result["predicted_memory"],
        predicted_instance_count=result["predicted_instance_count"],
        confidence_lower=result.get("confidence_lower_cpu") if body.include_confidence else None,
        confidence_upper=result.get("confidence_upper_cpu") if body.include_confidence else None,
        model_accuracy=result["model_accuracy"],
        recommendation=recommendation,
    )


@router.post("/train")
async def train_models(
    body: TrainingDataRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Train (or retrain) the prediction and anomaly detection models
    with the provided historical data.
    """
    prediction_model = request.app.state.prediction_model
    anomaly_detector = request.app.state.anomaly_detector
    settings = request.app.state.settings

    if len(body.data_points) < settings.min_training_samples and not body.force_retrain:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Need {settings.min_training_samples} samples, "
                f"got {len(body.data_points)}. Set force_retrain=true to override."
            ),
        )

    async def _do_train():
        try:
            pred_result = prediction_model.train(body.data_points)
            anom_result = anomaly_detector.train(body.data_points)
            logger.info("models_trained_via_api", prediction=pred_result, anomaly=anom_result)
        except Exception as e:
            logger.error("api_training_failed", error=str(e))

    background_tasks.add_task(_do_train)
    return {
        "message": "Model training started in background.",
        "samples": len(body.data_points),
        "force_retrain": body.force_retrain,
    }


@router.post("/train/synthetic")
async def train_with_synthetic_data(
    request: Request,
    background_tasks: BackgroundTasks,
    n_points: int = 500,
):
    """Bootstrap the models with synthetic data (useful for demos/dev environments)."""
    from app.utils.helpers import generate_synthetic_training_data
    from app.models.schemas import MetricDataPoint as MDP

    if n_points < 50 or n_points > 10000:
        raise HTTPException(400, "n_points must be between 50 and 10000")

    prediction_model = request.app.state.prediction_model
    anomaly_detector = request.app.state.anomaly_detector
    monitoring = request.app.state.monitoring

    async def _do_train():
        raw = generate_synthetic_training_data(n_points=n_points)
        dps = [MDP(**p) for p in raw]
        # Also push to buffer so prediction endpoint has recent data
        for p in raw:
            monitoring.ingest_metric(p)
        try:
            pred_r = prediction_model.train(dps)
            anom_r = anomaly_detector.train(dps)
            logger.info("synthetic_training_complete", prediction=pred_r, anomaly=anom_r)
        except Exception as e:
            logger.error("synthetic_training_failed", error=str(e))

    background_tasks.add_task(_do_train)
    return {
        "message": f"Synthetic training started with {n_points} data points.",
        "estimated_completion": "a few seconds",
    }


@router.post("/anomaly/detect")
async def detect_anomaly(body: MetricDataPoint, request: Request):
    """Run anomaly detection on a single metric observation."""
    anomaly_detector = request.app.state.anomaly_detector
    if not anomaly_detector.is_trained:
        raise HTTPException(
            status_code=422,
            detail="Anomaly model not yet trained.",
        )
    result = anomaly_detector.detect(body)
    return AnomalyResult(
        timestamp=datetime.utcnow(),
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        affected_metrics=result["affected_metrics"],
        severity=result["severity"],
        recommended_action=result["recommended_action"],
    )


@router.post("/anomaly/batch")
async def detect_anomaly_batch(body: List[MetricDataPoint], request: Request):
    """Run anomaly detection on a sequence of observations."""
    anomaly_detector = request.app.state.anomaly_detector
    if not anomaly_detector.is_trained:
        raise HTTPException(status_code=422, detail="Anomaly model not yet trained.")
    results = anomaly_detector.detect_batch(body)
    return {
        "count": len(results),
        "anomalies_found": sum(1 for r in results if r["is_anomaly"]),
        "results": results,
    }


@router.get("/model/status")
async def model_status(request: Request):
    """Return current ML model health metrics."""
    pm = request.app.state.prediction_model
    ad = request.app.state.anomaly_detector
    fl = request.app.state.feedback_loop
    return {
        "prediction_model": {
            "trained": pm.is_trained,
            "last_trained": pm.last_trained.isoformat() if pm.last_trained else None,
            "training_samples": pm.training_samples,
            "cpu_mae": round(pm.cpu_mae, 3),
            "mem_mae": round(pm.mem_mae, 3),
            "r2_score": round(pm.r2_score, 4),
            "accuracy": round(pm.get_accuracy(), 4),
        },
        "anomaly_model": {
            "trained": ad.is_trained,
            "last_trained": ad.last_trained.isoformat() if ad.last_trained else None,
            "training_samples": ad.training_samples,
            "contamination_rate": ad.contamination,
        },
        "feedback_loop": {
            "next_retrain": fl.get_next_retrain_time().isoformat()
            if fl.get_next_retrain_time()
            else None,
        },
    }


@router.post("/feedback/retrain")
async def manual_retrain(request: Request, background_tasks: BackgroundTasks):
    """Force an immediate model retrain using buffered observations."""
    fl = request.app.state.feedback_loop
    background_tasks.add_task(fl.retrain_if_due)
    return {"message": "Retraining triggered in background."}


def _dict_to_metric_obj(d: dict) -> MetricDataPoint:
    return MetricDataPoint(
        timestamp=d.get("timestamp", datetime.utcnow()),
        cpu_utilization=float(d.get("cpu_utilization", 0.0)),
        memory_utilization=float(d.get("memory_utilization", 0.0)),
        network_in=float(d.get("network_in", 0.0)),
        network_out=float(d.get("network_out", 0.0)),
        request_count=int(d.get("request_count", 0)),
        instance_count=max(1, int(d.get("instance_count", 1))),
    )
