from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
import structlog

from app.models.schemas import (
    ScalingRequest, ScalingResponse, ScalingDecision,
    ScalingDirection, ScalingTrigger, ASGStatus, InstanceInfo, InstanceState,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/scaling", tags=["Auto Scaling"])


@router.get("/status", response_model=ASGStatus)
async def get_asg_status(request: Request):
    """Return current Auto Scaling Group state including all instances."""
    aws = request.app.state.aws
    try:
        data = aws.describe_asg()
        if data is None:
            raise HTTPException(status_code=404, detail="Auto Scaling Group not found")
        instances = [
            InstanceInfo(
                instance_id=i["instance_id"],
                state=_map_state(i["state"]),
                instance_type=i["instance_type"],
                launch_time=i.get("launch_time"),
                private_ip=i.get("private_ip"),
                public_ip=i.get("public_ip"),
                availability_zone=i.get("availability_zone", ""),
            )
            for i in data.get("instances", [])
        ]
        return ASGStatus(
            group_name=data["group_name"],
            min_size=data["min_size"],
            max_size=data["max_size"],
            desired_capacity=data["desired_capacity"],
            current_capacity=data["current_capacity"],
            instances=instances,
            health_check_type=data.get("health_check_type", "EC2"),
            created_time=data.get("created_time"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("asg_status_error", error=str(e))
        raise HTTPException(status_code=503, detail=f"AWS error: {e}")


@router.post("/scale", response_model=ScalingResponse)
async def manual_scale(body: ScalingRequest, request: Request):
    """Manually set the desired instance count."""
    scaling = request.app.state.scaling
    settings = request.app.state.settings

    if body.desired_capacity < settings.min_instances:
        raise HTTPException(
            400,
            f"desired_capacity must be ≥ min_instances ({settings.min_instances})",
        )
    if body.desired_capacity > settings.max_instances:
        raise HTTPException(
            400,
            f"desired_capacity must be ≤ max_instances ({settings.max_instances})",
        )

    try:
        decision = await scaling.scale_manually(body.desired_capacity, body.reason)
        return ScalingResponse(
            success=True,
            decision=decision,
            message=f"Scaling to {body.desired_capacity} instances initiated.",
        )
    except Exception as e:
        logger.error("manual_scale_error", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/evaluate", response_model=ScalingResponse)
async def trigger_evaluation(request: Request, background_tasks: BackgroundTasks):
    """Immediately run the AI scaling evaluation cycle."""
    scaling = request.app.state.scaling
    try:
        decision = await scaling.evaluate_and_scale()
        return ScalingResponse(
            success=True,
            decision=decision,
            message="Evaluation complete. " + decision.reason,
        )
    except Exception as e:
        logger.error("evaluation_error", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/setup-alarms")
async def setup_cloudwatch_alarms(request: Request):
    """Create the standard CPU scale-out / scale-in CloudWatch alarms."""
    aws = request.app.state.aws
    settings = request.app.state.settings
    try:
        # Create simple scaling policies first, then wire alarms to them
        scale_out_arn = aws.create_scaling_policy(
            "autoscale-ai-scale-out", adjustment=2,
            cooldown=settings.scale_out_cooldown_seconds,
        )
        scale_in_arn = aws.create_scaling_policy(
            "autoscale-ai-scale-in", adjustment=-1,
            cooldown=settings.scale_in_cooldown_seconds,
        )
        aws.setup_standard_alarms(scale_out_arn, scale_in_arn)
        return {"message": "CloudWatch alarms configured", "scale_out_arn": scale_out_arn}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/cost")
async def get_cost_analysis(request: Request):
    """Return current cost analysis and optimization suggestions."""
    cost_service = request.app.state.cost_service
    monitoring = request.app.state.monitoring
    aws = request.app.state.aws

    recent = monitoring.get_recent_data_points(hours=2)
    cpu_values = [p.get("cpu_utilization", 50.0) for p in recent]
    try:
        asg = aws.describe_asg()
        current = asg["current_capacity"] if asg else 1
    except Exception:
        current = 1

    return cost_service.analyze(
        current_instances=current,
        proposed_instances=current,
        predicted_cpu=cpu_values,
    )


@router.get("/cost/report")
async def get_cost_report(request: Request, hours: int = 24):
    """Return a scaling activity cost report for the last N hours."""
    cost_service = request.app.state.cost_service
    return cost_service.get_scaling_cost_report(hours=hours)


def _map_state(raw: str) -> InstanceState:
    mapping = {
        "InService": InstanceState.RUNNING,
        "Pending": InstanceState.PENDING,
        "Terminating": InstanceState.STOPPING,
        "Terminated": InstanceState.TERMINATED,
        "running": InstanceState.RUNNING,
        "stopped": InstanceState.STOPPED,
        "pending": InstanceState.PENDING,
        "stopping": InstanceState.STOPPING,
        "terminated": InstanceState.TERMINATED,
    }
    return mapping.get(raw, InstanceState.RUNNING)
