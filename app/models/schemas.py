from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ScalingDirection(str, Enum):
    OUT = "scale_out"
    IN = "scale_in"
    NONE = "no_action"


class ScalingTrigger(str, Enum):
    CPU = "cpu_utilization"
    MEMORY = "memory_utilization"
    PREDICTION = "ml_prediction"
    ANOMALY = "anomaly_detection"
    MANUAL = "manual"
    COST = "cost_optimization"


class InstanceState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"


# ── Request Schemas ──────────────────────────────────────────────────────────

class ScalingRequest(BaseModel):
    desired_capacity: int = Field(..., ge=1, le=100, description="Target number of instances")
    reason: str = Field(default="Manual scaling request")
    trigger: ScalingTrigger = Field(default=ScalingTrigger.MANUAL)


class MetricDataPoint(BaseModel):
    timestamp: datetime
    cpu_utilization: float = Field(..., ge=0.0, le=100.0)
    memory_utilization: float = Field(..., ge=0.0, le=100.0)
    network_in: float = Field(default=0.0, ge=0.0)
    network_out: float = Field(default=0.0, ge=0.0)
    request_count: int = Field(default=0, ge=0)
    instance_count: int = Field(default=1, ge=1)


class TrainingDataRequest(BaseModel):
    data_points: List[MetricDataPoint]
    force_retrain: bool = Field(default=False)


class PredictionRequest(BaseModel):
    horizon_hours: int = Field(default=2, ge=1, le=24)
    include_confidence: bool = Field(default=True)


class AlarmConfigRequest(BaseModel):
    alarm_name: str
    metric_name: str
    threshold: float
    comparison_operator: str = Field(default="GreaterThanThreshold")
    evaluation_periods: int = Field(default=2)
    period_seconds: int = Field(default=300)


# ── Response Schemas ─────────────────────────────────────────────────────────

class InstanceInfo(BaseModel):
    instance_id: str
    state: InstanceState
    instance_type: str
    launch_time: Optional[datetime]
    private_ip: Optional[str]
    public_ip: Optional[str]
    availability_zone: str


class ASGStatus(BaseModel):
    group_name: str
    min_size: int
    max_size: int
    desired_capacity: int
    current_capacity: int
    instances: List[InstanceInfo]
    health_check_type: str
    created_time: Optional[datetime]


class ScalingDecision(BaseModel):
    direction: ScalingDirection
    current_capacity: int
    target_capacity: int
    trigger: ScalingTrigger
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    estimated_cost_delta: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScalingResponse(BaseModel):
    success: bool
    decision: ScalingDecision
    message: str
    activity_id: Optional[str] = None


class MetricSummary(BaseModel):
    metric_name: str
    current_value: float
    average_1h: float
    average_24h: float
    peak_24h: float
    trend: str


class MonitoringStatus(BaseModel):
    timestamp: datetime
    asg_status: ASGStatus
    metrics: List[MetricSummary]
    active_alarms: List[str]
    system_healthy: bool


class PredictionResult(BaseModel):
    timestamp: datetime
    horizon_hours: int
    predicted_cpu: List[float]
    predicted_memory: List[float]
    predicted_instance_count: List[int]
    confidence_lower: Optional[List[float]]
    confidence_upper: Optional[List[float]]
    model_accuracy: float
    recommendation: ScalingDecision


class AnomalyResult(BaseModel):
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    affected_metrics: List[str]
    severity: str
    recommended_action: ScalingDirection


class CostAnalysis(BaseModel):
    current_hourly_cost: float
    projected_daily_cost: float
    projected_monthly_cost: float
    potential_savings: float
    optimization_suggestions: List[str]
    cost_per_request: Optional[float]


class ModelStatus(BaseModel):
    prediction_model_trained: bool
    anomaly_model_trained: bool
    last_trained: Optional[datetime]
    training_samples: int
    prediction_accuracy: float
    next_retrain: Optional[datetime]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    aws_connected: bool
    model_status: ModelStatus
    uptime_seconds: float
