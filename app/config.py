from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # AWS
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")
    aws_region: str = Field(default="us-east-1")

    # Auto Scaling
    auto_scaling_group_name: str = Field(default="autoscale-ai-asg")
    min_instances: int = Field(default=1)
    max_instances: int = Field(default=20)
    desired_capacity: int = Field(default=2)

    # CloudWatch
    cloudwatch_namespace: str = Field(default="AutoScaleAI/Metrics")
    cpu_scale_out_threshold: float = Field(default=80.0)
    cpu_scale_in_threshold: float = Field(default=20.0)
    memory_scale_out_threshold: float = Field(default=85.0)
    alarm_evaluation_periods: int = Field(default=2)
    alarm_period_seconds: int = Field(default=300)

    # ML Model
    model_retrain_interval_hours: int = Field(default=6)
    prediction_horizon_hours: int = Field(default=2)
    anomaly_contamination: float = Field(default=0.05)
    min_training_samples: int = Field(default=100)

    # App
    app_name: str = Field(default="AutoScaleAI")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    api_prefix: str = Field(default="/api/v1")

    # Cost
    cost_per_instance_hour: float = Field(default=0.096)
    scale_in_cooldown_seconds: int = Field(default=300)
    scale_out_cooldown_seconds: int = Field(default=180)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
