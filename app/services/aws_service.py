"""
AWS Service Layer
Central Boto3 wrapper for EC2, Auto Scaling, and CloudWatch operations.
Uses tenacity for retry logic on transient AWS API failures.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import boto3
import structlog
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import Settings

logger = structlog.get_logger(__name__)

RETRYABLE = (ClientError, BotoCoreError)


class AWSService:
    """Thread-safe Boto3 client manager with retry and error handling."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
        )
        self._ec2 = self._session.client("ec2")
        self._asg = self._session.client("autoscaling")
        self._cw = self._session.client("cloudwatch")
        self._asg_name = settings.auto_scaling_group_name

    # ── Connectivity ─────────────────────────────────────────────────────────

    async def check_connectivity(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._ec2.describe_regions)
            return True
        except Exception as e:
            logger.warning("aws_connectivity_check_failed", error=str(e))
            return False

    # ── EC2 ──────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )
    def describe_instances(self, filters: Optional[List[Dict]] = None) -> List[Dict]:
        params: Dict[str, Any] = {}
        if filters:
            params["Filters"] = filters
        response = self._ec2.describe_instances(**params)
        instances = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                instances.append({
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "instance_type": inst.get("InstanceType", "unknown"),
                    "launch_time": inst.get("LaunchTime"),
                    "private_ip": inst.get("PrivateIpAddress"),
                    "public_ip": inst.get("PublicIpAddress"),
                    "availability_zone": inst.get("Placement", {}).get("AvailabilityZone", ""),
                })
        return instances

    # ── Auto Scaling Group ────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def describe_asg(self) -> Optional[Dict]:
        response = self._asg.describe_auto_scaling_groups(
            AutoScalingGroupNames=[self._asg_name]
        )
        groups = response.get("AutoScalingGroups", [])
        if not groups:
            return None
        g = groups[0]
        instances = []
        for inst in g.get("Instances", []):
            instances.append({
                "instance_id": inst["InstanceId"],
                "state": inst["LifecycleState"],
                "instance_type": inst.get("InstanceType", "unknown"),
                "launch_time": None,
                "private_ip": None,
                "public_ip": None,
                "availability_zone": inst.get("AvailabilityZone", ""),
            })
        return {
            "group_name": g["AutoScalingGroupName"],
            "min_size": g["MinSize"],
            "max_size": g["MaxSize"],
            "desired_capacity": g["DesiredCapacity"],
            "current_capacity": len([i for i in g["Instances"] if i["LifecycleState"] == "InService"]),
            "instances": instances,
            "health_check_type": g.get("HealthCheckType", "EC2"),
            "created_time": g.get("CreatedTime"),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def set_desired_capacity(self, capacity: int) -> Dict:
        """Scale the ASG to the given desired capacity."""
        clamped = max(
            self.settings.min_instances,
            min(capacity, self.settings.max_instances),
        )
        self._asg.set_desired_capacity(
            AutoScalingGroupName=self._asg_name,
            DesiredCapacity=clamped,
            HonorCooldown=True,
        )
        logger.info("asg_capacity_set", desired=clamped, requested=capacity)
        return {"new_desired_capacity": clamped}

    def create_auto_scaling_group(
        self,
        launch_template_id: str,
        vpc_zone_ids: List[str],
        target_group_arns: Optional[List[str]] = None,
    ) -> Dict:
        """Provision a new Auto Scaling Group with sensible defaults."""
        params: Dict[str, Any] = {
            "AutoScalingGroupName": self._asg_name,
            "LaunchTemplate": {
                "LaunchTemplateId": launch_template_id,
                "Version": "$Latest",
            },
            "MinSize": self.settings.min_instances,
            "MaxSize": self.settings.max_instances,
            "DesiredCapacity": self.settings.desired_capacity,
            "VPCZoneIdentifier": ",".join(vpc_zone_ids),
            "HealthCheckType": "ELB",
            "HealthCheckGracePeriod": 300,
            "DefaultCooldown": self.settings.scale_out_cooldown_seconds,
            "Tags": [
                {
                    "Key": "ManagedBy",
                    "Value": "AutoScaleAI",
                    "PropagateAtLaunch": True,
                    "ResourceId": self._asg_name,
                    "ResourceType": "auto-scaling-group",
                }
            ],
        }
        if target_group_arns:
            params["TargetGroupARNs"] = target_group_arns

        self._asg.create_auto_scaling_group(**params)
        logger.info("asg_created", name=self._asg_name)
        return {"asg_name": self._asg_name, "status": "created"}

    def create_scaling_policy(
        self,
        policy_name: str,
        adjustment: int,
        cooldown: int,
        policy_type: str = "SimpleScaling",
    ) -> str:
        response = self._asg.put_scaling_policy(
            AutoScalingGroupName=self._asg_name,
            PolicyName=policy_name,
            PolicyType=policy_type,
            AdjustmentType="ChangeInCapacity",
            ScalingAdjustment=adjustment,
            Cooldown=cooldown,
        )
        arn = response["PolicyARN"]
        logger.info("scaling_policy_created", policy=policy_name, arn=arn)
        return arn

    # ── CloudWatch ────────────────────────────────────────────────────────────

    def get_metric_statistics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: Optional[List[str]] = None,
        dimensions: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        if statistics is None:
            statistics = ["Average", "Maximum"]
        params: Dict[str, Any] = {
            "Namespace": "AWS/EC2",
            "MetricName": metric_name,
            "StartTime": start_time,
            "EndTime": end_time,
            "Period": period,
            "Statistics": statistics,
        }
        if dimensions:
            params["Dimensions"] = dimensions
        response = self._cw.get_metric_statistics(**params)
        return sorted(
            response.get("Datapoints", []),
            key=lambda x: x["Timestamp"],
        )

    def get_asg_cpu_metrics(self, hours: int = 1) -> List[Dict]:
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        return self.get_metric_statistics(
            "CPUUtilization",
            start_time=start,
            end_time=end,
            dimensions=[{"Name": "AutoScalingGroupName", "Value": self._asg_name}],
        )

    def put_metric_data(
        self,
        metric_name: str,
        value: float,
        unit: str = "Percent",
        dimensions: Optional[List[Dict]] = None,
    ):
        metric_data: Dict[str, Any] = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.utcnow(),
        }
        if dimensions:
            metric_data["Dimensions"] = dimensions
        self._cw.put_metric_data(
            Namespace=self.settings.cloudwatch_namespace,
            MetricData=[metric_data],
        )

    def create_cpu_alarm(
        self,
        alarm_name: str,
        threshold: float,
        comparison: str,
        alarm_actions: List[str],
        evaluation_periods: Optional[int] = None,
        period: Optional[int] = None,
    ) -> Dict:
        self._cw.put_metric_alarm(
            AlarmName=alarm_name,
            AlarmDescription=f"AutoScaleAI managed alarm: {alarm_name}",
            MetricName="CPUUtilization",
            Namespace="AWS/EC2",
            Statistic="Average",
            Dimensions=[{"Name": "AutoScalingGroupName", "Value": self._asg_name}],
            Period=period or self.settings.alarm_period_seconds,
            EvaluationPeriods=evaluation_periods or self.settings.alarm_evaluation_periods,
            Threshold=threshold,
            ComparisonOperator=comparison,
            AlarmActions=alarm_actions,
            TreatMissingData="notBreaching",
        )
        logger.info("cloudwatch_alarm_created", name=alarm_name, threshold=threshold)
        return {"alarm_name": alarm_name, "threshold": threshold}

    def describe_alarms(self, alarm_names: Optional[List[str]] = None) -> List[Dict]:
        params: Dict[str, Any] = {"AlarmTypes": ["MetricAlarm"]}
        if alarm_names:
            params["AlarmNames"] = alarm_names
        response = self._cw.describe_alarms(**params)
        alarms = []
        for alarm in response.get("MetricAlarms", []):
            alarms.append({
                "name": alarm["AlarmName"],
                "state": alarm["StateValue"],
                "metric": alarm["MetricName"],
                "threshold": alarm["Threshold"],
                "description": alarm.get("AlarmDescription", ""),
            })
        return alarms

    def get_alarm_history(self, alarm_name: str, hours: int = 24) -> List[Dict]:
        start = datetime.utcnow() - timedelta(hours=hours)
        response = self._cw.describe_alarm_history(
            AlarmName=alarm_name,
            StartDate=start,
            EndDate=datetime.utcnow(),
        )
        return response.get("AlarmHistoryItems", [])

    def setup_standard_alarms(self, scale_out_arn: str, scale_in_arn: str):
        """Create the standard CPU scale-out and scale-in alarms."""
        self.create_cpu_alarm(
            alarm_name=f"{self._asg_name}-cpu-scale-out",
            threshold=self.settings.cpu_scale_out_threshold,
            comparison="GreaterThanThreshold",
            alarm_actions=[scale_out_arn],
        )
        self.create_cpu_alarm(
            alarm_name=f"{self._asg_name}-cpu-scale-in",
            threshold=self.settings.cpu_scale_in_threshold,
            comparison="LessThanThreshold",
            alarm_actions=[scale_in_arn],
        )
        logger.info("standard_alarms_configured", asg=self._asg_name)
