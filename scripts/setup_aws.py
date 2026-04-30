#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script
Run this once to provision:
  - A Launch Template for EC2 instances
  - An Auto Scaling Group
  - CloudWatch scale-out / scale-in alarms
  - Scaling policies wired to those alarms

Usage:
    python scripts/setup_aws.py \
        --ami-id ami-0c55b159cbfafe1f0 \
        --instance-type t3.medium \
        --key-name my-keypair \
        --security-groups sg-xxxxxxxx \
        --vpc-zone-ids subnet-aaa,subnet-bbb \
        --target-group-arns arn:aws:elasticloadbalancing:...
"""

import argparse
import sys
import os

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.services.aws_service import AWSService


def create_launch_template(
    ec2_client,
    template_name: str,
    ami_id: str,
    instance_type: str,
    key_name: str,
    security_group_ids: list,
) -> str:
    try:
        response = ec2_client.create_launch_template(
            LaunchTemplateName=template_name,
            LaunchTemplateData={
                "ImageId": ami_id,
                "InstanceType": instance_type,
                "KeyName": key_name,
                "SecurityGroupIds": security_group_ids,
                "Monitoring": {"Enabled": True},
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "ManagedBy", "Value": "AutoScaleAI"},
                            {"Key": "Name", "Value": f"{template_name}-instance"},
                        ],
                    }
                ],
                "UserData": _get_user_data(),
            },
        )
        lt_id = response["LaunchTemplate"]["LaunchTemplateId"]
        print(f"✓ Launch Template created: {lt_id}")
        return lt_id
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidLaunchTemplateName.AlreadyExistsException":
            response = ec2_client.describe_launch_templates(
                LaunchTemplateNames=[template_name]
            )
            lt_id = response["LaunchTemplates"][0]["LaunchTemplateId"]
            print(f"  Launch Template already exists: {lt_id}")
            return lt_id
        raise


def _get_user_data() -> str:
    import base64
    script = """#!/bin/bash
yum update -y
yum install -y amazon-cloudwatch-agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \\
    -a fetch-config -m ec2 -c default -s
"""
    return base64.b64encode(script.encode()).decode()


def setup_infrastructure(args):
    settings = get_settings()
    aws = AWSService(settings)

    session = boto3.Session(
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
        region_name=settings.aws_region,
    )
    ec2 = session.client("ec2")

    print(f"\n{'='*60}")
    print(f"  AutoScaleAI — AWS Infrastructure Setup")
    print(f"  Region : {settings.aws_region}")
    print(f"  ASG    : {settings.auto_scaling_group_name}")
    print(f"{'='*60}\n")

    # 1. Launch Template
    template_name = f"{settings.auto_scaling_group_name}-lt"
    lt_id = create_launch_template(
        ec2,
        template_name=template_name,
        ami_id=args.ami_id,
        instance_type=args.instance_type,
        key_name=args.key_name,
        security_group_ids=args.security_groups.split(","),
    )

    # 2. Auto Scaling Group
    vpc_zones = args.vpc_zone_ids.split(",")
    target_arns = args.target_group_arns.split(",") if args.target_group_arns else None
    try:
        result = aws.create_auto_scaling_group(lt_id, vpc_zones, target_arns)
        print(f"✓ Auto Scaling Group: {result['asg_name']} ({result['status']})")
    except ClientError as e:
        if "AlreadyExists" in str(e):
            print(f"  ASG already exists: {settings.auto_scaling_group_name}")
        else:
            raise

    # 3. Scaling Policies + CloudWatch Alarms
    print("  Creating scaling policies and CloudWatch alarms...")
    scale_out_arn = aws.create_scaling_policy(
        "autoscale-ai-scale-out",
        adjustment=2,
        cooldown=settings.scale_out_cooldown_seconds,
    )
    scale_in_arn = aws.create_scaling_policy(
        "autoscale-ai-scale-in",
        adjustment=-1,
        cooldown=settings.scale_in_cooldown_seconds,
    )
    aws.setup_standard_alarms(scale_out_arn, scale_in_arn)
    print(f"✓ CloudWatch alarms configured")
    print(f"    Scale-out at CPU > {settings.cpu_scale_out_threshold}%")
    print(f"    Scale-in  at CPU < {settings.cpu_scale_in_threshold}%")

    print(f"\n{'='*60}")
    print("  Setup complete. Start the API server:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Provision AWS infrastructure for AutoScaleAI"
    )
    parser.add_argument("--ami-id", required=True, help="AMI ID for EC2 instances")
    parser.add_argument("--instance-type", default="t3.medium", help="EC2 instance type")
    parser.add_argument("--key-name", required=True, help="EC2 key pair name")
    parser.add_argument("--security-groups", required=True, help="Comma-separated security group IDs")
    parser.add_argument("--vpc-zone-ids", required=True, help="Comma-separated subnet IDs")
    parser.add_argument("--target-group-arns", default="", help="Comma-separated ALB target group ARNs")
    args = parser.parse_args()
    setup_infrastructure(args)


if __name__ == "__main__":
    main()
