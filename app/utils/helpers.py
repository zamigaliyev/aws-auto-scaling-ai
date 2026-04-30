"""General-purpose helpers used across the application."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)


def generate_synthetic_training_data(
    n_points: int = 500,
    start_time: datetime = None,
    interval_minutes: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate realistic synthetic metric data for bootstrapping / testing.
    Simulates daily seasonality, weekly patterns, and random spikes.
    """
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(minutes=n_points * interval_minutes)

    rng = np.random.default_rng(42)
    data = []

    for i in range(n_points):
        ts = start_time + timedelta(minutes=i * interval_minutes)
        hour = ts.hour
        dow = ts.weekday()

        # Diurnal pattern: peaks at 10:00 and 15:00 UTC
        base_cpu = (
            15
            + 35 * np.sin(np.pi * max(0, hour - 8) / 10) ** 2
            + 20 * (1 if dow < 5 else 0)
        )
        cpu = float(np.clip(base_cpu + rng.normal(0, 8), 5, 98))

        base_mem = 40 + 0.3 * cpu + rng.normal(0, 5)
        mem = float(np.clip(base_mem, 10, 95))

        # Occasional traffic spike
        spike = float(rng.uniform(0, 1) > 0.97) * rng.uniform(20, 40)
        cpu = float(np.clip(cpu + spike, 5, 98))

        instance_count = max(1, int(np.ceil(cpu / 60)))
        data.append({
            "timestamp": ts,
            "cpu_utilization": round(cpu, 2),
            "memory_utilization": round(mem, 2),
            "network_in": float(rng.uniform(1e6, 5e7)),
            "network_out": float(rng.uniform(5e5, 2e7)),
            "request_count": int(rng.integers(50, 2000)),
            "instance_count": instance_count,
        })
    return data


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else default


def format_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"
