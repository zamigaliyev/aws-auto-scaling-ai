"""
Cost Analysis Service
Evaluates the financial impact of scaling decisions and surfaces optimization
opportunities to prevent wasteful over-provisioning.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import structlog

from app.config import Settings

logger = structlog.get_logger(__name__)


class CostService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cost_per_hour = settings.cost_per_instance_hour
        self._scaling_history: List[Dict] = []

    def analyze(
        self,
        current_instances: int,
        proposed_instances: int,
        predicted_cpu: List[float],
        current_request_rate: float = 0.0,
    ) -> Dict:
        current_hourly = current_instances * self.cost_per_hour
        proposed_hourly = proposed_instances * self.cost_per_hour
        delta_hourly = proposed_hourly - current_hourly

        projected_daily = proposed_hourly * 24
        projected_monthly = projected_daily * 30

        potential_savings = self._compute_potential_savings(
            current_instances, predicted_cpu
        )
        suggestions = self._generate_suggestions(
            current_instances, proposed_instances, predicted_cpu, current_request_rate
        )
        cost_per_request = (
            (proposed_hourly / (current_request_rate * 3600))
            if current_request_rate > 0
            else None
        )

        analysis = {
            "current_hourly_cost": round(current_hourly, 4),
            "proposed_hourly_cost": round(proposed_hourly, 4),
            "delta_hourly": round(delta_hourly, 4),
            "projected_daily_cost": round(projected_daily, 2),
            "projected_monthly_cost": round(projected_monthly, 2),
            "potential_savings": round(potential_savings, 2),
            "optimization_suggestions": suggestions,
            "cost_per_request": round(cost_per_request, 6) if cost_per_request else None,
            "is_cost_effective": self._is_cost_effective(
                current_instances, proposed_instances, predicted_cpu
            ),
        }
        logger.info(
            "cost_analysis_complete",
            current=current_instances,
            proposed=proposed_instances,
            delta_hourly=analysis["delta_hourly"],
        )
        return analysis

    def record_scaling_action(
        self,
        from_count: int,
        to_count: int,
        trigger: str,
        cpu_at_decision: float,
    ):
        self._scaling_history.append({
            "timestamp": datetime.utcnow(),
            "from_count": from_count,
            "to_count": to_count,
            "trigger": trigger,
            "cpu_at_decision": cpu_at_decision,
            "cost_delta_hourly": (to_count - from_count) * self.cost_per_hour,
        })
        # Retain only last 1000 records in memory
        if len(self._scaling_history) > 1000:
            self._scaling_history = self._scaling_history[-1000:]

    def get_scaling_cost_report(self, hours: int = 24) -> Dict:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [e for e in self._scaling_history if e["timestamp"] >= cutoff]

        total_scale_out = sum(
            1 for e in recent if e["to_count"] > e["from_count"]
        )
        total_scale_in = sum(
            1 for e in recent if e["to_count"] < e["from_count"]
        )
        unnecessary_actions = self._detect_thrashing(recent)

        return {
            "period_hours": hours,
            "total_scaling_actions": len(recent),
            "scale_out_actions": total_scale_out,
            "scale_in_actions": total_scale_in,
            "unnecessary_actions": unnecessary_actions,
            "estimated_wasted_cost": unnecessary_actions * self.cost_per_hour * 0.5,
        }

    def _compute_potential_savings(
        self, current_instances: int, predicted_cpu: List[float]
    ) -> float:
        if not predicted_cpu:
            return 0.0
        avg_cpu = sum(predicted_cpu) / len(predicted_cpu)
        optimal = max(1, int(avg_cpu / 60) + 1)
        wasted = max(0, current_instances - optimal)
        return wasted * self.cost_per_hour * 24

    def _generate_suggestions(
        self,
        current: int,
        proposed: int,
        predicted_cpu: List[float],
        request_rate: float,
    ) -> List[str]:
        suggestions = []
        avg_cpu = sum(predicted_cpu) / len(predicted_cpu) if predicted_cpu else 50.0

        if avg_cpu < 20 and current > self.settings.min_instances:
            suggestions.append(
                f"CPU predicted at {avg_cpu:.1f}% — consider scaling in to reduce cost."
            )
        if avg_cpu > 85:
            suggestions.append(
                "High CPU predicted — ensure scale-out happens before saturation."
            )
        if proposed > current * 2:
            suggestions.append(
                "Large scale-out requested — consider stepped scaling to avoid over-provisioning."
            )
        if current > self.settings.max_instances * 0.8:
            suggestions.append(
                "Approaching max instance limit — review capacity reservation strategy."
            )
        if not suggestions:
            suggestions.append("Current scaling configuration appears cost-efficient.")
        return suggestions

    def _is_cost_effective(
        self, current: int, proposed: int, predicted_cpu: List[float]
    ) -> bool:
        if not predicted_cpu:
            return True
        avg_cpu = sum(predicted_cpu) / len(predicted_cpu)
        # Scale-out is cost-effective if avg CPU > 65% on current fleet
        if proposed > current:
            return avg_cpu > 65
        # Scale-in is cost-effective if avg CPU < 30%
        if proposed < current:
            return avg_cpu < 30
        return True

    def _detect_thrashing(self, history: List[Dict]) -> int:
        """Count rapid alternating scale-out/in sequences (thrashing)."""
        if len(history) < 4:
            return 0
        thrash_count = 0
        for i in range(2, len(history)):
            a = history[i - 2]["to_count"] - history[i - 2]["from_count"]
            b = history[i - 1]["to_count"] - history[i - 1]["from_count"]
            c = history[i]["to_count"] - history[i]["from_count"]
            # Sign alternates: out, in, out OR in, out, in
            if (a > 0 and b < 0 and c > 0) or (a < 0 and b > 0 and c < 0):
                thrash_count += 1
        return thrash_count
