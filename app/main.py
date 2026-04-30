"""
AutoScaleAI — FastAPI Application Entry Point
Wires all services, registers routers, and manages the application lifecycle.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.prediction import LoadPredictionModel
from app.models.anomaly_detection import AnomalyDetector
from app.services.aws_service import AWSService
from app.services.cost_service import CostService
from app.services.monitoring_service import MonitoringService
from app.services.scaling_service import ScalingService
from app.core.feedback_loop import FeedbackLoop
from app.core.scheduler import BackgroundScheduler
from app.api.routes import health, scaling, monitoring, prediction

# ── Logging ──────────────────────────────────────────────────────────────────

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


# ── Application Lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("autoscale_ai_starting", version=settings.app_version)

    # Instantiate all singletons
    aws = AWSService(settings)
    prediction_model = LoadPredictionModel(min_samples=settings.min_training_samples)
    anomaly_detector = AnomalyDetector(contamination=settings.anomaly_contamination)
    cost_service = CostService(settings)
    monitoring = MonitoringService(settings, aws)
    scaling = ScalingService(
        settings, aws, prediction_model, anomaly_detector, cost_service, monitoring
    )
    feedback_loop = FeedbackLoop(settings, prediction_model, anomaly_detector, monitoring)
    scheduler = BackgroundScheduler(settings, monitoring, scaling, anomaly_detector)

    # Attach to app state for dependency access in routes
    app.state.settings = settings
    app.state.aws = aws
    app.state.prediction_model = prediction_model
    app.state.anomaly_detector = anomaly_detector
    app.state.cost_service = cost_service
    app.state.monitoring = monitoring
    app.state.scaling = scaling
    app.state.feedback_loop = feedback_loop
    app.state.scheduler = scheduler

    # Start background workers
    await feedback_loop.start()
    await scheduler.start()

    logger.info("autoscale_ai_ready")
    yield  # ← application is running

    # Graceful shutdown
    logger.info("autoscale_ai_shutting_down")
    await scheduler.stop()
    await feedback_loop.stop()
    logger.info("autoscale_ai_stopped")


# ── FastAPI App ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AutoScaleAI",
        description=(
            "AI-powered auto-scaling system for AWS cloud infrastructure. "
            "Uses ML predictions and anomaly detection to scale EC2 instances "
            "efficiently and cost-effectively."
        ),
        version=settings.app_version,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global Exception Handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(scaling.router, prefix=prefix)
    app.include_router(monitoring.router, prefix=prefix)
    app.include_router(prediction.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "AutoScaleAI",
            "version": settings.app_version,
            "docs": f"{prefix}/docs",
        }

    return app


app = create_app()
