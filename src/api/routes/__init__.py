"""Routes package."""

from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router

__all__ = ["predict_router", "health_router"]
