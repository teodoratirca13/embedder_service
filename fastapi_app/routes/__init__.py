from .health import router as health_router
from .documents import router as documents_router
from .query import router as query_router

__all__ = ["health_router", "documents_router", "query_router"]