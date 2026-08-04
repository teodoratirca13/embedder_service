import logging
import time
import uuid

from fastapi import Request

from fastapi_app.utils.logging_ctx import request_id_var
from fastapi_app.utils.logger import setup_logger

log = setup_logger("access")

HEADER = "X-Request-ID"


async def request_context(request: Request, call_next):
    rid = request.headers.get(HEADER) or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers[HEADER] = rid
        log.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response
    except Exception:
        log.exception(
            "http_error",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        raise
    finally:
        request_id_var.reset(token)