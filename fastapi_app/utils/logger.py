import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict

from opentelemetry import trace                      # ← nou

from fastapi_app.utils.logging_ctx import request_id_var

_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName", "extra_data",
}

class JsonFormatter(logging.Formatter):
    """Format logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "service": "embedder",
            "request_id": request_id_var.get(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }

        # Trace context OTel, format W3C — identic cu ce scoate agentul Java.  ← nou
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            log_data["trace_id"] = format(ctx.trace_id, "032x")
            log_data["span_id"] = format(ctx.span_id, "016x")

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        for key, value in record.__dict__.items():
            if key not in _STD and key not in log_data:
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False, default=str)

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a logger with JSON formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    logger.propagate = False

    return logger