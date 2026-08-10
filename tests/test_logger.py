"""Unit tests for fastapi_app.utils.logger.JsonFormatter and setup_logger."""
import json
import logging

from fastapi_app.utils.logger import JsonFormatter, setup_logger
from fastapi_app.utils.logging_ctx import request_id_var


def _make_record(msg="hello", level=logging.INFO, extra_data=None, exc_info=None):
    record = logging.LogRecord(
        name="test.logger", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )
    if extra_data is not None:
        record.extra_data = extra_data
    return record


class TestJsonFormatter:

    def test_produces_valid_json(self):
        formatted = JsonFormatter().format(_make_record())
        data = json.loads(formatted)  # should not raise
        assert data["msg"] == "hello"

    def test_includes_standard_fields(self):
        data = json.loads(JsonFormatter().format(_make_record(level=logging.WARNING)))
        assert data["service"] == "embedder"
        assert data["level"] == "WARNING"
        assert data["logger"] == "test.logger"
        assert "ts" in data

    def test_merges_extra_data_into_top_level(self):
        data = json.loads(JsonFormatter().format(_make_record(extra_data={"document_id": 42})))
        assert data["document_id"] == 42

    def test_includes_request_id_from_context(self):
        token = request_id_var.set("req-abc-123")
        try:
            data = json.loads(JsonFormatter().format(_make_record()))
            assert data["request_id"] == "req-abc-123"
        finally:
            request_id_var.reset(token)

    def test_defaults_request_id_when_unset(self):
        data = json.loads(JsonFormatter().format(_make_record()))
        assert data["request_id"] == "-"

    def test_includes_exception_info(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = _make_record(exc_info=sys.exc_info())
        data = json.loads(JsonFormatter().format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestSetupLogger:

    def test_returns_logger_with_json_formatter(self):
        logger = setup_logger("test.some.module")
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)

    def test_does_not_propagate_to_root(self):
        logger = setup_logger("test.some.other.module")
        assert logger.propagate is False

    def test_calling_twice_does_not_duplicate_handlers(self):
        setup_logger("test.dup.module")
        logger = setup_logger("test.dup.module")
        assert len(logger.handlers) == 1
