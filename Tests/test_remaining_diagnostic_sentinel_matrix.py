from __future__ import annotations

import logging
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    _forward_loguru_to_standard,
)
from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    log_persistent_metadata,
)


PRIVATE_SENTINEL = "REMAINING-PRIVATE-SENTINEL-sk-not-a-real-key"
DOMAIN_OWNERS = [
    ("rag_search", "tldw_chatbook.RAG_Search.simplified.search_service"),
    ("ingestion", "tldw_chatbook.Local_Ingestion.Document_Processing_Lib"),
    ("media_database", "tldw_chatbook.DB.Client_Media_DB_v2"),
    ("notes_sync", "tldw_chatbook.Notes.notes_sync_runtime"),
    # content_processor was retired in TASK-1220; monitoring_engine is the live
    # subscriptions module that WatchlistCheckHandler actually calls.
    ("subscriptions", "tldw_chatbook.Subscriptions.monitoring_engine"),
    ("web", "tldw_chatbook.Web_Scraping.Article_Extractor_Lib"),
    ("application", "tldw_chatbook.app"),
]


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _private_handler(path: Path) -> PrivateRotatingFileHandler:
    handler = PrivateRotatingFileHandler(
        path,
        maxBytes=180,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    handler.addFilter(PersistentDiagnosticFilter())
    return handler


def _all_generations(path: Path) -> str:
    return "\n".join(
        candidate.read_text(encoding="utf-8")
        for candidate in sorted(path.parent.glob(path.name + "*"))
        if candidate.is_file()
    )


def _emit_owned_loguru_payload(module_name: str, message: str) -> None:
    source = (
        Path(__file__).resolve().parents[1] / Path(*module_name.split("."))
    ).with_suffix(".py")
    code = compile("logger.error(message)", str(source), "exec")
    exec(
        code,
        {"__name__": module_name, "logger": loguru_logger, "message": message},
    )


@pytest.mark.parametrize(("operation", "module_name"), DOMAIN_OWNERS)
def test_remaining_domain_records_are_metadata_only(
    operation: str,
    module_name: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    private_handler = _private_handler(path)
    collecting = _CollectingHandler()
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(private_handler)
    root.addHandler(collecting)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    domain_logger = logging.getLogger(module_name)
    try:
        domain_logger.debug("query=%s", PRIVATE_SENTINEL)
        domain_logger.info("content=%s", PRIVATE_SENTINEL)
        try:
            raise RuntimeError(PRIVATE_SENTINEL)
        except RuntimeError:
            domain_logger.exception("operation failed: %s", PRIVATE_SENTINEL)
        _emit_owned_loguru_payload(
            module_name,
            f"response body and config value: {PRIVATE_SENTINEL}",
        )
        for index in range(2):
            log_persistent_metadata(
                domain_logger,
                logging.INFO,
                "operation_complete",
                operation=operation,
                status="success",
                item_count=index + 1,
                duration_ms=12,
            )
        log_persistent_metadata(
            domain_logger,
            logging.WARNING,
            "operation_failed",
            operation=operation,
            status="error",
            error_category="processing_failed",
            exception_type="RuntimeError",
            duration_ms=14,
        )
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(collecting)
        root.removeHandler(private_handler)
        root.setLevel(old_level)
        private_handler.close()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "sk-not-a-real-key" not in persisted
    assert "event=operation_complete" in persisted
    assert f"operation={operation}" in persisted
    assert "status=success" in persisted
    assert "event=operation_failed" in persisted
    assert "error_category=processing_failed" in persisted
    assert "exception_type=RuntimeError" in persisted
    assert any(PRIVATE_SENTINEL in message for message in collecting.messages)
    assert (tmp_path / "application.log.1").exists()


@pytest.mark.parametrize(
    "logger_name",
    ["transformers", "feedparser", "bs4", "third_party.custom"],
)
def test_third_party_payload_is_file_filtered_but_remains_available_to_ui(
    logger_name: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    private_handler = _private_handler(path)
    collecting = _CollectingHandler()
    third_party_logger = logging.Logger(logger_name, level=logging.DEBUG)
    third_party_logger.addHandler(private_handler)
    third_party_logger.addHandler(collecting)
    try:
        third_party_logger.error("response=%s", PRIVATE_SENTINEL)
    finally:
        private_handler.close()

    assert any(PRIVATE_SENTINEL in message for message in collecting.messages)
    assert PRIVATE_SENTINEL not in _all_generations(path)


# ADR-029 applies credential and recognized PII redaction to the real
# in-app collector and its Copy all buffer.


@pytest.mark.parametrize(("operation", "module_name"), DOMAIN_OWNERS)
def test_remaining_domain_records_stay_out_of_the_clipboard_artifact(
    operation: str,
    module_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real collector retains diagnostics with credentials and PII masked."""
    from tldw_chatbook.app import TldwCli

    class _AppStub:
        pass

    stub = _AppStub()
    root = logging.getLogger()
    caplog.set_level(logging.INFO, logger="tldw_chatbook")
    caplog.set_level(logging.DEBUG, logger=module_name)
    TldwCli._setup_buffered_logging(stub)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    domain_logger = logging.getLogger(module_name)
    email = "fixture-reader@example.test"
    try:
        domain_logger.debug("query failed api_key=%s", PRIVATE_SENTINEL)
        domain_logger.info("content check email=%s", email)
        try:
            raise RuntimeError(f"operation failed token={PRIVATE_SENTINEL}")
        except RuntimeError:
            domain_logger.exception("operation interrupted")
        _emit_owned_loguru_payload(
            module_name,
            f"response rejected Authorization: Bearer {PRIVATE_SENTINEL}",
        )
        log_persistent_metadata(
            domain_logger,
            logging.INFO,
            "operation_complete",
            operation=operation,
            status="success",
            duration_ms=12,
        )
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(stub._persistent_log_handler)

    shareable = "\n".join(stub._log_buffer)
    rendered = "\n".join(message for _l, _n, message in stub._log_records)
    for artifact in (shareable, rendered):
        assert PRIVATE_SENTINEL not in artifact
        assert "sk-not-a-real-key" not in artifact
        assert email not in artifact
        assert "query failed api_key=***REDACTED***" in artifact
        assert "content check" in artifact
        assert "response rejected" in artifact
        assert "operation interrupted" in artifact
        assert "Traceback (most recent call last)" in artifact
        assert "RuntimeError: operation failed" in artifact
        assert "event=operation_complete" in artifact
        assert f"operation={operation}" in artifact
