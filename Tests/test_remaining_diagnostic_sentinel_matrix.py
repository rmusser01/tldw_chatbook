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
    ("notes_sync", "tldw_chatbook.Notes.sync_engine"),
    ("subscriptions", "tldw_chatbook.Subscriptions.content_processor"),
    ("web", "tldw_chatbook.Web_Scraping.Article_Extractor_Lib"),
    ("ui", "tldw_chatbook.UI.Chat_Window_Enhanced"),
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
