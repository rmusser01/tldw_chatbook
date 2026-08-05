from __future__ import annotations

import logging
from pathlib import Path

from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    _forward_loguru_to_standard,
)
from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    log_persistent_metadata,
)


PRIVATE_SENTINEL = "PRIVATE-PROMPT-SENTINEL-sk-not-a-real-key"


def _real_private_sink(path: Path) -> PrivateRotatingFileHandler:
    handler = PrivateRotatingFileHandler(
        path,
        maxBytes=256,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    handler.addFilter(PersistentDiagnosticFilter())
    return handler


def _all_generations(path: Path) -> str:
    return "\n".join(
        candidate.read_text(encoding="utf-8")
        for candidate in sorted(path.parent.glob(f"{path.name}*"))
        if candidate.is_file()
    )


def _emit_owned_loguru_payload(module_name: str, message: str) -> None:
    source = (
        Path(__file__).resolve().parents[1] / Path(*module_name.split("."))
    ).with_suffix(".py")
    code = compile("loguru_logger.debug(message)", str(source), "exec")
    exec(
        code,
        {
            "__name__": module_name,
            "loguru_logger": loguru_logger,
            "message": message,
        },
    )


def test_real_rotating_sink_rejects_owned_standard_payloads_but_keeps_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    handler = _real_private_sink(path)
    owned_logger = logging.getLogger("tldw_chatbook.Chat.privacy_test")
    owned_logger.handlers = [handler]
    owned_logger.propagate = False
    owned_logger.setLevel(logging.DEBUG)
    try:
        owned_logger.debug("prompt=%s", PRIVATE_SENTINEL)
        owned_logger.error("response=%s", PRIVATE_SENTINEL)
        log_persistent_metadata(
            owned_logger,
            logging.INFO,
            "provider_request",
            provider="openai",
            status="success",
            payload_length=len(PRIVATE_SENTINEL),
        )
    finally:
        handler.close()
        owned_logger.handlers.clear()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "event=provider_request" in persisted
    assert "provider=openai" in persisted
    assert "payload_length=" in persisted


def test_real_rotating_sink_rejects_loguru_payload_from_owned_module(
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    handler = _real_private_sink(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    try:
        _emit_owned_loguru_payload(
            "tldw_chatbook.Agents.builtin_tool_gate",
            f"tool={PRIVATE_SENTINEL}",
        )
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted


def test_metadata_helper_rejects_unregistered_fields_and_private_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "application.log"
    handler = _real_private_sink(path)
    owned_logger = logging.getLogger("tldw_chatbook.Tools.privacy_test")
    owned_logger.handlers = [handler]
    owned_logger.propagate = False
    owned_logger.setLevel(logging.DEBUG)
    try:
        try:
            log_persistent_metadata(
                owned_logger,
                logging.INFO,
                "tool_execution",
                raw_argument=PRIVATE_SENTINEL,
            )
        except ValueError as exc:
            assert "raw_argument" in str(exc)
        else:
            raise AssertionError("unregistered persistent metadata was accepted")

        log_persistent_metadata(
            owned_logger,
            logging.INFO,
            "tool_execution",
            tool_name=PRIVATE_SENTINEL,
            status="success",
        )
    finally:
        handler.close()
        owned_logger.handlers.clear()

    persisted = _all_generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "tool_name=invalid" in persisted
