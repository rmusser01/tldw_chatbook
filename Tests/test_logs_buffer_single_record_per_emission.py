"""TASK-15422: one log emission must appear once in the Logs screen buffer.

Every Console speak failure showed TWO identical
``TTS generation failed (outcome_code=...)`` ERROR rows in the Logs screen
per single click, which read as two generation attempts. The generation ran
once (the mock server received exactly one request on the success path); the
duplication is in the logging pipeline: ``Logging_Config._setup_logging``
installs the canonical loguru->stdlib forward
(``_forward_loguru_to_standard``, TRACE, diagnose=False per task-2119), and
``TldwCli._setup_buffered_logging`` then installed a SECOND loguru sink
bridging to stdlib — so every loguru record reached the root logger's
``PersistentLogHandler`` twice, doubling both the Logs screen rows and its
Errors count.

This test replicates the production sink layout (config forward active, then
the app's buffered handler setup) and pins that one loguru emission produces
exactly one buffered record.
"""

from __future__ import annotations

import logging

import pytest
from loguru import logger as loguru_logger

pytestmark = pytest.mark.unit


class _AppStub:
    """Bare object for binding ``_setup_buffered_logging``."""


def test_one_loguru_error_lands_once_in_the_logs_buffer():
    """One ``logger.error`` -> one row in ``_log_records``, not two.

    Returns:
        None.
    """
    from tldw_chatbook.Logging_Config import _forward_loguru_to_standard
    from tldw_chatbook.app import TldwCli

    sinks_before = set(loguru_logger._core.handlers)
    config_sink_id = loguru_logger.add(
        _forward_loguru_to_standard,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        level="TRACE",
        diagnose=False,
        backtrace=True,
    )
    stub = _AppStub()
    try:
        TldwCli._setup_buffered_logging(stub)
        marker = "single-record-marker-15422"
        loguru_logger.error("TTS generation failed (outcome_code={})", marker)
        matching = [
            record for record in stub._log_records if marker in record[2]
        ]
        assert len(matching) == 1, (
            f"one emission buffered {len(matching)} times: {matching}"
        )
    finally:
        for sink_id in set(loguru_logger._core.handlers) - sinks_before:
            try:
                loguru_logger.remove(sink_id)
            except ValueError:
                pass
        handler = getattr(stub, "_persistent_log_handler", None)
        if handler is not None:
            logging.getLogger().removeHandler(handler)
