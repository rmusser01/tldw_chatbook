# test_worker_events_contract.py
# Description: Regression coverage for worker_events.chat_wrapper_function's
# error-handling contract (task-634).
#
# A failing LLM call previously fell through to an unconditional
# `return "STREAMING_HANDLED_BY_EVENTS"` regardless of whether the caller
# requested streaming. Non-streaming callers (e.g. media analysis) consume
# the return value directly, so the sentinel string rendered as if it were a
# valid (and successful) LLM response. Non-streaming failures must now
# propagate instead. The streaming branch's existing contract (StreamDone
# posted + sentinel returned) must remain byte-identical.
#
# Imports
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Event_Handlers.worker_events import (
    StreamDone,
    chat_wrapper_function,
)


def _mock_app() -> Mock:
    app = Mock()
    app.loguru_logger = Mock()
    app.post_message = Mock()
    app.current_chat_worker = None
    return app


def test_chat_wrapper_function_nonstreaming_failure_raises():
    """A non-streaming caller must see the real exception, not a sentinel string."""
    app = _mock_app()

    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            chat_wrapper_function(
                app,
                streaming=False,
                api_endpoint="test-endpoint",
                model="test-model",
            )

    app.post_message.assert_not_called()


def test_chat_wrapper_function_streaming_failure_keeps_sentinel_contract():
    """The legacy streaming contract (StreamDone posted + sentinel returned) is untouched."""
    app = _mock_app()

    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
        side_effect=RuntimeError("boom"),
    ):
        result = chat_wrapper_function(
            app,
            streaming=True,
            api_endpoint="test-endpoint",
            model="test-model",
        )

    assert result == "STREAMING_HANDLED_BY_EVENTS"
    app.post_message.assert_called_once()
    posted = app.post_message.call_args[0][0]
    assert isinstance(posted, StreamDone)
    assert posted.full_text == ""
    assert posted.error is not None
    assert "boom" in posted.error
