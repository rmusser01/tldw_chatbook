# test_worker_events_contract.py
# Description: Regression coverage for worker_events.chat_wrapper_function's
# error-handling contract (task-634, updated for the task-650 contract).
#
# A failing LLM call previously fell through to an unconditional
# `return "STREAMING_HANDLED_BY_EVENTS"` regardless of whether the caller
# requested streaming. Non-streaming callers (e.g. media analysis) consume
# the return value directly, so the sentinel string rendered as if it were a
# valid (and successful) LLM response. Non-streaming failures must propagate.
#
# TASK-650 then removed the legacy streaming branch entirely (StreamDone and
# the sentinel are gone): native Console's provider gateway owns streaming,
# and this adapter now rejects streaming requests outright. task-1456 updated
# this file accordingly — the old streaming-sentinel test asserted removed
# behavior and its StreamDone import made the whole suite uncollectable.
#
# Imports
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Event_Handlers.worker_events import chat_wrapper_function


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


def test_chat_wrapper_function_nonstreaming_success_returns_core_result():
    """The adapter returns the core chat result verbatim and forwards its kwargs."""
    app = _mock_app()

    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
        return_value="the-llm-answer",
    ) as core:
        result = chat_wrapper_function(
            app,
            strip_thinking_tags=False,
            streaming=False,
            api_endpoint="test-endpoint",
            model="test-model",
        )

    assert result == "the-llm-answer"
    core.assert_called_once_with(
        strip_thinking_tags=False,
        streaming=False,
        api_endpoint="test-endpoint",
        model="test-model",
    )
    app.post_message.assert_not_called()


def test_chat_wrapper_function_streaming_request_rejected():
    """Streaming is owned by native Console; the adapter must refuse, not degrade."""
    app = _mock_app()

    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
    ) as core:
        with pytest.raises(ValueError, match="provider gateway"):
            chat_wrapper_function(
                app,
                streaming=True,
                api_endpoint="test-endpoint",
                model="test-model",
            )

    core.assert_not_called()
    app.post_message.assert_not_called()
