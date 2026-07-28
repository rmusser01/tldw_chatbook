# test_worker_events_contract.py
# Description: Regression coverage for worker_events.chat_wrapper_function's
# surviving non-streaming and cancellation-boundary contracts.
#
# Native Console now owns streaming. This retained adapter serves only
# non-streaming callers such as media analysis: failures propagate, while a
# streaming request is rejected before the core call.
#
# Imports
from unittest.mock import patch

import pytest

from tldw_chatbook.Event_Handlers.worker_events import chat_wrapper_function


def test_chat_wrapper_function_nonstreaming_failure_raises():
    """A non-streaming caller must see the real exception, not a sentinel string."""
    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            chat_wrapper_function(
                None,
                streaming=False,
                api_endpoint="test-endpoint",
                model="test-model",
            )


def test_chat_wrapper_function_nonstreaming_success_returns_core_result():
    """The adapter returns the core chat result verbatim and forwards its kwargs."""
    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
        return_value="the-llm-answer",
    ) as core:
        result = chat_wrapper_function(
            None,
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


def test_chat_wrapper_function_rejects_streaming_before_core_call():
    """Streaming belongs to native Console and never reaches this adapter."""
    with patch(
        "tldw_chatbook.Event_Handlers.worker_events.core_chat_function",
    ) as core_chat:
        with pytest.raises(ValueError, match="no longer owns streaming calls"):
            chat_wrapper_function(
                None,
                streaming=True,
                api_endpoint="test-endpoint",
                model="test-model",
            )

    core_chat.assert_not_called()
