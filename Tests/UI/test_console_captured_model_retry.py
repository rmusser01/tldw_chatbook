"""Captured model retries complete through the mounted Console send action."""

import pytest

from Tests.UI.test_console_blocked_send_recovery import (
    _assistant_texts,
    _frame_rows,
    _mounted_console,
    _send,
    _until_completed,
)
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError, ChatRateLimitError

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("status", [429, 500])
async def test_captured_provider_retry_completes_without_unknown_delivery(
    tmp_path, monkeypatch, status
):
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        attempts = []

        def provider(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                if status == 429:
                    raise ChatRateLimitError("fixture rate limit", provider="openai")
                raise ChatProviderError(
                    "fixture server failure", provider="openai", status_code=500
                )
            return {"choices": [{"message": {"content": "Recovered captured reply"}}]}

        monkeypatch.setattr(
            h.controller.provider_gateway, "_chat_api_call_fn", provider
        )
        await _send(h, "Try the captured model retry")
        await _until_completed(h)
        assert len(attempts) == 2
        assert _assistant_texts(h) == ["Recovered captured reply"]
        assert h.store.dispatch_recovery_for_session(h.session.id) is None
        frame = "\n".join(_frame_rows(h.host)).lower()
        assert "trace capture blocked" not in frame
        assert "delivery unknown" not in frame
        with h.database.transaction() as cursor:
            calls = cursor.execute(
                "SELECT state, call_sequence FROM console_trace_calls ORDER BY call_sequence"
            ).fetchall()
        assert [tuple(row) for row in calls] == [("error", 0), ("complete", 1)]
