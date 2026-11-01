"""TASK-33621.5: provider 429/500/504 errors with capture on, at the HTTP edge.

Each test mounts the real ``ChatScreen`` with its real controller, store,
agent loop, gateway, provider adapter (``chat_with_openai``) and durable trace
runtime on a file-backed ChaChaNotes database. Only the HTTP response is
fake: ``requests``' transport answers every POST with the status the test
names, so the provider adapter, the agent's in-place transient retry and the
trace reservations all run as they do in the app.
"""

from __future__ import annotations

import json

import pytest
import requests

from Tests.UI.test_console_blocked_send_recovery import (
    _frame_rows,
    _mounted_console,
    _send,
    _until,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    project_console_send_authority,
)

pytestmark = pytest.mark.bootstrap_profile

HTTP_REPLY = "Reply over HTTP 33621"
#: What each status means, as the visible copy must say it (AC#1).
MEANINGS = {
    429: "rate limit exceeded",
    500: "internal server error",
    504: "gateway timed out",
}


def answer_http(monkeypatch, statuses: list[int], posts: list[str]) -> None:
    """Answer the n-th provider POST with ``statuses[n]`` (the last one repeats).

    Patching ``HTTPAdapter.send`` replaces only the socket exchange: the
    provider adapter still builds, sends and classifies the request.
    """

    from urllib3.response import HTTPResponse
    from urllib3.util.retry import Retry

    def send(self, request, **_kwargs):
        posts.append(request.url)
        status = statuses[min(len(posts), len(statuses)) - 1]
        response = requests.Response()
        response.status_code = status
        # This socket-edge replacement makes exactly one physical request.
        # Preserve the real response's empty transport-retry evidence.
        response.raw = HTTPResponse(status=status, retries=Retry(total=0))
        response.reason = "fixture"
        response.headers["Content-Type"] = "application/json"
        body = (
            {"choices": [{"message": {"role": "assistant", "content": HTTP_REPLY}}]}
            if status == 200
            else {"error": {"message": f"fixture status {status}", "type": "x"}}
        )
        response._content = json.dumps(body).encode()
        response.request = request
        response.url = request.url
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)


def real_provider_adapter(h, monkeypatch) -> None:
    """Undo the harness recorder so the real ``chat_api_call`` runs."""
    monkeypatch.setattr(h.controller.provider_gateway, "_chat_api_call_fn", None)


def visible_text(h) -> str:
    return " ".join(" ".join(_frame_rows(h.host)).split())


@pytest.mark.asyncio
@pytest.mark.parametrize("status", sorted(MEANINGS))
async def test_exhausted_provider_error_names_status_and_meaning(
    tmp_path, monkeypatch, status
):
    """AC#1/#4/#5: a failed row and copy naming provider, status and meaning."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        real_provider_adapter(h, monkeypatch)
        answer_http(monkeypatch, [status], posts)

        await _send(h, "Say hi")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.FAILED,
            timeout=30.0,
            what=f"the run did not fail: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        # The failed run state is published before the queued transcript
        # projection replaces its waiting row and paints the System error.
        await _until(
            h.pilot,
            lambda: f"Status: {status}." in visible_text(h),
            timeout=30.0,
            what=f"HTTP {status} was not painted in the exhausted error row",
        )

        # AC#3: the agent's in-place retries reached the provider (1 + 2).
        assert len(posts) == 3, posts
        assistant = [
            row
            for row in h.store.messages_for_session(h.store.active_session_id)
            if row.role is ConsoleMessageRole.ASSISTANT
        ]
        assert [row.status for row in assistant] == ["failed"]
        text = visible_text(h)
        assert "OpenAI" in text
        assert f"Status: {status}." in text, text
        assert MEANINGS[status] in text
        assert "Trace capture blocked" not in text
        assert "provider was not contacted" not in text
        run = project_console_send_authority(
            h.console._build_console_inspector_state(None)
        ).run
        assert run.startswith("Failed"), run
