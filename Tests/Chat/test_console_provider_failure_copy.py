"""TASK-335: provider failures must surface the server's message, never MDN links.

The agent-runtime send path rendered raw httpx text — `Server error '500
...' For more information check: https://developer.mozilla.org/...` — while
the response body's actionable hint ("you may need to provide the mmproj")
was discarded, and the re-sent image poisoned every later send with the
same undiagnosable 500 (UX review finding
j3-provider-error-discards-detail-poisons-conversation, REGRESSION).
"""

from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    describe_stream_failure,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

MMPROJ_BODY = (
    '{"error": {"message": "image input is not supported - '
    'hint: you may need to provide the mmproj", "code": 500}}'
)


def _http_500(body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:9099/v1/chat/completions")
    response = httpx.Response(500, request=request, text=body)
    return httpx.HTTPStatusError(
        "Server error '500 Internal Server Error' for url "
        "'http://127.0.0.1:9099/v1/chat/completions'\n"
        "For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
        request=request,
        response=response,
    )


def test_describe_stream_failure_surfaces_json_error_body():
    copy = describe_stream_failure(_http_500(MMPROJ_BODY))
    assert "HTTP 500" in copy
    assert "provide the mmproj" in copy
    assert "developer.mozilla.org" not in copy


def test_describe_stream_failure_surfaces_plain_text_body_truncated():
    copy = describe_stream_failure(_http_500("plain provider explosion " * 40))
    assert "HTTP 500" in copy
    assert "plain provider explosion" in copy
    assert len(copy) < 400
    assert "developer.mozilla.org" not in copy


def test_describe_stream_failure_never_emits_mdn_link_without_body():
    copy = describe_stream_failure(_http_500(""))
    assert "HTTP 500" in copy
    assert "developer.mozilla.org" not in copy


class _ExplodingGateway:
    async def resolve_for_send(self, _selection):
        return SimpleNamespace(ready=True, provider="llama_cpp", visible_copy="")

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        raise _http_500(MMPROJ_BODY)
        yield  # pragma: no cover — makes this an async generator


@pytest.mark.asyncio
async def test_agent_failure_row_carries_body_and_image_recovery_hint(tmp_path):
    gateway = _ExplodingGateway()
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="what is in this image?",
        image_data=b"\x89PNG-fake",
        image_mime_type="image/png",
    )

    result = await controller.submit_draft("and now this text send fails too")
    assert result.accepted is True

    messages = store.messages_for_session(session.id)
    system_rows = [
        m.content for m in messages if m.role is ConsoleMessageRole.SYSTEM
    ]
    assert system_rows, "no failure system row appended"
    failure_row = system_rows[-1]
    assert "provide the mmproj" in failure_row
    assert "developer.mozilla.org" not in failure_row
    # Recovery hint: the conversation carries an image the provider may be
    # rejecting — point at it and at the existing remove/switch affordances.
    assert "image" in failure_row.lower()
    assert "remove" in failure_row.lower() or "vision" in failure_row.lower()
