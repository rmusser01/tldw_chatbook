"""TASK-335: provider failures must surface the server's message, never MDN links.

The agent-runtime send path rendered raw httpx text — `Server error '500
...' For more information check: https://developer.mozilla.org/...` — while
the response body's actionable hint ("you may need to provide the mmproj")
was discarded, and the re-sent image poisoned every later send with the
same undiagnosable 500 (UX review finding
j3-provider-error-discards-detail-poisons-conversation, REGRESSION).
"""

import httpx
import pytest

from tldw_chatbook.Agents.hook_permissions import HookPermissions

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatModelUnavailableError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    describe_stream_failure,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from Tests.console_provider_doubles import provider_resolution

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
        return provider_resolution(ready=True, provider="llama_cpp", visible_copy="")

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        raise _http_500(MMPROJ_BODY)
        yield  # pragma: no cover — makes this an async generator


@pytest.mark.asyncio
async def test_agent_failure_row_carries_body_and_image_recovery_hint(tmp_path):
    gateway = _ExplodingGateway()
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    # This regression exercises provider failure rendering, not durable-chat
    # publication. A temporary chat is the supported persistence-free path.
    session = store.create_session(title="Chat 1", ephemeral=True)
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
    system_rows = [m.content for m in messages if m.role is ConsoleMessageRole.SYSTEM]
    assert system_rows, "no failure system row appended"
    failure_row = system_rows[-1]
    assert "provide the mmproj" in failure_row
    assert "developer.mozilla.org" not in failure_row
    # Recovery hint: the conversation carries an image the provider may be
    # rejecting — point at it and at the existing remove/switch affordances.
    assert "image" in failure_row.lower()
    assert "remove" in failure_row.lower() or "vision" in failure_row.lower()


# F5: one status per error. The stream worker used to stringify the adapter
# exception into prose (baking the REAL status into text), then the
# consumer re-raised with only that text -- dropping the status code, so
# ChatProviderError's 502 wrapper default won and describe_stream_failure
# read "HTTP 502" while the baked-in prose said "Status: 400." in the same
# sentence.


@pytest.mark.asyncio
async def test_stream_chat_provider_400_reports_one_consistent_status() -> None:
    def fake_chat_api_call(**_kwargs):
        raise ChatBadRequestError("invalid temperature", provider="openai")

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    with pytest.raises(ChatBadRequestError) as exc_info:
        _ = [
            item
            async for item in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

    err = exc_info.value
    # The real adapter status (400) must survive onto the re-raised wrapper
    # -- not the wrapper's own 502 upstream-error default.
    assert err.status_code == 400

    copy = describe_stream_failure(err)
    assert "HTTP 400" in copy
    assert "502" not in copy


@pytest.mark.asyncio
async def test_stream_chat_generic_failure_without_status_still_defaults_502() -> None:
    """A failure with no real HTTP status (e.g. a bare RuntimeError from the
    adapter layer) keeps the wrapper's 502 upstream-error default -- there is
    no real status to carry, so this is not a regression of the same bug."""

    def fake_chat_api_call(**_kwargs):
        raise RuntimeError("adapter exploded before any HTTP response")

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [
            item
            async for item in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

    assert exc_info.value.status_code == 502


@pytest.fixture
def provider_failure_hooks():
    """Use the admitted private profile's real permission owner, without retargeting."""
    owner = HookPermissions()
    assert owner.snapshot().ready
    try:
        yield owner
    finally:
        owner.close()


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "status"),
    [
        (ChatBadRequestError, 400),
        (ChatAuthenticationError, 401),
        (ChatBadRequestError, 404),
        (ChatModelUnavailableError, 404),
        (ChatRateLimitError, 429),
    ],
)
async def test_typed_gateway_error_reaches_console_without_changing_diagnostics(
    error_type, status, provider_failure_hooks
):
    """Actual Console copy survives typed projection; diagnostics stay content-free."""
    model = "claude-3-haiku-20240307"
    raw_message = "retired model; Authorization: Bearer SECRET-CANARY"
    if error_type is ChatRateLimitError:
        original = error_type(raw_message, provider="anthropic", retry_after=17)
    elif error_type is ChatAuthenticationError:
        original = error_type(raw_message, provider="anthropic")
    else:
        original = error_type(raw_message, provider="anthropic", status_code=status)

    def fail(**_kwargs):
        raise original

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "test-key"}}
        },
        chat_api_call_fn=fail,
    )
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="anthropic",
        model=model,
        agent_runtime_enabled=False,
        hook_permissions_accessor=lambda: provider_failure_hooks,
    )
    session = store.create_session(title="Provider failure", ephemeral=True)
    toasts = []
    controller.notify_run_failure = toasts.append
    result = await controller.submit_draft("hello")
    rows = [
        message.content
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]
    assert result.accepted
    assert rows
    visible = rows[-1]
    assert toasts == [visible]
    assert "Provider error from Anthropic" in visible
    assert f"Status: {status}" in visible
    assert "SECRET-CANARY" not in visible
    if status in (400, 404):
        assert model in visible
        assert "choose another model from the model picker" in visible
    if status == 404:
        assert "The provider could not find this model or endpoint" in visible
        assert "Check the model name and the key" in visible

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model=model)
    )
    with pytest.raises(ChatAPIError) as caught:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hello"}]
            )
        ]
    projected = caught.value
    assert type(projected) is error_type
    assert projected.provider == "anthropic"
    assert projected.status_code == status
    assert str(projected) == str(error_type())
    assert projected.message == str(error_type())
    assert model not in describe_stream_failure(projected)
    assert "SECRET-CANARY" not in describe_stream_failure(projected)
    assert "Provider error from Anthropic" not in describe_stream_failure(projected)
    if error_type is ChatRateLimitError:
        assert projected.retry_after == 17


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404])
async def test_typed_console_presentation_never_enters_durable_agent_error(
    tmp_path, provider_failure_hooks, status
):
    """A real gateway projection must retain content-free STEP_ERROR persistence."""

    def fail(**_kwargs):
        raise ChatBadRequestError(
            "Authorization: Bearer SECRET-CANARY",
            provider="anthropic",
            status_code=status,
        )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "test-key"}}
        },
        chat_api_call_fn=fail,
    )
    model = "claude-3-haiku-20240307"
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model=model)
    )
    with pytest.raises(ChatBadRequestError) as caught:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hello"}]
            )
        ]
    assert model in caught.value.console_copy

    class ProjectedGateway(_ExplodingGateway):
        async def stream_chat(self, _resolution, _messages, **_kwargs):
            raise caught.value
            yield  # pragma: no cover

    projected_gateway = ProjectedGateway()
    store = ConsoleChatStore()
    db = AgentRunsDB(tmp_path / "typed-runs.db", client_id="typed-copy")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=projected_gateway
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=projected_gateway,
        provider="llama_cpp",
        model="test-model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
        hook_permissions_accessor=lambda: provider_failure_hooks,
    )
    session = store.create_session(title="Typed failure", ephemeral=True)
    result = await controller.submit_draft("hello")
    assert result.accepted
    visible = [
        message.content
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.SYSTEM
    ][-1]
    assert "Provider error from Anthropic" in visible
    assert model in visible
    assert "choose another model from the model picker" in visible
    assert f"Status: {status}" in visible
    assert "SECRET-CANARY" not in visible
    runs = db.list_runs(session.id)
    assert runs
    steps = [step for run in runs for step in run["steps"]]
    errors = [step for step in steps if step["kind"] == "error"]
    assert errors
    summary = errors[-1]["summary"]
    assert f"HTTP {status}" in summary
    assert "Invalid request sent to the chat provider." in summary
    assert model not in summary
    assert "Provider error from Anthropic" not in summary
    assert "SECRET-CANARY" not in summary

    from tldw_chatbook.Agents.run_log import resolve_existing_log_dir

    authority = bridge._run_log_authority_for(runs[0]["id"])
    assert authority is not None
    with authority.access_scope():
        log_dir = resolve_existing_log_dir(runs[0]["id"], root=authority.root)
        assert log_dir is not None
        records = [path.read_text() for path in log_dir.iterdir() if path.is_file()]
    assert records
    durable = "\n".join(records)
    assert model not in durable
    assert "Provider error from Anthropic" not in durable
    assert "SECRET-CANARY" not in durable
    db.close()
