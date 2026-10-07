"""A request stopped by a local check is not reported as a provider error (TASK-32369).

The Anthropic and Cohere handlers refuse a request with no message to send
before anything leaves the client. They used to raise ``ChatBadRequestError``
(status 400), so Console said "provider returned HTTP 400" for a request the
provider never saw; streaming then rewrapped it as a provider 502. A
status-less ``ChatConfigurationError`` naming the field keeps the blame local,
as task-32342 did for the custom OpenAI-compatible path.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderSelection,
    safe_provider_error_copy,
)
from tldw_chatbook.Chat.provider_failures import (
    describe_console_stream_failure,
    describe_stream_failure,
)
from tldw_chatbook.LLM_Calls import LLM_API_Calls
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_anthropic, chat_with_cohere
from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request

# The handlers read provider settings through the guarded config loader
# before their local check (same admission signature as test_hosted_chat.py
# in Tests/conftest.py), so keep the bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile

_SYSTEM_ONLY = [{"role": "system", "content": "Be brief."}]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test, rather than reach a real API, if a handler sends anything.

    Args:
        monkeypatch: Replaces the handlers' session factory.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the local check must stop the request before any HTTP")

    monkeypatch.setattr(LLM_API_Calls, "create_default_session", refuse)


def _assert_local(error: ChatConfigurationError, provider: str) -> None:
    assert error.status_code is None
    assert error.provider == provider
    assert error.field == "messages"


def test_anthropic_without_a_user_message_is_stopped_locally() -> None:
    """Anthropic needs a user message; the refusal carries no HTTP status."""
    with pytest.raises(ChatConfigurationError) as caught:
        chat_with_anthropic(
            input_data=_SYSTEM_ONLY, model="claude-sonnet-4-5", api_key="test-key"
        )
    _assert_local(caught.value, "anthropic")


def test_cohere_without_a_conversation_message_is_stopped_locally() -> None:
    """Cohere needs a non-system message; the refusal carries no HTTP status."""
    with pytest.raises(ChatConfigurationError) as caught:
        chat_with_cohere(
            input_data=_SYSTEM_ONLY, model="command-a-03-2025", api_key="test-key"
        )
    _assert_local(caught.value, "cohere")


def test_console_copy_says_not_sent_and_names_the_field() -> None:
    """The copy names the field and never blames the provider or shows a status."""
    error = ChatConfigurationError(
        "raw detail SECRET-CANARY",
        provider="anthropic",
        status_code=None,
        field="messages",
    )
    copy = safe_provider_error_copy("anthropic", error)
    assert copy == "Request to Anthropic not sent: it failed a local check on messages."
    assert "SECRET-CANARY" not in copy
    # No field: possibly a reply that could not be read (task-32342), so the
    # request may have been sent -- the copy must not claim otherwise.
    unnamed = ChatConfigurationError("x", provider="anthropic", status_code=None)
    assert safe_provider_error_copy("anthropic", unnamed) == (
        "Provider error from Anthropic: configuration error."
    )
    # A configuration error the provider DID answer keeps the provider copy.
    answered = ChatConfigurationError(
        "x", provider="anthropic", status_code=500, field="messages"
    )
    assert safe_provider_error_copy("anthropic", answered).startswith(
        "Provider error from Anthropic: configuration error."
    )


@pytest.mark.asyncio
async def test_a_streamed_local_refusal_reaches_console_as_not_sent() -> None:
    """Through the Console stream path with the real handler: no invented 502."""

    def call_real_handler(**_kwargs: Any) -> Any:
        return chat_with_anthropic(
            input_data=_SYSTEM_ONLY, model="claude-sonnet-4-5", api_key="test-key"
        )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "test-key"}}
        },
        chat_api_call_fn=call_real_handler,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic", explicit_model="claude-sonnet-4-5"
        )
    )
    with pytest.raises(ChatConfigurationError) as caught:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hi"}]
            )
        ]

    assert caught.value.status_code is None
    diagnostic = describe_stream_failure(caught.value)
    assert "provider returned HTTP" not in diagnostic
    assert diagnostic.startswith("the app could not build the request")
    assert diagnostic == describe_stream_failure(
        ChatConfigurationError(status_code=None)
    )
    assert "messages" not in diagnostic
    visible = describe_console_stream_failure(caught.value)
    assert "provider returned HTTP" not in visible
    assert "Anthropic not sent: it failed a local check on messages" in visible


@pytest.mark.asyncio
async def test_a_sensitive_run_through_the_default_adapter_still_names_the_field() -> (
    None
):
    """Qodo #2974: automatic runs mark the request sensitive, and chat_api_call
    then rebuilds the error with redacted text. The field must survive that."""
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "test-key"}}
        },
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="anthropic", explicit_model="claude-sonnet-4-5"
        )
    )
    # Console refuses an empty history itself, so reach the handler's check the
    # way a real turn can: Anthropic inlines only data-URL images, so a user
    # turn holding just a remote image leaves it no user message to send.
    remote_image_only = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.invalid/a.png"},
                }
            ],
        }
    ]
    with sensitive_llm_request():
        with pytest.raises(ChatConfigurationError) as caught:
            _ = [
                chunk
                async for chunk in gateway.stream_chat(resolution, remote_image_only)
            ]

    assert caught.value.status_code is None
    assert (
        "Anthropic not sent: it failed a local check on messages"
        in describe_console_stream_failure(caught.value)
    )
    assert describe_stream_failure(caught.value) == describe_stream_failure(
        ChatConfigurationError(status_code=None)
    )
    assert "messages" not in describe_stream_failure(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_enabled", [False, True])
async def test_local_refusal_copy_reaches_console_without_entering_agent_audit(
    tmp_path, agent_enabled: bool
) -> None:
    """Real handler refusal reaches the visible row and keeps durable errors generic."""
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Agents.run_log import resolve_existing_log_dir
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    def call_real_handler(**_kwargs: Any) -> Any:
        return chat_with_anthropic(
            input_data=_SYSTEM_ONLY, model="claude-sonnet-4-5", api_key="test-key"
        )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"anthropic": {"api_key": "test-key"}}
        },
        chat_api_call_fn=call_real_handler,
    )
    store = ConsoleChatStore()
    owner = HookPermissions()
    assert owner.snapshot().ready
    db = (
        AgentRunsDB(tmp_path / "local-runs.db", client_id="local-copy")
        if agent_enabled
        else None
    )
    bridge = (
        ConsoleAgentBridge(agent_runs_db=db, store=store, provider_gateway=gateway)
        if db is not None
        else None
    )
    try:
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider="anthropic",
            model="claude-sonnet-4-5",
            agent_bridge=bridge,
            agent_runtime_enabled=agent_enabled,
            hook_permissions_accessor=lambda: owner,
        )
        session = store.create_session(title="Local refusal", ephemeral=True)
        toasts = []
        controller.notify_run_failure = toasts.append
        result = await controller.submit_draft("hello")
        assert result.accepted
        rows = [
            message.content
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.SYSTEM
        ]
        assert rows
        visible = rows[-1]
        assert "Anthropic not sent: it failed a local check on messages" in visible
        assert "provider returned HTTP" not in visible
        if not agent_enabled:
            assert toasts == [visible]
        if db is not None:
            runs = db.list_runs(session.id)
            assert runs
            errors = [
                step for run in runs for step in run["steps"] if step["kind"] == "error"
            ]
            assert errors
            assert errors[-1]["summary"] == describe_stream_failure(
                ChatConfigurationError(status_code=None)
            )
            assert "messages" not in errors[-1]["summary"]
            authority = bridge._run_log_authority_for(runs[0]["id"])
            assert authority is not None
            with authority.access_scope():
                log_dir = resolve_existing_log_dir(runs[0]["id"], root=authority.root)
                assert log_dir is not None
                records = [
                    path.read_text() for path in log_dir.iterdir() if path.is_file()
                ]
            assert records
            durable = "\n".join(records)
            assert "not sent" not in durable
            assert "local check on messages" not in durable
    finally:
        owner.close()
        if db is not None:
            db.close()
