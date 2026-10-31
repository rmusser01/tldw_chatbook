"""TASK-33621.11: Console failure copy follows typed failure facts, not text.

Every case drives a provider failure through the real
``ConsoleProviderGateway`` (only ``chat_api_call`` is replaced, raising the
same exception shape ``Chat_Functions.chat_api_call`` maps an HTTP error to),
the real ``ConsoleChatController`` and store, and -- for the default Console
path -- the real ``ConsoleAgentBridge`` and agent loop. The assertion is the
System row the user reads.
"""

from __future__ import annotations

import json

import pytest
import requests

from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatProviderError,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.provider_failures import describe_stream_failure
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile

MODEL = "claude-haiku-4-5"
_CONFIG = {"api_settings": {"anthropic": {"api_key": "test-key"}}}

#: The gateway's finished sentence for the injected 401 (category, status,
#: the provider's own words, the fix) -- the whole System row, once.
AUTH_COPY = (
    "Provider error from Anthropic: authentication failed. Status: 401. "
    "Anthropic says: “invalid x-api-key”. Update the API key in Settings ▸ "
    "Providers & Models, or run Ctrl+P ▸ Setup: Run setup wizard."
)
LEAD_INS = ("Agent run failed:", "Provider stream failed:", "provider returned HTTP")


def _failing_call(status: int, error: dict, factory):
    """A ``chat_api_call`` that fails as the real one maps an HTTP error."""

    def fail(**_kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps({"type": "error", "error": error}).encode()
        try:
            raise requests.HTTPError(f"{status} Error", response=response)
        except requests.HTTPError:
            raise factory()

    return fail


def _auth_failure():
    return _failing_call(
        401,
        {"type": "authentication_error", "message": "invalid x-api-key"},
        lambda: ChatAuthenticationError("Authentication failed", provider="anthropic"),
    )


def _server_failure():
    return _failing_call(
        500,
        {"type": "api_error", "message": "Internal server error"},
        lambda: ChatProviderError("boom", provider="anthropic", status_code=500),
    )


@pytest.fixture
def hooks():
    """The admitted bootstrap profile's real permission owner."""
    owner = HookPermissions()
    assert owner.snapshot().ready
    try:
        yield owner
    finally:
        owner.close()


def _console(tmp_path, hooks, call, *, agent: bool, bridge_type=ConsoleAgentBridge):
    gateway = ConsoleProviderGateway(
        config_provider=lambda: _CONFIG, chat_api_call_fn=call
    )
    store = ConsoleChatStore()
    extra = {}
    if agent:
        extra["agent_bridge"] = bridge_type(
            agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="copy"),
            store=store,
            provider_gateway=gateway,
        )
        store.bridge = extra["agent_bridge"]  # test handle only
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="anthropic",
        model=MODEL,
        agent_runtime_enabled=agent,
        hook_permissions_accessor=lambda: hooks,
        **extra,
    )
    session = store.create_session(title="Failure copy", ephemeral=True)
    return gateway, store, controller, session


def _with_image_turn(store, session) -> None:
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="what is in this image?",
        image_data=b"\x89PNG-fake",
        image_mime_type="image/png",
    )


def _system_row(store, session) -> str:
    rows = [
        message.content
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]
    assert rows, "no failure System row"
    return rows[-1]


async def _projected(gateway, call_kind: str = "stream") -> ChatAPIError:
    """The exception the real gateway raises for the injected failure."""
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model=MODEL)
    )
    with pytest.raises(ChatAPIError) as caught:
        _ = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "hello"}]
            )
        ]
    return caught.value


# --- AC#4: one wrapper sentence ------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
async def test_provider_401_reads_as_one_sentence(tmp_path, hooks, agent):
    """The 401 row is the gateway's sentence alone, with no second lead-in.

    Live on dev: "Agent run failed: Provider error from Anthropic:
    authentication failed. Status: 401. ..." (and "Provider stream failed:
    ..." on the direct path).
    """
    _gateway, store, controller, session = _console(
        tmp_path, hooks, _auth_failure(), agent=agent
    )
    # The live 401 followed a turn whose image the same model had just read.
    _with_image_turn(store, session)
    result = await controller.submit_draft("hello")
    assert result.accepted
    visible = _system_row(store, session)
    assert visible == AUTH_COPY
    assert not any(lead in visible for lead in LEAD_INS)
    assert controller.run_state.visible_copy == AUTH_COPY


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
async def test_provider_500_reads_as_one_sentence(tmp_path, hooks, agent):
    """A generic provider failure (no typed projection) is not wrapped either."""
    gateway, store, controller, session = _console(
        tmp_path, hooks, _server_failure(), agent=agent
    )
    # The gateway's finished sentence (its message for an unprojected error).
    expected = str(await _projected(gateway))
    assert expected.startswith("Provider error from Anthropic: ")
    assert "Status: 500." in expected
    await controller.submit_draft("hello")
    visible = _system_row(store, session)
    assert visible == expected
    assert not any(lead in visible for lead in LEAD_INS)


class _EscapingBridge(ConsoleAgentBridge):
    """A bridge whose failure escapes the agent loop's own guard.

    The controller's catch-all (bridge failures outside ``run_agent_loop``)
    renders it; the raised error is the real gateway projection.
    """

    escaping: BaseException | None = None

    def run_reply(self, **_kwargs):
        assert self.escaping is not None
        raise self.escaping


@pytest.mark.asyncio
async def test_escaped_bridge_failure_reads_as_one_sentence(tmp_path, hooks):
    gateway, store, controller, session = _console(
        tmp_path, hooks, _auth_failure(), agent=True, bridge_type=_EscapingBridge
    )
    projected = await _projected(gateway)
    assert projected.console_copy == AUTH_COPY
    store.bridge.escaping = projected
    await controller.submit_draft("hello")
    visible = _system_row(store, session)
    assert visible.startswith(AUTH_COPY)
    assert not any(lead in visible for lead in LEAD_INS)


def test_durable_summary_wrapping_is_decided_by_type_not_text():
    """The run log's summary wraps by what raised it, never by its text.

    Copy that merely CONTAINS "Status: 500." (an upstream message, say) is
    not the gateway's finished sentence and keeps its classified summary;
    dev returned it bare because it matched the gateway's wording.
    """
    foreign = ChatProviderError(
        "upstream said Status: 500.", provider="anthropic", status_code=500
    )
    assert describe_stream_failure(foreign) == (
        "provider returned HTTP 500 (upstream said Status: 500)"
    )


@pytest.mark.asyncio
async def test_durable_summary_of_a_gateway_failure_is_not_doubled(tmp_path, hooks):
    gateway, _store, _controller, _session = _console(
        tmp_path, hooks, _server_failure(), agent=False
    )
    projected = await _projected(gateway)
    summary = describe_stream_failure(projected)
    assert summary == str(projected)
    assert summary.count("500") == 1
    assert "provider returned HTTP" not in summary

