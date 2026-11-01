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
import base64

import httpx
import pytest
import requests

from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
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


async def _projected(gateway) -> ChatAPIError:
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
    controller._agent_bridge.escaping = projected
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


# --- AC#3: hints follow what the provider's answer established -----------

IMAGE_HINT = ConsoleChatController._IMAGE_REJECTION_RECOVERY_HINT
MODEL_HINTS = ("choose another model", "Confirm the model", "Selected model")
NOT_FOUND = (
    f" Selected model: {MODEL}. The provider could not find this model or "
    "endpoint, or this key cannot use it. Check the model name and the key, or "
    "choose another model from the model picker (Alt+M: Switch model)."
)


def _rejected(status: int, message: str, *, code: str = "invalid_request_error"):
    """A 4xx refusal whose body carries ``message`` (Anthropic's envelope)."""
    return _failing_call(
        status,
        {"type": code, "message": message},
        lambda: ChatBadRequestError("bad", provider="anthropic", status_code=status),
    )


async def _row_for(tmp_path, hooks, call, *, image_turn: bool, agent: bool = True):
    gateway, store, controller, session = _console(tmp_path, hooks, call, agent=agent)
    if image_turn:
        _with_image_turn(store, session)
    await controller.submit_draft("hello")
    return gateway, _system_row(store, session)


def _anthropic_sentence(status: int, message: str) -> str:
    return (
        f"Provider error from Anthropic: bad request. Status: {status}. "
        f"Anthropic says: “{message}”."
    )


def _raw_http_error(status: int, message: str) -> httpx.HTTPStatusError:
    """What a direct (llama.cpp-style) adapter raises: httpx, body attached."""
    request = httpx.Request("POST", "http://127.0.0.1:9099/v1/chat/completions")
    body = json.dumps({"error": {"message": message}})
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(f"{status} Error", request=request, response=response)


@pytest.mark.asyncio
async def test_escaped_401_with_an_image_in_history_gets_no_image_hint(tmp_path, hooks):
    """Dev appended the image hint to ANY escaped failure with a response
    status (a 401 included) on the controller's catch-all path."""
    _gateway, store, controller, session = _console(
        tmp_path, hooks, _auth_failure(), agent=True, bridge_type=_EscapingBridge
    )
    _with_image_turn(store, session)
    assert isinstance(controller._agent_bridge, _EscapingBridge)
    controller._agent_bridge.escaping = _raw_http_error(401, "invalid x-api-key")
    await controller.submit_draft("hello")
    assert _system_row(store, session) == (
        "Agent run failed: provider returned HTTP 401 (invalid x-api-key)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
async def test_a_plain_400_names_neither_the_model_nor_the_image(
    tmp_path, hooks, agent
):
    """Dev told every 400 to "Confirm the model is still available, or choose
    another model" -- here the provider refused a sampling value."""
    message = "temperature: range: 0..1"
    _gateway, visible = await _row_for(
        tmp_path, hooks, _rejected(400, message), image_turn=True, agent=agent
    )
    assert visible == _anthropic_sentence(400, message)
    assert not any(hint in visible for hint in MODEL_HINTS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (
            400,
            "messages.0.content.1.image.source.base64.data: Image does not "
            "match the provided media type image/png",
        ),
        (413, "image exceeds 5 MB maximum: 7340032 bytes > 5242880 bytes"),
        (415, "Unsupported media type for image input"),
        (422, "This model does not support vision input"),
    ],
)
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
async def test_an_image_refusal_names_the_image_not_the_model(
    tmp_path, hooks, agent, status, message
):
    """A refusal whose body is about the image gets the image hint, and only
    the image hint (live on dev: OpenAI's "You uploaded an unsupported image"
    400 read "Confirm the model is still available" and never named it)."""
    _gateway, visible = await _row_for(
        tmp_path, hooks, _rejected(status, message), image_turn=True, agent=agent
    )
    assert visible == _anthropic_sentence(status, message) + IMAGE_HINT
    assert not any(hint in visible for hint in MODEL_HINTS)


@pytest.mark.asyncio
async def test_an_image_refusal_without_an_image_in_the_chat_gets_no_image_hint(
    tmp_path, hooks
):
    """The hint says "This conversation includes an image attachment"; with
    no image in the conversation that would be false."""
    message = "Image does not match the provided media type image/png"
    _gateway, visible = await _row_for(
        tmp_path, hooks, _rejected(400, message), image_turn=False
    )
    assert visible == _anthropic_sentence(400, message)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 429])
async def test_image_words_on_another_status_get_no_image_hint(tmp_path, hooks, status):
    """Only 400/413/415/422 can be an image refusal; a 500 that mentions the
    image (llama.cpp's mmproj reply) keeps the body but no hint."""
    message = "image input is not supported - hint: you may need to provide the mmproj"
    call = _failing_call(
        status,
        {"type": "api_error", "message": message},
        lambda: ChatProviderError("boom", provider="anthropic", status_code=status),
    )
    _gateway, visible = await _row_for(tmp_path, hooks, call, image_turn=True)
    assert message in visible
    assert IMAGE_HINT not in visible
    assert not any(lead in visible for lead in LEAD_INS)


@pytest.mark.asyncio
async def test_the_model_hint_is_only_for_404_and_model_not_found(tmp_path, hooks):
    """A 404, and an explicit ``model_not_found`` even on a 400, name the
    model; dev's 400 branch said "Confirm the model is still available"."""
    from tldw_chatbook.Chat.Chat_Deps import ChatModelUnavailableError

    not_found_404 = _failing_call(
        404,
        {"type": "not_found_error", "message": f"model: {MODEL}"},
        lambda: ChatBadRequestError("gone", provider="anthropic", status_code=404),
    )
    _gateway, visible = await _row_for(tmp_path, hooks, not_found_404, image_turn=True)
    assert visible == (
        "Provider error from Anthropic: model or endpoint not found. Status: 404. "
        f"Anthropic says: “model: {MODEL}”.{NOT_FOUND}"
    )

    unavailable_400 = _failing_call(
        400,
        {"type": "invalid_request_error", "message": "model_not_found"},
        lambda: ChatModelUnavailableError(provider="anthropic", status_code=400),
    )
    (tmp_path / "second").mkdir()
    _gateway, visible = await _row_for(
        tmp_path / "second", hooks, unavailable_400, image_turn=True
    )
    assert visible.endswith(NOT_FOUND), visible
    assert IMAGE_HINT not in visible


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
@pytest.mark.parametrize("staged", [True, False], ids=["staged", "history"])
async def test_tool_schema_image_name_is_not_an_image_rejection(
    tmp_path, hooks, monkeypatch, agent, staged
):
    """A sent tool schema and an admitted image must remain separate facts."""
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.Chat.provider_error_reason import is_image_input_rejection

    message = (
        "Invalid schema for function 'generate_image': schema must have type 'object'."
    )
    calls = []
    failure = _failing_call(
        400,
        {
            "type": "invalid_request_error",
            "message": message,
            "param": "tools[0].function.parameters",
            "code": "invalid_function_parameters",
        },
        lambda: ChatBadRequestError(message, provider="anthropic", status_code=400),
    )

    def call(**kwargs):
        calls.append(kwargs)
        return failure(**kwargs)

    outcomes = []

    class ObservedBridge(ConsoleAgentBridge):
        def run_reply(self, **kwargs):
            result = super().run_reply(**kwargs)
            outcomes.append(result[1])
            return result

    gateway, store, controller, session = _console(
        tmp_path, hooks, call, agent=agent, bridge_type=ObservedBridge
    )
    # The fixture supplies the exact sent schema at the existing preparation
    # seam; provider dispatch/projection/controller/agent loop remain real.
    prepare = gateway.prepare_chat_request

    def prepared(*args, **kwargs):
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Synthetic schema control",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        return prepare(*args, **kwargs)

    monkeypatch.setattr(gateway, "prepare_chat_request", prepared)
    if staged:
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
        )
        assert store.add_pending_attachment(
            session.id,
            PendingAttachment(
                file_path="leaf.png",
                display_name="leaf.png",
                file_type="image",
                insert_mode="attachment",
                data=png,
                mime_type="image/png",
                original_size=len(png),
                processed_size=len(png),
            ),
        )
    else:
        _with_image_turn(store, session)
    result = await controller.submit_draft("hello")
    assert result.accepted
    assert calls, "provider must be reached, rather than a capability/preflight refusal"
    sent = json.dumps(calls[0], default=str)
    assert "generate_image" in sent and "image" in sent
    assert any(isinstance(m.get("content"), list) for m in calls[0]["messages_payload"])
    visible = _system_row(store, session)
    assert "tool" in visible.lower() and "generate_image" in visible
    assert IMAGE_HINT not in visible
    assert not any(hint in visible for hint in MODEL_HINTS)
    if agent:
        assert outcomes and outcomes[-1].console_image_rejection is False
    projected = await _projected(gateway)
    assert projected.image_input_rejected is False
    assert is_image_input_rejection(projected) is False


@pytest.mark.parametrize(
    "body",
    [
        {
            "error": {
                "message": "Invalid schema for function 'generate_image': bad schema"
            }
        },
        {
            "error": {
                "message": "Invalid schema for function 'image_to_text': bad schema"
            }
        },
        {"error": {"code": "generate_image", "type": "vision_error"}},
        {"error": {"param": "tools[0].function.parameters.properties.image"}},
        {"error": {"message": "Schema allows image/png; see image documentation"}},
        {
            "error": {
                "message": "Schema quotes 'messages.0.content.1.image.source.data: invalid'"
            }
        },
        {"metadata": {"error": {"code": "invalid_image_format"}}},
        {"error": {"code": ["invalid_image_format"]}},
    ],
)
def test_tool_schema_carriers_do_not_establish_image_rejection(body):
    from tldw_chatbook.Chat.provider_error_reason import is_image_input_rejection

    response = requests.Response()
    response.status_code = 400
    response._content = json.dumps(body).encode()
    assert not is_image_input_rejection(requests.HTTPError(response=response))


@pytest.mark.parametrize("carrier", ["reason", "text"])
def test_tool_schema_plain_carriers_do_not_establish_image_rejection(carrier):
    from tldw_chatbook.Chat.provider_error_reason import is_image_input_rejection

    message = (
        "Invalid schema for function 'generate_image': image/png property is invalid"
    )
    error = ChatBadRequestError(status_code=400)
    if carrier == "reason":
        error.provider_reason = message
    else:
        response = requests.Response()
        response.status_code = 400
        response._content = message.encode()
        error.response = response
    assert not is_image_input_rejection(error)


@pytest.mark.parametrize(
    "error",
    [
        {"code": "invalid_image_format", "message": "Invalid request"},
        {"type": "INVALID_IMAGE_FORMAT", "message": "Invalid request"},
        {"param": "messages[0].content[1].image_url", "message": "Invalid request"},
        {"param": "content.1.image.source.base64.data", "message": "Invalid request"},
    ],
)
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
@pytest.mark.asyncio
async def test_structured_image_evidence_survives_projection(
    tmp_path, hooks, error, agent
):
    call = _failing_call(400, error, lambda: ChatBadRequestError(status_code=400))
    gateway, visible = await _row_for(
        tmp_path, hooks, call, image_turn=True, agent=agent
    )
    assert visible == _anthropic_sentence(400, "Invalid request") + IMAGE_HINT
    assert (await _projected(gateway)).image_input_rejected is True


def test_unknown_malformed_and_severed_image_evidence_fails_closed():
    from tldw_chatbook.Chat.provider_error_reason import is_image_input_rejection

    for body in (
        '{"error": {"message": "Invalid image format"',
        '{"padding":"' + "x" * 65536 + '","error":{"code":"invalid_image_format"}}',
        "image documentation",
        "null",
        "[]",
    ):
        response = requests.Response()
        response.status_code = 400
        response._content = body.encode()
        assert not is_image_input_rejection(requests.HTTPError(response=response))

    class BrokenResponse:
        @property
        def status_code(self):
            raise RuntimeError("unusable status")

    broken = RuntimeError()
    broken.response = BrokenResponse()
    assert not is_image_input_rejection(broken)
    positive = _raw_http_error(400, "Invalid image format")
    severed = RuntimeError()
    severed.__context__ = positive
    severed.__suppress_context__ = True
    assert not is_image_input_rejection(severed)
    severed.__cause__ = severed
    assert not is_image_input_rejection(severed)
    for verdict in (1, "true"):
        severed.image_input_rejected = verdict
        assert not is_image_input_rejection(severed)


def test_refused_stream_model_recovery_survives_recursion():
    from tldw_chatbook.Chat.console_provider_gateway import (
        _provider_error_copy_with_model_recovery,
    )
    from tldw_chatbook.Chat.refused_stream import REFUSED_STREAM_ADVICE

    copy = _provider_error_copy_with_model_recovery(
        "failure",
        model=MODEL,
        status_code=400,
        refused_stream=True,
        model_unavailable=True,
    )
    assert copy.endswith(REFUSED_STREAM_ADVICE)
    assert copy.count(REFUSED_STREAM_ADVICE) == 1
    assert "Selected model" in copy


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [True, False], ids=["agent", "direct"])
async def test_401_with_image_rejection_words_is_still_authentication(
    tmp_path, hooks, agent
):
    call = _failing_call(
        401,
        {"code": "invalid_image_format", "message": "Invalid image format"},
        lambda: ChatAuthenticationError(provider="anthropic"),
    )
    _gateway, visible = await _row_for(
        tmp_path, hooks, call, image_turn=True, agent=agent
    )
    assert IMAGE_HINT not in visible
    assert not any(hint in visible for hint in MODEL_HINTS)
