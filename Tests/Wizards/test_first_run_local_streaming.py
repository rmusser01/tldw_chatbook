"""TASK-34100.2 AC#10 (entry-exit-handoff-25): a local server streams.

LM Studio users set up through the Custom OpenAI-compatible provider got a
long silent wait and then a wall of text: the template shipped
``[api_settings.custom] streaming = false`` and setup never chose a streaming
mode. The template now ships streaming on for the local OpenAI-compatible
Custom sections, and setup turns streaming on when it commits a local
provider; a cloud provider's streaming setting is left alone (task-33620.10).
"""

from __future__ import annotations

import copy

import pytest

from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.UI.Wizards import first_run_setup_state as setup_state


def test_the_template_streams_local_openai_compatible_sections() -> None:
    api = DEFAULT_CONFIG_FROM_TOML["api_settings"]

    assert api["custom"]["streaming"] is True
    assert api["custom_2"]["streaming"] is True


def test_the_streaming_set_comes_from_the_registrys_local_servers() -> None:
    keys = setup_state.FIRST_RUN_LOCAL_STREAMING_PROVIDER_KEYS

    assert {
        "custom", "custom_2", "llama_cpp", "ollama", "vllm", "tabbyapi",
        "aphrodite", "oobabooga", "local_llm", "local_mlx_lm",
    } <= keys
    assert "koboldcpp" not in keys  # its native handler forces streaming off
    assert not keys & {"openai", "anthropic", "google", "openrouter"}


def _commit(provider: str, endpoint: str, *, streaming: bool = False):
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    # An older profile that still carries the old template value.
    config["api_settings"][provider]["streaming"] = streaming
    draft = setup_state.FirstRunProviderDraft(
        provider=provider,
        endpoint=endpoint,
        credential=setup_state.ProviderCredentialDraft("none", "", 0),
    )
    resolved = setup_state.resolve_first_run_provider_draft(draft, config)
    return setup_state.build_first_run_provider_commit(resolved, "qwen2.5-0.5b", config)


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("custom", "http://127.0.0.1:9099/v1"),
        ("custom_2", "http://127.0.0.1:1234/v1"),
        ("llama_cpp", "http://127.0.0.1:8181"),
    ],
)
def test_setup_turns_streaming_on_when_it_commits_a_local_provider(
    provider, endpoint
) -> None:
    mutation = _commit(provider, endpoint)

    assert mutation.section_values[f"api_settings.{provider}"]["streaming"] is True


def test_setup_leaves_a_cloud_providers_streaming_alone() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    draft = setup_state.FirstRunProviderDraft(
        provider="openai",
        endpoint="",
        credential=setup_state.ProviderCredentialDraft("draft", "sk-test-local-streaming", 1),
    )
    resolved = setup_state.resolve_first_run_provider_draft(draft, config)
    mutation = setup_state.build_first_run_provider_commit(resolved, "gpt-4.1", config)

    assert "streaming" not in mutation.section_values["api_settings.openai"]


def test_saving_a_template_sample_provider_clears_its_marker() -> None:
    """AC#9: setup's save makes the provider the user's own (the marker is
    written false -- deleting it would let the template's true merge back)."""
    mutation = _commit("custom", "http://127.0.0.1:9099/v1")

    assert mutation.section_values["api_settings.custom"]["template_sample"] is False


# --- Review F2: the committed section's first reply, streamed and not ---------
#
# The tests above stop at config. These send a real request through the same
# dispatcher Console uses (``chat_api_call``) to the handler the committed
# Custom section names, at a loopback stub that answers a streamed request
# with SSE and any other request with one JSON completion. The stub records
# what the handler asked for, so the non-streaming fallback is pinned at the
# wire: one plain request, one parsed reply.

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _ChatStub:
    """An OpenAI-compatible /v1/chat/completions: SSE when asked to stream."""

    def __init__(self) -> None:
        stub = self
        self.requests: list[dict] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append({"path": self.path, "body": body})
                if body.get("stream") is True:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for piece in ("Par", "is"):
                        chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                payload = json.dumps(
                    {
                        "id": "cmpl-stub",
                        "object": "chat.completion",
                        "model": body.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Paris"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_ChatStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _reply_text(reply) -> str:
    """The assistant text of a non-streamed reply (a string or a completion)."""
    if isinstance(reply, str):
        return reply
    return reply["choices"][0]["message"]["content"]


def _streamed_text(reply) -> str:
    text = ""
    for event in reply:
        data = event.strip().removeprefix("data:").strip()
        if data and data != "[DONE]":
            text += json.loads(data)["choices"][0]["delta"].get("content", "")
    return text


# The handler reads the runtime config snapshot, which needs the
# collection-time profile (as Tests/Chat/test_rate_limit_reads_as_rate_limit.py).
@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.parametrize("streaming", [True, False], ids=("streamed", "non-streaming"))
def test_the_committed_custom_section_streams_when_asked_and_falls_back(
    streaming,
) -> None:
    """Review F2 (AC#10), relabelled in review v5r2 (TE-4).

    RED part: setup commits ``streaming = true`` for the Custom section (dev
    has no such key: KeyError). The request half is a REGRESSION PIN of the
    Custom handler at the committed endpoint and model: it streams when
    asked, and with streaming off (the non-streaming fallback) sends one
    plain request and parses the reply. The test passes ``streaming`` in
    itself, so it does not show which value a Console send asks for; that is
    decided by ``[chat_defaults] streaming`` (dev's), not by this section,
    and pinned by the Console-half test below (review v5r4, TE-R4-1).
    """
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    with _ChatStub() as stub:
        section = _commit("custom", stub.base_url).section_values["api_settings.custom"]
        assert section["streaming"] is True  # what setup commits
        reply = chat_api_call(
            "custom-openai-api",
            messages_payload=[{"role": "user", "content": "Capital of France?"}],
            model=section["model"],
            streaming=streaming,
            api_base_url=section["api_url"],
        )
        text = _streamed_text(reply) if streaming else _reply_text(reply)

    assert text == "Paris"
    assert len(stub.requests) == 1
    request = stub.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["model"] == "qwen2.5-0.5b"
    assert request["body"]["stream"] is streaming


# --- Review v5r4 (TE-R4-1): AC#10's Console half ----------------------------
#
# The tests above stop at the committed section or call the handler with a
# ``streaming`` value they choose. This one sends the way Console does: the
# wizard's commit is applied to the shipped template, a new Console
# session's default settings are built from that config, the session's
# selection is resolved by the real gateway, and the real agent bridge runs
# the reply against the loopback stub. What reaches the wire is what a
# Console send asks for.


def _committed_config(endpoint: str) -> dict:
    """The shipped template with setup's Custom commit applied to it."""
    config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    draft = setup_state.FirstRunProviderDraft(
        provider="custom",
        endpoint=endpoint,
        credential=setup_state.ProviderCredentialDraft("none", "", 0),
    )
    resolved = setup_state.resolve_first_run_provider_draft(draft, config)
    mutation = setup_state.build_first_run_provider_commit(
        resolved, "qwen2.5-0.5b", config
    )
    for dotted, values in mutation.section_values.items():
        node = config
        for part in dotted.split("."):
            node = node.setdefault(part, {})
        node.update(values)
    for dotted, keys in mutation.delete_keys.items():
        node = config
        for part in dotted.split("."):
            node = node.setdefault(part, {})
        for key in keys:
            node.pop(key, None)
    return config


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("section_streaming", "chat_defaults_streaming", "streamed"),
    [
        (None, None, True),  # the commit as setup writes it
        (False, None, True),  # the section's own value does not decide
        (None, False, False),  # control: [chat_defaults] streaming decides
    ],
    ids=("committed", "section-off", "chat-defaults-off-control"),
)
def test_a_wizard_committed_custom_providers_console_send_asks_to_stream(
    tmp_path, section_streaming, chat_defaults_streaming, streamed
) -> None:
    """Review v5r4 (TE-R4-1), AC#10's Console half. REGRESSION PIN of dev's
    TASK-33620.10: passes on dev 4b90eb860b, whose shipped
    ``[chat_defaults] streaming = true`` is what a Console send reads; it
    outranks ``[api_settings.custom] streaming``, so this branch's section
    default and the wizard's ``streaming = true`` write do not change the
    request (the 'section-off' case streams too). The control turns
    ``[chat_defaults] streaming`` off and the same send asks for one plain
    reply, so the pin can see the value it guards."""
    import asyncio

    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import (
        build_console_provider_selection_from_settings,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        default_console_session_settings,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    with _ChatStub() as stub:
        config = _committed_config(stub.base_url)
        assert config["chat_defaults"]["provider"] == "custom"
        assert config["chat_defaults"]["streaming"] is True  # the shipped value
        if section_streaming is not None:
            config["api_settings"]["custom"]["streaming"] = section_streaming
        if chat_defaults_streaming is not None:
            config["chat_defaults"]["streaming"] = chat_defaults_streaming

        settings = default_console_session_settings(config)
        assert (settings.provider, settings.model) == ("custom", "qwen2.5-0.5b")
        selection = build_console_provider_selection_from_settings(
            settings, app_config=config, workspace_context=None
        )
        gateway = ConsoleProviderGateway(config_provider=lambda: config, environ={})
        resolution = asyncio.run(gateway.resolve_for_send(selection))
        assert resolution.ready, resolution.visible_copy

        store = ConsoleChatStore()
        session = store.ensure_session()
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Capital of France?"
        )
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="te-r4-1"),
            store=store,
            provider_gateway=gateway,
        )
        _run_id, outcome = bridge.run_reply(
            conversation_id="te-r4-1",
            session_id=session.id,
            resolution=resolution,
            assistant_message_id=assistant.id,
            model=resolution.model,
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "Capital of France?"}],
            should_cancel=lambda: False,
            provider_stream_signals=ConsoleProviderStreamSignals(),
        )

    assert outcome.status == "done"
    assert outcome.final_text == "Paris"
    first = stub.requests[0]
    assert first["path"] == "/v1/chat/completions"
    assert first["body"]["model"] == "qwen2.5-0.5b"
    assert first["body"]["stream"] is streamed
