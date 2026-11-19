"""TASK-33620.10: a fresh Console streams a cloud reply as it arrives.

Owner decision, 2026-10-27: every provider streams by default
(``[chat_defaults] streaming = true`` in the shipped template). Before it,
OpenAI and Anthropic resolved Streaming Off from the shipped template: the row
read "Generating…" for the whole provider call and the full reply then landed
in one frame.

The real ChatScreen, store, controller, agent bridge and provider gateway run
here; only the provider handler (``chat_api_call``) is replaced, by a drip
provider that honours the ``streaming`` flag the gateway sends -- an
OpenAI-shaped SSE generator that parks after its first content chunk when
streaming, one blocking full completion when not. No streaming setting is
written: the session resolves it from the shipped defaults.
"""

from __future__ import annotations

import json
import threading

import pytest

from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

FIRST_CHUNK = "Partial-drip-one "
SECOND_CHUNK = "partial-drip-two."
#: Long enough for a slow CI worker to render the first chunk; the provider
#: is released by the test long before this in every passing run.
PROVIDER_HOLD_SECONDS = 20.0


class _DripProvider:
    """``chat_api_call`` stand-in that drips when asked to stream."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.streaming_flags: list[object] = []

    def _chunk(self, payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def _stream(self):
        yield self._chunk(
            {
                "id": "drip",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": FIRST_CHUNK},
                        "finish_reason": None,
                    }
                ],
            }
        )
        self.release.wait(PROVIDER_HOLD_SECONDS)
        yield self._chunk(
            {
                "id": "drip",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": SECOND_CHUNK},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        yield self._chunk(
            {
                "id": "drip",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                    "total_tokens": 18,
                },
            }
        )
        yield "data: [DONE]\n\n"

    def __call__(self, **kwargs):
        self.streaming_flags.append(kwargs.get("streaming"))
        if kwargs.get("streaming"):
            return self._stream()
        # Non-streamed: the whole reply arrives only when the call returns.
        self.release.wait(PROVIDER_HOLD_SECONDS)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": FIRST_CHUNK + SECOND_CHUNK,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        }


def _assistant_rows(store):
    return [
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ]


@pytest.mark.asyncio
async def test_a_fresh_openai_console_shows_partial_reply_text_before_the_reply_completes(
    monkeypatch,
):
    app = _build_test_app()
    attach_chachanotes_db(app)
    # This harness has no world-books schema for exchange capture (the same
    # ordinary-send policy the composer send tests use).
    assert config_module.save_settings_to_cli_config(
        {
            "console": {"exchange_capture": False},
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "chat_defaults": {"provider": "openai", "model": "gpt-4.1-mini"},
            "api_settings.openai": {"api_key": "DUMMY_OPENAI_API_KEY"},
        }
    )
    app.app_config = config_module.load_settings(force_reload=True)
    # Nothing above (or in the sandbox profile) chooses streaming: the
    # session must take it from the shipped template.
    assert "streaming" not in config_module.load_cli_config_and_ensure_existence(
        force_reload=True
    ).get("console", {}).get("provider_defaults", {}).get("openai", {})

    provider = _DripProvider()
    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", provider)
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(140, 44)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            composer.load_draft("Tell me a long story")
            await pilot.pause(0.1)
            await pilot.click("#console-send-message")
            store = console._ensure_console_chat_store()

            partial_seen = False
            for _ in range(160):
                await pilot.pause(0.05)
                rows = _assistant_rows(store)
                if (
                    FIRST_CHUNK.strip() in _visible_text(console)
                    and rows
                    and rows[-1].status == "streaming"
                ):
                    partial_seen = True
                    break
            assert partial_seen, (
                "no partial reply text appeared before the reply completed; "
                f"the provider was called with streaming={provider.streaming_flags!r}; "
                f"visible text: {_visible_text(console)[-400:]!r}"
            )
            assert provider.streaming_flags == [True]
            # The in-flight marker is the "Streaming…" header label, and the
            # rest of the reply has not arrived yet.
            assert SECOND_CHUNK not in _visible_text(console)
            assert "Streaming…" in _visible_text(console)

            provider.release.set()
            for _ in range(160):
                await pilot.pause(0.05)
                rows = _assistant_rows(store)
                if rows and rows[-1].status == "complete":
                    break
            rows = _assistant_rows(store)
            assert rows[-1].status == "complete"
            assert rows[-1].content == FIRST_CHUNK + SECOND_CHUNK

            # A reply streamed by default still records the usage the
            # stream's final chunk reported (TASK-2122), in the store and on
            # the durable row the cost display is priced from.
            usage = rows[-1].usage
            assert usage is not None
            assert (usage.uncached_input, usage.output) == (12, 6)
            persisted = None
            for _ in range(80):
                persisted = (
                    app.chachanotes_db.get_connection()
                    .execute(
                        "SELECT usage_json FROM messages "
                        "WHERE content = ? AND deleted = 0",
                        (FIRST_CHUNK + SECOND_CHUNK,),
                    )
                    .fetchone()
                )
                if persisted is not None and persisted[0]:
                    break
                await pilot.pause(0.05)
            assert persisted is not None and persisted[0], (
                "the streamed reply's durable row carries no usage_json"
            )
            stored = ProviderUsage.from_json(persisted[0])
            assert stored is not None
            assert (stored.uncached_input, stored.output) == (12, 6)
    finally:
        provider.release.set()
