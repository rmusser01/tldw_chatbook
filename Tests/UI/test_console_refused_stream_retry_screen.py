"""TASK-37421: one Console turn to a Custom server that cannot stream.

Owner decision 2026-10-29: a self-hosted or Custom OpenAI-compatible server
that refuses a streamed request is asked once more without streaming, the
way llama.cpp already was. The retry must not double anything: one user
turn, one assistant reply in the transcript and in the database, and the
usage the server reported for the retry recorded once.

The real ChatScreen, store, controller, agent bridge, provider gateway,
``chat_api_call`` and ``chat_with_custom_openai`` run here, under the shipped
streaming default. Nothing is replaced: the provider is a loopback
OpenAI-compatible server that answers ``"stream": true`` with HTTP 400.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_refused_stream_retry import (
    _MODEL,
    _REFUSAL,
    _REPLY,
    _USAGE,
    _server,
)
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


def _rows(store, role):
    return [
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is role
    ]


@pytest.mark.asyncio
@pytest.mark.loopback_network
async def test_a_custom_server_that_refuses_streaming_answers_one_turn_once():
    with _server() as server:
        app = _build_test_app()
        attach_chachanotes_db(app)
        # This harness has no world-books schema for exchange capture (the
        # same ordinary-send policy the composer send tests use).
        assert config_module.save_settings_to_cli_config(
            {
                "console": {"exchange_capture": False},
                "first_run": {"setup_completed": True},
                "splash_screen": {"enabled": False},
                "chat_defaults": {"provider": "custom", "model": _MODEL},
                "api_settings.custom": {"api_url": server.base_url, "model": _MODEL},
            }
        )
        app.app_config = config_module.load_settings(force_reload=True)
        host = ConsoleHarness(app)

        async with host.run_test(size=(140, 44)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            composer.load_draft("Say hello")
            await pilot.pause(0.1)
            await pilot.click("#console-send-message")
            store = console._ensure_console_chat_store()

            for _ in range(200):
                await pilot.pause(0.05)
                rows = _rows(store, ConsoleMessageRole.ASSISTANT)
                if rows and rows[-1].status in {"complete", "failed", "error"}:
                    break
            rows = _rows(store, ConsoleMessageRole.ASSISTANT)
            for _ in range(100):
                visible = _visible_text(console)
                if _REPLY in visible:
                    break
                await pilot.pause(0.05)
            assert rows and rows[-1].status == "complete", (
                f"status={rows[-1].status if rows else None!r}; "
                f"requests={server.stream_flags!r}; visible={visible[-600:]!r}"
            )
            # Streamed first under the shipped default, refused, then exactly
            # one retry without streaming.
            assert server.stream_flags == [True, False]
            # One turn: one user row, one assistant reply, no failure row.
            assert len(_rows(store, ConsoleMessageRole.USER)) == 1
            assert len(rows) == 1
            assert rows[0].content == _REPLY
            assert _REPLY in visible
            assert _REFUSAL not in visible
            assert "Agent run failed" not in visible
            # The usage the server reported for the retry, recorded once.
            usage = rows[0].usage
            assert usage is not None
            assert (usage.uncached_input, usage.output) == (
                _USAGE["prompt_tokens"],
                _USAGE["completion_tokens"],
            )
            persisted: list = []
            for _ in range(80):
                persisted = (
                    app.chachanotes_db.get_connection()
                    .execute(
                        "SELECT usage_json FROM messages "
                        "WHERE content = ? AND deleted = 0",
                        (_REPLY,),
                    )
                    .fetchall()
                )
                if persisted and persisted[0][0]:
                    break
                await pilot.pause(0.05)
            assert len(persisted) == 1, persisted
            stored = ProviderUsage.from_json(persisted[0][0])
            assert stored is not None
            assert (stored.uncached_input, stored.output) == (
                _USAGE["prompt_tokens"],
                _USAGE["completion_tokens"],
            )
