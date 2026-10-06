"""TASK-34100.3 (gap-02): a Moonshot reasoning reply in a saved Console chat.

Every model Moonshot lists (kimi-k2.6, kimi-k2.7-code, kimi-k2.7-code-highspeed,
kimi-k3) returns ``reasoning_content`` on a plain reply, so the Moonshot
adapter attaches a complete reasoning-replay checkpoint to every turn. In a
saved chat the reply was generated, then thrown away: the turn showed
"Provider continuation could not be persisted", a recovery panel said delivery
status was unknown, and the composer stayed blocked. Temporary chats worked.

Root cause: the durable dispatch row records the provider destination as a
credential-free ORIGIN (``https://api.moonshot.ai``), while the continuation
carries the full base URL (``https://api.moonshot.ai/v1``). The repository
compared the two for equality, so terminal settlement always reported a
conflict. The store-level tests passed because they hand-built a destination
whose identity already carried the ``/v1`` path; nothing drove the path with
the destination the real gateway computes.

These tests mount the real ``ChatScreen`` with its real controller, store,
runtime, gateway and a file-backed ChaChaNotes database. Only the HTTP call
is replaced: the adapter returns the real ``MoonshotResponse`` that
``chat_with_moonshot`` builds from a provider turn, so the real Moonshot
continuation builder runs. The parametrization is the provider smoke matrix
for this path: every listed Kimi model, plus a non-reasoning provider as the
control, each sent once in a saved chat and once in a Temporary chat.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_blocked_send_recovery import _frame_rows, _until
from Tests.UI.test_console_native_chat_flow import _persist_console_provider_config
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat import console_runtime
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatTurn
from tldw_chatbook.LLM_Calls.moonshot import (
    _moonshot_turn_response,
    resolve_moonshot_request,
)
from tldw_chatbook.model_capabilities import moonshot_model_returns_reasoning_content

pytestmark = pytest.mark.bootstrap_profile

REPLY = "Kimi reply 34100"
REASONING = "private reasoning that never reaches the transcript"
SYNTHETIC_KEY = "synthetic-test-key"

# The models Moonshot's /v1/models listed on 2026-10-03 (gap-02 evidence).
MOONSHOT_MODELS = ("kimi-k2.6", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k3")
SMOKE_MATRIX = [("moonshot", model) for model in MOONSHOT_MODELS] + [
    ("openai", "gpt-4.1"),
]
BLOCKED_COPY = (
    "could not be persisted",
    "delivery status is unknown",
    "Send blocked",
)


def _moonshot_adapter(calls: list[dict]):
    def adapter(**kwargs):
        calls.append(kwargs)
        resolution = resolve_moonshot_request(
            explicit_api_key=SYNTHETIC_KEY,
            explicit_base_url=kwargs.get("api_base_url"),
            explicit_model=kwargs.get("model"),
            app_config={},
            environ={},
        )
        turn = HostedChatTurn(
            text=REPLY,
            tool_calls=(),
            assistant_message={
                "role": "assistant",
                "content": REPLY,
                "reasoning_content": REASONING,
            },
            finish_reason="stop",
            reasoning_content=REASONING,
            usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        )
        return _moonshot_turn_response(
            turn,
            resolution=resolution,
            provider_continuations=tuple(kwargs.get("provider_continuations") or ()),
        )

    return adapter


def _plain_adapter(calls: list[dict]):
    def adapter(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": REPLY}}]}

    return adapter


@asynccontextmanager
async def _mounted_console(tmp_path, monkeypatch, *, provider: str, model: str):
    monkeypatch.setattr(
        console_runtime, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 600.0
    )
    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "task-34100-3")
    app.chachanotes_db = database
    _persist_console_provider_config(
        app,
        provider=provider,
        model=model,
        provider_settings={"api_key": SYNTHETIC_KEY},
    )
    # The send boundary checks the saved chat is not archived; the test app
    # has no local conversation service of its own.
    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_archive_states=lambda ids: {}
    )
    host = ConsoleHarness(app)
    host.CSS_PATH = TldwCli.CSS_PATH
    calls: list[dict] = []
    adapter = (
        _moonshot_adapter(calls) if provider == "moonshot" else _plain_adapter(calls)
    )
    console = None
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            controller = console._ensure_console_chat_controller()
            gateway = controller.provider_gateway
            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            original_resolve = gateway.resolve_for_send

            async def resolve(selection):
                return replace(await original_resolve(selection), streaming=False)

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            runtime = console._console_runtime()
            tasks: list[asyncio.Task] = []
            accept = runtime.accept_turn

            def capture(request, **kwargs):
                turn_id = accept(request, **kwargs)
                tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", capture)
            yield SimpleNamespace(
                host=host,
                pilot=pilot,
                console=console,
                controller=controller,
                store=controller.store,
                database=database,
                calls=calls,
                tasks=tasks,
            )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


async def _wait_for_turns(h, started: int) -> None:
    for task in h.tasks[started:]:
        await asyncio.wait({task}, timeout=30)
    await h.console._sync_native_console_chat_ui()
    await h.pilot.pause(0.3)


async def _send(h, text: str) -> None:
    """Send ``text`` in the active chat through the visible send action."""
    started = len(h.tasks)
    h.console._session._sync_console_session_draft()
    h.console._console_composer_or_none().load_draft(text)
    await asyncio.wait_for(
        h.console._send_console_message_from_visible_action(
            session_id=h.store.active_session_id
        ),
        10,
    )
    await _wait_for_turns(h, started)


async def _send_in_temporary_chat(h, text: str) -> None:
    """Send, then answer the Temporary chat's capture question as a user would.

    With trace capture on, a Temporary chat pauses before the provider and
    offers Save & Send, Send without capture, or Cancel. The live gap-02 run
    chose Send without capture; so does this one, through the same button.
    """
    await _send(h, text)
    calls_before = len(h.calls)
    session_id = h.store.active_session_id
    await _wait_for_selector(h.console, h.pilot, "#console-trace-send-without")
    await h.pilot.click("#console-trace-send-without")
    await _until(
        h.pilot,
        lambda: len(h.calls) > calls_before,
        timeout=30.0,
        what="Send without capture did not reach the provider",
    )
    await _until(
        h.pilot,
        lambda: (
            h.controller.run_state_for(session_id).status
            is not ConsoleRunStatus.STREAMING
        ),
        timeout=30.0,
        what="the Temporary chat run did not finish",
    )
    await h.console._sync_native_console_chat_ui()
    await h.pilot.pause(0.3)


async def _observed(h, session_id: str) -> dict[str, object]:
    """What a user sees and can do after one send in ``session_id``."""
    await _until(
        h.pilot,
        lambda: (
            h.controller.run_state_for(session_id).status
            in {
                ConsoleRunStatus.COMPLETED,
                ConsoleRunStatus.FAILED,
                ConsoleRunStatus.BLOCKED,
            }
        ),
        timeout=30.0,
        what="the run did not finish",
    )
    try:
        await _until(h.pilot, lambda: REPLY in " ".join(_frame_rows(h.host)))
    except AssertionError:
        pass  # recorded as reply_on_screen=False below
    screen = "\n".join(_frame_rows(h.host))
    return {
        "assistant_texts": [
            message.content
            for message in h.store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ],
        "assistant_states": [
            message.status
            for message in h.store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ],
        "run_status": h.controller.run_state_for(session_id).status,
        "dispatch_recovery": h.store.dispatch_recovery_for_session(session_id),
        "blocked_copy": [copy for copy in BLOCKED_COPY if copy in screen],
        "reply_on_screen": REPLY in screen,
    }


EXPECTED = {
    "assistant_texts": [REPLY],
    "assistant_states": ["complete"],
    "run_status": ConsoleRunStatus.COMPLETED,
    "dispatch_recovery": None,
    "blocked_copy": [],
    "reply_on_screen": True,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "model"), SMOKE_MATRIX)
async def test_saved_chat_replies_like_a_temporary_chat(
    tmp_path, monkeypatch, provider: str, model: str
) -> None:
    async with _mounted_console(
        tmp_path, monkeypatch, provider=provider, model=model
    ) as h:
        saved = h.store.ensure_session()
        assert saved.ephemeral is False
        await _send(h, "Say hello")
        saved_view = await _observed(h, saved.id)

        await h.console._session._create_native_console_session_from_active_context(
            ephemeral=True
        )
        await h.console._sync_native_console_chat_ui()
        temporary_id = h.store.active_session_id
        assert temporary_id != saved.id
        assert h.store.session_is_ephemeral(temporary_id)
        await _send_in_temporary_chat(h, "Say hello")
        temporary_view = await _observed(h, temporary_id)

        assert saved_view == EXPECTED
        assert temporary_view == saved_view
        assert len(h.calls) == 2, "each chat must reach the provider exactly once"

        # A second turn in the saved chat replays the first reply's saved
        # reasoning (when the model keeps it) and is not blocked either.
        h.store.switch_session(saved.id)
        await h.console._sync_native_console_chat_ui()
        await _send(h, "And again")
        second_view = await _observed(h, saved.id)
        assert second_view["assistant_texts"] == [REPLY, REPLY]
        assert second_view["blocked_copy"] == []
        assert second_view["dispatch_recovery"] is None
        assert len(h.calls) == 3
        reasoning_model = provider == "moonshot" and (
            moonshot_model_returns_reasoning_content(model)
        )
        replayed = tuple(h.calls[2].get("provider_continuations") or ())
        assert [
            (checkpoint.model, checkpoint.state, checkpoint.rounds[-1].reasoning_blocks)
            for checkpoint in replayed
        ] == ([(model, "complete", (REASONING,))] if reasoning_model else [])

        conversation_id = next(
            session for session in h.store.sessions() if session.id == saved.id
        ).persisted_conversation_id
        assert conversation_id is not None
        rows = h.database.get_messages_for_conversation(conversation_id, limit=100)
        assistants = [row for row in rows if row["role"] == "assistant"]
        assert [
            (row["content"], row["assistant_generation_state"]) for row in assistants
        ] == [(REPLY, "complete"), (REPLY, "complete")]
        # The reasoning is kept for replay on the next turn, never shown.
        assert [
            row["provider_continuation_json"] is not None for row in assistants
        ] == [reasoning_model, reasoning_model]
        assert REASONING not in "\n".join(_frame_rows(h.host))
        assert (
            h.database.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_unstored_reasoning_checkpoint_keeps_the_reply(
    tmp_path, monkeypatch
) -> None:
    """AC#2: if the optional replay checkpoint still cannot be stored, keep the reply.

    The checkpoint's atomic terminal settlement is made to conflict, as every
    saved-chat Moonshot reply did before the destination fix. The reply must
    still be shown and saved as an ordinary assistant message, with no
    recovery panel and no blocked composer; only replay of the reasoning is
    lost, and the run log says so.
    """
    from loguru import logger

    from tldw_chatbook.Chat.console_dispatch_repository import (
        ConsoleDispatchRepository,
    )

    monkeypatch.setattr(
        ConsoleDispatchRepository,
        "_continuation_matches_destination",
        staticmethod(lambda row, continuation: False),
    )
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        async with _mounted_console(
            tmp_path, monkeypatch, provider="moonshot", model="kimi-k3"
        ) as h:
            saved = h.store.ensure_session()
            await _send(h, "Say hello")

            assert await _observed(h, saved.id) == EXPECTED
            conversation_id = next(
                session for session in h.store.sessions() if session.id == saved.id
            ).persisted_conversation_id
            rows = h.database.get_messages_for_conversation(conversation_id, limit=100)
            assistants = [row for row in rows if row["role"] == "assistant"]
            assert [
                (
                    row["content"],
                    row["assistant_generation_state"],
                    row["provider_continuation_json"],
                )
                for row in assistants
            ] == [(REPLY, "complete", None)]
            assert (
                h.database.get_connection()
                .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )

            # The chat stays usable: the next send reaches the provider too.
            await _send(h, "And again")
            assert len(h.calls) == 2
            assert (await _observed(h, saved.id))["blocked_copy"] == []
    finally:
        logger.remove(sink)
    assert any(
        "Reasoning not preserved for replay: store_failure=terminal_settlement_failed"
        in line
        for line in warnings
    ), warnings
