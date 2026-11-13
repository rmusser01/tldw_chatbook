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
control, each sent once in a saved chat and once in a Temporary chat. One
kimi-k3 row streams: its adapter returns the real ``MoonshotStream`` over the
real ``HostedChatStream`` SSE parser, so the gateway's streamed consumption of
the continuation (read after the stream is exhausted) is covered too.
"""

from __future__ import annotations

import asyncio
import json
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
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatStream, HostedChatTurn
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.moonshot import (
    _FINISH_POLICY,
    MoonshotStream,
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
SMOKE_MATRIX = [
    pytest.param("moonshot", model, False, id=f"moonshot-{model}")
    for model in MOONSHOT_MODELS
] + [
    pytest.param("moonshot", "kimi-k3", True, id="moonshot-kimi-k3-streaming"),
    pytest.param("openai", "gpt-4.1", False, id="openai-gpt-4.1"),
]
BLOCKED_COPY = (
    "could not be persisted",
    "delivery status is unknown",
    "Send blocked",
)


def _sse(model: str, *, delta=None, finish=None, usage=None) -> SSERecord:
    body: dict = {
        "id": "chatcmpl-34100",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": (
            []
            if usage is not None
            else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
        ),
    }
    if usage is not None:
        body["usage"] = usage
    return SSERecord(None, json.dumps(body))


def _moonshot_stream(model: str, resolution, provider_continuations) -> MoonshotStream:
    """The real stream parser over the SSE records Moonshot sends for one reply."""
    records = [
        _sse(
            model,
            delta={"role": "assistant", "content": "", "reasoning_content": REASONING},
        ),
        _sse(model, delta={"content": REPLY}),
        _sse(model, finish="stop"),
        _sse(
            model,
            usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        ),
        SSERecord(None, "[DONE]"),
    ]
    return MoonshotStream(
        HostedChatStream(iter(records), finish_policy=_FINISH_POLICY),
        resolution=resolution,
        provider_continuations=provider_continuations,
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
        if kwargs.get("streaming"):
            return _moonshot_stream(
                resolution.model,
                resolution,
                tuple(kwargs.get("provider_continuations") or ()),
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
async def _mounted_console(
    tmp_path, monkeypatch, *, provider: str, model: str, streaming: bool = False
):
    monkeypatch.setattr(
        console_runtime, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 600.0
    )
    # Native configuration capture needs the profile-bound file sources the
    # factory installs for this directory (dev's TASK-39147 fixture repair);
    # without them every mounted send is refused before the controller.
    app = _build_test_app(user_data_dir=get_user_data_dir())
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
                return replace(await original_resolve(selection), streaming=streaming)

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            yield SimpleNamespace(
                host=host,
                pilot=pilot,
                console=console,
                controller=controller,
                store=controller.store,
                database=database,
                calls=calls,
            )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


async def _wait_for_turns(h, session_id: str) -> None:
    """Wait until the runtime no longer holds a turn for ``session_id``.

    Dev's TASK-39147 harness repair waits on the runtime's own custody record;
    the earlier ``accept_turn`` wrapper no longer sees a mounted send's turn,
    so waiting on the tasks it captured returned at once.
    """
    await _until(
        h.pilot,
        lambda: not h.console._console_runtime().has_custodied_turns(session_id),
        timeout=30,
        what="The Console turn did not retire.",
    )
    await h.console._sync_native_console_chat_ui()
    await h.pilot.pause(0.3)


async def _send(h, text: str) -> None:
    """Send ``text`` in the active chat through the visible send action."""
    session_id = h.store.active_session_id
    h.console._session._sync_console_session_draft()
    h.console._console_composer_or_none().load_draft(text)
    await asyncio.wait_for(
        h.console._send_console_message_from_visible_action(
            session_id=h.store.active_session_id
        ),
        10,
    )
    await _wait_for_turns(h, session_id)


async def _send_in_temporary_chat(h, text: str) -> None:
    """Send in a Temporary chat, answering a capture question if one is asked.

    When gap-02 was filed, a Temporary chat with trace capture on paused
    before the provider and offered Save & Send, Send without capture, or
    Cancel; the live run chose Send without capture. Dev's
    ``temporary_chat_capture`` now leaves capture off for a Temporary chat
    unless the user turns it on, so the send goes straight to the provider.
    Either way the chat must reach the provider exactly once.
    """
    calls_before = len(h.calls)
    session_id = h.store.active_session_id
    await _send(h, text)
    if len(h.calls) == calls_before:
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
@pytest.mark.parametrize(("provider", "model", "streaming"), SMOKE_MATRIX)
async def test_saved_chat_replies_like_a_temporary_chat(
    tmp_path, monkeypatch, provider: str, model: str, streaming: bool
) -> None:
    async with _mounted_console(
        tmp_path, monkeypatch, provider=provider, model=model, streaming=streaming
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
        # The streaming row really streamed: the adapter returned a stream.
        assert [bool(call.get("streaming")) for call in h.calls] == [streaming] * 3
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

            # The run's recorded trace says what was lost, by code.
            from tldw_chatbook.Chat.provider_continuation_diagnostics import (
                continuation_reasons,
            )

            runs = h.controller._agent_bridge.runs_db.list_runs(conversation_id)
            notes = [
                step
                for run in runs
                for step in run["steps"]
                if step.get("summary") == "Reasoning not preserved for replay"
            ]
            assert len(notes) == 1, runs
            assert continuation_reasons(notes[0]["field_states"]) == {
                "continuation_error": "final_persist_failed",
                "continuation_store_failure": "terminal_settlement_failed",
            }

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


TRACE_BLOCKED = "Trace capture blocked"
SAVED_TURNS = 5


def _assistant_rows(database, conversation_id: str) -> list:
    rows = database.get_messages_for_conversation(conversation_id, limit=100)
    return [row for row in rows if row["role"] == "assistant"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "model", "streaming"), SMOKE_MATRIX)
async def test_saved_chat_five_turns_with_trace_capture_on(
    tmp_path, monkeypatch, provider: str, model: str, streaming: bool
) -> None:
    """A saved chat keeps replying, and tracing, past its second turn.

    With trace capture on (the default), turn 3 of a saved Kimi chat used to
    stop before the provider with "Trace capture blocked"
    (``unsupported_surface_change``): each turn replays every earlier reply's
    reasoning checkpoint, so the request grows in two domains at once (new
    messages, plus a new checkpoint after the recorded ones), and the planner
    compared the request as one flat list. TASK-34100.3 pass 3 (folding in
    TASK-34100.3.1 AC#1/AC#5).
    """
    async with _mounted_console(
        tmp_path, monkeypatch, provider=provider, model=model, streaming=streaming
    ) as h:
        saved = h.store.ensure_session()
        for turn in range(1, SAVED_TURNS + 1):
            await _send(h, f"Turn {turn}")

            def settled(turn: int = turn) -> bool:
                screen = " ".join(_frame_rows(h.host))
                replies = [
                    message.content
                    for message in h.store.messages_for_session(saved.id)
                    if message.role is ConsoleMessageRole.ASSISTANT
                ]
                return TRACE_BLOCKED in screen or replies == [REPLY] * turn

            await _until(
                h.pilot, settled, timeout=30.0, what=f"turn {turn} did not settle"
            )
            assert TRACE_BLOCKED not in " ".join(_frame_rows(h.host)), (
                f"turn {turn} hit the trace pause"
            )
            view = await _observed(h, saved.id)
            assert view["blocked_copy"] == [], f"turn {turn}"
            assert view["assistant_texts"] == [REPLY] * turn, f"turn {turn}"
            assert view["run_status"] is ConsoleRunStatus.COMPLETED, f"turn {turn}"
            assert view["dispatch_recovery"] is None, f"turn {turn}"
        assert len(h.calls) == SAVED_TURNS

        # Each call replays every earlier reply's reasoning, oldest first.
        reasoning_model = provider == "moonshot" and (
            moonshot_model_returns_reasoning_content(model)
        )
        for index, call in enumerate(h.calls):
            replayed = tuple(call.get("provider_continuations") or ())
            assert [(item.model, item.state) for item in replayed] == (
                [(model, "complete")] * index if reasoning_model else []
            ), f"call {index + 1}"

        # The trace records each request exactly: one traced call per turn,
        # whose ledger reconstruction is the request the provider received.
        from tldw_chatbook.Chat.console_trace_native_reader import (
            ConsoleTraceNativeReader,
        )

        users = [
            message
            for message in h.store.messages_for_session(saved.id)
            if message.role is ConsoleMessageRole.USER
        ]
        assert len(users) == SAVED_TURNS
        reader = ConsoleTraceNativeReader(h.database)
        for index, (user, call) in enumerate(zip(users, h.calls, strict=True)):
            traced = reader.read_calls(user.persisted_message_id)
            assert len(traced) == 1, f"turn {index + 1}: {traced}"
            request = traced[0].capture.request
            assert [
                (row.get("role"), row.get("content"))
                for row in request["messages_payload"]
            ] == [
                (row.get("role"), row.get("content"))
                for row in call["messages_payload"]
            ], f"turn {index + 1}"
            assert len(request.get("provider_continuations") or ()) == len(
                call.get("provider_continuations") or ()
            ), f"turn {index + 1}"

        # A relaunch reads exactly one row per reply, in one unbroken chain.
        conversation_id = next(
            session for session in h.store.sessions() if session.id == saved.id
        ).persisted_conversation_id
        reopened = CharactersRAGDB(tmp_path / "chat.sqlite", "task-34100-3-relaunch")
        try:
            rows = reopened.get_messages_for_conversation(conversation_id, limit=100)
            assert [row["role"] for row in rows] == ["user", "assistant"] * SAVED_TURNS
            assert [row["parent_message_id"] for row in rows[1:]] == [
                row["id"] for row in rows[:-1]
            ]
            assert [
                (row["content"], row["assistant_generation_state"])
                for row in _assistant_rows(reopened, conversation_id)
            ] == [(REPLY, "complete")] * SAVED_TURNS
            assert [
                row["provider_continuation_json"] is not None
                for row in _assistant_rows(reopened, conversation_id)
            ] == [reasoning_model] * SAVED_TURNS
            assert (
                reopened.get_connection()
                .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )
        finally:
            reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        pytest.param("moonshot", "kimi-k3", id="moonshot-kimi-k3"),
        pytest.param("moonshot", "kimi-k2.6", id="moonshot-kimi-k2.6"),
        pytest.param("openai", "gpt-4.1", id="openai-gpt-4.1"),
    ],
)
async def test_continue_and_regenerate_keep_one_row_per_reply(
    tmp_path, monkeypatch, provider: str, model: str
) -> None:
    """Continue and Regenerate in a saved chat, then keep chatting with capture on.

    Brief item (c): Continue in a saved Kimi chat once stored its reply twice
    (a truncated streaming copy plus the full reply with its checkpoint, same
    parent). The race itself is pinned deterministically in
    ``Tests/Chat/test_continuation_create_flush_race.py``; this drives both
    actions through the real controller and checks the durable outcome a
    relaunch reads: exactly one row per reply, each complete with its own
    checkpoint, and no duplicate-request warning.
    """
    async with _mounted_console(
        tmp_path, monkeypatch, provider=provider, model=model
    ) as h:
        saved = h.store.ensure_session()

        def replies() -> list:
            return [
                message
                for message in h.store.messages_for_session(saved.id)
                if message.role is ConsoleMessageRole.ASSISTANT
            ]

        async def act(action, what: str) -> None:
            result = await asyncio.wait_for(action, 30)
            assert result.accepted, f"{what}: {result.visible_copy}"
            await _wait_for_turns(h, saved.id)
            await _until(
                h.pilot,
                lambda: h.controller.run_state_for(saved.id).status
                is ConsoleRunStatus.COMPLETED,
                timeout=30.0,
                what=f"{what} did not complete",
            )

        for turn in (1, 2):
            await _send(h, f"Turn {turn}")
            await _until(
                h.pilot,
                lambda turn=turn: [m.content for m in replies()] == [REPLY] * turn,
                timeout=30.0,
                what=f"turn {turn} did not reply",
            )
        await act(h.controller.continue_from_message(replies()[-1].id), "Continue")
        await act(h.controller.regenerate_message(replies()[-1].id), "Regenerate")
        await _send(h, "Turn 3")
        await _until(
            h.pilot,
            lambda: TRACE_BLOCKED in " ".join(_frame_rows(h.host))
            or h.controller.run_state_for(saved.id).status
            is ConsoleRunStatus.COMPLETED
            and len(h.calls) == 5,
            timeout=30.0,
            what="turn 3 did not settle",
        )
        view = await _observed(h, saved.id)
        assert TRACE_BLOCKED not in " ".join(_frame_rows(h.host))
        assert view["blocked_copy"] == []
        assert view["dispatch_recovery"] is None
        assert len(h.calls) == 5, "two turns, Continue, Regenerate, one turn"

        conversation_id = next(
            session for session in h.store.sessions() if session.id == saved.id
        ).persisted_conversation_id
        reopened = CharactersRAGDB(tmp_path / "chat.sqlite", "task-34100-3-relaunch")
        try:
            rows = _assistant_rows(reopened, conversation_id)
            # Turn 1, turn 2, the Continue reply, the Regenerate sibling and
            # turn 3: five replies, five rows, no orphaned streaming copy.
            assert len(rows) == 5, [
                (row["id"], row["parent_message_id"], row["content"]) for row in rows
            ]
            assert len({row["id"] for row in rows}) == 5
            assert [
                (row["content"], row["assistant_generation_state"]) for row in rows
            ] == [(REPLY, "complete")] * 5
            reasoning_model = provider == "moonshot"
            assert [
                row["provider_continuation_json"] is not None for row in rows
            ] == [reasoning_model] * 5
            assert (
                reopened.get_connection()
                .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )
        finally:
            reopened.close()
