"""TASK-34100.3 (gap-02): a Moonshot reasoning reply in a saved Console chat.

Every Kimi model Moonshot lists returns ``reasoning_content`` on a plain
reply, so the Moonshot adapter attaches a complete reasoning-replay checkpoint
to every turn. In a saved chat the reply was generated, then thrown away: the
turn showed "Provider continuation could not be persisted", a recovery panel
said delivery status was unknown, and the composer stayed blocked. Temporary
chats worked.

The original cause: the durable dispatch row recorded the provider destination
as a credential-free ORIGIN (``https://api.moonshot.ai``), while the
continuation carried the full base URL (``https://api.moonshot.ai/v1``), and
the repository compared the two for equality. Dev's TASK-35582 has since fixed
that comparison (the destination freezes the continuation tuple), so on the
current base the saved-vs-Temporary matrix below is a regression guard, not a
reproduction. What this file still proves RED on that base: turn 3+ of a saved
chat with trace capture on (the ADR-337 per-domain append), each Kimi call's
trace settling, and the kept reply when the optional checkpoint cannot be
stored.

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
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
from tldw_chatbook.model_capabilities import moonshot_model_returns_reasoning_content

pytestmark = pytest.mark.bootstrap_profile

REPLY = "Kimi reply 34100"
REASONING = "private reasoning"
SYNTHETIC_KEY = "synthetic-test-key"


def _reply(call: int) -> str:
    """The reply the provider gives to its ``call``-th request (1-based)."""
    return f"{REPLY} #{call}"


def _reasoning(call: int) -> str:
    """The reasoning the provider returns with its ``call``-th reply."""
    return f"{REASONING} #{call} that never reaches the transcript"


# The models Moonshot's /v1/models listed on 2026-11-13 (live, review v5r1).
MOONSHOT_MODELS = (
    "kimi-k2.6",
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
    "kimi-k3",
    "kimi-k3.1-pro",
    "kimi-k3.1-pro-highspeed",
)
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


def _moonshot_stream(
    model: str, resolution, provider_continuations, *, reply: str, reasoning: str
) -> MoonshotStream:
    """The real stream parser over the SSE records Moonshot sends for one reply."""
    records = [
        _sse(
            model,
            delta={"role": "assistant", "content": "", "reasoning_content": reasoning},
        ),
        _sse(model, delta={"content": reply}),
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
        reply, reasoning = _reply(len(calls)), _reasoning(len(calls))
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
                reply=reply,
                reasoning=reasoning,
            )
        turn = HostedChatTurn(
            text=reply,
            tool_calls=(),
            assistant_message={
                "role": "assistant",
                "content": reply,
                "reasoning_content": reasoning,
            },
            finish_reason="stop",
            reasoning_content=reasoning,
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
        return {"choices": [{"message": {"content": _reply(len(calls))}}]}

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
    texts = [
        message.content
        for message in h.store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ]
    latest = texts[-1] if texts else None
    try:
        await _until(
            h.pilot,
            lambda: bool(latest) and latest in " ".join(_frame_rows(h.host)),
        )
    except AssertionError:
        pass  # recorded as reply_on_screen=False below
    screen = "\n".join(_frame_rows(h.host))
    return {
        "assistant_texts": texts,
        "assistant_states": [
            message.status
            for message in h.store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ],
        "run_status": h.controller.run_state_for(session_id).status,
        "dispatch_recovery": h.store.dispatch_recovery_for_session(session_id),
        "blocked_copy": [copy for copy in BLOCKED_COPY if copy in screen],
        "reply_on_screen": bool(latest) and latest in screen,
    }


def _expected(*replies: str) -> dict[str, object]:
    """What a user sees after a clean send whose chat holds ``replies``."""
    return {
        "assistant_texts": list(replies),
        "assistant_states": ["complete"] * len(replies),
        "run_status": ConsoleRunStatus.COMPLETED,
        "dispatch_recovery": None,
        "blocked_copy": [],
        "reply_on_screen": True,
    }


def _replayed(checkpoints) -> list[tuple]:
    """Each replayed checkpoint's model, state and rounds, in replay order."""
    return [
        (
            checkpoint.model,
            checkpoint.state,
            [
                (round_.assistant_content, tuple(round_.reasoning_blocks))
                for round_ in checkpoint.rounds
            ],
        )
        for checkpoint in checkpoints
    ]


def _traced(checkpoints) -> list[tuple]:
    """``_replayed`` for the checkpoint mappings a trace reconstruction holds."""
    return [
        (
            checkpoint["model"],
            checkpoint["state"],
            [
                (round_["assistant_content"], tuple(round_["reasoning_blocks"]))
                for round_ in checkpoint["rounds"]
            ],
        )
        for checkpoint in checkpoints
    ]


def _kimi_checkpoint(model: str, call: int) -> tuple:
    """The checkpoint the ``call``-th Kimi reply leaves for replay."""
    return (model, "complete", [(_reply(call), (_reasoning(call),))])


def _saved_checkpoints(database, conversation_id: str) -> list[tuple]:
    """Each saved reply's own checkpoint, in reply order (``_replayed`` form)."""
    return [
        _replayed(
            (parse_provider_continuation_json(row["provider_continuation_json"]),)
        )[0]
        for row in _assistant_rows(database, conversation_id)
        if row["provider_continuation_json"] is not None
    ]


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

        assert saved_view == _expected(_reply(1))
        # The same view, holding the Temporary chat's own (second) reply.
        assert temporary_view == {**saved_view, "assistant_texts": [_reply(2)]}
        assert len(h.calls) == 2, "each chat must reach the provider exactly once"

        # A second turn in the saved chat replays the first reply's saved
        # reasoning (when the model keeps it) and is not blocked either.
        h.store.switch_session(saved.id)
        await h.console._sync_native_console_chat_ui()
        await _send(h, "And again")
        second_view = await _observed(h, saved.id)
        assert second_view["assistant_texts"] == [_reply(1), _reply(3)]
        assert second_view["blocked_copy"] == []
        assert second_view["dispatch_recovery"] is None
        assert len(h.calls) == 3
        # The streaming row really streamed: the adapter returned a stream.
        assert [bool(call.get("streaming")) for call in h.calls] == [streaming] * 3
        reasoning_model = provider == "moonshot" and (
            moonshot_model_returns_reasoning_content(model)
        )
        # Only the saved chat's own reply is replayed, never the Temporary one.
        assert _replayed(h.calls[2].get("provider_continuations") or ()) == (
            [_kimi_checkpoint(model, 1)] if reasoning_model else []
        )

        conversation_id = next(
            session for session in h.store.sessions() if session.id == saved.id
        ).persisted_conversation_id
        assert conversation_id is not None
        rows = h.database.get_messages_for_conversation(conversation_id, limit=100)
        assistants = [row for row in rows if row["role"] == "assistant"]
        assert [
            (row["content"], row["assistant_generation_state"]) for row in assistants
        ] == [(_reply(1), "complete"), (_reply(3), "complete")]
        # The reasoning is kept for replay on the next turn, never shown.
        assert [
            row["provider_continuation_json"] is not None for row in assistants
        ] == [reasoning_model, reasoning_model]
        assert REASONING not in "\n".join(_frame_rows(h.host))
        # Each reply keeps its own checkpoint, not a neighbour's.
        assert [
            _replayed((parse_provider_continuation_json(row["provider_continuation_json"]),))
            for row in assistants
            if row["provider_continuation_json"] is not None
        ] == (
            [[_kimi_checkpoint(model, 1)], [_kimi_checkpoint(model, 3)]]
            if reasoning_model
            else []
        )
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

            assert await _observed(h, saved.id) == _expected(_reply(1))
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
            ] == [(_reply(1), "complete", None)]
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


@pytest.mark.asyncio
async def test_destination_refused_reasoning_checkpoint_fails_the_reply(
    tmp_path, monkeypatch
) -> None:
    """A first Kimi final refused by a destination binding check fails the reply.

    The reply-keeping fallback above covers a checkpoint the store could not
    write. TASK-35582's destination and binding refusals stay fail-closed
    (review v5r1 F9): the checkpoint names another provider destination than
    the one the send was accepted for. This pins what the user then gets, as
    the User Guide states it (pre-merge review, 2026-11-13): the reply text
    stays visible but is saved as failed, without its reasoning, under an
    "Agent run failed: Provider continuation could not be persisted" error
    row; the run fails; there is NO response-recovery card or dispatch
    checkpoint; and the next Send reaches the provider and completes.

    The refusal is the real one: the dispatch the store holds is presented
    once as accepted for Moonshot's China endpoint, so the real binding check
    (``validate_durable_continuation_binding``) raises "Provider continuation
    destination mismatch."
    """
    from tldw_chatbook.Chat import console_continuation_binding as binding
    from tldw_chatbook.Chat.provider_continuation_diagnostics import (
        continuation_reasons,
    )

    original = binding.validate_durable_continuation_binding
    moved: list[str] = []

    def moved_destination(message, checkpoint, target, recovery, **kwargs):
        if not moved and recovery is not None and recovery.checkpoint is not None:
            accepted = recovery.checkpoint
            recovery = replace(
                recovery,
                checkpoint=replace(
                    accepted,
                    resolved_destination=replace(
                        accepted.resolved_destination,
                        continuation_base_url="https://api.moonshot.cn/v1",
                    ),
                ),
            )
            moved.append(message.id)
        return original(message, checkpoint, target, recovery, **kwargs)

    monkeypatch.setattr(
        binding, "validate_durable_continuation_binding", moved_destination
    )
    async with _mounted_console(
        tmp_path, monkeypatch, provider="moonshot", model="kimi-k3"
    ) as h:
        saved = h.store.ensure_session()
        await _send(h, "Say hello")

        view = await _observed(h, saved.id)
        assert len(moved) == 1, "the first final must meet the moved destination"
        assert view == {
            "assistant_texts": [_reply(1)],
            "assistant_states": ["failed"],
            "run_status": ConsoleRunStatus.FAILED,
            "dispatch_recovery": None,
            "blocked_copy": ["could not be persisted"],
            "reply_on_screen": True,
        }
        screen = " ".join(_frame_rows(h.host))
        assert (
            "Agent run failed: Provider continuation could not be persisted; "
            "retry or recover the interrupted run." in screen
        )
        assert "delivery status is unknown" not in screen
        assert "Send blocked" not in screen
        conversation_id = next(
            session for session in h.store.sessions() if session.id == saved.id
        ).persisted_conversation_id
        assert [
            (
                row["content"],
                row["assistant_generation_state"],
                row["provider_continuation_json"],
            )
            for row in _assistant_rows(h.database, conversation_id)
        ] == [(_reply(1), "failed", None)]
        assert (
            h.database.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
        # The run failed on the binding refusal itself, by code.
        runs = h.controller._agent_bridge.runs_db.list_runs(conversation_id)
        errors = [
            step
            for run in runs
            for step in run["steps"]
            if continuation_reasons(step.get("field_states") or {})
        ]
        assert [continuation_reasons(step["field_states"]) for step in errors] == [
            {
                "continuation_error": "final_persist_failed",
                "continuation_store_failure": "destination_mismatch",
            }
        ]
        assert not [
            step
            for run in runs
            for step in run["steps"]
            if step.get("summary") == "Reasoning not preserved for replay"
        ]

        # Send stays available: the next turn reaches the provider and
        # completes, with no reasoning to replay from the failed reply.
        await _send(h, "And again")
        second = await _observed(h, saved.id)
        assert len(h.calls) == 2
        assert second["assistant_texts"] == [_reply(1), _reply(2)]
        assert second["assistant_states"] == ["failed", "complete"]
        assert second["run_status"] is ConsoleRunStatus.COMPLETED
        assert second["dispatch_recovery"] is None
        assert _replayed(h.calls[1].get("provider_continuations") or ()) == []
        # A failed reply is not sent as history either: only the two turns.
        assert [
            (row.get("role"), row.get("content"))
            for row in h.calls[1]["messages_payload"]
            if row.get("role") != "system"
        ] == [("user", "Say hello"), ("user", "And again")]


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
                return TRACE_BLOCKED in screen or replies == [
                    _reply(call) for call in range(1, turn + 1)
                ]

            await _until(
                h.pilot, settled, timeout=30.0, what=f"turn {turn} did not settle"
            )
            assert TRACE_BLOCKED not in " ".join(_frame_rows(h.host)), (
                f"turn {turn} hit the trace pause"
            )
            view = await _observed(h, saved.id)
            assert view["blocked_copy"] == [], f"turn {turn}"
            assert view["assistant_texts"] == [
                _reply(call) for call in range(1, turn + 1)
            ], f"turn {turn}"
            assert view["run_status"] is ConsoleRunStatus.COMPLETED, f"turn {turn}"
            assert view["dispatch_recovery"] is None, f"turn {turn}"
        assert len(h.calls) == SAVED_TURNS

        # Each call replays every earlier reply's own reasoning, oldest first.
        reasoning_model = provider == "moonshot" and (
            moonshot_model_returns_reasoning_content(model)
        )
        for index, call in enumerate(h.calls):
            assert _replayed(call.get("provider_continuations") or ()) == (
                [_kimi_checkpoint(model, earlier) for earlier in range(1, index + 1)]
                if reasoning_model
                else []
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
        conversation_id = next(
            session for session in h.store.sessions() if session.id == saved.id
        ).persisted_conversation_id
        saved_checkpoints = _saved_checkpoints(h.database, conversation_id)
        assert saved_checkpoints == (
            [_kimi_checkpoint(model, turn) for turn in range(1, SAVED_TURNS + 1)]
            if reasoning_model
            else []
        )
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
            # Every replayed checkpoint, in order, with its own reply and
            # reasoning: a misordered, duplicated or reattached one fails.
            assert _traced(request.get("provider_continuations") or ()) == (
                _replayed(call.get("provider_continuations") or ())
            ), f"turn {index + 1}"
            # ... and in each slot the checkpoint saved with that reply: the
            # ledger stores a checkpoint as an artifact, so its position must
            # name the reply that owns it (turn N carries replies 1..N-1).
            assert _traced(request.get("provider_continuations") or ()) == (
                saved_checkpoints[:index]
            ), f"turn {index + 1}"

        # A relaunch reads exactly one row per reply, in one unbroken chain.
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
            ] == [(_reply(call), "complete") for call in range(1, SAVED_TURNS + 1)]
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
                lambda turn=turn: [m.content for m in replies()]
                == [_reply(call) for call in range(1, turn + 1)],
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
            # One row per provider call: each holds exactly that call's reply.
            assert sorted(
                (row["content"], row["assistant_generation_state"]) for row in rows
            ) == [(_reply(call), "complete") for call in range(1, 6)]
            reasoning_model = provider == "moonshot"
            assert [
                row["provider_continuation_json"] is not None for row in rows
            ] == [reasoning_model] * 5
            if reasoning_model:
                # ... and that call's own checkpoint, never a sibling's.
                for row in rows:
                    call = int(row["content"].rsplit("#", 1)[1])
                    assert _replayed(
                        (
                            parse_provider_continuation_json(
                                row["provider_continuation_json"]
                            ),
                        )
                    ) == [_kimi_checkpoint(model, call)], row["content"]
            assert (
                reopened.get_connection()
                .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
                .fetchone()[0]
                == 0
            )
        finally:
            reopened.close()
