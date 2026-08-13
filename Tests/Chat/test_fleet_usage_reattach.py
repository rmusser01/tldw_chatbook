"""PR 3a-2 Task 3 (tasks 15660 + 15667): fold a survivor's spend back onto
its own message when the last fleet child settles.

The gap (audit F3, made observable-not-fixed in PR 3a-1 Task 6b): the agent
path attaches usage exactly ONCE, the instant the turn finalizes. A fleet
child that outlives its turn keeps billing into the SAME
``ConsoleProviderStreamSignals``; that spend reached only the cost chip's
"Sub-agents: N tok (not priced)" line -- never the assistant message's own
usage row, never a conversation export, and it died with the controller.

The fix consumes Task 2's ``FleetDrained`` fan-out: the controller records,
per originating assistant message, the turn's signals + resolution + the
partial flag its last attach used (only while the turn still has unsettled
children -- ``ConsoleAgentBridge.has_unsettled_children``); the drain
consumer hops from the child's thread to the loop ``_attach_stream_usage``
normally runs on and re-attaches (recompute-all + REPLACE, the idempotence
the 6b pin certifies), then syncs the session watch so
``unattributed_fleet_tokens`` falls to zero.

Threading contract under test: the fold's store write happens on the loop
captured at watch time (the app loop in production -- which OUTLIVES the
Console screen), never on the child's thread; with no loop ever captured
(sync tests) the consumer runs inline.

Durability contract under test: ``set_message_usage`` on an already-
terminal message flushes ``usage_json`` through the persistence adapter's
version-neutral write, and that chain still lands AFTER
``controller.shutdown()`` (the ``ChatScreen.on_unmount`` sequence) because
nothing in teardown closes the ChaChaNotes DB or the app loop -- Task 1
A2/A5 established the child thread and its DB writes survive teardown; this
file proves the usage flush does too. What nothing can recover: spend not
yet folded when the PROCESS exits -- it is durable nowhere (``agent_runs``
rows carry no token data), which is why there is deliberately NO
mount-time usage reconcile.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _FleetTwoChildGateway,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    FleetDrained,
    SettledChild,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


async def _settle(predicate, seconds: float = 5.0) -> bool:
    """Yield the loop until ``predicate()`` is true (the drain consumer's
    fold arrives via ``call_soon_threadsafe`` and needs the loop to run)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


class _FakeFleetBridge:
    """Registration + unsettled-check seams only -- no real fleet.

    Used by the controller-level tests that deliver ``FleetDrained`` events
    directly; the full-path tests use a real ``ConsoleAgentBridge``.
    """

    def __init__(self, unsettled: bool = True):
        self.registered: dict[str, object] = {}
        self.unsettled = unsettled

    def on_fleet_drained(self, name, consumer):
        self.registered[name] = consumer

    def has_unsettled_children(self, conversation_id):
        return self.unsettled


def _turn_signals(prompt=100, completion=20) -> ConsoleProviderStreamSignals:
    signals = ConsoleProviderStreamSignals()
    signals.record_usage_payload(
        {"prompt_tokens": prompt, "completion_tokens": completion}
    )
    signals.close_usage_call()
    return signals


def _resolution() -> SimpleNamespace:
    return SimpleNamespace(provider="openai", model="gpt-4o")


def _drain_event(session_id, assistant_message_id, *, status="done", run_id=None):
    return FleetDrained(
        conversation_id="conv-reattach",
        children=(
            SettledChild(
                run_id=run_id,
                status=status,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
            ),
        ),
    )


async def _finalize(controller, aid, session_id, signals, resolution, outcome=None):
    """The production turn-end pair: ONE attach + the post-turn watch."""
    if outcome is None:
        outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")
    await controller._finalize_agent_reply(
        aid,
        session_id,
        outcome,
        variant_mode=False,
        stream_signals=signals,
        resolution=resolution,
    )


# ---------------------------------------------------------------------------
# The gap, red-first: full production path, real bridge, real drain.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_survivor_spend_folds_into_the_message_row_when_the_last_child_settles(
    tmp_path,
):
    """15660 AC#1 + AC#4: after the last child settles, the originating
    assistant message's OWN usage row includes the survivor's post-turn
    spend, the chip's unattributed line reads zero, and the store write ran
    on the loop the turn-end attach used -- never on the child's thread."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn final"],
        ],
        needed=1,
    )
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    resolution = _resolution()
    signals = _turn_signals()
    attach_threads: list[str] = []
    real_set_usage = store.set_message_usage

    def recording_set_usage(message_id, usage):
        attach_threads.append(threading.current_thread().name)
        return real_set_usage(message_id, usage)

    store.set_message_usage = recording_set_usage
    try:
        outcome = _run(
            bridge,
            store,
            session,
            aid,
            conversation_id=session.id,
            resolution=resolution,
            provider_stream_signals=signals,
        )
        assert outcome.status == "done"
        assert gateway.entered_event.wait(5), "the child never started"
        await _finalize(controller, aid, session.id, signals, resolution, outcome)
        assert store.get_message(aid).usage.total_tokens == 120
        assert controller.unattributed_fleet_tokens(session.id) == 0

        # The turn is over. The SURVIVOR makes one more provider call --
        # real money, invisible to the message row before this task.
        signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
        signals.close_usage_call()
        assert store.get_message(aid).usage.total_tokens == 120
        assert controller.unattributed_fleet_tokens(session.id) == 45
    finally:
        gate.set()
    _join_fleet_threads()

    folded = await _settle(
        lambda: (store.get_message(aid).usage or ProviderUsage()).total_tokens == 165
    )
    assert folded, (
        "the last child settled and its spend never folded into the "
        "message row: usage is "
        f"{store.get_message(aid).usage!r}, expected total 165"
    )
    assert controller.unattributed_fleet_tokens(session.id) == 0, (
        "after the fold the chip's unattributed line must fall to zero"
    )
    loop_thread = threading.current_thread().name
    post_turn_attaches = attach_threads[1:]
    assert post_turn_attaches and all(
        name == loop_thread for name in post_turn_attaches
    ), (
        "the fold's store write must hop to the loop the turn-end attach "
        f"used, never run on the child's thread: {attach_threads}"
    )
    assert controller._fleet_usage_reattach_sources == {}, (
        "a folded turn's source must be popped -- nothing can bill into "
        "its signals again after the drain"
    )


@pytest.mark.asyncio
async def test_the_fold_lands_durably_when_the_child_settles_after_console_teardown(
    tmp_path,
):
    """15667 AC#3, the hard half: the user closes Console (the exact
    ``on_unmount`` sequence: ``busy_fleet_session_count`` then
    ``shutdown()``) while a survivor is still running; the child settles
    afterwards and the fold still lands in the persisted conversation --
    read back through a FRESH DB handle, not the old store's memory."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session()
        store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="hi", persist=True
        )
        # `persist=persistence is not None` is the production placeholder
        # append (`submit_draft`'s assistant placeholder).
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        aid = assistant.id
        gate = threading.Event()
        gateway = _FleetTwoChildGateway(
            parent_script=[
                [_fence("spawn_subagent", {"task": "long job"})],
                ["turn final"],
            ],
            child_result=["child answer"],
            gate=gate,
            needed=1,
        )
        runs_db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db, store=store, provider_gateway=gateway
        )
        controller = ConsoleChatController(
            store=store, provider_gateway=object(), agent_bridge=bridge
        )
        resolution = _resolution()
        signals = _turn_signals()
        conversation_id = controller._agent_conversation_id(session.id)
        try:
            outcome = _run(
                bridge,
                store,
                session,
                aid,
                conversation_id=conversation_id,
                resolution=resolution,
                provider_stream_signals=signals,
            )
            assert outcome.status == "done"
            assert gateway.entered_event.wait(5), "the child never started"
            await _finalize(
                controller, aid, session.id, signals, resolution, outcome
            )
            persisted_id = store.get_message(aid).persisted_message_id
            assert persisted_id is not None, (
                "precondition: the turn's terminal mark persisted the reply"
            )

            # The exact ChatScreen.on_unmount sequence, controller level
            # (Task 1's harness): snapshot the count, then shut down.
            assert controller.busy_fleet_session_count() == 1
            await controller.shutdown()

            # The screen is gone. The survivor bills one more call.
            signals.record_usage_payload(
                {"prompt_tokens": 40, "completion_tokens": 5}
            )
            signals.close_usage_call()
        finally:
            gate.set()
        _join_fleet_threads()

        folded = await _settle(
            lambda: (
                store.get_message(aid).usage or ProviderUsage()
            ).total_tokens
            == 165
        )
        assert folded, (
            "the post-teardown fold never ran: usage is "
            f"{store.get_message(aid).usage!r}"
        )
        fresh = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "verifier")
        try:
            row = fresh.get_message_by_id(persisted_id)
            usage = ProviderUsage.from_json(row["usage_json"])
            assert usage is not None and usage.total_tokens == 165, (
                "the fold must be DURABLE -- a fresh handle on the same "
                f"file must read it back: {row.get('usage_json')!r}"
            )
        finally:
            fresh.close()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_conversation_export_includes_survivor_spend_after_the_fold(
    tmp_path,
):
    """15660 AC#3 / 15667 AC#1: once the re-attach has run, the JSON
    conversation export carries the message's usage -- survivor spend
    included -- read from the persisted row, not from any live object."""
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        export_conversation_to_json,
    )

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session()
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="hi", persist=True
        )
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        aid = assistant.id
        # Stream real content in (as the bridge does): an EMPTY assistant
        # row's persistence is deferred and would never reach the export.
        store.append_stream_chunk(aid, "the answer")
        controller = ConsoleChatController(
            store=store, provider_gateway=object(), agent_bridge=_FakeFleetBridge()
        )
        resolution = _resolution()
        signals = _turn_signals()
        await _finalize(controller, aid, session.id, signals, resolution)
        assert store.get_message(aid).persisted_message_id is not None, (
            "precondition: the terminal mark persisted the reply"
        )
        signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
        signals.close_usage_call()

        controller._on_fleet_drained_reattach_usage(
            _drain_event(session.id, aid)
        )
        folded = await _settle(
            lambda: (
                store.get_message(aid).usage or ProviderUsage()
            ).total_tokens
            == 165
        )
        assert folded, "precondition: the fold itself must have landed"

        raw = export_conversation_to_json(db, conversation_id)
        assert raw is not None
        payload = json.loads(raw)
        exported_usages = [
            ProviderUsage.from_json(json.dumps(message["usage"]))
            for message in payload["messages"]
            if "usage" in message
        ]
        assert [u.total_tokens for u in exported_usages if u is not None] == [
            165
        ], (
            "the export must include the folded usage row (survivor spend "
            f"included): {payload['messages']}"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Controller-level contracts of the new consumer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_earlier_turns_survivor_folds_onto_its_own_message_after_a_later_turn():
    """A later turn in the same session REPLACES the per-session watch; the
    earlier turn's survivor must still fold onto ITS OWN message (the
    message-keyed source map, not the session watch, is the re-attach
    source). The settled child is ``cancelled`` with ``run_id=None`` --
    a cancelled child's partial spend is still real spend, and a child
    that died before ``create_run`` must be tolerated."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=_FakeFleetBridge()
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    first = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    resolution = _resolution()
    signals_one = _turn_signals()
    await _finalize(controller, first.id, session.id, signals_one, resolution)

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="more")
    second = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    signals_two = _turn_signals(prompt=10, completion=2)
    await _finalize(controller, second.id, session.id, signals_two, resolution)

    # Turn 1's survivor bills after turn 2 replaced the session watch.
    signals_one.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
    signals_one.close_usage_call()

    controller._on_fleet_drained_reattach_usage(
        _drain_event(session.id, first.id, status="cancelled", run_id=None)
    )
    folded = await _settle(
        lambda: (store.get_message(first.id).usage or ProviderUsage()).total_tokens
        == 165
    )
    assert folded, (
        "the earlier turn's survivor spend must fold onto its OWN message "
        f"even after a later turn: {store.get_message(first.id).usage!r}"
    )
    assert store.get_message(second.id).usage.total_tokens == 12, (
        "the later turn's message must be untouched by the earlier fold"
    )


@pytest.mark.asyncio
async def test_re_attaching_twice_yields_the_same_stored_total():
    """15660 AC#2, end to end: delivering the same drain twice leaves the
    stored usage byte-identical (the first fold replaced; the second finds
    the source popped and is a no-op)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=_FakeFleetBridge()
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    resolution = _resolution()
    signals = _turn_signals()
    await _finalize(controller, placeholder.id, session.id, signals, resolution)
    signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
    signals.close_usage_call()

    event = _drain_event(session.id, placeholder.id)
    controller._on_fleet_drained_reattach_usage(event)
    folded = await _settle(
        lambda: (
            store.get_message(placeholder.id).usage or ProviderUsage()
        ).total_tokens
        == 165
    )
    assert folded
    first_json = store.get_message(placeholder.id).usage.to_json()

    controller._on_fleet_drained_reattach_usage(event)
    await _settle(lambda: False, seconds=0.2)  # give a (wrong) second fold time
    assert store.get_message(placeholder.id).usage.to_json() == first_json, (
        "re-attaching twice must yield the SAME stored record"
    )


@pytest.mark.asyncio
async def test_a_drain_for_an_unwatched_turn_is_a_harmless_no_op():
    """The drain also fires for WITHIN-turn children (before the turn's
    finalize has watched anything): the consumer must do nothing -- even
    when the SESSION has a watch from some other turn (folding that
    other turn's signals onto this message would be wrong-turn
    attribution) -- and the turn's own attach then covers everything
    billed so far."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=_FakeFleetBridge()
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    resolution = _resolution()
    # An EARLIER turn already finalized: the session watch exists and
    # points at that turn's signals.
    earlier = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    await _finalize(
        controller, earlier.id, session.id, _turn_signals(prompt=7, completion=1),
        resolution,
    )

    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    signals = _turn_signals()
    # A within-turn child's spend is already in the signals object...
    signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
    signals.close_usage_call()

    # ...when the drain fires BEFORE the finalize. This turn is unwatched.
    controller._on_fleet_drained_reattach_usage(
        _drain_event(session.id, placeholder.id)
    )
    await _settle(lambda: False, seconds=0.2)  # give a (wrong) fold time
    assert store.get_message(placeholder.id).usage is None, (
        "an unwatched drain must attach NOTHING -- not even the session "
        "watch's (other turn's) signals"
    )

    # The turn's own finalize then bills the whole thing, child included.
    await _finalize(controller, placeholder.id, session.id, signals, resolution)
    assert store.get_message(placeholder.id).usage.total_tokens == 165


@pytest.mark.asyncio
async def test_the_fold_preserves_the_turns_partial_flag():
    """A Stop ends the turn (attach ``partial=True``, watch set -- the
    cancel-path pair); the survivor's later fold must keep the stored
    record marked partial, not silently launder it to complete."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=_FakeFleetBridge()
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    resolution = _resolution()
    signals = _turn_signals()
    # The cancel path's exact pair (production order): partial attach,
    # then the watch with the same flag.
    controller._attach_stream_usage(
        placeholder.id, signals, resolution, partial=True
    )
    controller._watch_post_turn_usage(
        session.id,
        signals,
        resolution,
        assistant_message_id=placeholder.id,
        partial=True,
    )
    assert store.get_message(placeholder.id).usage.partial is True

    signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
    signals.close_usage_call()
    controller._on_fleet_drained_reattach_usage(
        _drain_event(session.id, placeholder.id)
    )
    folded = await _settle(
        lambda: (
            store.get_message(placeholder.id).usage or ProviderUsage()
        ).total_tokens
        == 165
    )
    assert folded, f"fold never landed: {store.get_message(placeholder.id).usage!r}"
    usage = store.get_message(placeholder.id).usage
    assert usage.partial is True, (
        "the fold must reuse the flag the turn's own attach used -- a "
        "stopped turn's record stays partial"
    )


@pytest.mark.asyncio
async def test_source_map_records_only_turns_that_still_owe_a_drain():
    """Memory hygiene: a turn whose conversation has no unsettled children
    at watch time records NO re-attach source (no drain will ever pop it);
    a turn that does records one, and the fold pops it."""
    store = ConsoleChatStore()
    bridge = _FakeFleetBridge(unsettled=False)
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    childless = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    resolution = _resolution()
    await _finalize(
        controller, childless.id, session.id, _turn_signals(), resolution
    )
    assert controller._fleet_usage_reattach_sources == {}, (
        "a turn owing no drain must not retain its signals object"
    )

    bridge.unsettled = True
    survivor_turn = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    signals = _turn_signals()
    await _finalize(controller, survivor_turn.id, session.id, signals, resolution)
    assert set(controller._fleet_usage_reattach_sources) == {survivor_turn.id}

    controller._on_fleet_drained_reattach_usage(
        _drain_event(session.id, survivor_turn.id)
    )
    await _settle(
        lambda: controller._fleet_usage_reattach_sources == {}, seconds=2.0
    )
    assert controller._fleet_usage_reattach_sources == {}


def test_has_unsettled_children_reads_the_drain_paired_counter(tmp_path):
    """The gate's bridge seam must answer "is a drain still owed?", which
    is the UNSETTLED counter -- in the scope-exit -> settle-hook window the
    live count already reads 0 while the drain has not fired (Task 1's
    pinned ordering), and a gate reading the live counter there would drop
    the re-attach source for a turn finalizing in that window."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=None
    )
    assert bridge.has_unsettled_children("conv-x") is False
    with bridge._change_window_lock:
        bridge._live_child_counts["conv-x"] = 0
        bridge._unsettled_child_counts["conv-x"] = 1
    assert bridge.has_unsettled_children("conv-x") is True, (
        "live already 0, drain still owed: the gate must say True here"
    )


def test_the_consumer_is_registered_at_construction_and_on_runtime_refresh():
    """The wiring the two-turn fan-out pin demands: registration next to
    bridge attachment (constructor + ``update_agent_runtime``), never from
    ``run_reply``; a bridge without the seam (older fakes) is tolerated."""
    store = ConsoleChatStore()
    bridge = _FakeFleetBridge()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    # PR3a-2 Task 5 widened the exact-list pin: the auto-wake consumer
    # registers at the same two call sites by the same rule, so the pin
    # is now "usage-reattach present, registered before fleet-wake"
    # (mark/usage effects land before a wake attempt reads them).
    assert list(bridge.registered) == ["usage-reattach", "fleet-wake"]
    assert (
        bridge.registered["usage-reattach"]
        == controller._on_fleet_drained_reattach_usage
    )

    refreshed = _FakeFleetBridge()
    controller.update_agent_runtime(enabled=True, bridge=refreshed)
    assert list(refreshed.registered) == ["usage-reattach", "fleet-wake"]

    # No bridge / a bridge without the seam: constructor must not raise.
    ConsoleChatController(store=store, provider_gateway=object(), agent_bridge=None)
    ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=object()
    )
