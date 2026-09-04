"""Viewless hook defaults for the app-owned Console runtime (task-15860, Task 4).

The runtime now outlives the screen, so "no view attached" is a real,
supported state rather than a transient. Every slot in
``CONSOLE_VIEW_HOOK_SLOTS`` therefore needs a value that is *semantically
correct for a runtime with no UI* -- not merely a value that does not
crash. Task 0's P3 measured the failure mode this closes: with the slots
still bound to a DEAD ``ChatScreen`` nothing raised, and a silent wrong
answer is worse than a raise.

The two safety-critical ones, each pinned here by its OBSERVABLE
CONSEQUENCE (never by asserting on the callable):

* ``wake_conversation_in_view`` -- viewless means **not in view**, so a
  wake delivered with no Console attached KEEPS the ``FLEET_UNSEEN`` ◈
  mark (task-15971's whole point: the user must learn a wake happened
  off-view). The lifetime landing deliberately left this at ``None``,
  whose read site (``_conversation_in_view``) treats "unwired" as
  IN-VIEW and clears the mark. That is the red below.
* ``wake_user_priority_probe`` -- viewless means **no user claim**: there
  is no composer, so no user can be mid-thought and no wake may be
  deferred behind one.

The remaining slots are covered by consequence too: a full viewless turn
runs end to end, a terminal run state reaches nobody, a skill confirm
fails closed immediately instead of blocking for its timeout, and an MCP
approval round armed with no view is still registered and still carries
its payload, so the next mount can claim it (plan Task 5 owns the
surfacing policy; what must hold HERE is that the viewless default does
not LOSE the round).

Rig note: the viewless state is produced through the production
``ConsoleRuntime.detach_view`` seam. ``leave_console`` (the unmount path)
calls that same ``detach_view`` and then additionally sets
``_shutdown_requested``, which is what still refuses a real headless wake
today -- that gate is the *wake-fires-headless* slice and is deliberately
untouched here.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _quiet,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_console_runtime_lifetime import _View, _pending_call
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)


def _marked(app, conversation_id) -> bool:
    return app.conversation_local_marks_service.has_mark(
        conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
    )


def _mounted_view(**overrides) -> _View:
    """A view whose hooks answer the way a MOUNTED Console's would.

    Every hook this file cares about is explicit, so a test can say which
    answer it is removing when it detaches.
    """
    hooks: dict = {
        "wake_conversation_in_view": lambda conversation_id, session_id: True,
        "wake_user_priority_probe": lambda session_id: False,
        "delivery_ui_hook": lambda session_id: None,
    }
    hooks.update(overrides)
    return _View(hooks)


def _runtime_for(rig) -> ConsoleRuntime:
    """An app-owned runtime holding the rig's already-built objects."""
    _chacha, app, _runs_db, store, _session, _gateway, _bridge, controller = rig
    runtime = ConsoleRuntime(app=app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    return runtime


async def _deliver_one(runs_db, app, session, gateway, controller):
    """Drive one survivor settle through to an accepted, stamped delivery.

    Same shape as ``test_console_fleet_wake_view_mark._deliver_one``: the
    production chain (``on_fleet_drained`` -> ``_attempt`` -> ``_deliver``),
    never a hand-called private.
    """
    _parent, run_id = _terminal_subagent_run(runs_db, session.id)
    app.conversation_local_marks_service.set_mark(
        session.id, ConversationLocalMarksService.FLEET_UNSEEN
    )
    wake = controller.fleet_wake
    wake.on_fleet_drained(_drain(session.id, _survivor(run_id, session_id=session.id)))
    assert await _settle(lambda: gateway.payloads), "the wake never delivered"
    assert await _settle(lambda: not wake.has_pending(session.id))
    assert runs_db.get_run(run_id).get("wake_delivered_at"), (
        "harness precondition: the delivery must commit (stamped ledger)"
    )


# ---------------------------------------------------------------------------
# Safety red 1 -- the ◈ mark survives a wake delivered with no view.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_wake_delivered_with_no_view_keeps_the_unseen_mark(tmp_path):
    """RED before Task 4: the viewless default cleared the mark.

    ``detach_view`` restored ``wake_conversation_in_view`` to ``None``, and
    ``_conversation_in_view`` reads an unwired probe as IN VIEW -- so a
    delivery nobody could possibly have watched committed as "seen" and
    the ◈ badge was cleared. The user was told nothing.

    Observable consequence, not the callable: the FLEET_UNSEEN mark must
    SURVIVE the delivery commit.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, _store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        view = _mounted_view()
        runtime.attach_view(view)
        assert runtime.detach_view(view) is True, "the rig never went viewless"

        await _deliver_one(runs_db, app, session, gateway, controller)

        assert _marked(app, session.id), (
            "a wake delivered with NO Console attached cleared the ◈ mark: "
            "the user has no way to learn the supervisor turn ever ran"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_wake_delivered_to_the_attached_view_still_clears_the_mark(
    tmp_path,
):
    """The preserved side: an ATTACHED view that reports in-view still clears.

    Without this, "keep the mark" could be implemented by simply ignoring
    the probe, and the mark would go stale on every watched delivery.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, _store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        runtime.attach_view(_mounted_view())

        await _deliver_one(runs_db, app, session, gateway, controller)

        assert not _marked(app, session.id), (
            "a wake the attached view reported as in-view must still clear"
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# Safety red 2 -- no view means no user claim, so no deferral.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_wake_is_not_deferred_by_a_user_claim_once_the_view_is_gone(
    tmp_path,
):
    """Both directions of user-wins-ties across the view seam.

    While a view is attached and its probe claims the user is mid-thought,
    the wake defers (that is the shipped tie-break). The moment the view
    detaches there is no composer, so nobody can be mid-thought: the wake
    must fire on the next retry rather than defer forever behind a claim
    made by a screen that no longer exists.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, _store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        view = _mounted_view(wake_user_priority_probe=lambda session_id: True)
        runtime.attach_view(view)

        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(lambda: gateway.payloads), (
            "harness precondition: an ATTACHED view's user claim must defer "
            "the wake, or this test cannot say anything about removing it"
        )

        assert runtime.detach_view(view) is True
        wake.retry_soon()

        assert await _settle(lambda: gateway.payloads), (
            "the wake stayed deferred behind a user claim made by a view "
            "that is gone -- with no composer there is no user to lose to"
        )
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# delivery_ui_hook -- inert while detached, RE-ARMED by the next attach.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_delivery_started_with_no_view_re_arms_at_the_next_attach(
    tmp_path,
):
    """The 4-minute mid-delivery freeze, pinned.

    A wake that starts while nothing is attached has no repaint target --
    correct, and inert. But the user who opens Console DURING that turn
    gets a live delivery and (before this) no transcript poll: PR 3a-2
    Task 7 measured that live as a 4+ minute frozen Console. ``attach_view``
    must therefore fire the newly-bound hook when a wake is still
    delivering.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, runs_db, _store, session, gateway, _bridge, controller = rig
    try:
        gateway.stream_gate = asyncio.Event()
        runtime = _runtime_for(rig)
        stale: list[str] = []
        view_one = _mounted_view(delivery_ui_hook=stale.append)
        runtime.attach_view(view_one)
        assert runtime.detach_view(view_one) is True

        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        wake = controller.fleet_wake
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: gateway.payloads), "the wake never started"
        assert wake.delivering_conversation_id() == session.id, (
            "harness precondition: the turn must still be delivering"
        )
        assert stale == [], (
            "a detached view's repaint hook was fired for a delivery it "
            "cannot possibly repaint"
        )

        armed: list[str] = []
        runtime.attach_view(_mounted_view(delivery_ui_hook=armed.append))

        assert armed == [session.id], (
            "opening Console during a wake delivery left the transcript "
            f"poll unarmed (the live 4-minute freeze); got {armed}"
        )
    finally:
        gateway.stream_gate.set()
        await _settle(lambda: wake.delivering_conversation_id() is None)
        chacha.close()


@pytest.mark.asyncio
async def test_attaching_with_no_delivery_in_flight_arms_nothing(tmp_path):
    """The re-arm is conditional: no delivery, no spurious poll.

    A poll armed with nothing to repaint is the recurring-idle-repaint
    regression 15664 AC#2 forbids.
    """
    rig = _controller_rig(tmp_path)
    chacha, _app, _runs_db, _store, _session, _gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        armed: list[str] = []
        assert controller.fleet_wake.delivering_conversation_id() is None
        runtime.attach_view(_mounted_view(delivery_ui_hook=armed.append))
        assert armed == []
    finally:
        chacha.close()


# ---------------------------------------------------------------------------
# The remaining slots, by consequence.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_whole_turn_runs_with_no_view_attached(tmp_path):
    """Every constructor-supplied slot's viewless default, executed.

    ``_chat_dictionary_applier``, ``_world_info_applier``,
    ``_rag_capture_provider``, ``_default_session_settings``,
    ``_library_provider_factory``, ``_global_user_display_name``,
    ``_turn_context_provider``, ``on_submission_accepted`` and
    ``prompt_history`` are all read on the send path. A viewless turn must
    run to a reply through every one of them -- in particular
    ``_global_user_display_name``, whose read site CALLS the slot with no
    ``is None`` guard, so ``None`` would be a hard failure there rather
    than an inert one.
    """
    rig = _controller_rig(tmp_path)
    chacha, _app, _runs_db, store, session, gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        view = _mounted_view()
        runtime.attach_view(view)
        assert runtime.detach_view(view) is True

        result = await controller.submit_draft("hello", session_id=session.id)

        assert getattr(result, "accepted", False), (
            f"a viewless turn was refused: {result}"
        )
        assert gateway.payloads, "the viewless turn never reached the provider"
        replies = [
            message
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert replies, "the viewless turn produced no assistant reply"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_viewless_turn_calls_none_of_the_departed_views_hooks(tmp_path):
    """Task 0's P3 finding, as a consequence test.

    P3 measured hook slots still bound to a DEAD ``ChatScreen`` after a
    real unmount, none of them raising. The property that closes it is not
    "the slot holds None" (that is asserting on the callable) but "a turn
    running after the view is gone calls NOTHING the view supplied".

    Run twice on the same controller so the assertion cannot be vacuous:
    turn one, attached, must fire the recorders; turn two, detached, must
    fire none of them.
    """
    rig = _controller_rig(tmp_path)
    chacha, _app, _runs_db, _store, session, _gateway, _bridge, controller = rig
    try:
        calls: list[str] = []

        class _RecordingHistory:
            async def append(self, text: str) -> None:
                calls.append(f"prompt_history:{text}")

        runtime = _runtime_for(rig)
        view = _mounted_view(
            on_submission_accepted=lambda: calls.append("on_submission_accepted"),
            prompt_history=_RecordingHistory(),
        )
        runtime.attach_view(view)

        first = await controller.submit_draft("attached", session_id=session.id)
        assert getattr(first, "accepted", False), first
        assert calls, (
            "harness precondition: an ATTACHED view's hooks must fire, or "
            "the detached assertion below proves nothing"
        )

        assert runtime.detach_view(view) is True
        calls.clear()

        second = await controller.submit_draft("detached", session_id=session.id)
        assert getattr(second, "accepted", False), second

        assert calls == [], (
            "a turn running with no view attached called into the view that "
            f"is gone: {calls}"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_the_display_name_slot_is_never_cleared_to_none(tmp_path):
    """`None` is UNSAFE for this one slot, and the pin says why.

    ``_presentation_context_for`` calls ``self._global_user_display_name()``
    unconditionally (the constructor's ``... or (lambda: "User")`` is the
    only guard, and it runs once, at construction). Cleared to ``None`` the
    call raises ``TypeError`` on every read; the broad ``except`` around it
    turns that into a per-read warning rather than a crash, which is
    exactly the kind of silently-degraded default this task exists to
    remove.
    """
    rig = _controller_rig(tmp_path)
    chacha, _app, _runs_db, _store, session, _gateway, _bridge, controller = rig
    try:
        runtime = _runtime_for(rig)
        view = _mounted_view(_global_user_display_name=lambda: "Ada")
        runtime.attach_view(view)
        assert controller._presentation_context_for(session.id).user_name == "Ada"

        assert runtime.detach_view(view) is True

        assert callable(controller._global_user_display_name), (
            "the display-name slot was cleared to a non-callable"
        )
        assert controller._presentation_context_for(session.id).user_name == "User"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_skill_install_confirm_with_no_view_fails_closed_at_once(
    tmp_path,
):
    """The two skill-confirm slots: deny immediately, never hang.

    ``None`` is the CORRECT viewless value here and the read site says so:
    "no UI bridge wired means the marshal below is a no-op and nothing can
    ever set the Event -- fail closed immediately instead of blocking for
    the full timeout". Pinned by consequence: denied, and denied fast.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, _runs_db, _store, session, _gateway, _bridge, controller = rig
    try:
        controller.app = app
        runtime = _runtime_for(rig)
        view = _mounted_view(
            set_pending_skill_install=lambda payload: None,
            set_pending_skill_script=lambda payload: None,
        )
        runtime.attach_view(view)
        assert runtime.detach_view(view) is True
        controller.mcp_approval_timeout_seconds = lambda: 60.0

        verdicts: dict[str, object] = {}

        def _ask() -> None:
            verdicts["install"] = controller.request_skill_install_confirm(
                "https://example.invalid/skill.zip", session_id=session.id
            )
            verdicts["script"] = controller.request_skill_script_confirm(
                {"skill": "demo", "mechanism": "shell", "args": []},
                session_id=session.id,
            )

        worker = threading.Thread(target=_ask, daemon=True)
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive(), (
            "a viewless skill confirm blocked instead of failing closed"
        )
        assert verdicts["install"] is False
        assert verdicts["script"] == {"allow": False, "remember": False}
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_an_approval_round_armed_with_no_view_is_not_lost(tmp_path):
    """The approval slots must not SWALLOW a card.

    ``set_pending_approval``/``park_pending_approval`` are UI bridges and
    their viewless value is inert -- but the round itself is registered
    and its payload retained unconditionally, before either hook is
    consulted. So a round armed headless is still there to be claimed at
    the next mount. (Plan Task 5 owns the surfacing policy and the clock;
    what must hold here is only that nothing is lost.)
    """
    rig = _controller_rig(tmp_path)
    chacha, app, _runs_db, _store, session, _gateway, _bridge, controller = rig
    try:
        controller.app = app
        runtime = _runtime_for(rig)
        view = _mounted_view(
            set_pending_approval=lambda payload: None,
            park_pending_approval=lambda session_id: None,
        )
        runtime.attach_view(view)
        assert runtime.detach_view(view) is True
        controller.mcp_approval_timeout_seconds = lambda: 60.0

        decisions: dict[str, str] = {}
        armed = threading.Event()

        def _run_round() -> None:
            armed.set()
            decisions.update(
                controller.request_mcp_approvals(
                    [_pending_call()], session_id=session.id
                )
            )

        worker = threading.Thread(target=_run_round, daemon=True)
        worker.start()
        assert armed.wait(timeout=2), "the round never armed"
        await asyncio.sleep(0.3)

        assert controller.has_pending_approval_round(session.id), (
            "a round armed with no view is unregistered -- the next mount "
            "has nothing to claim and the card is silently swallowed"
        )
        # PR0 (task-15661): keyed by ROUND now -- this session's retained
        # payload is its FIFO head, not a `.get(session_id)`.
        parked = controller._head_round_payload(
            controller._parked_approval_payloads, session.id
        )
        assert parked is not None and parked.get("calls"), (
            "the round's payload was not retained, so a mount could not "
            "re-derive the card even knowing the round exists"
        )

        # Unblock: ending the visit is the shipped resolution for an
        # undecided round (AC#2), and it must still deny.
        await asyncio.wait_for(runtime.leave_console(), timeout=5)
        worker.join(timeout=10)
        assert decisions == {"write_file": "deny"}, decisions
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_runtime_that_never_had_a_view_answers_viewless(tmp_path):
    """A runtime can be viewless from BIRTH, not only after a detach.

    ``ensure_chat_controller`` builds from parameters a caller supplies;
    nothing about that caller makes it an attached view. A controller
    built with no view claimed must answer the same way a detached one
    does, or the wake-at-launch case (Console never opened) inherits the
    exact silent wrong answer this task removes.
    """
    rig = _controller_rig(tmp_path)
    chacha, app, _runs_db, store, session, gateway, bridge, _controller = rig
    try:
        runtime = ConsoleRuntime(app=app)
        runtime.set_chat_store(store)
        assert runtime.view is None
        controller = runtime.ensure_chat_controller(
            store=store,
            provider_gateway=gateway,
            agent_bridge=bridge,
            agent_runtime_enabled=False,
            global_user_display_name=lambda: "Ada",
        )
        assert (
            controller.fleet_wake._conversation_in_view(session.id, session.id) is False
        ), (
            "a runtime built with no view reported the conversation as "
            "watched -- the ◈ mark would be cleared for a delivery nobody "
            "could have seen"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_every_slot_names_a_real_attribute_on_the_target_it_declares():
    """A total sweep over the enumerated list, on real objects.

    Two things a per-slot consequence test cannot catch, because both
    fail SILENTLY: a slot whose ``name`` does not exist on its target (a
    rename on the controller leaves the list writing a brand-new,
    never-read attribute) and a slot whose ``target`` kind is wrong
    (``detach_view`` then clears the wrong object entirely and the real
    one keeps a dead screen's callable). ``setattr`` succeeds in both
    cases, so only a pre-attach ``hasattr`` on freshly built objects says
    so. Also asserts every default actually lands, and that each carries
    the ``why`` Task 4 requires.
    """
    from tldw_chatbook.Chat.console_runtime import CONSOLE_VIEW_HOOK_SLOTS

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=None)
    targets = {
        "controller": controller,
        "store": store,
        "wake": controller.fleet_wake,
    }
    for slot in CONSOLE_VIEW_HOOK_SLOTS:
        target = targets[slot.target]
        assert hasattr(target, slot.name), (
            f"slot {slot.name!r} does not exist on the {slot.target} it "
            "declares -- detach writes an attribute nothing reads"
        )
        assert slot.why, f"slot {slot.name!r} has no viewless justification"

    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    view = _View({})
    runtime.attach_view(view)

    assert runtime.detach_view(view) is True

    for slot in CONSOLE_VIEW_HOOK_SLOTS:
        assert getattr(targets[slot.target], slot.name) == slot.viewless_default, (
            slot.name
        )
