"""Fleet PR 3b Task 5, Qodo #1808 finding 3: closing a session kills its fleet.

`ConsoleChatStore.close_session` is DESTRUCTIVE -- it purges every
message the session owns and drops the session outright. With Stop
decoupled from the children (this branch), a session's live decoupled
children would otherwise outlive their own conversation: the panel has
no rows left to cancel them from, wake delivery would target a dead
conversation, and its unseen-mark would leak. Navigation-away preserves
the conversation and its survivors rightly continue; DESTRUCTION must
take the fleet with it.

The seam pinned here: `ConsoleChatController.close_session` calls the
explicit whole-fleet path this branch built --
`ConsoleAgentBridge.cancel_all_subagents` -- with the conversation id
derived WHILE THE SESSION STILL EXISTS (the persisted conversation id
when set; the bridge keys everything by it). That method's own walk,
per-handle cancel reuse, approval revocation, and
cancelled-is-never-retained semantics are pinned by execution in
`Tests/Chat/test_console_agent_bridge_cancel_all.py` /
`Tests/Agents/test_fleet_stop_semantics.py`; the recording bridge here
asserts THIS layer's own seam (the call, its target id, its timing),
per the layered-guard rule.

All three were committed RED at the pre-fix branch (`cancel_all_calls
== []`: the close path made no attempt to stop the fleet -- the child
"keeps running with no cancel path", exactly the bot's framing).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Chat.test_console_chat_controller import StreamingGateway

hook_file = _hook_file


class _RecordingFleetBridge:
    """Exposes exactly the seam `close_session` must call, recording it.

    `sessions_present_at_call` captures whether the closed session was
    still in the store when the cancel fired -- the conversation-id
    derivation (`_agent_conversation_id`) reads the store, so a cancel
    issued after the purge would silently target the bare session id
    even for a persisted conversation.
    """

    def __init__(self, store: ConsoleChatStore) -> None:
        self._store = store
        self.cancel_all_calls: list[str] = []
        self.sessions_present_at_call: list[bool] = []

    def cancel_all_subagents(self, conversation_id: str) -> int:
        self.cancel_all_calls.append(conversation_id)
        self.sessions_present_at_call.append(bool(self._store.sessions()))
        return 1

    async def await_fleet_terminal(self, _conversation_id: str) -> bool:
        return True


def _controller_with_bridge(store: ConsoleChatStore):
    bridge = _RecordingFleetBridge(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=bridge,
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_agent_bridge(bridge)
    runtime.set_chat_controller(controller)
    return runtime, controller, bridge


@pytest.mark.asyncio
async def test_closing_a_session_cancels_its_fleet_through_cancel_all():
    """The active-close path: one close, one whole-fleet cancel, keyed by
    the session's own conversation id, issued while the session still
    exists in the store."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, bridge = _controller_with_bridge(store)

    await runtime.close_session(
        session.id,
        expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
    )

    assert bridge.cancel_all_calls == [session.id]
    assert bridge.sessions_present_at_call == [True], (
        "the cancel must fire while the session (and its conversation-id "
        "mapping) still exists"
    )
    assert store.sessions() == []


@pytest.mark.asyncio
async def test_closing_an_ephemeral_session_cancels_its_fleet_too():
    """The ephemeral-teardown path: a temporary session's close is the
    same destruction (nothing was ever persisted to return to), so its
    fleet dies with it identically."""
    store = ConsoleChatStore()
    session = store.create_session(title="temp", ephemeral=True)
    assert store.session_is_ephemeral(session.id)
    runtime, controller, bridge = _controller_with_bridge(store)

    await runtime.close_session(
        session.id,
        expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
    )

    assert bridge.cancel_all_calls == [session.id]
    assert store.sessions() == []


@pytest.mark.asyncio
async def test_close_cancel_targets_the_persisted_conversation_id_when_set():
    """The bridge keys fleets by the DURABLE conversation id
    (`_agent_conversation_id`: persisted id when set, else the session
    id). A close must cancel under that key, or a persisted
    conversation's fleet would be looked up under a key it never used."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "conv-durable"
    runtime, controller, bridge = _controller_with_bridge(store)

    await runtime.close_session(
        session.id,
        expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
    )

    assert bridge.cancel_all_calls == [session.id, "conv-durable"]
    assert bridge.sessions_present_at_call == [True, True]


@pytest.mark.asyncio
async def test_close_survives_a_bridge_without_the_cancel_all_seam():
    """A bare bridge double (or none at all) must never break a close --
    the fleet cancel degrades to a no-op, exactly like every other
    getattr-guarded bridge read on the controller."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=object(),  # no cancel_all_subagents at all
    )
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_agent_bridge(controller._agent_bridge)
    runtime.set_chat_controller(controller)

    assert (
        await runtime.close_session(
            session.id,
            expected_revision=controller.lifecycle_impact(
                session_id=session.id
            ).revision,
        )
        is None
    )
    assert store.sessions() == []


def _begin_fenced_close(runtime, controller, session):
    runtime._admission_fenced_sessions.add(session.id)
    ticket = controller.begin_session_close(
        session.id,
        expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
    )
    return ticket, controller._session_close_states[ticket.close_id]


@pytest.mark.parametrize(
    "field,value",
    [
        ("incarnation_id", "different-incarnation"),
        ("conversation_binding_revision", 123456),
        ("ephemeral", True),
        ("workspace_id", "different-workspace"),
    ],
)
def test_legacy_close_retains_ticket_on_subject_identity_change(field, value):
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    ticket, state = _begin_fenced_close(runtime, controller, session)
    setattr(session, field, value)
    with pytest.raises(RuntimeError):
        controller.finalize_session_close(ticket)
    assert controller._session_close_states[ticket.close_id] is state
    assert store._sessions[session.id] is session
    assert session.id in runtime._admission_fenced_sessions


def test_legacy_close_rejects_coherent_replacement_before_candidate_access():
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    engine = runtime.ensure_run_hooks()
    witness = runtime._run_hooks_construction_witness
    assert (
        type(witness) is tuple
        and witness[0] is engine
        and witness[1] is runtime._legacy_hooks_source
    )

    class Foreign:
        def __getattribute__(self, name):
            raise AssertionError(f"foreign object accessed: {name}")

        def __eq__(self, other):
            raise AssertionError("foreign equality invoked")

    foreign = Foreign()
    runtime._run_hooks_engine = foreign
    runtime._legacy_hooks_source = foreign
    runtime._hook_permissions = foreign
    ticket, state = _begin_fenced_close(runtime, controller, session)
    with pytest.raises(RuntimeError):
        controller.finalize_session_close(ticket)
    assert controller._session_close_states[ticket.close_id] is state
    assert store._sessions[session.id] is session
    engine.close()


def test_legacy_close_rejects_equal_witness_replacement_after_pin():
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    engine = runtime.ensure_run_hooks()
    witness = runtime._run_hooks_construction_witness
    ticket, state = _begin_fenced_close(runtime, controller, session)
    subject = state[4]
    controller._legacy_close_ready(ticket, state, subject)
    runtime._run_hooks_construction_witness = (*witness,)
    with pytest.raises(RuntimeError):
        controller.finalize_session_close(ticket)
    assert controller._session_close_states[ticket.close_id] is state
    assert store._sessions[session.id] is session
    engine.close()


def test_failed_construction_stays_partial_and_cannot_be_rebuilt(monkeypatch):
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)

    def fail_permissions():
        raise RuntimeError("construction failed")

    monkeypatch.setattr(runtime, "ensure_hook_permissions", fail_permissions)
    with pytest.raises(RuntimeError, match="construction failed"):
        runtime.ensure_run_hooks()
    assert runtime._run_hooks_construction_witness is True
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="incomplete"):
        runtime.ensure_run_hooks()
    ticket, state = _begin_fenced_close(runtime, controller, session)
    with pytest.raises(RuntimeError):
        controller.finalize_session_close(ticket)
    assert controller._session_close_states[ticket.close_id] is state
    assert store._sessions[session.id] is session


def test_sealed_absence_survives_original_other_chat_construction():
    store = ConsoleChatStore()
    session = store.ensure_session()
    other = store.create_session(title="other")
    runtime, controller, _ = _controller_with_bridge(store)
    ticket, state = _begin_fenced_close(runtime, controller, session)
    subject = state[4]
    controller._legacy_close_ready(ticket, state, subject)
    assert subject.mode == "absent"
    engine = runtime.ensure_run_hooks()
    assert runtime._run_hooks_construction_witness[0] is engine
    controller.finalize_session_close(ticket)
    assert session.id not in store._sessions
    assert store._sessions[other.id] is other
    engine.close()


def test_legacy_close_busy_hook_lock_is_pending_without_consuming_ticket():
    import threading

    from tldw_chatbook.Chat.console_chat_controller import _LegacyClosePending

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    ticket, state = _begin_fenced_close(runtime, controller, session)
    held, release = threading.Event(), threading.Event()

    def hold_lock():
        with runtime._run_hooks_lock:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    try:
        assert held.wait(2)
        with pytest.raises(_LegacyClosePending):
            controller.finalize_session_close(ticket)
        assert controller._session_close_states[ticket.close_id] is state
        assert store._sessions[session.id] is session
    finally:
        release.set()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize("mutation", ["registry", "store", "generation", "capture"])
def test_legacy_close_rejects_changed_originals_before_finalization(
    monkeypatch, mutation
):
    from tldw_chatbook.Agents.legacy_hook_ownership import _NativeLegacyHookSource

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    ticket, state = _begin_fenced_close(runtime, controller, session)
    controller._legacy_close_ready(ticket, state, state[4])
    if mutation == "registry":
        store._sessions[session.id] = store.create_session(title="replacement")
    elif mutation == "store":
        controller.store = ConsoleChatStore()
    elif mutation == "generation":
        controller._session_close_generations[session.id] += 1
    else:
        monkeypatch.setattr(_NativeLegacyHookSource, "capture", lambda *_: None)
    with pytest.raises(RuntimeError):
        controller.finalize_session_close(ticket)
    assert controller._session_close_states[ticket.close_id] is state
    assert session.id in runtime._admission_fenced_sessions


@pytest.mark.parametrize("failure", ["issuer", "attachment"])
def test_failed_published_construction_cannot_be_reclassified(monkeypatch, failure):
    from tldw_chatbook.Agents import legacy_hook_ownership
    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    if failure == "issuer":

        def fail_issuer(*_):
            raise RuntimeError("issuer failure")

        monkeypatch.setattr(
            legacy_hook_ownership, "_NativeLegacyHookSource", fail_issuer
        )
    else:

        def fail_attachment(self, name, value):
            if name == "_native_attached" and value is True:
                raise RuntimeError("attachment failure")
            object.__setattr__(self, name, value)

        monkeypatch.setattr(RunHooksEngine, "__setattr__", fail_attachment)
    with pytest.raises(RuntimeError):
        runtime.ensure_run_hooks()
    engine = runtime._run_hooks_engine
    assert runtime._run_hooks_construction_witness is True
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="incomplete"):
        runtime.ensure_run_hooks()
    ticket, state = _begin_fenced_close(runtime, controller, session)
    with pytest.raises(RuntimeError):
        controller.finalize_session_close(ticket)
    assert controller._session_close_states[ticket.close_id] is state
    assert store._sessions[session.id] is session
    engine.close()


def test_legacy_close_busy_issuer_condition_is_pending():
    import threading

    from tldw_chatbook.Chat.console_chat_controller import _LegacyClosePending

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    engine = runtime.ensure_run_hooks()
    issuer = runtime._legacy_hooks_source
    ticket, state = _begin_fenced_close(runtime, controller, session)
    held, release = threading.Event(), threading.Event()

    def hold_condition():
        with issuer._condition:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold_condition)
    thread.start()
    try:
        assert held.wait(2)
        with pytest.raises(_LegacyClosePending):
            controller.finalize_session_close(ticket)
        assert controller._session_close_states[ticket.close_id] is state
        assert store._sessions[session.id] is session
    finally:
        release.set()
        thread.join(5)
        engine.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("replacement", ["none", "foreign"])
def test_legacy_close_cannot_downgrade_replaced_state_to_no_subject(replacement):
    calls = []

    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("foreign close subject accessed")

        def __eq__(self, other):
            calls.append("equality")
            raise AssertionError("foreign equality invoked")

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    engine = runtime.ensure_run_hooks()
    ticket, state = _begin_fenced_close(runtime, controller, session)
    controller._session_close_states[ticket.close_id] = (
        *state[:4],
        None if replacement == "none" else Foreign(),
    )
    try:
        with pytest.raises(RuntimeError):
            controller.finalize_session_close(ticket)
        assert store._sessions[session.id] is session
        assert session.id in runtime._admission_fenced_sessions
        assert not calls
    finally:
        engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["none", "foreign"])
async def test_runtime_close_refuses_foreign_subject_without_access(
    monkeypatch, replacement
):
    import asyncio

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    runtime.ensure_run_hooks()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def held_drain(_session_id):
        entered.set()
        await release.wait()
        return False

    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("foreign close subject accessed")

    monkeypatch.setattr(runtime, "_drain_hook_review_operations", held_drain)
    revision = controller.lifecycle_impact(session_id=session.id).revision
    task = asyncio.create_task(
        runtime.close_session(session.id, expected_revision=revision)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        ticket_id, state = next(iter(controller._session_close_states.items()))
        controller._session_close_states[ticket_id] = (
            *state[:4],
            None if replacement == "none" else Foreign(),
        )
        release.set()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(task, 3)
        assert not calls
        assert store._sessions[session.id] is session
        assert session.id in runtime._admission_fenced_sessions
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.undo()
        if "ticket_id" in locals():
            controller._session_close_states[ticket_id] = state
        await runtime.dispose()


@pytest.mark.skipif(
    not native_workflow_qualified(), reason="native POSIX group custody required"
)
def test_legacy_close_rejects_coherent_original_class_replacement_before_candidate_calls():
    import sys

    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Agents.legacy_hook_ownership import _NativeLegacyHookSource
    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    original_engine = runtime.ensure_run_hooks()
    witness = runtime._run_hooks_construction_witness
    original_issuer, original_permissions = witness[1:]
    candidate_permissions = HookPermissions()
    candidate_engine = RunHooksEngine(
        candidate_permissions.targets,
        lambda: ".",
        notification_targets=candidate_permissions.notification_targets,
        launch_guard=candidate_permissions.launch_guard,
    )
    candidate_issuer = _NativeLegacyHookSource(
        runtime, candidate_engine, candidate_permissions
    )
    candidate_engine._native_source = candidate_issuer
    candidate_engine._native_attached = True
    assert type(candidate_engine) is type(original_engine)
    assert type(candidate_issuer) is type(original_issuer)
    assert type(candidate_permissions) is type(original_permissions)
    assert (
        candidate_issuer.runtime is runtime
        and candidate_issuer.engine is candidate_engine
    )
    assert candidate_issuer.permissions is candidate_permissions
    assert (
        candidate_engine._native_source is candidate_issuer
        and candidate_engine._native_attached is True
    )
    runtime._run_hooks_engine = candidate_engine
    runtime._legacy_hooks_source = candidate_issuer
    runtime._hook_permissions = candidate_permissions
    candidate_calls = []
    previous_profile = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and any(
            frame.f_locals.get("self") is candidate
            for candidate in (
                candidate_engine,
                candidate_issuer,
                candidate_permissions,
            )
        ):
            candidate_calls.append(frame.f_code.co_name)

    try:
        sys.setprofile(observe)
        try:
            ticket, state = _begin_fenced_close(runtime, controller, session)
            with pytest.raises(RuntimeError):
                controller.finalize_session_close(ticket)
        finally:
            sys.setprofile(previous_profile)
        assert not candidate_calls
        assert runtime._run_hooks_construction_witness is witness
        assert controller._session_close_states[ticket.close_id] is state
        assert store._sessions[session.id] is session
        assert session.id in runtime._admission_fenced_sessions
    finally:
        sys.setprofile(previous_profile)
        runtime._run_hooks_engine = original_engine
        runtime._legacy_hooks_source = original_issuer
        runtime._hook_permissions = original_permissions
        candidate_engine.close()
        original_engine.close()


@pytest.mark.skipif(
    not native_workflow_qualified(), reason="native POSIX group custody required"
)
def test_legacy_close_refuses_removed_original_native_issuer():
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    engine = runtime.ensure_run_hooks()
    witness = runtime._run_hooks_construction_witness
    issuer = runtime._legacy_hooks_source
    del runtime._legacy_hooks_source
    try:
        ticket, state = _begin_fenced_close(runtime, controller, session)
        with pytest.raises(RuntimeError):
            controller.finalize_session_close(ticket)
        assert runtime._run_hooks_construction_witness is witness
        assert controller._session_close_states[ticket.close_id] is state
        assert store._sessions[session.id] is session
        assert session.id in runtime._admission_fenced_sessions
    finally:
        runtime._legacy_hooks_source = issuer
        engine.close()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_genuine_unqualified_ordinary_producer_fires_and_closes_selected_chat(
    tmp_path, monkeypatch, hook_file
):
    import os
    import sys

    import toml

    from Tests.Agents.test_hook_permissions import _approve
    from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

    marker = tmp_path / "ordinary-selected-hook"
    raw = toml.loads(hook_file.read_text())
    raw["hooks"] = {
        "enabled": True,
        "hook": [
            {
                "id": "ordinary",
                "event": "PreToolUse",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                "timeout_s": 3,
            }
        ],
    }
    hook_file.write_text(toml.dumps(raw))
    store = ConsoleChatStore()
    session = store.ensure_session()
    runtime, controller, _ = _controller_with_bridge(store)
    with monkeypatch.context() as platform:
        platform.delattr(os, "killpg", raising=False)
        assert not native_workflow_qualified()
        engine = runtime.ensure_run_hooks()
    witness = runtime._run_hooks_construction_witness
    assert type(witness) is tuple and witness[0] is engine and witness[1] is None
    assert witness[2] is runtime._hook_permissions
    assert engine._native_attached is False and engine._native_source is None
    assert "_legacy_hooks_source" not in vars(runtime)
    _approve(runtime.ensure_hook_permissions())
    try:
        result = engine.fire("PreToolUse", session_id=session.id, cwd=str(tmp_path))
        assert not result.blocked and marker.exists()
        assert (
            runtime._execution_capacity is None and runtime._worktree_activity is None
        )
        revision = controller.lifecycle_impact(session_id=session.id).revision
        from tldw_chatbook.Agents.legacy_hook_ownership import _NativeLegacyHookSource

        readiness_calls = []
        previous_profile = sys.getprofile()

        def observe(frame, event, arg):
            if (
                event == "call"
                and frame.f_code is _NativeLegacyHookSource.selected_ready.__code__
            ):
                readiness_calls.append(True)

        try:
            sys.setprofile(observe)
            await runtime.close_session(session.id, expected_revision=revision)
        finally:
            sys.setprofile(previous_profile)
        assert not readiness_calls
        assert session.id not in store._sessions
        assert runtime._execution_capacity is None
        assert runtime._run_hooks_construction_witness is witness
    finally:
        await runtime.dispose()
