"""Final Console unmount and application disposal retain their cancellation fences.

TASK-15860 split visit cleanup from permanent runtime shutdown. TASK-31520
subsequently made ordinary navigation reuse and suspend the mounted Console;
that route does not call ``leave_console``. These direct runtime tests cover
actual final-unmount cleanup and permanent app disposal. Real navigation with
streams, queues and pending decisions is covered by test_console_navigation_decisions.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore as _ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_prompt_queue import QueueMutationStatus
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnCustodyRequest,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def test_runtime_composes_one_shared_canvas_owner_into_real_store(tmp_path) -> None:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "runtime-canvas.sqlite", "runtime-canvas")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=db))
    try:
        store = runtime.ensure_chat_store()
        session = store.create_session(ephemeral=True)

        assert runtime.canvas_controller is not None
        assert store.canvas_turn_controller is runtime.canvas_controller
        assert store.canvas_promotion_participant is runtime.canvas_controller
        assert session.id in runtime.canvas_controller._session_owners
    finally:
        asyncio.run(runtime.dispose())
        db.close_connection()


def test_runtime_forwards_live_session_and_branch_transitions_to_canvas_authority():
    observed: list[str | None] = []
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
    runtime._canvas_native_authority = SimpleNamespace(
        sync_live_context=observed.append
    )
    store = runtime.ensure_chat_store()

    first = store.create_session(ephemeral=True)
    root = store.append_message(
        first.id,
        role="user",
        content="root",
    )
    child = store.append_message(
        first.id,
        role="assistant",
        content="child",
    )
    observed.clear()

    store.set_active_leaf(first.id, root.id)
    second = store.create_session(ephemeral=True)
    store.switch_session(first.id)
    store.set_active_leaf(first.id, child.id)

    assert observed == [first.id, second.id, first.id, first.id]


@pytest.mark.asyncio
async def test_runtime_disposes_canvas_authority_after_revoking_gateway():
    order: list[str] = []

    class CanvasGateway:
        async def aclose(self):
            order.append("gateway")

    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
    runtime._canvas_gateway = CanvasGateway()
    runtime._canvas_native_authority = SimpleNamespace(
        dispose=lambda: order.append("authority")
    )

    await runtime.dispose()

    assert order[:2] == ["gateway", "authority"]


@pytest.mark.unit
def test_trace_rollout_modules_stay_off_the_ui_ready_import_path() -> None:
    """Metrics and write planning load only when a trace actually uses them."""

    probe = """
import asyncio
import sys
from types import SimpleNamespace

from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

class Database:
    def transaction(self):
        raise AssertionError("the lazy factory must not touch storage")

runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=Database()))
runtime._chat_store = SimpleNamespace(
    persistence=SimpleNamespace(console_trace_repository=object())
)
gateway = runtime.ensure_provider_gateway()
assert "tldw_chatbook.Chat.console_trace_metrics" not in sys.modules
assert "tldw_chatbook.Chat.console_trace_runtime" not in sys.modules
asyncio.run(gateway.aclose())
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


class ConsoleChatStore(_ConsoleChatStore):
    """Test store whose intentionally db-less sessions are explicitly ephemeral."""

    def create_session(self, **kwargs):
        kwargs.setdefault("ephemeral", self.persistence is None)
        return super().create_session(**kwargs)


class _StalledGateway:
    """Streams one chunk, then never finishes until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.never_release = asyncio.Event()

    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
                "resolved_destination": ConsoleResolvedDestination(
                    provider="llama_cpp",
                    model="test-model",
                    endpoint_identity="http://127.0.0.1:9099",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            },
        )()

    async def stream_chat(self, resolution, messages, **kwargs):
        self.started.set()
        yield "partial"
        await self.never_release.wait()
        yield "never"


class _ThreadApp:
    """The app surface `request_mcp_approvals` needs to reach its poll loop.

    **Not decoration.** ADR-067 added a no-`app` guard to
    `request_mcp_approvals` that denies every name on the spot when
    `controller.app is None` -- so the two approval-round tests below,
    written before that guard existed, stopped exercising the poll loop
    at all and passed on the guard's verdict instead. Measured
    (task-15860 Task 5): with the visit-cancel check deleted outright --
    fail-open for every session-scoped round -- this whole file was
    still 14/14 green in 0.98s, which is less than one poll interval.
    Wiring an app restores what these tests claim to pin.
    """

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class _SensitiveCustodyData:
    """Weak-referenceable evidence owned only by one custody request."""


class _View:
    """The smallest thing `ConsoleRuntime` accepts as a view."""

    def __init__(self, hooks: dict | None = None) -> None:
        self._hooks = hooks or {}
        self.attachment_generation: int | None = None

    def console_view_hooks(self) -> dict:
        return dict(self._hooks)


def _runtime_with(controller: ConsoleChatController, view: _View) -> ConsoleRuntime:
    """A runtime holding `controller`, attached to `view`."""
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(controller.store)
    runtime.set_chat_controller(controller)
    view.attachment_generation = runtime.attach_view(view)
    runtime.finish_view_reconciliation(view, view.attachment_generation)
    return runtime


def _pending_call() -> MCPPendingCall:
    return MCPPendingCall(
        llm_name="write_file",
        server_key="agent:builtin",
        tool_name="write_file",
        server_label="Built-in",
        arguments={},
        reason="risk_floored",
    )


def _custody_request(
    *,
    turn_id: str = "turn-custody-1",
    draft: str = "sensitive prompt",
    launch: ConsoleLiveWorkLaunch | None = None,
    attachment_ids: tuple[str, ...] = ("attachment-opaque-id",),
) -> ConsoleTurnCustodyRequest:
    """Build one detached runtime-custody request."""
    return ConsoleTurnCustodyRequest(
        turn_id=turn_id,
        session_id="session-custody-1",
        draft=draft,
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id="session-custody-1",
            provider_selection=ConsoleProviderSelection(
                provider="openai",
                system_prompt="credential-secret",
            ),
            rag_defaults={"rag": "rag-secret"},
            tool_configuration={"tool": "tool-secret"},
            provider_payload_settings={"credential": "credential-secret"},
        ),
        attachment_ids=attachment_ids,
        staged_evidence_launch=launch,
    )


def _runtime_with_custody_inputs(
    request: ConsoleTurnCustodyRequest,
) -> tuple[ConsoleRuntime, ConsoleChatStore, tuple[PendingAttachment, ...]]:
    """Attach the request's real session and exact staged attachments."""
    store = ConsoleChatStore()
    store.create_session(session_id=request.session_id)
    attachments = tuple(
        PendingAttachment(
            file_path=f"/{attachment_id}",
            display_name=attachment_id,
            file_type="image",
            insert_mode="attachment",
            attachment_id=attachment_id,
        )
        for attachment_id in request.attachment_ids
    )
    for attachment in attachments:
        assert store.add_pending_attachment(request.session_id, attachment)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    return runtime, store, attachments


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_start", [True, False])
async def test_archive_reservation_and_recovery_follow_actual_custody(
    cancel_before_start, monkeypatch
):
    from tldw_chatbook.Chat import conversation_archive_actions as archive

    request = _custody_request()
    runtime, store, attachments = _runtime_with_custody_inputs(request)
    session = store.sessions()[0]
    session.persisted_conversation_id = "original-conversation"
    runtime._app = SimpleNamespace(console_runtime=runtime)
    entered = asyncio.Event()
    release = asyncio.Event()
    observed = []

    async def refusal(app, conversation_id):
        observed.append(conversation_id)
        entered.set()
        await release.wait()
        return "This conversation is archived."

    async def chain(*, initial_turn, **kwargs):
        return await initial_turn()

    async def submit(*args, **kwargs):
        pytest.fail("Archived custody reached provider submission")

    runtime._chat_controller = SimpleNamespace(
        run_prompt_chain=chain, submit_draft=submit
    )
    monkeypatch.setattr(archive, "conversation_send_refusal", refusal)
    turn_id = runtime.accept_turn(request)
    record = runtime._turn_custody[turn_id]
    task = record.task
    try:
        assert getattr(runtime._app, "_conversation_send_inflight", {}) == {
            "original-conversation": 1
        }
        session.persisted_conversation_id = "new-conversation"
        if not cancel_before_start:
            await asyncio.wait_for(entered.wait(), 1)
            assert observed == ["original-conversation"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert runtime._app._conversation_send_inflight == {}
        recovery = runtime._turn_recoveries[turn_id]
        assert recovery.draft == request.draft
        assert recovery.attachments == attachments
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_custody_registers_before_the_runtime_task_starts():
    """An accepted turn is retained synchronously before it can run."""
    request = _custody_request()
    runtime, _, _ = _runtime_with_custody_inputs(request)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run(record, **_kwargs):
        assert runtime._turn_custody[record.turn_id] is record
        started.set()
        await release.wait()

    runtime._run_custodied_turn = run
    turn_id = runtime.accept_turn(request)

    record = runtime._turn_custody[turn_id]
    assert turn_id == "turn-custody-1"
    assert record.session_id == "session-custody-1"
    assert record.task is not None
    assert not started.is_set()

    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(record.task, timeout=1)
    await asyncio.sleep(0)

    assert turn_id not in runtime._turn_custody


@pytest.mark.asyncio
async def test_custody_terminal_cleanup_releases_sensitive_request_references():
    """Terminal cleanup removes the only retained launch reference."""
    release = asyncio.Event()

    async def run(record, **_kwargs):
        await release.wait()

    launch = ConsoleLiveWorkLaunch.from_values(
        source="RAG secret",
        title="attachment-name-secret",
        payload={"tool": "tool-secret", "credential": "credential-secret"},
    )
    launch_ref = weakref.ref(launch)
    request = _custody_request(launch=launch)
    runtime, _, _ = _runtime_with_custody_inputs(request)
    runtime._run_custodied_turn = run
    turn_id = runtime.accept_turn(request)
    task = runtime._turn_custody[turn_id].task
    del request, launch

    release.set()
    await asyncio.wait_for(task, timeout=1)
    await asyncio.sleep(0)
    gc.collect()

    assert turn_id not in runtime._turn_custody
    assert launch_ref() is None


@pytest.mark.asyncio
async def test_custody_exception_cleanup_severs_traceback_retained_references(
    monkeypatch,
):
    """A failed task cannot retain its custody request through its traceback."""
    started = asyncio.Event()
    release = asyncio.Event()
    terminal = asyncio.Event()

    async def run(record, **_kwargs):
        started.set()
        await release.wait()
        raise LookupError(record.turn_id)

    def warning(*_args):
        terminal.set()

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime.logger.warning",
        warning,
    )
    evidence = _SensitiveCustodyData()
    evidence_ref = weakref.ref(evidence)
    launch = ConsoleLiveWorkLaunch(
        source="RAG evidence",
        title="staged evidence",
        payload={"evidence": evidence},
    )
    request = _custody_request(launch=launch)
    runtime, _, _ = _runtime_with_custody_inputs(request)
    runtime._run_custodied_turn = run
    turn_id = runtime.accept_turn(request)
    task = runtime._turn_custody[turn_id].task
    del request, launch, evidence

    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(terminal.wait(), timeout=1)
    gc.collect()

    assert task.done()
    assert turn_id not in runtime._turn_custody
    assert evidence_ref() is None


@pytest.mark.asyncio
async def test_custody_cancellation_cleanup_severs_retained_record_references():
    """Cancellation releases sensitive inputs even if a caller retains the record."""
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def run(record, **_kwargs):
        started.set()
        await blocked.wait()

    evidence = _SensitiveCustodyData()
    evidence_ref = weakref.ref(evidence)
    launch = ConsoleLiveWorkLaunch(
        source="RAG evidence",
        title="staged evidence",
        payload={"evidence": evidence},
    )
    request = _custody_request(launch=launch)
    runtime, _, _ = _runtime_with_custody_inputs(request)
    runtime._run_custodied_turn = run
    turn_id = runtime.accept_turn(request)
    record = runtime._turn_custody[turn_id]
    task = record.task
    del request, launch, evidence

    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    gc.collect()

    assert turn_id not in runtime._turn_custody
    assert record.request is None
    assert record.task is None
    assert evidence_ref() is None


@pytest.mark.asyncio
async def test_custody_consumes_task_failures_before_releasing_the_record(
    monkeypatch,
):
    """The done callback retrieves failures instead of leaking task warnings."""
    failure_seen = asyncio.Event()
    warning_args: list[object] = []

    async def run(record, **_kwargs):
        del record
        raise LookupError("provider failure")

    def warning(*args):
        warning_args.extend(args)
        failure_seen.set()

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime.logger.warning",
        warning,
    )
    request = _custody_request()
    runtime, _, _ = _runtime_with_custody_inputs(request)
    runtime._run_custodied_turn = run
    turn_id = runtime.accept_turn(request)

    await asyncio.wait_for(failure_seen.wait(), timeout=1)

    assert turn_id not in runtime._turn_custody
    assert warning_args[-1] == "LookupError"


def test_custody_request_repr_redacts_sensitive_turn_inputs():
    """Process-local custody objects never expose request contents in reprs."""
    request = _custody_request(
        draft="prompt-secret",
        launch=ConsoleLiveWorkLaunch.from_values(
            source="RAG secret",
            title="attachment-name-secret",
            payload={"tool": "tool-secret", "credential": "credential-secret"},
        ),
    )

    representation = repr(request)
    runtime = ConsoleRuntime(app=None)
    record = runtime._register_custody(request)

    for secret in (
        "prompt-secret",
        "attachment-name-secret",
        "RAG secret",
        "tool-secret",
        "credential-secret",
    ):
        assert secret not in representation
        assert secret not in repr(record)


@pytest.mark.asyncio
async def test_custody_task_creation_failure_releases_registration(monkeypatch):
    """Admission leaves draft recovery with its caller if scheduling fails."""
    draft = "recover this draft"
    attachment_ids = (
        "attachment-recovery-first",
        "attachment-recovery-second",
    )
    evidence = ConsoleLiveWorkLaunch.from_values(
        source="RAG evidence",
        title="recovery evidence",
        payload={"evidence": "sensitive recovery payload"},
    )
    request = _custody_request(
        draft=draft,
        attachment_ids=attachment_ids,
        launch=evidence,
    )
    runtime, store, attachments = _runtime_with_custody_inputs(request)

    def fail_create_task(coroutine):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime.asyncio.create_task",
        fail_create_task,
    )

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        runtime.accept_turn(request)

    assert runtime._turn_custody == {}
    restored = store.pending_attachments(request.session_id)
    assert restored == list(attachments)
    assert all(
        actual is expected
        for actual, expected in zip(restored, attachments, strict=True)
    )
    assert request.draft == draft
    assert request.attachment_ids is attachment_ids
    assert request.staged_evidence_launch is evidence
    assert request.staged_evidence_launch.payload == {
        "evidence": "sensitive recovery payload"
    }


# -- Navigation is a pure view detach ---------------------------------------


@pytest.mark.asyncio
async def test_leaving_console_preserves_stream_queue_and_controller_state():
    """Navigation clears the view without mutating app-owned work."""
    gateway = _StalledGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    view = _View()
    runtime = _runtime_with(controller, view)
    visit_event = controller._shutdown_requested
    coordinator = controller.prompt_queue_coordinator

    turn = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)
    session_id = store.active_session_id
    assert controller._active_stream_tasks.get(session_id) is not None

    try:
        assert (
            await asyncio.wait_for(
                runtime.leave_console(view, view.attachment_generation),
                timeout=2,
            )
            is True
        )
        assert runtime.view is None
        assert controller._shutdown_requested is visit_event
        assert not visit_event.is_set()
        assert not turn.done()
        assert controller._active_stream_tasks.get(session_id) is not None
        revision = coordinator.registry.snapshot(session_id).revision
        assert (
            coordinator.admit(
                session_id, text="queued", expected_revision=revision
            ).status
            is QueueMutationStatus.REROUTE_NORMAL_SEND
        )
        assert runtime.chat_controller is controller
        assert runtime.chat_store is store
        assert runtime.generation == 0
    finally:
        gateway.never_release.set()
        await asyncio.wait_for(turn, timeout=1)

    assert store.messages_for_session(session_id)[-1].status == "complete"


# -- Pending decisions remain authoritative while detached -----------------


@pytest.mark.asyncio
async def test_leaving_console_does_not_resolve_a_pending_approval_round():
    """Detaching neither denies nor otherwise resolves a human decision."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    # Without an app the round never reaches the poll loop -- see `_ThreadApp`.
    controller.app = _ThreadApp()
    mounted = threading.Event()
    view = _View({"set_pending_approval": lambda payload: mounted.set()})
    runtime = _runtime_with(controller, view)

    decisions: dict[str, str] = {}

    def _run_round() -> None:
        decisions.update(
            controller.request_mcp_approvals([_pending_call()], session_id=session.id)
        )

    worker = threading.Thread(target=_run_round, daemon=True)
    worker.start()
    assert mounted.wait(timeout=2), "the round never mounted"
    round_id = next(iter(controller._pending_approval_rounds))
    round_state = controller._pending_approval_rounds[round_id]

    try:
        await asyncio.wait_for(
            runtime.leave_console(view, view.attachment_generation), timeout=5
        )

        assert not controller._shutdown_requested.is_set()
        assert round_id in controller._pending_approval_rounds
        assert not round_state["event"].is_set()
        assert worker.is_alive()
        assert decisions == {}
    finally:
        controller.resolve_pending_approval({"write_file": "deny"}, round_id=round_id)
        worker.join(timeout=10)

    assert decisions == {"write_file": "deny"}


# -- the owner ruling: a wake turn is NOT a user turn ----------------------


@pytest.mark.asyncio
async def test_leaving_console_does_not_cancel_an_in_flight_wake_turn():
    """Owner ruling: `leave_console` never cancels an `AGENT_WAKE` turn.

    Cancelling it would re-create the "only completes if you stay" gap
    this whole arc exists to close, and a wake turn is structurally the
    same class of work as the fleet survivor AC#2 keeps running.
    """
    gateway = _StalledGateway()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = _runtime_with(controller, _View())

    # Stand in for a wake turn in flight: the exemption registry and a real
    # (stalled) stream task for the same session, exactly as `submit_draft`
    # leaves them mid-turn.
    async def _stalled_turn() -> None:
        await gateway.never_release.wait()

    task = asyncio.create_task(_stalled_turn())
    await asyncio.sleep(0)
    controller._active_stream_tasks[session.id] = task
    controller._agent_wake_turn_sessions.add(session.id)

    await asyncio.wait_for(runtime.leave_console(), timeout=2)

    assert not task.cancelled() and not task.done(), (
        "leaving Console cancelled a wake turn -- the owner ruling is that it must not"
    )
    assert controller._active_stream_tasks.get(session.id) is task
    gateway.never_release.set()
    task.cancel()


@pytest.mark.asyncio
async def test_the_wake_exemption_never_outlives_its_turn():
    """A wake turn that finishes leaves nothing exempt behind."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    with pytest.raises(PermissionError):
        # No coordinator-issued token -> refused before anything runs, which
        # is enough to prove the registry is not populated by the attempt.
        await controller.submit_draft(
            "notice", origin=ConsoleSubmissionOrigin.AGENT_WAKE
        )
    assert controller._agent_wake_turn_sessions == set()


# -- the per-visit Event, captured at ARM time -----------------------------


def test_begin_visit_installs_a_fresh_cancellation_event():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    first = controller._shutdown_requested
    first.set()

    controller.begin_visit()

    assert controller._shutdown_requested is not first
    assert not controller._shutdown_requested.is_set()
    assert first.is_set(), "the previous visit's Event must stay set forever"


def test_a_disposed_controller_is_never_re_opened():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    controller.begin_shutdown()
    event = controller._shutdown_requested

    controller.begin_visit()

    assert controller._shutdown_requested is event
    assert controller._shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_a_round_from_the_previous_visit_is_not_resurrected():
    """A pending decision remains the same unresolved round after reattach."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    controller.mcp_approval_timeout_seconds = lambda: 60.0
    # Without an app the round never reaches the poll loop -- see `_ThreadApp`.
    controller.app = _ThreadApp()
    runtime = _runtime_with(controller, _View())

    decisions: dict[str, str] = {}
    armed = threading.Event()
    resolved = threading.Event()

    def _run_round() -> None:
        armed.set()
        decisions.update(
            controller.request_mcp_approvals([_pending_call()], session_id=session.id)
        )
        resolved.set()

    worker = threading.Thread(target=_run_round, daemon=True)
    worker.start()
    assert armed.wait(timeout=2)
    await asyncio.sleep(0.2)

    round_id = next(iter(controller._pending_approval_rounds))
    visit_event = controller._shutdown_requested
    await asyncio.wait_for(runtime.leave_console(), timeout=5)
    runtime.attach_view(_View())
    assert controller._shutdown_requested is visit_event
    assert not visit_event.is_set()
    assert round_id in controller._pending_approval_rounds
    assert not resolved.is_set()
    assert decisions == {}

    controller.resolve_pending_approval({"write_file": "deny"}, round_id=round_id)
    worker.join(timeout=10)
    assert resolved.is_set()
    assert decisions == {"write_file": "deny"}


# -- per-visit queue admission --------------------------------------------


@pytest.mark.asyncio
async def test_the_prompt_queue_admits_again_on_the_next_visit():
    """Navigation never tombstones or replaces the prompt queue."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    runtime = _runtime_with(controller, _View())
    coordinator = controller.prompt_queue_coordinator

    def _admit_status():
        revision = coordinator.registry.snapshot(session.id).revision
        return coordinator.admit(
            session.id, text="queued", expected_revision=revision
        ).status

    # An empty queue reroutes to a normal send -- the "admission is open"
    # answer, and emphatically not `SHUTTING_DOWN`.
    assert _admit_status() is QueueMutationStatus.REROUTE_NORMAL_SEND

    await asyncio.wait_for(runtime.leave_console(), timeout=2)
    assert _admit_status() is QueueMutationStatus.REROUTE_NORMAL_SEND

    runtime.attach_view(_View())

    assert _admit_status() is QueueMutationStatus.REROUTE_NORMAL_SEND, (
        "the prompt queue changed admission across a view-only navigation"
    )
    assert controller.prompt_queue_coordinator is coordinator


# -- dispose keeps today's behaviour exactly -------------------------------


@pytest.mark.asyncio
async def test_runtime_closes_its_thread_local_db_after_coordinator_disposal(tmp_path):
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(db_path=tmp_path / "chatbook.db")
    )
    runtime = ConsoleRuntime(app)

    bridge = runtime.ensure_agent_bridge(
        store_factory=ConsoleChatStore,
        provider_gateway_factory=lambda: _StalledGateway(),
    )

    assert bridge is not None
    assert runtime.change_review_coordinator is not None
    assert bridge._change_finalization_coordinator is runtime.change_review_coordinator
    assert runtime._agent_runs_db._thread_local.conn is not None

    await runtime.dispose()

    assert runtime._agent_runs_db._thread_local.conn is None


@pytest.mark.asyncio
async def test_dispose_orders_change_review_before_db_and_gateway_close():
    calls = []

    class _Controller:
        async def shutdown(self):
            calls.append("controller")

    class _DB:
        def close(self):
            calls.append("runtime-db")

    class _Coordinator:
        def shutdown(self, timeout):
            calls.append(("coordinator", timeout))
            calls.append("publisher-db")
            return True

    class _Gateway:
        async def aclose(self):
            calls.append("gateway")

    runtime = ConsoleRuntime(app=None)
    runtime._chat_controller = _Controller()
    runtime._change_review_coordinator = _Coordinator()
    runtime._agent_runs_db = _DB()
    runtime._provider_gateway = _Gateway()

    await runtime.dispose()

    assert calls == [
        "controller",
        ("coordinator", 2.0),
        "publisher-db",
        "runtime-db",
        "gateway",
    ]


@pytest.mark.asyncio
async def test_dispose_publishes_content_free_trace_compatibility_totals():
    snapshots: list[dict[str, int]] = []

    class _Metrics:
        def snapshot(self):
            snapshot = {
                "normalized_write": 4,
                "normalized_read": 3,
                "legacy_read": 2,
                "fallback_read": 1,
                "incomplete": 0,
            }
            snapshots.append(snapshot)
            return snapshot

    runtime = ConsoleRuntime(app=None)
    runtime.trace_compatibility_metrics = _Metrics()

    await runtime.dispose()

    assert snapshots == [
        {
            "normalized_write": 4,
            "normalized_read": 3,
            "legacy_read": 2,
            "fallback_read": 1,
            "incomplete": 0,
        }
    ]


@pytest.mark.asyncio
async def test_dispose_closes_its_thread_local_db_after_coordinator_timeout():
    calls = []

    class _Coordinator:
        def shutdown(self, timeout):
            calls.append(("coordinator", timeout))
            return False

    class _DB:
        def close(self):
            calls.append("db")

    runtime = ConsoleRuntime(app=None)
    runtime._change_review_coordinator = _Coordinator()
    runtime._agent_runs_db = _DB()

    await runtime.dispose()

    assert calls == [("coordinator", 2.0), "db"]


@pytest.mark.asyncio
async def test_dispose_shuts_the_controller_down_and_closes_the_gateway():
    closed: list[bool] = []

    class _Gateway(_StalledGateway):
        async def aclose(self) -> None:
            closed.append(True)

    gateway = _Gateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())

    await asyncio.wait_for(runtime.dispose(), timeout=2)

    assert controller._shutdown_requested.is_set()
    assert controller._disposed is True
    assert closed == [True], "dispose must close the app-owned gateway"
    assert runtime.view is None
    assert runtime.generation == 1


@pytest.mark.asyncio
async def test_dispose_does_not_let_a_late_ensure_rebuild_the_runtime():
    """A quit-time tick must not resurrect what dispose just tore down.

    `_shutdown_app_owned_lifecycles` runs BEFORE Textual closes screen
    state, so a Console screen and its 0.2s timers can still be live while
    `dispose()` runs -- and ~75 `_ensure_console_chat_*` call sites are
    reachable from them. If `ensure_*` built a fresh object then, quit
    would leave a brand-new controller alive that nothing ever shuts down.
    """
    store = ConsoleChatStore()
    gateway = _StalledGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())

    await asyncio.wait_for(runtime.dispose(), timeout=2)

    # (a) an ALREADY-BUILT slot hands back the torn-down object rather than
    #     a fresh one -- dispose keeps its references precisely for this.
    assert runtime.ensure_chat_controller() is controller, (
        "dispose must not let a late ensure build a SECOND controller"
    )
    assert runtime.ensure_chat_store() is store
    assert runtime.ensure_provider_gateway() is gateway
    # ...and what it hands back is genuinely torn down, so it refuses work.
    assert controller._shutdown_requested.is_set()


@pytest.mark.asyncio
async def test_dispose_does_not_let_a_late_ensure_build_an_unbuilt_slot():
    """The `_disposed` latch itself, on the case references cannot cover.

    A slot that was never built before quit has no reference to hand back,
    so only the latch stops `ensure_*` constructing a brand-new store (and,
    with it, a fresh `ChatPersistenceService`) while the app is exiting.
    """
    runtime = ConsoleRuntime(app=None)
    runtime.attach_view(_View())
    assert runtime.chat_store is None, "nothing built yet -- that is the point"

    await asyncio.wait_for(runtime.dispose(), timeout=2)

    assert runtime.ensure_chat_store() is None, (
        "a late tick built a fresh store DURING QUIT"
    )
    assert runtime.ensure_provider_gateway() is None
    assert runtime.ensure_chat_controller() is None
    assert runtime.chat_store is None


@pytest.mark.asyncio
async def test_leaving_console_does_not_close_the_provider_gateway():
    """The gateway is app-owned now; a surviving turn still needs it."""
    closed: list[bool] = []

    class _Gateway(_StalledGateway):
        async def aclose(self) -> None:
            closed.append(True)

    gateway = _Gateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    runtime.attach_view(_View())

    await asyncio.wait_for(runtime.leave_console(), timeout=2)

    assert closed == []
    assert runtime.provider_gateway is gateway


def test_lifecycle_impact_counts_a_delegated_child_after_parent_turn_finishes():
    """A survivor is loss-impact even when no parent run occupies a slot."""

    class _Handle:
        status = "running"

    class _FleetBridge:
        def __init__(self) -> None:
            self.activity = None

        def on_fleet_drained(self, _name, _consumer) -> None:
            return None

        def on_fleet_activity(self, _name, consumer) -> None:
            self.activity = consumer

        def fleet_snapshot(self, _conversation_id):
            return [_Handle()]

    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = _FleetBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StalledGateway(),
        agent_bridge=bridge,
    )

    impact = controller.lifecycle_impact(session_id=session.id)

    assert impact.live_run_count == 0
    assert impact.delegated_child_count == 1
    assert impact.has_loss_risk is True


def test_fleet_activity_advances_global_and_owning_session_lifecycle_revisions():
    """Spawn/settle notifications invalidate revision-pinned consent."""

    class _FleetBridge:
        def __init__(self) -> None:
            self.activity = None

        def on_fleet_drained(self, _name, _consumer) -> None:
            return None

        def on_fleet_activity(self, _name, consumer) -> None:
            self.activity = consumer

        def fleet_snapshot(self, _conversation_id):
            return []

    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = _FleetBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StalledGateway(),
        agent_bridge=bridge,
    )
    before_session = controller.lifecycle_impact(session_id=session.id)
    before_global = controller.lifecycle_impact()

    assert bridge.activity is not None
    bridge.activity(session.id)

    assert controller.lifecycle_impact(session_id=session.id).revision > (
        before_session.revision
    )
    assert controller.lifecycle_impact().revision > before_global.revision


def test_fleet_activity_uses_snapshot_seeded_owner_without_reading_store():
    """Child-thread admission never iterates the mutable session dictionary."""

    class _FleetBridge:
        def __init__(self) -> None:
            self.activity = None

        def on_fleet_drained(self, _name, _consumer) -> None:
            return None

        def on_fleet_activity(self, _name, consumer) -> None:
            self.activity = consumer

        def fleet_snapshot(self, _conversation_id):
            return []

    store = ConsoleChatStore()
    session = store.ensure_session()
    session.persisted_conversation_id = "saved-conversation"
    bridge = _FleetBridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StalledGateway(),
        agent_bridge=bridge,
    )
    before = controller.lifecycle_impact(session_id=session.id).revision
    original_sessions = store.sessions

    def forbid_cross_thread_store_read():
        raise AssertionError("fleet callback read the mutable store")

    store.sessions = forbid_cross_thread_store_read
    try:
        bridge.activity("saved-conversation")
    finally:
        store.sessions = original_sessions

    assert controller.lifecycle_impact(session_id=session.id).revision > before


def test_lifecycle_impact_does_not_treat_a_failed_fleet_snapshot_as_idle():
    """Unknown delegated state must block destructive confirmation."""

    class _FleetBridge:
        def on_fleet_drained(self, _name, _consumer) -> None:
            return None

        def on_fleet_activity(self, _name, _consumer) -> None:
            return None

        def fleet_snapshot(self, _conversation_id):
            raise RuntimeError("fleet unavailable")

    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_StalledGateway(),
        agent_bridge=_FleetBridge(),
    )

    with pytest.raises(RuntimeError, match="fleet unavailable"):
        controller.lifecycle_impact(session_id=session.id)


# -- the wake gate is NOT relaxed here -------------------------------------


@pytest.mark.asyncio
async def test_the_visit_cancellation_event_is_set_between_visits():
    """Navigation preserves the exact unset cancellation-event identity."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    runtime = _runtime_with(controller, _View())
    visit_event = controller._shutdown_requested
    assert not visit_event.is_set()

    await asyncio.wait_for(runtime.leave_console(), timeout=2)
    assert controller._shutdown_requested is visit_event
    assert not visit_event.is_set()

    runtime.attach_view(_View())
    assert controller._shutdown_requested is visit_event
    assert not visit_event.is_set()


@pytest.mark.asyncio
async def test_leave_console_is_idempotent_and_cheap_with_no_work():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=_StalledGateway())
    runtime = _runtime_with(controller, _View())

    started = time.monotonic()
    assert await runtime.leave_console() is True
    # A second leave finds no view attached and does nothing.
    assert await runtime.leave_console(object()) is False
    assert time.monotonic() - started < 2.0
