"""Local control integration retains the actual Console owners."""

from __future__ import annotations

import asyncio

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case
from Tests.Chat.test_console_durable_commit_plan import prepared

domain_case = _domain_case  # Reuse the original fixture without duplicating its owners.

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
@pytest.mark.parametrize("refusal", ["stale_revision", "draft_custody"])
async def test_refused_original_close_releases_only_provisional_marker(
    domain_case, refusal
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        CloseSession,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
    )

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )

    def identity(sequence):
        return ControlIdentity(
            "profile",
            "service",
            sequence,
            f"close-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    revision = case.controller.lifecycle_impact(session_id=case.session.id).revision
    if refusal == "draft_custody":
        # Actual original session state: drain_agent_handoff refuses before
        # taking its irreversible close ticket. No close method is replaced.
        case.session.agent_handoff_state = "review_required"
    close = controls.dispatch(
        CloseSession(identity(1), revision + (refusal == "stale_revision"))
    )
    with pytest.raises(RuntimeError, match="closing"):
        controls.dispatch(UpdateInputs(identity(2), "still pending", (1, 1)))
    with pytest.raises(RuntimeError):
        await close.applied
    assert case.store._sessions[case.session.id] is case.session
    assert case.session.id not in controls._closing_sessions
    assert case.session.id not in case.runtime._admission_fenced_sessions
    updated = controls.dispatch(UpdateInputs(identity(3), "surviving input", (1, 2)))
    received = controls.dispatch(
        ReceiveTurn(
            identity(4), "surviving-turn", updated.applied.result(), _selection(case)
        )
    )
    record = case.runtime._turn_custody["surviving-turn"]
    assert received.work is record.task and record.received_claim is not None
    controls.dispatch(CancelTurn(identity(5), "surviving-turn"))
    await asyncio.gather(received.work, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_original_irreversible_close_fence_keeps_selected_intake_closed(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        CloseSession,
        ControlIdentity,
        UpdateInputs,
    )

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    revision = case.controller.lifecycle_impact(session_id=case.session.id).revision
    ticket = case.controller.begin_session_close(
        case.session.id, expected_revision=revision
    )

    def identity(sequence):
        return ControlIdentity(
            "profile",
            "service",
            sequence,
            f"fenced-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    try:
        receipt = controls.dispatch(CloseSession(identity(1), revision))
        with pytest.raises(RuntimeError):
            await receipt.applied
        assert case.session.id in case.controller._session_close_generations
        assert case.session.id in controls._closing_sessions
        with pytest.raises(RuntimeError, match="closing"):
            controls.dispatch(UpdateInputs(identity(2), "refused", (1, 1)))
    finally:
        case.controller.finalize_session_close(ticket)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_more_than_4096_controls_preserve_intake_cancel_and_shutdown(domain_case):
    from Tests.Chat.test_console_domain_boundary import _selection
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        CloseIntake,
        ControlIdentity,
        Drain,
        ReceiveTurn,
        Shutdown,
        UpdateInputs,
    )

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )

    def identity(sequence, request_id=None):
        return ControlIdentity(
            "profile",
            "service",
            sequence,
            request_id or f"many-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    first = controls.dispatch(UpdateInputs(identity(1), "pressed", (1, 1)))
    received = controls.dispatch(
        ReceiveTurn(
            identity(2), "retained-turn", first.applied.result(), _selection(case)
        )
    )
    original = case.runtime._turn_custody["retained-turn"].task
    for sequence in range(3, 4103):
        controls.dispatch(
            UpdateInputs(identity(sequence), f"draft {sequence}", (1, sequence))
        )
    assert controls._intake_open and len(controls._requests) == 4102
    cancelled = controls.dispatch(CancelTurn(identity(4103), "retained-turn"))
    assert cancelled.work is original is received.work
    controls.dispatch(CloseIntake(identity(4104)))
    assert await controls.dispatch(Drain(identity(4105))).applied is True
    with pytest.raises(RuntimeError, match="Conflicting"):
        controls.dispatch(
            UpdateInputs(identity(4106, "many-1"), "duplicate", (1, 4106))
        )
    with pytest.raises(RuntimeError, match="out-of-order"):
        controls.dispatch(
            UpdateInputs(identity(2, "reordered"), "reordered", (1, 4107))
        )
    assert await controls.dispatch(Shutdown(identity(4107))).applied is True
    assert case.runtime._disposed


class _CaptureRouteDiagnostics:
    """Observe original code-owned route facts without replacing producers."""

    def __init__(self, controller):
        import sys
        import time
        from types import CodeType
        from tldw_chatbook.Chat.console_configuration_preparation import (
            capture_console_turn_configuration_owned,
        )
        from tldw_chatbook.Chat.console_send_diagnostics import record_send_stage

        self.monitoring = sys.monitoring
        self.slot = next(
            value for value in range(6) if self.monitoring.get_tool(value) is None
        )
        self.monitoring.use_tool_id(self.slot, "task60-capture-route")
        self.rows = []
        self.phases = []
        self.started = time.monotonic()
        capture = capture_console_turn_configuration_owned.__code__
        self.codes = [capture, record_send_stage.__code__] + [
            value
            for value in capture.co_consts
            if isinstance(value, CodeType) and value.co_name == "capture"
        ]

        def observe(code, *args):
            frame = sys._getframe(1)
            values = frame.f_locals
            elapsed = round(time.monotonic() - self.started, 3)
            if code is record_send_stage.__code__:
                if len(self.phases) < 64:
                    self.phases.append(
                        (elapsed, values.get("phase"), values.get("status"))
                    )
            elif values.get("creator") is controller and "planning" in values:
                self.rows.append(
                    {
                        "elapsed": elapsed,
                        "site": code.co_name,
                        "planning": values["planning"] is not None,
                        "bounded": values.get("bounded_mcp_sources") is not None,
                        "capture_performed": values.get("mcp_capture_performed"),
                        "complete": values.get("mcp_complete"),
                        "pending": values.get("mcp_pending"),
                        "stock": values.get("stock_mcp"),
                        "composition": values.get("composition_mcp"),
                        "definitions": len(values.get("maximum") or {}),
                    }
                )

        events = self.monitoring.events
        self.monitoring.register_callback(self.slot, events.PY_START, observe)
        self.monitoring.register_callback(self.slot, events.PY_RETURN, observe)
        for code in self.codes:
            self.monitoring.set_local_events(
                self.slot, code, events.PY_START | events.PY_RETURN
            )

    def close(self):
        for code in self.codes:
            self.monitoring.set_local_events(self.slot, code, 0)
        self.monitoring.register_callback(
            self.slot, self.monitoring.events.PY_START, None
        )
        self.monitoring.register_callback(
            self.slot, self.monitoring.events.PY_RETURN, None
        )
        self.monitoring.free_tool_id(self.slot)


@pytest.mark.parametrize("domain_case", ["control_trace"], indirect=True)
@pytest.mark.parametrize("provider", ["openai", "llama_cpp"])
@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.loopback_network
async def test_exact_ack_reaches_original_loopback_provider_and_terminal(
    domain_case, provider, capture_enabled
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from Tests.Chat.test_console_refused_stream_retry import (
        _MODEL,
        _REPLY,
        _gateway_config,
        _server,
    )
    from tldw_chatbook.Chat.console_control_dispatch import (
        CommitPlanAck,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = domain_case
    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "positive-history.jsonl"
    )
    with _server() as server:
        case.sources.configuration.update(_gateway_config(provider, server.base_url))
        case.store.replace_session_settings(
            case.session.id,
            ConsoleSessionSettings(
                provider=provider,
                model=_MODEL,
                base_url=server.base_url,
                streaming=False,
            ),
        )
        case.store.replace_session_trace_privacy_override(
            case.session.id,
            capture_enabled=capture_enabled,
            pii_redaction_enabled=None,
            expected_policy_revision=case.store.capture_policy_state(
                case.session.id
            ).policy_revision,
        )
        # This positive provider turn offers no tools. Keep the original
        # definitive-empty capture through the actual singleton policy owner.
        await case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            True
        )
        controls = case.runtime._execution_service.open_local_controls(
            profile_epoch="profile", service_epoch="service"
        )

        def identity(sequence):
            return ControlIdentity(
                "profile",
                "service",
                sequence,
                f"positive-{sequence}",
                case.session.id,
                case.session.incarnation_id,
            )

        diagnostics = _CaptureRouteDiagnostics(case.controller)
        controls.dispatch(UpdateInputs(identity(1), "Actual loopback turn", (1, 1)))
        receipt = controls.dispatch(
            ReceiveTurn(
                identity(2),
                "positive-turn",
                case.session.draft_revision,
                _selection(case),
            )
        )
        try:
            reached = await _until(lambda: controls.commit_plans or receipt.work.done())
            if not reached:
                cursor = receipt.work.get_coro()
                awaiting = []
                while cursor is not None:
                    frame = getattr(cursor, "cr_frame", None) or getattr(
                        cursor, "gi_frame", None
                    )
                    if frame is not None:
                        awaiting.append((frame.f_code.co_name, frame.f_lineno))
                        values = frame.f_locals
                        if "bounded_mcp_sources" in values:
                            diagnostics.rows.append(
                                {
                                    "site": frame.f_code.co_name,
                                    "planning": values.get("planning") is not None,
                                    "bounded": values["bounded_mcp_sources"]
                                    is not None,
                                    "capture_performed": values.get(
                                        "mcp_capture_performed"
                                    ),
                                    "complete": values.get("mcp_complete"),
                                    "pending": values.get("mcp_pending"),
                                    "stock": values.get("stock_mcp"),
                                    "composition": values.get("composition_mcp"),
                                }
                            )
                    cursor = getattr(cursor, "cr_await", None) or getattr(
                        cursor, "gi_yieldfrom", None
                    )
                print("Task60 original pending await chain:", awaiting)
                print("Task60 admitted effects:", controls.admitted_effects)
                print("Task60 original capture routes:", diagnostics.rows)
                print("Task60 original phase timeline:", diagnostics.phases)
                import sys

                stacks = []
                for frame in sys._current_frames().values():
                    stack = []
                    while frame is not None:
                        if "tldw_chatbook" in frame.f_code.co_filename:
                            stack.append((frame.f_code.co_name, frame.f_lineno))
                        frame = frame.f_back
                    if stack:
                        stacks.append(stack)
                print("Task60 original worker locations:", stacks)
            assert reached
            if receipt.work.done():
                receipt.work.result()
            assert len(controls.commit_plans) == 1
            plan = controls.commit_plans[0]
            controls.dispatch(
                CommitPlanAck(
                    identity(3),
                    plan.received_request_id,
                    plan.turn_id,
                    plan.ordinal,
                    plan.fingerprint,
                )
            )
            controls.dispatch(
                CommitPlanAck(
                    identity(4),
                    plan.received_request_id,
                    plan.turn_id,
                    plan.ordinal,
                    plan.fingerprint,
                )
            )
            result = await asyncio.wait_for(asyncio.shield(receipt.work), timeout=20)
            assert result.accepted
            assert len(server.payloads) == 1
            assert (
                _REPLY
                in case.store.read_only_messages_for_session(case.session.id)[
                    -1
                ].content
            )
            from tldw_chatbook.Chat.console_control_trace import drain_trace_worker

            await drain_trace_worker(case.store)
            assert not controls.admitted_effects
            if capture_enabled:
                assert len(controls._trace_coordinators) == 1
                assert next(iter(controls._trace_coordinators)).pending_count == 0
                with case.database.transaction() as cursor:
                    rows = cursor.execute(
                        "SELECT state FROM console_trace_calls"
                    ).fetchall()
                from tldw_chatbook.Chat.console_trace_models import TraceCallState

                assert len(rows) == 1 and rows[0][0] == TraceCallState.COMPLETE.value
            else:
                assert not controls._trace_coordinators
            assert (
                case.controller._ordinary_native_commit_owner(
                    plan.fingerprint.preparation_id
                )
                is None
            )
        finally:
            diagnostics.close()
            if not receipt.work.done():
                receipt.work.cancel()
            await asyncio.gather(receipt.work, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
@pytest.mark.parametrize("kill_switch", [False, True])
async def test_original_completed_mcp_capture_keeps_its_actual_maximum(
    domain_case, kill_switch
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_received_configuration_owned,
    )

    case = domain_case
    operations = case.runtime._execution_service.ensure_permission_operations()
    await operations.set_kill_switch(kill_switch)
    captured = await capture_received_configuration_owned(
        case.controller, case.session.id, selection=_selection(case)
    )
    assert captured.mcp_definition_complete and not captured.mcp_capture_pending
    assert captured.mcp_definition_capture == "captured"
    assert captured.mcp_tool_maximum == frozenset(captured.mcp_definition_maximum)
    assert bool(captured.mcp_definition_maximum) is not kill_switch


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
async def test_saved_open_uses_original_scope_database_and_whole_tree(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, OpenSession

    case = domain_case
    local = case.sources.conversation_scope_service.local_service
    target = local.create_conversation(
        title="Saved original", runtime_backend="local", scope_type="global"
    )
    case.database.add_message(
        {
            "conversation_id": target,
            "sender": "user",
            "role": "user",
            "content": "Saved input",
        }
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    receipt = controls.dispatch(
        OpenSession(
            ControlIdentity("profile", "service", 1, "open"),
            conversation_id=target,
        )
    )
    session_id, incarnation = await receipt.applied
    assert await receipt.retired is True
    session = case.store._sessions[session_id]
    assert session.incarnation_id == incarnation
    assert session.persisted_conversation_id == target
    assert (
        case.store.read_only_messages_for_session(session_id)[0].content
        == "Saved input"
    )
    assert (
        case.domain.conversation_scope_service()
        is case.sources.conversation_scope_service
    )
    assert case.domain.conversation_scope_service().local_service is local
    assert not controls.admitted_effects


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
async def test_saved_open_cancelled_before_entry_retires_without_loading(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, OpenSession

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    original_sessions = tuple(case.store._sessions)
    receipt = controls.dispatch(
        OpenSession(
            ControlIdentity("profile", "service", 1, "unentered-open"),
            conversation_id="unentered-saved-conversation",
        )
    )
    receipt.work.cancel()  # No event-loop handoff occurred after dispatch.
    with pytest.raises(asyncio.CancelledError):
        await receipt.applied
    assert await receipt.retired is True
    assert receipt.work.cancelled()
    assert tuple(case.store._sessions) == original_sessions
    assert not controls.admitted_effects


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
async def test_saved_open_rejects_replaced_original_reader_after_wait(domain_case):
    import sys
    import threading

    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, OpenSession

    case = domain_case
    scope = case.sources.conversation_scope_service
    original = scope.local_service
    target = original.create_conversation(title="Replaced reader", scope_type="global")
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    entered, release = threading.Event(), threading.Event()
    observed = []
    previous, thread_previous = sys.getprofile(), threading.getprofile()

    def hold(frame, event, _result):
        if (
            event == "call"
            and frame.f_code is ChatConversationService.get_conversation_tree.__code__
        ):
            observed.append(frame.f_locals["self"])
            entered.set()
            assert release.wait(10)

    before = tuple(case.store._sessions)
    threading.setprofile_all_threads(hold)
    receipt = controls.dispatch(
        OpenSession(
            ControlIdentity("profile", "service", 1, "open"),
            conversation_id=target,
        )
    )
    try:
        assert await _until(entered.is_set)
        scope.local_service = None
        release.set()
        with pytest.raises(RuntimeError):
            await receipt.applied
        assert await receipt.retired is True
        assert observed == [original]
        assert tuple(case.store._sessions) == before
        assert not controls.admitted_effects
    finally:
        release.set()
        threading.setprofile_all_threads(thread_previous)
        sys.setprofile(previous)
        await asyncio.gather(receipt.work, return_exceptions=True)
        scope.local_service = original


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_absent_saved_owner_preserves_original_new_session(domain_case):
    from tldw_chatbook.Chat.console_conversation_hydration import (
        ConversationServiceUnavailable,
    )
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, OpenSession

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    with pytest.raises(ConversationServiceUnavailable):
        controls.dispatch(
            OpenSession(
                ControlIdentity("profile", "service", 1, "saved"),
                conversation_id="missing",
            )
        )
    receipt = controls.dispatch(
        OpenSession(
            ControlIdentity("profile", "service", 2, "new"),
            title="New original",
        )
    )
    session_id, incarnation = await receipt.applied
    assert case.store._sessions[session_id].title == "New original"
    assert case.store._sessions[session_id].incarnation_id == incarnation


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_fence_waits_for_original_admitted_permission_mutation(domain_case):
    import sys
    import threading

    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        FenceSource,
        PermissionAction,
    )
    from tldw_chatbook.Chat.console_control_permissions import SetKillSwitch
    from tldw_chatbook.MCP.local_permission_operation import LocalPermissionOperation

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    entered, release = threading.Event(), threading.Event()
    owners = []
    previous, thread_previous = sys.getprofile(), threading.getprofile()

    def hold_original_entry(frame, event, _result):
        if (
            event == "call"
            and frame.f_code is LocalPermissionOperation.__enter__.__code__
        ):
            owners.append(frame.f_locals["self"])
            entered.set()
            assert release.wait(10), "Original permission owner was not released"

    threading.setprofile_all_threads(hold_original_entry)
    mutation = controls.dispatch(
        PermissionAction(
            ControlIdentity("profile", "service", 1, "mutation"),
            SetKillSwitch(value=True),
        )
    )
    try:
        assert await _until(entered.is_set)
        assert controls.admitted_effects == ("permission",)
        fence = controls.dispatch(
            FenceSource(
                ControlIdentity("profile", "service", 2, "fence"), "configuration"
            )
        )
        await asyncio.sleep(0)
        assert not fence.applied.done() and not mutation.applied.done()
        release.set()
        await mutation.applied
        assert await mutation.retired is True
        applied = await fence.applied
        assert applied.revision == 1 and controls.source_revision == 1
        assert len(owners) == 1 and owners[0].retired
        assert owners[0].observations["raw"]
        assert not controls.admitted_effects
        assert (
            await asyncio.to_thread(case.sources.control_plane.get_kill_switch) is True
        )
    finally:
        release.set()
        threading.setprofile_all_threads(thread_previous)
        sys.setprofile(previous)
        await asyncio.gather(mutation.work, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_fence_winning_refuses_queued_original_permission_mutation(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        FenceSource,
        PermissionAction,
    )
    from tldw_chatbook.Chat.console_control_permissions import SetKillSwitch

    case = domain_case
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            False
        )
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    mutation = controls.dispatch(
        PermissionAction(
            ControlIdentity("profile", "service", 1, "mutation"),
            SetKillSwitch(value=True),
        )
    )
    fence = controls.dispatch(
        FenceSource(ControlIdentity("profile", "service", 2, "fence"), "configuration")
    )
    with pytest.raises(RuntimeError, match="fenced"):
        await mutation.applied
    assert await mutation.retired is True
    await fence.applied
    assert not controls.admitted_effects
    assert not controls._permissions._unresolved
    assert await asyncio.to_thread(case.sources.control_plane.get_kill_switch) is False


@pytest.mark.parametrize("domain_case", ["textual"], indirect=True)
async def test_reserved_commit_reuses_original_outer_acceptance_owner(domain_case):
    from tldw_chatbook.Chat.console_native_commit import (
        commit_reserved_durable_turn_owned,
        reserve_durable_commit_plan_owned,
    )

    case = domain_case
    async with prepared(case) as (acceptance, record):
        owner = case.controller._register_ordinary_native_commit(
            case.session, acceptance
        )
        try:
            plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
            assert record.task is asyncio.current_task()
            completion = await commit_reserved_durable_turn_owned(
                plan, _native_owner=owner
            )
            assert completion.error is None and completion.commit is not None
            assert completion.native_retired is True
            assert (
                case.controller._ordinary_native_commit_owner(acceptance.preparation_id)
                is owner
            )
            assert not owner.retired.done()
            assert owner.task is record.task
        finally:
            case.controller._release_ordinary_native_commit(owner)
        assert owner.retired.done()


@pytest.mark.parametrize("domain_case", ["textual"], indirect=True)
@pytest.mark.parametrize("changed", ["issuer", "acceptance", "task", "registration"])
async def test_reserved_commit_refuses_foreign_outer_owner(domain_case, changed):
    from dataclasses import replace

    from tldw_chatbook.Chat.console_native_commit import (
        commit_reserved_durable_turn_owned,
        reserve_durable_commit_plan_owned,
    )

    case = domain_case
    async with prepared(case) as (acceptance, _record):
        owner = case.controller._register_ordinary_native_commit(
            case.session, acceptance
        )
        original = (owner._issuer, owner.acceptance, owner.task)
        try:
            plan = await reserve_durable_commit_plan_owned(case.store, acceptance)
            if changed == "issuer":
                owner._issuer = object()
            elif changed == "acceptance":
                owner.acceptance = replace(acceptance)
            elif changed == "task":
                owner.task = None
            else:
                case.controller._ordinary_native_commits.pop(acceptance.preparation_id)
            completion = await commit_reserved_durable_turn_owned(
                plan, _native_owner=owner
            )
            assert isinstance(completion.error, RuntimeError)
            assert str(completion.error) == "Separated plan native owner changed."
            assert completion.commit is None
            assert not owner.retired.done()
            assert (
                case.store.durable_turn_commit_for(
                    acceptance.preparation_id, fingerprint=plan.fingerprint
                )
                is None
            )
        finally:
            owner._issuer, owner.acceptance, owner.task = original
            case.controller._ordinary_native_commits[acceptance.preparation_id] = owner
            case.controller._release_ordinary_native_commit(owner)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_local_receive_claim_precedes_later_edit_and_cancel(domain_case):
    from Tests.Chat.test_console_domain_boundary import _selection
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
    )

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile-one", service_epoch="service-one"
    )

    def identity(sequence):
        return ControlIdentity(
            "profile-one",
            "service-one",
            sequence,
            f"command-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    first = controls.dispatch(UpdateInputs(identity(1), "pressed", (1, 1)))
    assert first.applied.done()
    inputs = case.store.session_input_snapshot(case.session.id)
    received = controls.dispatch(
        ReceiveTurn(identity(2), "turn-one", inputs.draft_revision, _selection(case))
    )
    assert received.applied.done()
    record = case.runtime._turn_custody["turn-one"]
    assert record.task is not None and not record.task.done()
    assert record.received_claim is not None and record.request is None
    assert record.received_intent.inputs.draft == "pressed"
    controls.dispatch(UpdateInputs(identity(3), "later edit", (1, 2)))
    assert case.session.draft == "later edit"
    assert record.received_intent.inputs.draft == "pressed"
    cancelled = controls.dispatch(CancelTurn(identity(4), "turn-one"))
    assert cancelled.applied.done()
    await asyncio.gather(record.task, return_exceptions=True)
    assert case.session.draft == "later edit"


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_selected_receive_waits_for_exact_plan_ack_without_native_lease(
    domain_case,
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        CommitPlanAck,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = domain_case
    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "history.jsonl"
    )
    case.sources.configuration["api_settings"] = {
        "openai": {"api_key": "task60-local-readiness-only"}
    }
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(
            provider="openai", model="gpt-4o-mini", base_url="http://127.0.0.1:1/v1"
        ),
    )
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            False
        )
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile-one", service_epoch="service-one"
    )

    def identity(sequence):
        return ControlIdentity(
            "profile-one",
            "service-one",
            sequence,
            f"ack-command-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    import sys
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    refusal_sites = []
    monitoring = sys.monitoring
    monitor_id = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(monitor_id, "task60-original-refusal")

    def refused(code, offset, exception):
        if isinstance(exception, RecoveryRequired):
            frame = sys._getframe(1)
            source = frame.f_locals.get("source")
            mismatch = None
            if source is not None and hasattr(source, "config_identity"):
                mismatch = (
                    source.config_identity != source.config.current_config_identity()
                )
            refusal_sites.append((code.co_filename, code.co_name, offset, mismatch))

    monitoring.register_callback(monitor_id, monitoring.events.RAISE, refused)
    monitoring.set_events(monitor_id, monitoring.events.RAISE)
    controls.dispatch(UpdateInputs(identity(1), "hold this exact plan", (1, 1)))
    received = controls.dispatch(
        ReceiveTurn(
            identity(2), "ack-turn", case.session.draft_revision, _selection(case)
        )
    )
    record = case.runtime._turn_custody["ack-turn"]
    task = received.work
    assert task is record.task
    try:
        assert await _until(lambda: controls.commit_plans or task.done())
        if task.done():
            print("Task60 original refusal sites:", refusal_sites)
            print(
                "Task60 original refusal reason:",
                getattr(task.exception(), "reason", None),
            )
            task.result()
        assert len(controls.commit_plans) == 1
        planned = controls.commit_plans[0]
        assert planned.received_request_id == "ack-command-2"
        assert planned.turn_id == "ack-turn" and planned.ordinal == 1
        assert planned.recovery.fingerprint == planned.fingerprint
        assert record.task is received.work and not task.done()
        owner = case.controller._ordinary_native_commit_owner(
            planned.fingerprint.preparation_id
        )
        assert owner is not None and owner.task is record.task
        assert (
            case.store.durable_turn_commit_for(
                planned.fingerprint.preparation_id, fingerprint=planned.fingerprint
            )
            is None
        )
        assert not controls.admitted_effects
        controls.dispatch(CancelTurn(identity(3), "ack-turn"))
        await asyncio.gather(task, return_exceptions=True)
        with pytest.raises(RuntimeError, match="no longer pending"):
            controls.dispatch(
                CommitPlanAck(
                    identity(4),
                    planned.received_request_id,
                    planned.turn_id,
                    planned.ordinal,
                    planned.fingerprint,
                )
            )
        assert (
            case.store.durable_turn_commit_for(
                planned.fingerprint.preparation_id, fingerprint=planned.fingerprint
            )
            is None
        )
    finally:
        monitoring.set_events(monitor_id, 0)
        monitoring.register_callback(monitor_id, monitoring.events.RAISE, None)
        monitoring.free_tool_id(monitor_id)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
