"""Causal controls on the actual selected Console graph and original owners."""

from __future__ import annotations

import asyncio
import sys
import threading
import tomllib
from dataclasses import fields

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case
from Tests.Chat.test_console_first_send_atomicity import _until

domain_case = _domain_case
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _controls(case):
    return case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )


def _identity(case, sequence, *, revision=0, session=True):
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity

    return ControlIdentity(
        "profile",
        "service",
        sequence,
        f"causal-{sequence}",
        case.session.id if session else "",
        case.session.incarnation_id if session else "",
        revision,
    )


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_new_chat_uses_original_controller_and_rejects_changed_origin(domain_case):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_control_dispatch import OpenSession

    case = domain_case
    controls = _controls(case)
    observed = []
    monitoring = sys.monitoring
    slot = next(value for value in range(6) if monitoring.get_tool(value) is None)
    code = ConsoleChatController.new_session.__code__
    monitoring.use_tool_id(slot, "task62-new-chat")

    def observe(_code, *_args):
        frame = sys._getframe(1)
        if frame.f_locals.get("self") is case.controller:
            observed.append(dict(frame.f_locals))

    monitoring.register_callback(slot, monitoring.events.PY_START, observe)
    monitoring.set_local_events(slot, code, monitoring.events.PY_START)
    try:
        receipt = controls.dispatch(OpenSession(
            _identity(case, 1), workspace_id=case.session.workspace_id,
            new_chat=True, ephemeral=True, new_chat_default_generation=3,
        ))
        session_id, incarnation_id = await receipt.applied
        current = next(row for row in case.store.sessions() if row.id == session_id)
        assert current.incarnation_id == incarnation_id
        assert case.store.active_session_id == session_id
        assert current.ephemeral and current.new_chat_default_generation == 3
        assert len(observed) == 1
        assert observed[0]["settings"] is observed[0]["canonical_settings_baseline"]
        assert observed[0]["workspace_id"] == case.session.workspace_id
        with pytest.raises(RuntimeError, match="origin changed"):
            controls.dispatch(OpenSession(_identity(case, 2), new_chat=True))
        assert len(case.store.sessions()) == 2 and len(observed) == 1
    finally:
        monitoring.set_local_events(slot, code, 0)
        monitoring.register_callback(slot, monitoring.events.PY_START, None)
        monitoring.free_tool_id(slot)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_original_configuration_publication_rolls_only_configuration(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import FenceSource, UpdateInputs
    from tldw_chatbook.Chat.console_control_publication import (
        ApplySourceRevision,
        RefreshConfiguration,
    )

    case = domain_case
    controls = _controls(case)
    original = case.domain.sources
    fence = await controls.dispatch(
        FenceSource(_identity(case, 1), "configuration")
    ).applied
    publication = await controls.dispatch(
        RefreshConfiguration(_identity(case, 2, revision=1), fence)
    ).applied
    assert publication.previous_sources is original
    assert publication.view == "normalized_settings"
    assert (
        publication.sources.configuration is publication.original_result.settings_view
    )
    assert case.domain.sources is original  # Original publication still needs Apply.
    assert not controls.admitted_effects
    await controls.dispatch(
        ApplySourceRevision(_identity(case, 3, revision=1), publication)
    ).applied
    successor = case.domain.sources
    assert successor is publication.sources and successor is not original
    for item in fields(original):
        if item.name != "configuration":
            assert getattr(successor, item.name) is getattr(original, item.name)
    with pytest.raises(RuntimeError):
        case.domain.require_current(original)
    with pytest.raises(RuntimeError, match="fenced"):
        controls.dispatch(UpdateInputs(_identity(case, 4), "old", (1, 1)))
    assert controls.dispatch(
        UpdateInputs(_identity(case, 5, revision=1), "new", (1, 2))
    ).applied.done()


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_original_invalid_configuration_stays_fenced_after_native_close(
    domain_case,
):
    from tldw_chatbook.Chat.console_control_dispatch import FenceSource, UpdateInputs
    from tldw_chatbook.Chat.console_control_publication import ReplaceConfiguration

    case = domain_case
    controls = _controls(case)
    original = case.domain.sources
    fence = await controls.dispatch(
        FenceSource(_identity(case, 1), "configuration")
    ).applied
    failed = controls.dispatch(
        ReplaceConfiguration(_identity(case, 2, revision=1), fence, "[invalid")
    )
    with pytest.raises(tomllib.TOMLDecodeError):
        await failed.applied
    assert case.domain.sources is original
    assert not controls._intake_open and not controls.admitted_effects
    with pytest.raises(RuntimeError, match="fenced"):
        controls.dispatch(
            UpdateInputs(_identity(case, 3, revision=1), "refused", (1, 1))
        )


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_actual_question_owner_waits_at_zero_timeout_and_rejects_late_answer(
    domain_case,
):
    from tldw_chatbook.Agents.ask_user_questions import validate_questions
    from tldw_chatbook.Chat.console_control_decisions import QuestionDecision
    from tldw_chatbook.Chat.console_control_dispatch import ResolveDecision

    case = domain_case
    controls = _controls(case)
    host = case.controller._interrupt_host
    # Original explicit Buddy-opener capability on this exact live session.
    host.retain_decision_target(case.session.id)
    assert host._resolve_ask_user_timeout_seconds() == 0
    questions = validate_questions(
        [
            {
                "question": "Choose a result",
                "header": "Choice",
                "options": [{"label": "One"}, {"label": "Two"}],
            }
        ]
    )
    worker = asyncio.create_task(
        asyncio.to_thread(
            case.controller.request_user_questions,
            questions,
            session_id=case.session.id,
        )
    )
    try:
        assert await _until(lambda: host.pending_question_ids() or worker.done())
        assert not worker.done()
        request_id = host.pending_question_ids()[0]
        with host.lock:
            state = host.registries["question"][request_id]
            assert not state["event"].is_set()
        wrong = controls.dispatch(
            ResolveDecision(_identity(case, 1), QuestionDecision("different", []))
        )
        assert wrong.applied.result() is False and not worker.done()
        answers = [{"question": "Choose a result", "selected": ["One"]}]
        controls.dispatch(
            ResolveDecision(_identity(case, 2), QuestionDecision(request_id, answers))
        )
        result = await asyncio.wait_for(asyncio.shield(worker), 10)
        assert result["answered"] and result["answers"][0]["selected"] == ["One"]
        late = controls.dispatch(
            ResolveDecision(_identity(case, 3), QuestionDecision(request_id, answers))
        )
        assert late.applied.result() is False
        assert request_id not in host.registries["question"]
    finally:
        case.controller._signal_stop(session_id=case.session.id)
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
async def test_saved_open_cancel_keeps_fence_waiting_for_original_read_thread(
    domain_case,
):
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_control_dispatch import FenceSource, OpenSession

    case = domain_case
    local = case.sources.conversation_scope_service.local_service
    target = local.create_conversation(
        title="Held read", runtime_backend="local", scope_type="global"
    )
    controls = _controls(case)
    entered, release = threading.Event(), threading.Event()
    code = ChatConversationService.get_conversation_tree.__code__
    monitoring = sys.monitoring
    slot = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(slot, "task60-saved-read")

    def observe(_code, *_args):
        if sys._getframe(1).f_locals.get("self") is local:
            entered.set()
            assert release.wait(10), "Original saved read was not released"

    monitoring.register_callback(slot, monitoring.events.PY_START, observe)
    monitoring.set_local_events(slot, code, monitoring.events.PY_START)
    receipt = controls.dispatch(
        OpenSession(_identity(case, 1, session=False), conversation_id=target)
    )
    try:
        assert await _until(entered.is_set)
        receipt.work.cancel()
        await asyncio.sleep(0)
        fence = controls.dispatch(FenceSource(_identity(case, 2), "configuration"))
        await asyncio.sleep(0)
        assert not receipt.work.done() and not fence.applied.done()
        assert "saved_open" in controls.admitted_effects
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await receipt.applied
        await fence.applied
        assert not controls.admitted_effects
        assert len(case.store.sessions()) == 1
    finally:
        release.set()
        await asyncio.gather(receipt.work, return_exceptions=True)
        monitoring.set_local_events(slot, code, 0)
        monitoring.register_callback(slot, monitoring.events.PY_START, None)
        monitoring.free_tool_id(slot)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_viewless_whole_projection_is_explicitly_unavailable(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import ReadProjection

    controls = _controls(domain_case)
    receipt = controls.dispatch(ReadProjection(_identity(domain_case, 1)))
    with pytest.raises(RuntimeError, match="actual attached view"):
        await receipt.applied


@pytest.mark.parametrize("domain_case", ["control_index"], indirect=True)
async def test_original_workspace_publication_retains_registry(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import FenceSource, UpdateInputs
    from tldw_chatbook.Chat.console_control_publication import (
        ApplySourceRevision,
        SetWorkspaceBindingAccess,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )

    case = domain_case
    controls = _controls(case)
    original = case.domain.sources
    registry = original.workspace_registry
    binding_id = case.index_binding.binding_id
    fence = await controls.dispatch(
        FenceSource(_identity(case, 1), "workspace")
    ).applied
    publication = await controls.dispatch(
        SetWorkspaceBindingAccess(
            _identity(case, 2, revision=1), fence, binding_id, True
        )
    ).applied
    assert publication.sources is original
    assert publication.binding.binding_id == binding_id
    assert publication.binding.metadata["access"] == "rw"

    def persisted_binding():
        with project_index_registry_connections(registry.db):
            return registry.get_runtime_binding(binding_id)

    persisted = await asyncio.to_thread(persisted_binding)
    assert persisted.metadata["access"] == "rw"
    assert not controls.admitted_effects
    assert not controls._intake_open
    assert (
        await controls.dispatch(
            ApplySourceRevision(_identity(case, 3, revision=1), publication)
        ).applied
        == 1
    )
    assert case.domain.sources is original
    assert case.domain.sources.workspace_registry is registry
    with pytest.raises(RuntimeError, match="fenced"):
        controls.dispatch(UpdateInputs(_identity(case, 4), "old", (1, 1)))
    assert controls.dispatch(
        UpdateInputs(_identity(case, 5, revision=1), "current", (1, 2))
    ).applied.done()


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
@pytest.mark.parametrize("failure", ["postcommit", "identity"])
async def test_committed_configuration_publication_failure_stays_fenced(
    domain_case, failure
):
    import inspect

    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_control_dispatch import FenceSource, UpdateInputs
    from tldw_chatbook.Chat.console_control_publication import (
        ApplySourceRevision,
        RefreshConfiguration,
        ReplaceConfiguration,
    )

    case = domain_case
    controls = _controls(case)
    original = case.domain.sources
    before = await asyncio.to_thread(config.read_cli_config_snapshot)
    serialized = (
        before.serialized or ""
    ) + '\n[task60_publication_evidence]\nvalue = "committed"\n'
    fence = await controls.dispatch(
        FenceSource(_identity(case, 1), "configuration")
    ).applied
    try:
        if failure == "postcommit":
            observed = []

            def fail_publication(frame):
                if frame.f_locals["raw_config"].get("task60_publication_evidence"):
                    observed.append(threading.get_ident())
                    raise RuntimeError("Original postcommit publication failed")

            with _observe(
                inspect.unwrap(config._publish_runtime_config_unlocked).__code__,
                sys.monitoring.events.PY_START,
                fail_publication,
            ):
                failed = controls.dispatch(
                    ReplaceConfiguration(
                        _identity(case, 2, revision=1),
                        fence,
                        serialized,
                        expected_snapshot=before,
                        create_backup=False,
                    )
                )
                with pytest.raises(config.ConfigPostCommitError) as caught:
                    await failed.applied
            assert observed and observed[0] != threading.get_ident()
            outcome = caught.value
            committed = await asyncio.to_thread(config.read_cli_config_snapshot)
            assert outcome.snapshot == committed
            assert (
                tomllib.loads(committed.serialized)["task60_publication_evidence"][
                    "value"
                ]
                == "committed"
            )
            assert outcome.backup_path is None
            assert not controls._publications
        else:
            publication = await controls.dispatch(
                RefreshConfiguration(_identity(case, 2, revision=1), fence)
            ).applied
            await asyncio.to_thread(
                config.replace_cli_config_snapshot,
                serialized,
                before,
                create_backup=False,
            )
            with pytest.raises(RuntimeError, match="publication identity changed"):
                controls.dispatch(
                    ApplySourceRevision(_identity(case, 3, revision=1), publication)
                )
            assert publication in controls._publications
            assert (
                publication.sources.configuration
                is publication.original_result.settings_view
            )
        assert case.domain.sources is original
        assert not controls._intake_open and not controls.admitted_effects
        with pytest.raises(RuntimeError, match="fenced"):
            controls.dispatch(
                UpdateInputs(_identity(case, 4, revision=1), "refused", (1, 1))
            )
    finally:
        current = await asyncio.to_thread(config.read_cli_config_snapshot)
        await asyncio.to_thread(
            config.replace_cli_config_snapshot,
            before.serialized or "",
            current,
            create_backup=False,
        )


@pytest.mark.loopback_network
@pytest.mark.parametrize("domain_case", ["control_trace"], indirect=True)
async def test_existing_conversation_archive_change_during_original_read(domain_case):
    import sqlite3

    from Tests.Chat.test_console_control_trace import _observe, _prepare
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_refused_stream_retry import _server
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_control_dispatch import (
        Drain,
        ReceiveTurn,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_control_sessions import _saved_read
    from tldw_chatbook.Chat.console_control_trace import drain_trace_worker
    from tldw_chatbook.Chat.console_runtime import _ConsoleTurnRefusedError
    from tldw_chatbook.Chat.conversation_archive_actions import (
        change_conversation_archive,
    )

    case = domain_case
    local = case.sources.conversation_scope_service.local_service
    assert (
        case.parent.chat_conversation_scope_service
        is case.sources.conversation_scope_service
    )
    other = local.create_conversation(
        title="Independent archive target", runtime_backend="local", scope_type="global"
    )
    version = local.get_conversation_metadata(other)["version"]
    entered, release = threading.Event(), threading.Event()
    bindings = []

    def hold_read(frame):
        if frame.f_locals["self"] is local:
            binding = _saved_read.get()
            assert binding is not None and binding.database is case.database
            bindings.append(binding)
            entered.set()
            assert release.wait(10), "Original archive read was not released"

    with _server() as server:
        journey = await _prepare(case, server)
        first = await journey.start()
        assert (await asyncio.wait_for(asyncio.shield(first.work), 20)).accepted
        await drain_trace_worker(case.store)
        conversation_id = case.store._sessions[
            case.session.id
        ].persisted_conversation_id
        assert conversation_id and conversation_id != other
        original_requests = len(server.payloads)
        original_generation = getattr(
            case.parent, "_conversation_archive_generation", 0
        )
        with _observe(
            ChatConversationService.get_conversation_archive_states.__code__,
            sys.monitoring.events.PY_RETURN,
            hold_read,
        ):
            updated = journey.controls.dispatch(
                UpdateInputs(journey.identity(), "Preserve this draft", (2, 1))
            )
            receipt = journey.controls.dispatch(
                ReceiveTurn(
                    journey.identity(),
                    "archive-race",
                    updated.applied.result(),
                    _selection(case),
                )
            )
            try:
                assert await _until(lambda: entered.is_set() or receipt.work.done())
                assert entered.is_set(), (
                    repr(receipt.work.result()) if receipt.work.done() else "pending"
                )
                assert "archive_read" in journey.controls.admitted_effects
                binding = bindings[0]
                assert not binding.turn_owner.retired
                changed = await change_conversation_archive(
                    case.parent,
                    [other],
                    archived=True,
                    expected_versions={other: version},
                )
                assert other in changed["changed"] and not changed["failures"]
                assert (
                    case.parent._conversation_archive_generation
                    == original_generation + 1
                )
                assert case.parent._conversation_archive_states[other] is True
                release.set()
                with pytest.raises(_ConsoleTurnRefusedError) as refused:
                    await asyncio.wait_for(asyncio.shield(receipt.work), 10)
                assert refused.value.reason == (
                    "Wait for the archive change to finish; your draft is preserved."
                )
                assert (
                    case.store.session_draft(case.session.id) == "Preserve this draft"
                )
                assert len(server.payloads) == original_requests
                # Generation changed after the actual read: no stale cache publication.
                assert conversation_id not in case.parent._conversation_archive_states
                assert binding.turn_owner.retired and binding.retirement.confirmed
                assert binding.retirement.borrowed is None
                assert binding.retirement.connection is not None
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    binding.retirement.connection.execute("SELECT 1")
                assert not journey.controls.admitted_effects
                assert await journey.controls.dispatch(
                    Drain(journey.identity())
                ).applied
            finally:
                release.set()
                await asyncio.gather(receipt.work, return_exceptions=True)


@pytest.mark.loopback_network
@pytest.mark.parametrize("domain_case", ["control_trace"], indirect=True)
async def test_archive_parent_checkpoints_finish_only_after_original_native_retirement(
    domain_case, request
):
    import pytest_timeout
    import sqlite3

    from tldw_chatbook.Chat.console_presentation_port import ConsolePresentationPort

    from Tests.Chat.test_console_control_trace import _observe, _prepare
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_refused_stream_retry import _server
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_control_dispatch import (
        Drain,
        ReceiveTurn,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_control_sessions import _saved_read
    from tldw_chatbook.Chat.console_control_trace import drain_trace_worker
    from tldw_chatbook.Chat.console_runtime import _ConsoleTurnRefusedError
    from tldw_chatbook.Chat.conversation_archive_actions import (
        change_conversation_archive,
    )

    case = domain_case
    local = case.sources.conversation_scope_service.local_service
    assert (
        case.parent.chat_conversation_scope_service
        is case.sources.conversation_scope_service
    )
    other = local.create_conversation(
        title="Independent archive target", runtime_backend="local", scope_type="global"
    )
    version = local.get_conversation_metadata(other)["version"]
    entered, release = threading.Event(), threading.Event()
    bindings = []
    checkpoints = []
    records = []
    order = []
    cache_after_change = []
    port = case.domain.presentation
    assert port._app is case.parent

    def inflight_return(frame, values):
        if frame.f_locals["self"] is not port:
            return
        record = frame.f_locals["record"]
        assert record.archive_conversation_id == conversation_id
        assert values[-1] is None
        assert conversation_id not in getattr(
            case.parent, "_conversation_archive_inflight", ()
        )
        assert not bindings and not checkpoints
        records.append(record)
        order.append("inflight")

    def capture_return(frame, values):
        if frame.f_locals["self"] is not port:
            return
        checkpoint = values[-1]
        assert frame.f_locals["record"] is records[0]
        assert checkpoint.port is port and checkpoint.record is records[0]
        assert checkpoint.association is port.association
        assert checkpoint.conversation_id == conversation_id
        assert checkpoint.generation == original_generation
        assert not checkpoint.consumed and not bindings
        checkpoints.append(checkpoint)
        order.append("capture")

    def finish_entry(frame):
        if frame.f_locals["self"] is not port:
            return
        assert len(bindings) == len(checkpoints) == len(records) == 1
        checkpoint = frame.f_locals["checkpoint"]
        assert checkpoint is checkpoints[0]
        assert frame.f_locals["record"] is records[0]
        assert checkpoint.association is port.association and not checkpoint.consumed
        binding = bindings[0]
        owner = binding.turn_owner
        assert owner.controls is journey.controls
        assert owner.reads and owner.workers  # Never let all([]) prove retirement.
        assert binding.retirement in owner.reads
        assert all(task.done() and not task.cancelled() for task in owner.workers)
        assert all(retirement.confirmed for retirement in owner.reads)
        assert owner.projection_retired and owner.retired
        assert binding.retirement.confirmed and binding.retirement.borrowed is None
        assert binding.retirement.connection is not None
        assert "archive_read" in journey.controls.admitted_effects
        assert case.parent._conversation_archive_generation == original_generation + 1
        assert case.parent._conversation_archive_states is cache_after_change[0]
        assert case.parent._conversation_archive_states[other] is True
        assert conversation_id not in case.parent._conversation_archive_states
        assert conversation_id not in getattr(
            case.parent, "_conversation_archive_inflight", ()
        )
        order.append("finish")

    def hold_read(frame):
        if frame.f_locals["self"] is local:
            binding = _saved_read.get()
            assert binding is not None and binding.database is case.database
            assert order == ["inflight", "capture"]
            bindings.append(binding)
            order.append("native-return")
            entered.set()
            assert release.wait(10), "Original archive read was not released"

    with _server() as server:
        journey = await _prepare(case, server)
        first = await journey.start()
        assert (await asyncio.wait_for(asyncio.shield(first.work), 20)).accepted
        await drain_trace_worker(case.store)
        conversation_id = case.store._sessions[
            case.session.id
        ].persisted_conversation_id
        assert conversation_id and conversation_id != other
        original_requests = len(server.payloads)
        original_generation = getattr(
            case.parent, "_conversation_archive_generation", 0
        )
        with (
            _observe(
                ConsolePresentationPort.archive_send_inflight.__code__,
                sys.monitoring.events.PY_RETURN,
                inflight_return,
                values=True,
            ),
            _observe(
                ConsolePresentationPort.capture_archive_send.__code__,
                sys.monitoring.events.PY_RETURN,
                capture_return,
                values=True,
            ),
            _observe(
                ConsolePresentationPort.finish_archive_send.__code__,
                sys.monitoring.events.PY_START,
                finish_entry,
            ),
            _observe(
                ChatConversationService.get_conversation_archive_states.__code__,
                sys.monitoring.events.PY_RETURN,
                hold_read,
            ),
        ):
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            updated = journey.controls.dispatch(
                UpdateInputs(journey.identity(), "Preserve this draft", (2, 1))
            )
            receipt = journey.controls.dispatch(
                ReceiveTurn(
                    journey.identity(),
                    "archive-race",
                    updated.applied.result(),
                    _selection(case),
                )
            )
            body_error = None
            try:
                assert await _until(lambda: entered.is_set() or receipt.work.done())
                assert entered.is_set(), (
                    repr(receipt.work.result()) if receipt.work.done() else "pending"
                )
                assert "archive_read" in journey.controls.admitted_effects
                binding = bindings[0]
                assert not binding.turn_owner.retired
                assert order == ["inflight", "capture", "native-return"]
                changed = await change_conversation_archive(
                    case.parent,
                    [other],
                    archived=True,
                    expected_versions={other: version},
                )
                assert other in changed["changed"] and not changed["failures"]
                assert (
                    case.parent._conversation_archive_generation
                    == original_generation + 1
                )
                assert case.parent._conversation_archive_states[other] is True
                cache_after_change.append(case.parent._conversation_archive_states)
                release.set()
                with pytest.raises(_ConsoleTurnRefusedError) as refused:
                    await asyncio.wait_for(asyncio.shield(receipt.work), 10)
                assert refused.value.reason == (
                    "Wait for the archive change to finish; your draft is preserved."
                )
                assert (
                    case.store.session_draft(case.session.id) == "Preserve this draft"
                )
                assert len(server.payloads) == original_requests
                # Generation changed after the actual read: no stale cache publication.
                assert conversation_id not in case.parent._conversation_archive_states
                assert order == ["inflight", "capture", "native-return", "finish"]
                assert checkpoints[0].consumed
                assert case.parent._conversation_archive_states is cache_after_change[0]
                assert case.parent._conversation_archive_states[other] is True
                assert binding.turn_owner.retired and binding.retirement.confirmed
                assert binding.retirement.borrowed is None
                assert binding.retirement.connection is not None
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    binding.retirement.connection.execute("SELECT 1")
                assert not journey.controls.admitted_effects
                assert await journey.controls.dispatch(
                    Drain(journey.identity())
                ).applied
            except BaseException as error:
                body_error = error
                raise
            finally:
                release.set()
                try:
                    async with asyncio.timeout(10):
                        await asyncio.shield(
                            asyncio.gather(receipt.work, return_exceptions=True)
                        )
                except BaseException:
                    if body_error is None:
                        raise
