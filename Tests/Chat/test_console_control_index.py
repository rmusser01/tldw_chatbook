"""Actual selected index stages; mechanism coverage, not Settings dispatch."""

from __future__ import annotations

import asyncio
import contextvars
import sys
import threading
from contextlib import ExitStack

import pytest

from Tests.Chat.test_console_control_trace import _Journey, _observe
from Tests.Chat.test_console_domain_boundary import (
    _selection,
    domain_case as _domain_case,
)
from Tests.Chat.test_console_first_send_atomicity import _until

domain_case = _domain_case
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def _received_index_source(case):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ConsoleLocalControls,
        ReceiveTurn,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "index-history.jsonl"
    )
    case.sources.configuration["api_settings"] = {
        "openai": {"api_key": "task60-local-readiness-only"}
    }
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(
            provider="openai",
            model="gpt-4o-mini",
            base_url="http://127.0.0.1:1/v1",
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
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile",
        service_epoch="service",
    )
    journey = _Journey(case, controls)
    contexts = []

    def capture(frame):
        if frame.f_locals["self"] is controls:
            contexts.append(contextvars.copy_context())

    with _observe(
        ConsoleLocalControls.commit_received_turn.__code__,
        sys.monitoring.events.PY_START,
        capture,
    ):
        updated = controls.dispatch(
            UpdateInputs(journey.identity(), "Hold index source", (1, 1))
        )
        receipt = controls.dispatch(
            ReceiveTurn(
                journey.identity(),
                "index-source",
                updated.applied.result(),
                _selection(case),
            )
        )
        assert await _until(lambda: controls.commit_plans or receipt.work.done())
        assert controls.commit_plans, (
            repr(receipt.work.result()) if receipt.work.done() else "pending"
        )
    assert len(contexts) == 1
    # No ACK: this actual received turn remains owned without provider dispatch.
    return journey, receipt, contexts[0]


@pytest.mark.parametrize("domain_case", ["control_index"], indirect=True)
@pytest.mark.parametrize("phase", ["fence", "writer", "supervisor"])
async def test_index_finite_stage_and_original_physical_owners(domain_case, phase):
    from tldw_chatbook.Chat.console_control_dispatch import (
        Drain,
        FenceSource,
        _command_source,
    )
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.runtime import _project_index_key_owned
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    case = domain_case
    journey, receipt, context = await _received_index_source(case)
    source = context.get(_command_source)
    assert source.controls is journey.controls and source.sources is case.sources
    entered, release = threading.Event(), threading.Event()
    commits = []
    retained = {}

    def setup():
        owner = case.domain.project_index_owner()
        assert owner.service is None
        service = owner.get()
        assert owner is case.domain._project_index_owner
        key = _project_index_key_owned(
            service, "index-control", case.index_binding.binding_id, case.index_root
        )
        status = service.status(key)
        enabled = service.enable(
            key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=status.scope
        )
        retained.update(owner=owner, service=service, key=key)
        job_id = service.refresh(
            key,
            expected_scope=enabled.scope,
            expected_job_revision=enabled.job_revision,
        )
        retained["run"] = service._runs[job_id]

    def hold_unit(frame):
        if (
            phase != "supervisor"
            and frame.f_locals["self"] is retained["service"]._scanner
        ):
            entered.set()
            assert release.wait(10), "Original scanner unit not released"

    def hold_supervisor(frame):
        if phase == "supervisor" and frame.f_locals["self"] is retained["service"]:
            entered.set()
            assert release.wait(10), "Original supervisor not released"

    def observe_commit(frame):
        commits.append(frame.f_locals["self"])

    try:
        with ExitStack() as observations:
            observations.enter_context(
                _observe(
                    ProjectScanner._unit.__code__,
                    sys.monitoring.events.PY_START,
                    hold_unit,
                )
            )
            observations.enter_context(
                _observe(
                    ProjectIndexService._retire.__code__,
                    sys.monitoring.events.PY_RETURN,
                    hold_supervisor,
                )
            )
            observations.enter_context(
                _observe(
                    WorkspaceProjectIndexDB.commit_scan_unit.__code__,
                    sys.monitoring.events.PY_START,
                    observe_commit,
                )
            )
            await asyncio.to_thread(context.run, setup)
            assert await _until(entered.is_set)
            run, service = retained["run"], retained["service"]
            assert run.control_source is source
            assert not any(
                name.startswith("index_") for name in journey.controls.admitted_effects
            )
            if phase == "supervisor":
                assert run.native_retired and run.pin.closed and run.lease.closed
                assert service._scanner.retired and run.supervisor.is_alive()
                assert commits
            else:
                assert (
                    not run.native_retired
                    and not run.pin.closed
                    and not run.lease.closed
                )
                assert run.writer.is_alive() and not service._scanner.retired
                assert not commits
            if phase == "fence":
                await journey.controls.dispatch(
                    FenceSource(journey.identity(), "configuration")
                ).applied
                release.set()
                assert await _until(lambda: run.native_retired)
                assert not commits, "Old-source emit entered after its source fence"
            drain = journey.controls.dispatch(Drain(journey.identity()))
            if phase != "fence":
                await asyncio.sleep(0)
                assert not drain.applied.done()
                release.set()
            assert await drain.applied is True
            assert (
                run.native_retired
                and not run.writer.is_alive()
                and not run.supervisor.is_alive()
            )
            assert run.pin.closed and run.lease.closed and service._scanner.retired
            assert run.control_connections
            assert retained["owner"].closed
    finally:
        release.set()
        receipt.work.cancel()
        await asyncio.gather(receipt.work, return_exceptions=True)
        await case.domain._project_index_owner.aclose()


@pytest.mark.parametrize("domain_case", ["control_index"], indirect=True)
async def test_selected_detached_cleanup_uncertainty_retains_original_effect(
    domain_case,
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_control_dispatch import Drain
    from tldw_chatbook.Workspaces.project_index.lifetime import (
        BindingLifetime,
        LifetimeBusy,
    )
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexPolicy,
        ProjectIndexRetirementUnproven,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _project_index_key_owned

    case = domain_case
    owner = case.domain.project_index_owner()
    prepared = {}

    def build():
        # Original idle-ready composition precedes control selection. No active
        # unrecorded run is adopted by the selected source boundary.
        service = owner.get()
        key = _project_index_key_owned(
            service, "index-control", case.index_binding.binding_id, case.index_root
        )
        status = service.status(key)
        enabled = service.enable(
            key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=status.scope
        )
        job = service.refresh(
            key,
            expected_scope=enabled.scope,
            expected_job_revision=enabled.job_revision,
        )
        prepared.update(service=service, run=service._runs[job])

    await asyncio.to_thread(build)
    service = prepared["service"]
    assert await _until(lambda: prepared["run"].native_retired)

    def detach():
        service._unsubscribe()
        with service.registry_connection_scope():
            case.sources.workspace_registry.remove_runtime_binding(
                case.index_binding.binding_id
            )

    await asyncio.to_thread(detach)
    assert not service._operations and service._cleanup_thread is None
    journey, receipt, context = await _received_index_source(case)
    assert owner._console_controls is journey.controls
    failed, repair = threading.Event(), threading.Event()
    observed, errors = [], []

    def deny_receipt_close(frame):
        lease = frame.f_locals["self"]
        if lease.path.name == "cleanup-owner-v1.lock" and not repair.is_set():
            observed.append((lease, service._operation_local.operation))
            raise OSError("task60 original native receipt close refused")

    def cleanup():
        try:
            context.run(service.retry_detached_cleanup)
        except ProjectIndexRetirementUnproven as error:
            errors.append(error)
        finally:
            failed.set()
            assert repair.wait(10), "Original cleanup owner repair not released"
            # Same original creator closes its retained physical capabilities;
            # uncertainty records remain and are never changed into success.
            for operation in tuple(service._operations):
                for capability in operation.retained:
                    capability.close()

    task = None
    try:
        with _observe(
            BindingLifetime.close.__code__,
            sys.monitoring.events.PY_START,
            deny_receipt_close,
        ):
            task = asyncio.create_task(asyncio.to_thread(cleanup))
            assert await _until(failed.is_set)
            assert errors and len(observed) == 1
            lease, operation = observed[0]
            assert not lease.closed and operation in service._operations
            assert operation.done.is_set() and operation.retirement_unproven
            assert lease in operation.retained
            pins = [
                item
                for item in operation.retained
                if getattr(item, "factory", None) is service._factory
            ]
            assert len(pins) == 1 and not pins[0].closed
            with storage._lock:
                assert pins[0].lease in storage._live_leases
            with pytest.raises(LifetimeBusy):
                BindingLifetime.acquire(lease.path, existing_only=True)
            effects = [
                effect
                for effect in journey.controls._effects
                if effect.owner is operation
            ]
            assert len(effects) == 1 and effects[0].kind == "index_operation"
            assert (
                effects[0].completed.result() is False
                and not effects[0].physically_retired
            )
            with pytest.raises(
                ProjectIndexRetirementUnproven,
                match="index_operation_retirement_unproven",
            ):
                await journey.controls.dispatch(Drain(journey.identity())).applied
            assert not lease.closed and not pins[0].closed and not task.done()
    finally:
        repair.set()
        if task is not None:
            await asyncio.shield(task)
        receipt.work.cancel()
        await asyncio.gather(receipt.work, return_exceptions=True)
    assert lease.closed and pins[0].closed
    assert operation.retirement_unproven and effects[0].completed.result() is False
