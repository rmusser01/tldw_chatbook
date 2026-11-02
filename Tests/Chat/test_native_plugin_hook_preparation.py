"""Native custody follows real plugin admission and its original Console host."""

import asyncio
import threading
from dataclasses import replace
from uuid import uuid4

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from tldw_chatbook.Agents.execution_capacity import (
    ExecutionOwner,
    OwnedOperation,
    RuntimeCapacity,
    WorkOrigin,
    current_execution_owner,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified
from tldw_chatbook.Workspaces.worktree_activity import (
    WorktreeActivity,
    WorktreeFootprint,
)

hook_file = _hook_file
native_console = _native_console
native_package = _native_package
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native platform required"
    ),
]


async def until(probe):
    async with asyncio.timeout(5):
        while not probe():
            await asyncio.sleep(0.005)


def opaque_busy(runtime):
    with runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        return claim is None


@pytest.fixture
async def preparation(native_console, hook_file, monkeypatch):
    rig = native_console
    _save_hooks(hook_file, {})
    await rig.install()
    runtime = ConsoleRuntime(rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    entered, release = threading.Event(), threading.Event()
    owners = []
    check = rig.service._admission.check

    def held_check(*args, **kwargs):
        owners.append(current_execution_owner())
        entered.set()
        assert release.wait(10), "test failed to release actual plugin admission"
        return check(*args, **kwargs)

    monkeypatch.setattr(rig.service._admission, "check", held_check)
    rig.runtime, rig.entered, rig.release, rig.owners = (
        runtime,
        entered,
        release,
        owners,
    )
    rig.tasks = []
    try:
        yield rig
    finally:
        release.set()
        await asyncio.gather(*rig.tasks, return_exceptions=True)
        await runtime.dispose()


async def test_actual_plugin_admission_has_one_native_operation(preparation):
    rig = preparation
    task = asyncio.create_task(
        rig.controller.submit_draft("hello", session_id=rig.session.id)
    )
    rig.tasks.append(task)
    await until(rig.entered.is_set)
    assert opaque_busy(rig.runtime), "plugin admission ran outside native activity"
    snapshot = rig.runtime.execution_capacity.snapshot()
    assert snapshot.tool_workers == 1 and snapshot.model_lifelines == 0
    assert snapshot.child_executions == 0
    assert rig.owners[0] is not None
    assert rig.owners[0].origin is WorkOrigin.MANUAL
    rig.release.set()
    assert (await task).accepted
    assert not opaque_busy(rig.runtime)
    assert rig.runtime.execution_capacity.snapshot().tool_workers == 0


async def test_cancelled_preparation_keeps_original_admission_until_return(preparation):
    rig = preparation
    task = asyncio.create_task(
        rig.controller.submit_draft("hello", session_id=rig.session.id)
    )
    rig.tasks.append(task)
    await until(rig.entered.is_set)
    for _ in range(2):
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done(), "cancelled Future abandoned original plugin callback"
        assert opaque_busy(rig.runtime)
        assert rig.runtime.execution_capacity.snapshot().tool_workers == 1
    rig.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not rig.gateway.payloads
    assert not opaque_busy(rig.runtime)
    assert rig.runtime.execution_capacity.snapshot().tool_workers == 0
    assert len(rig.owners) == 1, "cancellation admitted a later plugin callback"


def configuration_for(rig):
    configuration = rig.controller.resolve_turn_configuration_snapshot(rig.session.id)
    maximum = dict(
        configuration.skill_context_maximum, plugin_run_id="pending:" + uuid4().hex
    )
    return replace(configuration, skill_context_maximum=maximum)


async def test_original_native_result_handoff_remains_owned(preparation, monkeypatch):
    rig = preparation
    configuration = configuration_for(rig)
    entered, release = asyncio.Event(), asyncio.Event()
    original = rig.service.hook_configuration

    async def held_handoff(maximum):
        result = await original(maximum)
        assert result.service is rig.service
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(rig.service, "hook_configuration", held_handoff)
    rig.release.set()
    task = asyncio.create_task(
        rig.runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
    )
    rig.tasks.append(task)
    try:
        await until(entered.is_set)
        assert opaque_busy(rig.runtime)
        assert rig.runtime.execution_capacity.snapshot().tool_workers == 1
        assert rig.service._thread.is_alive()
    finally:
        release.set()
    assert await task is None
    assert not opaque_busy(rig.runtime)
    assert rig.service._thread.is_alive(), (
        "finite admission must not retire plugin infrastructure"
    )


@pytest.mark.parametrize("close", ["runtime", "session"])
async def test_close_keeps_native_callback_and_storage_through_repeated_cancel(
    preparation,
    close,
):
    rig = preparation
    configuration = configuration_for(rig)
    task = asyncio.create_task(
        rig.runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
    )
    rig.tasks.append(task)
    await until(rig.entered.is_set)
    dispose = asyncio.create_task(
        rig.runtime.dispose(timeout_seconds=0)
        if close == "runtime"
        else rig.runtime.close_session(
            rig.session.id,
            expected_revision=rig.controller.lifecycle_impact(
                session_id=rig.session.id
            ).revision,
            timeout_seconds=0,
        )
    )
    rig.tasks.append(dispose)
    await until(
        lambda: (
            rig.runtime.execution_capacity.snapshot().closed
            if close == "runtime"
            else rig.session.id in rig.runtime._admission_fenced_sessions
        )
    )
    for _ in range(2):
        dispose.cancel()
        await asyncio.sleep(0.03)
        assert not dispose.done()
        assert opaque_busy(rig.runtime)
        assert rig.runtime._preparation_reads
        assert rig.service._thread.is_alive()
    with pytest.raises(RuntimeError):
        await rig.runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
    rig.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(asyncio.CancelledError):
        await dispose
    assert not rig.runtime._preparation_reads
    assert len(rig.owners) == 1
    assert not opaque_busy(rig.runtime)
    if close == "session":
        assert rig.session.id not in rig.store._sessions


@pytest.mark.parametrize("resource", ["capacity", "activity"])
async def test_first_use_original_resource_replacement_refuses_before_plugin_worker(
    preparation, monkeypatch, resource
):
    rig = preparation
    capacity = rig.runtime.execution_capacity
    activity = rig.runtime.worktree_activity
    permissions = rig.runtime.ensure_hook_permissions()
    original = permissions.v2_configuration
    entered, release = threading.Event(), threading.Event()

    def held_review():
        result = original()
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(permissions, "v2_configuration", held_review)
    task = asyncio.create_task(
        rig.controller.submit_draft("hello", session_id=rig.session.id)
    )
    rig.tasks.append(task)
    try:
        await until(entered.is_set)
        if resource == "capacity":
            rig.runtime._execution_capacity = RuntimeCapacity()
        else:
            rig.runtime._worktree_activity = WorktreeActivity()
    finally:
        release.set()
    assert not (await task).accepted
    assert not rig.entered.is_set()
    assert capacity.snapshot().tool_workers == 0
    with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
        assert claim is not None


@pytest.mark.parametrize("manual", [False, True])
async def test_only_original_manual_submission_can_use_reserved_capacity(
    preparation, manual
):
    rig = preparation
    rig.runtime._execution_capacity = RuntimeCapacity(
        max_tool_workers=1, reserved_manual_tool_workers=1
    )
    if manual:
        rig.release.set()
        assert (
            await rig.controller.submit_draft("hello", session_id=rig.session.id)
        ).accepted
        assert rig.owners[0].origin is WorkOrigin.MANUAL
    else:
        with pytest.raises(RuntimeError):
            # A caller-written initiator label cannot select manual capacity.
            await rig.runtime.prepare_hooks_v2(
                rig.session.id, initiator="manual", configuration=configuration_for(rig)
            )
        assert not rig.entered.is_set()
    assert rig.runtime.execution_capacity.snapshot().tool_workers == 0


async def test_activity_busy_is_original_visible_submission_refusal(preparation):
    rig = preparation
    with rig.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        assert claim is not None
        result = await rig.controller.submit_draft("hello", session_id=rig.session.id)
    assert not result.accepted
    assert not rig.entered.is_set()
    assert not rig.gateway.payloads
    assert not opaque_busy(rig.runtime)


async def test_no_plugin_rows_leave_native_resources_cold(preparation):
    rig = preparation
    configuration = replace(configuration_for(rig), skill_context_maximum={})
    assert (
        await rig.runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
        is None
    )
    assert rig.runtime._execution_capacity is None
    assert rig.runtime._worktree_activity is None
    assert not rig.entered.is_set()
    assert not rig.runtime._preparation_reads


async def test_unqualified_ordinary_preparation_uses_existing_path(
    preparation, monkeypatch
):
    from tldw_chatbook.Chat import console_workflow_runs

    rig = preparation
    configuration = configuration_for(rig)
    await rig.runtime.dispose()
    monkeypatch.setattr(
        console_workflow_runs, "native_workflow_qualified", lambda: False
    )
    runtime = ConsoleRuntime(rig.controller.app)
    runtime.set_chat_store(rig.store)
    # Keep the original live controller for source checks; the preceding dispose
    # sealed it, so this compatibility probe uses the real standalone service.
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    controller = ConsoleChatController(
        store=rig.store, provider_gateway=rig.gateway, skills_service=rig.skills
    )
    controller.app = rig.controller.app
    runtime.set_chat_controller(controller)
    rig.release.set()
    try:
        assert (
            await runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
            is None
        )
        assert rig.entered.is_set()
        assert all(owner is None for owner in rig.owners)
        assert runtime._execution_capacity is None
        assert runtime._worktree_activity is None
    finally:
        await runtime.dispose()


@pytest.mark.parametrize(
    "changed",
    [
        "revoke",
        "permission",
        "service",
        "maximum",
        "cancel",
        "infrastructure",
        "issuer",
        "snapshot",
    ],
)
async def test_change_during_native_reserve_cannot_enqueue_plugin_admission(
    preparation, monkeypatch, changed
):
    from types import SimpleNamespace

    from tldw_chatbook.Plugins.revocation import RevocationTarget

    rig = preparation
    original = ExecutionOwner.reserve_tool
    entered, release = threading.Event(), threading.Event()

    def held_reserve(owner):
        operation = original(owner)
        assert rig.runtime._run_hooks_lock.acquire(blocking=False)
        rig.runtime._run_hooks_lock.release()
        assert rig.service._lock.acquire(blocking=False)
        rig.service._lock.release()
        entered.set()
        assert release.wait(5)
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", held_reserve)
    configured = configuration_for(rig)
    maximum = dict(configured.skill_context_maximum)
    configuration = SimpleNamespace(skill_context_maximum=maximum)
    task = asyncio.create_task(
        rig.runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
    )
    rig.tasks.append(task)
    old_service = rig.local.plugin_service
    old_owner = rig.service._coordinator.owner
    old_issuer = rig.runtime._native_plugin_hook_preparation
    old_snapshots = rig.service._snapshots
    try:
        await until(entered.is_set)
        if changed == "revoke":
            identity = maximum["available_skills"][0]["plugin_installation_id"]
            request = rig.service.begin_disable(
                RevocationTarget(identity, "workspace-a", False)
            )
            await rig.service.finish_revocation(request)
        elif changed == "permission":
            _save_hooks(
                rig.runtime.ensure_hook_permissions()._published.config.config_path,
                {"enabled": False},
            )
            await asyncio.to_thread(
                rig.runtime.ensure_hook_permissions().v2_configuration
            )
        elif changed == "service":
            rig.local._plugin_service = object()
        elif changed == "maximum":
            maximum["available_skills"] = ()
        elif changed == "infrastructure":
            rig.service._coordinator.owner = object()
        elif changed == "issuer":
            rig.runtime._native_plugin_hook_preparation = None
        elif changed == "snapshot":
            rig.service._snapshots = dict(old_snapshots)
        else:
            task.cancel()
            await asyncio.sleep(0.02)
        release.set()
        rig.release.set()
        with pytest.raises((RuntimeError, PermissionError, asyncio.CancelledError)):
            await task
        assert not rig.entered.is_set(), (
            "changed source reached actual plugin admission"
        )
        assert not opaque_busy(rig.runtime)
        assert rig.runtime.execution_capacity.snapshot().tool_workers == 0
    finally:
        release.set()
        rig.local._plugin_service = old_service
        rig.service._coordinator.owner = old_owner
        rig.runtime._native_plugin_hook_preparation = old_issuer
        rig.service._snapshots = old_snapshots


async def test_original_operation_release_callback_return_keeps_claim_and_close_hold(
    preparation, monkeypatch
):
    rig = preparation
    original = OwnedOperation.finish
    entered, release = threading.Event(), threading.Event()

    def held_finish(operation):
        original(operation)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(OwnedOperation, "finish", held_finish)
    rig.release.set()
    task = asyncio.create_task(
        rig.runtime.prepare_hooks_v2(
            rig.session.id, configuration=configuration_for(rig)
        )
    )
    rig.tasks.append(task)
    try:
        await until(entered.is_set)
        assert opaque_busy(rig.runtime)
        assert rig.runtime._preparation_reads
        assert rig.runtime.execution_capacity.snapshot().tool_workers == 0
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert opaque_busy(rig.runtime)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    assert not opaque_busy(rig.runtime)
    assert not rig.runtime._preparation_reads


async def test_cancelled_service_bridge_is_not_original_callback_return(preparation):
    rig = preparation
    task = asyncio.create_task(
        rig.runtime.prepare_hooks_v2(
            rig.session.id, configuration=configuration_for(rig)
        )
    )
    rig.tasks.append(task)
    await until(rig.entered.is_set)
    read = next(
        read for read in rig.runtime._preparation_reads if read._producer is not None
    )
    try:
        for _ in range(2):
            read._producer.cancel()
            await asyncio.sleep(0.03)
            assert not task.done(), (
                "service Future cancellation released a running callback"
            )
            assert opaque_busy(rig.runtime)
            assert rig.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        rig.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not opaque_busy(rig.runtime)


async def test_original_host_check_cannot_accept_replaced_native_source(
    preparation, monkeypatch
):
    rig = preparation
    original = rig.runtime._require_hook_preparation_source
    original_native = rig.service.hook_configuration
    native_returned = asyncio.Event()
    replaced = []

    async def observed_native(maximum):
        result = await original_native(maximum)
        native_returned.set()
        return result

    monkeypatch.setattr(rig.service, "hook_configuration", observed_native)

    def replace_during_handoff(source):
        original(source)
        if not replaced and native_returned.is_set():
            replaced.append(rig.runtime._execution_capacity)
            rig.runtime._execution_capacity = RuntimeCapacity()

    monkeypatch.setattr(
        rig.runtime, "_require_hook_preparation_source", replace_during_handoff
    )
    rig.release.set()
    with pytest.raises(RuntimeError):
        await rig.runtime.prepare_hooks_v2(
            rig.session.id, configuration=configuration_for(rig)
        )
    assert len(replaced) == 1
    assert replaced[0].snapshot().tool_workers == 0
    assert not opaque_busy(rig.runtime)


async def test_blocked_installed_components_alone_create_no_preparation_effect(
    preparation,
):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    rig = preparation
    maximum = rig.service.capture_maximum("workspace-a")
    identity = maximum["available_skills"][0]["plugin_installation_id"]
    request = rig.service.begin_disable(
        RevocationTarget(identity, "workspace-a", False)
    )
    await rig.service.finish_revocation(request)
    configuration = configuration_for(rig)
    assert rig.service.list_components("workspace-a"), (
        "the installation must still exist"
    )
    assert not any(
        row.get("plugin_owned")
        for row in configuration.skill_context_maximum.get("available_skills", ())
    )
    assert (
        await rig.runtime.prepare_hooks_v2(rig.session.id, configuration=configuration)
        is None
    )
    assert not rig.entered.is_set()
    assert rig.runtime._execution_capacity is None
    assert rig.runtime._worktree_activity is None


@pytest.mark.parametrize("outcome", ["install", "revoke", "cancel"])
async def test_actual_engine_install_keeps_preparation_through_host_context_read(
    preparation, native_package, monkeypatch, outcome
):
    import json
    import sys

    rig = preparation
    rig.release.set()
    member = "io.github.rmusser01.chatbook/hooks.json"
    package = native_package(extension={"version": 1, "hooks": member})
    (package / member).parent.mkdir(parents=True, exist_ok=True)
    (package / member).write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "observe",
                        "event": "Stop",
                        "type": "command",
                        "effects": [],
                        "argv": [sys.executable, "-c", "pass"],
                    }
                ],
            }
        )
    )
    install = await rig.service.review_install(
        package, selection=("hook:observe",), workspace_id="workspace-a"
    )
    await rig.service.commit(install, install.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enable = await rig.service.review_activation(
        install.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enable, enable.operation_id)
    entered, release = threading.Event(), threading.Event()
    original_key = rig.runtime._hooks_v2_context_key
    original_install = rig.runtime.ensure_hooks_v2
    installed = []

    def held_context(session_id):
        result = original_key(session_id)
        entered.set()
        assert release.wait(5)
        return result

    def actual_install(*args, **kwargs):
        assert opaque_busy(rig.runtime), "NativeHooks installation lost its custody"
        assert rig.runtime.execution_capacity.snapshot().tool_workers == 1
        result = original_install(*args, **kwargs)
        installed.append(result)
        return result

    monkeypatch.setattr(rig.runtime, "_hooks_v2_context_key", held_context)
    monkeypatch.setattr(rig.runtime, "ensure_hooks_v2", actual_install)
    # This direct preparation call uses the same submission map initialized by
    # prepare_submission_hooks; it deliberately stops before any model/Stop work.
    rig.controller._hooks_v2_submissions = {}
    task = asyncio.create_task(
        rig.runtime.prepare_hooks_v2(
            rig.session.id, configuration=configuration_for(rig)
        )
    )
    rig.tasks.append(task)
    try:
        await until(entered.is_set)
        assert opaque_busy(rig.runtime), "host context read escaped preparation custody"
        assert rig.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not installed
        if outcome == "revoke":
            from tldw_chatbook.Plugins.revocation import RevocationTarget

            request = rig.service.begin_disable(
                RevocationTarget(install.installation_id, "workspace-a", False)
            )
            await rig.service.finish_revocation(request)
        elif outcome == "cancel":
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert opaque_busy(rig.runtime)
    finally:
        release.set()
    if outcome != "install":
        with pytest.raises((PermissionError, RuntimeError, asyncio.CancelledError)):
            await task
        assert not installed
        assert not opaque_busy(rig.runtime)
        return
    lifecycle = await task
    assert installed == [lifecycle.engine]
    assert installed[0].native_plugins.service is rig.service
    assert not opaque_busy(rig.runtime)
