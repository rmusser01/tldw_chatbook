"""Selected local permission operations use the original authority owners."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def permission_case(tmp_path):
    from tldw_chatbook.Chat.console_permission_operations import (
        ConsolePermissionOperations,
    )
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    from tldw_chatbook.config import get_user_data_dir

    store = MCPPermissionStore(get_user_data_dir() / "mcp_permissions.json")
    service = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    service._permission_store = store
    from tldw_chatbook.MCP.local_permission_operation import LocalPermissionOperation
    from tldw_chatbook.MCP.permission_store import _fresh_payload
    from tldw_chatbook.Backup_Recovery import mcp_source_participants

    operations = ConsolePermissionOperations(service)
    assert mcp_source_participants.source_receipt(store)[0] is store
    with LocalPermissionOperation(store, operations._require_current):
        store.save(_fresh_payload())
    return operations, service, store


@pytest.mark.asyncio
async def test_permission_operations_share_actual_singleton_and_publication(tmp_path):
    from tldw_chatbook.Chat.console_permission_operations import (
        ConsolePermissionOperations,
    )
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    from tldw_chatbook.config import get_user_data_dir

    store = MCPPermissionStore(get_user_data_dir() / "mcp_permissions.json")
    service = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    service._permission_store = store
    operations = ConsolePermissionOperations(service)
    publication, grants = store._permission_publication, service._session_approvals
    await operations.set_kill_switch(True)
    assert await operations.read_kill_switch() is True
    assert service.permission_store is store
    assert store._permission_publication is publication
    assert service._session_approvals is grants
    await operations.set_tool_state("agent:builtin", "Calculator", "deny")
    snapshot = await operations.read_permission_snapshot(strict=True)
    assert (
        snapshot.payload["profiles"]["default"]["servers"]["agent:builtin"]["tools"][
            "Calculator"
        ]["state"]
        == "deny"
    )
    assert not operations._reads


@pytest.mark.asyncio
async def test_original_session_grant_is_visible_and_revoke_changes_same_set(
    permission_case,
):
    operations, service, store = permission_case
    from tldw_chatbook.MCP.local_permission_operation import LocalPermissionOperation
    from tldw_chatbook.Tool_Packs.binding import (
        profile_policy_digest,
        ProfileMutationError,
    )

    snapshot = await operations.read_permission_snapshot()
    digest = profile_policy_digest(snapshot.payload["profiles"]["default"])
    # The original provider/gate is the grant producer; this surface has no
    # approve-session command and cannot turn a display listing into a grant.
    with LocalPermissionOperation(store, operations._require_current):
        service.approve_for_session(
            "agent:builtin", "Calculator", expected_profile_digest=digest
        )
    assert await operations.list_session_approvals() == [
        ("agent:builtin", "Calculator")
    ]
    assert service.is_session_approved("agent:builtin", "Calculator")
    assert await operations.revoke_session_approval("agent:builtin", "Calculator")
    assert not service.is_session_approved("agent:builtin", "Calculator")
    await operations.set_global_default("deny")
    with LocalPermissionOperation(store, operations._require_current):
        with pytest.raises(ProfileMutationError):
            service.approve_for_session(
                "agent:builtin", "Calculator", expected_profile_digest=digest
            )
    assert await operations.list_session_approvals() == []


@pytest.mark.asyncio
async def test_stale_profile_definition_and_actual_arg_rule_actions(permission_case):
    operations, _service, _store = permission_case
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool
    from tldw_chatbook.Tool_Packs.binding import (
        ProfileMutationError,
        profile_policy_digest,
    )

    tool = HubTool(
        server_key="local:docs",
        server_label="Docs",
        source="local",
        name="read",
        description="original",
        input_schema={"type": "object"},
        tags=(),
        stale=False,
        executable=True,
    )
    snapshot = await operations.read_permission_snapshot()
    digest = profile_policy_digest(snapshot.payload["profiles"]["default"])
    await operations.set_tool_state(
        tool.server_key, tool.name, "allow", tool=tool, expected_profile_digest=digest
    )
    with pytest.raises(ProfileMutationError):
        await operations.set_global_default("allow", expected_profile_digest=digest)
    changed = HubTool(
        server_key=tool.server_key,
        server_label="Docs",
        source="local",
        name=tool.name,
        description="changed",
        input_schema=tool.input_schema,
        tags=(),
        stale=False,
        executable=True,
    )
    states = await operations.read_effective_states([changed])
    assert states[(tool.server_key, tool.name)].state == "ask"
    assert states[(tool.server_key, tool.name)].config_changed
    await operations.add_arg_rule(
        tool.server_key, tool.name, args={"path": "README"}, tool=changed
    )
    rules = await operations.read_arg_rules(tool.server_key, tool.name)
    assert len(rules) == 1
    assert await operations.remove_arg_rule(
        tool.server_key, tool.name, rules[0]["rule_id"]
    )
    assert await operations.read_arg_rules(tool.server_key, tool.name) == []
    await operations.dismiss_suggestion(tool.server_key, tool.name)
    await operations.restore_suggestion(tool.server_key, tool.name)
    await operations.dismiss_suggestion(tool.server_key, tool.name)
    await operations.clear_suggestion_dismissals()


@pytest.mark.asyncio
async def test_strict_corruption_refuses_and_compatibility_read_keeps_original_repair(
    permission_case,
):
    operations, _service, store = permission_case
    from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshotError

    store.path.write_bytes(b"not JSON")
    with pytest.raises(PermissionStoreSnapshotError):
        await operations.read_permission_snapshot(strict=True)
    assert store.path.read_bytes() == b"not JSON"
    payload = await operations.read_permission_snapshot(strict=False)
    assert payload["kill_switch"] is False
    assert store.path.with_suffix(".json.bak").read_bytes() == b"not JSON"
    assert not store.path.exists()


@pytest.mark.asyncio
async def test_replaced_selected_definition_refuses_without_legacy_retry(
    permission_case, monkeypatch
):
    operations, service, _store = permission_case
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    calls = []
    monkeypatch.setattr(
        type(service), "set_kill_switch", lambda *_args: calls.append(True)
    )
    with pytest.raises(RecoveryRequired, match="permission_operation_source_changed"):
        await operations.set_kill_switch(True)
    assert calls == []


@pytest.mark.asyncio
async def test_native_call_retains_exact_future_through_cancellation(
    permission_case, monkeypatch
):
    operations, _service, store = permission_case
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.MCP import permission_process

    entered, release = threading.Event(), threading.Event()
    captured = []
    original = raw._file_in_state

    def held(state, *args, **kwargs):
        if state.source is store and not entered.is_set():
            captured.append(state)
            entered.set()
            assert release.wait(10)
        return original(state, *args, **kwargs)

    monkeypatch.setattr(raw, "_file_in_state", held)
    monkeypatch.setattr(
        permission_process,
        "PermissionProcess",
        lambda *_args: pytest.fail("nested permission process"),
    )
    task = asyncio.create_task(operations.set_kill_switch(True))
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        handles = tuple(operations._reads)
        assert len(handles) == 1
        handle = handles[0]
        assert handle._producer is not None and not handle._producer.done()
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert handle in operations._reads and not handle.retired.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handle._producer.done() and handle._producer.result() is None
    assert handle.callback_returned and handle.retired.done()
    assert handle not in operations._reads
    assert captured and all(
        not state.active and not state.files and not state.descriptors
        for state in captured
    )
    assert not operations._unresolved
    assert await operations.read_kill_switch() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [1, 2])
async def test_complete_render_capture_recaptures_changed_policy_in_original_fence(
    permission_case, changes
):
    operations, service, store = permission_case
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        "local:docs",
        "Docs",
        "local",
        "read",
        "read",
        {"type": "object"},
        (),
        False,
        True,
    )
    original = service.effective_tool_states.__func__.__code__
    mutations = []

    def observe(frame, event, _arg):
        if (
            frame.f_code is original
            and event == "return"
            and frame.f_locals.get("self") is service
        ):
            assert store._path_lock._is_owned()
            if len(mutations) < changes:
                state = "deny" if not mutations else "ask"
                mutations.append(state)
                store.set_global_default(state)

    previous, threads = sys.getprofile(), threading.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        result = await operations.capture_permission_render([tool])
    finally:
        threading.setprofile_all_threads(threads)
        sys.setprofile(previous)
    assert mutations == (["deny"] if changes == 1 else ["deny", "ask"])
    assert (
        result.snapshot.payload["profiles"]["default"]["global_default"]
        == mutations[-1]
    )
    assert (result.profile_identity is not None) is (changes == 1)
    assert result.effective_states[(tool.server_key, tool.name)].state == (
        "deny" if changes == 1 else "ask"
    )


@pytest.mark.asyncio
async def test_complete_tool_pack_lifecycle_retains_receipts_guard_and_real_cas(
    permission_case, tmp_path
):
    operations, service, store = permission_case
    from Tests.Tool_Packs.test_importer import _archive
    from tldw_chatbook.Chat.console_permission_operations import (
        ConsolePermissionOperations,
    )
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventoryRegistry
    from tldw_chatbook.Tool_Packs.contracts import ToolPackError
    from tldw_chatbook.Tool_Packs.service import ToolPackService
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    database = WorkspaceDB(tmp_path / "workspace.sqlite")
    registry = LocalWorkspaceRegistryService(database)
    local_catalog = SimpleNamespace(
        get_inventory=lambda: {"tools": []}, get_external_servers=lambda: []
    )
    inventory = PermissionInventoryRegistry.v1(local_catalog, fallback_root=tmp_path)
    packs = ToolPackService.compose(
        permission_store=store,
        inventory=inventory,
        workspace_registry=registry,
        receipt_root=tmp_path / "receipts",
    )
    registry.attach_tool_profile_guard(packs.binding_guard)
    operations = ConsolePermissionOperations(service, tool_packs=packs)
    archive = _archive(tmp_path / "pack.zip", rules=())
    try:
        review = await operations.inspect_import(archive, destination_id="research")
        activated = await operations.import_unbound_profile(review)
        assert activated is not None
        snapshot = await operations.read_permission_snapshot()
        profile = snapshot.payload["profiles"]["research"]
        lifecycle = profile["tool_pack_lifecycle"]
        assert lifecycle["revision"] == 1
        receipt = packs._receipt_store.read(
            lifecycle["receipt_id"], expected_digest=lifecycle["receipt_digest"]
        )
        assert receipt.receipt.profile_id == "research"
        listing = await operations.list_tool_profiles()
        assert listing.by_id("research").removal_eligible
        defaults = WorkspaceAssistantDefaults(
            assistant_id="persona", tool_policy_profile_id="research"
        )
        binding_review = await operations.review_first_bind(
            "workspace-future", defaults, action="create"
        )
        token = await operations.confirm_first_bind(binding_review)
        assert packs.binding_guard._tokens[token].review is binding_review
        with pytest.raises(ToolPackError, match="confirmation_invalid"):
            await operations.confirm_first_bind(binding_review)
        with packs.lifecycle.lease("research"):
            with pytest.raises(ToolPackError, match="in_use"):
                await operations.remove_imported_profile(
                    "research", expected_revision=1
                )
        with pytest.raises(ToolPackError, match="stale"):
            await operations.remove_imported_profile("research", expected_revision=2)
        removed = await operations.remove_imported_profile(
            "research", expected_revision=1
        )
        assert removed.tombstone.revision == 2
        snapshot = await operations.read_permission_snapshot()
        assert (
            snapshot.payload["profiles"]["research"]["profile_kind"]
            == "tool_pack_tombstone"
        )
        assert (await operations.list_tool_profiles()).by_id("research") is None
        assert not operations._unresolved
    finally:
        database.close()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Original Windows receipt limitation"
)
@pytest.mark.asyncio
async def test_original_unavailable_composition_refuses_selected_profiles_and_binding(
    permission_case, tmp_path
):
    """Exercise the original unavailable path without substituting receipt IO."""
    operations, service, store = permission_case
    from types import MethodType

    from tldw_chatbook.app_service_wiring import ServiceWiringMixin
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
    from tldw_chatbook.Workspaces.registry_service import (
        DeferredWorkspaceToolProfileGuard,
        LocalWorkspaceRegistryService,
        WorkspaceRegistryServiceError,
    )

    database = WorkspaceDB(tmp_path / "workspace.sqlite")
    registry = LocalWorkspaceRegistryService(database)
    bootstrap = DeferredWorkspaceToolProfileGuard()
    registry.attach_tool_profile_guard(bootstrap)
    app = SimpleNamespace(
        unified_mcp_service=service,
        local_mcp_control_service=SimpleNamespace(
            get_inventory=lambda: {"tools": []}, get_external_servers=lambda: []
        ),
        workspace_registry_service=registry,
        _tool_pack_guard_bootstrap=bootstrap,
        tool_pack_service=None,
        call_from_thread=lambda callback, *args: callback(*args),
    )
    for name in ("_attach_tool_pack_service", "_mark_tool_pack_service_unavailable"):
        setattr(app, name, MethodType(getattr(ServiceWiringMixin, name), app))
    before = await operations.read_permission_snapshot()
    defaults = WorkspaceAssistantDefaults(
        assistant_id="persona", tool_policy_profile_id="research"
    )
    try:
        await asyncio.to_thread(
            ServiceWiringMixin._compose_tool_pack_service_off_thread, app
        )
        assert app.tool_pack_service is None
        assert app.tool_pack_service_unavailable_reason == "composition_unavailable"
        assert bootstrap.active_guard is None
        assert registry.tool_profile_guard is bootstrap
        for action in (
            lambda: operations.inspect_import(
                tmp_path / "absent.zip", destination_id="research"
            ),
            lambda: operations.import_unbound_profile(object()),
            lambda: operations.remove_imported_profile("research", expected_revision=1),
            operations.list_tool_profiles,
            lambda: operations.review_first_bind("new", defaults, action="create"),
            lambda: operations.confirm_first_bind(object()),
        ):
            with pytest.raises(
                RecoveryRequired, match="permission_profile_owner_unavailable"
            ):
                await action()
        with pytest.raises(
            WorkspaceRegistryServiceError, match="authority is unavailable"
        ):
            registry.create_workspace(
                workspace_id="new", name="New", assistant_defaults=defaults
            )
        assert registry.get_workspace("new") is None
        assert (await operations.read_permission_snapshot()).payload == before.payload
        assert not operations._reads and not operations._unresolved
        await operations.set_kill_switch(True)
        assert await operations.read_kill_switch() is True
    finally:
        database.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Original receipt directory durability prevents genuine ToolPack composition",
)
@pytest.mark.parametrize(
    "change",
    (
        "nested_registry",
        "importer_store",
        "importer_inventory",
        "importer_references",
        "reader_receiver",
        "reader_body",
        "clock_replace",
        "clock_body",
        "clock_defaults",
        "clock_keywords",
    ),
)
@pytest.mark.asyncio
async def test_composed_profile_authority_refuses_nested_and_callback_changes(
    permission_case, tmp_path, monkeypatch, change
):
    """Requires genuine original composition; never substitute receipt durability."""
    operations, service, store = permission_case
    from types import MethodType

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat.console_permission_operations import (
        ConsolePermissionOperations,
    )
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventoryRegistry
    from tldw_chatbook.Tool_Packs.service import ToolPackService, _WorkspaceReferences
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    database = WorkspaceDB(tmp_path / "workspace.sqlite")
    registry = LocalWorkspaceRegistryService(database)
    inventory = PermissionInventoryRegistry.v1(
        SimpleNamespace(
            get_inventory=lambda: {"tools": []}, get_external_servers=lambda: []
        ),
        fallback_root=tmp_path,
    )
    try:
        packs = ToolPackService.compose(
            permission_store=store,
            inventory=inventory,
            workspace_registry=registry,
            receipt_root=tmp_path / "receipts",
        )
        registry.attach_tool_profile_guard(packs.binding_guard)
        selected = ConsolePermissionOperations(service, tool_packs=packs)
        before = await operations.read_permission_snapshot()
        if change == "nested_registry":
            monkeypatch.setattr(packs._references, "_registry", object())
        elif change.startswith("importer_"):
            field = {
                "importer_store": "_permission_store",
                "importer_inventory": "_inventory",
                "importer_references": "_reference_checker",
            }[change]
            monkeypatch.setattr(packs._importer, field, object())
        elif change == "reader_receiver":
            monkeypatch.setattr(
                packs.binding_guard,
                "_workspace_defaults_reader",
                MethodType(
                    _WorkspaceReferences.defaults_for, _WorkspaceReferences(registry)
                ),
            )
        elif change == "reader_body":

            def changed_reader(self, workspace_id):
                raise AssertionError("changed defaults reader executed")

            monkeypatch.setattr(
                _WorkspaceReferences.defaults_for, "__code__", changed_reader.__code__
            )
        else:
            clock = packs.binding_guard._now

            def changed_clock():
                raise AssertionError("changed clock executed")

            if change == "clock_replace":
                monkeypatch.setattr(packs.binding_guard, "_now", changed_clock)
            elif change == "clock_body":
                monkeypatch.setattr(clock, "__code__", changed_clock.__code__)
            elif change == "clock_defaults":
                monkeypatch.setattr(clock, "__defaults__", (object(),))
            else:
                monkeypatch.setattr(clock, "__kwdefaults__", {"changed": object()})
        with pytest.raises(RecoveryRequired, match="permission_profile_owner_changed"):
            ConsolePermissionOperations(service, tool_packs=packs)
        with pytest.raises(RecoveryRequired, match="permission_profile_owner_changed"):
            await selected.list_tool_profiles()
        assert not selected._reads and not selected._unresolved
        assert (await operations.read_permission_snapshot()).payload == before.payload
    finally:
        database.close()


@pytest.mark.parametrize(
    "change", ("body", "defaults", "keywords", "closure", "receiver")
)
def test_callback_definition_records_refuse_changed_inputs(change):
    """Finite provenance primitive only; this does not claim composed profile execution."""
    from types import MethodType
    from tldw_chatbook.Utils.definition_records import (
        capture_definitions,
        definitions_current,
    )

    captured = object()

    def callback(self, value=None, *, option=None):
        return captured, self, value, option

    owner = SimpleNamespace()
    owner.reader = MethodType(callback, owner)
    records = capture_definitions(((owner, "reader", owner.reader),))
    assert definitions_current(records)
    if change == "body":

        def replacement(self, value=None, *, option=None):
            return captured, None

        callback.__code__ = replacement.__code__
    elif change == "defaults":
        callback.__defaults__ = (object(),)
    elif change == "keywords":
        callback.__kwdefaults__["option"] = object()
    elif change == "closure":
        callback.__closure__[0].cell_contents = object()
    else:
        owner.reader = MethodType(callback, SimpleNamespace())
    assert not definitions_current(records)


@pytest.mark.asyncio
async def test_original_gate_keeps_once_session_and_builtin_always_distinct(
    permission_case,
):
    operations, service, store = permission_case
    from Tests.Agents.test_builtin_tool_gate import _Mutating
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.MCP.local_permission_operation import LocalPermissionOperation

    await operations.set_tool_state("agent:builtin", "write_thing", "ask")
    gate = BuiltinToolGate(service)
    baseline = (await operations.read_permission_snapshot()).generation
    with LocalPermissionOperation(store, operations._require_current):
        gate.stamp("once", "write_thing", "approve_once")
        assert gate.check(_Mutating(), "once") is None
        assert gate.check(_Mutating(), "other") is not None
        gate.stamp("always", "write_thing", "always_allow")
        assert gate.check(_Mutating(), "always") is None
        assert gate.check(_Mutating(), "other") is not None
        gate.stamp("session", "write_thing", "approve_session")
        assert gate.check(_Mutating(), "other") is None
    assert (await operations.read_permission_snapshot()).generation == baseline
    assert await operations.list_session_approvals() == [
        ("agent:builtin", "write_thing")
    ]
    await operations.clear_session_approvals()
    with LocalPermissionOperation(store, operations._require_current):
        assert gate.check(_Mutating(), "other") is not None


@pytest.mark.asyncio
async def test_selected_source_replacement_never_uses_successor(
    permission_case, tmp_path
):
    operations, service, original = permission_case
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    successor = MCPPermissionStore(tmp_path / "successor.json")
    service._permission_store = successor
    with pytest.raises(RecoveryRequired, match="permission_operation_source_changed"):
        await operations.set_kill_switch(True)
    assert not successor.path.exists()
    service._permission_store = original
    assert await operations.read_kill_switch() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["swallowed", "raised"])
async def test_failed_native_close_retains_original_owner_before_publication(
    permission_case, monkeypatch, mode
):
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP import local_permission_operation as local
    from tldw_chatbook.MCP.permission_store import _PermissionPathFence

    operations, _service, store = permission_case
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    submit = loop.run_in_executor
    original_close = storage.StorageLease.close
    armed = True
    retained = []
    completions = []

    def close(lease):
        operation = local.active()
        if armed and operation is not None:
            for _key, state in operation.observations["raw"]:
                if (
                    state.source is store
                    and lease in state.leases
                    and lease._key is not None
                ):
                    retained.append((lease, state, threading.current_thread()))
                    if mode == "raised":
                        raise OSError("injected original native lease close failure")
                    return None
        return original_close(lease)

    def observe(frame, event, _arg):
        if event == "call" and frame.f_code is _PermissionPathFence.complete.__code__:
            if frame.f_locals.get("self") is store._permission_publication:
                completions.append(True)

    def cleanup():
        nonlocal armed
        armed = False
        assert all(
            thread is threading.current_thread() for _lease, _state, thread in retained
        )
        with store.mutation_fence():
            for key, state in tuple(raw._states.items()):
                if state.source is not store:
                    continue
                assert not state.files and not state.descriptors
                state.uncertain = False
                assert raw._retire(state)
                with storage._changed:
                    del raw._states[key]
                    storage._raw_operations.discard(key)
                    storage._changed.notify_all()
        for operation in operations._unresolved:
            operation.verify_retirement()
        store._mcp_persistence_error = None
        store._permission_publication.pending = False
        store._permission_publication.uncertain = False

    previous, threads = sys.getprofile(), threading.getprofile()
    monkeypatch.setattr(
        loop,
        "run_in_executor",
        lambda selected, fn, *args: submit(
            executor if selected is None else selected, fn, *args
        ),
    )
    monkeypatch.setattr(storage.StorageLease, "close", close)
    threading.setprofile_all_threads(observe)
    try:
        expected = OSError if mode == "raised" else RecoveryRequired
        with pytest.raises(expected):
            await operations.set_kill_switch(True)
        assert retained and not completions
        assert operations._unresolved
        assert all(
            operation.retirement_error is not None
            for operation in operations._unresolved
        )
        for lease, state, _thread in retained:
            assert lease in storage._live_leases and lease._key is not None
            assert any(value is state for value in raw._states.values())
        assert store._permission_publication.uncertain
        assert store._permission_publication.token is None
        assert store._mcp_persistence_error == "mcp_persistence_incomplete"
    finally:
        threading.setprofile_all_threads(threads)
        sys.setprofile(previous)
        await submit(executor, cleanup)
        executor.shutdown(wait=True)
    assert all(operation.retired for operation in operations._unresolved)
    assert all(
        lease not in storage._live_leases and lease._key is None
        for lease, _state, _thread in retained
    )


def test_local_operation_rejects_copied_creator(permission_case):
    operations, _service, store = permission_case
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP import local_permission_operation as local

    results = []
    with local.LocalPermissionOperation(
        store, operations._require_current
    ) as operation:

        def copied():
            local._local.operation = operation
            try:
                with pytest.raises(RecoveryRequired, match="creator_changed"):
                    local.selected(store)
                results.append(True)
            finally:
                local._local.operation = None

        thread = threading.Thread(target=copied)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
    assert results == [True] and operation.retired


@pytest.mark.asyncio
async def test_selected_call_retires_only_its_new_leases_with_shared_owner_alive(
    permission_case,
):
    operations, _service, store = permission_case
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    lease = storage.acquire_storage(store.path)
    with storage._lock:
        hold = storage._holds[lease._key]
    try:
        assert await operations.read_kill_switch() is False
        assert lease in storage._live_leases and hold.thread.is_alive()
        assert not operations._unresolved
    finally:
        lease.close()
    assert lease not in storage._live_leases


@pytest.mark.asyncio
async def test_selected_runtime_uses_same_singleton_and_retained_host(permission_case):
    _operations, service, _permissions = permission_case
    from Tests.Chat.test_console_domain_boundary import _Parent
    from tldw_chatbook.Chat import console_runtime as runtime_source
    from tldw_chatbook.Chat.console_domain_services import (
        ConsoleDomainServices,
        ConsoleDomainSources,
    )
    from tldw_chatbook.Chat.console_presentation_port import ConsolePresentationPort
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    parent = _Parent()
    domain = ConsoleDomainServices(
        ConsoleDomainSources(
            configuration={"console": {"native_tool_calls": False}},
            control_plane=service,
        ),
        runtime_source=runtime_source,
        presentation=ConsolePresentationPort(parent),
    )
    runtime = runtime_source.ConsoleRuntime(parent, domain_services=domain)
    gateway = ConsoleProviderGateway(config_provider=domain.provider_config)
    store = runtime.ensure_chat_store()
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
        provider="deepseek",
        model="deepseek-chat",
    )
    session = store.create_session(title="Permission owner", workspace_id="global")
    operations = runtime._execution_service.ensure_permission_operations()
    host = controller._interrupt_host
    event = threading.Event()
    answers = {}
    state = {
        "decisions": answers,
        "event": event,
        "names": {"read"},
        "session_id": session.id,
        "run_id": "original-run",
    }
    try:
        assert operations._control_plane is domain.sources.control_plane is service
        assert runtime._execution_service.ensure_permission_operations() is operations
        assert operations._host is host
        with host.lock:
            host.registries["approval"]["original-round"] = state
        operations.resolve_approval_round(
            {"read": "approve_session"}, round_id="foreign-round"
        )
        assert not event.is_set() and answers == {}
        operations.resolve_approval_round(
            {"read": "approve_once"}, round_id="original-round"
        )
        assert event.is_set() and answers == {"read": "approve_once"}
        assert not service._session_approvals
        operations.resolve_approval_round(
            {"read": "always_allow"}, round_id="original-round"
        )
        assert answers == {"read": "approve_once"}
        controller._interrupt_host = object()
        with pytest.raises(RecoveryRequired, match="approval_host_unavailable"):
            operations.resolve_approval_round({}, round_id="original-round")
        controller._interrupt_host = host
    finally:
        controller._interrupt_host = host
        with host.lock:
            host.registries["approval"].pop("original-round", None)
        await runtime.dispose()
        await gateway.aclose()
