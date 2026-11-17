"""Production permission routing with real Windows admission and retirement."""

import asyncio
import json
import time
from contextlib import nullcontext

import pytest

from Tests.MCP import test_permission_process_native as native_controls

source = native_controls.source


@pytest.fixture(autouse=True)
def owners(tmp_path, monkeypatch):
    from tldw_chatbook.MCP import permission_process as process

    created = []
    initialize = process.PermissionProcess.__init__

    def record(broker, source):
        initialize(broker, source)
        created.append(broker)

    monkeypatch.setattr(process.PermissionProcess, "__init__", record)
    yield
    records = []
    for broker in created:
        if broker.unresolved:
            broker.contain_unresolved()
        else:
            broker.close()
        records.append(
            {
                "pid": None if broker.process is None else broker.process.pid,
                "retired": broker.retired,
                "unresolved": bool(broker.unresolved),
                "receipts": broker.receipts,
                "origins": broker.origins,
            }
        )
        assert broker.retired
    (tmp_path / "permission-routing-audit.json").write_text(
        json.dumps(records, indent=2)
    )
    # Retained outcomes are recorded above. Clear only this test's owner census.
    process._brokers.difference_update(created)


def test_captured_console_routes_without_parent_permission_raw_scope(
    source, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.console_snapshot import _CapturedSources
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    local = LocalMCPControlService(
        store=LocalMCPStore(source.path.with_name("local_mcp_store.json")),
        manifest_provider=lambda: {},
    )
    service = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=local, server_service=None
    )
    service._permission_store = source
    captured = _CapturedSources(service)
    source.load()
    token = source._permission_publication.token
    read = process.read_message
    checks = []

    def observe(stream):
        message = read(stream)
        if message.get("type") == "Check":
            assert not any(state.source is source for state in raw._states.values())
            captured.require_current()
            checks.append(message["data"]["site"])
        return message

    monkeypatch.setattr(process, "read_message", observe)
    assert captured.read_permission_payload()["kill_switch"] is False
    assert source._permission_publication.token is token
    assert captured.permission_call(lambda store: store.get_kill_switch()) is False
    assert captured.permission_identity[0] is source
    assert "protected_open" in checks or not source.path.exists()
    assert len(checks) > 3


def test_same_path_receiver_and_external_composite_keep_original_authority(
    source, monkeypatch
):
    import threading
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    source.set_kill_switch(False)
    second = MCPPermissionStore(source.path)
    broker = process.select(source)
    assert process.select(second) is broker
    entered, finished = threading.Event(), threading.Event()
    errors = []

    def mutate():
        entered.set()
        try:
            second.set_kill_switch(True)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    with source.mutation_fence():
        thread = threading.Thread(target=mutate)
        thread.start()
        assert entered.wait(5)
        assert not finished.wait(0.05)
        assert source.get_kill_switch() is False
    thread.join(10)
    assert finished.is_set() and not errors
    assert second.get_kill_switch() is True
    read = process.read_message
    changed = []

    def substitute(stream):
        message = read(stream)
        if message.get("type") == "Check" and not changed:
            changed.append(True)
            monkeypatch.setattr(second, "path", second.path.with_name("foreign.json"))
        return message

    monkeypatch.setattr(process, "read_message", substitute)
    with pytest.raises(process.ProtocolError, match="source_changed"):
        second.get_kill_switch()
    assert changed and broker.unresolved
    assert source._permission_publication.uncertain


def test_compatibility_is_selected_before_issue(source, monkeypatch):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.Backup_Recovery import bootstrap

    original = bootstrap.default_bootstrap_root
    with monkeypatch.context() as patch:
        patch.setattr(bootstrap, "default_bootstrap_root", lambda: original())
        assert process.select(source) is None
        assert source.get_kill_switch() is False
        assert not process._brokers

    class Truth:
        def __bool__(self):
            return True

    assert source.set_kill_switch(Truth()) is None
    assert not process._brokers


def test_healthy_context_change_retires_old_helper(source, monkeypatch):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources

    source.get_kill_switch()
    before = process.select(source)
    config = sources._BINDINGS[source].config
    monkeypatch.setattr(config, "_CONFIG_CACHE", dict(config._CONFIG_CACHE))
    assert source.get_kill_switch() is False
    after = process.select(source)
    assert before is not after and before.retired and before.closed
    assert before.process.pid != after.process.pid


@pytest.mark.parametrize(
    "state,error_type", [("ask", AttributeError), (None, TypeError)]
)
def test_legacy_mutator_pre_effect_error_settles_without_poison(
    source, monkeypatch, state, error_type
):
    from tldw_chatbook.MCP import permission_process as process

    payload = {
        "schema_version": 1,
        "kill_switch": False,
        "profiles": {"default": {"global_default": "ask", "servers": {"s": 1}}},
    }
    source.path.write_text(json.dumps(payload), encoding="utf-8")
    before = source.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(process, "forward", lambda *args, **kwargs: (False, None))
        with pytest.raises(error_type) as local:
            source.set_server_default("s", state)
        assert source.path.read_bytes() == before
        assert source.get_kill_switch() is False
        assert source._mcp_persistence_error is None
    with pytest.raises(error_type) as routed:
        source.set_server_default("s", state)
    assert str(routed.value) == str(local.value)
    assert source.path.read_bytes() == before
    broker = process.select(source)
    assert broker.unresolved is None and broker.receipts[-1]["retired"]
    assert source._mcp_persistence_error is None
    assert not source._permission_publication.uncertain
    assert source.get_kill_switch() is False
    broker.close()
    assert broker.retired and broker.process.returncode == 0


def test_uncertain_old_profile_blocks_selection_until_physical_containment(
    source, monkeypatch, tmp_path
):
    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    source.get_kill_switch()
    old = process.select(source)
    # Hold a real idle helper and all its physical ownership pending. The
    # unresolved record is deliberately independent from containment status.
    outcome = {"injected_unresolved": True}
    old._poison(outcome)
    assert not old.retired and old.process.poll() is None
    data = tmp_path / "next-data"
    data.mkdir(mode=0o700)
    config_path = tmp_path / "next-config.toml"
    config_path.write_text(f'[paths]\ndata_dir = "{data.as_posix()}"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config = install_config_source(monkeypatch)
    new_source = MCPPermissionStore(config.get_user_data_dir() / "mcp_permissions.json")
    assert new_source._path_lock is not old.path_lock
    owners = set(process._brokers)
    with pytest.raises(process.ProtocolError, match="permission_context_busy"):
        process.select(new_source)
    assert set(process._brokers) == owners
    assert old.unresolved is outcome and not old.retired
    old.contain_unresolved()
    assert old.retired and old.unresolved is outcome and old in process._brokers
    assert source._permission_publication.uncertain
    assert new_source.get_kill_switch() is False
    new = process.select(new_source)
    assert new is not old and new.process.pid != old.process.pid
    new.close()
    assert new.retired and old.unresolved is outcome


def test_custom_mutation_helper_keeps_checked_compatibility_path(source, monkeypatch):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    reached = []

    def refuse(*args, **kwargs):
        reached.append(True)
        raise RecoveryRequired("custom_replace_refused")

    monkeypatch.setattr(raw, "replace", refuse)
    assert process.select(source) is None
    with pytest.raises(RecoveryRequired, match="custom_replace_refused"):
        source.set_kill_switch(True)
    assert reached and not process._brokers


def test_direct_receiver_gc_retires_idle_worker(source):
    import gc
    import weakref
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    second = MCPPermissionStore(source.path)
    second.get_kill_switch()
    broker = process.select(second)
    reference = weakref.ref(second)
    del second
    gc.collect()
    assert reference() is None
    assert broker.retired and broker.closed and broker not in process._brokers


def test_session_approval_rechecks_changed_policy_after_review(source):
    import threading
    from tldw_chatbook.MCP.permission_store import (
        profile_policy_digest,
        ProfileMutationError,
    )
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    service = object.__new__(UnifiedMCPControlPlaneService)
    service._permission_store = source
    service._session_approvals = set()
    service._session_approvals_lock = threading.Lock()
    source.ensure_profile("writer")
    digest = profile_policy_digest(
        source.read_snapshot_strict().payload["profiles"]["writer"]
    )
    service.approve_for_session(
        "local:example", "read", profile_id="writer", expected_profile_digest=digest
    )
    assert service.is_session_approved("local:example", "read", profile_id="writer")
    service._session_approvals.clear()
    source.set_global_default("deny", profile_id="writer")
    with pytest.raises(ProfileMutationError, match="stale_profile"):
        service.approve_for_session(
            "local:example", "read", profile_id="writer", expected_profile_digest=digest
        )
    assert not service._session_approvals


def test_tool_pack_binding_holds_parent_permission_fence_through_registry_commit(
    source, tmp_path, monkeypatch
):
    import threading
    from contextlib import contextmanager
    from Tests.Tool_Packs import test_binding_guard as controls
    from tldw_chatbook.MCP import permission_process as process

    monkeypatch.setattr(controls, "MCPPermissionStore", lambda path: source)
    registry, _, guard, _ = controls._build(tmp_path, monkeypatch)
    intended = controls._defaults()
    token = guard.confirm(guard.review("w-1", intended, action="set"))
    transaction = registry.db.transaction
    entered, release, changed = threading.Event(), threading.Event(), threading.Event()
    errors = []

    @contextmanager
    def blocked_commit(*, immediate=False):
        with transaction(immediate=immediate) as connection:
            yield connection
            entered.set()
            assert release.wait(15)

    monkeypatch.setattr(registry.db, "transaction", blocked_commit)

    def bind():
        try:
            registry.set_assistant_defaults(
                "w-1", intended, tool_profile_confirmation_token=token
            )
        except BaseException as error:
            errors.append(error)

    def mutate():
        try:
            source.set_kill_switch(True)
            changed.set()
        except BaseException as error:
            errors.append(error)

    binding = threading.Thread(target=bind)
    binding.start()
    mutation = None
    try:
        assert entered.wait(15)
        mutation = threading.Thread(target=mutate)
        mutation.start()
        assert not changed.wait(0.1)
    finally:
        release.set()
        binding.join(20)
        if mutation is not None:
            mutation.join(20)
    assert not binding.is_alive() and mutation is not None and not mutation.is_alive()
    assert not errors and changed.is_set()
    assert process.select(source).process is not None
    registry.db.close()


def test_public_get_set_route_before_parent_raw_scope(source):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.MCP import permission_process as process

    assert source.get_kill_switch() is False
    broker = process.select(source)
    assert broker.process is not None
    assert not any(state.source is source for state in raw._states.values())
    token = source._permission_publication.token
    assert source.get_kill_switch() is False
    assert source._permission_publication.token is token
    assert source.set_kill_switch(True) is None
    assert source.get_kill_switch() is True


def test_all_public_commands_keep_results_and_profile_cas(source, monkeypatch):
    from Tests.Tool_Packs.test_permission_profile_authority import (
        _imported_profile,
        _tombstone_profile,
    )
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import (
        ProfileMutationError,
        ProfileMutationResult,
    )
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    commands = []
    write = process.write_message

    def observe(stream, message):
        if message.get("type") == "Command":
            commands.append(message["data"]["operation"])
            assert not any(state.source is source for state in raw._states.values())
        return write(stream, message)

    monkeypatch.setattr(process, "write_message", observe)
    assert source.get_kill_switch() is False
    assert source.set_kill_switch(False) is None

    snapshot = source.read_snapshot_strict()
    assert snapshot.file_exists is False
    with pytest.raises(TypeError):
        snapshot.payload["kill_switch"] = True
    assert source.catalog_fingerprint() == ("fresh",)
    payload = source.load()
    assert source.save(payload, expected_generation=snapshot.generation) is None
    assert "updated_at" in payload
    fingerprint = source.catalog_fingerprint()
    assert type(fingerprint) is tuple and type(fingerprint[2]) is tuple
    assert source.ensure_profile("workspace") is None
    assert source.list_profiles() == ["default", "workspace"]
    assert source.get_global_default() == "ask"
    assert source.set_global_default("deny") is None
    assert source.get_global_default() == "deny"
    assert source.get_server_entry("local:example") is None
    assert source.set_server_default("local:example", "ask") is None
    assert source.get_server_entry("local:example")["default"] == "ask"
    assert source.get_tool_entry("local:example", "read") is None
    assert (
        source.set_tool_state(
            "local:example", "read", "allow", definition_hash="a" * 64
        )
        is None
    )
    assert source.get_tool_entry("local:example", "read")["state"] == "allow"
    assert source.mark_config_changed("local:example", "read") is True
    assert source.mark_config_changed("local:example", "read") is False
    args = {"file": "safe.txt"}
    assert (
        source.add_tool_arg_rule(
            "local:example", "read", args=args, definition_hash="a" * 64
        )
        is None
    )
    assert source.has_current_tool_arg_rule(
        "local:example", "read", args=args, definition_hash="a" * 64
    )
    rules = source.list_tool_arg_rules("local:example", "read")
    assert source.remove_tool_arg_rule("local:example", "read", rules[0]["rule_id"])
    assert not source.remove_tool_arg_rule("local:example", "read", rules[0]["rule_id"])
    assert source.dismiss_suggestion("local:example", "read") is None
    assert "local:example::read" in source.list_suggestion_dismissals()
    assert source.restore_suggestion("local:example", "read") is None
    assert source.clear_suggestion_dismissals() is None
    assert source.list_suggestion_dismissals() == {}
    assert source.read_profile_inventory_snapshot().file_exists
    with pytest.raises(ValueError, match="state"):
        source.set_global_default("invalid")
    with pytest.raises(ProfileMutationError, match="stale_store"):
        source.save(payload, expected_generation=snapshot.generation)
    candidate = _imported_profile()
    result = source.install_profile_if_absent(
        "imported",
        candidate,
        expected_generation=source.read_snapshot_strict().generation,
        max_profiles=20,
        max_store_bytes=1000000,
    )
    assert type(result) is ProfileMutationResult and result.revision == 1
    result = source.update_imported_profile(
        "imported", candidate, expected_revision=1, max_store_bytes=1000000
    )
    assert result.revision == 2
    with pytest.raises(ProfileMutationError, match="stale_revision"):
        source.update_imported_profile(
            "imported", candidate, expected_revision=1, max_store_bytes=1000000
        )
    result = source.replace_profile_with_tombstone(
        "imported", _tombstone_profile(), expected_revision=2, max_store_bytes=1000000
    )
    assert result.revision == 3
    assert process.select(source).process is not None
    assert set(commands) == set(process._SCHEMAS)


@pytest.mark.parametrize(
    "fault", ["control", "registry", "acl", "rename", "reparse", "missing", "replaced"]
)
def test_routed_native_boundary_refuses_changed_evidence(source, monkeypatch, fault):
    from Tests.Backup_Recovery.path_aliases import create_directory_alias
    from Tests.Backup_Recovery.test_raw_lock_stream_parent_posture import (
        _public_posture,
    )
    from Tests.Backup_Recovery.windows_acl_fixture import preserve_windows_dacl
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.MCP import permission_process as process

    source.set_kill_switch(True)
    source.load()
    broker = process.select(source)
    original = source.path.read_bytes()
    parent = source.path.parent
    moved = parent.with_name(parent.name + "-moved")
    marker = bootstrap.default_bootstrap_root() / "t23b-unknown-control.json"
    registry = bootstrap.default_bootstrap_root() / "admission" / "registry.json"
    registry_bytes = registry.read_bytes()
    read = process.read_message
    changed = []

    def changing(stream):
        message = read(stream)
        if (
            message.get("type") == "Check"
            # Parent ACL posture is an admission predicate in the original
            # permission path; protected-open refreshes the pinned identity.
            and message["data"]["site"]
            == ("entry" if fault == "acl" else "protected_open")
            and not changed
        ):
            changed.append(True)
            if fault == "control":
                marker.write_bytes(b"{}")
            elif fault == "registry":
                registry.write_bytes(b"{}")
            elif fault == "acl":
                _public_posture(parent, writable=True)
            elif fault in {"rename", "reparse"}:
                parent.rename(moved)
                if fault == "reparse":
                    create_directory_alias(parent, moved)
            elif fault == "missing":
                source.path.unlink()
            else:
                replacement = source.path.with_suffix(".replacement")
                replacement.write_bytes(original)
                replacement.replace(source.path)
        return message

    monkeypatch.setattr(process, "read_message", changing)
    with preserve_windows_dacl(parent) if fault == "acl" else nullcontext():
        try:
            with pytest.raises(
                (process.ProtocolError, bootstrap.RecoveryRequired, OSError)
            ):
                source.load()
            assert changed == [True]
        finally:
            marker.unlink(missing_ok=True)
            if fault == "registry":
                registry.write_bytes(registry_bytes)
            if fault == "reparse" and parent.is_junction():
                parent.rmdir()
            if moved.exists():
                moved.rename(parent)
            if fault == "missing":
                source.path.write_bytes(original)
    assert source.path.read_bytes() == original
    assert not source.path.with_suffix(".json.bak").exists()
    if broker.unresolved:
        with pytest.raises(process.ProtocolError):
            source.load()


def test_stock_parent_acl_post_admission_oracle(source, monkeypatch):
    """Compare the unchanged local read boundary with the initial IPC fault."""
    import sys
    from Tests.Backup_Recovery.test_raw_lock_stream_parent_posture import (
        _public_posture,
    )
    from Tests.Backup_Recovery.windows_acl_fixture import preserve_windows_dacl
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.MCP import permission_process as process

    monkeypatch.setattr(process, "forward", lambda *args, **kwargs: (False, None))
    source.set_kill_switch(True)
    changed = []

    def profile(frame, event, result):
        if (
            event == "call"
            and frame.f_code is raw._permission_read_boundary.__code__
            and frame.f_locals.get("destination")
            and not changed
        ):
            changed.append(True)
            _public_posture(source.path.parent, writable=True)

    with preserve_windows_dacl(source.path.parent):
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            assert source.load()["kill_switch"] is True
        finally:
            sys.setprofile(previous)
    assert changed == [True]


@pytest.mark.asyncio
async def test_routed_cancellation_and_maintenance_settle_exact_worker(
    source, monkeypatch
):
    import threading
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.console_snapshot import _owned_worker

    source.get_kill_switch()
    broker = process.select(source)
    entered, release = threading.Event(), threading.Event()
    read = process.read_message

    def waiting(stream):
        message = read(stream)
        if message.get("type") == "Check" and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return message

    monkeypatch.setattr(process, "read_message", waiting)
    task = asyncio.create_task(_owned_worker(source.get_kill_switch))
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done() and broker.lifetime.calls
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not broker.lifetime.calls and broker.unresolved is None
        before = broker.process.pid
        participant = process.participant
        participant._maintenance_close_admission()
        assert await participant._maintenance_drain(time.monotonic() + 10)
        with pytest.raises(process.ProtocolError, match="intake_closed"):
            source.get_kill_switch()
        participant._maintenance_resume()
        assert source.get_kill_switch() is False
        assert broker.process.pid != before
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
