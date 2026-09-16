"""Actual MCP durable-source lifetime regressions, isolated from transports."""

import json
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from Tests.Backup_Recovery.test_participant_lifetimes import local_root


@pytest.fixture
def mcp_sources(tmp_path, monkeypatch, local_root):
    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
    from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.MCP.execution_log import MCPExecutionLog

    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    target = tmp_path / "config.toml"
    target.write_text(f'[paths]\ndata_dir = "{data}"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config = install_config_source(monkeypatch)
    data = config.get_user_data_dir()
    sources = (
        LocalMCPStore(data / "local_mcp_store.json"),
        ConfiguredServerTargetStore(data / "mcp_server_targets.json"),
        UnifiedMCPContextStore(data / "unified_mcp_context.json"),
        MCPPermissionStore(data / "mcp_permissions.json"),
        MCPExecutionLog(data / "mcp_execution_log.jsonl"),
    )
    return sources


@pytest.mark.parametrize("index", range(5))
def test_actual_reader_refuses_before_read_or_recovery(mcp_sources, index):
    source = mcp_sources[index]
    read = source.read_recent if index == 4 else source.load
    read()
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            read()
        assert pause.drain(time.monotonic() + 0.2)
    finally:
        pause.resume()


def test_permission_rmw_admitted_before_pause_finishes(mcp_sources, monkeypatch):
    source = mcp_sources[3]
    source.set_kill_switch(False)
    original = json.loads
    pauses = []

    def loaded_then_pause(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(result, dict) and "kill_switch" in result and not pauses:
            pauses.append(storage._begin_local_pause())
        return result

    monkeypatch.setattr(json, "loads", loaded_then_pause)
    try:
        source.set_kill_switch(True)
        assert pauses
        assert original(source.path.read_text())["kill_switch"] is True
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.parametrize("index", range(5))
def test_constructor_selection_is_reserved_before_pause(mcp_sources, index):
    source = mcp_sources[index]
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            type(source)(source.path)
    finally:
        pause.resume()


@pytest.mark.parametrize("index", range(5))
def test_installed_retarget_refuses_and_custom_subclass_stays_unqualified(
    mcp_sources, tmp_path, index
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[index]
    selected = source.path
    source.path = tmp_path / "other.json"
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
            (source.read_recent if index == 4 else source.load)()
    finally:
        source.path = selected
    custom = type(source)(tmp_path / f"custom-{index}.json")
    (custom.read_recent if index == 4 else custom.load)()
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(custom)

    class Custom(type(source)):
        pass

    subclass = Custom(selected)
    (subclass.read_recent if index == 4 else subclass.load)()
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(subclass)


def test_same_path_distinct_source_cannot_borrow_admitted_permission(
    mcp_sources, monkeypatch
):
    source = mcp_sources[3]
    other = type(source)(source.path)
    source.set_kill_switch(False)
    original = json.loads
    pauses = []

    def loaded(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(result, dict) and "kill_switch" in result and not pauses:
            pauses.append(storage._begin_local_pause())
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                other.set_kill_switch(False)
        return result

    monkeypatch.setattr(json, "loads", loaded)
    try:
        source.set_kill_switch(True)
        assert original(source.path.read_text())["kill_switch"]
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.parametrize("corrupt", ["{broken", "[]", '{"schema_version":99}'])
def test_permission_corrupt_move_finishes_admitted_load_across_pause(
    mcp_sources, monkeypatch, corrupt
):
    source = mcp_sources[3]
    source.path.write_text(corrupt)
    backup = source.path.with_suffix(".json.bak")
    backup.write_text("old backup")
    original = json.loads
    pauses = []

    def parse(value, *args, **kwargs):
        if value == corrupt and not pauses:
            pauses.append(storage._begin_local_pause())
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", parse)
    try:
        assert source.load()["profiles"]["default"]["global_default"] == "ask"
        assert not source.path.exists()
        assert backup.read_text() == corrupt
    finally:
        for pause in pauses:
            pause.resume()


def _record(name):
    from tldw_chatbook.MCP.execution_log import build_record

    return build_record(
        server_key="local:test",
        tool_name=name,
        initiator="test",
        ok=True,
        duration_ms=1,
    )


@pytest.mark.parametrize("action", ["append", "read"])
def test_history_migration_and_rotation_finish_across_pause(
    mcp_sources, monkeypatch, action
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import private_paths

    source = mcp_sources[4]
    source.max_records_per_file = 1
    source.path.write_text('{"tool_name":"older","arguments":{"redacted":0}}\n{torn\n')
    rotated = source.path.with_name(source.path.name + ".1")
    rotated.write_text('{"tool_name":"oldest"}\n')
    original = private_paths._native_close
    pauses = []

    def closed(fd):
        operation = private_paths._runtime_operation()
        state = raw._states.get(operation)
        original(fd)
        if (
            state is not None
            and state.source is source
            and state.mcp_effects
            and not pauses
        ):
            pauses.append(storage._begin_local_pause())

    monkeypatch.setattr(private_paths, "_native_close", closed)
    try:
        if action == "append":
            source.append(_record("newest"))
        else:
            assert [r["tool_name"] for r in source.read_recent()] == ["older", "oldest"]
        assert pauses
        assert "arguments" not in source.path.read_text()
        assert "{torn" not in source.path.read_text()
        assert not list(source.path.parent.glob(".*.tmp"))
    finally:
        for pause in pauses:
            pause.resume()
    expected = ["newest", "older"] if action == "append" else ["older", "oldest"]
    assert [r["tool_name"] for r in source.read_recent()] == expected


def test_permission_serialization_failure_keeps_caller_timestamp(
    mcp_sources, monkeypatch
):
    source = mcp_sources[3]
    payload = source.load()
    source.save(payload)
    before = dict(payload)
    original = source.path.read_bytes()
    payload["unserializable"] = object()
    with pytest.raises(TypeError):
        source.save(payload)
    assert payload["updated_at"] == before["updated_at"]
    assert source.path.read_bytes() == original
    assert not source.path.with_suffix(".json.tmp").exists()


def test_permission_stale_temp_is_not_truncated_or_deleted(mcp_sources):
    source = mcp_sources[3]
    payload = source.load()
    temporary = source.path.with_suffix(".json.tmp")
    temporary.write_text("foreign temporary")
    with pytest.raises(FileExistsError):
        source.save(payload)
    assert temporary.read_text() == "foreign temporary"
    assert "updated_at" not in payload


def test_permission_preflight_pause_has_no_payload_or_backup_effects(
    mcp_sources, monkeypatch
):
    source = mcp_sources[3]
    source.path.write_text("{broken")
    before = source.path.read_bytes()
    original = storage.acquire_storage
    pauses = []

    def acquired(path=None):
        lease = original(path)
        if path is not None and str(path).endswith(".bak") and not pauses:
            pauses.append(storage._begin_local_pause())
        return lease

    monkeypatch.setattr(storage, "acquire_storage", acquired)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            source.load()
        assert source.path.read_bytes() == before
        assert not source.path.with_suffix(".json.bak").exists()
    finally:
        for pause in pauses:
            pause.resume()


def _private_child(request, kind):
    import os
    from pathlib import Path
    import subprocess
    import sys

    if os.environ.get("TASK10_MCP_CHILD") == kind:
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            request.node.nodeid,
            "-q",
            "-o",
            "cache_dir=/private/tmp/task10-phase11-child-cache",
        ],
        env={**os.environ, "TASK10_MCP_CHILD": kind, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    Path(f"/private/tmp/task10-phase11-child-{kind}.log").write_text(
        result.stdout + result.stderr
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


from Tests.Backup_Recovery.test_admission import launch, line, release


@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize(
    "point", ["permission_parent", "history_read", "history_append", "history_atomic"]
)
def test_actual_native_uncertainty_retains_source_and_independent_lease(
    mcp_sources, local_root, monkeypatch, request, launch, timing, point
):
    if _private_child(request, f"{point}-{timing}"):
        return
    import os
    import select
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import private_paths

    permission, history = mcp_sources[3:]
    permission.set_kill_switch(False)
    history.append(_record("before"))
    source = permission if point == "permission_parent" else history
    payload = permission.load()
    stamp = payload.get("updated_at")
    payload["kill_switch"] = True
    history.max_records_per_file = 1
    # Test child only: retire its known quiescent startup, then prove exclusion
    # comes from actual MCP source/native holds, not startup overlap.
    storage._startups[(os.getpid(), str(local_root))].close()
    target = []
    real_close = os.close
    real_replace = raw._replace
    real_stream_close = private_paths._close_runtime_stream
    real_native_close = private_paths._native_close

    def published(token, temporary, destination):
        result = real_replace(token, temporary, destination)
        state = raw._states[token]
        if point == "permission_parent" and state.source is source:
            target.append(state.pins[source.path.parent])
        return result

    def stream_close(operation, stream):
        state = raw._states[operation]
        mode = getattr(stream, "mode", "")
        if state.source is source and (
            point == "history_read"
            and "r" in mode
            or point == "history_append"
            and "a" in mode
        ):
            target.append(stream.fileno())
        return real_stream_close(operation, stream)

    def native_close(fd):
        operation = private_paths._runtime_operation()
        state = raw._states.get(operation)
        rotated = source.path.with_name(source.path.name + ".1")
        if (
            point == "history_atomic"
            and state is not None
            and state.source is source
            and rotated.exists()
        ):
            published_info = rotated.stat()
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) == (
                published_info.st_dev,
                published_info.st_ino,
            ):
                target.append(fd)
        return real_native_close(fd)

    def uncertain_close(fd):
        if target and fd == target[0]:
            if timing == "after":
                real_close(fd)
            raise OSError("injected native close uncertainty")
        return real_close(fd)

    monkeypatch.setattr(raw, "_replace", published)
    monkeypatch.setattr(private_paths, "_close_runtime_stream", stream_close)
    monkeypatch.setattr(private_paths, "_native_close", native_close)
    monkeypatch.setattr(os, "close", uncertain_close)
    if point == "history_append":
        history.max_records_per_file = 5
    with pytest.raises(bootstrap.RecoveryRequired, match="raw_resources_not_retired"):
        if point == "permission_parent":
            permission.save(payload)
        elif point == "history_read":
            history.read_recent()
        else:
            history.append(_record("after"))
    monkeypatch.setattr(os, "close", real_close)
    assert target
    if timing == "before":
        assert os.fstat(target[0])
    assert source._mcp_persistence_error == "mcp_persistence_incomplete"
    if point == "permission_parent":
        assert payload["updated_at"] == stamp
        assert json.loads(permission.path.read_text())["kill_switch"]
    participant = raw._raw_participant(source)
    participant.close_admission()
    pause = storage._begin_local_pause()
    observer = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        assert not participant.drain(time.monotonic() + 0.02)
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([observer.stdout], [], [], 0.06)[0]
    finally:
        observer.kill()
        observer.wait(timeout=5)
        pause.resume()
        # Strong uncertain resources intentionally retire only on child exit.


def test_independent_maintainer_enters_after_history_rotation_native_close(
    mcp_sources, local_root, monkeypatch, request, launch
):
    if _private_child(request, "rotation-observer"):
        return
    import os
    import select
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import private_paths

    source = mcp_sources[4]
    source.max_records_per_file = 1
    source.append(_record("old"))
    storage._startups[(os.getpid(), str(local_root))].close()
    original = private_paths._native_close
    observers, pauses = [], []

    def closed(fd):
        operation = private_paths._runtime_operation()
        state = raw._states.get(operation)
        original(fd)
        if (
            state is not None
            and state.source is source
            and state.mcp_effects
            and not observers
        ):
            pauses.append(storage._begin_local_pause())
            observer = launch(
                local_root / "admission", "maintenance", ("bootstrap.unbound",)
            )
            observers.append(observer)
            assert not select.select([observer.stdout], [], [], 0.05)[0]

    monkeypatch.setattr(private_paths, "_native_close", closed)
    try:
        source.append(_record("new"))
        assert observers
        assert line(observers[0]) == "entered"
        assert json.loads(source.path.read_text())["tool_name"] == "new"
        assert (
            json.loads(source.path.with_name(source.path.name + ".1").read_text())[
                "tool_name"
            ]
            == "old"
        )
        release(observers[0])
    finally:
        for observer in observers:
            if observer.poll() is None:
                observer.kill()
                observer.wait(timeout=5)
        for pause in pauses:
            pause.resume()


@pytest.mark.parametrize("which", ["destination", "backup", "parent"])
def test_permission_identity_swap_preserves_foreign_entries(
    mcp_sources, monkeypatch, tmp_path, which
):
    from tldw_chatbook.Backup_Recovery import mcp_source_participants as mcp

    source = mcp_sources[3]
    original = json.loads
    source.path.write_text("{broken" if which == "backup" else '{"schema_version":1}')
    backup = source.path.with_suffix(".json.bak")
    backup.write_text("old backup")
    destination = backup if which == "backup" else source.path
    foreign = tmp_path / "foreign.json"
    foreign.write_text("foreign bytes")
    changed = []
    original_text = source.path.read_text()

    def parse(value, *args, **kwargs):
        if not changed and value == original_text:
            changed.append(True)
            if which == "parent":
                old = source.path.parent.with_name("moved-parent")
                source.path.parent.rename(old)
                source.path.parent.mkdir(mode=0o700)
                destination.write_text("foreign bytes")
            else:
                foreign.replace(destination)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", parse)
    with pytest.raises(bootstrap.RecoveryRequired, match="identity_changed"):
        if which == "backup":
            source.load()
        else:
            source.set_kill_switch(True)
    assert destination.read_text() == "foreign bytes"
    assert mcp.drain_ready(source)


def test_history_partial_generations_stay_failed_after_unrelated_success(
    mcp_sources, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[4]
    source.max_records_per_file = 1
    source.append(_record("old"))
    real_allowed = raw.mcp_sources.helper_allowed
    refused = []

    def allowed(state, helper, selected):
        if (
            state.source is source
            and helper == "atomic_private_write_bytes"
            and selected == source.path
            and not refused
        ):
            refused.append(True)
            raise OSError("injected second generation failure")
        return real_allowed(state, helper, selected)

    monkeypatch.setattr(raw.mcp_sources, "helper_allowed", allowed)
    with pytest.raises(OSError):
        source.append(_record("new"))
    assert refused
    assert json.loads(source.path.read_text())["tool_name"] == "old"
    assert (
        json.loads(source.path.with_name(source.path.name + ".1").read_text())[
            "tool_name"
        ]
        == "old"
    )
    assert source._mcp_persistence_error == "mcp_persistence_incomplete"
    monkeypatch.setattr(raw.mcp_sources, "helper_allowed", real_allowed)
    source.append(_record("later"))
    assert source.read_recent()[0]["tool_name"] == "later"
    participant = raw._raw_participant(source)
    participant.close_admission()
    try:
        assert not participant.drain(time.monotonic() + 0.03)
    finally:
        participant.resume()


def test_history_reused_temporary_preserves_foreign_recreation(
    mcp_sources, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[4]
    source.max_records_per_file = 1
    source.path.write_text('{"tool_name":"old"}\n')
    rotated = source.path.with_name(source.path.name + ".1")
    rotated.write_text('{"tool_name":"older"}\n')
    real_published = raw.mcp_sources.published
    foreign = []

    def published(state, selected):
        real_published(state, selected)
        if state.source is source and selected == rotated and not foreign:
            temporary = state.temporaries[selected]
            temporary.write_text("foreign recreated temporary")
            foreign.append(temporary)

    monkeypatch.setattr(raw.mcp_sources, "published", published)
    with pytest.raises(OSError):
        source.append(_record("new"))
    assert foreign[0].read_text() == "foreign recreated temporary"
    assert source._mcp_persistence_error == "mcp_persistence_incomplete"
    assert raw._raw_participant(source).owner_id == "mcp.history"


def test_history_private_helpers_reject_other_paths_and_foreign_task(
    mcp_sources, tmp_path
):
    import asyncio
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import private_paths

    source = mcp_sources[4]
    outsider = tmp_path / "foreign"
    with raw._scope(source, raw.mcp_sources.ROUTE, writing=True):
        with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
            private_paths.atomic_private_write_bytes(outsider, b"x")
        with pytest.raises(RuntimeError, match="helper_not_supported"):
            private_paths.create_private_text(source.path, "x")

    async def check():
        with raw._scope(source, raw.mcp_sources.ROUTE, writing=True):

            async def foreign():
                with pytest.raises(
                    bootstrap.RecoveryRequired, match="provenance_invalid"
                ):
                    source.read_recent()

            await asyncio.create_task(foreign())

    asyncio.run(check())
    assert not outsider.exists()


def test_config_retarget_and_native_demotion_refuse_bound_source(
    mcp_sources, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook import config

    source = mcp_sources[0]
    original = config._CONFIG_CACHE["paths"]["data_dir"]
    config._CONFIG_CACHE["paths"]["data_dir"] = str(source.path.parent / "new")
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
            source.load()
    finally:
        config._CONFIG_CACHE["paths"]["data_dir"] = original
    monkeypatch.setattr(raw, "_pinned_io_available", lambda: False)
    with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
        source.load()


def test_history_foreign_destination_between_generations_is_preserved(
    mcp_sources, monkeypatch, tmp_path
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[4]
    source.max_records_per_file = 1
    source.append(_record("old"))
    foreign = tmp_path / "foreign.jsonl"
    foreign.write_text("foreign history")
    real_published = raw.mcp_sources.published
    changed = []

    def published(state, selected):
        real_published(state, selected)
        if state.source is source and selected.name.endswith(".1") and not changed:
            changed.append(True)
            foreign.replace(source.path)

    monkeypatch.setattr(raw.mcp_sources, "published", published)
    with pytest.raises(bootstrap.RecoveryRequired, match="identity_changed"):
        source.append(_record("new"))
    assert source.path.read_text() == "foreign history"
    assert source._mcp_persistence_error == "mcp_persistence_incomplete"


def test_history_source_publication_does_not_adopt_foreign_inode(
    mcp_sources, monkeypatch, tmp_path
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[4]
    source.max_records_per_file = 1
    source.append(_record("old"))
    foreign = tmp_path / "foreign.jsonl"
    foreign.write_text("foreign history")
    real_published = raw.mcp_sources.published
    changed = []

    def published(state, selected):
        if state.source is source and selected.name.endswith(".1") and not changed:
            changed.append(True)
            foreign.replace(selected)
        real_published(state, selected)

    monkeypatch.setattr(raw.mcp_sources, "published", published)
    with pytest.raises(bootstrap.RecoveryRequired, match="identity_changed"):
        source.append(_record("new"))
    assert (
        source.path.with_name(source.path.name + ".1").read_text() == "foreign history"
    )
    assert json.loads(source.path.read_text())["tool_name"] == "old"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_actual_local_target_context_publication_finishes_after_pause(
    mcp_sources, monkeypatch, index
):
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile
    from tldw_chatbook.MCP.unified_control_models import (
        ConfiguredServerTarget,
        UnifiedMCPContext,
    )

    source = mcp_sources[index]
    if index == 0:
        source.save_profile(
            LocalExternalMCPProfile(profile_id="before", command="python")
        )
        action = lambda: source.save_profile(
            LocalExternalMCPProfile(profile_id="after", command="python")
        )
    elif index == 1:
        source.save_targets(
            [
                ConfiguredServerTarget(
                    server_id="target",
                    label="Target",
                    base_url="https://example.invalid",
                    authority_scope_id=None,
                )
            ]
        )
        action = lambda: source.ensure_authority_scope_id("target")
    else:
        action = lambda: source.save(UnifiedMCPContext(selected_section="catalogs"))
    original = json.dump
    pauses = []

    def serialize(payload, stream, *args, **kwargs):
        if not pauses:
            pauses.append(storage._begin_local_pause())
        return original(payload, stream, *args, **kwargs)

    monkeypatch.setattr(json, "dump", serialize)
    try:
        result = action()
        assert pauses and source.path.exists()
        if index == 1:
            assert (
                json.loads(source.path.read_text())["targets"][0]["authority_scope_id"]
                == result
            )
    finally:
        for pause in pauses:
            pause.resume()
    if index == 0:
        assert [profile.profile_id for profile in source.list_profiles()] == [
            "before",
            "after",
        ]
    elif index == 2:
        assert source.load().selected_section == "catalogs"


def test_installed_exact_file_binding_does_not_enroll_permission_sidecars(
    mcp_sources, local_root, request
):
    if _private_child(request, "exact-file-binding"):
        return
    import os
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook import config

    source = mcp_sources[3]
    source.set_kill_switch(True)
    original = source.path.read_bytes()
    storage._startups[(os.getpid(), str(local_root))].close()
    authority = admission_authority(local_root)
    selected_config = config._get_effective_config_path()
    authority.register("file-only", (source.path, selected_config))
    bind_profile(local_root, selected_config, ("file-only",), local_root / "admission")
    with pytest.raises(bootstrap.RecoveryRequired):
        source.set_kill_switch(False)
    assert source.path.read_bytes() == original
    assert not source.path.with_suffix(".json.tmp").exists()
    assert not source.path.with_suffix(".json.bak").exists()


def test_none_native_hold_does_not_grant_history_pause_authority(
    mcp_sources, local_root, request, monkeypatch
):
    if _private_child(request, "none-native-hold"):
        return
    import os
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[4]
    storage._startups[(os.getpid(), str(local_root))].close()
    monkeypatch.setattr(
        storage, "qualified_for", lambda *args: (False, "test_native_unavailable")
    )
    with raw._scope(source, raw.mcp_sources.ROUTE, writing=True) as operation:
        assert all(hold is None for hold in raw._states[operation].holds)
        pause = storage._begin_local_pause()
        try:
            with pytest.raises(
                bootstrap.RecoveryRequired, match="native_scope_unqualified"
            ):
                source.append(_record("refused"))
            assert not source.path.exists()
        finally:
            pause.resume()


@pytest.mark.parametrize("index", range(5))
def test_installed_constructor_reentry_cannot_retarget_or_clear_failure(
    mcp_sources, tmp_path, index
):
    source = mcp_sources[index]
    original = source.path
    source._mcp_persistence_error = "mcp_persistence_incomplete"
    with pytest.raises(bootstrap.RecoveryRequired, match="already_bound"):
        source.__init__(tmp_path / "retargeted.json")
    assert source.path == original
    assert source._mcp_persistence_error == "mcp_persistence_incomplete"


def test_installed_history_private_posture_demotion_refuses_before_read(
    mcp_sources, monkeypatch
):
    from tldw_chatbook.Utils import private_paths

    source = mcp_sources[4]
    monkeypatch.setattr(private_paths, "_atomic_posix_guards_available", lambda: False)
    with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
        source.read_recent()
