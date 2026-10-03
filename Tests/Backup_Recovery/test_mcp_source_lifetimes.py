"""Actual MCP durable-source lifetime regressions, isolated from transports."""

import json
import sys
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from Tests.Backup_Recovery.test_participant_lifetimes import local_root


def _fixture_config(data):
    """Use the existing JSON string spelling for portable TOML fixture paths."""
    return "[paths]\ndata_dir = " + json.dumps(str(data), ensure_ascii=False) + "\n"


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
    target.write_text(_fixture_config(data), encoding="utf-8")
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
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("TASK10_MCP_CHILD") == kind:
        return False
    private = Path(
        request.config.option.basetemp or request.getfixturevalue("tmp_path")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            request.node.nodeid,
            "-q",
            "--basetemp="
            + str(
                Path(
                    request.config.option.basetemp
                    or request.getfixturevalue("tmp_path")
                )
                / ("child-" + kind)
            ),
            "-o",
            "cache_dir=" + str(private / "child-cache"),
        ],
        env={**os.environ, "TASK10_MCP_CHILD": kind, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    (private / f"task10-phase11-child-{kind}.log").write_text(
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


@pytest.mark.parametrize("index", range(4))
def test_guarded_json_publication_requires_positive_native_retirement(
    mcp_sources, monkeypatch, index
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Utils.platform_files import os as native_os

    source = mcp_sources[index]
    storage.admit_startup()
    payload = source.load() if index == 3 else {"schema_version": 4, "profiles": []}
    payload["cache_probe"] = {"nested": ["held"]}
    source.path.write_text(json.dumps(payload))
    read = source.load if index == 3 else source._read_payload
    original_reader, original_complete = shared.reader, shared.complete
    original_parse, original_scope = json.loads, storage._scope
    original_barrier = Admission._current_hold
    states, handles, parses, scopes, barriers = [], [], [], [], []

    @contextmanager
    def reader(current):
        with original_reader(current) as handle:
            if current is source:
                operation, state = shared._operation(current)
                assert state.participant is not None and state.pinned  # nosec B101
                assert state.active and raw._states[operation] is state  # nosec B101
                assert state.holds and all(  # nosec B101
                    isinstance(hold, storage._Hold)
                    and storage._holds.get(hold.key) is hold
                    and hold.native_context is not None
                    for hold in state.holds
                )
                states.append((operation, state))
                handles.append(handle)
            yield handle

    def complete(state):
        if state.source is source:
            assert not state.active and not state.uncertain  # nosec B101
            assert not state.files and not state.descriptors and not state.pins  # nosec B101
            assert all(lease not in storage._live_leases for lease in state.leases)  # nosec B101
            assert handles[-1].closed  # nosec B101
        return original_complete(state)

    def parse(value, *args, **kwargs):
        result = original_parse(value, *args, **kwargs)
        if isinstance(result, dict) and "cache_probe" in result:
            parses.append(value)
        return result

    def scope(*args, **kwargs):
        scopes.append(True)
        return original_scope(*args, **kwargs)

    @contextmanager
    def barrier(self, *args, **kwargs):
        with original_barrier(self, *args, **kwargs) as observed:
            barriers.append(self)
            yield observed

    monkeypatch.setattr(shared, "reader", reader)
    monkeypatch.setattr(shared, "complete", complete)
    monkeypatch.setattr(json, "loads", parse)
    monkeypatch.setattr(storage, "_scope", scope)
    monkeypatch.setattr(Admission, "_current_hold", barrier)
    first = read()
    assert len(states) == 1 and handles[0].closed  # nosec B101
    operation, state = states[0]
    assert operation not in raw._states and operation not in storage._raw_operations  # nosec B101
    assert any(key[0]() is source for hold in state.holds for key in hold.json_evidence)  # nosec B101
    first["cache_probe"]["nested"].append("caller")
    before_scopes, before_barriers = len(scopes), len(barriers)
    assert read()["cache_probe"] == {"nested": ["held"]}  # nosec B101
    assert len(parses) == 1 and len(states) == 2  # nosec B101
    assert all(handle.closed for handle in handles)  # nosec B101
    assert all(operation not in raw._states for operation, _ in states)  # nosec B101
    assert len(barriers) > before_barriers  # nosec B101
    if native_os.name == "nt":
        assert len(scopes) > before_scopes  # nosec B101
        assert all(not hold.evidence and not hold.path_evidence for hold in state.holds)  # nosec B101


@pytest.mark.skipif(sys.platform != "win32", reason="actual Windows native guards")
@pytest.mark.parametrize("change", ["registry", "pending_intent"])
def test_windows_warm_json_refuses_changed_current_control(
    mcp_sources, monkeypatch, change
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared

    source = mcp_sources[0]
    storage.admit_startup()
    source.path.write_text('{"cache_probe":true}')
    assert source._read_payload()["cache_probe"] is True  # nosec B101
    holds = tuple(storage._holds.values())
    assert any(key[0]() is source for hold in holds for key in hold.json_evidence)  # nosec B101
    assert all(not hold.evidence and not hold.path_evidence for hold in holds)  # nosec B101
    control = holds[0].authority.control_root
    registry = control / "registry.json"
    before = registry.read_bytes()
    intent = control / "registry.pending.json"
    assert not intent.exists()  # nosec B101
    original_reader, reads = shared.reader, []

    @contextmanager
    def reader(current):
        if current is source:
            reads.append(current)
        with original_reader(current) as handle:
            yield handle

    monkeypatch.setattr(shared, "reader", reader)
    selected_bytes = source.path.read_bytes()
    try:
        if change == "registry":
            registry.write_bytes(b"{" + b" " * (len(before) - 1))
        else:
            record = json.loads(before)
            intent.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "write_id": "a" * 32,
                        "before": record,
                        "after": record,
                    }
                )
            )
            intent.chmod(0o600)
        with pytest.raises((bootstrap.RecoveryRequired, ValueError, OSError)):
            source._read_payload()
        assert not reads  # nosec B101
        assert source.path.read_bytes() == selected_bytes  # nosec B101
    finally:
        if change == "registry":
            registry.write_bytes(before)
        else:
            intent.unlink()
    assert registry.read_bytes() == before and not intent.exists()  # nosec B101
    assert source._read_payload()["cache_probe"] is True  # nosec B101


@pytest.mark.skipif(sys.platform != "win32", reason="actual Windows parent DACL")
def test_windows_warm_json_refuses_shared_writable_parent(mcp_sources, monkeypatch):
    import subprocess  # nosec B404
    from contextlib import contextmanager

    from Tests.Backup_Recovery.windows_acl_fixture import preserve_windows_dacl
    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared
    from tldw_chatbook.Utils.platform_files import os as native_os

    source = mcp_sources[0]
    storage.admit_startup()
    parent = source.path.parent
    foreign = parent / "foreign.txt"
    foreign.write_bytes(b"foreign fixture bytes")
    foreign.chmod(0o600)
    before_files = {
        path: path.read_bytes() for path in parent.iterdir() if path.is_file()
    }
    with preserve_windows_dacl(parent):
        before = native_os.stat(parent)
        source.path.write_text('{"cache_probe":true}')
        assert source._read_payload()["cache_probe"] is True  # nosec B101
        assert any(  # nosec B101
            key[0]() is source
            for hold in storage._holds.values()
            for key in hold.json_evidence
        )
        selected_bytes = source.path.read_bytes()
        original_reader, reads = shared.reader, []

        @contextmanager
        def reader(current):
            if current is source:
                reads.append(current)
            with original_reader(current) as handle:
                yield handle

        monkeypatch.setattr(shared, "reader", reader)
        # Fixed native fixture ACL tool and arguments.
        subprocess.run(  # nosec B603, B607
            ["icacls", str(parent), "/grant", "*S-1-1-0:(W)"],
            check=True,
            capture_output=True,
        )
        assert native_os.stat(parent).st_mode & 0o022  # nosec B101
        with pytest.raises(bootstrap.RecoveryRequired):
            source._read_payload()
        assert not reads and source.path.read_bytes() == selected_bytes  # nosec B101
        assert foreign.read_bytes() == b"foreign fixture bytes"  # nosec B101
    after = native_os.stat(parent)
    assert (before.st_dev, before.st_ino, before.st_mode) == (  # nosec B101
        after.st_dev,
        after.st_ino,
        after.st_mode,
    )
    assert {  # nosec B101
        path: path.read_bytes() for path in before_files if path != source.path
    } == {path: value for path, value in before_files.items() if path != source.path}
    assert source._read_payload()["cache_probe"] is True  # nosec B101


@pytest.mark.parametrize("index", range(4))
def test_guarded_json_reuses_parse_but_reads_current_bytes(
    mcp_sources, monkeypatch, index
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared

    source = mcp_sources[index]
    storage.admit_startup()
    payload = {"schema_version": 4, "profiles": [], "nested": {"items": ["old"]}}
    if index == 3:
        source.set_kill_switch(False)
        payload = source.load()
        source.save(payload)
    else:
        source.path.write_text(json.dumps(payload))
    read = source.load if index == 3 else source._read_payload
    original_parse = json.loads
    original_reader = shared.reader
    parses, reads = [], []

    def parse(value, *args, **kwargs):
        result = original_parse(value, *args, **kwargs)
        if (
            isinstance(result, dict)
            and result.get("schema_version") == payload["schema_version"]
        ):
            parses.append(value)
        return result

    @contextmanager
    def reader(current):
        if current is source:
            reads.append(current)
        with original_reader(current) as handle:
            yield handle

    monkeypatch.setattr(json, "loads", parse)
    monkeypatch.setattr(shared, "reader", reader)
    first = read()
    first["profiles"] = {"mutated": []}
    second = read()
    assert first != second
    assert len(reads) == 2
    assert len(parses) == 1
    assert not storage._pending_acquisitions


@pytest.mark.parametrize("index", range(4))
def test_guarded_json_same_metadata_bytes_and_nested_returns(mcp_sources, index):
    import os

    source = mcp_sources[index]
    storage.admit_startup()
    payload = source.load() if index == 3 else {"schema_version": 4, "profiles": []}
    payload["cache_probe"] = {"nested": ["old"]}
    source.path.write_text(json.dumps(payload))
    read = source.load if index == 3 else source._read_payload
    first = read()
    first["cache_probe"]["nested"].append("caller")
    assert read()["cache_probe"] == {"nested": ["old"]}
    before = source.path.stat()
    raw = source.path.read_bytes()
    with source.path.open("r+b") as handle:
        handle.write(raw.replace(b'"old"', b'"new"'))
    os.utime(source.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.path.stat().st_ino == before.st_ino
    assert read()["cache_probe"] == {"nested": ["new"]}


@pytest.mark.parametrize("change", ["missing", "corrupt", "unreadable"])
def test_warm_permission_parse_never_supplies_last_good_policy(
    mcp_sources, monkeypatch, change
):
    import errno
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared
    from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshotError

    source = mcp_sources[3]
    storage.admit_startup()
    payload = source.load()
    source.save(payload)
    assert source.load()["kill_switch"] is False
    source.read_snapshot_strict()
    before = source.path.read_bytes()
    try:
        if change == "missing":
            source.path.unlink()
            assert source.read_snapshot_strict().generation.startswith("missing:")
        elif change == "corrupt":
            source.path.write_bytes(b'{"kill_switch":false,"kill_switch":true}')
            with pytest.raises(PermissionStoreSnapshotError, match="duplicate_key"):
                source.read_snapshot_strict()
            assert source.path.read_bytes().startswith(b'{"kill_switch"')
        else:
            original_reader = shared.reader
            attempts, handles = [], []

            @contextmanager
            def unreadable(current):
                with original_reader(current) as handle:
                    if current is source:
                        attempts.append(current)
                        handles.append(handle)
                        raise PermissionError(errno.EACCES, "synthetic read failure")
                    yield handle

            with monkeypatch.context() as fault:
                fault.setattr(shared, "reader", unreadable)
                with pytest.raises(OSError):
                    source.load()
                with pytest.raises(PermissionStoreSnapshotError, match="io_error"):
                    source.read_snapshot_strict()
                assert source._load_for_raw_getter()["kill_switch"] is True  # nosec B101
            assert len(attempts) == 3 and all(handle.closed for handle in handles)  # nosec B101
            assert source.path.read_bytes() == before  # nosec B101
        assert not source.path.with_suffix(".json.bak").exists()
    finally:
        source.path.write_bytes(before)
    assert source.load()["kill_switch"] is False


def test_guarded_permission_strict_inventory_bytes_and_policy_stay_distinct(
    mcp_sources,
):
    import hashlib

    from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshotError

    source = mcp_sources[3]
    storage.admit_startup()
    payload = source.load()
    source.save(payload)
    snapshot = source.read_snapshot_strict()
    assert (
        snapshot.generation
        == "sha256:" + hashlib.sha256(source.path.read_bytes()).hexdigest()
    )
    payload["profiles"]["broken"] = {
        "servers": {},
        "profile_kind": "tool_pack_imported",
    }
    source.path.write_text(json.dumps(payload))
    inventory = source.read_profile_inventory_snapshot()
    assert "broken" in inventory.payload["profiles"]
    with pytest.raises(PermissionStoreSnapshotError, match="invalid_shape"):
        source.read_snapshot_strict()
    assert inventory.generation != snapshot.generation
    assert (
        inventory.generation
        == "sha256:" + hashlib.sha256(source.path.read_bytes()).hexdigest()
    )


def test_guarded_parse_pause_invalidates_and_oversize_keeps_legacy_policy(
    mcp_sources, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared

    source = mcp_sources[0]
    storage.admit_startup()
    source.path.write_text('{"cache_probe": ["value"]}')
    original = json.loads
    calls = []

    def parse(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(result, dict) and "cache_probe" in result:
            calls.append(result)
        return result

    monkeypatch.setattr(json, "loads", parse)
    source._read_payload()
    source._read_payload()
    assert len(calls) == 1
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            source._read_payload()
    finally:
        pause.resume()
    source._read_payload()
    assert len(calls) == 2
    monkeypatch.setattr(shared, "_JSON_CACHE_BYTES", 1)
    source._read_payload()
    source._read_payload()
    assert len(calls) == 4


def test_guarded_parse_migration_does_not_publish_prewrite_bytes(mcp_sources):
    from tldw_chatbook.MCP.local_store import LocalMCPStoreLoadError

    source = mcp_sources[0]
    storage.admit_startup()
    source.path.write_text('{"schema_version":1,"profiles":[]}')
    old = source.path.read_bytes()
    source.load()
    assert json.loads(source.path.read_bytes())["schema_version"] == 4
    assert not any(
        entry[0] == old
        for hold in storage._holds.values()
        for entry in hold.json_evidence.values()
    )
    source.path.write_text('{"schema_version":99,"profiles":[]}')
    with pytest.raises(LocalMCPStoreLoadError):
        source.load()
    assert not any(
        key[0]() is source
        for hold in storage._holds.values()
        for key in hold.json_evidence
    )


@pytest.mark.parametrize("closed_first", [False, True])
def test_guarded_parse_unknown_read_close_never_publishes_or_retries(
    mcp_sources, monkeypatch, request, closed_first
):
    if _private_child(request, f"json-read-close-{closed_first}"):
        return
    import os
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = mcp_sources[0]
    storage.admit_startup()
    source.path.write_text('{"cache_probe":true}')
    source._read_payload()
    original_reader, original_close = shared.reader, os.close
    selected, attempts, recycled = [], [], []

    @contextmanager
    def reader(current):
        with original_reader(current) as handle:
            if current is source:
                selected.append(handle.buffer.fileno())
            yield handle

    def close(fd):
        if selected and fd == selected[-1]:
            attempts.append(fd)
            if closed_first:
                original_close(fd)
                recycled.append(os.open(os.devnull, os.O_RDONLY))
                assert recycled[-1] == fd
            raise OSError("unresolved read close")
        return original_close(fd)

    monkeypatch.setattr(shared, "reader", reader)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(bootstrap.RecoveryRequired):
        source._read_payload()
    assert len(attempts) == 1
    uncertain = [
        state
        for state in raw._states.values()
        if state.source is source and state.uncertain
    ]
    assert len(uncertain) == 1 and uncertain[0].descriptors
    assert all(hold.native_context is not None for hold in uncertain[0].holds)
    assert not any(
        key[0]() is source for hold in uncertain[0].holds for key in hold.json_evidence
    )
    with pytest.raises(bootstrap.RecoveryRequired):
        source._read_payload()
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.01)
        assert len(attempts) == 1
        if recycled:
            os.fstat(recycled[0])  # No retry closed a recycled descriptor.
    finally:
        pause.resume()
    # The private child exits with its failed native ownership intact.


def test_guarded_parse_last_owner_retirement_and_selection_error_discard(
    mcp_sources, monkeypatch, tmp_path
):
    source = mcp_sources[0]
    source.path.write_text('{"cache_probe":true}')
    original = json.loads
    calls = []

    def parse(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(result, dict) and "cache_probe" in result:
            calls.append(result)
        return result

    monkeypatch.setattr(json, "loads", parse)
    owner = storage.acquire_storage(source.path)
    hold = storage._holds[owner._key]
    startup = storage._startups.pop(owner._key, None)
    if startup is not None:
        startup.close()  # Positively retire only this fixture's startup owner.
    try:
        source._read_payload()
        source._read_payload()
        assert len(calls) == 1
        selected = source.path
        source.path = tmp_path / "different.json"
        try:
            with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
                source._read_payload()
        finally:
            source.path = selected
        source._read_payload()
        assert len(calls) == 2
    finally:
        owner.close()
    assert hold.native_context is None and not hold.thread.is_alive()
    source._read_payload()
    source._read_payload()
    assert len(calls) == 4  # No surviving owner means no reusable publication.


def test_guarded_parse_fork_cannot_use_inherited_evidence(mcp_sources):
    import os

    if not hasattr(os, "fork"):
        pytest.skip("POSIX fork control")
    source = mcp_sources[0]
    storage.admit_startup()
    source.path.write_text('{"cache_probe":true}')
    source._read_payload()
    child = os.fork()
    if child == 0:
        status = 1
        try:
            assert not storage._monitor_subscribers
            with pytest.raises(bootstrap.RecoveryRequired):
                source._read_payload()
            status = 0
        finally:
            os._exit(status)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert source._read_payload()["cache_probe"] is True


@pytest.mark.parametrize("index", range(4))
def test_guarded_json_replacement_after_read_refuses_detached_bytes(
    mcp_sources, monkeypatch, index
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mcp_source_participants as shared

    source = mcp_sources[index]
    storage.admit_startup()
    payload = source.load() if index == 3 else {"schema_version": 4, "profiles": []}
    payload["cache_probe"] = "old"
    source.path.write_text(json.dumps(payload))
    read = source.load if index == 3 else source._read_payload
    assert read()["cache_probe"] == "old"
    original_reader = shared.reader
    replaced = []

    @contextmanager
    def reader(current):
        with original_reader(current) as handle:
            yield handle
        if current is source:
            assert handle.closed  # nosec B101
            from tldw_chatbook.Utils.platform_files import os as native_os

            before = native_os.stat(source.path)
            replacement = source.path.with_name("replacement.json")
            payload["cache_probe"] = "new"
            replacement.write_text(json.dumps(payload))
            replacement.replace(source.path)
            after = native_os.stat(source.path)
            assert (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)  # nosec B101
            assert source.path.read_text() == json.dumps(payload)  # nosec B101
            replaced.append(True)

    monkeypatch.setattr(shared, "reader", reader)
    with pytest.raises(bootstrap.RecoveryRequired, match="raw_entry_identity_changed"):
        read()
    assert replaced == [True]  # nosec B101
    assert not any(
        key[0]() is source
        for hold in storage._holds.values()
        for key in hold.json_evidence
    )


def test_guarded_parse_config_generation_change_requires_fresh_parse(
    mcp_sources, monkeypatch
):
    from tldw_chatbook import config

    source = mcp_sources[0]
    storage.admit_startup()
    source.path.write_text('{"cache_probe":true}')
    original = json.loads
    parses = []

    def parse(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(result, dict) and "cache_probe" in result:
            parses.append(result)
        return result

    monkeypatch.setattr(json, "loads", parse)
    source._read_payload()
    source._read_payload()
    assert len(parses) == 1
    monkeypatch.setattr(config, "_CONFIG_GENERATION", config._CONFIG_GENERATION + 1)
    source._read_payload()
    assert len(parses) == 2


def test_guarded_json_target_shape_fallback_never_publishes(mcp_sources):
    source = mcp_sources[1]
    storage.admit_startup()
    source.path.write_text("42")
    assert source.load() == []
    assert not any(
        key[0]() is source
        for hold in storage._holds.values()
        for key in hold.json_evidence
    )
