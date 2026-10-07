"""TASK-33560: actual warm raw readers must reuse complete positive evidence."""

# Imported fixture names intentionally match pytest parameter names.
# ruff: noqa: F811

from __future__ import annotations

import os
import sys
from collections import Counter

import pytest

from Tests.Backup_Recovery.test_admission_evidence_reuse import (
    MUTATIONS,
    _Tracer,
)
from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from Tests.Backup_Recovery.test_mcp_source_lifetimes import mcp_sources  # noqa: F401
from Tests.Backup_Recovery.test_participant_lifetimes import local_root  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap, generation_witnesses
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import Admission
from tldw_chatbook.Backup_Recovery.control_records import bind_profile

pytestmark = pytest.mark.skipif(os.name == "nt", reason="ordinary reuse is POSIX-only")


def _derivations(monkeypatch):
    """Call through every expensive seam; never replace its returned authority."""
    counts = Counter()
    active = [False]
    for owner, name in (
        (bootstrap, "startup_permission"),
        (bootstrap, "_control_records"),
        (bootstrap, "_registry"),
        (storage, "_scope"),
        (Admission, "_read"),
        (Admission, "_groups"),
        (raw, "_open_verified_parent"),
    ):
        original = getattr(owner, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            if active[0]:
                counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)
    return counts, active


def _warm(call):
    for _ in range(3):
        call()


@pytest.mark.parametrize("index", range(5))
def test_actual_warm_mcp_readers_skip_all_admission_derivations(
    mcp_sources,
    monkeypatch,
    index,
):
    """Catches the witness handshake, native group walk and raw pin walk."""
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    source = mcp_sources[index]
    read = source.read_recent if index == 4 else source.load
    _warm(read)
    counts, active = _derivations(monkeypatch)
    active[0] = True
    result = read()
    active[0] = False
    assert result is not None
    assert not counts, f"warm reader {index} still derives: {dict(counts)}"
    assert not storage._raw_operations
    assert not any(source.path.parent.glob(".*.tmp"))


def test_actual_warm_native_probe_reuses_group_but_observes_native_gates(
    local_scope,
    monkeypatch,
):
    """Catches registry/group rereads; a later native request must remain visible."""
    import threading
    import time

    from tldw_chatbook.Backup_Recovery.admission import AdmissionCancelled

    root, selected, _, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage() as lease:
        hold = storage._holds[lease._key]
        _warm(storage._local_pause_requested)
        counts, active = _derivations(monkeypatch)
        active[0] = True
        assert storage._local_pause_requested() is False
        active[0] = False
        assert not counts, f"warm native probe still derives: {dict(counts)}"
        cancelled = threading.Event()
        failures = []

        def request():
            try:
                with hold.authority.maintenance(hold.names, 5, cancel=cancelled):
                    failures.append("entered while ordinary lease was live")
            except AdmissionCancelled:
                pass
            except (OSError, ValueError, RuntimeError) as error:
                failures.append(type(error).__name__)

        worker = threading.Thread(target=request)
        worker.start()
        try:
            deadline = time.monotonic() + 3
            while not storage._local_pause_requested():
                assert time.monotonic() < deadline, (
                    "warm false result hid native intent"
                )
                time.sleep(0.01)
        finally:
            cancelled.set()
            worker.join(5)
        assert not worker.is_alive() and not failures, failures
        assert storage._local_pause_requested() is False


def _verdict(call):
    try:
        return "allowed", call()
    except (OSError, ValueError, RuntimeError) as error:
        return "refused", type(error).__name__, str(error)


@pytest.mark.parametrize("route", ["witness", "pause"])
@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_complete_witness_and_pause_reuse_matches_original_mutation_verdict(
    local_scope,
    monkeypatch,
    tmp_path,
    route,
    mutation,
):
    """Require an engaged cache before comparing exact post-mutation outcomes."""
    root, selected, data, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    path = data / "store.json"
    with storage.acquire_storage(path) as lease:
        call = lambda: (
            generation_witnesses._witnesses(path, lease)
            if route == "witness"
            else storage._local_pause_requested()
        )
        _warm(call)
        counts, active = _derivations(monkeypatch)
        active[0] = True
        before = _verdict(call)
        active[0] = False
        assert before[0] == "allowed"
        assert not counts, f"oracle cache never engaged: {dict(counts)}"
        try:
            MUTATIONS[mutation](root, selected, data, tmp_path)
            reused = _verdict(call)
            monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
            original = _verdict(call)
            assert reused == original, (mutation, route, reused, original)
        finally:
            tmp_path.chmod(0o700)


def test_witness_result_mutation_cannot_change_the_next_reader(
    local_scope,
    monkeypatch,
):
    """Catches returning cached mutable witness lists to ordinary consumers."""
    _, _, data, _ = local_scope
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    path = data / "store.json"
    with storage.acquire_storage(path) as lease:
        call = lambda: generation_witnesses._witnesses(path, lease)
        _warm(call)
        first = call()
        assert first == []
        first.append({"generation": "caller-forged"})
        assert call() == []


def _hold_evidence(hold):
    """Collect the existing evidence type, without assuming a new cache layout."""
    found = []
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, storage._Evidence):
            found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    for value in vars(hold).values():
        visit(value)
    return found


def _unstamped(paths, evidence):
    stamped = {
        str(path)
        for entry in evidence
        for group in (entry.posture, entry.content)
        for path, _ in group
    }
    directories = {
        str(path)
        for entry in evidence
        for path, stamp in entry.content
        if stamp is not None and os.path.isdir(path)
    }
    return sorted(
        path
        for path in paths
        if path not in stamped
        and not (os.path.dirname(path) in directories and not os.path.lexists(path))
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="native FD paths use F_GETPATH")
def test_witness_cache_covers_every_input_of_the_complete_original_reader(
    local_scope,
    monkeypatch,
):
    """Catches a cache that omits records, registry or any skipped pin chain."""
    root, selected, data, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    path = data / "store.json"
    with storage.acquire_storage(path) as lease:
        call = lambda: generation_witnesses._witnesses(path, lease)
        _warm(call)
        counts, active = _derivations(monkeypatch)
        active[0] = True
        assert call() == []
        active[0] = False
        assert not counts, f"completeness cache never engaged: {dict(counts)}"
        evidence = _hold_evidence(storage._holds[lease._key])
        tracer = _Tracer(monkeypatch)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        tracer.active = True
        try:
            assert call() == []
        finally:
            tracer.active = False
        assert str(root / "admission" / "registry.json") in tracer.paths
        assert not _unstamped(tracer.paths, evidence)
        # The negative control changes the coverage input, never production.
        omitted = str(root / "admission" / "registry.json")
        shadow = []
        for entry in evidence:
            copy = storage._Evidence(entry.names, (), ())
            copy.posture = tuple(
                pair for pair in entry.posture if str(pair[0]) != omitted
            )
            copy.content = tuple(
                pair for pair in entry.content if str(pair[0]) != omitted
            )
            shadow.append(copy)
        assert omitted in _unstamped(tracer.paths, shadow)


def _activation(root, authority, selector, namespace, control, operation):
    from tldw_chatbook.Backup_Recovery.activation import bind_activation
    from tldw_chatbook.Backup_Recovery.control_records import register_pending

    control.mkdir(mode=0o700)
    register_pending(root, operation, (namespace,), control, (selector,))
    with authority.maintenance((namespace,), 2) as session:
        bind_activation(
            root, operation, selector, "generation", ("mcp.local",), session=session
        )
    (root / ("pending-" + bootstrap._key(operation) + ".json")).unlink()
    return control / "activation"


@pytest.mark.parametrize("damage", ["bytes", "mode", "replace"])
def test_warm_witness_requirements_have_exact_original_refusals(
    local_scope,
    tmp_path,
    monkeypatch,
    damage,
):
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore

    root, selected, data, authority = local_scope
    activation = ActivationStore(
        _activation(
            root, authority, selected, "profile", tmp_path / "control", "restore"
        )
    )
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    path = data / "source.json"
    with storage.acquire_storage(path) as lease:
        call = lambda: generation_witnesses._witnesses(path, lease)
        _warm(call)
        counts, active = _derivations(monkeypatch)
        active[0] = True
        value = call()
        active[0] = False
        assert len(value) == 1 and not counts
        value[0]["owners"].append("caller-forged")
        assert "caller-forged" not in call()[0]["owners"]
        requirement = activation._generation("generation") / "required.json"
        if damage == "bytes":
            requirement.write_bytes(b"{")
        elif damage == "mode":
            requirement.chmod(0o644)
        else:
            replacement = requirement.with_name("replacement.json")
            replacement.write_bytes(requirement.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, requirement)
        reused = _verdict(call)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        assert reused == _verdict(call)
        if damage != "replace":
            assert reused[0] == "refused"


@pytest.mark.skipif(sys.platform != "darwin", reason="maps native FD paths")
def test_complete_historical_chain_witness_falls_back_on_retarget(
    local_scope,
    tmp_path,
    monkeypatch,
):
    root, selected, data, authority = local_scope
    old = tmp_path / "old"
    current = tmp_path / "foreign-current"
    old.mkdir(mode=0o700)
    current.mkdir(mode=0o700)
    other = tmp_path / "foreign.toml"
    other.write_text("foreign")
    other.chmod(0o600)
    authority.register("foreign", (other, old))
    authority.remap("foreign", (other, current), 2)
    _activation(
        root,
        authority,
        other,
        "foreign",
        tmp_path / "foreign-control",
        "foreign-restore",
    )
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    path = data / "source.json"
    with storage.acquire_storage(path) as lease:
        call = lambda: generation_witnesses._witnesses(path, lease)
        _warm(call)
        counts, active = _derivations(monkeypatch)
        active[0] = True
        assert call() == []
        active[0] = False
        assert not counts
        evidence = _hold_evidence(storage._holds[lease._key])
        tracer = _Tracer(monkeypatch)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        tracer.active = True
        try:
            assert call() == []
        finally:
            tracer.active = False
        assert str(old) in tracer.paths and not _unstamped(tracer.paths, evidence)
        shadow = []
        for entry in evidence:
            copy = storage._Evidence(entry.names, (), ())
            copy.posture = tuple(pair for pair in entry.posture if pair[0] != old)
            copy.content = tuple(pair for pair in entry.content if pair[0] != old)
            shadow.append(copy)
        assert str(old) in _unstamped(tracer.paths, shadow)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", True)
        registry_before = (root / "admission" / "registry.json").read_bytes()
        old.rename(tmp_path / "preserved-old")
        old.symlink_to(data, target_is_directory=True)
        assert (root / "admission" / "registry.json").read_bytes() == registry_before
        reused = _verdict(call)
        assert reused == (
            "refused",
            "ValueError",
            "projection_source_scope_unavailable",
        )
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        assert _verdict(call) == reused


@pytest.fixture
def bound_config(local_scope, monkeypatch):
    from Tests.Backup_Recovery.config_test_support import install_config_source

    root, selected, data, authority = local_scope
    selected.write_text(
        f'[general]\nusers_name="fixture"\n[paths]\ndata_dir="{data.as_posix()}"\n'
    )
    selected.chmod(0o600)
    (data / "fixture").mkdir(mode=0o700)
    bind_profile(root, selected, ("profile",), root / "admission")
    config = install_config_source(monkeypatch)
    try:
        yield root, config, authority
    finally:
        lease = storage._startups.pop((os.getpid(), str(root)), None)
        if lease is not None:
            lease.close()


def test_warm_bound_config_companions_keep_foreign_member_refusal(
    bound_config, monkeypatch
):
    _, config, authority = bound_config
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    read = config.read_cli_config_serialized
    _warm(read)
    counts, active = _derivations(monkeypatch)
    active[0] = True
    assert read() is not None
    active[0] = False
    assert not counts, dict(counts)
    selected = config._get_effective_config_path()
    companion = config._advanced_backup_path(selected)
    companion.write_bytes(b"foreign bytes")
    companion.chmod(0o600)
    authority.register("foreign", (companion,))
    reused = _verdict(read)
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    assert reused == _verdict(read)
    assert reused[0] == "refused" and companion.read_bytes() == b"foreign bytes"


def test_warm_bound_history_keeps_fresh_temporary_authority(bound_config, monkeypatch):
    from tldw_chatbook.Backup_Recovery import mcp_source_participants
    from tldw_chatbook.MCP.execution_log import MCPExecutionLog

    _, config, _ = bound_config
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    source = MCPExecutionLog(config.get_user_data_dir() / "mcp_execution_log.jsonl")
    _warm(source.read_recent)
    counts, active = _derivations(monkeypatch)
    selections = []
    original = mcp_source_participants.members

    def members(*args):
        value = original(*args)
        selections.append(value)
        return value

    monkeypatch.setattr(mcp_source_participants, "members", members)
    active[0] = True
    assert source.read_recent() == []
    active[0] = False
    assert not counts, f"fresh history children still rederive: {dict(counts)}"
    assert len(selections) == 1 and len(selections[0][1]) == 2
    assert all(not path.exists() for path in selections[0][1].values())
    assert not storage._raw_operations


@pytest.mark.parametrize(
    "mutation", ["ancestor_mode", "ancestor_replace", "child_collision"]
)
def test_fresh_descendant_reuse_matches_complete_original(
    local_scope,
    tmp_path,
    monkeypatch,
    mutation,
):
    root, selected, data, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage(data / "seed.json"):
        _warm(lambda: _verdict(lambda: _acquire_result(data / "seed.json")))
        counts, active = _derivations(monkeypatch)
        active[0] = True
        assert _acquire_result(data / "fresh-before.json") == ("profile",)
        active[0] = False
        assert not counts, dict(counts)
        child = data / "fresh-after.json"
        if mutation == "ancestor_mode":
            tmp_path.chmod(0o777)
        elif mutation == "ancestor_replace":
            data.rename(tmp_path / "preserved-data")
            data.mkdir(mode=0o700)
        else:
            other = tmp_path / "foreign.json"
            other.write_text("foreign")
            child.symlink_to(other)
        try:
            reused = _verdict(lambda: _acquire_result(child))
            monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
            assert reused == _verdict(lambda: _acquire_result(child))
            if mutation == "child_collision":
                assert reused[0] == "refused"
        finally:
            tmp_path.chmod(0o700)


def _acquire_result(path):
    with storage.acquire_storage(path) as lease:
        return lease.execution_context(path)[1]


def test_file_only_admission_never_proves_its_parent(local_scope, monkeypatch):
    root, selected, data, authority = local_scope
    file = data / "only.json"
    file.write_text("only")
    file.chmod(0o600)
    authority.remap("profile", (selected, file), 2)
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage(file) as lease:
        _warm(lambda: _acquire_result(file))
        directory_key = ("directory", str(selected), str(data))
        assert directory_key not in storage._holds[lease._key].path_evidence
        child = data / "unowned.json"
        reused = _verdict(lambda: _acquire_result(child))
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        assert reused == _verdict(lambda: _acquire_result(child))
        assert reused == ("refused", "RecoveryRequired", "storage_scope_not_enrolled")


def test_fresh_descendant_foreign_inode_cannot_skip_source_scope(
    local_scope,
    tmp_path,
    monkeypatch,
):
    root, selected, data, authority = local_scope
    other = tmp_path / "foreign.toml"
    other.write_text("foreign")
    other.chmod(0o600)
    foreign = tmp_path / "foreign-data"
    foreign.mkdir(mode=0o700)
    source = foreign / "owned.json"
    source.write_text("owned")
    source.chmod(0o600)
    authority.register("foreign", (other, source))
    _activation(
        root,
        authority,
        other,
        "foreign",
        tmp_path / "foreign-control",
        "foreign-restore",
    )
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage(data / "seed.json"):
        _warm(lambda: _acquire_result(data / "seed.json"))
        child = data / "fresh-hardlink.json"
        os.link(source, child)

        def call():
            with storage.acquire_storage(child) as lease:
                return generation_witnesses._witnesses(child, lease)

        reused = _verdict(call)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        assert reused == _verdict(call)
        assert reused == (
            "refused",
            "ValueError",
            "projection_source_scope_unavailable",
        )


def test_warm_raw_operation_owns_and_retires_actual_parent_fd(mcp_sources, monkeypatch):
    from tldw_chatbook.Backup_Recovery import mcp_source_participants

    source = mcp_sources[1]
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    _warm(source.load)
    with raw._scope(source, mcp_source_participants.ROUTE, writing=True) as operation:
        state = raw._states[operation]
        fd = state.pins[source.path.parent]
        actual = os.fstat(fd)
        current = source.path.parent.stat()
        assert (actual.st_dev, actual.st_ino) == (current.st_dev, current.st_ino)
        raw._check(operation)
    with pytest.raises(OSError):
        os.fstat(fd)
    assert operation not in raw._states and operation not in storage._raw_operations


def test_wrong_warm_parent_fd_is_closed_before_original_fallback(
    mcp_sources, tmp_path, monkeypatch
):
    from tldw_chatbook.Utils import private_paths

    source = mcp_sources[1]
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    _warm(source.load)
    other = tmp_path / "other-parent"
    other.mkdir(mode=0o700)
    original = private_paths._native_open
    rejected = []
    full = raw._open_verified_parent
    fallback = []

    def opened(path, *args, **kwargs):
        if path == source.path.parent and not rejected:
            fd = original(other, *args, **kwargs)
            rejected.append(fd)
            return fd
        return original(path, *args, **kwargs)

    def walk(*args, **kwargs):
        fallback.append(True)
        return full(*args, **kwargs)

    monkeypatch.setattr(private_paths, "_native_open", opened)
    monkeypatch.setattr(raw, "_open_verified_parent", walk)
    assert source.load() is not None
    assert len(rejected) == len(fallback) == 1
    with pytest.raises(OSError):
        os.fstat(rejected[0])
    assert not storage._raw_operations


def test_parent_swap_during_open_has_the_same_original_refusal(
    mcp_sources, tmp_path, monkeypatch
):
    from tldw_chatbook.Utils import private_paths

    source = mcp_sources[1]
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    _warm(source.load)
    original_open = private_paths._native_open
    original_walk = raw._open_verified_parent
    parent = source.path.parent
    preserved = tmp_path / "preserved-parent"
    verdicts = []
    for reuse in (True, False):
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", reuse)
        swapped = []

        def swap(swapped=swapped):
            parent.rename(preserved)
            parent.mkdir(mode=0o700)
            swapped.append(True)

        def opened(path, *args, _swapped=swapped, _swap=swap, **kwargs):
            fd = original_open(path, *args, **kwargs)
            if path == parent and not _swapped:
                _swap()
            return fd

        def walked(*args, _reuse=reuse, _swapped=swapped, _swap=swap, **kwargs):
            if not _reuse and not _swapped:
                _swap()
            return original_walk(*args, **kwargs)

        monkeypatch.setattr(private_paths, "_native_open", opened)
        monkeypatch.setattr(raw, "_open_verified_parent", walked)
        try:
            verdicts.append(_verdict(source.load))
            assert swapped and not storage._raw_operations
        finally:
            parent.rmdir()
            preserved.rename(parent)
    assert (
        verdicts[0]
        == verdicts[1]
        == ("refused", "RecoveryRequired", "raw_parent_identity_changed")
    )


def test_reuse_stamp_io_never_runs_under_coordinator_lock(local_scope, monkeypatch):
    root, selected, data, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage(data / "seed.json"):
        _warm(lambda: _acquire_result(data / "seed.json"))
        observations = []
        for name in ("_posture", "_content", "_mount_read_only"):
            original = getattr(storage, name)

            def observed(*args, _original=original, _name=name, **kwargs):
                assert not storage._lock._is_owned(), _name
                observations.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(storage, name, observed)
        assert _acquire_result(data / "fresh.json") == ("profile",)
        assert observations and "_mount_read_only" in observations
        # Prove the oracle rejects stamp work inside the coordinator lock.
        with storage._lock, pytest.raises(AssertionError, match="_posture"):
            storage._posture(data)


_CUSTODY_SCRIPT = r"""
import os, sys, time
from pathlib import Path
from Tests.network_guard import install
install()
selected = Path(os.environ['TLDW_CONFIG_PATH'])
base = selected.parent.parent
selected.write_text('[general]\nusers_name="fixture"\n[paths]\ndata_dir="'+str(base/'data')+'"\n')
selected.chmod(0o600)
(base/'data'/'fixture').mkdir(mode=0o700)
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionTimeout
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
mode, timing = sys.argv[1:]
storage._EVIDENCE_SETTLE_NS = 0
source = (MCPPermissionStore(config.get_user_data_dir()/'mcp_permissions.json')
          if mode=='close' else ConfiguredServerTargetStore(config.get_user_data_dir()/'mcp_server_targets.json'))
if mode=='publish': source.save_targets([])
for _ in range(4): source.load()
root = bootstrap.default_bootstrap_root()
key = (os.getpid(),str(root))
assert storage._holds[key].derived_evidence
walk = raw._open_verified_parent
walks = []
def walked(*args, **kwargs):
 walks.append(True)
 return walk(*args, **kwargs)
raw._open_verified_parent = walked
if mode=='close':
 pin = raw._pin_parent
 target = []
 def pinned(state, anchor):
  fd = pin(state, anchor)
  if state.source is source: target.append(fd)
  return fd
 raw._pin_parent = pinned
 close = os.close
 def uncertain(fd):
  if target and fd==target[-1]:
   if timing=='after': close(fd)
   raise OSError('task33560 close outcome unknown')
  return close(fd)
 os.close = uncertain
 try:
  source.load()
 except bootstrap.RecoveryRequired as error:
  assert str(error)=='raw_resources_not_retired'
 else: raise AssertionError('warm parent uncertainty was retired')
 finally: os.close=close
 assert target and not walks
 if timing=='before': assert os.fstat(target[-1])
 else:
  try: os.fstat(target[-1])
  except OSError: pass
  else: raise AssertionError('native close did not happen')
else:
 foreign=base/'foreign.json';foreign.write_bytes(b'foreign bytes');foreign.chmod(0o600)
 replace=raw._replace
 changed=[]
 def replaced(operation, temporary, destination):
  if destination==source.path and not changed:
   foreign.replace(destination);changed.append(True)
  return replace(operation,temporary,destination)
 raw._replace=replaced
 try: source.save_targets([])
 except bootstrap.RecoveryRequired: pass
 else: raise AssertionError('warm publication adopted foreign destination')
 assert changed and source.path.read_bytes()==b'foreign bytes'
 assert not walks
# The warm operation's own uncertainty retains its independent tokens even after
# positively retiring this child's quiescent startup owner.
if mode=='close':
 state=next(state for state in raw._states.values() if state.source is source)
 assert state.uncertain and state.descriptors and state.pins
 assert all(lease in storage._live_leases for lease in state.leases)
 storage._startups.pop(key).close()
 assert key in storage._holds
 participant=raw._raw_participant(source);participant.close_admission()
 assert not participant.drain(time.monotonic()+.02)
 pause=storage._begin_local_pause()
 try:
  assert not pause.drain(time.monotonic()+.02)
  try:
   with Admission.open_existing(root/'admission').maintenance(storage._holds[key].names,.04):
    raise AssertionError('native maintenance crossed uncertain owner')
  except AdmissionTimeout: pass
 finally: pause.resume()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "mode,timing", [("close", "before"), ("close", "after"), ("publish", "before")]
)
def test_warm_descriptor_uncertainty_and_publication_keep_original_custody(
    tmp_path, mode, timing
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, mode, timing, script=_CUSTODY_SCRIPT, timeout=30)


def test_concurrent_full_witness_reads_cannot_demote_confirmed_evidence(
    local_scope, monkeypatch
):
    import threading

    root, selected, data, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage(data / "source.json") as lease:
        started = threading.Event()
        proceed = threading.Event()
        failures = []
        original = storage._note_derived

        def note(hold, key, entry, result, before):
            if (
                key[0] == "witness"
                and threading.current_thread().name == "early-cold-witness"
            ):
                started.set()
                assert proceed.wait(5)
            return original(hold, key, entry, result, before)

        monkeypatch.setattr(storage, "_note_derived", note)
        call = lambda: generation_witnesses._witnesses(data / "source.json", lease)

        def early():
            try:
                assert call() == []
            except (OSError, ValueError, RuntimeError, AssertionError) as error:
                failures.append(type(error).__name__)

        worker = threading.Thread(target=early, name="early-cold-witness")
        worker.start()
        try:
            assert started.wait(5)
            assert call() == []
            assert call() == []
        finally:
            proceed.set()
            worker.join(5)
        assert not worker.is_alive() and not failures
        counts, active = _derivations(monkeypatch)
        active[0] = True
        assert call() == []
        active[0] = False
        assert not counts, "an older concurrent derivation demoted warm evidence"


def test_witness_content_still_requires_the_original_settle_margin(
    local_scope, monkeypatch
):
    root, selected, data, _ = local_scope
    bind_profile(root, selected, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 24 * 60 * 60 * 1_000_000_000)
    with storage.acquire_storage(data / "source.json") as lease:
        call = lambda: generation_witnesses._witnesses(data / "source.json", lease)
        _warm(call)
        counts, active = _derivations(monkeypatch)
        active[0] = True
        assert call() == []
        active[0] = False
        assert counts["startup_permission"] and counts["_control_records"]


@pytest.mark.parametrize("reuse", (True, False))
def test_actual_permission_parse_pause_matches_original_refusal(
    mcp_sources, monkeypatch, reuse
):
    import json

    source = mcp_sources[3]
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    source.set_kill_switch(True)
    source.set_kill_switch(False)
    assert source.path.exists(), "a default False no-op alone leaves no document"
    _warm(source.load)
    counts, active = _derivations(monkeypatch)
    active[0] = True
    source.load()
    active[0] = False
    assert not counts, "the actual source must be warm before the pause stimulus"
    original = json.loads
    pauses = []

    def parse_then_pause(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(result, dict) and "kill_switch" in result and not pauses:
            pauses.append(storage._begin_local_pause())
        return result

    before = source.path.read_bytes()
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", reuse)
    monkeypatch.setattr(json, "loads", parse_then_pause)
    try:
        with pytest.raises(bootstrap.RecoveryRequired) as refusal:
            source.set_kill_switch(True)
        # Binding validation starts a fresh original acquisition after pause.
        # This task preserves that existing refusal and never publishes bytes.
        assert str(refusal.value) == "storage_locally_paused"
        assert len(pauses) == 1
        assert source.path.read_bytes() == before
    finally:
        for pause in pauses:
            pause.resume()
