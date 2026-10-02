"""PERF-07/08 (TASK-33266/33267): reused admission evidence never changes a verdict.

ADR-126's 2026-09-29 amendment lets ordinary storage admission reuse the allowed
result of an unmodified derivation while per-call stamps stay identical. The
oracle below applies each mutation in a catalog, then compares:

* ``acquire_storage`` with evidence reuse available, against
* ``acquire_storage`` with reuse switched off (the full derivation).

They must agree exactly: the same namespaces, or the same refusal reason. Some
writers run in a subprocess, so no in-process hook can be what notices them;
only the stamps can. The settle margin is patched to 0 so evidence recorded
milliseconds ago is reusable, the worst case for coarse timestamps.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, bind_profile

# Evidence reuse is POSIX-only (_acquire_storage gates it on os.name != "nt"):
# on Windows every acquisition derives, so the reuse assertions cannot hold and
# the reuse-vs-derivation oracles compare the derivation with itself (Qodo, #2919).
pytestmark = pytest.mark.skipif(os.name == "nt", reason="evidence reuse is POSIX-only")

REPO = Path(__file__).resolve().parents[2]


def _verdict(path: Path) -> tuple[str, object]:
    """Acquire once and release; report what the caller would have observed."""
    try:
        with storage.acquire_storage(path) as lease:
            return ("allowed", lease.execution_context(path)[1])
    except bootstrap.RecoveryRequired as error:
        return ("refused", str(error))


def _in_subprocess(code: str, **values: str) -> None:
    """Run a protocol writer in another process, so no in-process hook fires."""
    script = "import json, sys\nvalues = json.loads(sys.argv[1])\n" + code
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(values)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def _pending_from_subprocess(root, config, data, tmp_path):
    _in_subprocess(
        "from pathlib import Path\n"
        "from tldw_chatbook.Backup_Recovery.control_records import register_pending\n"
        "register_pending(Path(values['root']), 'op', ('profile',),"
        " Path(values['control']), (Path(values['config']),))\n",
        root=str(root), control=str(tmp_path / "control"), config=str(config),
    )


def _unrelated_pending_from_subprocess(root, config, data, tmp_path):
    other = tmp_path / "other.toml"
    other.write_text("other")
    elsewhere = tmp_path / "elsewhere-data"
    elsewhere.mkdir(mode=0o700)
    _in_subprocess(
        "from pathlib import Path\n"
        "from tldw_chatbook.Backup_Recovery.admission import Admission\n"
        "from tldw_chatbook.Backup_Recovery.control_records import register_pending\n"
        "Admission.open_existing(Path(values['root']) / 'admission')"
        ".register('elsewhere', (Path(values['elsewhere']),))\n"
        "register_pending(Path(values['root']), 'op2', ('elsewhere',),"
        " Path(values['control']), (Path(values['other']),))\n",
        root=str(root), control=str(tmp_path / "control2"),
        other=str(other), elsewhere=str(elsewhere),
    )


def _registry_remap_from_subprocess(root, config, data, tmp_path):
    moved = tmp_path / "moved"
    moved.mkdir(mode=0o700)
    _in_subprocess(
        "from pathlib import Path\n"
        "from tldw_chatbook.Backup_Recovery.admission import Admission\n"
        "Admission.open_existing(Path(values['root']) / 'admission')"
        ".register('extra', (Path(values['moved']),))\n",
        root=str(root), moved=str(moved),
    )


def _registry_intent(root, config, data, tmp_path):
    intent = root / "admission" / "registry.pending.json"
    intent.write_text("{}")
    intent.chmod(0o600)


def _profile_edited_in_place(root, config, data, tmp_path):
    record = next(root.glob("profile-*.json"))
    body = json.loads(record.read_text())
    body["roots"] = sorted([*body["roots"], str(tmp_path / "grafted")])
    with open(record, "r+", encoding="utf-8") as stream:  # same inode, new bytes
        stream.seek(0)
        stream.write(json.dumps(body))
        stream.truncate()


def _marker_replaced(root, config, data, tmp_path):
    marker = root / "unbound-owner"
    replacement = root / "unbound-owner.new"
    replacement.write_bytes(marker.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, marker)


def _selector_edited(root, config, data, tmp_path):
    config.write_text("scope2")


def _ancestor_group_writable(root, config, data, tmp_path):
    tmp_path.chmod(0o777)


def _data_renamed_and_recreated(root, config, data, tmp_path):
    data.rename(tmp_path / "data.old")
    data.mkdir(mode=0o700)


def _data_swapped_for_symlink(root, config, data, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    data.rename(tmp_path / "data.old")
    data.symlink_to(target, target_is_directory=True)


def _bootstrap_root_removed(root, config, data, tmp_path):
    shutil.rmtree(root)


def _nothing(root, config, data, tmp_path):
    pass


MUTATIONS = {
    "control-no-change": _nothing,
    "pending-record-from-subprocess": _pending_from_subprocess,
    "unrelated-pending-from-subprocess": _unrelated_pending_from_subprocess,
    "registry-replaced-from-subprocess": _registry_remap_from_subprocess,
    "registry-intent-file": _registry_intent,
    "profile-record-edited-in-place": _profile_edited_in_place,
    "enrollment-marker-replaced": _marker_replaced,
    "config-selector-edited": _selector_edited,
    "ancestor-made-group-writable": _ancestor_group_writable,
    "data-dir-renamed-and-recreated": _data_renamed_and_recreated,
    "data-dir-swapped-for-symlink": _data_swapped_for_symlink,
    "bootstrap-root-removed": _bootstrap_root_removed,
}

#: Mutations the full derivation is known to refuse. Pinned so the oracle can
#: never pass vacuously by having both sides allow everything.
REFUSED_BY_THE_DERIVATION = {
    "pending-record-from-subprocess",
    "registry-intent-file",
    "profile-record-edited-in-place",
    "enrollment-marker-replaced",
    "ancestor-made-group-writable",
}


@pytest.fixture
def reuse_switch(monkeypatch):
    """Toggle evidence reuse; a no-op until the reuse path exists.

    Args:
        monkeypatch: Sets the settle margin to zero and the reuse switch.

    Returns:
        A callable taking ``enabled``; it turns evidence reuse on or off for
        the rest of the test.
    """
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0, raising=False)

    def switch(enabled: bool) -> None:
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", enabled, raising=False)

    return switch


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_reused_evidence_matches_the_full_derivation(
    local_scope, reuse_switch, tmp_path, mutation, bound  # noqa: F811
):
    root, config, data, _ = local_scope
    if mutation == "profile-record-edited-in-place" and not bound:
        pytest.skip("an unbound selection has no profile record to edit")
    if bound:
        bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    reuse_switch(True)
    # A live lease keeps the native hold -- and any evidence on it -- alive, as
    # the startup enrollment does in the running app.
    startup = storage.acquire_storage()
    try:
        for _ in range(2):  # record evidence, then serve from it
            assert _verdict(target)[0] == "allowed"
        try:
            MUTATIONS[mutation](root, config, data, tmp_path)
            reused = _verdict(target)
            reuse_switch(False)
            derived = _verdict(target)
        finally:
            tmp_path.chmod(0o700)
    finally:
        startup.close()

    assert reused == derived, f"{mutation}: reuse gave {reused}, derivation {derived}"
    if mutation in REFUSED_BY_THE_DERIVATION:
        assert derived[0] == "refused", f"{mutation} was expected to refuse: {derived}"


def _count_derivations(monkeypatch) -> dict[str, int]:
    """Count entries into the derivation's two expensive phases."""
    calls = {"permission": 0, "scope": 0}
    permission, scope = bootstrap.startup_permission, storage._scope

    def counted_permission(*args, **kwargs):
        calls["permission"] += 1
        return permission(*args, **kwargs)

    def counted_scope(*args, **kwargs):
        calls["scope"] += 1
        return scope(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "startup_permission", counted_permission)
    monkeypatch.setattr(storage, "_scope", counted_scope)
    return calls


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
def test_a_warm_acquisition_reuses_evidence_without_rederiving(
    local_scope, reuse_switch, monkeypatch, bound  # noqa: F811
):
    """Two bracketing derivations confirm the evidence; the next call skips both."""
    root, config, data, _ = local_scope
    if bound:
        bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        calls = _count_derivations(monkeypatch)
        warm = _verdict(target)
    finally:
        startup.close()

    assert warm[0] == "allowed"
    assert calls == {"permission": 0, "scope": 0}, calls


@pytest.mark.parametrize("change", ["pause", "selector"])
def test_warm_observation_rechecks_selection_and_cancellation_before_io(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
    change,
):
    """A change during out-of-lock validation must fence the counted borrower."""
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "not-written"
    reuse_switch(True)
    startup = storage.acquire_storage()
    pauses = []
    try:
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        original = storage._Evidence.observe

        def changed(evidence, *args, **kwargs):
            result = original(evidence, *args, **kwargs)
            monkeypatch.setattr(storage._Evidence, "observe", original)
            if change == "pause":
                pauses.append(storage._begin_local_pause())
            else:
                monkeypatch.setenv("TLDW_CONFIG_PATH", str(config.with_name("other")))
            return result

        monkeypatch.setattr(storage._Evidence, "observe", changed)
        with pytest.raises(bootstrap.RecoveryRequired), storage.acquire_storage(target):
            target.write_text("unexpected")
        assert not target.exists()
    finally:
        for pause in pauses:
            pause.resume()
        startup.close()


def test_blocked_scope_validation_allows_unrelated_transaction(
    local_scope,  # noqa: F811
    monkeypatch,
):
    """An owner's filesystem validation must not own the global mutex."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Notifications.event_state_repository import EventStateRepository

    _, _, data, _ = local_scope
    repository = EventStateRepository(data / "other.sqlite")
    blocked, release, committed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    selected = data / "blocked.sqlite"
    original = storage._scope
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)

    def scoped(root, selector, path, **kwargs):
        if path == selected and not release.is_set():
            blocked.set()
            assert release.wait(5), "test did not release scope validation"
        return original(root, selector, path, **kwargs)

    def acquire():
        with storage.acquire_storage(selected):
            pass

    def transact():
        with repository.transaction() as connection:
            connection.execute("CREATE TABLE admission_progress (value INTEGER)")
            connection.execute("INSERT INTO admission_progress VALUES (42)")
        committed.set()

    monkeypatch.setattr(storage, "_scope", scoped)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(acquire)
            assert blocked.wait(5)
            second = pool.submit(transact)
            try:
                assert committed.wait(1), (
                    "unrelated transaction blocked behind validation"
                )
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        with repository.transaction() as connection:
            assert (
                connection.execute("SELECT value FROM admission_progress").fetchone()[0]
                == 42
            )
    finally:
        release.set()
        repository.close()


def test_evidence_newer_than_the_settle_margin_is_not_reused(
    local_scope, monkeypatch  # noqa: F811
):
    """Coarse clocks: stamps over a just-written file are never trusted yet."""
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 3_600 * 10**9)
    target = data / "store.db"
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        calls = _count_derivations(monkeypatch)
        assert _verdict(target)[0] == "allowed"
    finally:
        startup.close()

    assert calls["permission"] >= 1, "evidence inside the settle margin was reused"


class _Tracer:
    """Record every absolute path the admission derivation touches (macOS only).

    ``os`` functions are wrapped, and the wrappers are added to
    ``os.supports_dir_fd``/``supports_follow_symlinks`` so the private-path
    walk's capability check still passes. Descriptor-relative names are made
    absolute with ``F_GETPATH``; ``open()`` of plain paths comes from the
    ``open`` audit event.
    """

    def __init__(self, monkeypatch) -> None:
        import fcntl

        self.active = False
        self.paths: set[str] = set()
        self._fcntl = fcntl

        def absolute(target, dir_fd=None) -> None:
            if not self.active:
                return
            if isinstance(target, int):
                self.paths.add(self._fd_path(target))
            elif isinstance(target, (str, bytes, os.PathLike)):
                name = os.fsdecode(target)
                if dir_fd is not None and not os.path.isabs(name):
                    name = os.path.join(self._fd_path(dir_fd), name)
                self.paths.add(os.path.normpath(name))

        def wrap(original):
            def traced(target=".", *args, **kwargs):
                absolute(target, kwargs.get("dir_fd"))
                return original(target, *args, **kwargs)

            return traced

        wrapped = {name: wrap(getattr(os, name)) for name in
                   ("open", "stat", "lstat", "listdir", "readlink")}
        dir_fd = set(os.supports_dir_fd)
        follow = set(os.supports_follow_symlinks)
        fd_ok = set(os.supports_fd)
        for name, fn in wrapped.items():
            original = getattr(os, name)
            if original in dir_fd:
                dir_fd.add(fn)
            if original in follow:
                follow.add(fn)
            if original in fd_ok:
                fd_ok.add(fn)
            monkeypatch.setattr(os, name, fn)
        monkeypatch.setattr(os, "supports_dir_fd", dir_fd)
        monkeypatch.setattr(os, "supports_follow_symlinks", follow)
        monkeypatch.setattr(os, "supports_fd", fd_ok)

        def audit(event, args):
            # The open event omits dir_fd, so only absolute paths come from here;
            # descriptor-relative opens are recorded by the os.open wrapper.
            if (
                event == "open"
                and self.active
                and isinstance(args[0], (str, os.PathLike))
                and os.path.isabs(os.fsdecode(args[0]))
            ):
                absolute(args[0])

        sys.addaudithook(audit)  # inert whenever self.active is False

    def _fd_path(self, fd: int) -> str:
        raw = self._fcntl.fcntl(fd, self._fcntl.F_GETPATH, bytes(1024))
        return os.fsdecode(raw.split(b"\0", 1)[0])


@pytest.mark.skipif(sys.platform != "darwin", reason="maps descriptors with F_GETPATH")
@pytest.mark.parametrize("scope", ["bound", "unbound", "drifted"])
def test_the_evidence_stamps_every_path_the_derivation_reads(
    local_scope, reuse_switch, monkeypatch, scope  # noqa: F811
):
    """Dependency completeness: nothing the full derivation reads is unstamped.

    ``drifted`` is unbound with a profile on disk: the derivation fingerprints
    the selector to decide that, so its content is an input too.
    """
    root, config, data, _ = local_scope
    bound = scope == "bound"
    if scope != "unbound":
        bind_profile(root, config, ("profile",), root / "admission")
    if scope == "drifted":
        config.write_text("drifted")
    target = data / "store.db"
    reuse_switch(True)
    tracer = _Tracer(monkeypatch)
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        hold = storage._holds[(os.getpid(), str(root))]
        stamped = {
            str(p)
            for evidence in (
                hold.evidence[str(config)],
                *(hold.path_evidence.values() if bound else ()),
            )
            for group in (evidence.posture, evidence.content)
            for p, _ in group
        }
        reuse_switch(False)
        tracer.active = True
        try:
            assert _verdict(target)[0] == "allowed"
        finally:
            tracer.active = False
    finally:
        startup.close()

    content_dirs = {
        str(p)
        for p, stamp in hold.evidence[str(config)].content
        if stamp is not None and stat.S_ISDIR(os.stat(p).st_mode)
    }
    # _execution_selection_for resolves the selector's and the target's chains
    # on every call, the reuse path included. Resolving never reads the
    # selector's bytes, so the selector itself is not exempt.
    per_call = {str(p) for p in (*config.parents, *target.parents, target)}

    def covered(path: str) -> bool:
        if path in stamped or path in per_call:
            return True
        # An absent entry of a content-stamped directory: creating it changes
        # the directory's stamp.
        return os.path.dirname(path) in content_dirs and not os.path.lexists(path)

    assert str(root / "admission" / "registry.json") in tracer.paths, "trace saw nothing"
    unstamped = sorted(p for p in tracer.paths if not covered(p))
    assert not unstamped, "\n".join(unstamped)


def test_restoring_a_drifted_selector_is_not_served_from_unbound_evidence(
    local_scope, reuse_switch  # noqa: F811
):
    """A profile whose selector drifted leaves the hold unbound. Restoring the
    selector makes the derivation bind -- a scope change it refuses -- so the
    unbound evidence must notice the restore too (Qodo, #2919)."""
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    original = config.read_bytes()
    config.write_text("drifted")
    target = data / "store.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    try:
        for _ in range(2):
            assert _verdict(target)[0] == "allowed"
        config.write_bytes(original)
        reused = _verdict(target)
        reuse_switch(False)
        derived = _verdict(target)
    finally:
        startup.close()

    assert derived[0] == "refused", derived
    assert reused == derived, f"reuse gave {reused}, derivation {derived}"


@pytest.mark.parametrize("change", ["data-swapped-for-symlink", "epoch-advanced"])
def test_a_change_just_before_the_lease_is_counted_falls_back(
    local_scope, reuse_switch, monkeypatch, tmp_path, change  # noqa: F811
):
    """Reuse revalidates after counting its lease, as the derivation does, so a
    change landing between its first look and the count is seen (Qodo, #2919/#2924)."""
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    try:
        for _ in range(2):
            assert _verdict(target)[0] == "allowed"
        real = storage._mount_read_only  # runs between the first look and the count

        def race(path):
            monkeypatch.setattr(storage, "_mount_read_only", real)
            if change == "epoch-advanced":
                bootstrap.advance_admission_epoch()
            else:
                _data_swapped_for_symlink(root, config, data, tmp_path)
            return real(path)

        monkeypatch.setattr(storage, "_mount_read_only", race)
        calls = _count_derivations(monkeypatch)
        _verdict(target)  # a mid-call swap may refuse; the derivation decides
    finally:
        startup.close()

    assert calls["scope"] >= 1, "a lease was reused over a change made before it was counted"


def test_an_absence_proved_root_is_never_served_from_evidence(
    tmp_path, reuse_switch, monkeypatch
):
    """A deleted alias under an enrolled parent is admitted by the absence proof,
    which no stamp covers: re-creating it as a non-directory changes no parent
    posture. Reuse must not serve such a binding (Qodo, #2919).

    Args:
        tmp_path: Holds the bootstrap root, the data directory and its alias.
        reuse_switch: Turns evidence reuse on, then off for the derivation.
        monkeypatch: Points the bootstrap root and config selector at tmp_path.
    """
    root = tmp_path / "bootstrap"
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config = data / "config.toml"
    config.write_text('name = "profile"\n')
    alias = data / "history.jsonl"
    alias.write_text("")
    authority = admission_authority(root)
    authority.register("parent", (data,))
    authority.register("child", (alias,))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    bind_profile(root, config, ("parent", "child"), authority.control_root)
    alias.unlink()
    target = data / "store.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    try:
        for _ in range(2):
            assert _verdict(target)[0] == "allowed"
        os.mkfifo(alias)
        reused = _verdict(target)
        reuse_switch(False)
        derived = _verdict(target)
    finally:
        startup.close()

    assert derived[0] == "refused", derived
    assert reused == derived, f"reuse gave {reused}, derivation {derived}"


def test_a_failed_recheck_after_counting_closes_the_reused_lease(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    """An observation that raises after reuse counted its lease closes the lease,
    as the derivation's own failure path does, so the hold can still drain
    (Qodo, #2919).

    Args:
        local_scope: A private storage root, config and data directory.
        reuse_switch: Turns evidence reuse on.
        monkeypatch: Makes the post-count observation raise.
    """
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    try:
        for _ in range(2):
            assert _verdict(target)[0] == "allowed"
        held = sum(hold.count for hold in storage._holds.values())

        def unreadable(self, *args, **kwargs):
            raise PermissionError("an admitted directory became unreadable")

        monkeypatch.setattr(storage._Evidence, "observe", unreadable)
        with pytest.raises((PermissionError, bootstrap.RecoveryRequired)):
            with storage.acquire_storage(target):
                pass
        assert sum(hold.count for hold in storage._holds.values()) == held
    finally:
        startup.close()


def test_evidence_stamps_and_settle_margin_in_isolation(tmp_path, monkeypatch):
    """The stamp comparisons and the settle decision, without an acquisition."""
    record = tmp_path / "record"
    record.write_text("a")
    evidence = storage._Evidence(("n",), storage._chain(tmp_path), (record, tmp_path / "absent"))
    assert evidence.observe() == evidence.stamps()

    changed_at = evidence.content[0][1][4]
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 10)
    assert evidence.settled_before(changed_at + 10)  # an absent input is settled
    assert not evidence.settled_before(changed_at + 9)

    record.write_text("bb")
    assert evidence.observe() != evidence.stamps()
    (tmp_path / "absent").write_text("")
    record.write_text("a")
    assert evidence.observe()[1][1] is not None  # appearing is a mismatch too

    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    assert storage._path_evidence(("n",), link / "leaf") is None
    assert storage._path_evidence(("n",), tmp_path / "leaf") is not None


def test_an_in_process_admission_write_drops_reused_evidence(
    local_scope, reuse_switch, monkeypatch  # noqa: F811
):
    """In-process writers advance the admission epoch; stamps then need not fire."""
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        bootstrap.advance_admission_epoch()
        calls = _count_derivations(monkeypatch)
        assert _verdict(target)[0] == "allowed"
    finally:
        startup.close()

    assert calls["permission"] >= 1, "evidence survived an epoch advance"


def test_concurrent_derivations_keep_confirmed_evidence(
    local_scope, reuse_switch, monkeypatch  # noqa: F811
):
    """Parallel warm-ups on unrelated paths must converge, not churn the evidence.

    Each full derivation publishes fresh stamps. Before this was pinned, four
    threads kept replacing each other's confirmed selector evidence with
    unconfirmed copies, so no call ever reused it (234 vs 226 acquisitions/s).
    """
    from concurrent.futures import ThreadPoolExecutor

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    reuse_switch(True)
    paths = [data / f"db{i}.db" for i in range(4)]
    startup = storage.acquire_storage()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in range(4):
                assert all(v[0] == "allowed" for v in pool.map(_verdict, paths))
        calls = _count_derivations(monkeypatch)
        for path in paths:
            assert _verdict(path)[0] == "allowed"
    finally:
        startup.close()

    assert calls == {"permission": 0, "scope": 0}, calls


def test_warm_admission_reads_current_complete_control_bytes(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "bytes.db"
    reuse_switch(True)
    with storage.acquire_storage():
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        required = [
            config,
            root / "admission" / "registry.json",
            *root.glob("profile-*.json"),
        ]
        identities = {
            (p.stat().st_dev, p.stat().st_ino): p.read_bytes() for p in required
        }
        read = {key: bytearray() for key in identities}
        original = os.read

        def observed(fd, size):
            chunk = original(fd, size)
            info = os.fstat(fd)
            key = info.st_dev, info.st_ino
            if key in read:
                read[key].extend(chunk)
            return chunk

        monkeypatch.setattr(os, "read", observed)
        with storage.acquire_storage(target):
            target.write_text("checked")
        assert all(
            read[key] and bytes(read[key]) == raw * (len(read[key]) // len(raw))
            for key, raw in identities.items()
        )


@pytest.mark.parametrize("kind", ("gate", "lease", "registry"))
def test_warm_admission_refuses_replaced_native_lock_before_write(
    local_scope,  # noqa: F811
    reuse_switch,
    kind,
):
    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "foreign.db"
    target.write_bytes(b"foreign")
    reuse_switch(True)
    with storage.acquire_storage():
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        gate = (
            root
            / "admission"
            / (
                "registry.lock"
                if kind == "registry"
                else authority._key("profile", kind)
            )
        )
        gate.rename(gate.with_suffix(".retained"))
        gate.write_bytes(b"")
        gate.chmod(0o600)
        with pytest.raises(bootstrap.RecoveryRequired), storage.acquire_storage(target):
            target.write_bytes(b"unexpected")
        assert target.read_bytes() == b"foreign"


def test_counted_borrower_retains_predecessors_until_positive_close(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "retained.db"
    reuse_switch(True)
    startup = storage.acquire_storage()
    blocked, release = Event(), Event()
    try:
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        hold = storage._holds[startup._key]
        descriptors = [
            fd for chain in hold.predecessors.values() for fd in chain.descriptors
        ]
        assert descriptors
        original = storage._Evidence.observe

        def observe(evidence, *args, **kwargs):
            blocked.set()
            assert release.wait(5)
            return original(evidence, *args, **kwargs)

        monkeypatch.setattr(storage._Evidence, "observe", observe)

        def write():
            with storage.acquire_storage(target):
                target.write_text("accepted")

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(write)
            try:
                assert blocked.wait(5)
                startup.close()
                assert all(stat.S_ISDIR(os.fstat(fd).st_mode) for fd in descriptors)
            finally:
                release.set()
            pending.result(timeout=5)
        assert target.read_text() == "accepted"
        for fd in descriptors:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        release.set()
        startup.close()


def test_uncertain_predecessor_close_retains_native_exclusion(local_scope):  # noqa: F811
    root, _config, _data, _ = local_scope
    _in_subprocess(
        "from pathlib import Path\n"
        "import os, time\n"
        "from tldw_chatbook.Backup_Recovery import storage_admission as s, bootstrap\n"
        "from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionTimeout\n"
        "bootstrap.default_bootstrap_root = lambda: Path(values['root'])\n"
        "s._EVIDENCE_SETTLE_NS = 0\n"
        "lease = s.acquire_storage()\n"
        "hold = s._holds[lease._key]\n"
        "fd = next(iter(hold.predecessors.values())).descriptors[-1]\n"
        "close = os.close\n"
        "def uncertain(value):\n"
        "    if value == fd:\n"
        "        close(value)\n"
        "        raise OSError('injected_unknown_close')\n"
        "    close(value)\n"
        "os.close = uncertain\n"
        "lease.close()\n"
        "os.close = close\n"
        "assert hold in s._retiring_holds and hold.error is not None\n"
        "assert hold.native_context is not None\n"
        "pause = s._begin_local_pause()\n"
        "assert not pause.drain(time.monotonic())\n"
        "try:\n"
        "    with Admission.open_existing(Path(values['root']) / 'admission').maintenance(hold.names, .05):\n"
        "        raise AssertionError('released_native_exclusion')\n"
        "except AdmissionTimeout:\n"
        "    pass\n"
        "pause.resume()\n",
        root=str(root),
    )


def test_same_inode_bytes_cannot_hide_behind_equal_change_stamps(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    target = data / "unchanged.db"
    target.write_bytes(b"foreign")
    reuse_switch(True)
    with storage.acquire_storage():
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        record = next(root.glob("profile-*.json"))
        original = storage._content
        before = original(record)
        identity = before[:2]

        def equal_stamps(path, *args):
            current = original(path, *args)
            return (*before[:5], *current[5:]) if path == record else current

        monkeypatch.setattr(storage, "_content", equal_stamps)
        stamp = getattr(storage, "_content_stamp", None)
        if stamp is not None:

            def equal_stat(info):
                current = stamp(info)
                return (
                    (*before[:5], *current[5:]) if current[:2] == identity else current
                )

            monkeypatch.setattr(storage, "_content_stamp", equal_stat)
        raw = record.read_bytes()
        with record.open("r+b") as stream:
            stream.write(b"[" + raw[1:])
        with pytest.raises(bootstrap.RecoveryRequired), storage.acquire_storage(target):
            target.write_bytes(b"unexpected")
        assert target.read_bytes() == b"foreign"


def test_cold_scope_cannot_continue_a_retired_incumbent(
    local_scope, monkeypatch  # noqa: F811
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    startup = storage.acquire_storage()
    target = data / "not-created.db"
    config.write_text("changed selector in incumbent scope")
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    original = storage._scope

    def retired(*args, **kwargs):
        names = original(*args, **kwargs)
        monkeypatch.setattr(storage, "_scope", original)
        startup.close()
        return names

    monkeypatch.setattr(storage, "_scope", retired)
    try:
        with pytest.raises(bootstrap.RecoveryRequired), storage.acquire_storage(target):
            target.write_text("unexpected")
        assert not target.exists()
    finally:
        startup.close()


@pytest.mark.parametrize(
    "closed_first", [False, True], ids=["left-open", "closed-unknown"]
)
@pytest.mark.parametrize(
    "seam",
    [
        "registry.lock",
        "registry.json",
        "registry.pending.json",
        "gate",
        "lease",
        "content-file",
        "content-directory",
        "fallback-directory",
        "qualification",
    ],
)
def test_temporary_unknown_close_fences_actual_hold(local_scope, seam, closed_first):  # noqa: F811
    """Unknown temporary closes must not disappear when the borrower unwinds."""
    _temporary_close_case(local_scope, seam, closed_first)


def _temporary_close_case(local_scope, seam, closed_first, fault="normal"):  # noqa: F811
    import textwrap

    root, _config, _data, _ = local_scope
    _in_subprocess(
        textwrap.dedent("""
        from pathlib import Path
        import os, time
        from Tests import network_guard
        network_guard.install()
        import keyring
        from keyring.backends.null import Keyring
        keyring.set_keyring(Keyring())
        from tldw_chatbook.Backup_Recovery import storage_admission as s, bootstrap
        from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionTimeout
        bootstrap.default_bootstrap_root = lambda: Path(values['root'])
        s._EVIDENCE_SETTLE_NS = 0
        owner = s.acquire_storage()
        for _ in range(3):
            with s.acquire_storage():
                pass
        hold = s._holds[owner._key]
        seam = values['seam']
        dependent_path = Path(values['root']).parent / 'temporary-close-foreign.txt'
        dependent_path.write_bytes(b'foreign')
        original_open, original_close = os.open, os.close
        selected, calls, custody = [], [], []
        def opened(path, flags, *args, **kwargs):
            fd = original_open(path, flags, *args, **kwargs)
            name = Path(path).name
            match = name == seam
            if seam in ('gate', 'lease'):
                match = name.endswith('.' + seam)
            if seam == 'content-file':
                match = name == 'unbound-owner'
            if seam == 'content-directory':
                match = name == Path(values['root']).name
            if seam == 'fallback-directory':
                match = str(path) == Path(values['root']).anchor
            if seam == 'qualification':
                match = name == 'native_qualification.json'
            if match and not selected:
                selected.append(fd)
            return fd
        def closing(fd):
            if selected and fd == selected[0]:
                calls.append(fd)
                custody.append(any(fd in getattr(r, 'descriptors', ()) for r in hold.resources))
                if values['closed_first'] == 'yes':
                    original_close(fd)
                raise OSError('injected_temporary_close_unknown')
            original_close(fd)
        if seam == 'registry.pending.json':
            intent = Path(values['root']) / 'admission' / seam
            # Inject after candidate checking, at the real mandatory intent reader.
            original_read = Admission._read_intent
            def intent_read(self, parent, *args, **kwargs):
                intent.write_text('{}')
                intent.chmod(0o600)
                return original_read(self, parent, *args, **kwargs)
            Admission._read_intent = intent_read
        if seam == 'fallback-directory':
            hold.predecessor = lambda path: None
        original_read_bytes, original_stamp = os.read, s._content_stamp
        if values['fault'] == 'read':
            def read_bytes(fd, size):
                if selected and fd == selected[0]:
                    raise OSError('injected_content_read_failed')
                return original_read_bytes(fd, size)
            os.read = read_bytes
        if values['fault'] == 'validation':
            def stamp(info):
                if selected:
                    raise OSError('injected_content_validation_failed')
                return original_stamp(info)
            s._content_stamp = stamp
        os.open, os.close = opened, closing
        dependent = False
        try:
            if values['fault'] == 'optional':
                assert s._selector_evidence(Path(values['root']), bootstrap.effective_config_path(), hold.names, None, hold) is None
            with s.acquire_storage():
                dependent_path.write_bytes(b'forbidden')
                dependent = True
        except bootstrap.RecoveryRequired:
            pass
        finally:
            os.open = original_open
            os.read, s._content_stamp = original_read_bytes, original_stamp
        assert selected, 'fault seam not reached'
        assert not dependent, 'dependent I/O admitted after unknown close'
        assert dependent_path.read_bytes() == b'foreign'
        assert custody == [True], 'temporary descriptor had no actual Hold custody'
        assert hold.error is not None, 'unknown close was swallowed'
        recycled = original_open('/dev/null', os.O_RDONLY) if values['closed_first'] == 'yes' else None
        owner.close()
        if recycled is not None:
            os.fstat(recycled)
            original_close(recycled)
        assert hold in s._retiring_holds and hold.native_context is not None
        # A frame must never retry the ambiguous numeric fd at last-owner close.
        assert calls == selected
        os.close = original_close
        pause = s._begin_local_pause()
        assert not pause.drain(time.monotonic())
        pause.resume()
        # The retained original lease still excludes native maintenance.
        if seam == 'registry.pending.json':
            Admission._read_intent = original_read
            intent.unlink()
        try:
            with Admission.open_existing(Path(values['root']) / 'admission').maintenance(hold.names, .05):
                raise AssertionError('native exclusion released')
        except AdmissionTimeout:
            pass
    """),
        root=str(root),
        seam=seam,
        closed_first="yes" if closed_first else "no",
        fault=fault,
    )


def test_candidate_observation_reserves_last_owner_before_using_pins(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    """A cache miss must retain the old native Hold throughout its pin reads."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    root, config, data, _ = local_scope
    reuse_switch(True)
    owner = storage.acquire_storage()
    hold = storage._holds[owner._key]
    selected, release = Event(), Event()
    original = storage._Evidence.observe

    def observe(entry, *args, **kwargs):
        selected.set()
        assert release.wait(5)
        return original(entry, *args, **kwargs)

    monkeypatch.setattr(storage._Evidence, "observe", observe)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                storage._observe_candidates, root, config, data / "x.db", ()
            )
            try:
                assert selected.wait(5)
                owner.close()
                assert not hold.stop.is_set(), "unreserved observation lost its pins"
                assert hold.count == 1
                assert all(chain.descriptors for chain in hold.predecessors.values())
            finally:
                release.set()
            future.result(timeout=5)
        assert hold.stop.is_set() and hold.native_context is None
        assert hold not in storage._retiring_holds
    finally:
        release.set()
        owner.close()


def test_transaction_reuses_only_its_already_locked_file_descriptions(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    """One borrower need not reopen locks that its own native frame still holds."""
    from collections import Counter

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    reuse_switch(True)
    with storage.acquire_storage() as startup:
        db = CharactersRAGDB(data / "lending.db", "lending-test")
        try:
            for _ in range(3):
                with db.transaction() as cursor:
                    cursor.execute("SELECT 1").fetchone()
            hold = storage._holds[startup._key]
            group = hold.authority._observed_groups[hold.names]
            keys = {"registry.lock"} | {
                hold.authority._key(name, kind)
                for name in group
                for kind in ("gate", "lease")
            }
            opened = Counter()
            original = os.open

            def counted(path, *args, **kwargs):
                if Path(path).name in keys:
                    opened[Path(path).name] += 1
                return original(path, *args, **kwargs)

            monkeypatch.setattr(os, "open", counted)
            with db.transaction() as cursor:
                assert cursor.execute("SELECT 1").fetchone()[0] == 1
            assert opened == Counter({key: 1 for key in keys})
        finally:
            db.close_connection()


@pytest.mark.parametrize("closed_first", [False, True])
@pytest.mark.parametrize("fault", ["read", "validation", "optional"])
def test_content_error_unknown_close_is_not_optional_ineligibility(
    local_scope, fault, closed_first  # noqa: F811
):
    """Failed content validation cannot hide uncertain retirement as a miss."""
    _temporary_close_case(local_scope, "content-file", closed_first, fault)


def test_borrowed_nonempty_lock_bytes_still_require_current_full_read(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    lock = root / "admission" / "registry.lock"
    lock.write_bytes(b"original")
    reuse_switch(True)
    target = data / "foreign.db"
    target.write_bytes(b"foreign")
    with storage.acquire_storage():
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        stamp = storage._content_stamp
        original = storage._content(lock)
        identity = original[:2]
        before = storage._content
        mutated = []

        def content(path, parent=None, custody=None):
            if path == lock and not mutated:
                assert custody is not None and path in custody.borrowed
                with lock.open("r+b") as stream:
                    stream.write(b"modified")
                mutated.append(True)
            return before(path, parent, custody)

        def equal_stamp(info):
            current = stamp(info)
            return (*original[:5], *current[5:]) if current[:2] == identity else current

        monkeypatch.setattr(storage, "_content", content)
        monkeypatch.setattr(storage, "_content_stamp", equal_stamp)
        # A mismatch may fully rederive; it must not be a warm cache success.
        calls = _count_derivations(monkeypatch)
        with storage.acquire_storage(target):
            assert target.read_bytes() == b"foreign"
        assert mutated and calls["permission"] > 0


def test_independent_borrow_frames_survive_another_borrowers_read_exception(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    reuse_switch(True)
    target = data / "independent.db"
    lock = root / "admission" / "registry.lock"
    first_ready, release = Event(), Event()
    observed = {}
    with storage.acquire_storage() as owner:
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        hold = storage._holds[owner._key]
        resources = set(hold.resources)
        original = storage._content

        def content(path, parent=None, custody=None):
            if path == lock:
                fd = custody.borrowed[path]
                if not first_ready.is_set():
                    observed["first"] = fd
                    first_ready.set()
                    assert release.wait(5)
                    os.fstat(fd)
                else:
                    observed["second"] = fd
                    raise OSError("second borrower read failed")
            return original(path, parent, custody)

        monkeypatch.setattr(storage, "_content", content)

        def accepted():
            with storage.acquire_storage(target):
                target.write_bytes(b"accepted")

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(accepted)
            try:
                assert first_ready.wait(5)
                with (
                    pytest.raises(bootstrap.RecoveryRequired),
                    storage.acquire_storage(target),
                ):
                    target.write_bytes(b"forbidden")
                assert observed["first"] != observed["second"]
                os.fstat(observed["first"])
                with pytest.raises(OSError):
                    os.fstat(observed["second"])
            finally:
                release.set()
            future.result(timeout=5)
        assert target.read_bytes() == b"accepted"
        assert hold.error is None and hold.resources == resources


def test_prederivation_observation_cannot_confirm_a_replacement_hold(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    _root, config, data, _ = local_scope
    reuse_switch(True)
    owner = storage.acquire_storage()
    old = storage._holds[owner._key]
    original = storage._observe_candidates

    def after_last_owner(*args):
        observed = original(*args)
        owner.close()
        return observed

    monkeypatch.setattr(storage, "_observe_candidates", after_last_owner)
    try:
        with storage.acquire_storage(data / "new-generation.db") as borrower:
            current = storage._holds[borrower._key]
            assert current is not old
            assert not current.evidence[str(config)].confirmed
        assert old.native_context is None
    finally:
        owner.close()


def test_lent_lock_replacement_never_authorizes_detached_description(
    local_scope,  # noqa: F811
    reuse_switch,
    monkeypatch,
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    reuse_switch(True)
    target = data / "untouched.db"
    target.write_bytes(b"foreign")
    lock = root / "admission" / "registry.lock"
    with storage.acquire_storage():
        for _ in range(3):
            with storage.acquire_storage(target):
                pass
        original = storage._content
        replaced = []

        def content(path, parent=None, custody=None):
            if path == lock and not replaced:
                assert custody is not None and path in custody.borrowed
                lock.rename(lock.with_suffix(".detached"))
                lock.write_bytes(b"")
                lock.chmod(0o600)
                replaced.append(True)
            return original(path, parent, custody)

        monkeypatch.setattr(storage, "_content", content)
        with pytest.raises(bootstrap.RecoveryRequired), storage.acquire_storage(target):
            target.write_bytes(b"forbidden")
        assert replaced and target.read_bytes() == b"foreign"
