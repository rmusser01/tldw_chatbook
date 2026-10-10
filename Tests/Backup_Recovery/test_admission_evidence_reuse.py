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
        self.guard_paths: set[str] = set()
        self.relative_callers: dict[str, list[str]] = {}
        self._fcntl = fcntl
        from Tests import real_profile_guard

        guard_protected = real_profile_guard._protected.__code__
        guard_hook = real_profile_guard._hook.__code__
        realpath_code = os.path.realpath.__code__

        def guard_probe(name: str, frame) -> bool:
            """Recognize only a realpath probe of the guard's write-open input."""
            saw_realpath = False
            for _ in range(8):
                if frame is None:
                    return False
                if frame.f_code is realpath_code:
                    saw_realpath = True
                if frame.f_code is guard_protected:
                    caller = frame.f_back
                    return bool(
                        saw_realpath
                        and frame.f_locals.get("raw") == name
                        and frame.f_locals.get("dir_fd") is None
                        and caller is not None
                        and caller.f_code is guard_hook
                        and caller.f_locals.get("event") == "open"
                    )
                frame = frame.f_back
            return False

        def absolute(target, dir_fd=None) -> None:
            if not self.active:
                return
            if isinstance(target, int):
                self.paths.add(self._fd_path(target))
            elif isinstance(target, (str, bytes, os.PathLike)):
                name = os.fsdecode(target)
                if dir_fd is not None and not os.path.isabs(name):
                    name = os.path.join(self._fd_path(dir_fd), name)
                if not os.path.isabs(name) and guard_probe(name, sys._getframe(1)):
                    # The open audit event omits dir_fd. The unchanged profile
                    # guard therefore probes this leaf against cwd in realpath;
                    # it is observer IO, not an admission dependency. Exact code
                    # identity and input are required; leaf names grant nothing.
                    self.guard_paths.add(name)
                    return
                self.paths.add(os.path.normpath(name))
                if not os.path.isabs(name):
                    # Source-line formatting would itself open Python files
                    # and contaminate this dependency observer.
                    frames = []
                    frame = sys._getframe(1)
                    for _ in range(8):
                        if frame is None:
                            break
                        code = frame.f_code
                        frames.append(f"{code.co_filename}:{frame.f_lineno}:{code.co_name}")
                        frame = frame.f_back
                    self.relative_callers.setdefault(name, frames)

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
    assert not unstamped, "\n".join(unstamped) + "\n" + repr(tracer.relative_callers)



@pytest.mark.skipif(
    sys.platform != "darwin", reason="actual macOS guard realpath probes"
)
def test_guard_probes_are_separate_from_descriptor_and_unknown_relative_reads(
    tmp_path, monkeypatch
):
    """Attribution must not hide a real relative read with the same leaf name."""
    monkeypatch.chdir(tmp_path)
    tracer = _Tracer(monkeypatch)
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    tracer.active = True
    try:
        child = os.open(
            "observer-child", os.O_CREAT | os.O_WRONLY, 0o600, dir_fd=descriptor
        )
        os.close(child)
        assert str(tmp_path / "observer-child") in tracer.paths
        assert "observer-child" in tracer.guard_paths
        assert "observer-child" not in tracer.paths
        os.lstat("observer-child")
        assert "observer-child" in tracer.paths
        assert tracer.relative_callers["observer-child"]
    finally:
        tracer.active = False
        os.close(descriptor)


@pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS F_GETPATH observer")
def test_unknown_relative_read_is_never_classified_by_its_leaf_name(
    tmp_path, monkeypatch
):
    """Names resembling control records do not establish observer provenance."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "registry.lock").write_bytes(b"test")
    tracer = _Tracer(monkeypatch)
    tracer.active = True
    try:
        os.lstat("registry.lock")
    finally:
        tracer.active = False
    assert "registry.lock" in tracer.paths
    assert "registry.lock" not in tracer.guard_paths

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
    """An optional observation failure closes the counted lease before fallback,
    so the unchanged full derivation can succeed and the hold can still drain
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

        derivations = []
        original_scope = storage._scope

        def full_derivation(*args, **kwargs):
            derivations.append(True)
            return original_scope(*args, **kwargs)

        def unreadable(*paths):
            raise PermissionError("optional admission evidence became unreadable")

        monkeypatch.setattr(storage, "_scope", full_derivation)
        monkeypatch.setattr(storage, "_observe_stamps", unreadable)
        # An unavailable optional observation falls back to full authority;
        # it cannot fabricate a refusal while the real filesystem is unchanged.
        assert _verdict(target)[0] == "allowed"
        assert derivations
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
