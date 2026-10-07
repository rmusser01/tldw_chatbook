"""Persistent hook consent, invalidation, and real process launch boundaries."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import toml

from tldw_chatbook import config
from tldw_chatbook.Utils.toml_serialization import dumps_cli_config

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def hook_file(monkeypatch):
    # ADR-126 binds the source for this interpreter; mutate only its already
    # selected private test profile rather than switching a live source.
    root = Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).resolve()
    path = Path(os.environ["TLDW_CONFIG_PATH"])
    assert path.resolve().is_relative_to(root)
    data = Path(config.get_user_data_dir())
    assert data.resolve().is_relative_to(root)
    permissions = data / "hook_permissions.json"
    original = path.read_bytes()
    previous = permissions.read_bytes() if permissions.exists() else None
    permissions.unlink(missing_ok=True)
    raw = toml.loads(original.decode())
    raw["hooks"] = {
        "hook": [
            {
                "id": "one",
                "event": "PostToolUse",
                "command": [sys.executable, "-c", "pass"],
                "timeout_s": 5,
            }
        ]
    }
    path.write_text(dumps_cli_config(raw))
    try:
        yield path
    finally:
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
        path.write_bytes(original)
        if previous is None:
            permissions.unlink(missing_ok=True)
        else:
            permissions.write_bytes(previous)
        config.refresh_runtime_config_from_cli_config()


def _owner():
    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    return HookPermissions()


def _approve(owner):
    pending = owner.snapshot()
    return owner.approve(
        pending,
        [row.entry.key for row in pending.rows if row.entry and row.state == "pending"],
    )


def _edit(path, mutate):
    raw = toml.loads(path.read_text())
    mutate(raw["hooks"])
    path.write_text(dumps_cli_config(raw))


def _count_reads(owner, monkeypatch, *, settle_ns: int = 0) -> list[int]:
    """Count the owner's full reads (config section plus permission store).

    Args:
        owner: The hook permission owner under test.
        monkeypatch: Wraps the owner's full read.
        settle_ns: The visit settle window; ``0`` lets a visit keep a
            snapshot of files written moments ago, as these tests do.

    Returns:
        One entry per full read.
    """
    from tldw_chatbook.Agents import hook_permissions

    monkeypatch.setattr(hook_permissions, "_VISIT_SETTLE_NS", settle_ns)
    reads: list[int] = []
    real = owner._current

    def counted(*args, **kwargs):
        reads.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, "_current", counted)
    return reads


def _settle(owner, reads: list[int]) -> None:
    """Visit until a visit reads nothing (a reconcile write costs one more)."""
    for _ in range(4):
        before = len(reads)
        owner.visit_snapshot()
        if len(reads) == before:
            return
    raise AssertionError("visit snapshots never stopped reading")


def test_a_warm_visit_reads_nothing_until_a_file_changes(hook_file, monkeypatch):
    """TASK-33642: an unchanged Console visit reuses the last hook snapshot.

    An approval (a store write) or a hook edit (a config write) is read on
    the next visit and shown.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads.
    """
    owner = _owner()
    reads = _count_reads(owner, monkeypatch)
    first = owner.visit_snapshot()
    _settle(owner, reads)
    settled = len(reads)
    warm = owner.visit_snapshot()
    assert len(reads) == settled, "a warm visit re-read the hook files"
    assert warm.rows == first.rows and not warm.ready

    assert _approve(owner).ready
    count = len(reads)
    approved = owner.visit_snapshot()
    assert len(reads) > count and approved.ready, "an approval was not picked up"

    _settle(owner, reads)
    _edit(hook_file, lambda hooks: hooks["hook"][0].update(timeout_s=7))
    count = len(reads)
    edited = owner.visit_snapshot()
    assert len(reads) > count and not edited.ready, "a hook edit was not picked up"


def test_a_sealed_hook_is_not_served_from_a_warm_visit(hook_file, monkeypatch):
    """TASK-33642: in-memory sealing invalidates the reused visit snapshot too.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    assert approved.ready

    owner._seal(approved, [row.entry.key for row in approved.rows if row.entry])
    count = len(reads)
    sealed = owner.visit_snapshot()
    assert len(reads) > count and not sealed.ready, "a sealed hook was still approved"


def test_a_hook_sealed_while_the_stamp_is_probed_is_not_served(hook_file, monkeypatch):
    """TASK-33642: sealing is re-read under its lock before a warm visit returns.

    The seal lands after the reuse stamp copied the in-memory state but before
    the visit returns, so the stamp still matches the cached one.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads and seals mid-probe.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    real_stamp = owner._visit_stamp
    sealed_once: list[int] = []

    def stamp_then_seal(*args):
        stamp = real_stamp(*args)
        if not sealed_once:
            sealed_once.append(1)
            owner._seal(approved, [row.entry.key for row in approved.rows if row.entry])
        return stamp

    monkeypatch.setattr(owner, "_visit_stamp", stamp_then_seal)
    count = len(reads)
    sealed = owner.visit_snapshot()
    assert len(reads) > count and not sealed.ready, "a sealed hook was still approved"


def test_a_just_written_hook_file_is_not_kept_for_a_warm_visit(hook_file, monkeypatch):
    """TASK-33642: a snapshot is kept only once both files' last write settled.

    A same-size edit inside one coarse timestamp tick leaves an equal stamp,
    so a snapshot of a file changed within the settle window is re-read.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads; a window longer than the
            test (rather than the real second) keeps it independent of load.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch, settle_ns=10**15)
    for _ in range(3):
        count = len(reads)
        assert owner.visit_snapshot().ready
        assert len(reads) > count, "a visit kept a snapshot of a just-written file"


def test_a_retargeted_config_path_is_not_served_from_a_warm_visit(
    hook_file, monkeypatch
):
    """TASK-33642: a config selection moved in-process forces a full read.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads and retargets the config.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    other = hook_file.parent / "other-profile.toml"
    other.write_text("")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
    count = len(reads)
    moved = owner.visit_snapshot()
    assert len(reads) > count, "a retargeted config was served the old profile's hooks"
    assert moved is not approved


def test_an_unsafe_store_directory_is_not_served_from_a_warm_visit(
    hook_file, monkeypatch
):
    """TASK-33642: the store directory's posture is part of the reuse stamp.

    A full read refuses a group-writable profile directory; a warm visit must
    not keep showing the approved snapshot read before it changed.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    directory = approved.store_path.parent
    mode = directory.stat().st_mode & 0o777
    directory.chmod(mode | 0o020)
    try:
        count = len(reads)
        unsafe = owner.visit_snapshot()
    finally:
        directory.chmod(mode)
    assert len(reads) > count, "an unsafe store directory was served from the reuse"
    assert unsafe is not approved


def test_a_tampered_store_lock_is_not_served_from_a_warm_visit(
    hook_file, monkeypatch, tmp_path
):
    """TASK-33642: the store's lock file is part of the reuse stamp.

    A full read refuses a lock file swapped for a symlink; a warm visit must
    not keep showing the approval read before the swap.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads.
        tmp_path: Holds the symlink's target.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    lock = approved.store_path.with_name(approved.store_path.name + ".lock")
    assert lock.exists()
    target = tmp_path / "elsewhere.lock"
    target.write_text("")
    lock.unlink()
    lock.symlink_to(target)
    try:
        count = len(reads)
        tampered = owner.visit_snapshot()
    finally:
        lock.unlink()
    assert len(reads) > count, "a swapped lock file was served from the reuse"
    assert not tampered.ready


def test_a_maintenance_pause_is_not_served_from_a_warm_visit(hook_file, monkeypatch):
    """TASK-33642: while storage admission is paused, a visit reads in full.

    The full read is refused at the admission boundary and reports recovery;
    a warm visit must not keep showing the approval from before the pause.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads and opens a pause.
    """
    from tldw_chatbook.Backup_Recovery import storage_admission

    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    owner.visit_snapshot()
    monkeypatch.setattr(storage_admission, "_pause", object())
    count = len(reads)
    paused = owner.visit_snapshot()
    assert len(reads) > count, "a paused admission was served from the reuse"
    assert not paused.ready


def test_a_store_swapped_for_a_symlink_is_not_served_from_a_warm_visit(
    hook_file, monkeypatch
):
    """TASK-33642: the store is stamped without following links.

    The same file reached through a symlink keeps its identity and times
    under ``stat``; a full read refuses it as non-regular.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    store = approved.store_path
    moved = store.with_name("moved-hook-permissions.json")
    store.rename(moved)
    store.symlink_to(moved)
    try:
        count = len(reads)
        linked = owner.visit_snapshot()
    finally:
        store.unlink()
        moved.rename(store)
    assert len(reads) > count, (
        "a store reached through a symlink was served from the reuse"
    )
    assert not linked.ready


def test_a_retargeted_data_root_is_not_served_from_a_warm_visit(
    hook_file, monkeypatch, tmp_path
):
    """TASK-33642: the environment that selects the data root is stamped.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads and retargets XDG_DATA_HOME.
        tmp_path: The other data root.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    approved = owner.visit_snapshot()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "other-data"))
    count = len(reads)
    moved = owner.visit_snapshot()
    assert len(reads) > count, (
        "a retargeted data root was served the old store's grants"
    )
    assert moved is not approved


def test_a_tampered_config_lock_is_not_served_from_a_warm_visit(
    hook_file, monkeypatch, tmp_path
):
    """TASK-33642: the config writer's lock file is stamped too.

    Args:
        hook_file: The private profile's config with one hook defined.
        monkeypatch: Counts the owner's full reads.
        tmp_path: Holds the symlink's target.
    """
    owner = _owner()
    assert _approve(owner).ready
    reads = _count_reads(owner, monkeypatch)
    _settle(owner, reads)
    owner.visit_snapshot()
    lock = hook_file.with_name(hook_file.name + ".lock")
    assert lock.exists()
    target = tmp_path / "elsewhere.lock"
    target.write_text("")
    original = lock.read_bytes()
    lock.unlink()
    lock.symlink_to(target)
    try:
        count = len(reads)
        tampered = owner.visit_snapshot()
    finally:
        lock.unlink()
        lock.write_bytes(original)
        lock.chmod(0o600)
    assert len(reads) > count, "a tampered config lock was served from the reuse"
    assert not tampered.ready


def test_consent_survives_a_new_owner_and_state_contains_no_commands(hook_file):
    owner = _owner()
    assert not owner.snapshot().ready
    assert _approve(owner).ready
    current = _owner().snapshot()
    assert current.ready
    state = json.loads(current.store_path.read_text())
    assert state["schema_version"] == 1
    assert "pass" not in current.store_path.read_text()
    assert "command" not in current.store_path.read_text()
    # The platform adapter derives these bits from the actual Windows DACL;
    # stdlib Path.stat reports synthetic writable bits on Windows.
    from tldw_chatbook.Utils.platform_files import os as platform_os

    assert platform_os.stat(current.store_path, follow_symlinks=False).st_mode & 0o777 == 0o600


def test_deleting_an_approved_legacy_duplicate_cannot_transfer_grant(hook_file):
    def duplicate(section):
        row = section["hook"][0]
        row.pop("id")
        section["hook"] = [row.copy(), row.copy()]

    _edit(hook_file, duplicate)
    owner = _owner()
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    _edit(hook_file, lambda section: section["hook"].pop(0))
    assert not owner.snapshot().ready


@pytest.mark.parametrize(
    "change",
    [
        {"event": "Stop"},
        {"command": [sys.executable, "-c", "pass", "changed argument"]},
        {"matcher": "fs_read"},
        {"timeout_s": 6},
    ],
)
def test_changed_then_restored_definition_requires_new_review(hook_file, change):
    original = toml.loads(hook_file.read_text())["hooks"]["hook"][0]
    owner = _owner()
    assert _approve(owner).ready
    _edit(hook_file, lambda section: section["hook"][0].update(change))
    changed = owner.snapshot()
    assert not changed.ready and changed.rows[0].state == "pending"
    _edit(hook_file, lambda section: section["hook"].__setitem__(0, original))
    assert not owner.snapshot().ready


def test_disable_reenable_retains_consent_but_retires_queued_epoch(hook_file):
    from tldw_chatbook.Agents.run_hooks import HookLaunchRefused

    owner = _owner()
    _approve(owner)
    old = owner.targets("PostToolUse", None)[0]
    _edit(hook_file, lambda section: section.update(enabled=False))
    assert owner.snapshot().ready
    _edit(hook_file, lambda section: section.update(enabled=True))
    assert owner.snapshot().ready
    with pytest.raises(HookLaunchRefused), owner.launch_guard(old, tool_name=None):
        pytest.fail("stale epoch authorized")


def test_stale_approval_cannot_undo_another_owners_revocation(hook_file):
    from tldw_chatbook.Agents.hook_permissions import HookReviewConflict

    first = _owner()
    current = _approve(first)
    other = _owner()
    other.revoke(other.snapshot(), current.rows[0].entry.key)
    with pytest.raises(HookReviewConflict):
        first.approve(current, [current.rows[0].entry.key])
    assert not first.snapshot().ready


def test_config_change_rejects_old_review(hook_file):
    from tldw_chatbook.Agents.hook_permissions import HookReviewConflict

    owner = _owner()
    pending = owner.snapshot()
    _edit(
        hook_file,
        lambda section: section["hook"][0].update(
            command=[sys.executable, "-c", "print(1)"]
        ),
    )
    with pytest.raises(HookReviewConflict):
        owner.approve(pending, [pending.rows[0].entry.key])
    assert not owner.snapshot().ready


@pytest.mark.parametrize("contents", ['{"schema_version":99}', "broken json"])
def test_corrupt_or_unsupported_state_blocks_until_explicit_reset(hook_file, contents):
    owner = _owner()
    initial = owner.snapshot()
    initial.store_path.write_text(contents)
    broken = owner.snapshot()
    assert not broken.ready
    reset = owner.reset_invalid_state(broken)
    assert not reset.ready
    assert _approve(owner).ready


def test_master_disabled_allows_send_with_corrupt_store(hook_file):
    owner = _owner()
    owner.snapshot().store_path.write_text("broken json")
    _edit(hook_file, lambda section: section.update(enabled=False))
    assert owner.snapshot().ready
    assert owner.targets("PreToolUse", "fs_read") == ()


def test_failed_revoke_is_locally_sealed_and_reports_failure(hook_file, monkeypatch):
    from tldw_chatbook.Agents import hook_permissions

    owner = _owner()
    current = _approve(owner)

    def fail(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(hook_permissions, "atomic_private_write_text", fail)
    refused = owner.revoke(current, current.rows[0].entry.key)
    assert not refused.ready
    assert refused.notice
    assert not owner.notification_targets("PostToolUse", None)


def test_invalid_enabled_guard_refuses_even_without_a_valid_engine_target(hook_file):
    from tldw_chatbook.Agents.run_hooks import HookLaunchRefused

    _edit(hook_file, lambda section: section["hook"][0].update(event="unknown"))
    owner = _owner()
    with pytest.raises(HookLaunchRefused):
        owner.targets("PreToolUse", "fs_read")
    with pytest.raises(HookLaunchRefused):
        owner.targets("UserPromptSubmit", None)


def test_permission_file_and_companions_are_sensitive(hook_file):
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    owner = _owner()
    state = owner.snapshot().store_path
    assert is_sensitive_path(state)
    assert is_sensitive_path(state.with_name(state.name + ".lock"))
    assert is_sensitive_path(state.with_name(".hook_permissions.json.deadbeef.tmp"))


@pytest.mark.asyncio
async def test_real_command_starts_when_approved_and_cannot_start_after_revoke(
    hook_file, tmp_path
):
    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    marker = tmp_path / "executed"
    _edit(
        hook_file,
        lambda section: section["hook"][0].update(
            command=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path("
                + repr(str(marker))
                + ").write_text('ran')",
            ]
        ),
    )
    owner = _owner()
    _approve(owner)
    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=owner.launch_guard,
    )
    try:
        await engine.fire_async("PostToolUse", session_id="one")
        assert marker.read_text() == "ran"
        marker.unlink()
        current = owner.snapshot()
        owner.revoke(current, current.rows[0].entry.key)
        await engine.fire_async("PostToolUse", session_id="one")
        assert not marker.exists()
    finally:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)


def test_unreadable_config_cannot_reuse_a_cached_master_disable(hook_file, monkeypatch):
    from contextlib import contextmanager

    _edit(hook_file, lambda section: section.update(enabled=False))
    owner = _owner()
    assert owner.snapshot().ready

    @contextmanager
    def unavailable():
        raise OSError("test config unavailable")
        yield

    monkeypatch.setattr(config, "locked_hooks_config_snapshot", unavailable)
    assert not owner.snapshot().ready


@pytest.mark.asyncio
async def test_authority_error_refuses_prompt_but_process_error_keeps_fail_open(
    hook_file, tmp_path
):
    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    _edit(
        hook_file,
        lambda section: section["hook"][0].update(
            event="UserPromptSubmit",
            command=[str(tmp_path / "missing-executable")],
        ),
    )
    owner = _owner()
    _approve(owner)
    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=owner.launch_guard,
    )
    try:
        assert not (
            await engine.fire_async("UserPromptSubmit", session_id="one")
        ).blocked
    finally:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)

    def unavailable(*args, **kwargs):
        raise RuntimeError("test authority unavailable")

    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=unavailable,
    )
    try:
        assert (await engine.fire_async("UserPromptSubmit", session_id="one")).blocked
    finally:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)


@pytest.mark.parametrize("change", ["remove", "config_path", "data_root"])
def test_observed_removal_or_retarget_cannot_reuse_consent(
    hook_file, tmp_path, monkeypatch, change
):
    owner = _owner()
    _approve(owner)
    original = hook_file.read_text()
    if change == "remove":
        _edit(hook_file, lambda section: section.update(hook=[]))
        assert owner.snapshot().ready
        hook_file.write_text(original)
    elif change == "config_path":
        other = tmp_path / "other.toml"
        other.write_text(original)
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
    else:
        data = tmp_path / "other-data"
        data.mkdir(mode=0o700)
        monkeypatch.setattr(config, "get_user_data_dir", lambda: data)
    assert not owner.snapshot().ready
    assert not owner.notification_targets("PostToolUse", None)


def test_failure_after_visible_permission_replace_stays_locally_blocked(
    hook_file, monkeypatch
):
    from tldw_chatbook.Agents import hook_permissions

    owner = _owner()
    pending = owner.snapshot()
    real_write = hook_permissions.atomic_private_write_text

    def fail_after_replace(*args, **kwargs):
        real_write(*args, **kwargs)
        raise OSError("test publication failure")

    monkeypatch.setattr(
        hook_permissions, "atomic_private_write_text", fail_after_replace
    )
    result = owner.approve(pending, [pending.rows[0].entry.key])
    assert not result.ready
    assert "failed" in result.notice.lower()
    assert not owner.snapshot().ready
    assert _owner().snapshot().ready  # replacement really happened


def test_independent_python_process_reads_persisted_consent(hook_file):
    import os
    import subprocess

    owner = _owner()
    _approve(owner)
    code = (
        "from pathlib import Path; from tldw_chatbook import config; "
        "from tldw_chatbook.Agents.hook_permissions import HookPermissions; "
        "config.get_user_data_dir=lambda:Path("
        + repr(str(owner.snapshot().store_path.parent))
        + "); "
        "assert HookPermissions().snapshot().ready"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["PostToolUse", "PreToolUse", "UserPromptSubmit"])
@pytest.mark.parametrize("transition", ["reapprove", "reenable"])
async def test_stale_target_cannot_launch_or_bypass_a_blocking_hook(
    hook_file, tmp_path, event, transition
):
    import threading
    from contextlib import contextmanager

    from tldw_chatbook.Agents.run_hooks import BLOCKING_EVENTS, RunHooksEngine

    marker = tmp_path / "started"
    _edit(
        hook_file,
        lambda section: section["hook"][0].update(
            event=event,
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ],
        ),
    )
    owner = _owner()
    _approve(owner)
    entered, release = threading.Event(), threading.Event()

    @contextmanager
    def waiting_guard(target, *, tool_name):
        entered.set()
        assert release.wait(5)
        with owner.launch_guard(target, tool_name=tool_name):
            yield

    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=waiting_guard,
    )
    task = asyncio.create_task(engine.fire_async(event, session_id="s"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        current = owner.snapshot()
        if transition == "reapprove":
            owner.revoke(current, current.rows[0].entry.key)
            _approve(owner)
        else:
            _edit(hook_file, lambda section: section["hook"][0].update(enabled=False))
            assert owner.snapshot().ready
            _edit(hook_file, lambda section: section["hook"][0].update(enabled=True))
            assert owner.snapshot().ready
        release.set()
        result = await asyncio.wait_for(task, 5)
        assert not marker.exists()
        assert result.blocked is (event in BLOCKING_EVENTS)
    finally:
        release.set()
        engine.close()
        await task
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_revoke_does_not_wait_for_or_kill_an_already_started_hook(
    hook_file, tmp_path
):
    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    started, release, finished = (
        tmp_path / name for name in ("started", "release", "finished")
    )
    code = (
        f"from pathlib import Path; import time; Path({str(started)!r}).write_text('started'); "
        f"\nwhile not Path({str(release)!r}).exists(): time.sleep(.01)"
        f"\nPath({str(finished)!r}).write_text('finished')"
    )
    _edit(
        hook_file,
        lambda section: section["hook"][0].update(command=[sys.executable, "-c", code]),
    )
    owner = _owner()
    _approve(owner)
    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=owner.launch_guard,
    )
    task = asyncio.create_task(engine.fire_async("PostToolUse", session_id="s"))
    try:
        async with asyncio.timeout(3):
            while not started.exists():
                await asyncio.sleep(0.01)
        current = owner.snapshot()
        await asyncio.wait_for(
            asyncio.to_thread(owner.revoke, current, current.rows[0].entry.key), 1
        )
        assert not task.done()
        release.write_text("go")
        await asyncio.wait_for(task, 3)
        assert finished.read_text() == "finished"
    finally:
        release.write_text("go")
        engine.close()
        await task
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)


def test_queued_notification_never_selects_a_new_definition(hook_file, tmp_path):
    import threading

    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    marker = tmp_path / "executed"
    _edit(
        hook_file,
        lambda section: section["hook"][0].update(
            event="Stop",
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('old')",
            ],
        ),
    )
    owner = _owner()
    _approve(owner)
    engine = RunHooksEngine(
        owner.targets,
        lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=owner.launch_guard,
    )
    release, entered = threading.Event(), threading.Event()

    def hold_coordinator():
        entered.set()
        release.wait(5)

    engine._notify_worker.submit(hold_coordinator)
    assert entered.wait(2)
    try:
        engine.notify("Stop", session_id="s")
        _edit(hook_file, lambda section: section["hook"][0].update(timeout_s=6))
        _approve(owner)
        release.set()
        engine._notify_worker.shutdown(wait=True)
        assert not marker.exists()
    finally:
        release.set()
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            pool.shutdown(wait=True, cancel_futures=True)


def test_verified_master_disable_bypasses_unavailable_consent_lock(
    hook_file, monkeypatch
):
    from contextlib import contextmanager

    _edit(hook_file, lambda section: section.update(enabled=False))
    owner = _owner()

    @contextmanager
    def unavailable(path):
        raise OSError("test consent lock unavailable")
        yield

    monkeypatch.setattr(owner, "_store_lock", unavailable)
    assert owner.snapshot().ready
    assert owner.targets("PreToolUse", "fs_read") == ()


def test_guided_id_assignment_preserves_only_unchanged_owned_legacy_consent(hook_file):
    import copy

    _edit(hook_file, lambda section: section["hook"][0].pop("id"))
    owner = _owner()
    current = _approve(owner)
    key = current.rows[0].entry.key
    replacement = copy.deepcopy(current.config.section)
    replacement["hook"][0]["id"] = "assigned"
    saved, reviewed = owner.save_configuration(
        current, replacement, legacy_ids={key: "assigned"}
    )
    assert saved.file_replaced and reviewed.ready
    assert owner.targets("PostToolUse", None)[0].key == "id:assigned"


def test_guided_id_assignment_does_not_transfer_an_incomplete_duplicate_group(
    hook_file,
):
    import copy

    def duplicate(section):
        row = section["hook"][0]
        row.pop("id")
        section["hook"] = [row.copy(), row.copy()]

    _edit(hook_file, duplicate)
    owner = _owner()
    current = _approve(owner)
    replacement = copy.deepcopy(current.config.section)
    replacement["hook"].pop(1)
    replacement["hook"][0]["id"] = "assigned"
    saved, reviewed = owner.save_configuration(
        current, replacement, legacy_ids={current.rows[0].entry.key: "assigned"}
    )
    assert saved.file_replaced and not reviewed.ready


def test_settings_save_refresh_failure_blocks_launch_until_explicit_recovery(
    hook_file, monkeypatch
):
    import copy

    owner = _owner()
    current = _approve(owner)
    replacement = copy.deepcopy(current.config.section)
    replacement["hook"][0]["name"] = "friendly"
    writer = config.replace_hooks_config_snapshot

    def failed_refresh(expected, section):
        result = writer(expected, section)
        return config.LiteralConfigMutationResult(
            result.file_replaced, False, None, "cache_reload"
        )

    monkeypatch.setattr(config, "replace_hooks_config_snapshot", failed_refresh)
    result, pending = owner.save_configuration(current, replacement)
    assert result.file_replaced and not result.caches_reloaded
    assert not pending.ready and "refresh" in pending.blocked_reason
    from tldw_chatbook.Agents.run_hooks import HookLaunchRefused

    with pytest.raises(HookLaunchRefused):
        owner.targets("UserPromptSubmit", None)
    assert owner.recover().ready


def test_settings_save_cannot_resurrect_another_owners_revocation(
    hook_file, monkeypatch
):
    import copy

    _edit(hook_file, lambda section: section["hook"][0].pop("id"))
    owner = _owner()
    current = _approve(owner)
    key = current.rows[0].entry.key
    replacement = copy.deepcopy(current.config.section)
    replacement["hook"][0]["id"] = "new"
    writer = config.replace_hooks_config_snapshot

    def concurrent_revoke(expected, section):
        other = _owner()
        other.revoke(other.snapshot(), key)
        return writer(expected, section)

    monkeypatch.setattr(config, "replace_hooks_config_snapshot", concurrent_revoke)
    result, pending = owner.save_configuration(
        current, replacement, legacy_ids={key: "new"}
    )
    assert result.file_replaced and not pending.ready
    assert pending.rows[0].state == "pending"


def test_verified_master_off_does_not_require_a_permission_directory(
    hook_file, monkeypatch
):
    _edit(hook_file, lambda section: section.update(enabled=False))

    def unavailable():
        raise OSError("permission directory unavailable")

    monkeypatch.setattr(config, "get_user_data_dir", unavailable)
    assert _owner().snapshot().ready


def test_permission_review_does_not_discard_a_staged_configuration_edit(hook_file):
    import copy

    owner = _owner()
    original = owner.snapshot()
    _approve(owner)
    replacement = copy.deepcopy(original.config.section)
    replacement["hook"][0]["timeout_s"] = 7
    result, reviewed = owner.save_configuration(original, replacement)
    assert result.file_replaced and not reviewed.ready


def test_permission_store_scope_retains_config_lease_and_rejects_other_files(hook_file):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils.private_paths import atomic_private_write_text

    owner = _owner()
    store = owner.snapshot().store_path
    before = set(raw._states)
    with config.locked_hooks_config_snapshot():
        parent = raw._local.operation
        with owner._store_lock(store):
            assert raw._local.operation is not parent
            assert parent in raw._states
            with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
                atomic_private_write_text(store.parent / "unrelated.json", "{}")
        assert raw._local.operation is parent
        with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
            atomic_private_write_text(store, "{}")
    assert set(raw._states) == before


def test_recovery_closes_store_admission_without_reusing_cached_consent(hook_file):
    import time

    from tldw_chatbook.Agents.run_hooks import HookLaunchRefused
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    owner = _owner()
    _approve(owner)
    target = owner.notification_targets("PostToolUse", None)[0]
    participant = raw._raw_participant(owner)
    participant.close_admission()
    try:
        assert participant.drain(time.monotonic() + 1)
        assert not owner.snapshot().ready
        assert owner.notification_targets("PostToolUse", None) == ()
        with pytest.raises(HookLaunchRefused):
            owner.targets("UserPromptSubmit", None)
        with (
            pytest.raises(HookLaunchRefused),
            owner.launch_guard(target, tool_name=None),
        ):
            pytest.fail("recovery admitted a cached hook")
    finally:
        participant.resume()
    assert owner.snapshot().ready


def test_backup_recognizes_but_never_imports_hook_permission_authority(hook_file):
    from tldw_chatbook.Agents.recovery import recovery_adapters

    owner = _owner()
    _approve(owner)
    adapter = next(a for a in recovery_adapters() if a.owner_id == "hooks.permissions")
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )

    values = {
        **config.load_settings(),
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(hook_file, "hooks-test"),
    }
    items = adapter.discover(values)
    assert {item.path.name for item in items} == {
        "hook_permissions.json",
        "hook_permissions.json.lock",
    }
    # Unsupported host metadata is also a restrictive, nonportable result.
    # Neither classification may include device-local grant authority.
    assert all(
        item.status in {"intentionally_excluded", "unsupported"} for item in items
    )
    assert adapter.validate(owner.snapshot().store_path)
    with pytest.raises(
        ValueError, match="device_local_hook_permissions_not_restorable"
    ):
        adapter.relocate(owner.snapshot().store_path, {})


def test_decoder_depth_failure_keeps_permission_recovery_resettable(
    hook_file, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.Agents import hook_permissions

    owner = _owner()
    owner.snapshot()

    def decode(_encoded):
        raise RecursionError("decoder depth exceeded")

    monkeypatch.setattr(
        hook_permissions, "json", SimpleNamespace(loads=decode, dumps=json.dumps)
    )
    broken = owner.snapshot()
    assert not broken.ready
    reset = owner.reset_invalid_state(broken)
    assert not reset.ready
    assert reset.store_revision[0]


@pytest.mark.parametrize("action", ["revoke", "disable"])
def test_stale_mutation_leaves_current_approval_unsealed(hook_file, action):
    from tldw_chatbook.Agents.hook_permissions import HookReviewConflict

    owner = _owner()
    stale = _approve(owner)
    key = stale.rows[0].entry.key
    other = _owner()
    other.approve(other.snapshot(), [key])
    before = hook_file.read_bytes()
    with pytest.raises(HookReviewConflict):
        getattr(owner, action)(stale, key)
    assert owner.snapshot().ready
    assert owner.recover().ready
    assert owner.notification_targets("PostToolUse", None)
    assert hook_file.read_bytes() == before


@pytest.mark.parametrize(
    "event",
    [
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "ApprovalRequested",
        "Stop",
        "SubagentStop",
    ],
)
def test_invalid_master_refuses_cached_launch_and_publishes_no_targets(
    hook_file, event
):
    from tldw_chatbook.Agents.run_hooks import BLOCKING_EVENTS, HookLaunchRefused

    _edit(hook_file, lambda section: section["hook"][0].update(event=event))
    owner = _owner()
    _approve(owner)
    captured = owner.targets(event, None)[0]
    _edit(hook_file, lambda section: section.update(enabled="true"))
    with pytest.raises(HookLaunchRefused), owner.launch_guard(captured, tool_name=None):
        pytest.fail("invalid master switch admitted a captured hook")
    invalid = owner.snapshot()
    assert not invalid.ready
    assert not any(row.state == "approved" for row in invalid.rows)
    if event in BLOCKING_EVENTS:
        with pytest.raises(HookLaunchRefused):
            owner.targets(event, None)
    else:
        assert owner.targets(event, None) == ()
        assert owner.notification_targets(event, None) == ()


@pytest.mark.parametrize("selection", ["name", "directory"])
def test_raw_profile_change_cannot_use_a_throttled_previous_profile_grant(
    hook_file, monkeypatch, selection
):
    owner = _owner()
    _approve(owner)
    old = owner.targets("PostToolUse", None)[0]
    monkeypatch.setattr(config, "_external_edit_detected", lambda path: False)
    raw = toml.loads(hook_file.read_text())
    if selection == "name":
        raw.setdefault("general", {})["users_name"] = "different_profile"
    else:
        raw.setdefault("paths", {})["data_dir"] = str(hook_file.parent / "other_data")
    hook_file.write_text(dumps_cli_config(raw))
    stale = owner.snapshot()
    assert not stale.ready
    assert owner.notification_targets("PostToolUse", None) == ()
    from tldw_chatbook.Agents.run_hooks import HookLaunchRefused

    with pytest.raises(HookLaunchRefused), owner.launch_guard(old, tool_name=None):
        pytest.fail("previous profile consent admitted a new profile hook")


def test_failed_disable_stays_fenced_until_explicit_review(hook_file, monkeypatch):
    owner = _owner()
    current = _approve(owner)
    key = current.rows[0].entry.key
    monkeypatch.setattr(
        config,
        "replace_hooks_config_snapshot",
        lambda *_: config.LiteralConfigMutationResult(False, False, None, "write"),
    )
    assert not owner.disable(current, key).ready
    assert not owner.recover().ready
    assert owner.notification_targets("PostToolUse", None) == ()
    assert owner.approve(owner.snapshot(), [key]).ready
    assert owner.notification_targets("PostToolUse", None)


def test_v2_review_is_persistent_and_legacy_targets_remain_separate(hook_file):
    def add(section):
        section["handler"] = [
            {
                "id": "one",
                "event": "SessionStart",
                "type": "command",
                "argv": [sys.executable, "-c", "pass"],
                "effects": [],
            }
        ]

    _edit(hook_file, add)
    owner = _owner()
    pending = owner.snapshot()
    assert len(pending.rows) == 2
    assert not pending.ready
    assert pending.rows[0].entry.key == "id:one"
    assert pending.rows[1].entry.key == "v2:id:one"
    assert _approve(owner).ready
    assert _owner().snapshot().ready
    assert len(owner.targets("PostToolUse", None)) == 1
    assert owner.targets("SessionStart", None) == ()
    assert "argv" not in pending.store_path.read_text()


def test_changed_v2_policy_and_revoke_retire_exact_epoch(hook_file):
    from tldw_chatbook.Agents.run_hooks import HookLaunchRefused

    _edit(
        hook_file,
        lambda section: section.update(
            handler=[
                {
                    "id": "v2",
                    "event": "SessionStart",
                    "type": "command",
                    "argv": [sys.executable, "-c", "pass"],
                    "effects": [],
                }
            ]
        ),
    )
    owner = _owner()
    assert _approve(owner).ready
    snapshot, targets = owner.v2_configuration()
    assert snapshot.ready and len(targets) == 1
    old = targets[0]
    _edit(hook_file, lambda section: section["handler"][0].update(required=True))
    assert not owner.snapshot().ready
    with pytest.raises(HookLaunchRefused), owner.launch_guard(old, tool_name=None):
        pytest.fail("changed policy launched")
    assert _approve(owner).ready
    _, targets = owner.v2_configuration()
    assert targets[0] != old
    owner.revoke(owner.snapshot(), targets[0].key)
    assert not owner.target_current(targets[0])
    assert _approve(owner).ready
    assert not owner.target_current(targets[0])


@pytest.mark.asyncio
async def test_v2_actual_process_creation_serializes_revoke_and_rejects_late_result(
    hook_file, tmp_path, monkeypatch
):
    import threading

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    marker = tmp_path / "started"
    release_child = tmp_path / "child-release"
    argv = child_argv(
        "from pathlib import Path;import time;"
        f'Path({str(marker)!r}).write_text("started");'
        f"exec({f'while not Path({str(release_child)!r}).exists(): time.sleep(0.01)'!r});"
        'print(\'{"version":2,"decision":"pass"}\')'
    )
    _edit(
        hook_file,
        lambda section: section.update(
            handler=[
                {
                    "id": "guarded-v2",
                    "event": "PreToolUse",
                    "type": "command",
                    "argv": argv,
                    "effects": ["deny"],
                    "timeout_seconds": 30,
                }
            ]
        ),
    )
    owner = _owner()
    expected = _approve(owner)
    _, targets = owner.v2_configuration()
    target = targets[0]
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release_launch = asyncio.Event()
    original = loop.subprocess_exec

    async def paused_launch(*args, **kwargs):
        entered.set()
        await release_launch.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(loop, "subprocess_exec", paused_launch)
    engine = HookEngine(
        (target.spec,),
        lambda *_: owner.target_current(target),
        HookBudgetOwner(),
        launch_guard=lambda *_: owner.launch_guard(target, tool_name=None),
        effect_authority_check=lambda *_: owner.target_current(target, refresh=False),
    )
    firing = asyncio.create_task(engine.fire_async(event()))
    revoke_entered = threading.Event()

    def revoke():
        revoke_entered.set()
        return owner.revoke(expected, target.key)

    revoking = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        revoking = asyncio.create_task(asyncio.to_thread(revoke))
        assert await asyncio.to_thread(revoke_entered.wait, 5)
        await asyncio.sleep(0.05)
        assert not revoking.done(), "revoke passed an in-flight launch transaction"
        release_launch.set()
        assert not (await asyncio.wait_for(revoking, 5)).ready
        async with asyncio.timeout(10):
            while not marker.exists():
                await asyncio.sleep(0.01)
        release_child.write_text("release")
        outcome = await asyncio.wait_for(firing, 10)
        assert not outcome.allowed and not outcome.accepted
        assert not engine.processes.records
    finally:
        release_launch.set()
        release_child.write_text("release")
        await engine.close()
        await asyncio.gather(
            firing, *([revoking] if revoking else []), return_exceptions=True
        )
