"""Persistent hook consent, invalidation, and real process launch boundaries."""

import asyncio
import json
import sys

import pytest
import toml

from tldw_chatbook import config


@pytest.fixture
def hook_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: data)
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    path.write_text(
        toml.dumps(
            {
                "hooks": {
                    "hook": [
                        {
                            "id": "one",
                            "event": "PostToolUse",
                            "command": [sys.executable, "-c", "pass"],
                            "timeout_s": 5,
                        }
                    ]
                }
            }
        )
    )
    return path


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
    path.write_text(toml.dumps(raw))


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
    assert current.store_path.stat().st_mode & 0o777 == 0o600


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


def test_changed_then_restored_definition_requires_new_review(hook_file):
    owner = _owner()
    _approve(owner)
    _edit(hook_file, lambda section: section["hook"][0].update(timeout_s=6))
    assert not owner.snapshot().ready
    _edit(hook_file, lambda section: section["hook"][0].update(timeout_s=5))
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
async def test_reapprove_cannot_revive_a_target_waiting_to_launch(hook_file, tmp_path):
    import threading
    from contextlib import contextmanager

    from tldw_chatbook.Agents.run_hooks import RunHooksEngine

    marker = tmp_path / "started"
    _edit(
        hook_file,
        lambda section: section["hook"][0].update(
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ]
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
    task = asyncio.create_task(engine.fire_async("PostToolUse", session_id="s"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        current = owner.snapshot()
        owner.revoke(current, current.rows[0].entry.key)
        _approve(owner)
        release.set()
        await asyncio.wait_for(task, 5)
        assert not marker.exists()
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
