"""Qualified real Git runs retain ignores/history and never enter configured helpers."""

import subprocess

import pytest

from Tests.Workspaces.test_native_change_review_ownership import (
    native_review,  # noqa: F401 - shared actual runtime/SQLite fixture
    register,
    wait_for,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepo

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def private_git(native_review, monkeypatch, tmp_path):  # noqa: F811 - imported fixture
    h = native_review
    home = tmp_path / "private-git"
    home.mkdir()
    original = ShadowRepo._env
    monkeypatch.setattr(
        ShadowRepo,
        "_env",
        lambda repo: {
            **original(repo),
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    h.git_home = home
    return h


def test_real_native_git_pins_helpers_preserves_ignore_and_historical_bytes(
    private_git, tmp_path, monkeypatch
):
    h = private_git
    sentinel = tmp_path / "helper-entered"
    helper = tmp_path / "helper"
    helper.write_text(f"#!/bin/sh\nprintf entered > '{sentinel}'\nexit 1\n")
    helper.chmod(0o700)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name in (
        "pre-commit",
        "post-commit",
        "prepare-commit-msg",
        "reference-transaction",
    ):
        (hooks / name).write_bytes(helper.read_bytes())
        (hooks / name).chmod(0o700)
    ignores = tmp_path / "private-ignore"
    ignores.write_text("*.secret\nforced.txt\n")
    config = h.git_home / ".gitconfig"
    config.write_text(
        f"[core]\n hooksPath = {hooks}\n fsmonitor = {helper}\n excludesFile = {ignores}\n"
        f"[commit]\n gpgsign = true\n[gpg]\n program = {helper}\n"
        f'[diff]\n external = {helper}\n[diff "sentinel"]\n textconv = {helper}\n'
        "[gc]\n auto = 1\n autoDetach = true\n[maintenance]\n auto = true\n autoDetach = true\n"
    )
    config_before = config.read_bytes()
    (h.root / ".gitattributes").write_text("*.txt diff=sentinel\n")
    (h.root / "kept.secret").write_text("private ignored content\n")
    nested = h.root / "nested"
    (nested / ".git").mkdir(parents=True)
    (nested / "child.txt").write_text("child before\n")
    reservation = register(h)
    assert reservation.await_baseline(5)
    wait_for(lambda: reservation.baseline_completion.value != "pending")
    assert reservation.baseline_completion.value == "observed", (
        reservation._handle.errors
    )
    before = reservation._handle.baselines[str(h.root)]
    repo = h.tracker.service.repo_for_root(h.root)
    assert repo.file_bytes(before, "kept.secret") is None
    assert repo.file_bytes(before, "a.txt") == b"before\n"
    (h.root / "a.txt").write_text("after\n")
    (h.root / "forced.txt").write_text("explicit touch\n")
    (nested / "child.txt").write_text("child after\n")
    # Warm local drift must not override the invocation's helper pins.
    repo._run("config", "core.fsmonitor", str(helper))
    repo._run("config", "core.hooksPath", str(hooks))
    run_id = h.bridge._db.create_run(conversation_id="capture", agent_kind="primary")
    publish = h.coordinator._publish
    diffs = []

    def with_diff(item):
        for record in item.records:
            if record.root == str(h.root) and not record.tracking_error:
                diffs.append(
                    repo.diff_text(record.baseline_sha, record.end_sha, "a.txt")
                )
        publish(item)

    monkeypatch.setattr(h.coordinator, "_publish", with_diff)
    h.coordinator.finalize(
        reservation,
        run_id=run_id,
        kind="turn",
        touched_paths=[str(h.root / "forced.txt")],
    )
    wait_for(lambda: reservation.completion.value != "pending")
    assert reservation.completion.value == "published"
    end = reservation._handle.end_shas[str(h.root)]
    assert repo.file_bytes(end, "kept.secret") is None
    assert repo.file_bytes(end, "forced.txt") == b"explicit touch\n"
    assert repo.file_bytes(before, "a.txt") == b"before\n"
    assert diffs and "+after" in diffs[0]
    assert not sentinel.exists()
    assert config.read_bytes() == config_before
    assert not (repo.git_dir / "info" / "attributes").exists()
    assert len(h.bridge._db.change_snapshots_for_run(run_id)) == 2


@pytest.mark.parametrize("when", ["init", "warm", "forced", "nested"])
def test_real_filter_no_entry_across_capture_paths(private_git, tmp_path, when):
    h = private_git
    repo = h.tracker.service.repo_for_root(h.root)
    if when != "init":
        repo.snapshot("old supported baseline")
    nested = h.root / "nested"
    if when == "nested":
        (nested / ".git").mkdir(parents=True)
        (nested / "child.txt").write_text("child\n")
    sentinel = tmp_path / "filter-entered"
    helper = tmp_path / "filter"
    helper.write_text(f"#!/bin/sh\nprintf entered > '{sentinel}'\ncat\n")
    helper.chmod(0o700)
    (h.git_home / ".gitconfig").write_text(f'[filter "private"]\nclean = {helper}\n')
    (h.root / ".gitattributes").write_text("*.txt filter=private\n")
    reservation = register(h)
    assert reservation.await_baseline(4)
    assert reservation._handle.errors
    run_id = h.bridge._db.create_run(conversation_id="capture", agent_kind="primary")
    h.coordinator.finalize(
        reservation,
        run_id=run_id,
        kind="turn",
        touched_paths=[str(h.root / "a.txt")] if when == "forced" else (),
    )
    wait_for(lambda: reservation.completion.value != "pending")
    assert not sentinel.exists()
    assert all(
        "configured filters" in row["tracking_error"]
        for row in h.bridge._db.change_snapshots_for_run(run_id)
    )


def test_old_git_refuses_before_config_or_risky_entry(private_git, monkeypatch):
    h = private_git
    original = ShadowRepo._run_process
    commands = []

    def old_version(repo, cmd, env, **kwargs):
        commands.append(tuple(cmd))
        if cmd[-1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "git version 2.35.1\n", "")
        return original(repo, cmd, env, **kwargs)

    monkeypatch.setattr(ShadowRepo, "_run_process", old_version)
    reservation = register(h)
    assert reservation.await_baseline(4)
    h.coordinator.cancel(reservation)
    assert commands and all(cmd[-1] == "--version" for cmd in commands)
    assert ">=2.36" in reservation._handle.errors[str(h.root)]


def test_real_builtin_config_timeout_keeps_original_job_until_process_return(
    private_git, tmp_path, monkeypatch
):
    import os
    import threading

    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Workspaces import change_tracking
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

    if not hasattr(os, "mkfifo"):
        pytest.skip("requires a native FIFO to hold built-in Git config reading")
    h = private_git
    fifo = tmp_path / "held-config"
    os.mkfifo(fifo)
    (h.git_home / ".gitconfig").write_text(f"[include]\npath = {fifo}\n")
    monkeypatch.setattr(change_tracking, "_GIT_TIMEOUT_SECONDS", 0.4)
    entered, returned = threading.Event(), threading.Event()
    run = ShadowRepo._run_process
    owners = []

    def observe(repo, cmd, env, **kwargs):
        if cmd[-1] == "--version":
            owners.append(current_execution_owner())
            entered.set()
            try:
                return run(repo, cmd, env, **kwargs)
            finally:
                returned.set()
        return run(repo, cmd, env, **kwargs)

    monkeypatch.setattr(ShadowRepo, "_run_process", observe)
    reservation = register(h)
    assert entered.wait(3)
    assert owners[0] is not None
    assert not returned.is_set()
    with h.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as contender:
        assert contender is None
    assert not h.coordinator.await_baseline(reservation, 0)
    assert returned.wait(3)
    wait_for(lambda: h.runtime.execution_capacity.snapshot().tool_workers == 0)
    assert reservation.baseline_completion.value == "disclosed_failure"
    h.coordinator.cancel(reservation)


def test_configuration_is_requalified_before_each_risky_invocation(
    private_git, tmp_path, monkeypatch
):
    h = private_git
    reservation = register(h)
    assert reservation.await_baseline(4)
    wait_for(lambda: reservation.baseline_completion.value != "pending")
    sentinel = tmp_path / "late-filter-entered"
    helper = tmp_path / "late-filter"
    helper.write_text(f"#!/bin/sh\nprintf entered > '{sentinel}'\ncat\n")
    helper.chmod(0o700)
    (h.root / ".gitattributes").write_text("*.txt filter=late\n")
    (h.root / "a.txt").write_text("later\n")
    run = ShadowRepo._run
    installed = []

    def install_before_forced_stage(repo, *args, **kwargs):
        if args[0] == "update-index" and not installed:
            installed.append(True)
            (h.git_home / ".gitconfig").write_text(
                f'[filter "late"]\nclean = {helper}\n'
            )
        return run(repo, *args, **kwargs)

    monkeypatch.setattr(ShadowRepo, "_run", install_before_forced_stage)
    run_id = h.bridge._db.create_run(conversation_id="capture", agent_kind="primary")
    h.coordinator.finalize(
        reservation, run_id=run_id, kind="turn", touched_paths=[str(h.root / "a.txt")]
    )
    wait_for(lambda: reservation.completion.value != "pending")
    assert installed and not sentinel.exists()
    assert reservation.completion.value == "disclosed_failure"
    rows = h.bridge._db.change_snapshots_for_run(run_id)
    assert len(rows) == 1 and "configured filters" in rows[0]["tracking_error"]


def test_promisor_configuration_refuses_before_lazy_fetch_capable_entry(
    private_git, monkeypatch, tmp_path
):
    h = private_git
    (h.git_home / ".gitconfig").write_text(
        f'[remote "private"]\npromisor = true\nurl = file://{tmp_path / "not-a-remote"}\n'
    )
    original = ShadowRepo._run_process
    attempted = []

    def no_fetch_probe(repo, cmd, env, **kwargs):
        if "cat-file" in cmd and "-e" in cmd:
            attempted.append(True)
            raise AssertionError("unsafe lazy-fetch-capable entry")
        return original(repo, cmd, env, **kwargs)

    monkeypatch.setattr(ShadowRepo, "_run_process", no_fetch_probe)

    def check_missing_object(_handle):
        h.tracker.service.repo_for_root(h.root).has_snapshot("f" * 40)
        return ()

    monkeypatch.setattr(h.tracker, "discover_baseline", check_missing_object)
    reservation = register(h)
    assert reservation.await_baseline(4)
    h.coordinator.cancel(reservation)
    assert not attempted, "unqualified cat-file can spawn a promisor fetch"
    assert "partial-clone" in reservation._handle.errors[str(h.root)]
