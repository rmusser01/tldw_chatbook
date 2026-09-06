"""TASK-1971: B/E turn snapshots around agent runs + change_snapshots schema.

Tracker and bridge tests run against REAL git (no mocks — TASK-1970's rule).
The bridge tests drive the real run loop with a scripted gateway whose
streaming callback writes files mid-turn: that is literally the run-window
side effect the feature exists to catch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX
from Tests.Chat.test_console_agent_bridge import _FakeBuiltinGateForRegistry
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.agent_models import (
    STEP_TOOL_CALL,
    STEP_TOOL_RESULT,
    TOOL_OUTCOME_SUCCESS,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
import tldw_chatbook.Chat.console_agent_bridge as console_agent_bridge_module
import tldw_chatbook.Workspaces.change_tracking as change_tracking_module
from tldw_chatbook.Chat.console_agent_bridge import (
    CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    CHANGE_REVIEW_BASELINE_WAIT_SECONDS,
    ConsoleAgentBridge,
    _ChildChangeState,
    _PostTurnChangeWindow,
    build_change_review_dispatch_gate,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.change_tracking import (
    ChangeTrackingError,
    ShadowRepoService,
)
from tldw_chatbook.Workspaces.change_review_finalization import (
    ChangeReviewFinalizationCoordinator,
    ChangeReviewFinalizeResult,
)
from tldw_chatbook.Workspaces.change_turn_tracker import (
    ChangeTurnTracker,
    TurnChangeRecord,
    TurnHandle,
)
from tldw_chatbook.Workspaces.change_review_consent import SkippedReviewRoot

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp


@pytest.fixture()
def root(tmp_path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    (r / "seed.txt").write_text("seed\n")
    return r


@pytest.fixture()
def tracker(tmp_path) -> ChangeTurnTracker:
    return ChangeTurnTracker(service=ShadowRepoService(data_dir=tmp_path / "appdata"))


# -- tracker level ----------------------------------------------------------


def test_a_turn_records_disk_truth(tracker, root):
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / "new.txt").write_text("created\n")
    (root / "seed.txt").write_text("edited\n")

    records = tracker.end_turn(handle)

    assert len(records) == 1
    rec = records[0]
    assert rec.root == str(root)
    assert rec.tracking_error == ""
    assert rec.files_changed == 2
    assert rec.adds >= 2 and rec.baseline_sha != rec.end_sha


def test_a_clean_turn_yields_no_records(tracker, root):
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    records = tracker.end_turn(handle)
    assert records == []


def test_begin_turn_force_adds_an_ignored_path_into_the_baseline(tracker, root):
    target = root / "ignored-agent-output.txt"
    expected = b"present before the baseline\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    target.write_bytes(expected)

    handle = tracker.begin_turn([root], touched_paths=[str(target)])
    handle.await_baseline()

    baseline = handle.baselines[str(root.resolve())]
    repo = tracker.service.repo_for_root(root)
    assert repo.file_bytes(baseline, target.name) == expected


def test_snapshot_preserves_a_new_ordinary_symlink(tracker, root):
    repo = tracker.service.repo_for_root(root)
    baseline = repo.snapshot("baseline")
    link = root / "seed-link"
    link.symlink_to("seed.txt")

    end = repo.snapshot("new symlink")

    assert end != baseline
    assert repo.file_bytes(end, link.name) == b"seed.txt"
    mode = str(repo._run("ls-tree", end, "--", link.name).stdout).split()[0]
    assert mode == "120000"
    assert repo.last_nested_repos == ()
    assert repo.last_oversize_excluded == ()


def test_snapshot_drops_a_gitlink_that_appears_after_scan(tracker, root, monkeypatch):
    import subprocess as _sp

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    repo = tracker.service.repo_for_root(root)
    prepared = root.parent / "prepared-child"
    prepared.mkdir()
    _sp.run(["git", "init", "--quiet", str(prepared)], check=True)
    (prepared / "inner.txt").write_text("child content\n")
    _sp.run(["git", "-C", str(prepared), "add", "inner.txt"], check=True)
    _sp.run(
        [
            "git",
            "-C",
            str(prepared),
            "-c",
            "user.name=change tracking test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "child seed",
        ],
        check=True,
    )
    child = root / "late-child"
    original_run = type(repo)._run
    appeared = threading.Event()

    def install_child_before_add(self, *args, **kwargs):
        if (
            self.root == root.resolve()
            and not appeared.is_set()
            and args[:2] == ("add", "-A")
        ):
            prepared.rename(child)
            appeared.set()
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(type(repo), "_run", install_child_before_add)
    records = tracker.end_turn(handle)

    assert appeared.is_set()
    assert len(records) == 1
    record = records[0]
    assert record.tracking_error == ""
    assert record.baseline_sha == record.end_sha
    assert record.files_changed == 0
    assert record.nested_repos == (child.name,)
    assert str(repo._run("ls-files", "--stage", "--", child.name).stdout) == ""
    assert str(repo._run("ls-tree", record.end_sha, "--", child.name).stdout) == ""


def test_snapshot_drops_a_gitlink_replacing_a_tip_regular_file_after_scan(
    tracker, root, monkeypatch
):
    import subprocess as _sp

    child = root / "tracked-child"
    child.write_text("ordinary parent file\n")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    repo = tracker.service.repo_for_root(root)
    baseline = handle.baselines[str(root.resolve())]
    baseline_entry = str(repo._run("ls-tree", baseline, "--", child.name).stdout)
    assert baseline_entry.split()[0] == "100644"

    prepared = root.parent / "prepared-replacement-child"
    prepared.mkdir()
    _sp.run(["git", "init", "--quiet", str(prepared)], check=True)
    (prepared / "inner.txt").write_text("child content\n")
    _sp.run(["git", "-C", str(prepared), "add", "inner.txt"], check=True)
    _sp.run(
        [
            "git",
            "-C",
            str(prepared),
            "-c",
            "user.name=change tracking test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "child seed",
        ],
        check=True,
    )
    original_run = type(repo)._run
    appeared = threading.Event()

    def install_child_before_add(self, *args, **kwargs):
        if (
            self.root == root.resolve()
            and not appeared.is_set()
            and args[:2] == ("add", "-A")
        ):
            child.unlink()
            prepared.rename(child)
            appeared.set()
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(type(repo), "_run", install_child_before_add)
    records = tracker.end_turn(handle)

    assert appeared.is_set()
    assert len(records) == 1
    record = records[0]
    assert record.tracking_error == ""
    assert str(repo._run("ls-tree", record.end_sha, "--", child.name).stdout) == ""
    changed = repo.changed_files(record.baseline_sha, record.end_sha)
    assert [(change.path, change.status) for change in changed] == [(child.name, "D")]
    assert record.nested_repos == (child.name,)
    assert str(repo._run("ls-files", "--stage", "--", child.name).stdout) == ""


def test_snapshot_removes_tip_entries_when_directory_becomes_nested_repo(tracker, root):
    import subprocess as _sp

    child = root / "child"
    child.mkdir()
    target = child / "file.txt"
    target.write_text("parent version\n")
    link = child / "file-link"
    link.symlink_to(target.name)
    target_rel = target.relative_to(root).as_posix()
    link_rel = link.relative_to(root).as_posix()

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    repo = tracker.service.repo_for_root(root)
    baseline = handle.baselines[str(root.resolve())]
    assert repo.file_bytes(baseline, target_rel) == b"parent version\n"
    link_entry = str(repo._run("ls-tree", baseline, "--", link_rel).stdout)
    assert link_entry.split()[0] == "120000"

    _sp.run(["git", "init", "--quiet", str(child)], check=True)
    target.write_text("nested version\n")
    records = tracker.end_turn(handle)

    assert len(records) == 1
    record = records[0]
    assert record.tracking_error == ""
    assert repo.file_bytes(record.end_sha, target_rel) is None
    assert repo.file_bytes(record.end_sha, link_rel) is None
    changed = repo.changed_files(record.baseline_sha, record.end_sha)
    assert [(change.path, change.status) for change in changed] == [
        (link_rel, "D"),
        (target_rel, "D"),
    ]
    assert record.nested_repos == (child.name,)
    assert str(repo._run("ls-files", "--", child.name).stdout).strip() == ""


def test_final_index_validation_uses_nul_delimited_force_removals(
    tracker, root, monkeypatch
):
    repo = tracker.service.repo_for_root(root)
    ordinary_paths = tuple(
        f"nested/dir-{index:05d}/{'x' * 180}-{index:05d}.txt" for index in range(20_000)
    )
    overlong = f"nested/{'y' * 5_000}"
    paths = (*ordinary_paths[:10_000], overlong, *ordinary_paths[10_000:])
    object_id = "a" * 40
    stage_entries = tuple(f"100644 {object_id} 0\t{path}" for path in paths)
    calls: list[tuple[tuple[str, ...], dict]] = []

    def fake_z_tokens(*args):
        if args[:4] == ("ls-tree", "-r", "-z", "--name-only"):
            return list(paths)
        if args == ("ls-files", "--stage", "-z"):
            return list(stage_entries)
        raise AssertionError(f"unexpected git token request: {args!r}")

    def record_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(repo, "tip", lambda: "tip")
    monkeypatch.setattr(repo, "_z_tokens", fake_z_tokens)
    monkeypatch.setattr(repo, "_nested_owner", lambda _path: "nested")
    monkeypatch.setattr(repo, "_run", record_run)

    repo._validate_new_index_paths()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("update-index", "--force-remove", "-z", "--stdin")
    assert kwargs == {
        "binary": True,
        "input_data": b"".join(os.fsencode(path) + b"\0" for path in paths),
    }


@pytest.mark.parametrize(
    ("operation", "staging_rounds"),
    (("snapshot", 2), ("force_add", 1)),
)
def test_exact_force_add_paths_use_nul_delimited_stdin(
    tracker, root, monkeypatch, operation, staging_rounds
):
    import subprocess as _sp

    repo = tracker.service.repo_for_root(root)
    repo.snapshot("initial")
    ordinary_paths = tuple(
        f"ignored/dir-{index:05d}/{'x' * 180}-{index:05d}.txt" for index in range(2_000)
    )
    overlong = f"ignored/{'y' * 5_000}"
    paths = (*ordinary_paths[:1_000], overlong, *ordinary_paths[1_000:])
    calls: list[tuple[tuple[str, ...], dict]] = []
    original_run = repo._run

    def record_exact_add(*args, **kwargs):
        if args[:2] == ("update-index", "--add"):
            calls.append((args, kwargs))
            return _sp.CompletedProcess(args, 0, stdout="", stderr="")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(repo, "_exact_force_paths", lambda _paths: list(paths))
    monkeypatch.setattr(repo, "_validate_new_index_paths", lambda: ((), ()))
    monkeypatch.setattr(repo, "_run", record_exact_add)

    if operation == "snapshot":
        repo.snapshot("chunked exact add", force_paths=paths)
    else:
        repo.force_add(paths)

    assert len(calls) == staging_rounds
    expected_input = b"".join(os.fsencode(path) + b"\0" for path in paths)
    assert all(
        args == ("update-index", "--add", "-z", "--stdin")
        and kwargs == {"binary": True, "input_data": expected_input}
        for args, kwargs in calls
    )


def test_force_add_rejects_root_and_directory_paths(tracker, root, monkeypatch):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "100")
    ignored = root / "ignored"
    ignored.mkdir()
    (root / ".gitignore").write_text("ignored/\n")
    (ignored / "small.txt").write_text("small\n")
    oversized = ignored / "oversized.bin"
    oversized.write_bytes(b"x" * 101)

    repo = tracker.service.repo_for_root(root)
    sha = repo.snapshot("unsafe force paths", force_paths=[".", "ignored"])

    assert repo.file_bytes(sha, "ignored/small.txt") is None
    assert repo.file_bytes(sha, "ignored/oversized.bin") is None


def test_force_add_treats_pathspec_magic_as_a_literal_filename(tracker, root):
    target = root / "[ab]"
    expected = b"only this ignored file\n"
    sibling = root / "a"
    (root / ".gitignore").write_text("*\n!/.gitignore\n!/seed.txt\n")
    target.write_bytes(expected)
    sibling.write_text("must stay ignored\n")

    repo = tracker.service.repo_for_root(root)
    sha = repo.snapshot("literal force path", force_paths=[target.name])

    assert repo.file_bytes(sha, target.name) == expected
    assert repo.file_bytes(sha, sibling.name) is None


def test_force_add_refreshes_ignored_executable_bytes_and_mode(tracker, root):
    target = root / "ignored-tool.sh"
    (root / ".gitignore").write_text(f"{target.name}\n")
    target.write_bytes(b"#!/bin/sh\nexit 1\n")
    target.chmod(0o755)
    repo = tracker.service.repo_for_root(root)
    first = repo.snapshot("first executable", force_paths=[target.name])
    first_mode = str(repo._run("ls-tree", first, "--", target.name).stdout).split()[0]
    if first_mode != "100755":
        pytest.skip("filesystem does not expose executable bits to Git")

    expected = b"#!/bin/sh\nexit 0\n"
    target.write_bytes(expected)
    target.chmod(0o644)
    repo.force_add([target.name])

    assert repo._run("show", f":{target.name}", binary=True).stdout == expected
    staged_mode = str(
        repo._run("ls-files", "--stage", "--", target.name).stdout
    ).split()[0]
    assert staged_mode == "100644"
    second = repo.snapshot("updated non-executable")
    assert repo.file_bytes(second, target.name) == expected
    committed_mode = str(
        repo._run("ls-tree", second, "--", target.name).stdout
    ).split()[0]
    assert committed_mode == "100644"


def test_snapshot_force_path_drops_new_blob_that_grows_over_cap_before_index(
    tracker, root, monkeypatch
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "32")
    target = root / "ignored-race.bin"
    (root / ".gitignore").write_text(f"{target.name}\n")
    repo = tracker.service.repo_for_root(root)
    baseline = repo.snapshot("baseline")
    target.write_bytes(b"small")
    (root / "seed.txt").write_text("changed\n")
    original_run = repo._run
    grew = False

    def grow_before_index(*args, **kwargs):
        nonlocal grew
        if not grew and args and args[0] == "update-index" and "--add" in args:
            target.write_bytes(b"x" * 33)
            grew = True
        return original_run(*args, **kwargs)

    repo._run = grow_before_index  # type: ignore[method-assign]
    try:
        end = repo.snapshot("raced snapshot", force_paths=[target.name])
    finally:
        repo._run = original_run  # type: ignore[method-assign]

    assert grew
    assert end != baseline
    assert repo.file_bytes(end, target.name) is None
    assert str(repo._run("ls-files", "--", target.name).stdout).strip() == ""
    assert repo.last_oversize_excluded == (target.name,)


def test_snapshot_final_restage_captures_force_path_that_shrinks_after_drop(
    tracker, root, monkeypatch
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "32")
    target = root / "ignored-race.bin"
    expected = b"small at final boundary"
    (root / ".gitignore").write_text(f"{target.name}\n")
    repo = tracker.service.repo_for_root(root)
    repo.snapshot("baseline")
    target.write_bytes(b"small")
    original_run = repo._run
    grew = False
    shrank = False

    def race_provisional_stage(*args, **kwargs):
        nonlocal grew, shrank
        if not grew and args[:2] == ("update-index", "--add"):
            target.write_bytes(b"x" * 33)
            grew = True
        result = original_run(*args, **kwargs)
        if grew and not shrank and args[:2] == ("add", "-A"):
            target.write_bytes(expected)
            shrank = True
        return result

    repo._run = race_provisional_stage  # type: ignore[method-assign]
    try:
        end = repo.snapshot("raced snapshot", force_paths=[target.name])
    finally:
        repo._run = original_run  # type: ignore[method-assign]

    assert grew and shrank
    assert repo.file_bytes(end, target.name) == expected
    assert repo.last_oversize_excluded == ()


def test_force_path_final_restage_discloses_blob_still_over_cap(
    tracker, root, monkeypatch
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "32")
    target = root / "ignored-race.bin"
    (root / ".gitignore").write_text(f"{target.name}\n")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    target.write_bytes(b"small")
    repo = tracker.service.repo_for_root(root)
    original_run = type(repo)._run
    grew_provisionally = False
    shrank_for_scan = False
    grew_at_final_boundary = False

    def race_both_stages(self, *args, **kwargs):
        nonlocal grew_provisionally, shrank_for_scan, grew_at_final_boundary
        in_root = self.root == root.resolve()
        if in_root and not grew_provisionally and args[:2] == ("update-index", "--add"):
            target.write_bytes(b"x" * 33)
            grew_provisionally = True
        result = original_run(self, *args, **kwargs)
        if (
            in_root
            and grew_provisionally
            and not shrank_for_scan
            and args[:2] == ("update-index", "--add")
        ):
            target.write_bytes(b"small")
            shrank_for_scan = True
        if (
            in_root
            and shrank_for_scan
            and not grew_at_final_boundary
            and args[:2] == ("add", "-A")
        ):
            target.write_bytes(b"y" * 33)
            grew_at_final_boundary = True
        return result

    monkeypatch.setattr(type(repo), "_run", race_both_stages)
    records = tracker.end_turn(handle, touched_paths=[str(target)])

    assert grew_provisionally and shrank_for_scan and grew_at_final_boundary
    assert len(records) == 1
    record = records[0]
    assert record.files_changed == 0
    assert record.untracked_oversize == 1
    assert repo.file_bytes(record.end_sha, target.name) is None


def test_snapshot_force_path_keeps_committed_file_that_grows_over_cap(
    tracker, root, monkeypatch
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "32")
    target = root / "ignored-tracked.bin"
    (root / ".gitignore").write_text(f"{target.name}\n")
    target.write_bytes(b"small")
    repo = tracker.service.repo_for_root(root)
    first = repo.snapshot("small tracked file", force_paths=[target.name])

    expected = b"x" * 33
    target.write_bytes(expected)
    second = repo.snapshot("tracked file grew", force_paths=[target.name])

    assert second != first
    assert repo.file_bytes(second, target.name) == expected
    assert repo.last_oversize_excluded == (target.name,)


def test_force_path_growth_after_scan_is_disclosed_when_post_add_drops_it(
    tracker, root, monkeypatch
):
    from tldw_chatbook.Workspaces import change_bounds

    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "32")
    target = root / "ignored-race.bin"
    (root / ".gitignore").write_text(f"{target.name}\n")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    target.write_bytes(b"small")
    original_scan = change_bounds.scan_root
    grew = False

    def grow_after_scan(*args, **kwargs):
        nonlocal grew
        scan = original_scan(*args, **kwargs)
        if not grew and Path(args[0]).resolve() == root.resolve():
            assert scan.oversized == ()
            target.write_bytes(b"x" * 33)
            grew = True
        return scan

    monkeypatch.setattr(change_bounds, "scan_root", grow_after_scan)
    records = tracker.end_turn(handle, touched_paths=[str(target)])

    assert grew
    assert len(records) == 1
    record = records[0]
    assert record.files_changed == 0
    assert record.untracked_oversize == 1
    repo = tracker.service.repo_for_root(root)
    assert repo.file_bytes(record.end_sha, target.name) is None


def test_snapshot_force_path_file_to_directory_swap_never_stages_descendants(
    tracker, root
):
    target = root / "race-target"
    child_rel = "race-target/ignored-child.txt"
    replacement = root.parent / "snapshot-replacement"
    replacement.mkdir()
    (replacement / "ignored-child.txt").write_text("must not be staged\n")
    (root / ".gitignore").write_text(f"{target.name}\n")
    repo = tracker.service.repo_for_root(root)
    baseline = repo.snapshot("baseline")
    target.write_text("validated file\n")
    original_run = repo._run
    swapped = False

    def swap_before_index(*args, **kwargs):
        nonlocal swapped
        if not swapped and args and args[0] in {"add", "update-index"}:
            target.unlink()
            replacement.replace(target)
            swapped = True
        return original_run(*args, **kwargs)

    repo._run = swap_before_index  # type: ignore[method-assign]
    try:
        repo.snapshot("raced snapshot", force_paths=[target.name])
    except ChangeTrackingError:
        pass

    assert swapped
    assert str(repo._run("ls-files", "--", child_rel).stdout).strip() == ""
    tip = repo.tip()
    assert tip == baseline or repo.file_bytes(tip, child_rel) is None


def test_force_add_file_to_directory_swap_never_stages_descendants(tracker, root):
    target = root / "race-target"
    child_rel = "race-target/ignored-child.txt"
    replacement = root.parent / "force-add-replacement"
    replacement.mkdir()
    (replacement / "ignored-child.txt").write_text("must not be staged\n")
    (root / ".gitignore").write_text(f"{target.name}\n")
    repo = tracker.service.repo_for_root(root)
    baseline = repo.snapshot("baseline")
    target.write_text("validated file\n")
    original_run = repo._run
    swapped = False

    def swap_before_index(*args, **kwargs):
        nonlocal swapped
        if not swapped and args and args[0] in {"add", "update-index"}:
            target.unlink()
            replacement.replace(target)
            swapped = True
        return original_run(*args, **kwargs)

    repo._run = swap_before_index  # type: ignore[method-assign]
    try:
        repo.force_add([target.name])
    except ChangeTrackingError:
        pass

    assert swapped
    assert str(repo._run("ls-files", "--", child_rel).stdout).strip() == ""
    assert repo.tip() == baseline


def test_force_add_rejects_escapes_through_the_shared_path_validator(
    tracker, root, tmp_path, monkeypatch
):
    ignored = root / "ignored"
    ignored.mkdir()
    (ignored / "inside.txt").write_text("ignored\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    link = root / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unsupported on this platform/permission level")
    (root / ".gitignore").write_text("ignored/\noutside-link\n")

    real_validate = change_tracking_module.validate_path
    validated_inputs: list[str] = []

    def record_validation(user_path, base_directory, **kwargs):
        validated_inputs.append(str(user_path))
        return real_validate(user_path, base_directory, **kwargs)

    monkeypatch.setattr(change_tracking_module, "validate_path", record_validation)

    repo = tracker.service.repo_for_root(root)
    sha = repo.snapshot(
        "unsafe force paths",
        force_paths=["", "../outside.txt", str(outside), link.name],
    )

    assert "../outside.txt" in validated_inputs
    assert str(outside) in validated_inputs
    assert link.name in validated_inputs
    assert repo.file_bytes(sha, "ignored/inside.txt") is None
    assert repo.file_bytes(sha, link.name) is None


def test_begin_is_nonblocking_and_await_gates(tmp_path, root):
    """B must ride the model's first-token latency: begin_turn returns
    while the snapshot is still running; await_baseline blocks until done.
    """
    events: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                time.sleep(0.4)
                events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    started = time.monotonic()
    handle = tracker.begin_turn([root])
    events.append("begin-returned")
    begin_elapsed = time.monotonic() - started

    handle.await_baseline()
    events.append("await-returned")

    assert begin_elapsed < 0.3, "begin_turn blocked on the snapshot"
    assert events == ["begin-returned", "baseline-finished", "await-returned"]


def test_change_review_dispatch_gate_bypasses_only_fixed_pure_runtime_tools():
    waits: list[float] = []
    gate = build_change_review_dispatch_gate(
        lambda timeout: waits.append(timeout) or True
    )

    gate(
        [ToolCall(name=name, args={}) for name in CHANGE_REVIEW_BASELINE_BYPASS_TOOLS],
        CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    )
    assert waits == []

    for name in (
        "spawn_subagent",
        "install_skill",
        "run_skill_script",
        "send_to_agent",
        "provider_tool",
        "unknown_tool",
    ):
        gate = build_change_review_dispatch_gate(
            lambda timeout: waits.append(timeout) or True
        )
        gate([ToolCall(name=name, args={})], frozenset())

    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS] * 6

    collision_gate = build_change_review_dispatch_gate(
        lambda timeout: waits.append(timeout) or True
    )
    collision_gate(
        [ToolCall(name="skill_file", args={})],
        frozenset({"find_tools", "load_tools"}),
    )
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS] * 7


def test_change_review_dispatch_gate_waits_for_mixed_batch_and_warns():
    waits: list[float] = []
    warnings: list[bool] = []
    gate = build_change_review_dispatch_gate(
        lambda timeout: waits.append(timeout) or False,
        on_timeout=lambda: warnings.append(True),
    )

    gate(
        [
            ToolCall(name="find_tools", args={}),
            ToolCall(name="provider_tool", args={}),
        ],
        CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    )

    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]
    assert warnings == [True]

    gate(
        [ToolCall(name="provider_tool", args={})],
        CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    )
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]
    assert warnings == [True]


def test_change_review_dispatch_gate_coalesces_concurrent_waiters():
    entered = threading.Event()
    release = threading.Event()
    waits: list[float] = []

    def await_baseline(timeout: float) -> bool:
        waits.append(timeout)
        entered.set()
        assert release.wait(timeout=1)
        return False

    gate = build_change_review_dispatch_gate(await_baseline)
    calls = [ToolCall(name="provider_tool", args={})]
    first = threading.Thread(target=gate, args=(calls, frozenset()))
    second = threading.Thread(target=gate, args=(calls, frozenset()))
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]


def test_tracker_supports_a_caller_owned_synchronous_lifecycle(tracker, root):
    """The app-owned coordinator must be the only owner of worker threads."""
    handle = tracker.new_turn_handle([root])

    tracker.populate_baseline(handle)
    assert handle.await_baseline(timeout=0) is True

    (root / "caller-owned.txt").write_text("changed\n")
    records = tracker.finish_turn(handle)

    assert len(records) == 1
    assert records[0].root == str(root)
    assert records[0].files_changed == 1


def test_timed_out_baseline_rejects_late_success(tmp_path, root):
    entered = threading.Event()
    release = threading.Event()

    class _HeldService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def held_snapshot(message: str) -> str:
                if message == "turn baseline":
                    entered.set()
                    release.wait(timeout=2)
                return original(message)

            repo.snapshot = held_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_HeldService(data_dir=tmp_path / "app"))
    handle = tracker.new_turn_handle([root])
    worker = threading.Thread(target=tracker.populate_baseline, args=(handle,))
    worker.start()
    assert entered.wait(timeout=1)

    assert handle.await_baseline(timeout=0.01) is False
    release.set()
    worker.join(timeout=2)

    records = tracker.finish_turn(handle)
    assert len(records) == 1
    assert "baseline snapshot still running" in records[0].tracking_error
    assert records[0].baseline_sha == ""


def test_force_add_carveout_for_tool_touched_ignored_paths(tracker, root):
    """A tool write to a .gitignore'd path (.env is the canonical case) must
    surface; a SCRIPT write into an ignored directory stays a documented
    blind spot (force-adding everything would false-positive pre-existing
    ignored files as Added).
    """
    (root / ".gitignore").write_text(".env\nignored_dir/\n")
    ignored_dir = root / "ignored_dir"
    ignored_dir.mkdir()

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / ".env").write_text("SECRET=1\n")
    (ignored_dir / "side_effect.txt").write_text("script wrote this\n")

    records = tracker.end_turn(handle, touched_paths=[str(root / ".env")])

    assert len(records) == 1
    changed = tracker.service.repo_for_root(root).changed_files(
        records[0].baseline_sha, records[0].end_sha
    )
    paths = [c.path for c in changed]
    assert ".env" in paths, "the tool-touched ignored file is invisible"
    assert not any("side_effect" in p for p in paths), (
        "script writes into ignored dirs are OUT of scope by design"
    )


def test_snapshot_force_path_drops_file_when_nested_marker_appears_during_final_restage(
    tracker, root, monkeypatch
):
    child = root / "late-child"
    child.mkdir()
    target = child / "ignored-write.txt"
    target_rel = target.relative_to(root).as_posix()
    (root / ".gitignore").write_text(f"/{target_rel}\n")

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    target.write_text("must stay child-owned\n")
    repo = tracker.service.repo_for_root(root)
    original_run = type(repo)._run
    marker_created = threading.Event()
    exact_stages = 0

    def create_marker_before_index(self, *args, **kwargs):
        nonlocal exact_stages
        if self.root == root.resolve() and args[:2] == ("update-index", "--add"):
            exact_stages += 1
            if exact_stages == 2:
                (child / ".git").mkdir()
                marker_created.set()
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(type(repo), "_run", create_marker_before_index)
    records = tracker.end_turn(handle, touched_paths=[str(target)])

    assert marker_created.is_set()
    assert exact_stages == 2
    assert len(records) == 1
    record = records[0]
    assert record.files_changed == 0
    assert record.nested_repos == (child.name,)
    assert str(repo._run("ls-files", "--", target_rel).stdout).strip() == ""
    assert repo.file_bytes(record.end_sha, target_rel) is None


def test_force_path_under_auto_registered_nested_repo_is_owned_only_by_child(
    tracker, root
):
    import subprocess as _sp

    child = root / "childrepo"
    child.mkdir()
    _sp.run(["git", "init", "--quiet", str(child)], check=True)
    (child / ".gitignore").write_text("ignored-write.txt\n")
    (child / "seed.txt").write_text("child seed\n")
    target = child / "ignored-write.txt"
    expected = b"child-owned ignored write\n"

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    target.write_bytes(expected)
    records = tracker.end_turn(handle, touched_paths=[str(target)])

    by_root = {record.root: record for record in records}
    parent_key = str(root.resolve())
    child_key = str(child.resolve())
    assert parent_key not in by_root
    assert set(by_root) == {child_key}
    child_record = by_root[child_key]
    child_repo = tracker.service.repo_for_root(child)
    assert child_repo.file_bytes(child_record.end_sha, target.name) == expected
    parent_repo = tracker.service.repo_for_root(root)
    parent_rel = target.relative_to(root).as_posix()
    assert str(parent_repo._run("ls-files", "--", parent_rel).stdout).strip() == ""
    assert parent_repo.file_bytes(parent_repo.tip(), parent_rel) is None


def test_supplied_successor_sha_defers_a_late_ignored_path_to_successor_e(
    tracker, root
):
    target = root / "ignored-agent-output.txt"
    expected = b"created after successor baseline\n"
    (root / ".gitignore").write_text(f"{target.name}\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    successor = tracker.begin_turn([root])
    successor.await_baseline()
    key = str(root.resolve())
    supplied = successor.baselines[key]
    target.write_bytes(expected)

    continuation_records = tracker.end_turn(
        continuation,
        touched_paths=[str(target)],
        end_shas=successor.baselines,
        successor_handle=successor,
    )
    assert continuation_records == []
    assert continuation.end_shas[key] == supplied

    successor_records = tracker.end_turn(successor)
    assert len(successor_records) == 1
    assert successor_records[0].baseline_sha == supplied
    repo = tracker.service.repo_for_root(root)
    assert repo.file_bytes(successor_records[0].end_sha, target.name) == expected


def test_supplied_boundary_does_not_leak_force_paths_to_another_conversation(
    tracker, root
):
    target = root / "ignored-agent-output.txt"
    expected = b"owned by the claimed successor\n"
    (root / ".gitignore").write_text(f"{target.name}\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    successor = tracker.begin_turn([root])
    successor.await_baseline()
    unrelated = tracker.begin_turn([root])
    unrelated.await_baseline()
    target.write_bytes(expected)

    assert (
        tracker.end_turn(
            continuation,
            touched_paths=[str(target)],
            end_shas=successor.baselines,
            successor_handle=successor,
        )
        == []
    )

    assert tracker.end_turn(unrelated) == []
    successor_records = tracker.end_turn(successor)

    assert len(successor_records) == 1
    repo = tracker.service.repo_for_root(root)
    record = successor_records[0]
    assert repo.file_bytes(record.end_sha, target.name) == expected


def test_supplied_sha_deferred_file_growing_before_successor_e_is_disclosed(
    tracker, root, monkeypatch
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "8")
    target = root / "x"
    (root / ".gitignore").write_text(f"{target.name}\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    successor = tracker.begin_turn([root])
    successor.await_baseline()
    key = str(root.resolve())
    supplied = successor.baselines[key]
    target.write_bytes(b"small")

    assert (
        tracker.end_turn(
            continuation,
            touched_paths=[str(target)],
            end_shas=successor.baselines,
            successor_handle=successor,
        )
        == []
    )
    assert continuation.end_shas[key] == supplied
    repo = tracker.service.repo_for_root(root)
    assert str(repo._run("ls-files", "--", target.name).stdout).strip() == ""

    target.write_bytes(b"x" * 9)
    records = tracker.end_turn(successor)

    assert len(records) == 1
    record = records[0]
    assert record.baseline_sha == supplied
    assert record.end_sha == supplied
    assert record.files_changed == 0
    assert record.untracked_oversize == 1
    assert repo.file_bytes(record.end_sha, target.name) is None
    assert str(repo._run("ls-files", "--", target.name).stdout).strip() == ""


def test_supplied_sha_deferred_path_becoming_nested_before_successor_e_is_disclosed(
    tracker, root
):
    child = root / "late-child"
    child.mkdir()
    target = child / "ignored-write.txt"
    target_rel = target.relative_to(root).as_posix()
    (root / ".gitignore").write_text(f"/{target_rel}\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    successor = tracker.begin_turn([root])
    successor.await_baseline()
    key = str(root.resolve())
    supplied = successor.baselines[key]
    target.write_text("must stay child-owned\n")

    assert (
        tracker.end_turn(
            continuation,
            touched_paths=[str(target)],
            end_shas=successor.baselines,
            successor_handle=successor,
        )
        == []
    )
    assert continuation.end_shas[key] == supplied
    repo = tracker.service.repo_for_root(root)
    assert str(repo._run("ls-files", "--", target_rel).stdout).strip() == ""

    (child / ".git").mkdir()
    records = tracker.end_turn(successor)

    assert len(records) == 1
    record = records[0]
    assert record.baseline_sha == supplied
    assert record.end_sha == supplied
    assert record.files_changed == 0
    assert record.nested_repos == (child.name,)
    assert repo.file_bytes(record.end_sha, target_rel) is None
    assert str(repo._run("ls-files", "--", target_rel).stdout).strip() == ""


def test_supplied_sha_deferred_path_treats_pathspec_magic_as_a_literal_filename(
    tracker, root
):
    target = root / "[ab]"
    expected = b"literal target\n"
    sibling = root / "a"
    (root / ".gitignore").write_text("*\n!/.gitignore\n!/seed.txt\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    successor = tracker.begin_turn([root])
    successor.await_baseline()
    key = str(root.resolve())
    supplied = successor.baselines[key]
    target.write_bytes(expected)
    sibling.write_text("must stay ignored\n")

    assert (
        tracker.end_turn(
            continuation,
            touched_paths=[str(target)],
            end_shas=successor.baselines,
            successor_handle=successor,
        )
        == []
    )
    successor_records = tracker.end_turn(successor)

    assert len(successor_records) == 1
    assert successor_records[0].baseline_sha == supplied
    repo = tracker.service.repo_for_root(root)
    assert repo.file_bytes(successor_records[0].end_sha, target.name) == expected
    assert repo.file_bytes(successor_records[0].end_sha, sibling.name) is None


def test_deferred_supplied_path_growing_over_cap_is_disclosed_by_successor(
    tracker, root, monkeypatch
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "32")
    target = root / "ignored-race.bin"
    (root / ".gitignore").write_text(f"{target.name}\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    (root / "boundary.txt").write_text("at supplied boundary\n")
    successor = tracker.begin_turn([root])
    successor.await_baseline()
    key = str(root.resolve())
    supplied = successor.baselines[key]
    assert supplied != continuation.baselines[key]
    target.write_bytes(b"small")
    repo = tracker.service.repo_for_root(root)
    original_run = type(repo)._run
    grew = False

    def grow_before_index(self, *args, **kwargs):
        nonlocal grew
        if (
            not grew
            and self.root == root.resolve()
            and args
            and args[0] == "update-index"
            and "--add" in args
        ):
            target.write_bytes(b"x" * 33)
            grew = True
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(type(repo), "_run", grow_before_index)
    continuation_records = tracker.end_turn(
        continuation,
        touched_paths=[str(target)],
        end_shas=successor.baselines,
        successor_handle=successor,
    )
    records = tracker.end_turn(successor)

    assert grew
    assert continuation.end_shas[key] == supplied
    assert len(continuation_records) == 1
    assert len(records) == 1
    record = records[0]
    assert record.baseline_sha == supplied
    assert record.end_sha == supplied
    assert record.tracking_error == ""
    assert record.untracked_oversize == 1
    assert str(repo._run("ls-files", "--", target.name).stdout).strip() == ""


def test_deferred_supplied_path_becoming_nested_is_disclosed_by_successor(
    tracker, root, monkeypatch
):
    child = root / "late-child"
    child.mkdir()
    target = child / "ignored-write.txt"
    target_rel = target.relative_to(root).as_posix()
    (root / ".gitignore").write_text(f"/{target_rel}\n")

    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None

    (root / "boundary.txt").write_text("at supplied boundary\n")
    successor = tracker.begin_turn([root])
    successor.await_baseline()
    key = str(root.resolve())
    supplied = successor.baselines[key]
    assert supplied != continuation.baselines[key]
    target.write_text("must stay child-owned\n")
    repo = tracker.service.repo_for_root(root)
    original_run = type(repo)._run
    marker_created = threading.Event()

    def create_marker_before_index(self, *args, **kwargs):
        if (
            self.root == root.resolve()
            and not marker_created.is_set()
            and args[:2] == ("update-index", "--add")
        ):
            (child / ".git").mkdir()
            marker_created.set()
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(type(repo), "_run", create_marker_before_index)
    continuation_records = tracker.end_turn(
        continuation,
        touched_paths=[str(target)],
        end_shas=successor.baselines,
        successor_handle=successor,
    )
    records = tracker.end_turn(successor)

    assert marker_created.is_set()
    assert continuation.end_shas[key] == supplied
    assert len(continuation_records) == 1
    assert len(records) == 1
    record = records[0]
    assert record.baseline_sha == supplied
    assert record.end_sha == supplied
    assert record.tracking_error == ""
    assert record.nested_repos == (child.name,)
    assert str(repo._run("ls-files", "--", target_rel).stdout).strip() == ""
    assert repo.file_bytes(supplied, target_rel) is None


def test_deferred_force_path_snapshot_failure_is_disclosed_by_successor(
    tracker, root, monkeypatch
):
    target = root / "ignored-agent-output.txt"
    (root / ".gitignore").write_text(f"{target.name}\n")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    key = str(root.resolve())
    baseline = handle.baselines[key]

    (root / "boundary.txt").write_text("at supplied boundary\n")
    successor = tracker.begin_turn([root])
    successor.await_baseline()
    supplied = successor.baselines[key]
    assert supplied != baseline
    target.write_text("late ignored write\n")
    repo = tracker.service.repo_for_root(root)
    original_snapshot = type(repo).snapshot

    def fail_deferred_snapshot(self, message, *, force_paths=()):
        if force_paths:
            raise RuntimeError("injected deferred snapshot failure")
        return original_snapshot(self, message, force_paths=force_paths)

    monkeypatch.setattr(type(repo), "snapshot", fail_deferred_snapshot)
    records = tracker.end_turn(
        handle,
        touched_paths=[str(target)],
        end_shas={key: supplied},
        successor_handle=successor,
    )
    successor_records = tracker.end_turn(successor)

    assert handle.end_shas.get(key) == supplied
    assert len(records) == 1
    assert records[0].baseline_sha == baseline
    assert records[0].end_sha == supplied
    assert records[0].tracking_error == ""
    assert len(successor_records) == 1
    assert successor_records[0].tracking_error == "injected deferred snapshot failure"


def test_invalid_supplied_sha_does_not_defer_or_stage_the_path(tracker, root):
    target = root / "ignored-agent-output.txt"
    (root / ".gitignore").write_text(f"{target.name}\n")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    key = str(root.resolve())
    baseline = handle.baselines[key]
    target.write_text("must not be staged\n")
    invalid = "f" * 40
    repo = tracker.service.repo_for_root(root)
    records = tracker.end_turn(
        handle,
        touched_paths=[str(target)],
        end_shas={key: invalid},
    )

    assert handle.end_shas[key] == invalid
    assert len(records) == 1
    record = records[0]
    assert record.baseline_sha == baseline
    assert record.end_sha == invalid
    assert record.tracking_error
    assert str(repo._run("ls-files", "--", target.name).stdout).strip() == ""


def test_supplied_sha_preserves_nonempty_continuation_range_statistics(tracker, root):
    parent = tracker.begin_turn([root])
    parent.await_baseline()
    assert tracker.end_turn(parent) == []
    continuation = tracker.continuation(parent)
    assert continuation is not None
    key = str(root.resolve())
    baseline = continuation.baselines[key]

    (root / "between-boundaries.txt").write_text("one\ntwo\n")
    successor = tracker.begin_turn([root])
    successor.await_baseline()
    supplied = successor.baselines[key]
    assert supplied != baseline

    records = tracker.end_turn(continuation, end_shas=successor.baselines)

    assert continuation.end_shas[key] == supplied
    assert len(records) == 1
    assert records[0].baseline_sha == baseline
    assert records[0].end_sha == supplied
    assert records[0].files_changed == 1
    assert records[0].adds == 2
    assert records[0].dels == 0


def test_tracking_failure_yields_error_records_never_raises(tmp_path, root):
    tracker = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    records = tracker.end_turn(handle)
    assert len(records) == 1
    assert records[0].tracking_error != ""
    assert records[0].end_sha == ""


def test_tool_touched_paths_reads_write_tools_only():
    class _Step:
        def __init__(self, tool_name, args):
            self.tool_name = tool_name
            self.args = args

    steps = [
        _Step("write_file", {"file_path": "/w/a.txt", "content": "x"}),
        _Step("read_file", {"file_path": "/w/read-only.txt"}),
        _Step("calculator", {"expression": "1+1"}),
        _Step("write_file", {"file_path": "/w/b.txt", "content": "y"}),
    ]
    touched = ChangeTurnTracker.tool_touched_paths(steps)
    assert touched == ["/w/a.txt", "/w/b.txt"], (
        "read touches would force-add pre-existing ignored files and lie "
        f"an Added row: {touched}"
    )


# -- DB level ---------------------------------------------------------------


def test_v2_database_gains_the_change_snapshots_table_on_open(tmp_path):
    """The DB has no migration framework by design: CREATE IF NOT EXISTS on
    every open IS the mechanism. A file created at v2 must gain the table."""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
        INSERT INTO schema_version (version) VALUES (2);
        CREATE TABLE agent_runs (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
            parent_run_id TEXT, agent_kind TEXT NOT NULL, task TEXT,
            status TEXT NOT NULL, steps TEXT NOT NULL DEFAULT '[]',
            result TEXT, budget TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, assistant_message_id TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    db = AgentRunsDB(db_path, client_id="t")
    run_id = db.create_run(conversation_id="c1", agent_kind="primary")
    db.record_change_snapshot(
        run_id=run_id,
        root="/w/root",
        baseline_sha="b" * 8,
        end_sha="e" * 8,
        files_changed=2,
        adds=3,
        dels=1,
    )
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["root"] == "/w/root"
    assert rows[0]["adds"] == 3

    by_conv = db.change_snapshots_for_conversation("c1")
    assert [r["run_id"] for r in by_conv] == [run_id]


def test_change_snapshot_batch_commits_one_complete_window(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="c1", agent_kind="primary")
    records = [
        {
            "root": "/w/one",
            "baseline_sha": "b1",
            "end_sha": "e1",
            "files_changed": 1,
            "adds": 2,
            "dels": 0,
        },
        {
            "root": "/w/two",
            "baseline_sha": "b2",
            "end_sha": "e2",
            "tracking_error": "snapshot failed",
        },
    ]

    db.record_change_snapshots_batch(run_id=run_id, records=records, kind="turn")

    rows = db.change_snapshots_for_run(run_id)
    assert [(row["root"], row["tracking_error"]) for row in rows] == [
        ("/w/one", ""),
        ("/w/two", "snapshot failed"),
    ]


# -- bridge level -----------------------------------------------------------


class _SideEffectGateway:
    """Streams a scripted reply and, mid-stream, runs a side-effect callback
    — the exact run-window write the tracker must attribute to the turn.

    Matches the real gateway contract (async generator, positional
    resolution/messages) — the first version was a sync generator and every
    bridge test failed with an empty run.
    """

    def __init__(self, scripts, side_effect=None, explode=False, side_effect_on_call=1):
        self._scripts = list(scripts)
        self._side_effect = side_effect
        self._explode = explode
        self._side_effect_on_call = side_effect_on_call
        self._calls = 0

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        script = self._scripts[min(self._calls, len(self._scripts) - 1)]
        self._calls += 1
        if self._side_effect is not None and self._calls >= self._side_effect_on_call:
            self._side_effect()
            self._side_effect = None
        for chunk in script:
            yield chunk
        if self._explode and self._calls >= len(self._scripts):
            raise RuntimeError("provider died mid-turn")


def _bridge_with(tmp_path, gateway, tracker, coordinator=None):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        change_tracker=tracker,
        change_finalization_coordinator=coordinator,
    )
    return bridge, db, store, session, assistant.id


def _run(bridge, session, assistant_id, root, **over):
    kwargs = dict(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="",
            model="test-model",
            ready=True,
            execution_key="llama_cpp",
        ),
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
        change_roots=[root],
    )
    kwargs.update(over)
    return bridge.run_reply(**kwargs)


def _calc_fence() -> str:
    return (
        f"{FENCE_OPEN}\n"
        + json.dumps({"name": "calculator", "arguments": {"expression": "6*7"}})
        + "\n```"
    )


def test_bridge_run_records_a_change_row_matching_disk(tmp_path, root, tracker):
    """The side effect fires on the SECOND provider call -- i.e. after the
    first tool batch has passed the await-B gate. Writing during the FIRST
    provider stream would race the baseline thread (warm process: gateway
    wins, the write lands inside B and vanishes) -- a window that exists
    only for writers that bypass tools, which production has none of: every
    writer, scripts included, is a tool behind the gate. The first version
    of this test wrote pre-gate and passed only by cold-start luck.
    """
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made_by_run.txt").write_text("hello\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, outcome = _run(bridge, session, aid, root)

    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["files_changed"] == 1
    changed = tracker.service.repo_for_root(root).changed_files(
        rows[0]["baseline_sha"], rows[0]["end_sha"]
    )
    assert [c.path for c in changed] == ["made_by_run.txt"]


def test_bridge_run_with_no_changes_records_no_row(tmp_path, root, tracker):
    gateway = _SideEffectGateway([["done."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _outcome = _run(bridge, session, aid, root)
    assert db.change_snapshots_for_run(run_id) == []


def test_bridge_returns_before_coordinated_end_snapshot_finishes(tmp_path, root):
    end_entered = threading.Event()
    release_end = threading.Event()

    class _HeldEndTracker(ChangeTurnTracker):
        def finish_turn(self, handle, touched_paths=(), *, end_shas=None):
            end_entered.set()
            release_end.wait(timeout=2)
            return super().finish_turn(
                handle, touched_paths=touched_paths, end_shas=end_shas
            )

    tracker = _HeldEndTracker(service=ShadowRepoService(data_dir=tmp_path / "appdata"))
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made_by_run.txt").write_text("hello\n"),
    )
    publications = []
    db_holder = {}

    def publish(item):
        publications.append(item)
        db_holder["db"].record_change_snapshots_batch(
            run_id=item.run_id,
            records=[record.__dict__ for record in item.records],
            kind=item.kind,
        )

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=4,
    )
    bridge, db, store, session, aid = _bridge_with(
        tmp_path, gateway, tracker, coordinator
    )
    db_holder["db"] = db

    run_id, outcome = _run(bridge, session, aid, root)

    assert outcome.final_text.strip() == "done."
    assert end_entered.wait(timeout=1)
    assert db.change_snapshots_for_run(run_id) == []
    release_end.set()
    assert coordinator.wait_idle(timeout=2)
    assert len(db.change_snapshots_for_run(run_id)) == 1
    coordinator.shutdown(timeout=1)


def test_bridge_surfaces_capacity_error_when_error_channel_is_saturated(
    tmp_path, root, tracker
):
    class _Reservation:
        roots = (str(root),)
        admission_error = "change-review error publication channel is at capacity"

        @staticmethod
        def await_baseline(timeout=120.0):
            del timeout
            return True

    class _SaturatedCoordinator:
        @staticmethod
        def register(_roots, *, survivor_key=""):
            return _Reservation()

        @staticmethod
        def finalize(_reservation, **_kwargs):
            return ChangeReviewFinalizeResult.OVERLOAD_VISIBLE

    coordinator = _SaturatedCoordinator()
    bridge, _db, store, session, aid = _bridge_with(
        tmp_path,
        _SideEffectGateway([["done."]]),
        tracker,
        coordinator,
    )

    _run(
        bridge,
        session,
        aid,
        root,
        change_root_aliases=["folder-safe"],
    )

    failures = [
        message.content
        for message in _tool_rows(store, session)
        if "change tracking failed" in message.content
    ]
    assert len(failures) == 1
    assert "error publication channel is at capacity" in failures[0]
    assert "folder-safe" in failures[0]
    assert str(root.resolve()) not in failures[0]


def test_bridge_does_not_append_capacity_marker_after_coordinator_shutdown(
    tmp_path, root, tracker
):
    class _Reservation:
        roots = (str(root),)
        admission_error = "change-review coordinator is at capacity"

        @staticmethod
        def await_baseline(timeout=120.0):
            del timeout
            return True

    class _StoppedCoordinator:
        @staticmethod
        def register(_roots, *, survivor_key=""):
            return _Reservation()

        @staticmethod
        def finalize(_reservation, **_kwargs):
            return ChangeReviewFinalizeResult.REJECTED

    bridge, _db, store, session, aid = _bridge_with(
        tmp_path,
        _SideEffectGateway([["done."]]),
        tracker,
        _StoppedCoordinator(),
    )

    _run(bridge, session, aid, root)

    assert not [
        message
        for message in _tool_rows(store, session)
        if "change tracking failed" in message.content
    ]


def test_third_turn_starts_while_second_review_finalization_is_held(tmp_path, root):
    second_end_entered = threading.Event()
    release_second_end = threading.Event()

    class _HoldSecondEndTracker(ChangeTurnTracker):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ends = 0

        def finish_turn(self, handle, touched_paths=(), *, end_shas=None):
            self.ends += 1
            if self.ends == 2:
                second_end_entered.set()
                release_second_end.wait(timeout=3)
            return super().finish_turn(
                handle, touched_paths=touched_paths, end_shas=end_shas
            )

    class _ThreeTurnGateway(_SideEffectGateway):
        def __init__(self):
            super().__init__([["one"], ["two"], ["three"]])
            self.third_started = threading.Event()

        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            if self._calls == 2:
                self.third_started.set()
            async for chunk in super().stream_chat(
                resolution, messages, tools=tools, **kwargs
            ):
                yield chunk

    tracker = _HoldSecondEndTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    gateway = _ThreeTurnGateway()
    db_holder = {}

    def publish(item):
        db_holder["db"].record_change_snapshots_batch(
            run_id=item.run_id,
            records=[record.__dict__ for record in item.records],
            kind=item.kind,
        )

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=4,
    )
    bridge, db, store, session, first_assistant = _bridge_with(
        tmp_path, gateway, tracker, coordinator
    )
    db_holder["db"] = db

    _run(bridge, session, first_assistant, root)
    assert coordinator.wait_idle(timeout=2)

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="two")
    second_assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    second_run_id, _second_outcome = _run(bridge, session, second_assistant.id, root)
    bridge.record_run_assistant_message(second_run_id, "persisted-second")
    assert db.get_run(second_run_id)["assistant_message_id"] == "persisted-second"
    assert second_end_entered.wait(timeout=1)

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="three")
    third_assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    third_result = []
    third_thread = threading.Thread(
        target=lambda: third_result.append(
            _run(bridge, session, third_assistant.id, root)
        )
    )
    third_thread.start()

    assert gateway.third_started.wait(timeout=1), (
        "turn three remained blocked behind turn two's file-review E snapshot"
    )
    release_second_end.set()
    third_thread.join(timeout=3)
    assert third_result and third_result[0][1].final_text.strip() == "three"
    assert coordinator.wait_idle(timeout=3)
    coordinator.shutdown(timeout=1)


def test_cancelled_turn_still_schedules_coordinated_finalization(tmp_path, root):
    tracker = ChangeTurnTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    gateway = _SideEffectGateway([["never used"]])
    publications = []
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publications.append,
        worker_count=1,
        capacity=2,
    )
    bridge, _db, _store, session, aid = _bridge_with(
        tmp_path, gateway, tracker, coordinator
    )
    scheduled = []
    original_finalize = coordinator.finalize

    def recording_finalize(*args, **kwargs):
        scheduled.append(kwargs["run_id"])
        return original_finalize(*args, **kwargs)

    coordinator.finalize = recording_finalize  # type: ignore[method-assign]

    run_id, outcome = _run(
        bridge,
        session,
        aid,
        root,
        should_cancel=lambda: True,
    )

    assert outcome.status == "cancelled"
    assert scheduled == [run_id]
    assert coordinator.wait_idle(timeout=2)
    coordinator.shutdown(timeout=1)


def test_failed_run_still_records_its_end_snapshot(tmp_path, root, tracker):
    """A run that died halfway through editing is when review matters MOST."""
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["partial"]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "half_done.txt").write_text("partial\n"),
        explode=True,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, outcome = _run(bridge, session, aid, root)

    assert outcome.status != "completed"
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1, "the failed run's half-finished edits are unreviewable"


def test_tracking_never_blocks_the_reply(tmp_path, root):
    """Spec failure posture: git broken -> the agent reply still completes."""
    broken = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    gateway = _SideEffectGateway([["fine."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, broken)

    run_id, outcome = _run(bridge, session, aid, root)

    assert outcome.final_text.strip() == "fine."
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1 and rows[0]["tracking_error"] != ""


def test_review_runs_before_baseline_gate_and_tool_dispatch_waits(tmp_path, root):
    """Permission review precedes the bounded B gate; invocation follows B."""
    events: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                if "baseline" in message:
                    time.sleep(0.5)
                    events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    fence = (
        f"{FENCE_OPEN}\n"
        + json.dumps({"name": "calculator", "arguments": {"expression": "6*7"}})
        + "\n```"
    )
    gateway = _SideEffectGateway(
        [[fence], ["42."]],
        side_effect=lambda: events.append("tool-finished"),
        side_effect_on_call=2,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    # PR2a Task 5: an AgentService-wired hook takes `(calls, run_id)` --
    # the change-tracker wrapper passes it straight through.
    def probe_review(calls, run_id):
        events.append("review-called")
        return {}

    run_id, outcome = _run(bridge, session, aid, root, review_tool_calls=probe_review)

    assert "baseline-finished" in events and "review-called" in events
    assert events.index("review-called") < events.index("baseline-finished"), (
        f"permission review did not precede the baseline gate: {events}"
    )
    assert events.index("baseline-finished") < events.index("tool-finished"), (
        f"tool dispatch raced ahead of the baseline: {events}"
    )


def test_bridge_timeout_continues_dispatch_and_warns_with_root_alias(
    tmp_path, root, tracker
):
    waits: list[float] = []

    class _Reservation:
        def __init__(self) -> None:
            self.roots = (str(root.resolve()),)
            self.admission_error = ""
            self._handle = TurnHandle([root.resolve()])

    class _TimeoutCoordinator:
        @staticmethod
        def register(_roots, *, survivor_key=""):
            return _Reservation()

        @staticmethod
        def await_baseline(reservation, timeout):
            waits.append(timeout)
            reservation._handle.errors[str(root.resolve())] = (
                "baseline snapshot still running after 3s"
            )
            return False

        @staticmethod
        def finalize(_reservation, **_kwargs):
            return ChangeReviewFinalizeResult.SCHEDULED

    bridge, _db, store, session, aid = _bridge_with(
        tmp_path,
        _SideEffectGateway([[_calc_fence()], ["done."]]),
        tracker,
        _TimeoutCoordinator(),
    )

    _run(
        bridge,
        session,
        aid,
        root,
        change_root_aliases=["folder-safe"],
    )

    warnings = [
        row.content
        for row in _tool_rows(store, session)
        if row.content.startswith("⚠ change review skipped")
    ]
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]
    assert warnings == [
        "⚠ change review skipped folder-safe: baseline timed out; "
        "this turn's changes are not tracked"
    ]
    assert str(root.resolve()) not in warnings[0]


# -- wiring: roots resolution + registration hook ---------------------------


def test_folder_binding_roots_includes_ro_and_never_sandbox(tmp_path, monkeypatch):
    """Tracking is about what happened on DISK: a script can write into a
    read-only root even though the file tools cannot, so ro bindings are in.
    The sandbox is app-managed scratch and would be pure review noise.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="t")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="A")
    rw = tmp_path / "rw"
    rw.mkdir()
    ro = tmp_path / "ro"
    ro.mkdir()
    registry.add_folder_binding("ws-a", rw, allow_write=True)
    registry.add_folder_binding("ws-a", ro)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

    registry.set_change_review_enabled("ws-a", True)
    roots = wfr.folder_binding_roots("ws-a")

    assert set(roots) == {rw.resolve(), ro.resolve()}
    assert wfr.folder_binding_roots(None) == ()


def test_app_owner_snapshots_an_enabled_folder_binding_in_background(tmp_path):
    """The attached bounded owner, not registry persistence, prepares roots."""
    import time as _time

    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import (
        ChangeReviewConsentService,
        LocalWorkspaceRegistryService,
    )

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="t")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="A")
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "code.py").write_text("x = 1\n")

    review = ChangeReviewConsentService(registry)
    registry.attach_change_review_consent_service(review)
    registry.set_change_review_enabled("ws-a", True)
    try:
        registry.add_folder_binding("ws-a", folder)

        service = ShadowRepoService()
        deadline = _time.monotonic() + 15.0
        tip = None
        while _time.monotonic() < deadline:
            tip = service.repo_for_root(folder).tip()
            if tip:
                break
            _time.sleep(0.05)
        assert tip, "the registered root never received its initial snapshot"
    finally:
        review.shutdown(timeout=1.0)


def test_carveout_survives_a_symlink_spelled_root(tmp_path):
    """Review finding: `_paths_within` resolves each touched path but the
    roots were stored UNRESOLVED — a run whose root arrived spelled through
    a symlink made `relative_to` fail, silently skipping the force-add: the
    `.env` carve-out dying without a trace.
    """
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    (real_root / ".gitignore").write_text(".env\n")
    link = tmp_path / "root_link"
    try:
        link.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported on this platform/permission level")

    tracker = ChangeTurnTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    handle = tracker.begin_turn([link])  # the SYMLINK spelling
    handle.await_baseline()
    (real_root / ".env").write_text("SECRET=1\n")

    records = tracker.end_turn(handle, touched_paths=[str(real_root / ".env")])

    assert len(records) == 1 and not records[0].tracking_error
    changed = tracker.service.repo_for_root(real_root).changed_files(
        records[0].baseline_sha, records[0].end_sha
    )
    assert ".env" in [c.path for c in changed], (
        "the carve-out silently died for a symlink-spelled root"
    )


# -- TASK-1972: the transcript summary row ----------------------------------


def _tool_rows(store, session):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    return [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]


def test_a_change_turn_emits_the_summary_row_with_real_counts(tmp_path, root, tracker):
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("one\ntwo\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, _ = _run(bridge, session, aid, root)

    rows = [m for m in _tool_rows(store, session) if m.content.startswith("✎")]
    assert len(rows) == 1
    assert "1 file" in rows[0].content
    assert "+2" in rows[0].content
    assert rows[0].change_review_run_id == run_id, (
        "the row does not know WHICH turn it reviews"
    )


def test_a_clean_turn_emits_no_summary_row(tmp_path, root, tracker):
    gateway = _SideEffectGateway([["done."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    _run(bridge, session, aid, root)
    assert not [m for m in _tool_rows(store, session) if m.content.startswith("✎")]


def test_tracking_failure_emits_the_warning_row(tmp_path, root):
    broken = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    gateway = _SideEffectGateway([["fine."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, broken)
    _run(bridge, session, aid, root)
    warns = [
        m for m in _tool_rows(store, session) if "change tracking failed" in m.content
    ]
    assert len(warns) == 1, "a tracking failure must be DISCLOSED in the transcript"


@pytest.mark.parametrize(
    ("alias", "reason"),
    [
        ("folder-preparing", "Preparing change history"),
        ("folder-failed", "Change history preparation failed"),
    ],
)
def test_skipped_review_root_emits_alias_only_warning_without_snapshot_state(
    tmp_path, tracker, alias, reason
):
    """Readiness warnings never masquerade as canonical-root snapshots."""
    gateway = _SideEffectGateway([["done."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, outcome = _run(
        bridge,
        session,
        aid,
        tmp_path / "unused-root",
        change_roots=[],
        change_review_skipped_roots=(SkippedReviewRoot(alias=alias, reason=reason),),
    )

    assert outcome.status == "done"
    warnings = [
        row
        for row in _tool_rows(store, session)
        if "change review skipped" in row.content.lower()
    ]
    assert [row.content for row in warnings] == [
        f"⚠ change review skipped {alias}: {reason}"
    ]
    assert db.change_snapshots_for_run(run_id) == []
    assert db.roots_with_change_snapshots() == set()


def test_summary_row_survives_the_next_message(tmp_path, root, tracker):
    """TASK-1842's whole arc: display-only rows must survive recompute."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    _run(bridge, session, aid, root)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="follow-up")
    rows = [m for m in _tool_rows(store, session) if m.content.startswith("✎")]
    assert len(rows) == 1, "the summary row was destroyed by the next message"


def test_resume_re_derives_the_summary_row_byte_identical(tmp_path, root, tracker):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _ = _run(bridge, session, aid, root)
    live = [m for m in _tool_rows(store, session) if m.content.startswith("✎")]
    assert live, "precondition"

    fresh = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    resumed = [
        m
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if m.content.startswith("✎")
    ]
    assert [m.content for m in resumed] == [m.content for m in live]
    assert resumed[0].change_review_run_id == run_id
    projected = [
        message
        for _anchor, block in fresh.change_review_marker_messages("conv-1")
        for message in block
    ]
    assert [message.content for message in projected] == [
        message.content for message in live
    ]
    assert all(message.change_review_run_id == run_id for message in projected)


def test_review_changes_action_offered_only_for_summary_rows():
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_message_actions import (
        ConsoleMessageActionService,
    )

    summary = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="✎ Edited 1 file  +2 −0 — review with `v`",
        change_review_run_id="run-1",
    )
    plain_marker = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, content="⚙ calculator → 42"
    )
    svc = ConsoleMessageActionService()
    assert "review-changes" in [a.action_id for a in svc.available_actions(summary)]
    assert "review-changes" not in [
        a.action_id for a in svc.available_actions(plain_marker)
    ]


def test_bridge_exposes_a_provider_for_the_review_screen(tmp_path, root, tracker):
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _ = _run(bridge, session, aid, root)

    provider = bridge.change_review_provider("conv-1")
    assert provider is not None
    turns = provider.turns()
    assert [t.run_id for t in turns] == [run_id]

    untracked = ConsoleAgentBridge = None  # noqa: F841 -- reuse import below
    from tldw_chatbook.Chat.console_agent_bridge import (
        ConsoleAgentBridge as _B,
    )

    no_tracker = _B(agent_runs_db=db, store=store, provider_gateway=gateway)
    assert no_tracker.change_review_provider("conv-1") is None


@pytest.mark.asyncio
async def test_the_opener_pushes_the_screen_and_selects_the_turn(
    tmp_path, root, tracker
):
    """The `v`/inspector opener on the PRODUCTION ChatScreen: derives the
    run-store conversation id, builds the provider through the bridge, pushes
    the Review screen, and selects THAT turn. The opener is where an invented
    method name already slipped in once during this task -- it needs a test
    on the real screen object, not a reading.
    """
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.change_review_screen import ChangeReviewScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    # run_reply spins its own event loop; inside an async test that loop
    # collides with pytest-asyncio's ("Cannot run the event loop while
    # another loop is running"). Production calls it via asyncio.to_thread
    # -- so does this test.
    import asyncio as _asyncio

    run_id, outcome = await _asyncio.to_thread(_run, bridge, session, aid, root)
    assert outcome.status not in ("error",), outcome.steps
    assert db.change_snapshots_for_run(run_id), "precondition: the run recorded rows"

    class _ConsoleHarness(ConsolidatedCSSApp):
        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(ChatScreen(self.app_instance))

    app = _build_test_app()
    # Same native-ready configuration the workbench harness applies -- the
    # Console controller is built lazily and stays None without it.
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    harness = _ConsoleHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chat_screen = harness.screen_stack[-1]
        assert isinstance(chat_screen, ChatScreen)
        # The harness app has no chachanotes db, so its own bridge factory
        # returns None by design -- substitute THIS test's real bridge
        # (real tracker, real db, real turn) at the accessor seam.
        chat_screen._ensure_console_agent_bridge = lambda: bridge
        chat_screen._ensure_console_chat_controller()
        controller = chat_screen._console_chat_controller
        assert controller is not None
        controller.store.ensure_session()
        # The run-store id for the harness's session is the session id
        # itself (no persisted conversation) -- point the bridge's provider
        # at the id the run actually recorded under instead.
        chat_screen._console_chat_controller._agent_conversation_id = lambda _sid: (
            "conv-1"
        )

        chat_screen._open_change_review(run_id)
        review = await _wait_for_screen(harness, pilot, ChangeReviewScreen)
        assert review is not None, "the opener never pushed the Review screen"

        turns = review._provider.turns()
        assert [t.run_id for t in turns] == [run_id]


@pytest.mark.asyncio
async def test_the_summary_rows_own_review_action_opens_the_screen(
    tmp_path, root, tracker
):
    """TASK-2030 (live-UAT headline defect): selecting the rendered ✎ row
    and invoking its review action must open the Review screen.

    The defect class: TOOL markers are display-only rows, deliberately NOT
    tree nodes, so `store.get_message(marker_id)` ALWAYS raises -- and the
    action handler resolved the store row before dispatching, killing the
    row's own advertised affordance ("review with `v`") on the live app
    while the direct-call opener test stayed green. This test goes through
    the REAL chain the user does: transcript render -> row selection ->
    `invoke_selected_action("review-changes")` -> button dispatch ->
    handler -> pushed screen.
    """
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.change_review_screen import ChangeReviewScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleTranscript,
    )

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    import asyncio as _asyncio

    run_id, outcome = await _asyncio.to_thread(_run, bridge, session, aid, root)
    assert outcome.status not in ("error",), outcome.steps
    marker = next(m for m in _tool_rows(store, session) if m.content.startswith("✎"))
    assert marker.change_review_run_id == run_id

    class _ConsoleHarness(ConsolidatedCSSApp):
        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(ChatScreen(self.app_instance))

    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    harness = _ConsoleHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chat_screen = harness.screen_stack[-1]
        assert isinstance(chat_screen, ChatScreen)
        # The run's rows live in THIS test's store -- make it the screen's
        # store before the lazy controller builds, then render it for real.
        chat_screen._console_chat_store = store
        chat_screen._ensure_console_agent_bridge = lambda: bridge
        chat_screen._ensure_console_chat_controller()
        controller = chat_screen._console_chat_controller
        assert controller is not None
        chat_screen._console_chat_controller._agent_conversation_id = lambda _sid: (
            "conv-1"
        )
        await chat_screen._sync_native_console_transcript()
        await pilot.pause()

        transcript = chat_screen.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        rendered = [
            m
            for m in transcript._messages
            if str(getattr(m, "content", "")).startswith("✎")
        ]
        assert rendered, "precondition: the ✎ row rendered in the transcript"
        transcript.select_message(rendered[0].id)
        await pilot.pause()
        transcript.action_invoke_selected_action("review-changes")
        await pilot.pause()

        review = await _wait_for_screen(harness, pilot, ChangeReviewScreen)
        assert review is not None, (
            "the ✎ row's own review action never opened the Review screen"
        )
        turns = review._provider.turns()
        assert [t.run_id for t in turns] == [run_id]

        # AC#3: a genuinely-unknown target still gets the failure toast --
        # the display-model path is not a blanket bypass of resolution.
        harness.pop_screen()
        await pilot.pause()
        toasts: list[str] = []
        app.notify = lambda msg, **kw: toasts.append(str(msg))

        class _Btn:
            id = "console-message-action-review-changes-not-a-row"

        class _Ev:
            button = _Btn()

            def stop(self) -> None:
                pass

        handled = await chat_screen.handle_console_message_action(_Ev())
        assert handled is True
        assert any("no longer exists" in t for t in toasts), toasts
        assert not isinstance(harness.screen, ChangeReviewScreen)


async def _wait_for_screen(harness, pilot, screen_type, timeout: float = 8.0):
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        if isinstance(harness.screen, screen_type):
            return harness.screen
        await pilot.pause(0.05)
    return None


def test_resume_re_derives_the_failure_row_too(tmp_path, root):
    """Review finding: live emitted the ⚠ tracking-failed row but resume
    did not -- a resumed transcript silently hid that a turn's tracking
    failed, breaking the byte-identical marker parity rule."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    broken = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    gateway = _SideEffectGateway([["fine."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, broken)
    _run(bridge, session, aid, root)
    live = [
        m.content
        for m in _tool_rows(store, session)
        if "change tracking failed" in m.content
    ]
    assert live, "precondition: the live run disclosed the failure"

    fresh = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    resumed = [
        m.content
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if "change tracking failed" in m.content
    ]
    assert resumed == live, "the failure disclosure vanished on resume"


# -- PR3a-1 Task 6c (audit F2): a survivor's writes vs the turn window ------
#
# PR 3a lets a sub-agent outlive the `run_reply` that spawned it. The turn's
# E snapshot is taken in that call's `finally`, so every byte a survivor
# writes afterwards falls OUTSIDE its own turn's window -- into the NEXT
# turn's baseline (invisible in every record) or into the next turn's diff
# (attributed to an agent that never made it). Both are silent.


class _FleetSurvivorGateway:
    """One primary script per turn, plus a sub-agent turn gated on an Event.

    The child's disk write fires on the child's own thread the moment the
    gate opens -- the same "run-window side effect" technique
    `_SideEffectGateway` uses for the primary, moved onto a survivor.

    The gate is awaited through ``run_in_executor`` for the reason
    `Tests/Chat/test_console_agent_bridge.py::_FleetTwoChildGateway`
    documents at length: a bare ``.wait()`` inside a coroutine blocks the
    one thread driving that loop.
    """

    def __init__(
        self,
        parent_scripts,
        gate: threading.Event,
        child_side_effect=None,
        parent_side_effect=None,
        parent_side_effect_on_call: int = 0,
        child_scripts=None,
        second_gate: "threading.Event | None" = None,
    ):
        self._parent = list(parent_scripts)
        self._child = list(child_scripts or [["child answer"]])
        self._gate = gate
        self._second_gate = second_gate
        self._child_side_effect = child_side_effect
        self._parent_side_effect = parent_side_effect
        self._parent_side_effect_on_call = parent_side_effect_on_call
        self._lock = threading.Lock()
        self.parent_calls = 0
        self.child_calls = 0
        self.child_started = threading.Event()
        self.child_second_started = threading.Event()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        if system.startswith(SUBAGENT_PROMPT_PREFIX):
            with self._lock:
                self.child_calls += 1
                first_call = self.child_calls == 1
                chunks = self._child.pop(0) if self._child else ["child answer"]
            loop = asyncio.get_running_loop()
            if first_call:
                self.child_started.set()
                await loop.run_in_executor(None, self._gate.wait)
                if self._child_side_effect is not None:
                    side_effect, self._child_side_effect = (
                        self._child_side_effect,
                        None,
                    )
                    side_effect()
            elif self._second_gate is not None:
                self.child_second_started.set()
                # The child keeps RUNNING after its write -- how a test
                # pins a window that must be closed by the next turn
                # rather than by the child finishing.
                await loop.run_in_executor(None, self._second_gate.wait)
            for chunk in chunks:
                yield chunk
            return
        with self._lock:
            assert self._parent, "parent script exhausted"
            chunks = self._parent.pop(0)
            self.parent_calls += 1
            fire = (
                self._parent_side_effect is not None
                and self.parent_calls == self._parent_side_effect_on_call
            )
        if fire:
            self._parent_side_effect()
        for chunk in chunks:
            yield chunk


def _spawn_fence(task: str) -> str:
    return (
        f"{FENCE_OPEN}\n"
        + json.dumps({"name": "spawn_subagent", "arguments": {"task": task}})
        + "\n```"
    )


def _write_fence(path: Path, content: str) -> str:
    return (
        f"{FENCE_OPEN}\n"
        + json.dumps(
            {
                "name": "write_file",
                "arguments": {"file_path": str(path), "content": content},
            }
        )
        + "\n```"
    )


def _join_fleet_threads(timeout: float = 5.0) -> None:
    """Block until every live fleet child thread has fully finished.

    Copied from `Tests/Chat/test_console_agent_bridge.py` for the same
    reason it exists there: a child's run row goes terminal slightly
    BEFORE its thread unwinds, so joining the thread -- not polling -- is
    what guarantees any scope wrapping that run has already exited.
    """
    for thread in list(threading.enumerate()):
        if thread.name.startswith("fleet-"):
            thread.join(timeout)


def _next_turn(store, session):
    """Append the next user/assistant pair, as a real second Send does."""
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="again")
    return store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id


class _WaitRecordingEvent(threading.Event):
    """Event whose waiter-entry is itself a deterministic test barrier."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_started = threading.Event()

    def wait(self, timeout=None):
        self.wait_started.set()
        return super().wait(timeout)


class _BlockingParentGateway:
    """Keep one primary reply inside the provider until its test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.entered.set()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release.wait)
        yield "successor done"


def test_post_turn_real_write_file_surfaces_a_new_ignored_path(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "ignored-agent-output.txt"
    sentinel = "written by the surviving child\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    assert not target.exists()

    gate = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[[_spawn_fence("write ignored output")], ["parent done"]],
        gate=gate,
        child_scripts=[[_write_fence(target, sentinel)], ["child done"]],
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_id, outcome = _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
        assert outcome.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        assert not target.exists(), "the child wrote before its parent returned"
    finally:
        gate.set()
        _join_fleet_threads()

    child_runs = [
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1, child_runs
    assert child_runs[0]["status"] == "done"
    successful_writes = [
        step
        for step in child_runs[0]["steps"]
        if step["kind"] == STEP_TOOL_RESULT
        and step.get("tool_name") == "write_file"
        and step.get("tool_outcome") == TOOL_OUTCOME_SUCCESS
    ]
    assert len(successful_writes) == 1, child_runs[0]["steps"]
    assert target.read_text() == sentinel

    repo = tracker.service.repo_for_root(root)
    rows = db.change_snapshots_for_run(run_id)
    rows_listing_target = [
        row
        for row in rows
        if row["kind"] == "subagent_post_turn"
        and target.name
        in [
            changed.path
            for changed in repo.changed_files(row["baseline_sha"], row["end_sha"])
        ]
    ]
    assert len(rows_listing_target) == 1, rows
    assert (
        repo.file_bytes(rows_listing_target[0]["end_sha"], target.name)
        == sentinel.encode()
    )


def test_pending_child_before_scope_entry_keeps_ignored_write_reviewable(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "pending-child-output.txt"
    sentinel = "written after delayed child scope entry\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    child_write_gate = threading.Event()
    scope_waiting = threading.Event()
    enter_scope = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[[_spawn_fence("write delayed output")], ["parent done"]],
        gate=child_write_gate,
        # Relative tool input exercises scratch-root normalization before
        # the path crosses parent E.
        child_scripts=[[_write_fence(Path(target.name), sentinel)], ["child done"]],
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    original_scope = bridge._child_run_scope

    @contextlib.contextmanager
    def delayed_scope(*args, **kwargs):
        scope_waiting.set()
        assert enter_scope.wait(5), "test barrier timed out before child scope entry"
        with original_scope(*args, **kwargs):
            yield

    monkeypatch.setattr(bridge, "_child_run_scope", delayed_scope)
    try:
        run_id, outcome = _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
        assert outcome.status == "done"
        assert scope_waiting.wait(5), "child never reached the pre-scope barrier"
        assert not target.exists(), "child wrote before its parent returned"
        enter_scope.set()
        assert gateway.child_started.wait(5), "child never entered its real scope"
        child_write_gate.set()
        _join_fleet_threads()
    finally:
        enter_scope.set()
        child_write_gate.set()
        _join_fleet_threads()

    assert target.read_text() == sentinel
    assert bridge._child_change_states == {}
    child_runs = [
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1, child_runs
    successful_writes = [
        step
        for step in child_runs[0]["steps"]
        if step["kind"] == STEP_TOOL_RESULT
        and step.get("tool_name") == "write_file"
        and step.get("tool_outcome") == TOOL_OUTCOME_SUCCESS
    ]
    assert len(successful_writes) == 1, child_runs[0]["steps"]

    repo = tracker.service.repo_for_root(root)
    rows = db.change_snapshots_for_run(run_id)
    survivor_rows = [row for row in rows if row["kind"] == "subagent_post_turn"]
    assert len(survivor_rows) == 1, (
        "the pending child's ignored WRITE path was omitted from survivor close: "
        f"{rows}"
    )
    changed = repo.changed_files(
        survivor_rows[0]["baseline_sha"], survivor_rows[0]["end_sha"]
    )
    assert [item.path for item in changed] == [target.name], changed
    assert (
        repo.file_bytes(survivor_rows[0]["end_sha"], target.name) == sentinel.encode()
    )


def test_pending_inherited_child_at_successor_b_marks_concurrent_turn(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "pending-successor-output.txt"
    sentinel = "written by a child that was pending at successor B\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    child_write_gate = threading.Event()
    scope_waiting = threading.Event()
    enter_scope = threading.Event()

    def release_child_during_successor() -> None:
        enter_scope.set()
        child_write_gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("write during successor")],
            ["parent one done"],
            [_calc_fence()],
            ["parent two done"],
        ],
        gate=child_write_gate,
        parent_side_effect=release_child_during_successor,
        parent_side_effect_on_call=4,
        child_scripts=[[_write_fence(Path(target.name), sentinel)], ["child done"]],
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    original_scope = bridge._child_run_scope

    @contextlib.contextmanager
    def delayed_scope(*args, **kwargs):
        scope_waiting.set()
        assert enter_scope.wait(5), "test barrier timed out before child scope entry"
        with original_scope(*args, **kwargs):
            yield

    monkeypatch.setattr(bridge, "_child_run_scope", delayed_scope)
    try:
        _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
        assert scope_waiting.wait(5), "child never reached the pre-scope barrier"
        assert bridge._live_child_count("conv-1") == 0
        assert not target.exists()

        run_2, outcome_2 = _run(
            bridge,
            session,
            _next_turn(store, session),
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
        assert outcome_2.status == "done"
    finally:
        enter_scope.set()
        child_write_gate.set()
        _join_fleet_threads()

    repo = tracker.service.repo_for_root(root)
    rows = db.change_snapshots_for_run(run_2)
    rows_listing_target = [
        row
        for row in rows
        if target.name
        in [
            changed.path
            for changed in repo.changed_files(row["baseline_sha"], row["end_sha"])
        ]
    ]
    assert len(rows_listing_target) == 1, rows
    assert rows_listing_target[0]["kind"] == "turn_concurrent_subagent"
    assert any(
        "earlier turn" in message.content and "sub-agent" in message.content
        for message in _tool_rows(store, session)
    )


def test_child_write_during_blocked_parent_e_is_retried_by_immediate_close(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "blocked-parent-e-output.txt"
    sentinel = "written while parent E was blocked\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    child_write_gate = threading.Event()
    scope_waiting = threading.Event()
    enter_scope = threading.Event()
    end_started = threading.Event()
    release_end = threading.Event()
    scope_exited = threading.Event()
    allow_settle = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[[_spawn_fence("write during parent E")], ["parent done"]],
        gate=child_write_gate,
        child_scripts=[[_write_fence(target, sentinel)], ["child done"]],
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    original_scope = bridge._child_run_scope
    repo = tracker.service.repo_for_root(root)
    original_snapshot = repo.snapshot
    original_repo_for_root = tracker.service.repo_for_root

    @contextlib.contextmanager
    def pause_after_real_scope(*args, **kwargs):
        scope_waiting.set()
        assert enter_scope.wait(5), "test barrier timed out before child scope entry"
        with original_scope(*args, **kwargs):
            yield
        scope_exited.set()
        assert allow_settle.wait(5), "test barrier timed out before child settle"

    def instrumented_repo_for_root(candidate):
        if Path(candidate).expanduser().resolve() == root.resolve():
            return repo
        return original_repo_for_root(candidate)

    def blocked_snapshot(message, *args, **kwargs):
        if message == "turn end":
            end_started.set()
            assert release_end.wait(5), "test barrier timed out inside parent E"
        return original_snapshot(message, *args, **kwargs)

    monkeypatch.setattr(bridge, "_child_run_scope", pause_after_real_scope)
    monkeypatch.setattr(tracker.service, "repo_for_root", instrumented_repo_for_root)
    monkeypatch.setattr(repo, "snapshot", blocked_snapshot)
    result: dict[str, object] = {}

    def run_parent() -> None:
        try:
            result["value"] = _run(
                bridge,
                session,
                aid,
                root,
                builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
                scratch_root=root,
                scratch_lease=lambda: contextlib.nullcontext(root),
            )
        except BaseException as exc:  # noqa: BLE001 -- re-raised on test thread
            result["error"] = exc

    parent = threading.Thread(target=run_parent, name="blocked-parent-e")
    parent.start()
    try:
        assert scope_waiting.wait(5), "child never reached the pre-scope barrier"
        assert end_started.wait(5), "parent never reached its E snapshot"
        assert not target.exists(), "child wrote before parent E capture"
        enter_scope.set()
        assert gateway.child_started.wait(5), "child never reached its WRITE gate"
        child_write_gate.set()
        assert scope_exited.wait(5), "real child scope never exited"
        assert bridge._live_child_count("conv-1") == 0
        assert target.read_text() == sentinel
        assert parent.is_alive(), "parent E was not held by the test barrier"
        assert bridge._child_change_states.get("conv-1"), (
            "child settle crossed the test barrier and removed live state"
        )
        release_end.set()
        parent.join(10)
        assert not parent.is_alive(), "parent did not leave E after barrier release"

        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        run_id, outcome = result["value"]  # type: ignore[misc]
        assert outcome.status == "done"
        assert bridge._post_turn_change_windows.get("conv-1") is None, (
            "a pre-scope child that exited during E stranded its window"
        )

        rows = db.change_snapshots_for_run(run_id)
        survivor_rows = [row for row in rows if row["kind"] == "subagent_post_turn"]
        assert len(survivor_rows) == 1, (
            "the pre-scope child's ignored WRITE path published during E was "
            f"omitted from immediate survivor close: {rows}"
        )
        changed = repo.changed_files(
            survivor_rows[0]["baseline_sha"], survivor_rows[0]["end_sha"]
        )
        assert [item.path for item in changed] == [target.name], changed
        assert (
            repo.file_bytes(survivor_rows[0]["end_sha"], target.name)
            == sentinel.encode()
        )
    finally:
        enter_scope.set()
        child_write_gate.set()
        release_end.set()
        allow_settle.set()
        parent.join(10)
        _join_fleet_threads()

    assert bridge._child_change_states == {}


def test_settled_child_before_parent_e_keeps_ignored_write_reviewable(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "settled-child-output.txt"
    sentinel = "written by child before parent E\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    child_write_gate = threading.Event()
    child_settled = threading.Event()

    def settle_child_before_parent_final() -> None:
        assert gateway.child_started.wait(5), "child never reached its WRITE gate"
        child_write_gate.set()
        _join_fleet_threads()
        child_settled.set()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("write before parent E")],
            ["parent done"],
        ],
        gate=child_write_gate,
        child_scripts=[[_write_fence(target, sentinel)], ["child done"]],
        parent_side_effect=settle_child_before_parent_final,
        parent_side_effect_on_call=2,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_id, outcome = _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
    finally:
        child_write_gate.set()
        _join_fleet_threads()

    assert outcome.status == "done"
    assert child_settled.is_set(), "child did not fully settle before parent E"
    assert bridge._child_change_states == {}
    assert target.read_text() == sentinel

    child_runs = [
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1, child_runs
    assert any(
        step["kind"] == STEP_TOOL_RESULT
        and step.get("tool_name") == "write_file"
        and step.get("tool_outcome") == TOOL_OUTCOME_SUCCESS
        for step in child_runs[0]["steps"]
    ), child_runs[0]["steps"]

    rows = db.change_snapshots_for_run(run_id)
    turn_rows = [row for row in rows if row["kind"] == "turn"]
    assert len(turn_rows) == 1, (
        f"the settled child's local WRITE state was omitted from parent E: {rows}"
    )
    assert not [row for row in rows if row["kind"] == "subagent_post_turn"], rows
    assert bridge._post_turn_change_windows.get("conv-1") is None
    repo = tracker.service.repo_for_root(root)
    changed = repo.changed_files(turn_rows[0]["baseline_sha"], turn_rows[0]["end_sha"])
    assert [item.path for item in changed] == [target.name], changed
    assert repo.file_bytes(turn_rows[0]["end_sha"], target.name) == sentinel.encode()


def test_child_write_path_normalization_failure_is_best_effort(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "bad-normalization.txt"
    sibling = root / "good-sibling.txt"
    sentinel = "the child still completed its WRITE\n"
    child_write_gate = threading.Event()
    child_settled = threading.Event()
    normalization_failed = threading.Event()

    def settle_child_before_parent_final() -> None:
        assert gateway.child_started.wait(5), "child never reached its WRITE gate"
        child_write_gate.set()
        _join_fleet_threads()
        child_settled.set()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("write despite tracking failure")],
            ["parent done"],
        ],
        gate=child_write_gate,
        child_scripts=[
            [_write_fence(Path(target.name), sentinel)],
            ["child done"],
        ],
        parent_side_effect=settle_child_before_parent_final,
        parent_side_effect_on_call=2,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    child_states: list[_ChildChangeState] = []
    original_scope = bridge._child_run_scope
    original_touched_paths = ChangeTurnTracker.tool_touched_paths
    original_resolve = Path.resolve

    @contextlib.contextmanager
    def capture_child_state(conversation_id, adapter, child_change_state):
        child_states.append(child_change_state)
        with original_scope(conversation_id, adapter, child_change_state):
            yield

    def touched_paths_with_sibling(steps):
        steps = tuple(steps)
        if (
            len(steps) == 1
            and steps[0].kind == STEP_TOOL_CALL
            and steps[0].tool_name == "write_file"
            and steps[0].args
            == {
                "file_path": target.name,
                "content": sentinel,
            }
        ):
            return [target.name, sibling.name]
        return original_touched_paths(steps)

    def fail_target_once(path, *args, **kwargs):
        if path == target and not normalization_failed.is_set():
            normalization_failed.set()
            raise OSError("synthetic path normalization failure")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(bridge, "_child_run_scope", capture_child_state)
    monkeypatch.setattr(
        ChangeTurnTracker,
        "tool_touched_paths",
        staticmethod(touched_paths_with_sibling),
    )
    monkeypatch.setattr(Path, "resolve", fail_target_once)
    try:
        _, outcome = _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
    finally:
        child_write_gate.set()
        _join_fleet_threads()

    assert normalization_failed.is_set(), "the normalization failure did not fire"
    assert outcome.status == "done"
    assert child_settled.is_set(), "child did not fully settle before parent E"
    assert target.read_text() == sentinel
    child_runs = [
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1, child_runs
    assert child_runs[0]["status"] == "done"
    assert any(
        step["kind"] == STEP_TOOL_RESULT
        and step.get("tool_name") == "write_file"
        and step.get("tool_outcome") == TOOL_OUTCOME_SUCCESS
        for step in child_runs[0]["steps"]
    ), child_runs[0]["steps"]
    assert len(child_states) == 1
    assert child_states[0].touched_paths == {str(sibling)}


def test_scratch_root_normalization_failure_keeps_child_step_processing(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "raw-relative-output.txt"
    sentinel = "the scratch authority still handled this WRITE\n"
    child_write_gate = threading.Event()
    child_settled = threading.Event()
    normalization_failed = threading.Event()

    class FailingProjectionRoot(type(root)):
        def resolve(self, *args, **kwargs):
            if self == root and not normalization_failed.is_set():
                normalization_failed.set()
                raise OSError("synthetic scratch-root normalization failure")
            return super().resolve(*args, **kwargs)

    scratch_root = FailingProjectionRoot(root)

    def settle_child_before_parent_final() -> None:
        assert gateway.child_started.wait(5), "child never reached its WRITE gate"
        child_write_gate.set()
        _join_fleet_threads()
        child_settled.set()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("write after scratch projection failure")],
            ["parent done"],
        ],
        gate=child_write_gate,
        child_scripts=[
            [_write_fence(Path(target.name), sentinel)],
            ["child done"],
        ],
        parent_side_effect=settle_child_before_parent_final,
        parent_side_effect_on_call=2,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    child_states: list[_ChildChangeState] = []
    original_scope = bridge._child_run_scope

    @contextlib.contextmanager
    def capture_child_state(conversation_id, adapter, child_change_state):
        child_states.append(child_change_state)
        with original_scope(conversation_id, adapter, child_change_state):
            yield

    monkeypatch.setattr(bridge, "_child_run_scope", capture_child_state)
    try:
        _, outcome = _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=scratch_root,
            scratch_lease=lambda: contextlib.nullcontext(scratch_root),
        )
    finally:
        child_write_gate.set()
        _join_fleet_threads()

    assert normalization_failed.is_set(), "the scratch-root failure did not fire"
    assert outcome.status == "done"
    assert child_settled.is_set(), "child did not fully settle before parent E"
    assert target.read_text() == sentinel
    child_runs = [
        row for row in db.list_runs("conv-1") if row["agent_kind"] == "subagent"
    ]
    assert len(child_runs) == 1, child_runs
    assert child_runs[0]["status"] == "done"
    assert len(child_states) == 1
    assert child_states[0].touched_paths == {target.name}


def test_a_survivors_write_after_its_turn_lands_in_a_change_record(
    tmp_path, root, tracker
):
    """A child released after its turn returned writes to disk. That write
    must be reviewable SOMEWHERE -- today it is in no record at all.
    """
    gate = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[[_spawn_fence("long job")], ["turn 1 final"]],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text(
            "written after the turn returned\n"
        ),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_id, outcome = _run(bridge, session, aid, root)
        assert outcome.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        # Nothing has changed on disk yet: the turn's own record is empty,
        # which is correct and is the state the survivor writes into.
        assert db.change_snapshots_for_run(run_id) == []
    finally:
        gate.set()
    _join_fleet_threads()

    assert (root / "survivor.txt").exists(), "precondition: the child wrote"
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1, f"the survivor's write landed in NO change record: {rows}"
    row = rows[0]
    assert row["files_changed"] == 1, row
    changed = tracker.service.repo_for_root(root).changed_files(
        row["baseline_sha"], row["end_sha"]
    )
    assert [c.path for c in changed] == ["survivor.txt"], changed
    marker = [
        m
        for m in _tool_rows(store, session)
        if m.content.startswith("✎") and "sub-agent" in m.content
    ]
    assert len(marker) == 1, (
        "nothing in the transcript says a sub-agent changed files after "
        f"the turn: {[m.content for m in _tool_rows(store, session)]}"
    )
    assert marker[0].change_review_run_id == run_id


def test_a_survivors_write_during_the_next_turn_is_disclosed_on_it(
    tmp_path, root, tracker
):
    """The survivor writes strictly AFTER turn 2's baseline settled, so the
    write is inside turn 2's diff -- a record attributing one turn's file
    writes to another. The tracker is a working-tree differ and cannot
    un-mix concurrent writers, so the record must SAY so.
    """
    gate = threading.Event()

    def release_and_join():
        gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],  # turn 1, call 1
            ["turn 1 final"],  # turn 1, call 2
            [_calc_fence()],  # turn 2, call 1 -- a tool, so B2 is awaited
            ["turn 2 final"],  # turn 2, call 2 -- fires the release
        ],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text(
            "written during turn 2\n"
        ),
        parent_side_effect=release_and_join,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, outcome_1 = _run(bridge, session, aid, root)
        assert outcome_1.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        run_2, outcome_2 = _run(bridge, session, _next_turn(store, session), root)
        assert outcome_2.status == "done"
    finally:
        gate.set()
    _join_fleet_threads()

    # Characterisation: the survivor's file IS inside turn 2's diff. That
    # is not fixable (one tree, two writers) -- what follows is.
    rows_2 = db.change_snapshots_for_run(run_2)
    assert len(rows_2) == 1, rows_2
    changed = tracker.service.repo_for_root(root).changed_files(
        rows_2[0]["baseline_sha"], rows_2[0]["end_sha"]
    )
    assert [c.path for c in changed] == ["survivor.txt"], changed

    assert rows_2[0]["kind"] == "turn_concurrent_subagent", (
        "turn 2's record does not record that an earlier turn's sub-agent "
        "was writing during it"
    )
    disclosures = [
        m
        for m in _tool_rows(store, session)
        if "earlier turn" in m.content and "sub-agent" in m.content
    ]
    assert len(disclosures) == 1, (
        "turn 2's changes silently include a sub-agent's: "
        f"{[m.content for m in _tool_rows(store, session)]}"
    )


def test_a_survivors_write_racing_the_next_baseline_is_still_reviewable(tmp_path, root):
    """The audit's second half: a survivor's tool dispatch passes turn 1's
    ALREADY-SATISFIED `await_baseline()`, so it is gated on nothing while
    turn 2's baseline is being taken -- and a write during that window is
    swallowed into B2 and vanishes from turn 2's diff.

    The baseline is made slow so the window is deterministic rather than a
    coin flip, exactly as `test_baseline_completes_before_the_first_tool_
    executes` does.
    """
    events: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                if "baseline" in message:
                    time.sleep(0.6)
                    events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    gate = threading.Event()

    def release_into_the_baseline_window():
        gate.set()
        _join_fleet_threads()

    def child_writes():
        # Recorded HERE, at the write itself -- an event appended after
        # joining the child's thread would time the JOIN instead, and the
        # join now waits for the very baseline this test is ordering
        # against.
        (root / "raced.txt").write_text(
            "written while turn 2's baseline was still snapshotting\n"
        )
        events.append("survivor-wrote")

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],  # turn 1, call 1
            ["turn 1 final"],  # turn 1, call 2
            ["turn 2 final"],  # turn 2, call 1 -- fires while B2 runs
        ],
        gate=gate,
        child_side_effect=child_writes,
        parent_side_effect=release_into_the_baseline_window,
        parent_side_effect_on_call=3,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, outcome_1 = _run(bridge, session, aid, root)
        assert outcome_1.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        events.clear()  # drop turn 1's baseline; only turn 2's matters now
        run_2, outcome_2 = _run(bridge, session, _next_turn(store, session), root)
        assert outcome_2.status == "done"
    finally:
        gate.set()
    _join_fleet_threads()

    assert "survivor-wrote" in events and "baseline-finished" in events, events
    assert events.index("survivor-wrote") < events.index("baseline-finished"), (
        "the survivor did NOT write inside turn 2's baseline window, so "
        f"this test proves nothing: {events}"
    )
    # Turn 2's own diff cannot see it -- B2 swallowed it.
    assert db.change_snapshots_for_run(run_2) == [], (
        "expected the raced write to be inside turn 2's baseline"
    )
    rows_1 = db.change_snapshots_for_run(run_1)
    assert len(rows_1) == 1, (
        f"a write that raced the next turn's baseline is in NO record: {rows_1}"
    )
    changed = tracker.service.repo_for_root(root).changed_files(
        rows_1[0]["baseline_sha"], rows_1[0]["end_sha"]
    )
    assert [c.path for c in changed] == ["raced.txt"], changed


def test_a_survivors_tool_dispatch_is_gated_on_nothing_across_turns(tmp_path, root):
    """CHARACTERISATION (green before and after this task's fix): the
    mechanism behind the test above.

    A survivor's tool batch still calls TURN 1's `before_tool_dispatch`
    gate, whose baseline was satisfied before turn 1 even answered. So the
    survivor dispatches a tool while turn 2's baseline is still being taken
    -- the exact "a tool writing before B settles races its own change into
    the baseline" hazard the gate exists to prevent, now reachable across
    turns and gated by nothing.

    This is NOT fixed by re-gating (a survivor must not block on an
    unrelated turn's snapshot); it is made harmless by the windows sharing
    a boundary sha, which the test above pins.
    """
    events: list[str] = []
    child_run_ids: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                if "baseline" in message:
                    time.sleep(0.6)
                    events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    gate = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            ["turn 2 final"],
        ],
        gate=gate,
        # The child calls a REAL tool once released, so its batch goes
        # through the review hook (i.e. through turn 1's baseline gate).
        child_scripts=[[_calc_fence()], ["child answer"]],
        parent_side_effect=lambda: (gate.set(), _join_fleet_threads()),
        parent_side_effect_on_call=3,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    def review(calls, run_id):
        events.append(f"review:{run_id}")
        return {}

    try:
        run_1, outcome_1 = _run(bridge, session, aid, root, review_tool_calls=review)
        assert outcome_1.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        child_run_ids.extend(
            row["id"]
            for row in db.list_runs("conv-1")
            if row["agent_kind"] == "subagent"
        )
        assert child_run_ids, "no sub-agent run row"
        events.clear()
        _run(
            bridge,
            session,
            _next_turn(store, session),
            root,
            review_tool_calls=review,
        )
    finally:
        gate.set()
    _join_fleet_threads()

    child_reviews = [
        index
        for index, event in enumerate(events)
        if event == f"review:{child_run_ids[0]}"
    ]
    assert child_reviews, f"the survivor never dispatched a tool after turn 1: {events}"
    baseline_done = events.index("baseline-finished")
    assert child_reviews[0] < baseline_done, (
        "the survivor's tool batch waited for turn 2's baseline (it does "
        f"not, and this test exists to say so out loud): {events}"
    )


def test_the_survivor_window_ends_exactly_where_the_next_turn_begins(
    tmp_path, root, tracker
):
    """The load-bearing invariant of the fix: the two windows ABUT.

    A survivor's window is closed at the NEXT turn's baseline sha rather
    than at a snapshot of its own, so the disk history is partitioned
    (B1..E1, E1..B2, B2..E2) with no crack for a write to fall into and no
    overlap for one to be counted twice in.
    """
    gate = threading.Event()
    keep_running = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            [_calc_fence()],
            ["turn 2 final"],
        ],
        gate=gate,
        # Write, then keep working: the window must be closed by turn 2,
        # not by the child finishing.
        child_scripts=[[_calc_fence()], ["child answer"]],
        second_gate=keep_running,
        child_side_effect=lambda: (root / "survivor.txt").write_text("a\n"),
        # Turn 2 writes its OWN file, after its baseline settled (its
        # first call was a tool, so the gate has been awaited) -- without
        # a change of its own turn 2 records no row to abut.
        parent_side_effect=lambda: (root / "by_turn_2.txt").write_text("b\n"),
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, _ = _run(bridge, session, aid, root)
        assert gateway.child_started.wait(5), "the child never started"
        gate.set()
        deadline = time.monotonic() + 5
        while not (root / "survivor.txt").exists():
            assert time.monotonic() < deadline, "the survivor never wrote"
            time.sleep(0.02)
        run_2, _ = _run(bridge, session, _next_turn(store, session), root)
        # Recorded by the END of turn 2, while the child is still working:
        # the window's content is settled the moment turn 2's baseline
        # exists, so waiting for a survivor that may run for an hour
        # before showing the user anything would be a choice, not a
        # necessity.
        post_turn = [
            r
            for r in db.change_snapshots_for_run(run_1)
            if r["kind"] == "subagent_post_turn"
        ]
        assert post_turn, (
            "the survivor's window was still open after the next turn "
            "ended, so its record waits on a child that need never finish"
        )
    finally:
        gate.set()
        keep_running.set()
    _join_fleet_threads()

    post_turn = [
        r
        for r in db.change_snapshots_for_run(run_1)
        if r["kind"] == "subagent_post_turn"
    ]
    assert len(post_turn) == 1, db.change_snapshots_for_run(run_1)
    assert post_turn[0]["files_changed"] == 1, post_turn
    turn_2_rows = [
        r
        for r in db.change_snapshots_for_conversation("conv-1")
        if r["run_id"] == run_2
    ]
    assert turn_2_rows, "turn 2 recorded no row to abut"
    assert post_turn[0]["end_sha"] == turn_2_rows[0]["baseline_sha"], (
        "the survivor's window and turn 2's window do not share a "
        "boundary, so a write between them belongs to neither"
    )
    # ... and therefore the write is in exactly ONE record.
    changed_2 = tracker.service.repo_for_root(root).changed_files(
        turn_2_rows[0]["baseline_sha"], turn_2_rows[0]["end_sha"]
    )
    assert "survivor.txt" not in [c.path for c in changed_2], (
        "the same write is counted in both windows"
    )
    # The survivor's row must also be emitted in the position resume will
    # re-derive it in -- BEFORE the next turn's own row, since it belongs
    # to the earlier turn's block. (Closing the window as a side effect of
    # opening the next one produces the same record in the wrong place.)
    live = [m.content for m in _tool_rows(store, session) if m.content.startswith("✎")]
    fresh = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    resumed = [
        m.content
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if m.content.startswith("✎")
    ]
    assert resumed == live, (
        f"live and resumed transcripts disagree: {live} vs {resumed}"
    )


def test_a_child_that_finishes_inside_its_turn_opens_no_survivor_window(
    tmp_path, root, tracker
):
    """Negative control: the post-turn window exists for survivors, not
    for every fleet turn. A child collected before the turn answers is
    fully inside the turn's own window and must add nothing."""
    gate = threading.Event()
    gate.set()  # the child never blocks
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("quick job")],
            [_calc_fence()],
            ["turn 1 final"],
        ],
        gate=gate,
        child_side_effect=lambda: (root / "by_the_child.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, outcome = _run(bridge, session, aid, root)
    _join_fleet_threads()
    assert outcome.status == "done"

    rows = db.change_snapshots_for_run(run_id)
    assert [r["kind"] for r in rows] == ["turn"], rows
    assert not [
        m for m in _tool_rows(store, session) if "after this turn" in m.content
    ], "a survivor row was emitted for a child that never survived"


def test_a_turn_without_a_foreign_survivor_is_not_stamped_concurrent(
    tmp_path, root, tracker
):
    """Negative control for the disclosure: a turn whose own child is the
    only sub-agent must NOT claim someone else's writes may be in it."""
    gate = threading.Event()
    gate.set()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("quick job")],
            [_calc_fence()],
            ["turn 1 final"],
        ],
        gate=gate,
        child_side_effect=lambda: (root / "by_the_child.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _ = _run(bridge, session, aid, root)
    _join_fleet_threads()

    assert [r["kind"] for r in db.change_snapshots_for_run(run_id)] == ["turn"]
    assert not [m for m in _tool_rows(store, session) if "earlier turn" in m.content]


def test_resume_re_derives_the_survivor_rows_byte_identical(tmp_path, root, tracker):
    """Both new transcript rows are re-derived from the stored `kind`.

    Without the column a post-turn row and a turn row are indistinguishable
    and resume would collapse them into ONE summary showing a turn that
    never happened -- the same parity rule TASK-1972 set for the summary
    and failure rows.
    """
    gate = threading.Event()

    def release_and_join():
        gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            [_calc_fence()],
            ["turn 2 final"],
        ],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text("a\n"),
        parent_side_effect=release_and_join,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, _ = _run(bridge, session, aid, root)
        assert gateway.child_started.wait(5), "the child never started"
        # Turn 2 both closes turn 1's window (at its baseline) AND absorbs
        # the survivor's write, so this run exercises both new rows.
        (root / "by_turn_1.txt").write_text("edited before turn 2\n")
        run_2, _ = _run(bridge, session, _next_turn(store, session), root)
    finally:
        gate.set()
    _join_fleet_threads()

    live = [
        m.content
        for m in _tool_rows(store, session)
        if m.content.startswith("✎") or "earlier turn" in m.content
    ]
    assert len(live) >= 2, live

    fresh = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    resumed = [
        m.content
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if m.content.startswith("✎") or "earlier turn" in m.content
    ]
    assert resumed == live, "the survivor rows did not survive resume byte-identical"


def test_opening_a_window_whose_last_child_already_left_closes_it_at_once(
    tmp_path, root, tracker
):
    """The sliver the open path exists to cover, probed directly.

    A child can finish between the turn's E snapshot and the moment the
    window is installed. Its final writes are then after E and before the
    window exists, and nothing else will ever close that window -- the
    last-child signal has already fired. Opening therefore re-checks the
    live count and closes immediately.

    Driven through the bridge's own methods rather than a scenario,
    because the race is microseconds wide and no scripted run can land in
    it reliably (see PR3a-1 Task 6b's P4: a branch no test can kill should
    be probed directly, not shipped untested).
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=None,
        change_tracker=tracker,
    )
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    tracker.end_turn(handle)  # the turn's own E; nothing changed
    (root / "written_in_the_sliver.txt").write_text("x\n")

    bridge._open_post_turn_change_window(
        "conv-1", run_id=run_id, session_id=session.id, handle=handle
    )

    assert bridge._post_turn_change_windows.get("conv-1") is None, (
        "a window nobody will ever close was left open"
    )
    rows = db.change_snapshots_for_run(run_id)
    assert [r["kind"] for r in rows] == ["subagent_post_turn"], rows
    assert rows[0]["files_changed"] == 1, rows


def test_shared_pending_scope_count_closes_only_after_each_child_enters(
    tmp_path, monkeypatch
):
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="t"),
        store=None,
        provider_gateway=None,
    )
    state = _ChildChangeState(owner_key="owner", pending_scopes=2)
    window = _PostTurnChangeWindow(
        run_id="parent-run",
        session_id="session",
        handle=object(),
        child_states=(state,),
    )
    bridge._post_turn_change_windows["conv-1"] = window
    closes: list[str] = []

    def record_close(conversation_id: str) -> None:
        with bridge._change_window_lock:
            removed = bridge._post_turn_change_windows.pop(conversation_id, None)
        if removed is not None:
            closes.append(conversation_id)

    monkeypatch.setattr(bridge, "_close_post_turn_change_window", record_close)
    adapter = SimpleNamespace(child_lifeline=contextlib.nullcontext)

    with bridge._child_run_scope("conv-1", adapter, state):
        assert state.pending_scopes == 1
    assert bridge._post_turn_change_windows.get("conv-1") is window
    assert closes == []

    with bridge._child_run_scope("conv-1", adapter, state):
        assert state.pending_scopes == 0
    assert bridge._post_turn_change_windows.get("conv-1") is None
    assert closes == ["conv-1"]


def test_final_settle_cleanup_keeps_window_for_other_pending_state(
    tmp_path, monkeypatch
):
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="t"),
        store=None,
        provider_gateway=None,
    )
    settling = _ChildChangeState(owner_key="settling", pending_scopes=1)
    other = _ChildChangeState(owner_key="other", pending_scopes=1)
    window = _PostTurnChangeWindow(
        run_id="parent-run",
        session_id="session",
        handle=object(),
        child_states=(settling, other),
    )
    bridge._post_turn_change_windows["conv-1"] = window
    bridge._child_change_states["conv-1"] = {
        settling.owner_key: settling,
        other.owner_key: other,
    }
    closes: list[str] = []

    def record_close(conversation_id: str) -> None:
        with bridge._change_window_lock:
            removed = bridge._post_turn_change_windows.pop(conversation_id, None)
        if removed is not None:
            closes.append(conversation_id)

    monkeypatch.setattr(bridge, "_close_post_turn_change_window", record_close)

    with bridge._change_window_lock:
        settling.pending_scopes = 0
        bridge._child_change_states["conv-1"].pop(settling.owner_key)
    bridge._close_post_turn_change_window_if_idle("conv-1")

    assert bridge._post_turn_change_windows.get("conv-1") is window
    assert closes == []

    with bridge._change_window_lock:
        other.pending_scopes = 0
        bridge._child_change_states.pop("conv-1")
    bridge._close_post_turn_change_window_if_idle("conv-1")

    assert bridge._post_turn_change_windows.get("conv-1") is None
    assert closes == ["conv-1"]


def test_a_survivor_finishing_mid_turn_is_counted_in_exactly_one_window(
    tmp_path, root, tracker
):
    """The other half of the abutment rule: a survivor that finishes
    DURING the next turn must not have its write counted twice.

    Its window cannot end at a snapshot of its own once the next turn's
    baseline exists -- that would overlap the turn's window, and the same
    file would appear on two cards as if it had been written twice.
    """
    gate = threading.Event()

    def release_and_join():
        gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            [_calc_fence()],  # turn 2 awaits B2 here
            ["turn 2 final"],  # ... then the survivor is released
        ],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text("a\n"),
        parent_side_effect=release_and_join,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        _run(bridge, session, aid, root)
        assert gateway.child_started.wait(5), "the child never started"
        _run(bridge, session, _next_turn(store, session), root)
    finally:
        gate.set()
    _join_fleet_threads()

    holding = [
        row
        for row in db.change_snapshots_for_conversation("conv-1")
        if "survivor.txt"
        in [
            c.path
            for c in tracker.service.repo_for_root(root).changed_files(
                row["baseline_sha"], row["end_sha"]
            )
        ]
    ]
    assert len(holding) == 1, (
        "the survivor's single write is on "
        f"{len(holding)} change records: {[r['kind'] for r in holding]}"
    )


def test_successor_claim_uses_window_paths_after_live_state_cleanup(
    tmp_path, root, tracker, monkeypatch
):
    target = root / "claimed-window-only.txt"
    expected = b"present before successor B\n"
    (root / ".gitignore").write_text(f"{target.name}\n")

    gateway = _SideEffectGateway([["successor done"]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None

    target.write_bytes(expected)
    state = _ChildChangeState(
        owner_key="old-owner",
        touched_paths={str(target)},
    )
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        child_states=(state,),
    )
    bridge._post_turn_change_windows["conv-1"] = window
    bridge._child_change_states["conv-1"] = {state.owner_key: state}

    cleanup_ready = threading.Event()
    release_cleanup = threading.Event()
    cleanup_done = threading.Event()

    def clean_live_state() -> None:
        cleanup_ready.set()
        assert release_cleanup.wait(5), "cleanup barrier was never released"
        with bridge._change_window_lock:
            states = bridge._child_change_states.get("conv-1")
            assert states is not None
            states.pop(state.owner_key)
            bridge._child_change_states.pop("conv-1")
        cleanup_done.set()

    cleanup = threading.Thread(target=clean_live_state, name="state-cleanup")
    cleanup.start()
    assert cleanup_ready.wait(5), "cleanup thread never reached its barrier"
    release_cleanup.set()
    assert cleanup_done.wait(5), "cleanup thread did not remove live state"
    cleanup.join(5)
    assert not cleanup.is_alive()
    assert bridge._child_change_states == {}
    assert window.child_states == (state,)

    real_begin = tracker.begin_turn
    captured: dict[str, object] = {}

    def capture_begin(roots, touched_paths=()):
        captured["paths"] = tuple(touched_paths)
        handle = real_begin(roots, touched_paths=touched_paths)
        captured["handle"] = handle
        return handle

    monkeypatch.setattr(tracker, "begin_turn", capture_begin)
    _, outcome = _run(bridge, session, aid, root)

    assert outcome.status == "done"
    assert str(target) in captured["paths"]
    successor = captured["handle"]
    successor.await_baseline()
    baseline = successor.baselines[str(root.resolve())]
    repo = tracker.service.repo_for_root(root)
    assert repo.file_bytes(baseline, target.name) == expected


def test_successor_b_waits_for_an_already_started_fresh_close(
    tmp_path, root, tracker, monkeypatch
):
    gateway = _SideEffectGateway([["successor done"]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None
    (root / "fresh-close.txt").write_text("closed before successor B\n")

    close_done = _WaitRecordingEvent()
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        close_done=close_done,
    )
    bridge._post_turn_change_windows["conv-1"] = window
    fresh_close_owned = threading.Event()
    release_close = threading.Event()
    successor_b_started = threading.Event()
    successor_handles: list[object] = []
    real_end = tracker.end_turn
    real_begin = tracker.begin_turn

    def block_fresh_close(handle, *args, **kwargs):
        if handle is window.handle:
            assert kwargs.get("end_shas") is None
            fresh_close_owned.set()
            assert release_close.wait(5), "fresh close was never released"
        return real_end(handle, *args, **kwargs)

    def record_successor_b(roots, touched_paths=()):
        successor_b_started.set()
        handle = real_begin(roots, touched_paths=touched_paths)
        successor_handles.append(handle)
        return handle

    monkeypatch.setattr(tracker, "end_turn", block_fresh_close)
    monkeypatch.setattr(tracker, "begin_turn", record_successor_b)
    errors: list[BaseException] = []
    results: list[object] = []

    def close_old_window() -> None:
        try:
            bridge._close_post_turn_change_window("conv-1")
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    def run_successor() -> None:
        try:
            results.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    closer = threading.Thread(target=close_old_window, name="fresh-close-owner")
    successor_thread = threading.Thread(target=run_successor, name="successor-b")
    closer.start()
    try:
        assert fresh_close_owned.wait(5), "fresh close never reached the tracker"
        successor_thread.start()
        assert close_done.wait_started.wait(5), (
            "successor did not wait on the already-owned fresh close"
        )
        assert not successor_b_started.is_set(), (
            "successor B started before the fresh close completed"
        )
    finally:
        release_close.set()
        closer.join(10)
        successor_thread.join(10)

    assert not closer.is_alive()
    assert not successor_thread.is_alive()
    assert errors == []
    assert results[0][1].status == "done"
    assert len(successor_handles) == 1
    successor = successor_handles[0]
    successor.await_baseline()
    old_rows = db.change_snapshots_for_run(old_run_id)
    assert len(old_rows) == 1, old_rows
    assert old_rows[0]["end_sha"] == successor.baselines[str(root.resolve())]


@pytest.mark.parametrize("close_failure", ["exception", "tracking_error"])
def test_successor_b_rejects_a_completed_but_failed_fresh_close(
    tmp_path, root, tracker, monkeypatch, close_failure
):
    gateway = _SideEffectGateway([["successor still replies"]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None

    close_done = _WaitRecordingEvent()
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        close_done=close_done,
    )
    bridge._post_turn_change_windows["conv-1"] = window
    real_begin = tracker.begin_turn
    real_end = tracker.end_turn
    close_owned = threading.Event()
    release_close = threading.Event()
    successor_b_called = threading.Event()
    successor_handles: list[object] = []

    def fail_fresh_close(handle, *args, **kwargs):
        if handle is not window.handle:
            return real_end(handle, *args, **kwargs)
        close_owned.set()
        assert release_close.wait(5), "failed close was never released"
        if close_failure == "exception":
            raise RuntimeError("injected fresh-close failure")
        return [
            TurnChangeRecord(
                root=str(root.resolve()),
                tracking_error="injected fresh-close tracking failure",
            )
        ]

    def record_successor_b(roots, touched_paths=()):
        successor_b_called.set()
        handle = real_begin(roots, touched_paths=touched_paths)
        successor_handles.append(handle)
        return handle

    monkeypatch.setattr(tracker, "end_turn", fail_fresh_close)
    monkeypatch.setattr(tracker, "begin_turn", record_successor_b)
    owner_results: list[bool] = []
    results: list[object] = []
    failures: list[BaseException] = []

    def close_old_window() -> None:
        try:
            owner_results.append(bridge._close_post_turn_change_window("conv-1"))
        except BaseException as exc:  # noqa: BLE001 -- asserted below
            failures.append(exc)

    def run_successor() -> None:
        try:
            results.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted below
            failures.append(exc)

    closer = threading.Thread(target=close_old_window, name="failed-fresh-owner")
    successor = threading.Thread(target=run_successor, name="failed-fresh-b")
    closer.start()
    try:
        assert close_owned.wait(5), "fresh close never reached its failure seam"
        successor.start()
        assert close_done.wait_started.wait(5), (
            "successor B never waited on the closing window"
        )
        assert not successor_b_called.is_set(), (
            "successor B started before close completion"
        )
    finally:
        release_close.set()
        closer.join(10)
        successor.join(10)

    assert not closer.is_alive()
    assert not successor.is_alive()
    for handle in successor_handles:
        handle.await_baseline()
    assert failures == []
    assert owner_results == [False]
    assert results[0][1].status == "done"
    assert close_done.is_set()
    assert not successor_b_called.is_set()
    assert successor_handles == []
    assert getattr(window, "close_succeeded", None) is False


def test_successor_b_freezes_paths_atomically_with_older_publication(
    tmp_path, root, tracker, monkeypatch
):
    target = root / "published-at-b-boundary.txt"
    (root / ".gitignore").write_text(f"{target.name}\n")
    gateway = _BlockingParentGateway()
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None
    state = _ChildChangeState(owner_key="older-owner")
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        child_states=(state,),
    )
    bridge._post_turn_change_windows["conv-1"] = window
    bridge._child_change_states["conv-1"] = {state.owner_key: state}

    class OwnerRecordingLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._meta_lock = threading.Lock()
            self._owner: str | None = None
            self.publisher_attempted = threading.Event()

        def acquire(self, blocking=True, timeout=-1):
            if threading.current_thread().name == "older-path-publisher":
                self.publisher_attempted.set()
            if timeout == -1:
                acquired = self._lock.acquire(blocking)
            else:
                acquired = self._lock.acquire(blocking, timeout)
            if acquired:
                with self._meta_lock:
                    self._owner = threading.current_thread().name
            return acquired

        def release(self) -> None:
            with self._meta_lock:
                self._owner = None
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.release()

        @property
        def owner(self) -> str | None:
            with self._meta_lock:
                return self._owner

    boundary_lock = OwnerRecordingLock()
    bridge._change_window_lock = boundary_lock
    real_begin = tracker.begin_turn
    begin_entered = threading.Event()
    release_begin = threading.Event()
    publication_done = threading.Event()
    order_lock = threading.Lock()
    order: list[str] = []
    captured: dict[str, object] = {}

    def wedge_successor_b(roots, touched_paths=()):
        captured["paths"] = tuple(touched_paths)
        begin_entered.set()
        assert release_begin.wait(5), "successor B was never released"
        handle = real_begin(roots, touched_paths=touched_paths)
        captured["handle"] = handle
        with order_lock:
            order.append("begin")
        return handle

    def publish_older_path() -> None:
        with bridge._change_window_lock:
            with order_lock:
                order.append("publish")
            target.write_text("older write at successor boundary\n")
            state.touched_paths.add(str(target))
        publication_done.set()

    monkeypatch.setattr(tracker, "begin_turn", wedge_successor_b)
    results: list[object] = []
    failures: list[BaseException] = []

    def run_successor() -> None:
        try:
            results.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted below
            failures.append(exc)

    successor = threading.Thread(target=run_successor, name="successor-freeze")
    publisher = threading.Thread(
        target=publish_older_path,
        name="older-path-publisher",
    )
    successor.start()
    assert begin_entered.wait(5), "successor never reached B"
    publisher.start()
    try:
        assert boundary_lock.publisher_attempted.wait(5), (
            "older publisher never attempted the boundary lock"
        )
        owner_during_begin = boundary_lock.owner
        publication_completed_before_kick = publication_done.is_set()
        if owner_during_begin != "successor-freeze":
            assert publication_done.wait(5), (
                "pre-B publisher did not complete across the open gap"
            )
            publication_completed_before_kick = True
    finally:
        release_begin.set()
        assert publication_done.wait(5), "older publication never completed"
        assert gateway.entered.wait(5), "successor never entered its provider"
        gateway.release.set()
        publisher.join(10)
        successor.join(10)

    assert not publisher.is_alive()
    assert not successor.is_alive()
    assert failures == []
    run_id, outcome = results[0]
    assert outcome.status == "done"
    assert owner_during_begin == "successor-freeze"
    assert not publication_completed_before_kick
    assert order == ["begin", "publish"]
    assert str(target) not in captured["paths"]
    successor_handle = captured["handle"]
    successor_handle.await_baseline()
    repo = tracker.service.repo_for_root(root)
    baseline = successor_handle.baselines[str(root.resolve())]
    assert repo.file_bytes(baseline, target.name) is None
    old_rows = db.change_snapshots_for_run(old_run_id)
    assert not any(
        target.name
        in [
            item.path
            for item in repo.changed_files(row["baseline_sha"], row["end_sha"])
        ]
        for row in old_rows
        if row["baseline_sha"] and row["end_sha"]
    )
    successor_rows = db.change_snapshots_for_run(run_id)
    assert (
        sum(
            target.name
            in [
                item.path
                for item in repo.changed_files(row["baseline_sha"], row["end_sha"])
            ]
            for row in successor_rows
            if row["baseline_sha"] and row["end_sha"]
        )
        == 1
    )
    assert window.close_done.is_set()


def test_inherited_child_write_waits_for_claimed_successor_baseline(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Tools.file_operation_tools import WriteFileTool

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )

    target = root / "inherited-after-successor-b.txt"
    sentinel = "written by the inherited child after successor B\n"
    child_release = threading.Event()
    child_joined = threading.Event()

    def join_child_before_successor_final() -> None:
        _join_fleet_threads()
        child_joined.set()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("write after the next turn starts")],
            ["first turn done"],
            [_calc_fence()],
            ["successor done"],
        ],
        gate=child_release,
        child_scripts=[
            [_write_fence(Path(target.name), sentinel)],
            ["child done"],
        ],
        parent_side_effect=join_child_before_successor_final,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_1, outcome_1 = _run(
        bridge,
        session,
        aid,
        root,
        builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
        scratch_root=root,
        scratch_lease=lambda: contextlib.nullcontext(root),
    )
    assert outcome_1.status == "done"
    assert gateway.child_started.wait(5), "inherited child never started"
    assert not target.exists(), "inherited child wrote before its turn returned"
    with bridge._change_window_lock:
        old_window = bridge._post_turn_change_windows.get("conv-1")
    assert old_window is not None

    successor_before_add = threading.Event()
    release_successor_add = threading.Event()
    successor_handle_ready = threading.Event()
    successor_handle: dict[str, object] = {}
    real_repo_for_root = tracker.service.repo_for_root
    real_begin = tracker.begin_turn

    def block_successor_baseline_add(requested_root):
        repo = real_repo_for_root(requested_root)
        real_run = repo._run

        def run_git(*args, **kwargs):
            if (
                threading.current_thread().name == "change-review-baseline"
                and args[:4] == ("add", "-A", "--", ".")
                and not successor_before_add.is_set()
            ):
                successor_before_add.set()
                assert release_successor_add.wait(5), (
                    "successor baseline add was never released"
                )
            return real_run(*args, **kwargs)

        repo._run = run_git
        return repo

    def capture_successor_handle(roots, touched_paths=()):
        handle = real_begin(roots, touched_paths=touched_paths)
        successor_handle["value"] = handle
        successor_handle_ready.set()
        return handle

    monkeypatch.setattr(
        tracker.service,
        "repo_for_root",
        block_successor_baseline_add,
    )
    monkeypatch.setattr(tracker, "begin_turn", capture_successor_handle)

    write_finished = threading.Event()
    child_claim_wait_started = threading.Event()
    child_reached_boundary = threading.Event()
    real_write_execute = WriteFileTool.execute

    async def observe_real_write(tool, **kwargs):
        result = await real_write_execute(tool, **kwargs)
        if Path(str(kwargs.get("file_path", ""))).name == target.name:
            write_finished.set()
            child_reached_boundary.set()
        return result

    monkeypatch.setattr(WriteFileTool, "execute", observe_real_write)
    results: list[object] = []
    failures: list[BaseException] = []

    def run_successor() -> None:
        try:
            results.append(
                _run(
                    bridge,
                    session,
                    _next_turn(store, session),
                    root,
                    builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
                    scratch_root=root,
                    scratch_lease=lambda: contextlib.nullcontext(root),
                )
            )
        except BaseException as exc:  # noqa: BLE001 -- asserted below
            failures.append(exc)

    successor = threading.Thread(target=run_successor, name="claimed-successor")
    successor.start()
    try:
        assert successor_before_add.wait(5), (
            "successor B never reached its pre-add barrier"
        )
        assert successor_handle_ready.wait(5), "successor handle was not published"
        handle = successor_handle["value"]
        real_await_baseline = handle.await_baseline

        def observe_child_claim_wait(*args, **kwargs):
            if threading.current_thread().name.startswith("fleet-"):
                child_claim_wait_started.set()
                child_reached_boundary.set()
            return real_await_baseline(*args, **kwargs)

        handle.await_baseline = observe_child_claim_wait
        with bridge._change_window_lock:
            claim = old_window.successor_claim
        assert claim is not None
        assert claim.ready.wait(5), "successor claim attachment did not publish"
        with bridge._change_window_lock:
            assert claim.handle is handle
            assert not claim.failed

        child_release.set()
        assert child_reached_boundary.wait(5), (
            "inherited child reached neither its baseline gate nor its WRITE"
        )
        wrote_before_b_release = write_finished.is_set()
        waited_before_write = child_claim_wait_started.is_set()
        target_existed_before_b_release = target.exists()
    finally:
        child_release.set()
        release_successor_add.set()
        successor.join(10)
        _join_fleet_threads()

    assert not successor.is_alive()
    assert failures == []
    run_2, outcome_2 = results[0]
    assert outcome_2.status == "done"
    assert child_joined.is_set(), "successor E raced the inherited child"
    assert write_finished.is_set(), "the real WRITE tool never completed"
    handle.await_baseline()
    repo = tracker.service.repo_for_root(root)
    baseline = handle.baselines[str(root.resolve())]
    assert repo.file_bytes(baseline, target.name) is None, (
        "the inherited WRITE was absorbed into successor B"
    )
    assert waited_before_write
    assert not wrote_before_b_release
    assert not target_existed_before_b_release
    assert target.read_text() == sentinel

    def changed_paths(row):
        if not row["baseline_sha"] or not row["end_sha"]:
            return []
        return [
            item.path
            for item in repo.changed_files(row["baseline_sha"], row["end_sha"])
        ]

    assert not any(
        target.name in changed_paths(row) for row in db.change_snapshots_for_run(run_1)
    )
    successor_rows = [
        row
        for row in db.change_snapshots_for_run(run_2)
        if target.name in changed_paths(row)
    ]
    assert len(successor_rows) == 1, db.change_snapshots_for_run(run_2)
    assert successor_rows[0]["kind"] == "turn_concurrent_subagent"
    assert any(
        "earlier turn" in message.content and "sub-agent" in message.content
        for message in _tool_rows(store, session)
    )
    assert old_window.close_done.is_set()


def test_successor_e_waits_for_close_time_force_path_handoff(
    tmp_path, root, tracker, monkeypatch
):
    target = root / "late-close-handoff.txt"
    expected = b"created after successor B\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    gateway = _BlockingParentGateway()
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None

    state = _ChildChangeState(
        owner_key="old-owner",
        touched_paths={str(target)},
    )
    close_done = _WaitRecordingEvent()
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        child_states=(state,),
        close_done=close_done,
    )
    bridge._post_turn_change_windows["conv-1"] = window

    real_begin = tracker.begin_turn
    real_end = tracker.end_turn
    successor_b_ready = threading.Event()
    close_owned = threading.Event()
    release_close = threading.Event()
    successor_e_started = threading.Event()
    successor_handle: dict[str, object] = {}

    def capture_successor_b(roots, touched_paths=()):
        handle = real_begin(roots, touched_paths=touched_paths)
        successor_handle["value"] = handle
        successor_b_ready.set()
        return handle

    def block_supplied_close(handle, *args, **kwargs):
        if handle is window.handle:
            assert kwargs.get("end_shas") is not None
            close_owned.set()
            assert release_close.wait(5), "supplied-SHA close was never released"
        elif handle is successor_handle.get("value"):
            successor_e_started.set()
        return real_end(handle, *args, **kwargs)

    monkeypatch.setattr(tracker, "begin_turn", capture_successor_b)
    monkeypatch.setattr(tracker, "end_turn", block_supplied_close)
    errors: list[BaseException] = []
    result: list[object] = []

    def run_successor() -> None:
        try:
            result.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    def close_old_window() -> None:
        try:
            bridge._close_post_turn_change_window("conv-1")
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    successor_thread = threading.Thread(target=run_successor, name="successor-e")
    closer = threading.Thread(target=close_old_window, name="supplied-close-owner")
    successor_thread.start()
    try:
        assert successor_b_ready.wait(5), "successor B was never started"
        successor = successor_handle["value"]
        successor.await_baseline()
        assert gateway.entered.wait(5), "successor never entered its provider"
        baseline = successor.baselines[str(root.resolve())]
        repo = tracker.service.repo_for_root(root)
        assert repo.file_bytes(baseline, target.name) is None
        target.write_bytes(expected)

        closer.start()
        assert close_owned.wait(5), "child closer never owned supplied-SHA close"
        gateway.release.set()
        assert close_done.wait_started.wait(5), (
            "successor E did not wait for close-time force-path handoff"
        )
        assert not successor_e_started.is_set(), (
            "successor E overtook close-time force-path handoff"
        )
    finally:
        release_close.set()
        gateway.release.set()
        closer.join(10)
        successor_thread.join(10)

    assert not closer.is_alive()
    assert not successor_thread.is_alive()
    assert errors == []
    run_id, outcome = result[0]
    assert outcome.status == "done"
    successor_rows = db.change_snapshots_for_run(run_id)
    rows_with_target = [
        row
        for row in successor_rows
        if target.name
        in [
            item.path
            for item in tracker.service.repo_for_root(root).changed_files(
                row["baseline_sha"], row["end_sha"]
            )
        ]
    ]
    assert len(rows_with_target) == 1, successor_rows
    assert rows_with_target[0]["baseline_sha"] == baseline
    assert repo.file_bytes(rows_with_target[0]["end_sha"], target.name) == expected
    assert not db.change_snapshots_for_run(old_run_id)


def test_successor_e_waiter_timeout_disables_turn_tracking(
    tmp_path, root, tracker, monkeypatch
):
    monkeypatch.setattr(
        console_agent_bridge_module,
        "_CHANGE_BOUNDARY_WAIT_SECONDS",
        0.01,
    )
    target = root / "timeout-before-handoff.txt"
    expected = b"created after successor B\n"
    (root / ".gitignore").write_text(f"{target.name}\n")
    gateway = _BlockingParentGateway()
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None

    state = _ChildChangeState(
        owner_key="old-owner",
        touched_paths={str(target)},
    )
    close_done = _WaitRecordingEvent()
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        child_states=(state,),
        close_done=close_done,
    )
    bridge._post_turn_change_windows["conv-1"] = window

    real_begin = tracker.begin_turn
    real_end = tracker.end_turn
    successor_handle: dict[str, object] = {}
    close_owned = threading.Event()
    release_close = threading.Event()
    successor_e_called = threading.Event()

    def capture_successor_b(roots, touched_paths=()):
        handle = real_begin(roots, touched_paths=touched_paths)
        successor_handle["value"] = handle
        return handle

    def block_close_before_handoff(handle, *args, **kwargs):
        if handle is window.handle:
            assert kwargs.get("end_shas") is not None
            close_owned.set()
            assert release_close.wait(5), "first-owner close was never released"
        elif handle is successor_handle.get("value"):
            successor_e_called.set()
        return real_end(handle, *args, **kwargs)

    monkeypatch.setattr(tracker, "begin_turn", capture_successor_b)
    monkeypatch.setattr(tracker, "end_turn", block_close_before_handoff)
    errors: list[BaseException] = []
    result: list[object] = []

    def run_successor() -> None:
        try:
            result.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    def close_old_window() -> None:
        try:
            bridge._close_post_turn_change_window("conv-1")
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    successor_thread = threading.Thread(
        target=run_successor,
        name="successor-e-timeout",
    )
    closer = threading.Thread(
        target=close_old_window,
        name="blocked-close-owner",
    )
    successor_thread.start()
    try:
        assert gateway.entered.wait(5), "successor never entered its provider"
        successor = successor_handle["value"]
        successor.await_baseline()
        target.write_bytes(expected)

        closer.start()
        assert close_owned.wait(5), "first-owner close never reached handoff"
        gateway.release.set()
        assert close_done.wait_started.wait(5), (
            "successor E never waited on the first-owner close"
        )
        successor_thread.join(5)
        assert not successor_thread.is_alive(), (
            "successor reply did not survive the boundary timeout"
        )
        assert not successor_e_called.is_set(), (
            "successor E tracking overtook unfinished close-time handoff"
        )
    finally:
        gateway.release.set()
        release_close.set()
        successor_thread.join(10)
        closer.join(10)

    assert not successor_thread.is_alive()
    assert not closer.is_alive()
    assert errors == []
    run_id, outcome = result[0]
    assert outcome.status == "done"
    assert not db.change_snapshots_for_run(run_id)
    assert close_done.is_set()


@pytest.mark.parametrize(
    "boundary_failure",
    ["missing", "error", "tracking_error", "exception"],
)
def test_untrusted_claimed_successor_boundary_fails_closed(
    tmp_path, root, tracker, monkeypatch, boundary_failure
):
    second_root = tmp_path / "second-root"
    second_root.mkdir()
    (second_root / "seed.txt").write_text("seed\n")
    gateway = _BlockingParentGateway()
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root, second_root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None

    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
    )
    bridge._post_turn_change_windows["conv-1"] = window
    baselines = dict(follow_on.baselines)
    errors: dict[str, str] = {}
    second_key = str(second_root.resolve())
    if boundary_failure == "missing":
        baselines.pop(second_key)
    elif boundary_failure == "error":
        errors[second_key] = "injected claimed baseline failure"
    baseline_awaited = threading.Event()
    claimed_handle = SimpleNamespace(
        roots=list(follow_on.roots),
        baselines=baselines,
        errors=errors,
        await_baseline=baseline_awaited.set,
    )
    monkeypatch.setattr(tracker, "begin_turn", lambda *args, **kwargs: claimed_handle)
    real_end = tracker.end_turn
    old_end_calls: list[dict[str, str] | None] = []
    successor_e_called = threading.Event()

    def record_boundary_end(handle, *args, **kwargs):
        if handle is window.handle:
            old_end_calls.append(kwargs.get("end_shas"))
            if boundary_failure == "tracking_error":
                return [
                    TurnChangeRecord(
                        root=str(root.resolve()),
                        tracking_error="injected close-time handoff failure",
                    )
                ]
            if boundary_failure == "exception":
                raise RuntimeError("injected close-time exception")
            return []
        if handle is claimed_handle:
            successor_e_called.set()
            return []
        return real_end(handle, *args, **kwargs)

    monkeypatch.setattr(tracker, "end_turn", record_boundary_end)
    results: list[object] = []
    failures: list[BaseException] = []
    close_results: list[bool] = []

    def run_successor() -> None:
        try:
            results.append(
                _run(
                    bridge,
                    session,
                    aid,
                    root,
                    change_roots=[root, second_root],
                )
            )
        except BaseException as exc:  # noqa: BLE001 -- asserted below
            failures.append(exc)

    successor = threading.Thread(target=run_successor, name="invalid-claim-run")
    successor.start()
    try:
        assert gateway.entered.wait(5), "successor never entered its provider"
        with bridge._change_window_lock:
            claim = window.successor_claim
        assert claim is not None
        assert claim.ready.is_set()
        closer = threading.Thread(
            target=lambda: close_results.append(
                bridge._close_post_turn_change_window("conv-1")
            ),
            name="invalid-claim-closer",
        )
        closer.start()
        closer.join(5)
        assert not closer.is_alive(), "invalid claim close did not finish"
    finally:
        gateway.release.set()
        successor.join(10)

    assert not successor.is_alive()
    assert failures == []
    assert results[0][1].status == "done"
    assert close_results == [False]
    assert baseline_awaited.is_set()
    if boundary_failure in {"missing", "error"}:
        assert old_end_calls == []
    else:
        assert old_end_calls == [baselines]
    assert not successor_e_called.is_set()
    with bridge._change_window_lock:
        assert claim.failed
    assert claim.ready.is_set()
    assert window.close_done.is_set()


def test_claim_timeout_failure_cannot_be_cleared_by_late_attachment(
    tmp_path, root, tracker, monkeypatch
):
    gateway = _BlockingParentGateway()
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None
    window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
    )
    bridge._post_turn_change_windows["conv-1"] = window

    class ForcedTimeoutEvent(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            self.wait_started = threading.Event()
            self.release_timeout = threading.Event()

        def wait(self, timeout=None):
            self.wait_started.set()
            assert self.release_timeout.wait(5), "forced timeout was not released"
            return False

    class InstrumentedClaim:
        def __init__(self) -> None:
            self.ready = ForcedTimeoutEvent()
            self.handle = None
            self._failed = False
            self.attachment_read = threading.Event()
            self.failure_written = threading.Event()
            self._instrumented = False

        @property
        def failed(self):
            observed = self._failed
            if (
                threading.current_thread().name == "late-claim-attachment"
                and not self._instrumented
            ):
                self._instrumented = True
                lock_was_free = bridge._change_window_lock.acquire(blocking=False)
                if lock_was_free:
                    bridge._change_window_lock.release()
                self.attachment_read.set()
                if lock_was_free:
                    assert self.failure_written.wait(5), (
                        "closer never published its failure"
                    )
            return observed

        @failed.setter
        def failed(self, value):
            self._failed = value
            if value:
                self.failure_written.set()

    monkeypatch.setattr(
        console_agent_bridge_module,
        "_SuccessorBoundaryClaim",
        InstrumentedClaim,
    )
    real_begin = tracker.begin_turn
    real_end = tracker.end_turn
    begin_returned = threading.Event()
    release_begin = threading.Event()
    successor_handle: dict[str, object] = {}
    successor_e_called = threading.Event()

    def block_after_begin(roots, touched_paths=()):
        handle = real_begin(roots, touched_paths=touched_paths)
        successor_handle["value"] = handle
        begin_returned.set()
        assert release_begin.wait(5), "successor begin was never released"
        return handle

    def record_successor_e(handle, *args, **kwargs):
        if handle is successor_handle.get("value"):
            successor_e_called.set()
        return real_end(handle, *args, **kwargs)

    monkeypatch.setattr(tracker, "begin_turn", block_after_begin)
    monkeypatch.setattr(tracker, "end_turn", record_successor_e)
    results: list[object] = []
    failures: list[BaseException] = []
    close_results: list[bool] = []

    def run_successor() -> None:
        try:
            results.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted below
            failures.append(exc)

    successor = threading.Thread(
        target=run_successor,
        name="late-claim-attachment",
    )
    successor.start()
    assert begin_returned.wait(5), "successor begin did not return"
    with bridge._change_window_lock:
        claim = window.successor_claim
    assert isinstance(claim, InstrumentedClaim)
    closer = threading.Thread(
        target=lambda: close_results.append(
            bridge._close_post_turn_change_window("conv-1")
        ),
        name="claim-timeout-closer",
    )
    closer.start()
    try:
        assert claim.ready.wait_started.wait(5), "closer never waited on claim"
        release_begin.set()
        assert claim.attachment_read.wait(5), "attachment never read claim state"
        claim.ready.release_timeout.set()
        closer.join(5)
        assert not closer.is_alive(), "claim timeout closer did not finish"
    finally:
        release_begin.set()
        gateway.release.set()
        successor.join(10)
        closer.join(10)

    assert not successor.is_alive()
    assert not closer.is_alive()
    successor_handle["value"].await_baseline()
    assert failures == []
    assert results[0][1].status == "done"
    assert close_results == [False]
    with bridge._change_window_lock:
        assert claim.failed
    assert claim.failure_written.is_set()
    assert claim.ready.is_set()
    assert window.close_done.is_set()
    assert not successor_e_called.is_set()


def test_claim_and_close_failures_release_waiters_without_breaking_runs(
    tmp_path, root, tracker, monkeypatch
):
    monkeypatch.setattr(
        console_agent_bridge_module,
        "_CHANGE_BOUNDARY_WAIT_SECONDS",
        1.0,
    )
    gateway = _SideEffectGateway([["successor still replies"]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    real_begin = tracker.begin_turn
    real_end = tracker.end_turn

    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = real_begin([root])
    old_turn.await_baseline()
    real_end(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None
    claim_close_done = _WaitRecordingEvent()
    claim_window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        close_done=claim_close_done,
    )
    bridge._post_turn_change_windows["conv-1"] = claim_window

    begin_entered = threading.Event()
    release_begin_failure = threading.Event()

    def fail_successor_begin(roots, touched_paths=()):
        begin_entered.set()
        assert release_begin_failure.wait(5), "claim failure was never released"
        raise RuntimeError("injected successor claim attachment failure")

    monkeypatch.setattr(tracker, "begin_turn", fail_successor_begin)
    errors: list[BaseException] = []
    results: list[object] = []

    def run_successor() -> None:
        try:
            results.append(_run(bridge, session, aid, root))
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    def close_claimed_window() -> None:
        claim_close_started.set()
        try:
            claim_close_results.append(bridge._close_post_turn_change_window("conv-1"))
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)
        finally:
            claim_close_finished.set()

    successor_thread = threading.Thread(target=run_successor, name="failed-claim")
    claim_closer = threading.Thread(target=close_claimed_window, name="claim-waiter")
    claim_close_started = threading.Event()
    claim_close_finished = threading.Event()
    claim_close_results: list[bool] = []
    successor_thread.start()
    try:
        assert begin_entered.wait(5), "successor never reached injected failure"
        claim = claim_window.successor_claim
        assert claim is not None, "successor did not install its pre-B claim"
        claim_ready = _WaitRecordingEvent()
        claim.ready = claim_ready
        claim_closer.start()
        assert claim_close_started.wait(5), "window closer never started"
        assert not claim_close_finished.is_set(), (
            "window closer crossed the atomic claim attachment"
        )
    finally:
        release_begin_failure.set()
        successor_thread.join(10)
        claim_closer.join(10)

    assert not successor_thread.is_alive()
    assert not claim_closer.is_alive()
    assert errors == []
    assert results[0][1].status == "done"
    assert claim_close_results == [False]
    assert claim.failed
    assert claim_ready.is_set()
    assert claim_close_done.is_set()

    second_run_id = db.create_run(conversation_id="conv-2", agent_kind="primary")
    second_turn = real_begin([root])
    second_turn.await_baseline()
    real_end(second_turn)
    second_follow_on = tracker.continuation(second_turn)
    assert second_follow_on is not None
    second_state = _ChildChangeState(owner_key="second-owner")
    close_done = _WaitRecordingEvent()
    failing_window = _PostTurnChangeWindow(
        run_id=second_run_id,
        session_id=session.id,
        handle=second_follow_on,
        child_states=(second_state,),
        close_done=close_done,
    )
    bridge._post_turn_change_windows["conv-2"] = failing_window
    close_started = threading.Event()
    release_close_failure = threading.Event()

    def fail_close(handle, *args, **kwargs):
        if handle is failing_window.handle:
            close_started.set()
            assert release_close_failure.wait(5), "close failure was never released"
            raise RuntimeError("injected close-time tracker failure")
        return real_end(handle, *args, **kwargs)

    monkeypatch.setattr(tracker, "end_turn", fail_close)
    adapter = SimpleNamespace(child_lifeline=contextlib.nullcontext)

    def finish_child_scope() -> None:
        try:
            with bridge._child_run_scope("conv-2", adapter, second_state):
                pass
        except BaseException as exc:  # noqa: BLE001 -- asserted on test thread
            errors.append(exc)

    owner = threading.Thread(target=finish_child_scope, name="failed-close-owner")
    waiter = threading.Thread(
        target=lambda: bridge._close_post_turn_change_window("conv-2"),
        name="failed-close-waiter",
    )
    owner.start()
    try:
        assert close_started.wait(5), "child teardown never entered tracker close"
        waiter.start()
        assert close_done.wait_started.wait(5), (
            "competing closer never waited for the close owner"
        )
    finally:
        release_close_failure.set()
        owner.join(10)
        waiter.join(10)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert errors == []
    assert close_done.is_set()
    assert bridge._post_turn_change_windows.get("conv-2") is None


def test_inherited_child_state_crosses_successor_e_and_second_window_without_backward_leakage(
    tmp_path, root, tracker, monkeypatch
):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True if section == "tools" and key == "write_file_enabled" else default
        ),
    )
    older_before_b = root / "older-before-b.txt"
    older_during_successor = root / "older-during-successor.txt"
    older_after_e = root / "older-after-e.txt"
    newer_child = root / "newer-child.txt"
    names = (
        older_before_b.name,
        older_during_successor.name,
        older_after_e.name,
        newer_child.name,
    )
    (root / ".gitignore").write_text("".join(f"{name}\n" for name in names))
    child_gate = threading.Event()
    keep_child_running = threading.Event()
    inherited_state = _ChildChangeState(
        owner_key="inherited-owner",
        pending_scopes=1,
    )

    def publish_inherited_path_and_release_child() -> None:
        older_during_successor.write_text("older child during successor\n")
        with bridge._change_window_lock:
            inherited_state.touched_paths.add(str(older_during_successor))
        child_gate.set()
        assert gateway.child_second_started.wait(5), (
            "successor child never completed its WRITE tool"
        )

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("write from successor child")],
            ["successor final"],
        ],
        gate=child_gate,
        child_scripts=[
            [_write_fence(newer_child, "newer child write\n")],
            ["newer child final"],
        ],
        second_gate=keep_child_running,
        parent_side_effect=publish_inherited_path_and_release_child,
        parent_side_effect_on_call=2,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    old_run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    old_turn = tracker.begin_turn([root])
    old_turn.await_baseline()
    tracker.end_turn(old_turn)
    follow_on = tracker.continuation(old_turn)
    assert follow_on is not None

    older_before_b.write_text("older child before B\n")
    inherited_state.touched_paths.add(str(older_before_b))
    old_window = _PostTurnChangeWindow(
        run_id=old_run_id,
        session_id=session.id,
        handle=follow_on,
        child_states=(inherited_state,),
    )
    bridge._post_turn_change_windows["conv-1"] = old_window
    bridge._child_change_states["conv-1"] = {inherited_state.owner_key: inherited_state}

    try:
        successor_run_id, outcome = _run(
            bridge,
            session,
            aid,
            root,
            builtin_gate=_FakeBuiltinGateForRegistry(refuse=False),
            scratch_root=root,
            scratch_lease=lambda: contextlib.nullcontext(root),
        )
        assert outcome.status == "done"
        assert gateway.child_second_started.is_set()
        assert newer_child.read_text() == "newer child write\n"

        repo = tracker.service.repo_for_root(root)
        old_rows = db.change_snapshots_for_run(old_run_id)
        assert len(old_rows) == 1, old_rows
        old_paths = [
            item.path
            for item in repo.changed_files(
                old_rows[0]["baseline_sha"], old_rows[0]["end_sha"]
            )
        ]
        assert old_paths == [older_before_b.name]
        assert newer_child.name not in old_paths

        successor_turn_rows = [
            row
            for row in db.change_snapshots_for_run(successor_run_id)
            if row["kind"] != "subagent_post_turn"
        ]
        assert len(successor_turn_rows) == 1, successor_turn_rows
        successor_paths = {
            item.path
            for item in repo.changed_files(
                successor_turn_rows[0]["baseline_sha"],
                successor_turn_rows[0]["end_sha"],
            )
        }
        assert successor_paths == {
            older_during_successor.name,
            newer_child.name,
        }

        second_window = bridge._post_turn_change_windows.get("conv-1")
        assert second_window is not None
        assert inherited_state in second_window.child_states
        assert len(second_window.child_states) == 2

        older_after_e.write_text("older child after successor E\n")
        with bridge._change_window_lock:
            inherited_state.touched_paths.add(str(older_after_e))
        bridge._close_post_turn_change_window("conv-1")

        post_turn_rows = [
            row
            for row in db.change_snapshots_for_run(successor_run_id)
            if row["kind"] == "subagent_post_turn"
        ]
        assert len(post_turn_rows) == 1, post_turn_rows
        post_turn_paths = [
            item.path
            for item in repo.changed_files(
                post_turn_rows[0]["baseline_sha"], post_turn_rows[0]["end_sha"]
            )
        ]
        assert post_turn_paths == [older_after_e.name]
    finally:
        child_gate.set()
        keep_child_running.set()
        _join_fleet_threads()
        with bridge._change_window_lock:
            bridge._child_change_states.pop("conv-1", None)
