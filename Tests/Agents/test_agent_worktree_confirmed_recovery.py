"""Confirmed recovery against owned real Git and SQLite fixtures."""

import importlib
import subprocess
from dataclasses import replace

import pytest

from tldw_chatbook.Agents.agent_worktree import (
    AgentWorktree,
    WorktreeRefusal,
    _git_common_directory_identity,
    _worktree_root_identity,
    create_agent_worktree,
)
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


@pytest.fixture
def work(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import agent_worktree

    monkeypatch.setattr(
        agent_worktree, "_worktrees_base", lambda: tmp_path / "children"
    )
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "a.txt").write_text("base\n")
    (root / ".gitignore").write_text("ignored\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "base")
    child = create_agent_worktree(root, "child-1")
    assert isinstance(child, AgentWorktree)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    db.create_run(conversation_id="chat", agent_kind="subagent", run_id=child.run_id)
    db.set_status(child.run_id, status="done", result="done")
    authority = RunAdmittedWorkspaceRoot(
        "workspace",
        "binding",
        "repo",
        root,
        "f" * 64,
        _worktree_root_identity(root),
        True,
        lambda write: True,
    )
    common, identity = _git_common_directory_identity(root)
    records = AgentWorktreeRepository(db)
    records.record_created(
        run_id=child.run_id,
        workspace_id=authority.workspace_id,
        binding_id=authority.binding_id,
        locator_fingerprint=authority.locator_fingerprint,
        repo_root=str(root),
        repo_identity=authority.root_identity,
        git_common_dir=str(common),
        git_common_identity=identity,
        child_path=str(child.worktree_path),
        child_identity=_worktree_root_identity(child.worktree_path),
        branch=child.branch,
        base_sha=child.base_sha,
        execution_id="execution",
    )
    records.mark_writer_finished(child.run_id, "execution", cleanup_proven=True)
    yield db, authority, child, records
    db.close()


def recover(work, action="apply", confirm=lambda payload: {"allow": True}, **kwargs):
    # Assertion gives an explicit RED feature failure before the module exists.
    assert importlib.util.find_spec("tldw_chatbook.Agents.agent_worktree_recovery"), (
        "shared recovery engine missing"
    )
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    db, authority, child, _ = work
    params = {
        "authority": authority,
        "conversation_id": "chat",
        "run_id": child.run_id,
        "action": action,
        "request_confirmation": confirm,
        "should_cancel": lambda: False,
    }
    params.update(kwargs)
    return module.recover_agent_worktree(db, **params)


def state(work):
    return work[3].get_for_conversation(work[2].run_id, "chat")["mutation_state"]


@pytest.mark.parametrize(
    "confirm", [None, lambda p: {"allow": False}, lambda p: {"allow": 1}]
)
def test_without_exact_allow_retains_unresolved_source(work, confirm):
    child = work[2].worktree_path
    (child / "a.txt").write_text("child\n")
    head = git(child, "rev-parse", "HEAD")
    assert isinstance(recover(work, confirm=confirm), WorktreeRefusal)
    assert git(child, "rev-parse", "HEAD") == head
    assert (child / "a.txt").read_text() == "child\n"
    assert state(work) == "unresolved"


def test_apply_original_base_binary_untracked_preserves_parent_index(work):
    root, child = work[1].root, work[2].worktree_path
    (root / "parent").write_text("advanced")
    git(root, "add", "-A")
    git(root, "commit", "-m", "advance")
    (root / "staged").write_text("keep")
    git(root, "add", "staged")
    index = git(root, "diff", "--cached", "--binary")
    (child / "a.txt").write_text("child\n")
    (child / "new\nfile").write_bytes(bytes(range(256)))
    outcome = recover(work)
    assert getattr(outcome, "state", None) == "applied", outcome
    assert (root / "a.txt").read_text() == "child\n"
    assert (root / "new\nfile").read_bytes() == bytes(range(256))
    assert git(root, "diff", "--cached", "--binary") == index
    assert (root / "parent").read_text() == "advanced"


def test_merge_creates_explicit_two_parent_commit(work):
    (work[2].worktree_path / "a.txt").write_text("child\n")
    result = recover(work, "merge")
    assert getattr(result, "state", None) == "merged", result
    assert (
        len(git(work[1].root, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
    )


@pytest.mark.parametrize("action", ["apply", "merge"])
def test_conflict_preserves_parent_and_source_releases_claim(work, action):
    root, child = work[1].root, work[2].worktree_path
    (root / "a.txt").write_text("parent\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "parent")
    head = git(root, "rev-parse", "HEAD")
    (child / "a.txt").write_text("child\n")
    assert isinstance(recover(work, action), WorktreeRefusal)
    assert git(root, "rev-parse", "HEAD") == head
    assert (root / "a.txt").read_text() == "parent\n"
    assert (child / "a.txt").read_text() == "child\n"
    assert state(work) == "unresolved"


@pytest.mark.parametrize("where", ["child", "parent"])
def test_changed_confirmation_snapshot_refuses_without_claim(work, where):
    def allow(payload):
        path = work[2].worktree_path if where == "child" else work[1].root
        (path / "new").write_text("changed while waiting")
        return {"allow": True}

    assert isinstance(recover(work, confirm=allow), WorktreeRefusal)
    assert state(work) == "unresolved"


@pytest.mark.parametrize(
    "change", ["conversation", "binding", "identity", "held", "uncertain"]
)
def test_ineligible_record_never_asks_confirmation(work, change):
    kwargs = {}
    if change == "conversation":
        kwargs["conversation_id"] = "other"
    if change == "binding":
        kwargs["authority"] = replace(work[1], binding_id="other")
    if change == "identity":
        kwargs["authority"] = replace(work[1], root_identity=(("/", 1, 2, 3),))
    if change in ("held", "uncertain"):
        with work[0].transaction() as connection:
            connection.execute("UPDATE agent_worktrees SET writer_state=?", (change,))

    def forbidden(payload):
        pytest.fail("ineligible record requested confirmation")

    assert isinstance(recover(work, confirm=forbidden, **kwargs), WorktreeRefusal)


@pytest.mark.parametrize("dirty", [False, True])
def test_discard_retains_detached_baseline_and_external_symlink_target(
    work, tmp_path, dirty
):
    child = work[2].worktree_path
    outside = tmp_path / "outside"
    outside.write_text("survives")
    if dirty:
        (child / "a.txt").write_text("child\n")
        (child / "ignored").write_text("ignored changes")
        (child / "link").symlink_to(outside)
    payloads = []
    result = recover(work, "discard", lambda p: payloads.append(p) or {"allow": True})
    assert getattr(result, "state", None) == "discarded_cleanup_pending", result
    assert payloads[0]["retains_checkout"] is True
    assert child.is_dir() and (child / ".git").is_file()
    assert (child / "a.txt").read_text() == "base\n"
    assert outside.read_text() == "survives"
    assert not (child / "ignored").exists() and not (child / "link").exists()
    assert git(child, "rev-parse", "HEAD").decode().strip() == work[2].base_sha
    assert git(work[1].root, "branch", "--list", work[2].branch) == b""


def test_nested_repository_discard_refuses_without_mutation(work):
    child = work[2].worktree_path
    nested = child / "nested"
    nested.mkdir()
    git(nested, "init")
    (child / "a.txt").write_text("child\n")
    assert isinstance(recover(work, "discard"), WorktreeRefusal)
    assert (child / "a.txt").read_text() == "child\n"
    assert state(work) == "unresolved"


def test_cancel_before_consent_leaves_source(work):
    assert isinstance(recover(work, should_cancel=lambda: True), WorktreeRefusal)
    assert state(work) == "unresolved"


def test_no_effect_release_requires_exact_operation_owner(work):
    records = work[3]
    assert records.claim("child-1", "chat", operation_id="one", action="apply")
    assert not records.claim("child-1", "chat", operation_id="two", action="apply")
    assert not records.finish_operation("child-1", "two", state="unresolved")
    assert records.finish_operation("child-1", "one", state="unresolved")
    assert records.claim("child-1", "chat", operation_id="two", action="merge")


def test_large_unchanged_baseline_does_not_consume_snapshot_budget(work, monkeypatch):
    root = work[1].root
    # A large unchanged parent file is unrelated to a small child delta.
    (root / "large").write_bytes(b"x" * 4096)
    git(root, "add", "large")
    git(root, "commit", "-m", "large parent baseline")
    (work[2].worktree_path / "a.txt").write_text("child\n")
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    monkeypatch.setattr(module, "MAX_OUTPUT", 1024)
    result = recover(work)
    assert getattr(result, "state", None) == "applied", result


def test_persistence_failure_after_effect_cannot_replay_on_reopen(work, monkeypatch):
    child = work[2].worktree_path
    (child / "a.txt").write_text("child\n")
    monkeypatch.setattr(
        AgentWorktreeRepository, "finish_operation", lambda *a, **k: False
    )
    result = recover(work)
    assert isinstance(result, WorktreeRefusal)
    assert (work[1].root / "a.txt").read_text() == "child\n"
    assert state(work) == "applying"
    work[0].close()
    assert isinstance(recover(work), WorktreeRefusal)
    assert state(work) == "applying"


def test_discard_refuses_missing_primitives_before_consent(work, monkeypatch):
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    monkeypatch.setattr(module, "_primitives", lambda: False)
    (work[2].worktree_path / "a.txt").write_text("child\n")
    result = recover(
        work, "discard", lambda p: pytest.fail("must refuse before consent")
    )
    assert isinstance(result, WorktreeRefusal)
    assert result.reason_code == "unsupported_primitives"
    assert state(work) == "unresolved"


def test_exact_branch_change_during_confirmation_refuses(work):
    def allow(payload):
        git(work[2].worktree_path, "checkout", "-b", "other")
        return {"allow": True}

    assert isinstance(recover(work, "discard", allow), WorktreeRefusal)
    assert git(work[1].root, "branch", "--list", work[2].branch)
    assert state(work) == "unresolved"


def test_clean_merge_does_not_report_a_nonexistent_merge_commit(work):
    result = recover(work, "merge")
    assert isinstance(result, WorktreeRefusal)
    assert result.reason_code == "nothing_to_merge"
    assert state(work) == "unresolved"


def test_discard_preview_discloses_ignored_entries(work):
    (work[2].worktree_path / "ignored").write_text("to discard")
    payloads = []
    recover(work, "discard", lambda p: payloads.append(p) or {"allow": False})
    assert "ignored" in payloads[0]["diffstat"]


def test_source_administrative_symlink_is_not_adopted(work, tmp_path):
    child = work[2].worktree_path
    link = child / ".git"
    saved = tmp_path / "administrative-link"
    link.rename(saved)
    link.symlink_to(saved)
    result = recover(work)
    assert isinstance(result, WorktreeRefusal)
    assert result.reason_code == "unsupported_layout"


def test_oversized_patch_retains_source_capture_and_releases_no_effect_claim(
    work, monkeypatch
):
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    from tldw_chatbook.Agents import agent_worktree_git

    (work[2].worktree_path / "new").write_bytes(bytes(range(256)) * 32)
    real_run = module.run_git

    def capped_patch(*args, **kwargs):
        if kwargs.get("output") is not None:
            monkeypatch.setattr(agent_worktree_git, "MAX_OUTPUT", 64)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(module, "run_git", capped_patch)
    result = recover(work)
    assert isinstance(result, WorktreeRefusal)
    assert result.reason_code == "output_limit"
    assert not (work[1].root / "new").exists()
    assert (
        git(work[2].worktree_path, "rev-parse", "HEAD").decode().strip()
        != work[2].base_sha
    )
    assert state(work) == "unresolved"


def test_missing_discard_only_primitive_does_not_disable_apply(work, monkeypatch):
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    monkeypatch.setattr(module, "_primitives", lambda: False)
    (work[2].worktree_path / "a.txt").write_text("child\n")
    result = recover(work)
    assert getattr(result, "state", None) == "applied", result


def test_git_reader_reaps_background_filter_process(work, tmp_path):
    import os
    import signal
    import sys
    import time

    from tldw_chatbook.Agents.agent_worktree_git import OperationError, run_git
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    root = work[1].root
    script = tmp_path / "filter.py"
    pidfile = tmp_path / "filter.pid"
    heartbeat = tmp_path / "heartbeat"
    script.write_text(
        "import os, sys, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    fd = os.open(os.devnull, os.O_RDWR)\n"
        "    for target in (0, 1, 2): os.dup2(fd, target)\n"
        f'    for i in range(1000):\n        open({str(heartbeat)!r}, "w").write(str(i)); time.sleep(.02)\n'
        "    os._exit(0)\n"
        f'open({str(pidfile)!r}, "w").write(str(child))\n'
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
    )
    git(root, "config", "filter.owned.clean", f"{sys.executable} {script}")
    (root / ".gitattributes").write_text("a.txt filter=owned\n")
    owner = RuntimeCapacity().begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="chat"
    )
    drained = []
    owner.on_drained(drained.append)
    cleanup_refused = False
    try:
        with owner.activate():
            try:
                run_git(root, "hash-object", "--path=a.txt", "a.txt")
            except OperationError as exc:
                assert exc.code == "cleanup_unproven"
                cleanup_refused = True
        owner.finish_root()
        assert drained == [not cleanup_refused]
        time.sleep(0.1)
        before = heartbeat.read_text() if heartbeat.exists() else None
        time.sleep(0.15)
        after = heartbeat.read_text() if heartbeat.exists() else None
        assert before == after, "Git descendant remains physically live"
    finally:
        owner.finish_root()
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_discard_exact_ref_cas_preserves_a_concurrently_changed_branch(
    work, monkeypatch
):
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    child = work[2].worktree_path
    (child / "a.txt").write_text("committed child\n")
    git(child, "add", "-A")
    git(child, "commit", "-m", "child")
    real_run = module.run_git

    def changed_ref(root, *args, **kwargs):
        if args[:2] == ("update-ref", "-d"):
            git(root, "update-ref", args[2], work[2].base_sha)
        return real_run(root, *args, **kwargs)

    monkeypatch.setattr(module, "run_git", changed_ref)
    result = recover(work, "discard")
    assert isinstance(result, WorktreeRefusal)
    assert state(work) == "uncertain"
    assert (
        git(work[1].root, "rev-parse", "refs/heads/" + work[2].branch).decode().strip()
        == work[2].base_sha
    )
    assert child.is_dir()


def test_preexisting_merge_is_not_aborted_or_confirmed(work):
    root = work[1].root
    git(root, "checkout", "-b", "side")
    (root / "a.txt").write_text("side\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "side")
    git(root, "checkout", "main")
    (root / "a.txt").write_text("main\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "main")
    merge = subprocess.run(
        ["git", "merge", "side"], cwd=root, capture_output=True, check=False
    )
    assert merge.returncode != 0
    before = git(root, "rev-parse", "MERGE_HEAD")
    result = recover(
        work, "merge", lambda p: pytest.fail("preexisting merge must refuse")
    )
    assert isinstance(result, WorktreeRefusal)
    assert git(root, "rev-parse", "MERGE_HEAD") == before
    assert state(work) == "unresolved"


@pytest.mark.parametrize("action", ["apply", "merge"])
def test_generated_mutations_disable_repository_hooks(work, action):
    common, _ = _git_common_directory_identity(work[1].root)
    sentinel = work[1].root / "hook-ran"
    for name in ("pre-commit", "pre-merge-commit"):
        hook = common / "hooks" / name
        hook.write_text(f'#!/bin/sh\ntouch "{sentinel}"\nexit 1\n')
        hook.chmod(0o700)
    (work[2].worktree_path / "a.txt").write_text("child\n")
    result = recover(work, action)
    assert getattr(result, "state", None) == (
        "applied" if action == "apply" else "merged"
    ), result
    assert not sentinel.exists()


@pytest.mark.parametrize("advance_parent", [False, True])
def test_already_incorporated_child_refuses_without_merge_receipt(work, advance_parent):
    root, child = work[1].root, work[2].worktree_path
    (child / "a.txt").write_text("child commit\n")
    git(child, "add", "-A")
    git(child, "commit", "-m", "child")
    git(root, "merge", "--ff-only", work[2].branch)
    if advance_parent:
        (root / "later").write_text("parent advancement\n")
        git(root, "add", "-A")
        git(root, "commit", "-m", "later parent")
    previous = git(root, "rev-parse", "HEAD")
    child_head = git(child, "rev-parse", "HEAD")

    result = recover(work, "merge")

    assert isinstance(result, WorktreeRefusal), result
    assert result.reason_code == "already_merged"
    assert git(root, "rev-parse", "HEAD") == previous
    assert git(child, "rev-parse", "HEAD") == child_head
    assert state(work) == "unresolved"


@pytest.mark.parametrize("interference", ["no_commit", "wrong_parents", "unreadable"])
def test_unverified_merge_receipt_preserves_uncertainty(
    work, monkeypatch, interference
):
    module = importlib.import_module("tldw_chatbook.Agents.agent_worktree_recovery")
    from tldw_chatbook.Agents.agent_worktree_git import OperationError

    root, child = work[1].root, work[2].worktree_path
    (child / "a.txt").write_text("child change\n")
    previous = git(root, "rev-parse", "HEAD")
    real_run = module.run_git
    merged = False

    def altered_merge(path, *args, **kwargs):
        nonlocal merged
        if args[:1] == ("merge",) and "--no-ff" in args:
            if interference == "no_commit":
                merged = True
                return b"Already up to date.\n"
            result = real_run(path, *args, **kwargs)
            merged = True
            if interference == "wrong_parents":
                git(root, "commit", "--allow-empty", "-m", "external advancement")
            return result
        if merged and interference == "unreadable" and args[:1] == ("rev-list",):
            raise OperationError("git_failed", "commit verification unavailable")
        return real_run(path, *args, **kwargs)

    monkeypatch.setattr(module, "run_git", altered_merge)
    result = recover(work, "merge")

    assert merged
    assert isinstance(result, WorktreeRefusal), result
    assert state(work) == "uncertain"
    assert (git(root, "rev-parse", "HEAD") == previous) is (interference == "no_commit")
    work[0].close()
    assert isinstance(recover(work, "merge"), WorktreeRefusal)
    assert state(work) == "uncertain"


def test_incorporated_child_with_new_dirty_work_still_merges(work):
    root, child = work[1].root, work[2].worktree_path
    (child / "a.txt").write_text("first child commit\n")
    git(child, "add", "-A")
    git(child, "commit", "-m", "first child")
    git(root, "merge", "--ff-only", work[2].branch)
    previous = git(root, "rev-parse", "HEAD").decode().strip()
    (child / "a.txt").write_text("additional child work\n")

    result = recover(work, "merge")

    assert getattr(result, "state", None) == "merged", result
    child_head = git(child, "rev-parse", "HEAD").decode().strip()
    assert git(root, "rev-list", "--parents", "-n", "1", "HEAD").decode().split() == [
        result.commit_sha,
        previous,
        child_head,
    ]
    assert (root / "a.txt").read_text() == "additional child work\n"


@pytest.mark.parametrize("boundary", ["construct", "start"])
@pytest.mark.parametrize("reader_number", [1, 2])
@pytest.mark.parametrize("retirement_denied", [False, True])
def test_git_partial_reader_startup_retires_gated_mutation(
    tmp_path, monkeypatch, boundary, reader_number, retirement_denied
):
    """A failed reader must not publish proven drain over a live Git mutation."""
    import os
    import select
    import signal
    import sys
    import threading
    from types import SimpleNamespace

    from tldw_chatbook.Agents import agent_worktree_git as module
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(
        root,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "base",
    )
    gate_read, gate_write = os.pipe()
    ready_read, ready_write = os.pipe()
    processes = []
    readers = []
    real_popen = subprocess.Popen
    real_killpg = os.killpg
    attempts = 0

    def gated_popen(argv, **kwargs):
        # Preserve the real Git argv; only delay exec until the parent opens its gate.
        script = (
            "import os,sys; "
            f"os.write({ready_write}, b'R'); os.close({ready_write}); "
            f"os.read({gate_read}, 1); os.close({gate_read}); "
            "os.execv(sys.argv[1], sys.argv[1:])"
        )
        proc = real_popen(
            [sys.executable, "-c", script, *argv],
            pass_fds=(gate_read, ready_write),
            **kwargs,
        )
        processes.append(proc)
        assert select.select([ready_read], [], [], 3)[0], (
            "owned child did not reach gate"
        )
        assert os.read(ready_read, 1) == b"R"
        return proc

    def make_reader(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        fail = attempts == reader_number
        if fail and boundary == "construct":
            raise RuntimeError("reader construction refused")
        reader = threading.Thread(*args, **kwargs)
        readers.append(reader)
        if fail and boundary == "start":

            def refused():
                raise RuntimeError("reader start refused")

            reader.start = refused
        return reader

    def denied_killpg(pid, sig):
        assert pid == processes[0].pid
        if sig == signal.SIGKILL:
            raise PermissionError("owned retirement refused")
        return real_killpg(pid, sig)

    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="chat")
    drained = []
    owner.on_drained(drained.append)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.subprocess, "Popen", gated_popen)
            patch.setattr(
                module,
                "threading",
                SimpleNamespace(Thread=make_reader, Event=threading.Event),
            )
            if retirement_denied:
                patch.setattr(module.os, "killpg", denied_killpg)
            try:
                with (
                    owner.activate(),
                    pytest.raises((RuntimeError, module.OperationError)),
                ):
                    module.run_git(
                        root, "update-ref", "refs/heads/startup-proof", "HEAD"
                    )
                owner.finish_root()
                assert drained == [not retirement_denied]
                proc = processes[0]
                if retirement_denied:
                    assert proc.poll() is None, (
                        "fake cleanup must retain the real live child"
                    )
                else:
                    assert proc.poll() == -signal.SIGKILL, (
                        "proven drain left gated Git alive"
                    )
                    with pytest.raises(ProcessLookupError):
                        real_killpg(proc.pid, 0)
                    assert all(not reader.is_alive() for reader in readers)
                    assert proc.stdout.closed and proc.stderr.closed
                    os.write(gate_write, b"G")
                    assert not (root / ".git/refs/heads/startup-proof").exists()
            finally:
                # Reap only our exact group before restoring patched helpers.
                for proc in processes:
                    try:
                        real_killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=3)
                for reader in readers:
                    if reader.ident is not None:
                        reader.join(timeout=3)
                        assert not reader.is_alive()
                for proc in processes:
                    proc.stdout.close()
                    proc.stderr.close()
        assert git(root, "branch", "--list", "startup-proof") == b""
    finally:
        for fd in (gate_read, gate_write, ready_read, ready_write):
            os.close(fd)
        owner.finish_root()
        capacity.close()
