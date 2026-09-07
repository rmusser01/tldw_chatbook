"""TASK-1970: ShadowRepoService — hardened per-root shadow git.

Every test drives REAL git in tmp dirs (no mocks — the whole point is that
git's actual behavior on real machines is the risk surface). The hostile-HOME
test is the crown jewel: each of its three hazards is a real first-turn
failure on a real dev machine.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.change_tracking import (
    ChangeTrackingUnavailableError,
    ShadowRepoService,
)


@pytest.fixture()
def service(tmp_path) -> ShadowRepoService:
    return ShadowRepoService(data_dir=tmp_path / "appdata")


@pytest.fixture()
def root(tmp_path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    return r


def _tree(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*")}


# -- the hostile HOME -------------------------------------------------------


def test_snapshot_succeeds_from_a_hostile_home(service, root, monkeypatch, tmp_path):
    """No identity + global gpgsign=true + a failing global hook: all three
    are real per-machine failures. Local pinned config must win over every
    one of them, silently.
    """
    home = tmp_path / "hostile_home"
    hooks = home / "hooks"
    hooks.mkdir(parents=True)
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\necho HOSTILE HOOK RAN >&2\nexit 1\n")
    hook.chmod(0o755)
    (home / ".gitconfig").write_text(
        "[commit]\n\tgpgsign = true\n"
        f"[core]\n\thooksPath = {hooks}\n"
        # deliberately NO [user] section: commit fails without identity
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))

    (root / "a.txt").write_text("one\n")
    repo = service.repo_for_root(root)
    first = repo.snapshot("baseline")
    (root / "a.txt").write_text("two\n")
    second = repo.snapshot("end")

    assert first and second and first != second
    changed = repo.changed_files(first, second)
    assert [c.path for c in changed] == ["a.txt"]


# -- identity & isolation ---------------------------------------------------


def test_symlinked_and_direct_root_resolve_to_one_shadow_repo(service, root, tmp_path):
    link = tmp_path / "root_link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported on this platform/permission level")
    direct = service.repo_for_root(root)
    via_link = service.repo_for_root(link)
    assert direct.git_dir == via_link.git_dir


def test_nothing_is_ever_created_inside_the_tracked_root(service, root):
    (root / "a.txt").write_text("content\n")
    before = _tree(root)
    repo = service.repo_for_root(root)
    repo.snapshot("baseline")
    (root / "b.txt").write_text("more\n")
    repo.snapshot("end")
    after = _tree(root)
    assert after - before == {"b.txt"}, (
        f"the service polluted the tracked root: {sorted(after - before)}"
    )


# -- concurrency ------------------------------------------------------------


def test_concurrent_snapshots_serialize(service, root):
    (root / "seed.txt").write_text("seed\n")
    repo = service.repo_for_root(root)
    repo.snapshot("baseline")
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            (root / f"file{n}.txt").write_text(f"{n}\n")
            service.repo_for_root(root).snapshot(f"turn {n}")
        except BaseException as exc:  # noqa: BLE001 -- surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"index.lock (or worse) leaked through: {errors!r}"


def test_a_stale_lockdir_is_taken_over(service, root):
    (root / "a.txt").write_text("x\n")
    repo = service.repo_for_root(root)
    repo.ensure_initialized()
    stale = repo.lock_dir
    stale.mkdir(parents=True, exist_ok=True)
    old = time.time() - 3600
    import os

    os.utime(stale, (old, old))
    sha = repo.snapshot("after stale lock")
    assert sha, "a crashed process's hour-old lockdir starved snapshots forever"


# -- availability -----------------------------------------------------------


def test_git_absent_is_reported_not_raised(tmp_path):
    svc = ShadowRepoService(
        data_dir=tmp_path / "appdata", git_executable="/nonexistent/git-binary"
    )
    assert svc.available is False
    with pytest.raises(ChangeTrackingUnavailableError):
        svc.repo_for_root(tmp_path)


def test_hostile_git_env_vars_are_scrubbed(service, root, monkeypatch, tmp_path):
    """The app itself can run under a git hook or IDE task that exports
    GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE — inherited by our subprocesses,
    they would redirect the snapshot into the WRONG repository.
    """
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-repo"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "wrong-index"))

    (root / "a.txt").write_text("x\n")
    repo = service.repo_for_root(root)
    sha = repo.snapshot("under hostile env")
    assert sha
    assert (repo.git_dir / "HEAD").exists()
    # GIT_DIR/GIT_WORK_TREE are already defeated by our explicit flags; the
    # live hazard is GIT_INDEX_FILE, which flags do NOT override -- without
    # scrubbing, the snapshot's index lands in the leaked path.
    assert not (tmp_path / "wrong-index").exists(), (
        "the leaked GIT_INDEX_FILE was used — env scrubbing is not happening"
    )
    assert (repo.git_dir / "index").exists()


# -- excludes ---------------------------------------------------------------


def test_forced_excludes_are_honored(service, root):
    (root / "app.py").write_text("print()\n")
    junk = root / "node_modules" / "pkg"
    junk.mkdir(parents=True)
    repo = service.repo_for_root(root)
    b = repo.snapshot("baseline")
    (junk / "index.js").write_text("junk\n")
    (root / "app.py").write_text("print(1)\n")
    e = repo.snapshot("end")
    assert [c.path for c in repo.changed_files(b, e)] == ["app.py"]


# -- hostile filenames (the -z contract) ------------------------------------


def test_hostile_filename_roundtrips_snapshot_diff_and_restore(service, root):
    """Spaces AND a newline in the name: paths are data, and restore executes
    file operations from parsed paths. Newline-in-filename is exactly what
    breaks non-z porcelain parsing silently.
    """
    name = "weird name\nwith newline.txt"
    hostile = root / name
    hostile.write_text("original\n")
    repo = service.repo_for_root(root)
    b = repo.snapshot("baseline")
    hostile.write_text("modified\n")
    e = repo.snapshot("end")

    changed = repo.changed_files(b, e)
    assert [c.path for c in changed] == [name]
    assert changed[0].status == "M"

    repo.restore_paths(b, [name])
    assert hostile.read_text() == "original\n"


def test_a_toplevel_git_file_is_never_tracked_or_restored(service, root):
    """A linked git WORKTREE carries a `.git` FILE at its root. Git's own
    path special-casing refuses to track it (verified before writing this),
    and the exclude pins that guarantee against future edits -- restoring a
    worktree's `.git` link would corrupt the user's worktree.
    """
    (root / ".git").write_text("gitdir: /main/repo/.git/worktrees/x\n")
    (root / "code.py").write_text("x = 1\n")
    repo = service.repo_for_root(root)
    b = repo.snapshot("baseline")
    (root / "code.py").write_text("x = 2\n")
    e = repo.snapshot("end")
    assert [c.path for c in repo.changed_files(b, e)] == ["code.py"]
    proc = repo._run("ls-tree", "-r", "--name-only", e)
    assert ".git" not in str(proc.stdout).split()


def test_mixed_case_git_env_vars_are_scrubbed(service, root, monkeypatch, tmp_path):
    """Windows env vars are case-insensitive: `Git_Index_File` reaches git
    exactly as GIT_INDEX_FILE does. Asserted on the env builder directly --
    POSIX git ignores the lowercase spelling, so a subprocess test could not
    tell scrubbed from ignored.
    """
    monkeypatch.setenv("Git_Index_File", str(tmp_path / "wrong-index"))
    repo = service.repo_for_root(root)
    assert "Git_Index_File" not in repo._env()


def test_a_missing_root_fails_fast_with_a_clear_error(service, tmp_path):
    from tldw_chatbook.Workspaces.change_tracking import ChangeTrackingError

    with pytest.raises(ChangeTrackingError, match="not a directory"):
        service.repo_for_root(tmp_path / "vanished")


def test_a_typechange_status_passes_through_verbatim(service, root):
    """file -> symlink is git status `T`; coercing it to A/M/D/R would lie.
    Documented pass-through, consumers bucket unknown letters as "other".
    """
    target = root / "pointee.txt"
    target.write_text("data\n")
    changer = root / "changer"
    changer.write_text("was a file\n")
    repo = service.repo_for_root(root)
    b = repo.snapshot("baseline")
    changer.unlink()
    try:
        changer.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unsupported on this platform/permission level")
    e = repo.snapshot("end")
    by_path = {c.path: c for c in repo.changed_files(b, e)}
    assert by_path["changer"].status == "T"


# -- change classification --------------------------------------------------


def test_changed_files_classifies_add_modify_delete_rename_binary(service, root):
    (root / "keep.txt").write_text("keep\n")
    (root / "edit.txt").write_text("before\n")
    (root / "gone.txt").write_text("delete me\n")
    (root / "old_name.txt").write_text("stable content that identifies a rename\n" * 5)
    (root / "image.bin").write_bytes(b"\x00\x01\x02")
    repo = service.repo_for_root(root)
    b = repo.snapshot("baseline")

    (root / "new.txt").write_text("created\n")
    (root / "edit.txt").write_text("after\n")
    (root / "gone.txt").unlink()
    (root / "old_name.txt").rename(root / "new_name.txt")
    (root / "image.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    e = repo.snapshot("end")

    by_path = {c.path: c for c in repo.changed_files(b, e)}
    assert by_path["new.txt"].status == "A"
    assert by_path["edit.txt"].status == "M"
    assert by_path["gone.txt"].status == "D"
    assert by_path["new_name.txt"].status == "R"
    assert by_path["new_name.txt"].old_path == "old_name.txt"
    assert by_path["image.bin"].binary is True
    assert by_path["edit.txt"].adds == 1 and by_path["edit.txt"].dels == 1
    assert "keep.txt" not in by_path

    diff = repo.diff_text(b, e, "edit.txt")
    assert "-before" in diff and "+after" in diff
    assert repo.file_bytes(b, "edit.txt") == b"before\n"


# -- snapshot economics -----------------------------------------------------


def test_clean_tree_snapshot_returns_existing_tip_without_new_commit(service, root):
    (root / "a.txt").write_text("x\n")
    repo = service.repo_for_root(root)
    first = repo.snapshot("baseline")
    second = repo.snapshot("no changes")
    assert first == second


def test_first_snapshot_of_an_empty_root_still_produces_a_tip(service, root):
    repo = service.repo_for_root(root)
    tip = repo.snapshot("baseline of empty root")
    assert tip


# -- portability ------------------------------------------------------------


def test_no_flock_dependency_in_the_module():
    """Windows CI lanes exist; fcntl does not exist there.

    Asserted on the module's IMPORTS (ast walk), not its prose -- the
    docstring legitimately explains WHY flock is not used, and a substring
    check would forbid the explanation.
    """
    import ast
    import inspect

    from tldw_chatbook.Workspaces import change_tracking

    tree = ast.parse(inspect.getsource(change_tracking))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.split(".")[0] == "fcntl" for m in imported), imported
