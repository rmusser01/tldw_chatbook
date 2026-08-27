"""Push engine for change-review git modes (TASK-16801 arc B, T4).

Every push test drives a REAL local bare remote (`git init --bare`) -- the
engine has no mockable seam by design (spec
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`,
AC #5, no mocked git).

`test_nonff_push_fails_honestly_no_force` is the regression pin for the
arc's no-force invariant: it wraps `_run_user_git` to capture every argv
issued during the failing push and asserts none of them carry
`--force`/`--force-with-lease` -- a non-fast-forward rejection must
surface git's own stderr excerpt honestly, never be silently retried with
force.

The `test_repo_config_*` family is the SECOND half of that invariant, and
the one an argv audit cannot see: a push carrying no refspec lets
`.git/config` -- which an agent can write, since no `.git` exclusion
exists in `workspace_file_roots.py` -- decide what the push actually does.
Each of those tests asserts the DESTRUCTION first (the other clone's
commit, the `release` branch, the `v1` tag, the private branch's SECRET
commit) and only then the returned `PushResult`; every one of them was
proven red against the pre-fix engine, which pushed with no refspec at
all.
"""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces import git_workspace
from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceError,
    PushResult,
    _push_failure_detail,
    detect_git_workspace,
    push_current,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


@pytest.fixture()
def bare(tmp_path: Path) -> Path:
    bare_root = tmp_path / "bare.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare_root)],
        check=True,
        capture_output=True,
    )
    _git(bare_root, "symbolic-ref", "HEAD", "refs/heads/main")
    return bare_root


def test_first_push_sets_upstream_and_moves_bare_ref(repo, bare):
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)
    assert info.upstream is None
    assert info.remotes == (("origin", str(bare)),)

    result = push_current(repo, info, None)
    assert isinstance(result, PushResult)
    assert result.state == "pushed"
    assert result.detail == ""

    bare_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert bare_head == _git(repo, "rev-parse", "HEAD")

    info_after = detect_git_workspace(repo)
    assert info_after.upstream == "origin/main"
    assert info_after.upstream_remote == "origin"


def test_second_push_up_to_date(repo, bare):
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)
    first = push_current(repo, info, None)
    assert first.state == "pushed"

    info2 = detect_git_workspace(repo)
    assert info2.upstream == "origin/main"
    second = push_current(repo, info2, None)
    assert second.state == "up_to_date"
    assert second.detail == ""


def test_nonff_push_fails_honestly_no_force(repo, bare, tmp_path, monkeypatch):
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)
    first = push_current(repo, info, None)
    assert first.state == "pushed"

    # A second clone commits and pushes first, so our next push is a
    # genuine non-fast-forward rejection (never a synthetic one).
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(clone)],
        check=True,
        capture_output=True,
    )
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")
    (clone / "a.txt").write_text("clone change\n")
    _git(clone, "commit", "-qam", "clone change")
    _git(clone, "push", "-q", "origin", "main")

    (repo / "a.txt").write_text("local change\n")
    _git(repo, "commit", "-qam", "local change")

    captured_argv: list[tuple[str, ...]] = []
    real_run = git_workspace._run_user_git

    def _spy(root, *args, **kwargs):
        captured_argv.append(args)
        return real_run(root, *args, **kwargs)

    monkeypatch.setattr(git_workspace, "_run_user_git", _spy)

    info2 = detect_git_workspace(repo)
    assert info2.upstream == "origin/main"
    result = push_current(repo, info2, None)

    assert result.state == "failed"
    assert "rejected" in result.detail

    push_calls = [args for args in captured_argv if args and args[0] == "push"]
    assert push_calls  # sanity: the push was actually attempted
    for args in push_calls:
        assert not any(arg.startswith("--force") for arg in args)

    # The bare remote's ref did NOT move -- the rejection was real.
    bare_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    clone_head = _git(clone, "rev-parse", "HEAD")
    assert bare_head == clone_head


def test_credential_hint_mapping():
    detail = _push_failure_detail(
        "fatal: could not read Username for 'https://host': "
        "terminal prompts disabled"
    )
    assert detail.endswith(
        " — credentials were not available non-interactively; push once "
        "from a terminal or configure a credential helper/ssh agent"
    )


def test_credential_hint_not_appended_for_unrelated_failure():
    detail = _push_failure_detail("fatal: unable to access 'https://host': "
                                   "Could not resolve host")
    assert detail == (
        "fatal: unable to access 'https://host': Could not resolve host"
    )


def test_detached_refused(repo):
    _git(repo, "checkout", "-q", "--detach")
    info = detect_git_workspace(repo)
    assert info.detached and info.branch is None
    with pytest.raises(GitWorkspaceError):
        push_current(repo, info, None)


def test_no_remote_refused(repo):
    info = detect_git_workspace(repo)
    assert info.remotes == ()
    with pytest.raises(GitWorkspaceError):
        push_current(repo, info, None)


def test_explicit_remote_overrides_derivation(repo, bare, tmp_path):
    other_bare = tmp_path / "other.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(other_bare)],
        check=True,
        capture_output=True,
    )
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "remote", "add", "other", str(other_bare))
    info = detect_git_workspace(repo)
    assert len(info.remotes) == 2

    result = push_current(repo, info, "other")
    assert result.state == "pushed"
    other_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=other_bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert other_head == _git(repo, "rev-parse", "HEAD")
    # The untargeted remote was never touched.
    bare_head_proc = subprocess.run(
        ["git", "rev-parse", "main"], cwd=bare, capture_output=True, text=True
    )
    assert bare_head_proc.returncode != 0  # "main" never pushed there


# ---------------------------------------------------------------------------
# Repository-supplied push configuration (the third argv-injection shape:
# the destructive option never appears in OUR argv at all).
# ---------------------------------------------------------------------------


def _bare_refs(bare: Path) -> dict[str, str]:
    """Every ref in the bare remote, as ``{refname: full sha}``."""
    out = subprocess.run(
        ["git", "--git-dir", str(bare), "for-each-ref", "--format=%(refname) %(objectname)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    refs: dict[str, str] = {}
    for line in out.splitlines():
        name, _, sha = line.partition(" ")
        if name:
            refs[name] = sha
    return refs


@pytest.fixture()
def shared(repo, bare, tmp_path):
    """A REAL shared remote another clone has already published work to.

    Returns ``(clone, precious_sha, refs_before)``. The bare carries the
    other clone's `main` commit, a `release` branch, and a `v1` tag -- the
    three things a repository-supplied push config destroys.
    """
    _git(repo, "remote", "add", "origin", str(bare))
    push_current(repo, detect_git_workspace(repo), None)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(clone)], check=True, capture_output=True
    )
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")
    (clone / "precious.txt").write_text("precious\n")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", "PRECIOUS")
    _git(clone, "push", "-q", "origin", "main")
    _git(clone, "tag", "v1")
    _git(clone, "push", "-q", "origin", "v1")
    _git(clone, "branch", "release")
    _git(clone, "push", "-q", "origin", "release")

    precious = _git(clone, "rev-parse", "HEAD")
    return clone, precious, _bare_refs(bare)


def test_repo_config_force_refspec_cannot_rewrite_the_shared_remote(repo, bare, shared):
    """`remote.origin.push = +refs/heads/*:refs/heads/*` (spec §6).

    Pre-fix this printed `+ <old>...<new> main -> main (forced update)`
    and the other clone's commit was GONE -- with no `--force` in our
    argv, a non-dash remote and a non-dash branch, so both existing
    option-injection guards passed cleanly.
    """
    _clone, precious, refs_before = shared
    _git(repo, "config", "remote.origin.push", "+refs/heads/*:refs/heads/*")
    (repo / "a.txt").write_text("local change\n")
    _git(repo, "commit", "-qam", "local change")

    result = push_current(repo, detect_git_workspace(repo), None)

    refs_after = _bare_refs(bare)
    assert refs_after["refs/heads/main"] == precious, (
        "the other clone's commit was DESTROYED by a repository-supplied "
        f"force refspec; refs now {refs_after!r}"
    )
    assert refs_after == refs_before, f"no ref may move; got {refs_after!r}"
    assert result.state == "failed"
    assert "rejected" in result.detail


def test_repo_config_mirror_cannot_delete_remote_refs(repo, bare, shared):
    """`remote.origin.mirror = true` -- forced update PLUS `- [deleted]`.

    Pre-fix this deleted `refs/heads/release` AND `refs/tags/v1` from the
    shared remote and force-rewound `main`. An explicit refspec makes git
    refuse the combination honestly instead.
    """
    _clone, precious, refs_before = shared
    _git(repo, "config", "remote.origin.mirror", "true")
    (repo / "a.txt").write_text("local change\n")
    _git(repo, "commit", "-qam", "local change")

    result = push_current(repo, detect_git_workspace(repo), None)

    refs_after = _bare_refs(bare)
    assert "refs/heads/release" in refs_after, "the release BRANCH was deleted"
    assert "refs/tags/v1" in refs_after, "the v1 TAG was deleted"
    assert refs_after["refs/heads/main"] == precious
    assert refs_after == refs_before, f"no ref may move; got {refs_after!r}"
    assert result.state == "failed"
    assert result.detail, "a refused push must carry git's own reason"


def test_repo_config_delete_refspec_cannot_delete_a_branch(repo, bare, shared):
    """`remote.origin.push = :refs/heads/release` deletes a branch we never named."""
    _clone, _precious, refs_before = shared
    _git(repo, "config", "remote.origin.push", ":refs/heads/release")
    (repo / "a.txt").write_text("local change\n")
    _git(repo, "commit", "-qam", "local change")

    push_current(repo, detect_git_workspace(repo), None)

    refs_after = _bare_refs(bare)
    assert "refs/heads/release" in refs_after, (
        "a repository-supplied DELETE refspec removed a branch this push "
        f"never named; refs now {refs_after!r}"
    )
    assert refs_after == refs_before


def test_repo_config_push_default_matching_cannot_publish_another_branch(
    repo, bare, shared
):
    """`push.default = matching` published an unrelated branch's SECRET commit.

    The modal names ONE branch; pre-fix a refspec-less `git push origin`
    also shipped every other local branch whose name matched a remote one.
    """
    clone, _precious, _refs_before = shared
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "reset", "-q", "--hard", "origin/main")
    _git(repo, "branch", "-q", "release", "origin/release")
    _git(repo, "checkout", "-q", "release")
    (repo / "secret.txt").write_text("secret\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "SECRET private wip")
    _git(repo, "checkout", "-q", "main")
    (repo / "b.txt").write_text("legit\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "legit work")
    _git(repo, "config", "push.default", "matching")

    release_before = _bare_refs(bare)["refs/heads/release"]
    result = push_current(repo, detect_git_workspace(repo), None)

    refs_after = _bare_refs(bare)
    assert refs_after["refs/heads/release"] == release_before, (
        "an unrelated branch's private commit was published to the shared "
        "remote by push.default=matching"
    )
    # The branch the user DID name still lands.
    assert result.state == "pushed"
    assert refs_after["refs/heads/main"] == _git(repo, "rev-parse", "HEAD")


def test_repo_config_force_refspec_cannot_rewrite_on_the_first_push(
    repo, bare, tmp_path
):
    """The `-u` (no-upstream) form leaks the same config force.

    `git push -u origin <branch>` looks the bare branch name up in the
    remote's configured push refspecs, and inherits that refspec's `+`.
    """
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(seed)], check=True, capture_output=True
    )
    _git(seed, "config", "user.email", "t@t")
    _git(seed, "config", "user.name", "t")
    (seed / "precious.txt").write_text("precious\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-qm", "PRECIOUS")
    _git(seed, "branch", "-M", "main")
    _git(seed, "push", "-q", "origin", "main")
    precious = _git(seed, "rev-parse", "HEAD")

    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "config", "remote.origin.push", "+refs/heads/*:refs/heads/*")
    info = detect_git_workspace(repo)
    assert info.upstream is None, "this test must exercise the `-u` form"

    result = push_current(repo, info, None)

    assert _bare_refs(bare)["refs/heads/main"] == precious, (
        "the first push force-rewound a branch it had never seen"
    )
    assert result.state == "failed"


def test_push_argv_carries_a_fully_qualified_refspec(repo, bare, monkeypatch):
    """Both push forms must name src AND dst explicitly (the C1 pin).

    A bare `("push", remote)` or `("push", "-u", remote, branch)` hands the
    decision to repository config; this asserts the shape that takes it
    back.
    """
    _git(repo, "remote", "add", "origin", str(bare))
    captured: list[tuple[str, ...]] = []
    real_run = git_workspace._run_user_git

    def _spy(root, *args, **kwargs):
        captured.append(args)
        return real_run(root, *args, **kwargs)

    monkeypatch.setattr(git_workspace, "_run_user_git", _spy)

    first = push_current(repo, detect_git_workspace(repo), None)
    assert first.state == "pushed"
    second = push_current(repo, detect_git_workspace(repo), None)
    assert second.state == "up_to_date", "the refspec must not break up-to-date"

    pushes = [args for args in captured if args and args[0] == "push"]
    assert len(pushes) == 2, f"expected one argv per push; got {pushes!r}"
    assert pushes[0] == (
        "push",
        "-u",
        "origin",
        "refs/heads/main:refs/heads/main",
    ), pushes[0]
    assert pushes[1] == (
        "push",
        "origin",
        "refs/heads/main:refs/heads/main",
    ), pushes[1]


def test_push_targets_the_upstreams_own_ref_when_the_names_differ(
    repo, bare, tmp_path
):
    """A branch tracking a DIFFERENTLY-named upstream pushes to that ref.

    The destination is the branch's `%(upstream:remoteref)`, never the
    local branch's name -- otherwise the refspec would create a new remote
    branch instead of updating the tracked one.
    """
    _git(repo, "remote", "add", "origin", str(bare))
    push_current(repo, detect_git_workspace(repo), None)  # publishes main
    _git(repo, "checkout", "-q", "-b", "feat/x")
    _git(repo, "config", "branch.feat/x.remote", "origin")
    _git(repo, "config", "branch.feat/x.merge", "refs/heads/main")
    (repo / "a.txt").write_text("on feat/x\n")
    _git(repo, "commit", "-qam", "on feat/x")

    info = detect_git_workspace(repo)
    assert info.upstream == "origin/main"
    result = push_current(repo, info, None)

    assert result.state == "pushed"
    refs = _bare_refs(bare)
    assert refs["refs/heads/main"] == _git(repo, "rev-parse", "HEAD")
    assert "refs/heads/feat/x" not in refs, (
        f"the push must not invent a remote branch; got {sorted(refs)!r}"
    )


def test_unresolvable_upstream_ref_refuses_rather_than_pushing_bare(
    repo, bare, monkeypatch
):
    """If the upstream's remote ref can't be read, REFUSE -- never fall back.

    Falling back to a refspec-less `git push <remote>` is exactly the
    vector this fix closes, so the failure mode has to be a refusal.
    """
    _git(repo, "remote", "add", "origin", str(bare))
    push_current(repo, detect_git_workspace(repo), None)
    info = detect_git_workspace(repo)
    assert info.upstream == "origin/main"

    monkeypatch.setattr(
        git_workspace, "_upstream_remote_ref", lambda root, branch: None
    )
    with pytest.raises(GitWorkspaceError):
        push_current(repo, info, None)


def test_push_result_is_frozen_dataclass():
    result = PushResult(state="pushed", detail="")
    with pytest.raises(Exception):
        result.state = "failed"  # type: ignore[misc]
