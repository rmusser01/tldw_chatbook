"""Sensitive-path coverage for the workspace-local ``git_*`` family.

TASK-19632. TASK-19551 made ``resolve_workspace_path`` enforce the
sensitive-path denylist for every path a model NAMES, and made the three
enumerating ``fs_*`` tools filter the entries they present but the model
never named. The ``git_*`` tools share that choke point for their path
ARGUMENT -- but ``path`` is OPTIONAL on ``git_status``/``git_log``/
``git_diff``, and with it omitted nothing but the repository root ever
reaches the denylist: git enumerates the repository on the tool's behalf
and the tool returns what comes back.

Measured before the fix, with the workspace root set to ``$HOME`` (exactly
what the shipped ``workspace_root`` default -- the app's cwd at startup --
produces when the app is launched from the user's home directory) and
``$HOME`` a git repository containing a synthetic ``.ssh/id_rsa``:

* ``git_diff(commit_range="HEAD~1..HEAD")`` returned the key's CONTENT
  from a CLEAN worktree -- no write primitive and no dirty tree needed.
* ``git_diff(stat=True)`` and ``git_status`` returned its NAME.
* ``git_log`` leaked nothing, and is deliberately left unfiltered.

Plus one vector found while building the fix: pathspec MAGIC in a
repository FILENAME. ``:(exclude)notes.txt`` is a legal POSIX filename, so
``git_diff(path=":(exclude)notes.txt")`` passed the choke point as an
ordinary confined path and then inverted the diff's scope, returning the
rest of the repository -- ``~/.ssh/id_rsa``'s content included -- while
nominally scoping to one file. ``--`` does not stop that: it ends OPTION
parsing, not magic parsing.

The fix constrains git's INPUT (``:(exclude)`` pathspecs computed per call
from the live denylist) rather than filtering its OUTPUT, so the negative
half of this file matters as much as the positive half: an ordinary diff,
stat, status and log must come back untouched, byte for byte.

``Tests/conftest.py``'s autouse ``isolate_test_environment`` fixture
redirects HOME/XDG/``TLDW_CONFIG_PATH`` to per-test tmp directories, so
every "credential" below is a synthetic marker under an isolated home --
no real credential file is ever created or read.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Tools.git_tool_impls import (
    _denylist_pathspecs,
    _glob_escape,
    git_diff,
    git_log,
    git_status,
    prepare_repository,
)
from tldw_chatbook.Tools.local_tool_impls import LocalToolError

GIT_AVAILABLE = shutil.which("git") is not None
pytestmark = pytest.mark.skipif(
    not GIT_AVAILABLE, reason="git is not available on this system"
)

SSH_MARKER = "SYNTHETIC-NOT-A-REAL-PRIVATE-KEY-19632"
NETRC_MARKER = "SYNTHETIC-NOT-A-REAL-PASSWORD-19632"
#: A legal POSIX filename that is ALSO a git pathspec meaning "everything
#: except notes.txt". The whole point of the injection case below.
MAGIC_NAME = ":(exclude)notes.txt"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")


def _home() -> Path:
    return Path(os.environ["HOME"]).resolve()


def _raw_git(repo: Path, *args: str) -> str:
    """Run git directly, so a tool's output can be compared against it."""
    completed = subprocess.run(
        ["git", "--no-pager", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


@pytest.fixture
def home_repo() -> Path:
    """``$HOME`` as both workspace root AND git repo, holding a synthetic key.

    Three commits, so the negative pins have something to hold onto:

    * ``initial``  -- ``notes.txt`` v1, ``.ssh/id_rsa`` v1, ``MAGIC_NAME``
    * ``second``   -- ``notes.txt`` v2 AND ``.ssh/id_rsa`` v2 (the range
      used for the content-leak cases: an ordinary file and a denylisted
      one change together, so "the secret is gone" and "the diff is not
      truncated" are asserted on the SAME output)
    * ``third``    -- ``.ssh/id_rsa`` v3 only (a commit touching NOTHING
      but a denylisted path -- what ``git_log`` must still list)
    """
    home = _home()
    _init(home)
    key = home / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\n")
    (home / "notes.txt").write_text("hello\n")
    (home / MAGIC_NAME).write_text("magic-filename\n")
    _git(home, "add", "-A", "-f")
    _git(home, "commit", "-m", "initial")

    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nROTATED\n")
    (home / "notes.txt").write_text("hello\nworld\n")
    _git(home, "add", "-A", "-f")
    _git(home, "commit", "-m", "second")

    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nAGAIN\n")
    _git(home, "add", "-A", "-f")
    _git(home, "commit", "-m", "third")
    return home


def _assert_no_leak(output: str, label: str) -> None:
    """Born-red evidence: at base each of these prints the leaked bytes."""
    assert SSH_MARKER not in output, (
        f"{label} leaked the private key's CONTENT -- it returned:\n{output}"
    )
    assert "id_rsa" not in output, (
        f"{label} leaked the private key's NAME -- it returned:\n{output}"
    )


# ---------------------------------------------------------------------------
# The leaks, one test per measured case (TASK-19632 AC1/AC2/AC4).
# ---------------------------------------------------------------------------


def test_git_diff_commit_range_never_returns_denylisted_content_on_a_clean_worktree(
    home_repo: Path,
) -> None:
    """The headline: a read-only agent on a CLEAN checkout was enough.

    ``commit_range`` reads the credential out of HISTORY, so neither a
    write primitive nor a dirty worktree is required -- which is what makes
    this reachable by prompt injection from fetched web content: the model
    only has to call a ``reads``-tagged git tool with no path.
    """
    assert _raw_git(home_repo, "status", "--porcelain").strip() == "", (
        "the worktree must be CLEAN for this case to mean what it claims"
    )

    out = git_diff(home_repo, commit_range="HEAD~2..HEAD~1")

    _assert_no_leak(out, "git_diff(commit_range=..., no path)")
    # ... and the ordinary file that changed in the SAME range is intact.
    assert "notes.txt" in out
    assert "+world" in out


def test_git_diff_stat_never_names_a_denylisted_path(home_repo: Path) -> None:
    """``stat=True`` disclosed the NAME (and the change size) only."""
    out = git_diff(home_repo, commit_range="HEAD~2..HEAD~1", stat=True)

    _assert_no_leak(out, "git_diff(stat=True, no path)")
    assert "notes.txt" in out


def test_git_diff_worktree_never_returns_denylisted_content(home_repo: Path) -> None:
    """The dirty-worktree case (the one the original draft over-claimed)."""
    (home_repo / ".ssh" / "id_rsa").write_text(
        f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nDIRTY\n"
    )
    (home_repo / "notes.txt").write_text("hello\nworld\ndirty\n")

    _assert_no_leak(git_diff(home_repo), "git_diff() dirty")
    _assert_no_leak(git_diff(home_repo, stat=True), "git_diff(stat=True) dirty")
    assert "notes.txt" in git_diff(home_repo)


def test_git_diff_index_never_returns_denylisted_content(home_repo: Path) -> None:
    """The INDEX scope (``staged=True``) -- AC1 names it explicitly."""
    (home_repo / ".ssh" / "id_rsa").write_text(
        f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nSTAGED\n"
    )
    (home_repo / "notes.txt").write_text("hello\nworld\nstaged\n")
    _git(home_repo, "add", "-A", "-f")

    out = git_diff(home_repo, staged=True)

    _assert_no_leak(out, "git_diff(staged=True)")
    assert "notes.txt" in out


def test_git_status_never_names_a_denylisted_path(home_repo: Path) -> None:
    """``git_status`` disclosed existence + name on a dirty tree."""
    (home_repo / ".ssh" / "id_rsa").write_text(
        f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nDIRTY\n"
    )
    (home_repo / "notes.txt").write_text("hello\nworld\ndirty\n")
    (home_repo / ".ssh" / "untracked_key").write_text(SSH_MARKER)

    out = git_status(home_repo)

    _assert_no_leak(out, "git_status()")
    assert "untracked_key" not in out, (
        f"git_status leaked an UNTRACKED denylisted path:\n{out}"
    )
    assert "notes.txt" in out


def test_git_diff_still_refuses_a_denylisted_path_argument(home_repo: Path) -> None:
    """The TASK-19551 behaviour is unchanged: naming the path is refused."""
    with pytest.raises(LocalToolError, match="protected path"):
        git_diff(home_repo, commit_range="HEAD~2..HEAD~1", path=".ssh/id_rsa")


# ---------------------------------------------------------------------------
# Pathspec magic in a repository FILENAME -- a pathspec is not a path.
# ---------------------------------------------------------------------------


def test_pathspec_magic_in_the_path_argument_cannot_invert_the_scope(
    home_repo: Path,
) -> None:
    """``path=":(exclude)notes.txt"`` returned the rest of the repository.

    The file genuinely exists, so it resolves, stays inside the workspace
    and is not denylisted -- the choke point has no reason to refuse it.
    git then read it as MAGIC rather than as a name. Before the fix this
    call returned ``.ssh/id_rsa``'s content; after it, the pathspec is
    rendered ``:(literal)`` and scopes to the one real file.
    """
    (home_repo / MAGIC_NAME).write_text("magic-filename\nchanged\n")
    (home_repo / "notes.txt").write_text("hello\nworld\nalso changed\n")
    # The key must be dirty too, or the injected ":(exclude)notes.txt"
    # scope would have nothing to leak and this test would pass at base
    # for the wrong reason (caught while checking the born-red run).
    (home_repo / ".ssh" / "id_rsa").write_text(
        f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nINJECTED\n"
    )

    out = git_diff(home_repo, path=MAGIC_NAME)

    _assert_no_leak(out, f"git_diff(path={MAGIC_NAME!r})")
    # Positive half: it scopes to the named file, and ONLY to it.
    assert "changed" in out, (
        f"the literal pathspec stopped matching its own file:\n{out}"
    )
    assert "also changed" not in out, (
        f"the scope leaked past the named file into notes.txt:\n{out}"
    )


def test_every_pathspec_this_family_builds_carries_explicit_magic() -> None:
    """Structural pin: no bare value may reach a pathspec position.

    ``--`` ends OPTION parsing, not pathspec-magic parsing, so a bare
    value after it is a repository-supplied injection point: a file
    legitimately named ``:(exclude)notes.txt`` inverts the scope of
    whatever command it lands in. Only two shapes are allowed through --
    ``_literal_pathspec(...)``, which disables magic for one value, and a
    splat of ``_denylist_pathspecs(...)``, which renders its own.

    Checked over EVERY list literal in the module that contains ``"--"``,
    and THROUGH a local list that is splatted into one. An earlier form of
    this test only inspected ``.extend([...])`` arguments and waved every
    ``*splat`` through, which post-fix left both leaking tools uncovered:
    ``git_status`` builds its whole argv as one literal handed to
    ``_run_git_checked``, and ``git_diff`` accumulates into a local
    ``pathspecs`` list and splats that. Measured: splicing a bare
    model-supplied value into either left the old test green. ``git_blame``
    stays exempt -- it takes a plain PATH, not a pathspec, and rejects
    magic outright (verified: ``fatal: no such path ':(literal)a.txt' in
    HEAD``).
    """
    import ast

    from tldw_chatbook.Tools import git_tool_impls

    source = Path(git_tool_impls.__file__).read_text()
    tree = ast.parse(source)

    def _is_call_to(node: ast.AST, names: set[str]) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        )

    def _local_list_is_safe(func: ast.FunctionDef, name: str) -> list[str]:
        """Every value that can reach the local list ``name``."""
        bad: list[str] = []
        for node in ast.walk(func):
            # Rebinding it to anything but a fresh empty list hides the flow.
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value  # type: ignore[union-attr]
                    if value is not None and not (
                        isinstance(value, ast.List) and not value.elts
                    ):
                        bad.append(f"{name} = {ast.dump(value)[:100]}")
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
            ):
                continue
            allowed = (
                {"_literal_pathspec"}
                if node.func.attr == "append"
                else {"_denylist_pathspecs"}
            )
            if node.func.attr not in {"append", "extend"}:
                bad.append(f"{name}.{node.func.attr}(...)")
                continue
            for arg in node.args:
                if not _is_call_to(arg, allowed):
                    bad.append(f"{name}.{node.func.attr}({ast.dump(arg)[:100]})")
        return bad

    offenders: list[str] = []
    for func in tree.body:
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func.name == "git_blame":
            continue  # plain path, see the docstring above
        for literal in ast.walk(func):
            if not isinstance(literal, (ast.List, ast.Tuple)):
                continue
            elements = list(literal.elts)
            separators = [
                index
                for index, element in enumerate(elements)
                if isinstance(element, ast.Constant) and element.value == "--"
            ]
            if not separators:
                continue
            for element in elements[separators[0] + 1 :]:
                if _is_call_to(element, {"_literal_pathspec"}):
                    continue
                if isinstance(element, ast.Starred):
                    if _is_call_to(element.value, {"_denylist_pathspecs"}):
                        continue
                    if isinstance(element.value, ast.Name):
                        offenders.extend(
                            f"{func.name}: via *{element.value.id} -- {reason}"
                            for reason in _local_list_is_safe(func, element.value.id)
                        )
                        continue
                offenders.append(f"{func.name}: {ast.dump(element)[:120]}")

    assert not offenders, (
        "these values reach a pathspec position without explicit magic, so a "
        "repository file named ':(exclude)x' can invert the command's scope: "
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# The negative side: the fix must not truncate a legitimate result.
# ---------------------------------------------------------------------------


@pytest.fixture
def ordinary_repo(tmp_path: Path) -> Path:
    """A repo with nothing denylisted anywhere near it."""
    repo = (tmp_path / "project").resolve()
    _init(repo)
    (repo / "pkg").mkdir()
    for name in ("a.txt", "b.txt", "pkg/c.txt"):
        (repo / name).write_text(f"{name} v1\n")
    (repo / ".github").mkdir()
    (repo / ".github" / "ci.yml").write_text("name: ci\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    for name in ("a.txt", "b.txt", "pkg/c.txt"):
        (repo / name).write_text(f"{name} v1\n{name} v2\n")
    (repo / ".github" / "ci.yml").write_text("name: ci\njobs: {}\n")
    return repo


def test_an_ordinary_multi_file_diff_is_byte_identical_to_raw_git(
    ordinary_repo: Path,
) -> None:
    """The whole risk of an input-side fix: over-exclusion.

    Compared against git run directly with the same flags, not against a
    hand-written expectation, so a pathspec that quietly drops one file
    cannot pass by matching a stale literal.
    """
    expected = _raw_git(
        ordinary_repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--unified=3",
    )
    assert git_diff(ordinary_repo) == expected
    for name in ("a.txt", "b.txt", "pkg/c.txt", ".github/ci.yml"):
        assert name in expected, "fixture drifted"


def test_an_ordinary_stat_and_status_are_unchanged(ordinary_repo: Path) -> None:
    stat = git_diff(ordinary_repo, stat=True)
    assert stat == _raw_git(
        ordinary_repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--stat",
    )

    status = git_status(ordinary_repo)
    for name in ("a.txt", "b.txt", "pkg/c.txt", ".github/ci.yml"):
        assert name in status, f"{name} vanished from git_status:\n{status}"


def test_an_ordinary_scoped_diff_is_unchanged(ordinary_repo: Path) -> None:
    """A ``path`` argument still scopes exactly as it did (file AND dir)."""
    out = git_diff(ordinary_repo, path="pkg")
    assert "pkg/c.txt" in out
    assert "a.txt" not in out

    out = git_diff(ordinary_repo, path="a.txt")
    assert "a.txt" in out
    assert "b.txt" not in out


def test_git_log_is_unfiltered_including_commits_that_touch_only_denied_paths(
    home_repo: Path,
) -> None:
    """AC5: ``git_log`` leaks nothing, so it must not be filtered either.

    The ``third`` commit touches NOTHING but ``.ssh/id_rsa``. Excluding
    denylisted paths from ``git log`` would delete it from the history the
    model sees while protecting nothing -- the format below emits commit
    metadata only, no paths and no content.
    """
    out = git_log(home_repo)

    assert "third" in out, f"a commit disappeared from git_log:\n{out}"
    assert "second" in out
    assert "initial" in out
    _assert_no_leak(out, "git_log()")


def test_git_log_scoped_by_path_is_unchanged(ordinary_repo: Path) -> None:
    out = git_log(ordinary_repo, path="pkg")
    assert "initial" in out


# ---------------------------------------------------------------------------
# Rule fidelity: each denial kind must be expressed as EXACTLY that rule.
# ---------------------------------------------------------------------------


def test_the_container_rule_hides_direct_child_files_and_nothing_deeper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``direct_children`` must not become a subtree exclusion.

    ``is_sensitive_path``'s container rule refuses loose FILES sitting
    directly inside one of this app's state directories while leaving the
    directories nested in them (``tool_sandbox``, ``skills``, ...) fully
    reachable. The pathspec has to draw the same line: ``:(exclude,glob)``
    is used precisely because ``*`` does not cross ``/`` under glob magic.

    The effective config directory is the container here, relocated into
    the repository through ``TLDW_CONFIG_PATH`` -- the same override the
    denylist itself honors, so nothing is stubbed.
    """
    repo = (tmp_path / "project").resolve()
    _init(repo)
    container = repo / "appstate"
    (container / "nested").mkdir(parents=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(container / "config.toml"))

    (repo / "keep.txt").write_text("v1\n")
    (container / "loose.txt").write_text("secret v1\n")
    (container / "nested" / "deep.txt").write_text("ordinary v1\n")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-m", "initial")
    (repo / "keep.txt").write_text("v1\nv2\n")
    (container / "loose.txt").write_text("secret v1\nsecret v2\n")
    (container / "nested" / "deep.txt").write_text("ordinary v1\nordinary v2\n")

    out = git_diff(repo)

    assert "appstate/loose.txt" not in out, (
        f"a loose file inside a protected container leaked:\n{out}"
    )
    assert "secret v2" not in out
    assert "appstate/nested/deep.txt" in out, (
        "the container exclusion swallowed a nested directory the denylist "
        f"deliberately leaves reachable:\n{out}"
    )
    assert "keep.txt" in out


def test_the_name_rule_is_excluded_at_every_depth(tmp_path: Path) -> None:
    """TASK-19633's name rule must reach git too, wherever the file sits."""
    repo = (tmp_path / "project").resolve()
    _init(repo)
    (repo / "sub" / "deeper").mkdir(parents=True)
    for rel in (".netrc", "sub/.netrc", "sub/deeper/.netrc"):
        (repo / rel).write_text(f"machine example.invalid password {NETRC_MARKER}\n")
    (repo / "sub" / "keep.txt").write_text("v1\n")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-m", "initial")
    for rel in (".netrc", "sub/.netrc", "sub/deeper/.netrc"):
        (repo / rel).write_text(f"machine example.invalid password {NETRC_MARKER}2\n")
    (repo / "sub" / "keep.txt").write_text("v1\nv2\n")

    out = git_diff(repo)

    assert NETRC_MARKER not in out, f".netrc content leaked through git_diff:\n{out}"
    assert ".netrc" not in out
    assert "sub/keep.txt" in out


def test_glob_escaping_confines_a_container_exclusion_to_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container whose NAME contains a glob metacharacter.

    Unescaped, ``:(exclude,glob)co*ntainer/*`` also swallows
    ``coXntainer/``. Verified against git 2.39 while building the fix; the
    escaping is what keeps the exclusion to the directory it names.
    """
    assert _glob_escape("co*ntainer") == "co\\*ntainer"
    assert _glob_escape("a?b[c]\\d") == "a\\?b\\[c]\\\\d"

    repo = (tmp_path / "project").resolve()
    _init(repo)
    container = repo / "co*ntainer"
    container.mkdir()
    (repo / "coXntainer").mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(container / "config.toml"))
    (container / "loose.txt").write_text("v1\n")
    (repo / "coXntainer" / "loose.txt").write_text("v1\n")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-m", "initial")
    (container / "loose.txt").write_text("v1\nv2\n")
    (repo / "coXntainer" / "loose.txt").write_text("v1\nv2\n")

    out = git_diff(repo)

    assert "co*ntainer/loose.txt" not in out
    assert "coXntainer/loose.txt" in out, (
        f"an unescaped glob exclusion swallowed a lookalike directory:\n{out}"
    )


# ---------------------------------------------------------------------------
# Drift pins between the denylist and its git rendering.
# ---------------------------------------------------------------------------


def test_exclusions_survive_a_case_variant_spelling(tmp_path: Path) -> None:
    """The git side must fold case exactly as the denylist does (TASK-19800).

    git records whatever spelling a path was added under, and on a
    case-insensitive filesystem that need not match the denylist's. An
    exclusion that misses ``.NETRC`` because the rule says ``.netrc``
    is a leak, so every exclusion carries ``icase``. The SCOPING pathspec
    deliberately does not -- folding it would add files to the output.
    """
    repo = (tmp_path / "project").resolve()
    _init(repo)
    (repo / "sub").mkdir()
    (repo / "sub" / ".NETRC").write_text(f"password {NETRC_MARKER}\n")
    (repo / "keep.txt").write_text("v1\n")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-m", "initial")
    (repo / "sub" / ".NETRC").write_text(f"password {NETRC_MARKER}2\n")
    (repo / "keep.txt").write_text("v1\nv2\n")

    out = git_diff(repo)

    assert NETRC_MARKER not in out, (
        f"a case-variant spelling of a name-rule credential leaked:\n{out}"
    )
    assert ".NETRC" not in out
    assert "keep.txt" in out


def test_a_location_exclusion_survives_a_case_variant_directory_spelling() -> None:
    """The same fold, on the LOCATION rules -- the case that was unpinned.

    The test above covers the name rule's ``:(exclude,glob,icase)**/<name>``
    form. Nothing covered ``:(exclude,literal,icase)<rel>``, which renders
    ``_SENSITIVE_DIRS`` and the resolved single-file denials -- and that is
    the form the module docstring's own headline example depends on:
    the rule is spelled ``~/.ssh``, git records whatever spelling the path
    was ADDED under, and on the case-insensitive filesystems this app ships
    on ``~/.SSH/id_rsa`` opens the very same file. ``is_sensitive_path``
    folds and refuses that spelling (TASK-19800); the pathspec rendered
    from it has to reach the same verdict or the denial is decorative.

    Found by mutation while reviewing TASK-19632: dropping ``icase`` from
    the ``subtree``/``file`` branch of ``_denylist_pathspecs`` left the
    whole suite GREEN while ``git_diff`` returned the key's content and
    ``stat``/``status`` returned its name. This test is what reds instead.
    """
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    home = _home()
    _init(home)
    # `.SSH`, not `.ssh`: the denylist's spelling and git's must differ.
    key = home / ".SSH" / "id_rsa"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\n")
    (home / "keep.txt").write_text("v1\n")
    _git(home, "add", "-A", "-f")
    _git(home, "commit", "-m", "initial")
    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nROTATED\n")
    (home / "keep.txt").write_text("v1\nv2\n")
    _git(home, "add", "-A", "-f")
    _git(home, "commit", "-m", "second")

    # The premise: the denylist itself already refuses this spelling, so
    # any disclosure below is the git rendering disagreeing with it.
    assert is_sensitive_path(key), (
        "premise broken: the denylist must already refuse the case variant "
        "(TASK-19800) for this test to be about the pathspec rendering"
    )

    _assert_no_leak(
        git_diff(home, commit_range="HEAD~1..HEAD"),
        "git_diff(commit_range) on a case-variant location",
    )
    _assert_no_leak(
        git_diff(home, commit_range="HEAD~1..HEAD", stat=True),
        "git_diff(stat) on a case-variant location",
    )
    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_MARKER}\nAGAIN\n")
    (home / "keep.txt").write_text("v1\nv2\nv3\n")
    _assert_no_leak(git_diff(home), "git_diff(worktree) on a case-variant location")
    _assert_no_leak(git_status(home), "git_status on a case-variant location")

    # Not vacuous: the ordinary file beside it is still reported.
    assert "keep.txt" in git_diff(home)
    assert "keep.txt" in git_status(home)


def test_every_exclusion_kind_the_denylist_emits_can_be_rendered(
    tmp_path: Path,
) -> None:
    """A new denial kind must fail loudly here, not pass through silently."""
    from tldw_chatbook.Utils.sensitive_paths import (
        SensitiveExclusion,
        sensitive_exclusions_under,
    )

    repo = (tmp_path / "project").resolve()
    repo.mkdir()
    kinds = {kind for kind, _ in sensitive_exclusions_under(repo)}
    assert "name" in kinds, "the name rule must always apply"

    specs = _denylist_pathspecs(repo)
    assert specs, "an empty exclusion list would mean no protection at all"
    assert all(spec.startswith(":(exclude,") for spec in specs), specs

    import tldw_chatbook.Utils.sensitive_paths as sp

    original = sp.sensitive_exclusions_under
    try:
        sp.sensitive_exclusions_under = lambda root, context=None: (
            SensitiveExclusion("a_kind_from_the_future", "x"),
        )
        import tldw_chatbook.Tools.git_tool_impls as gti

        gti.sensitive_exclusions_under = sp.sensitive_exclusions_under
        with pytest.raises(LocalToolError, match="unsupported sensitive-path"):
            _denylist_pathspecs(repo)
    finally:
        sp.sensitive_exclusions_under = original
        import tldw_chatbook.Tools.git_tool_impls as gti

        gti.sensitive_exclusions_under = original


def test_the_runner_environment_never_sets_a_pathspec_mode() -> None:
    """``GIT_LITERAL_PATHSPECS=1`` would disable every exclusion, silently.

    TASK-16801's lesson recommends exactly that variable as blanket
    hardening for git argv, and it is the natural thing for a future
    reader to add here. Under it ``:(exclude,literal)<path>`` is read as a
    literal FILENAME, matches nothing, and ``git_diff``/``git_status``
    return "(no changes)"/"(working tree clean)" with exit 0 -- a total
    functional break AND a total protection break, with no error anywhere.
    Verified on git 2.39 while writing this.

    ``_git_environment`` builds its environment from scratch, so an
    ambient one never reaches git either; this pins that neither direction
    changes.
    """
    from tldw_chatbook.Tools.git_tool_impls import _git_environment

    env = _git_environment()
    leaked = {key for key in env if key.endswith("_PATHSPECS")}
    assert not leaked, (
        "a pathspec-mode variable in the git runner's environment silently "
        f"disables the sensitive-path exclusions: {sorted(leaked)}"
    )


def test_prepare_repository_refuses_a_protected_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excluding denied paths from a repo that is ITSELF denied is nonsense.

    The repo root is DISCOVERED by git rather than supplied by the model,
    so it is the one path in this family the choke point never sees.
    """
    repo = (tmp_path / "project").resolve()
    _init(repo)
    (repo / "x.txt").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")

    import tldw_chatbook.Tools.git_tool_impls as gti

    monkeypatch.setattr(gti, "is_sensitive_path", lambda candidate: True)
    with pytest.raises(LocalToolError, match="protected path"):
        prepare_repository(repo, ".")
