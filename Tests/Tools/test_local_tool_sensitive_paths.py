"""Sensitive-path denylist coverage for the workspace-local ``fs_*`` family.

TASK-19551. The seven ``fs_*`` agent tools confine every path to the
configured ``[console] workspace_root`` but never consulted
``Utils.sensitive_paths.is_sensitive_path`` -- so with the shipped default
root (the app's cwd at startup; launch from ``$HOME`` and ``$HOME`` IS the
root) ``fs_read`` returned ``~/.ssh/id_rsa`` and ``fs_write``/``fs_patch``
could rewrite ``mcp_permissions.json``, turning every ``ask`` into
``allow`` -- a one-step bypass of the permission gate, reachable by prompt
injection from fetched web content.

These tests are the mirror image of ``Tests/Tools/test_file_tool_sandbox.py``
(which pins the same property for the OTHER file-tool family,
``Tools/file_operation_tools.py``); the last test here pins that the two
families cannot drift apart again, since that drift IS this bug.

``Tests/conftest.py``'s autouse ``isolate_test_environment`` fixture
redirects HOME/XDG/``TLDW_CONFIG_PATH`` to per-test tmp directories, so
every "credential" file below is a synthesized marker under an isolated
home -- no real credential file is ever created or read.

Each scenario deliberately puts the denied path under the workspace root
with NO dotted component in the RELATIVE portion where that is possible, so
``validate_path``'s own hidden-component check cannot mask whether the
denylist ran at all -- except the two tests that specifically exercise a
dotfile component, which is the shape ``allow_hidden=True`` (ADR-032)
deliberately lets past confinement and which therefore depends entirely on
the denylist.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tldw_chatbook.Tools.local_tool_impls import (
    LocalToolError,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    read_file,
    resolve_workspace_path,
    write_file,
)
from tldw_chatbook.Tools.patch_tool_impls import patch_files

SSH_KEY_MARKER = "SYNTHETIC-NOT-A-REAL-PRIVATE-KEY-19551"
AWS_MARKER = "aws_secret_access_key = SYNTHETIC-19551"


def _refused(call, label: str) -> str:
    """Run ``call``; return its refusal message or fail showing what leaked.

    Born-red evidence lives in the failure message: at base (before the
    fix) each of these calls SUCCEEDS, and the assertion failure prints the
    exact credential content / write receipt the tool handed back.
    """
    try:
        result = call()
    except LocalToolError as exc:
        return str(exc)
    pytest.fail(f"{label} was NOT refused -- the tool returned: {result!r}")


def _home() -> Path:
    return Path(os.environ["HOME"]).resolve()


def _plant_ssh_key() -> Path:
    key = _home() / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(f"-----BEGIN OPENSSH PRIVATE KEY-----\n{SSH_KEY_MARKER}\n")
    return key


def _plant_permission_store() -> Path:
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    user_data_dir.mkdir(parents=True, exist_ok=True)
    store = user_data_dir / "mcp_permissions.json"
    store.write_text('{"version": 1}\n')
    return store


# ---------------------------------------------------------------------------
# Shape 1: reading a denylisted credential path through fs_read.
# ---------------------------------------------------------------------------


def test_fs_read_refuses_denylisted_credential_path():
    """``fs_read`` must not hand ``~/.ssh/id_rsa`` to the model.

    The workspace root is ``$HOME`` -- exactly what the shipped
    ``workspace_root`` default (app cwd at startup) produces when the app is
    launched from the user's home directory.
    """
    key = _plant_ssh_key()
    message = _refused(
        lambda: read_file(".ssh/id_rsa", workspace_root=_home()), "fs_read(~/.ssh/id_rsa)"
    )
    assert "protected path" in message
    assert SSH_KEY_MARKER not in message
    assert key.read_text().count(SSH_KEY_MARKER) == 1  # untouched


def test_fs_read_refuses_this_apps_own_config_toml():
    """The file holding the user's provider API keys.

    Resolved through ``config._get_effective_config_path()`` (which honors
    ``TLDW_CONFIG_PATH``), never a ``~/.config/tldw_cli/config.toml``
    literal, and reached with NO dotted relative component -- the workspace
    root IS the config directory, so confinement alone cannot explain the
    refusal.
    """
    from tldw_chatbook import config as app_config

    config_path = app_config._get_effective_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("api_key = 'super-secret-19551'\n")

    message = _refused(
        lambda: read_file(config_path.name, workspace_root=config_path.parent),
        "fs_read(config.toml)",
    )
    assert "protected path" in message
    assert "super-secret-19551" not in message


def test_fs_read_refuses_this_apps_own_sqlite_db():
    """Reading ``chachanotes.db`` exfiltrates every conversation and note."""
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("not-a-real-sqlite-file-19551")

    message = _refused(
        lambda: read_file(db_path.name, workspace_root=db_path.parent),
        "fs_read(chachanotes.db)",
    )
    assert "protected path" in message
    assert "19551" not in message.replace("TASK-19551", "")


# ---------------------------------------------------------------------------
# Shape 2: writing the permission store -- the one-step gate bypass.
# ---------------------------------------------------------------------------


def test_fs_write_refuses_rewriting_the_permission_store():
    """``fs_write`` must not disarm the gate that authorized it."""
    store = _plant_permission_store()

    message = _refused(
        lambda: write_file(
            "mcp_permissions.json",
            '{"pwned": true}',
            workspace_root=store.parent,
        ),
        "fs_write(mcp_permissions.json)",
    )
    assert "protected path" in message
    assert store.read_text() == '{"version": 1}\n'  # untouched


def test_fs_edit_refuses_rewriting_the_permission_store():
    """Same gate, other write tool: ``fs_edit``'s exact-string replacement."""
    store = _plant_permission_store()

    message = _refused(
        lambda: edit_file(
            "mcp_permissions.json",
            '"version": 1',
            '"default_verdict": "allow"',
            workspace_root=store.parent,
        ),
        "fs_edit(mcp_permissions.json)",
    )
    assert "protected path" in message
    assert store.read_text() == '{"version": 1}\n'  # untouched


def test_fs_patch_refuses_rewriting_the_permission_store():
    """Same gate, third write tool: ``fs_patch``'s unified diff.

    ``patch_tool_impls.patch_files`` resolves every diff target through the
    SAME ``resolve_workspace_path`` choke point, so it inherits the check
    rather than re-implementing it.
    """
    store = _plant_permission_store()
    diff = (
        "--- mcp_permissions.json\n"
        "+++ mcp_permissions.json\n"
        "@@ -1 +1 @@\n"
        '-{"version": 1}\n'
        '+{"pwned": true}\n'
    )

    message = _refused(
        lambda: patch_files(diff, workspace_root=store.parent),
        "fs_patch(mcp_permissions.json)",
    )
    assert "protected path" in message
    assert store.read_text() == '{"version": 1}\n'  # untouched


def test_fs_patch_dry_run_also_refuses_the_permission_store():
    """``dry_run`` still reads the target, so it must refuse identically."""
    store = _plant_permission_store()
    diff = (
        "--- mcp_permissions.json\n"
        "+++ mcp_permissions.json\n"
        "@@ -1 +1 @@\n"
        '-{"version": 1}\n'
        '+{"pwned": true}\n'
    )

    message = _refused(
        lambda: patch_files(diff, workspace_root=store.parent, dry_run=True),
        "fs_patch(dry_run, mcp_permissions.json)",
    )
    assert "protected path" in message


# ---------------------------------------------------------------------------
# Shape 3: a dotfile-component path -- the shape allow_hidden=True permits.
# ---------------------------------------------------------------------------


def test_fs_read_refuses_dotfile_component_credential_path():
    """``~/.aws/credentials``, reached through a DOTTED relative component.

    ``resolve_workspace_path`` passes ``allow_hidden=True`` to
    ``validate_path`` (ADR-032: real workspaces need ``.git``/``.github``/
    dotfile configs), so confinement alone lets ``.aws/credentials``
    through. Only the denylist stops it -- which is precisely why this
    shape is pinned separately.
    """
    creds = _home() / ".aws" / "credentials"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text(f"[default]\n{AWS_MARKER}\n")

    message = _refused(
        lambda: read_file(".aws/credentials", workspace_root=_home()),
        "fs_read(~/.aws/credentials)",
    )
    assert "protected path" in message
    assert AWS_MARKER not in message


def test_fs_write_refuses_dotfile_component_credential_path():
    """The same dotted shape on the write side (``~/.ssh/authorized_keys``)."""
    ssh_dir = _home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)

    message = _refused(
        lambda: write_file(
            ".ssh/authorized_keys",
            "ssh-rsa ATTACKER-KEY-19551\n",
            workspace_root=_home(),
        ),
        "fs_write(~/.ssh/authorized_keys)",
    )
    assert "protected path" in message
    assert not (ssh_dir / "authorized_keys").exists()


def test_benign_workspace_dotfiles_stay_readable():
    """The allow_hidden=True decision, pinned.

    ADR-032 keeps hidden components reachable under the workspace root on
    purpose -- a coding agent that cannot read ``.github/workflows/ci.yml``
    or ``.gitignore`` is useless. The denylist is what makes that safe: it
    matches by RESOLVED ANCESTRY against real credential/state locations,
    not by "the name starts with a dot". If this test ever has to change,
    the allow_hidden decision is being reversed and ADR-032 needs amending.
    """
    ws = _home() / "projects" / "repo"
    (ws / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (ws / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (ws / ".gitignore").write_text("*.pyc\n")

    assert "name: ci" in read_file(".github/workflows/ci.yml", workspace_root=ws)
    assert "*.pyc" in read_file(".gitignore", workspace_root=ws)
    assert ".github/" in list_directory(".", workspace_root=ws)


# ---------------------------------------------------------------------------
# The enumerating tools: fs_list / fs_glob / fs_grep.
#
# resolve_workspace_path only ever sees the ROOT for these three, so the
# choke point alone cannot cover the entries they walk -- each result must
# be filtered too, exactly as GlobFiles/GrepFiles/ListDirectoryTool do in
# the sibling family. fs_grep is the sharpest of the three: it READS every
# file it walks and prints matching lines.
# ---------------------------------------------------------------------------


def test_fs_list_hides_sensitive_directories_from_a_home_rooted_workspace():
    """The shipped-default shape: workspace root == ``$HOME``."""
    _plant_ssh_key()
    (_home() / ".aws").mkdir(parents=True, exist_ok=True)
    (_home() / "notes.txt").write_text("fine\n")

    listing = list_directory(".", workspace_root=_home())

    assert "notes.txt" in listing  # ordinary entry still listed
    assert ".ssh" not in listing
    assert ".aws" not in listing


def test_fs_list_hides_the_permission_store_from_the_user_data_dir():
    """A denylisted FILE inside a listable directory, by name alone.

    Subdirectories of ``get_user_data_dir()`` stay listable (the
    direct-child-file rule is about loose FILES) -- ``sub_dir`` below is
    the control proving the listing itself still works.
    """
    store = _plant_permission_store()
    (store.parent / "sub_dir").mkdir(exist_ok=True)

    listing = list_directory(".", workspace_root=store.parent)

    assert "sub_dir/" in listing
    assert "mcp_permissions.json" not in listing


def test_fs_grep_never_reads_denylisted_file_contents():
    _plant_ssh_key()
    (_home() / "notes.txt").write_text(f"decoy {SSH_KEY_MARKER}\n")

    out = grep_files(SSH_KEY_MARKER, workspace_root=_home())

    assert "notes.txt" in out  # ordinary file still searched
    assert ".ssh/id_rsa" not in out
    assert out.count(SSH_KEY_MARKER) == 1  # only the decoy line


def test_fs_glob_omits_denylisted_matches():
    _plant_ssh_key()
    (_home() / "keep.txt").write_text("x\n")

    out = glob_files("**/*", workspace_root=_home())

    assert "keep.txt" in out
    assert "id_rsa" not in out


# ---------------------------------------------------------------------------
# Drift pin: the two file-tool families must agree on the denylist.
# ---------------------------------------------------------------------------


def _denylisted_candidates() -> list[tuple[str, Path]]:
    """Denylisted paths, each planted as a synthetic marker file.

    The third element says whether the file's own parent directory is
    DOTTED. That matters only for the second family: its
    ``validate_path_multi`` refuses a dotted base directory outright
    (``allow_hidden`` defaults False there), so for those candidates its
    refusal comes from confinement, not the denylist -- a distinction the
    assertions below keep rather than paper over.

    That distinction is now an honest one, and TASK-19633 is what made it
    so -- the phrase used to carry a caveat, because it was describing
    residue as well as design. When TASK-19551 shipped, the second
    family's dotted-component rejection ALSO refused credential files the
    denylist did not enumerate at all (``~/.netrc``,
    ``~/.git-credentials``, ``~/.npmrc``, ``~/.pypirc``,
    ``~/.cargo/credentials.toml``, ``~/.config/gh/hosts.yml``), every one
    of which the ``fs_*`` family returned in full: for those paths the
    ``fs_*`` family was strictly the weaker of the two, by accident rather
    than by decision. TASK-19633 closed that in the shared oracle -- a
    name rule for unambiguous credential FILENAMES plus ``~/.config/gh``
    as a location -- so both families now refuse all six, and
    ``test_both_families_refuse_the_TASK_19633_credential_paths`` below
    pins it in configurations where confinement cannot be what does the
    work. What remains is only the mechanism difference described above:
    a deliberate ADR-032 policy split about DOTTED NAMES, not a gap in
    what the denylist knows.
    """
    from tldw_chatbook import config as app_config

    home = _home()
    candidates: list[tuple[str, Path, bool]] = [
        ("ssh private key", home / ".ssh" / "id_rsa", True),
        ("aws credentials", home / ".aws" / "credentials", True),
        ("gnupg secring", home / ".gnupg" / "secring.gpg", True),
        ("app config.toml", app_config._get_effective_config_path(), False),
        (
            "mcp permission store",
            app_config.get_user_data_dir() / "mcp_permissions.json",
            False,
        ),
        ("chachanotes db", app_config.get_chachanotes_db_path(), False),
    ]
    for _label, path, _dotted in candidates:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic-19551\n")
    return candidates


def test_both_file_tool_families_refuse_the_same_denylisted_paths(monkeypatch):
    """Neither family may drift from the other on the denylist.

    ``fs_*`` (``Tools/local_tool_impls.py``, workspace-root confined) and
    ``read_file``/``write_file`` (``Tools/file_operation_tools.py``,
    sandbox-root confined) are two independent file-tool families reaching
    the same filesystem for the same agent runtime. The second one enforced
    ``is_sensitive_path``; the first never called it -- that drift IS
    TASK-19551. This pins them together: for every candidate, BOTH refuse
    and neither returns the file's contents.

    Family A's refusal must specifically be the DENYLIST's ("protected
    path"): it passes ``allow_hidden=True``, and each root here is the
    file's own directory, so nothing else could refuse it.
    """
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)

    for label, path, dotted_parent in _denylisted_candidates():
        # The shared oracle both families are supposed to consult.
        assert is_sensitive_path(path), label

        # Family A: fs_read, workspace root set to the file's own directory
        # (so confinement alone cannot be what refuses it).
        fs_message = _refused(
            lambda p=path: read_file(p.name, workspace_root=p.parent),
            f"fs_read({label})",
        )
        assert "protected path" in fs_message, label
        assert "synthetic-19551" not in fs_message, label

        # Family B: read_file, sandbox root set to the same directory.
        monkeypatch.setattr(fot, "_tool_sandbox_root", lambda p=path: p.parent.resolve())
        legacy = asyncio.run(fot.ReadFileTool().execute(file_path=path.name))
        assert "error" in legacy, label
        assert "synthetic-19551" not in str(legacy), label
        if not dotted_parent:
            assert "protected path" in legacy["error"], label


# ---------------------------------------------------------------------------
# Choke-point tripwire: a new fs_* tool cannot reach the filesystem without
# passing through resolve_workspace_path.
# ---------------------------------------------------------------------------


#: Helpers in ``git_tool_impls`` that resolve their path argument through
#: ``resolve_workspace_path`` themselves, so a git tool calling one of them
#: IS on the choke point -- just one hop away. Enumerated (rather than
#: chasing the call graph) so that adding a FOURTH indirect resolver is a
#: deliberate edit here, with the same "does it call the choke point?"
#: question asked of it directly below.
_INDIRECT_CHOKE_POINT_RESOLVERS = frozenset(
    {"prepare_repository", "_prepare_for_path", "_repo_relative_path"}
)


def test_every_workspace_rooted_function_uses_the_choke_point():
    """AST tripwire, TASK-19551 AC5.

    Any module-level function in the agent file/git tool core modules that
    accepts a ``workspace_root`` is a filesystem entry point for the agent,
    and must resolve its target through ``resolve_workspace_path`` -- the
    ONE place the denylist is enforced for this family. A new tool that
    resolves its own path (``Path(workspace_root) / arg``) would reintroduce
    exactly the hole this task closed, so it fails here instead.

    ``git_tool_impls`` is covered too (fix round): its four repo-scoped
    tools reach the choke point INDIRECTLY, through the three helpers in
    ``_INDIRECT_CHOKE_POINT_RESOLVERS`` -- each of which is itself required
    below to call ``resolve_workspace_path`` directly, so the indirection is
    verified rather than assumed. Covering only two of the family's three
    modules is what let this test tick AC5 while a new git tool would have
    got zero tripwire coverage.

    NOTE what this tripwire does NOT claim: reaching the choke point proves
    a tool's PATH ARGUMENT is denylist-checked, not that its OUTPUT is
    filtered. ``git_diff`` called without a path never presents one, and
    leaks committed file CONTENT for denylisted paths -- see TASK-19632.
    """
    import ast
    import importlib

    import tldw_chatbook.Tools as tools_pkg

    # DERIVED, not hand-listed. A hardcoded module tuple is the same shape as
    # the bug this task fixed (an enforcer list that can only name what existed
    # when it was written) -- a fourth module joining the family would silently
    # get zero coverage. Discover the family instead: any Tools/ module with a
    # module-level function taking ``workspace_root`` is in it by definition.
    tools_dir = Path(tools_pkg.__file__).parent
    modules = []
    for source in sorted(tools_dir.glob("*.py")):
        if source.name == "__init__.py":
            continue
        text = source.read_text()
        if "workspace_root" not in text:
            continue
        candidate = ast.parse(text)
        takes_root = any(
            "workspace_root"
            in {
                a.arg
                for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs
            }
            for node in candidate.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if takes_root:
            modules.append(importlib.import_module(f"tldw_chatbook.Tools.{source.stem}"))

    # The three known members must be present; discovery may only ADD.
    discovered = {m.__name__.rsplit(".", 1)[-1] for m in modules}
    assert {"local_tool_impls", "patch_tool_impls", "git_tool_impls"} <= discovered, (
        "the workspace-rooted tool family lost a known member -- discovery found "
        f"{sorted(discovered)}; if a module was renamed or retired, say so here"
    )

    offenders: list[str] = []
    verified_resolvers: set[str] = set()
    for module in modules:
        tree = ast.parse(Path(module.__file__).read_text())
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            names = {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
            if "workspace_root" not in names:
                continue
            if node.name == "resolve_workspace_path":
                continue  # the choke point itself IS the enforcement
            calls = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            accepted = {"resolve_workspace_path"}
            if node.name not in _INDIRECT_CHOKE_POINT_RESOLVERS:
                # A resolver itself must reach the choke point DIRECTLY;
                # only its callers may go through it.
                accepted |= _INDIRECT_CHOKE_POINT_RESOLVERS
            if not (calls & accepted):
                offenders.append(f"{module.__name__}.{node.name}")
            if node.name in _INDIRECT_CHOKE_POINT_RESOLVERS:
                verified_resolvers.add(node.name)

    assert not offenders, (
        "these workspace-rooted functions never reach resolve_workspace_path "
        "(directly or via one of "
        f"{sorted(_INDIRECT_CHOKE_POINT_RESOLVERS)}), so they bypass the "
        f"sensitive-path denylist: {offenders}"
    )

    # Every accepted resolver must itself have been VISITED and verified above.
    # Without this, the accept-set is a hole rather than a delegation: the loop
    # only inspects functions that take a parameter literally named
    # ``workspace_root``, so a resolver spelling it differently is added to the
    # accept-set, never checked itself, and silently launders its callers past
    # the choke point (demonstrated in review with a `root`-named resolver).
    unverified = set(_INDIRECT_CHOKE_POINT_RESOLVERS) - verified_resolvers
    assert not unverified, (
        "these names are trusted in _INDIRECT_CHOKE_POINT_RESOLVERS but were "
        "never themselves inspected by the scan above -- they are an unchecked "
        "hole, not a verified delegation. A resolver must live in one of the "
        "scanned modules and take a `workspace_root` parameter so that its own "
        f"call to the choke point is proven: {sorted(unverified)}"
    )


def test_fs_family_never_creates_directories_on_the_agents_behalf():
    """Companion to the tripwire above: no ``mkdir`` in the ``fs_*`` core.

    ``WriteFileTool`` in the sibling family creates parent directories on
    request, which is why it must consult
    ``Utils.sensitive_paths.refuses_new_directory_chain`` before calling
    ``mkdir(parents=True)`` (TASK-849: otherwise an agent can plant a
    directory where this app later expects its own state FILE -- a denial
    of service). The ``fs_*`` family deliberately requires the parent
    directory to already exist, so it has no such call site;
    ``resolve_workspace_path`` still applies the guard for write intents.
    If a ``mkdir`` ever appears here, that decision is being reversed and
    the guard's placement has to be re-examined.
    """
    import ast

    from tldw_chatbook.Tools import local_tool_impls, patch_tool_impls

    for module in (local_tool_impls, patch_tool_impls):
        tree = ast.parse(Path(module.__file__).read_text())
        attr_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "mkdir" not in attr_calls, (
            f"{module.__name__} now creates directories -- it must consult "
            "Utils.sensitive_paths.refuses_new_directory_chain first"
        )
        assert "makedirs" not in attr_calls, f"{module.__name__} now creates directories"


def test_resolve_workspace_path_still_confines_and_still_allows_hidden(tmp_path):
    """The pre-existing contract is unchanged by the denylist addition."""
    ws = tmp_path / "ws"
    (ws / ".github").mkdir(parents=True)
    assert resolve_workspace_path("a/b", ws) == (ws / "a/b").resolve()
    assert resolve_workspace_path(".github", ws) == (ws / ".github").resolve()
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        resolve_workspace_path("../x", ws)


# ---------------------------------------------------------------------------
# Shape 5: writes into a repository's own `.git/` metadata (TASK-19700).
#
# Surfaced by TASK-16801's git-modes arc, where a repository-supplied
# `.git/config` or `.git/HEAD` was the precondition for FOUR proven
# data-destruction vectors (an option-shaped remote or branch name reaching
# git's argv; `remote.push`/`mirror`/`push.default` turning an ordinary push
# into a forced update or a ref deletion). Those are all fixed defensively
# inside the git engine; this is the upstream cause -- an agent that can
# write `.git/` reconfigures git for EVERY feature that shells out to it.
#
# `.git` is a dotted component under the root, which ADR-032's
# `allow_hidden=True` deliberately lets past confinement, so this depends
# entirely on the write guard.
# ---------------------------------------------------------------------------


def _plant_repo(tmp_path: Path) -> Path:
    """A workspace root that is a real-looking git repo (no git binary needed)."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return root


def test_fs_write_refuses_rewriting_git_config(tmp_path):
    root = _plant_repo(tmp_path)
    before = (root / ".git" / "config").read_text()
    message = _refused(
        lambda: write_file(
            ".git/config",
            "[remote \"--force\"]\n\turl = https://example.invalid/x.git\n",
            workspace_root=root,
        ),
        "fs_write(.git/config)",
    )
    assert ".git" in message
    assert (root / ".git" / "config").read_text() == before, "config was rewritten"


def test_fs_write_refuses_rewriting_git_head(tmp_path):
    root = _plant_repo(tmp_path)
    before = (root / ".git" / "HEAD").read_text()
    message = _refused(
        lambda: write_file(
            ".git/HEAD", "ref: refs/heads/--mirror\n", workspace_root=root
        ),
        "fs_write(.git/HEAD)",
    )
    assert ".git" in message
    assert (root / ".git" / "HEAD").read_text() == before, "HEAD was rewritten"


def test_fs_write_refuses_a_nested_path_under_git(tmp_path):
    """The guard is on ANY `.git` component, not just its direct children."""
    root = _plant_repo(tmp_path)
    (root / ".git" / "hooks").mkdir()
    _refused(
        lambda: write_file(
            ".git/hooks/pre-commit", "#!/bin/sh\nexit 1\n", workspace_root=root
        ),
        "fs_write(.git/hooks/pre-commit)",
    )
    assert not (root / ".git" / "hooks" / "pre-commit").exists()


def test_fs_write_refuses_the_dot_git_FILE_of_a_linked_worktree(tmp_path):
    """A linked worktree carries `.git` as a FILE; rewriting it redirects the
    whole repository, so the guard must not assume `.git` is a directory."""
    root = tmp_path / "linked"
    root.mkdir()
    (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    before = (root / ".git").read_text()
    _refused(
        lambda: write_file(".git", "gitdir: /attacker\n", workspace_root=root),
        "fs_write(.git as a file)",
    )
    assert (root / ".git").read_text() == before


def test_fs_edit_refuses_git_config(tmp_path):
    """The guard sits at the shared boundary, so edit is covered too."""
    root = _plant_repo(tmp_path)
    _refused(
        lambda: edit_file(
            ".git/config",
            "repositoryformatversion = 0",
            "repositoryformatversion = 0\n\tfsmonitor = /tmp/evil",
            workspace_root=root,
        ),
        "fs_edit(.git/config)",
    )
    assert "fsmonitor" not in (root / ".git" / "config").read_text()


def test_fs_patch_refuses_git_config(tmp_path):
    root = _plant_repo(tmp_path)
    # A REAL unified diff -- the "*** Begin Patch" envelope is a different
    # tool's format and this parser rejects it as `invalid_diff`, which
    # would make this test pass without ever reaching the guard.
    patch = "--- /dev/null\n+++ b/.git/hooks/pre-commit\n@@ -0,0 +1,1 @@\n+evil\n"
    try:
        result = patch_files(patch, workspace_root=root)
    except LocalToolError as exc:
        result = str(exc)
    assert "invalid_diff" not in str(result), (
        f"the patch payload never reached the guard: {result}"
    )
    assert ".git" in str(result), f"expected a .git refusal, got: {result}"
    assert not (root / ".git" / "hooks" / "pre-commit").exists(), (
        f"patch_files wrote into .git/: {result}"
    )


def test_fs_patch_control_the_same_shape_succeeds_outside_git(tmp_path):
    """Proves the patch payload above is well-formed: the identical shape
    aimed at an ordinary path must APPLY, so the refusal is the guard's
    doing and not a malformed diff."""
    root = _plant_repo(tmp_path)
    patch = "--- /dev/null\n+++ b/notes.txt\n@@ -0,0 +1,1 @@\n+hello\n"
    patch_files(patch, workspace_root=root)
    assert (root / "notes.txt").read_text() == "hello\n"


# -- the deliberate exceptions: ordinary tracked dotfiles stay writable ------


def test_fs_write_still_allows_gitignore(tmp_path):
    """`.gitignore` is an ordinary tracked file; refusing it would break
    normal coding work. It only shares a name PREFIX with `.git`."""
    root = tmp_path / "repo2"
    (root / ".git").mkdir(parents=True)
    write_file(".gitignore", "build/\n", workspace_root=root)
    assert (root / ".gitignore").read_text() == "build/\n"


def test_fs_write_still_allows_gitattributes_and_github_dir(tmp_path):
    root = tmp_path / "repo3"
    (root / ".git").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)
    write_file(".gitattributes", "* text=auto\n", workspace_root=root)
    write_file(".github/workflows/ci.yml", "name: ci\n", workspace_root=root)
    assert (root / ".gitattributes").read_text() == "* text=auto\n"
    assert (root / ".github" / "workflows" / "ci.yml").read_text() == "name: ci\n"


def test_reads_under_git_are_unchanged_by_this_guard(tmp_path):
    """Scope pin: TASK-19700 governs WRITES. Read behaviour is deliberately
    untouched here so the change cannot quietly break an agent inspecting
    repository state; the read-side question is tracked separately."""
    root = _plant_repo(tmp_path)
    out = read_file(".git/HEAD", workspace_root=root)
    assert "refs/heads/main" in out


def test_fs_write_refuses_a_case_variant_of_git(tmp_path):
    """Qodo #1 (PR #1934): macOS and Windows filesystems are
    case-insensitive by default, so `.GIT/config` IS `.git/config`.

    Verified before the fix: `write_file(".GIT/config", ...)` succeeded and
    the hostile remote landed in the REAL `.git/config`.
    """
    root = _plant_repo(tmp_path)
    before = (root / ".git" / "config").read_text()
    for spelling in (".GIT/config", ".Git/config", ".gIt/HEAD"):
        _refused(
            lambda s=spelling: write_file(s, "[remote \"--force\"]\n", workspace_root=root),
            f"fs_write({spelling})",
        )
    assert (root / ".git" / "config").read_text() == before


def test_case_variants_do_not_over_refuse_gitignore(tmp_path):
    """The case-insensitive match must still be COMPONENT-exact: `.GITIGNORE`
    is an ordinary file, not repository metadata."""
    root = tmp_path / "repo_ci"
    (root / ".git").mkdir(parents=True)
    write_file(".GITIGNORE", "build/\n", workspace_root=root)
    assert (root / ".GITIGNORE").read_text() == "build/\n"


# ---------------------------------------------------------------------------
# Shape 6: case-variant spellings of a denylisted path (TASK-19800).
#
# Found while fixing TASK-19700's `.git` guard for the same weakness. macOS
# and Windows filesystems are case-insensitive by DEFAULT, and
# `Path.resolve()` does NOT canonicalise case there (it resolves symlinks
# and `..`, but preserves what the caller typed) -- so `~/.SSH/id_rsa` opens
# the same file as `~/.ssh/id_rsa` while comparing unequal to every
# denylist entry.
#
# Reproduced end-to-end before the fix against the app's OWN config file:
#   tldw_cli/config.toml -> refused
#   TLDW_CLI/config.toml -> ALLOWED, returned 32782 chars
# i.e. the user's provider API keys, read straight through the denylist by
# changing the case of one path component. The same shape reaches `~/.ssh`,
# `~/.aws` and `mcp_permissions.json` -- and bypassing that last one turns
# every `ask` into `allow`, which is the permission-gate bypass TASK-19551
# exists to prevent.
# ---------------------------------------------------------------------------


def test_is_sensitive_path_matches_case_variants_of_a_denied_directory():
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    _plant_ssh_key()
    home = _home()
    for spelling in (".ssh", ".SSH", ".Ssh", ".sSh"):
        assert is_sensitive_path(home / spelling / "id_rsa"), (
            f"~/{spelling}/id_rsa must be denied: on a case-insensitive "
            f"filesystem it IS ~/.ssh/id_rsa"
        )


def test_is_sensitive_path_matches_a_case_variant_of_a_denied_file():
    from tldw_chatbook import config as app_config
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    real = app_config._get_effective_config_path()
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("[API]\nopenai_api_key = \"SYNTHETIC-19800\"\n")
    variant = real.parent.parent / real.parent.name.upper() / real.name
    assert is_sensitive_path(real)
    assert is_sensitive_path(variant), (
        f"{variant} must be denied: it is the same file on a "
        f"case-insensitive filesystem"
    )


def test_fs_read_refuses_a_case_variant_of_this_apps_config():
    """End-to-end through the tool, not just the predicate."""
    from tldw_chatbook import config as app_config

    real = app_config._get_effective_config_path()
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("[API]\nopenai_api_key = \"SYNTHETIC-19800\"\n")
    root = real.parent.parent
    message = _refused(
        lambda: read_file(
            f"{real.parent.name.upper()}/{real.name}", workspace_root=root
        ),
        "fs_read(CASE-VARIANT config.toml)",
    )
    assert "protected path" in message
    assert "SYNTHETIC-19800" not in message


def test_case_folding_does_not_deny_an_unrelated_lookalike():
    """The match stays ancestry-based, not substring-based: `~/.sshfoo` is a
    genuinely different directory and must remain readable."""
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    home = _home()
    (home / ".sshfoo").mkdir(parents=True, exist_ok=True)
    (home / ".sshfoo" / "notes.txt").write_text("ordinary\n")
    assert not is_sensitive_path(home / ".sshfoo" / "notes.txt")
    assert not is_sensitive_path(home / ".SSHFOO" / "notes.txt")


def test_binding_gate_detects_a_case_variant_conflict():
    """Qodo #1 (PR #1936): `find_root_binding_conflict` enforces the SAME
    protected-path policy at folder-bind time, and was still comparing
    case-sensitively — so a case-variant spelling could bind a workspace
    root that overlaps a protected directory, widening tool reachability
    into it.

    Not a confinement check: this is the denylist's own overlap gate, so
    folding it fails safe (more conflicts reported) exactly as elsewhere.
    """
    from tldw_chatbook.Utils.sensitive_paths import find_root_binding_conflict

    _plant_ssh_key()
    home = _home()

    # (1) the root IS a protected directory, spelled differently
    assert find_root_binding_conflict(home / ".SSH") is not None

    # (3) the root CONTAINS a protected path, reached by a cased ancestor
    upper_home = Path(str(home).upper())
    assert find_root_binding_conflict(upper_home) is not None, (
        "a cased spelling of an ancestor still contains the protected paths"
    )


def test_binding_gate_still_allows_an_unrelated_root():
    from tldw_chatbook.Utils.sensitive_paths import find_root_binding_conflict

    plain = _home() / "projects" / "myapp"
    plain.mkdir(parents=True, exist_ok=True)
    assert find_root_binding_conflict(plain) is None


# ---------------------------------------------------------------------------
# TASK-19633: credential files the denylist did not cover.
#
# TASK-19551 kept `allow_hidden=True` (ADR-032) on the argument that "starts
# with a dot" is a name heuristic while `is_sensitive_path` answers the
# question properly, by resolved ancestry. That decision is right, but
# resolved-ancestry matching is only as strong as what the denylist knows:
# `_SENSITIVE_DIRS` listed seven locations and nothing else, so every file
# below returned its BODY through `fs_read` (measured, isolated $HOME).
#
# The fix is in the denylist's CONTENT: a NAME rule for filenames that are
# credential stores wherever they appear, plus `~/.config/gh` as a location
# where the filename (`hosts.yml`) is far too generic to be one. See
# `Utils/sensitive_paths.py`'s docstring for why both instruments exist.
# ---------------------------------------------------------------------------

CREDENTIAL_MARKER = "SYNTHETIC-NOT-A-REAL-CREDENTIAL-19633"

#: (label, path relative to $HOME, which rule refuses it). Exactly the six
#: paths measured in the task; the rule column is asserted, not decorative.
_TASK_19633_CREDENTIALS = (
    ("netrc", ".netrc", "name"),
    ("git credential store", ".git-credentials", "name"),
    ("npm auth token", ".npmrc", "name"),
    ("pypi upload credentials", ".pypirc", "name"),
    ("cargo registry token", ".cargo/credentials.toml", "name"),
    ("github cli oauth tokens", ".config/gh/hosts.yml", "location"),
)


def _plant_credential(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"machine example.invalid password {CREDENTIAL_MARKER}\n")
    return path


@pytest.mark.parametrize(
    ("label", "relative", "rule"),
    _TASK_19633_CREDENTIALS,
    ids=[relative for _label, relative, _rule in _TASK_19633_CREDENTIALS],
)
def test_fs_read_refuses_each_TASK_19633_credential_path(label, relative, rule):
    """One born-red case per added path shape.

    The workspace root is the file's OWN directory, so the relative
    portion carries no dotted component and ``allow_hidden=True`` is not
    what lets it through -- only the denylist can refuse it. At base every
    one of these returned the file's body.
    """
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    path = _plant_credential(_home() / relative)
    assert is_sensitive_path(path), f"{label}: the shared oracle must refuse it"

    message = _refused(
        lambda: read_file(path.name, workspace_root=path.parent),
        f"fs_read({relative})",
    )
    assert "protected path" in message, label
    assert CREDENTIAL_MARKER not in message, label


def test_the_name_rule_and_the_location_rule_each_do_their_own_work():
    """Which instrument refuses which path, asserted rather than assumed.

    A name-rule path must still be refused after being MOVED (that is the
    whole reason a name rule was chosen for it); a location-rule path must
    NOT be, because its filename is generic on purpose -- ``hosts.yml`` is
    just as often an Ansible inventory, and refusing it everywhere would
    be exactly the over-refusal the module docstring rules out.
    """
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    elsewhere = _home() / "projects" / "repo"
    for label, relative, rule in _TASK_19633_CREDENTIALS:
        moved = _plant_credential(elsewhere / Path(relative).name)
        assert is_sensitive_path(moved) is (rule == "name"), (
            f"{label}: expected the {rule} rule to decide, and it did not"
        )
        moved.unlink()


def test_the_name_rule_is_case_insensitive():
    """macOS and Windows filesystems are case-insensitive.

    ``.NETRC`` opens the same file an exact-case comparison would wave
    through, so the check is case-folded.
    """
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    assert is_sensitive_path(_plant_credential(_home() / ".NETRC"))
    assert is_sensitive_path(_plant_credential(_home() / "Credentials.TOML"))


def test_the_name_rule_does_not_refuse_near_misses_or_containers():
    """The cost of a name rule is over-refusal; this is where it is bounded.

    A container DIRECTORY carrying one of the names stays reachable (the
    same ``is_dir()`` gate the direct-child-file rule uses), and names that
    merely resemble a credential file are ordinary files. ``.env`` is the
    deliberate omission recorded in the module docstring.
    """
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    workspace = _home() / "projects" / "repo"
    (workspace / "credentials").mkdir(parents=True)
    (workspace / "credentials" / "README.md").write_text("docs\n")
    for name in (".npmrc.example", "netrc", "my.netrc", ".env", "credentials.json"):
        (workspace / name).write_text("ordinary\n")

    assert not is_sensitive_path(workspace / "credentials")
    assert "README.md" in list_directory("credentials", workspace_root=workspace)
    for name in (".npmrc.example", "netrc", "my.netrc", ".env", "credentials.json"):
        assert "ordinary" in read_file(name, workspace_root=workspace), name


def test_both_families_refuse_the_TASK_19633_credential_paths(monkeypatch):
    """AC1: BOTH families, in configurations confinement cannot explain.

    Constructing that configuration takes deliberate care, because the two
    families disagree about dotted names on purpose:

    * The NAME-rule paths are planted at a NON-dotted location under a
      NON-dotted root (``projects/repo/_netrc``, ``credentials.toml``,
      ``credentials``). Nothing in either family's confinement can object,
      so a refusal is the denylist's -- and that a moved credential is
      still refused is the name rule's whole point.
    * The LOCATION-rule path keeps its location, since that is what
      identifies it -- but rooted at ``~/.config/gh`` the relative portion
      is a plain ``hosts.yml`` and the root's own basename (``gh``) is not
      dotted either, so family 1's ``allow_hidden=False`` has nothing to
      reject.
    """
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)

    neutral = _home() / "projects" / "repo"
    cases: list[tuple[str, Path]] = [
        (f"moved name-rule credential ({name})", _plant_credential(neutral / name))
        for name in ("_netrc", "credentials.toml", "credentials")
    ]
    cases.append(
        ("github cli oauth tokens", _plant_credential(_home() / ".config/gh/hosts.yml"))
    )

    for label, path in cases:
        relative_parts = (path.name,)
        assert not any(part.startswith(".") for part in relative_parts), label
        assert not path.parent.name.startswith("."), label

        # Family A: fs_* (allow_hidden=True), root = the file's directory.
        message = _refused(
            lambda p=path: read_file(p.name, workspace_root=p.parent),
            f"fs_read({label})",
        )
        assert "protected path" in message, label
        assert CREDENTIAL_MARKER not in message, label

        # Family B: file_operation_tools, sandbox root = the same directory.
        monkeypatch.setattr(fot, "_tool_sandbox_root", lambda p=path: p.parent.resolve())
        legacy = asyncio.run(fot.ReadFileTool().execute(file_path=path.name))
        assert "error" in legacy, label
        assert "protected path" in legacy["error"], label
        assert CREDENTIAL_MARKER not in str(legacy), label


def test_family_b_hides_the_dotted_credential_names_by_denylist_not_confinement(
    monkeypatch,
):
    """The four inherently-dotted names, for family B, with the denylist alone.

    ``.netrc`` cannot be spelled without a dot, so ``ReadFileTool`` will
    always refuse it at confinement and prove nothing about the denylist.
    ``ListDirectoryTool`` with ``include_hidden=True`` is the seam where
    that is not true: it lists dotted entries by request, and the ONLY
    thing that can then withhold one is ``is_sensitive_path``. Before
    TASK-19633 all four were listed by name; ``.gitignore`` is the control
    proving ``include_hidden`` is honored and the listing itself works.
    """
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)

    workspace = _home() / "projects" / "repo"
    dotted = (".netrc", ".git-credentials", ".npmrc", ".pypirc")
    for name in dotted:
        _plant_credential(workspace / name)
    (workspace / ".gitignore").write_text("*.pyc\n")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: workspace.resolve())
    listing = asyncio.run(
        fot.ListDirectoryTool().execute(directory_path=".", include_hidden=True)
    )

    names = {entry["name"] for entry in listing.get("entries", [])}
    assert ".gitignore" in names, (
        f"the listing itself is broken or include_hidden was ignored: {listing}"
    )
    for name in dotted:
        assert name not in names, (
            f"ListDirectoryTool(include_hidden=True) disclosed {name}: {sorted(names)}"
        )


def test_fs_grep_never_reads_a_name_rule_credential(monkeypatch):
    """The sharpest ``fs_*`` tool, on the newly-covered shape.

    ``grep_files`` READS every file it walks and prints matching lines, so
    a home-rooted workspace and a pattern as bland as ``password`` emitted
    ``~/.netrc``'s body verbatim into the transcript.
    """
    _plant_credential(_home() / ".netrc")
    (_home() / "notes.txt").write_text(f"decoy password {CREDENTIAL_MARKER}\n")

    out = grep_files("password", workspace_root=_home())

    assert "notes.txt" in out
    assert ".netrc" not in out
    assert out.count(CREDENTIAL_MARKER) == 1  # the decoy line only


def test_the_name_rule_uses_the_modules_one_folding_rule(monkeypatch):
    """TASK-19633 must not open a SECOND normalization path (TASK-19800).

    ``_compare_key`` is this module's single definition of how two path
    spellings are compared. If the name rule ever grows its own
    ``.casefold()`` beside it, the two can drift -- so this pins the
    dependency by MUTATION: with folding removed from ``_compare_key``
    (and only from there), a case-variant credential name must stop being
    refused. It does not, if the name rule folds on its own.

    The mutation keeps ``_compare_key``'s SHAPE (a tuple of components) so
    the other rules keep working normally and this test isolates the one
    it is about.
    """
    import tldw_chatbook.Utils.sensitive_paths as sp

    variant = _plant_credential(_home() / "projects" / "repo" / "Credentials.TOML")
    assert sp.is_sensitive_path(variant)

    monkeypatch.setattr(sp, "_compare_key", lambda p: tuple(p.parts))
    assert not sp.is_sensitive_path(variant), (
        "the name rule still folded case after folding was removed from "
        "_compare_key, so it normalizes independently of the module's one "
        "folding rule"
    )
