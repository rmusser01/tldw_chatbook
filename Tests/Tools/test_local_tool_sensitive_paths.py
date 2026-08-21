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
    refusal comes from confinement, not the denylist -- an honest
    distinction the assertions below keep rather than paper over.
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


def test_every_workspace_rooted_function_uses_the_choke_point():
    """AST tripwire, TASK-19551 AC5.

    Any module-level function in the ``fs_*`` core modules that accepts a
    ``workspace_root`` is a filesystem entry point for the agent, and must
    resolve its target through ``resolve_workspace_path`` -- the ONE place
    the denylist is enforced for this family. A new tool that resolves its
    own path (``Path(workspace_root) / arg``) would reintroduce exactly the
    hole this task closed, so it fails here instead.
    """
    import ast

    from tldw_chatbook.Tools import local_tool_impls, patch_tool_impls

    offenders: list[str] = []
    for module in (local_tool_impls, patch_tool_impls):
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
            if "resolve_workspace_path" not in calls:
                offenders.append(f"{module.__name__}.{node.name}")

    assert not offenders, (
        "these workspace-rooted functions never call resolve_workspace_path, "
        f"so they bypass the sensitive-path denylist: {offenders}"
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
