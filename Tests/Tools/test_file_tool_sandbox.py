import asyncio
import os
from pathlib import Path

import pytest

from tldw_chatbook.Tools import file_operation_tools as fot
from tldw_chatbook.Tools import workspace_file_roots as wfr


@pytest.fixture(autouse=True)
def _sandbox_only_roots(monkeypatch):
    """Force ``allowed_file_roots`` to fall back to sandbox-only.

    These tests are about the sensitive-path denylist, not about the
    workspace-folder-roots feature (see ``Tests/Tools/test_file_tools_
    workspace_roots.py``). Raising from the registry factory drives
    ``allowed_file_roots`` into its own documented fail-safe fallback
    (``(sandbox_root,)``), which also sidesteps the process-wide
    ``_default_registry_instance`` cache in ``workspace_file_roots.py``
    picking up state from a different test's isolated HOME.
    """

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)


def test_sandbox_root_is_real_dir_not_literal(monkeypatch, tmp_path):
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path / "tool_sandbox"))
    root = fot._tool_sandbox_root()
    assert root == (tmp_path / "tool_sandbox").resolve()
    assert root.is_dir()  # created
    assert root.name != "file" and root.name != "directory"


def test_read_file_rejects_traversal_outside_sandbox(monkeypatch, tmp_path):
    sandbox = tmp_path / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)
    # write a secret OUTSIDE the sandbox
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    result = asyncio.run(fot.ReadFileTool().execute(file_path="../secret.txt"))
    assert result.get("success") is False or "error" in result  # rejected, not leaked
    assert "top secret" not in str(result)


def test_read_file_reads_inside_sandbox(monkeypatch, tmp_path):
    sandbox = tmp_path / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "hello.txt").write_text("inside content")
    result = asyncio.run(fot.ReadFileTool().execute(file_path="hello.txt"))
    assert "inside content" in str(result)


def test_read_write_work_under_dotted_ancestor_root(monkeypatch, tmp_path):
    """Regression test: the real default sandbox root lives under a dotted
    ancestor (``get_user_data_dir()/"tool_sandbox"`` resolves to something
    like ``~/.local/share/tldw_cli/.../tool_sandbox``). validate_path must
    not reject in-sandbox paths just because the sandbox *root* itself sits
    under a dotted directory component -- only a dotted component in the
    user-supplied (relative) portion of the path should be rejected.
    """
    # Simulate the real default: sandbox under a `.local`-style dotted ancestor.
    sandbox = tmp_path / ".local" / "share" / "tldw" / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)

    # A normal in-sandbox write must succeed even though the sandbox root
    # itself is nested under dotted directories.
    write_result = asyncio.run(
        fot.WriteFileTool().execute(file_path="note.txt", content="hi there")
    )
    assert "error" not in write_result
    assert (sandbox / "note.txt").read_text() == "hi there"

    # And the corresponding read must succeed too.
    read_result = asyncio.run(fot.ReadFileTool().execute(file_path="note.txt"))
    assert "error" not in read_result
    assert read_result.get("content") == "hi there"

    # A user-supplied path containing a dotted component is STILL rejected --
    # the security property is preserved, only the sandbox's own location is
    # exempted from the hidden-file check.
    bad_result = asyncio.run(fot.ReadFileTool().execute(file_path=".secret"))
    assert "error" in bad_result


# ---------------------------------------------------------------------------
# Sensitive-path coverage at the tool boundary (denylist-unreachable fix).
#
# `Tests/conftest.py`'s autouse `isolate_test_environment` fixture already
# redirects HOME to a per-test tmp directory, so `os.environ["HOME"]` below
# is that isolated home, not the real machine's.
#
# Each scenario deliberately configures `file_sandbox_root` to CONTAIN the
# denied path with NO dotted path component in the *relative* portion
# between root and target. That is the one configuration in which the bug
# is observable: `Utils.path_validation.validate_path`'s own hidden-file
# check would otherwise reject any ".dotted" component in the relative path
# on its own, masking whether `Utils.sensitive_paths.is_sensitive_path` ever
# ran at all. Isolating that confound is what makes these tests actually
# prove the fix rather than incidentally pass for an unrelated reason.
# ---------------------------------------------------------------------------


def test_read_file_refuses_sensitive_file_even_when_sandbox_root_contains_it(
    monkeypatch,
):
    """CRITICAL fix: read_file must not read config.toml just because a
    widened sandbox root happens to contain it.

    Written directly to the REAL effective config path
    (``config._get_effective_config_path()``, which honors
    ``TLDW_CONFIG_PATH``) rather than a hardcoded
    ``~/.config/tldw_cli/config.toml`` literal -- that literal is not
    necessarily where config.toml actually lives (Finding 3, substrate
    review); this project's own test suite is a live example, since
    ``Tests/conftest.py`` sets ``TLDW_CONFIG_PATH`` to a per-test path
    elsewhere entirely.
    """
    from tldw_chatbook import config as app_config

    config_path = app_config._get_effective_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("api_key = 'super-secret'\n")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: config_path.parent.resolve())

    result = asyncio.run(fot.ReadFileTool().execute(file_path=config_path.name))

    assert "error" in result
    assert "super-secret" not in str(result)


def test_write_file_refuses_to_overwrite_sensitive_file(monkeypatch):
    """CRITICAL fix: write_file must not overwrite mcp_permissions.json --
    that would be a one-step bypass of the whole permission gate -- just
    because the configured sandbox root contains it.

    Written to the REAL resolved location
    (``get_user_data_dir() / "mcp_permissions.json"``), not the
    ``~/.config/tldw_cli/`` literal the app never actually used (Finding 1,
    substrate review): the permission store is built under
    ``get_user_data_dir()`` (see ``MCP.unified_control_plane_service``'s
    ``permission_store`` property and ``app.py``'s ``LocalMCPStore``
    construction).
    """
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    target = user_data_dir / "mcp_permissions.json"
    target.write_text('{"version": 1}')

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: user_data_dir.resolve())

    result = asyncio.run(
        fot.WriteFileTool().execute(
            file_path="mcp_permissions.json", content='{"pwned": true}'
        )
    )

    assert "error" in result
    assert target.read_text() == '{"version": 1}'  # untouched


def test_list_directory_refuses_sensitive_directory_as_its_own_target(monkeypatch):
    """CRITICAL fix: list_directory's own top-level target was never checked
    against the denylist -- only the recursive-descent containment guard
    was, and that only gates whether a walk *descends into a subdirectory*.
    Root is set directly to ``~/.config/gcloud`` so the listing target is
    the sandbox root itself (``directory_path="."``).
    """
    home = Path(os.environ["HOME"])
    sandbox = home / ".config" / "gcloud"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "credentials.db").write_text("oauth-refresh-token-marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    result = asyncio.run(fot.ListDirectoryTool().execute(directory_path="."))

    assert "error" in result
    assert "credentials.db" not in str(result)


def test_read_file_refuses_this_apps_own_sqlite_db(monkeypatch):
    """Finding 2: this app's SQLite DBs live under ``get_user_data_dir()``,
    a SIBLING of ``~/.config/tldw_cli`` (not beneath it), so the static
    credential-path list cannot express their location. Uses the real,
    unmocked ``config.get_chachanotes_db_path()`` so this proves resolution
    happens through the app's own accessor, not a path hardcoded in the
    test (or in ``sensitive_paths.py``).
    """
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("not a real sqlite file, just a marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: db_path.parent.resolve())

    result = asyncio.run(fot.ReadFileTool().execute(file_path=db_path.name))

    assert "error" in result
    assert "marker" not in str(result)


def test_list_directory_filters_sensitive_entries_from_recursive_listing(monkeypatch):
    """Finding 5 (pre-merge review): a recursive listing correctly refuses to
    DESCEND into a sensitive directory (~/.ssh, ~/.aws), but individual
    sensitive FILES sitting inside an otherwise-ordinary, listable ancestor
    were still emitted by name and size -- this app's own
    ``mcp_permissions.json`` (under ``get_user_data_dir()``, its REAL
    location -- see Finding 1, substrate review) and the ChaChaNotes DB
    (plus its ``-wal`` sidecar), also under ``get_user_data_dir()``.
    Contents never leaked, only the listing row itself; this closes that.
    """
    home = Path(os.environ["HOME"])

    # A non-sensitive sibling directory, NOT get_user_data_dir(), so it
    # stays unaffected by Finding 2's "direct child of the user data dir"
    # rule and keeps testing what it always did: an ordinary file must
    # still be listed alongside sensitive ones.
    config_dir = home / ".config" / "tldw_cli"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "ordinary.toml").write_text("fine = true\n")

    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    (user_data_dir / "mcp_permissions.json").write_text('{"version": 1}')

    db_path = app_config.get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("marker")
    wal_path = db_path.with_name(db_path.name + "-wal")
    wal_path.write_text("wal-marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: home.resolve())

    result = asyncio.run(
        fot.ListDirectoryTool().execute(
            directory_path=".", recursive=True, include_hidden=True, max_depth=5
        )
    )

    names = {e["name"] for e in result["entries"]}
    assert "mcp_permissions.json" not in names
    assert db_path.name not in names
    assert wal_path.name not in names
    assert "ordinary.toml" in names  # non-sensitive sibling still listed


def test_read_file_refuses_wal_sidecar_of_this_apps_own_sqlite_db(monkeypatch):
    """The gap this task closes: WAL mode writes ``<name>.db-wal`` next to
    the database, carrying the same class of recent data, but exact-path
    equality on the ``.db`` file alone never reached it. Under the exact
    misconfiguration the DB denial exists to guard against -- a sandbox
    root widened to contain the user data directory -- ``read_file`` could
    still recover recent rows from the sidecar even though the ``.db`` path
    itself was refused. This is the tool-level observable property; the
    unit test on ``is_sensitive_path`` alone would not have caught the
    original gap because it never exercises the sandboxed ``execute()``
    boundary.
    """
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    wal_path = db_path.with_name(db_path.name + "-wal")
    wal_path.write_text("recent-uncommitted-row-marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: db_path.parent.resolve())

    result = asyncio.run(fot.ReadFileTool().execute(file_path=wal_path.name))

    assert "error" in result
    assert "recent-uncommitted-row-marker" not in str(result)


def test_read_file_refuses_chroma_vector_store_file_even_when_sandbox_root_contains_it(
    monkeypatch,
):
    """TASK-848 AC#1: ``chromadb/chroma.sqlite3`` -- plaintext chunks of the
    same conversations and notes ``ChaChaNotes.db`` protects -- must not
    become readable just because a widened sandbox root happens to contain
    it. ``chromadb`` itself stays reachable as a container (per the
    existing-directory exemption); only the file directly inside it is
    refused.
    """
    from tldw_chatbook.RAG_Search.simplified.config import (
        default_chroma_persist_directory,
    )

    chroma_dir = default_chroma_persist_directory()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "chroma.sqlite3").write_text("plaintext vector chunk marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: chroma_dir.resolve())

    result = asyncio.run(fot.ReadFileTool().execute(file_path="chroma.sqlite3"))

    assert "error" in result
    assert "plaintext vector chunk marker" not in str(result)


def test_write_file_refuses_to_plant_a_directory_shadowing_a_state_file(monkeypatch):
    """TASK-849: an agent must not be able to create a directory named
    after a state file this app has not created yet -- verified before this
    fix: ``search_history.db/`` (``get_user_data_dir() / "search_history.db"``,
    see ``UI/Views/RAGSearch/search_rag_window.py``) was permitted, and the
    app's own later ``sqlite3.connect(...)`` on that path would fail
    outright (a denial of service, not a disclosure).

    Widened sandbox root = ``get_user_data_dir()`` itself, the same
    reachability precondition the other TASK-848 gaps share -- under the
    shipped default (``tool_sandbox``, a SIBLING of ``search_history.db``)
    this path is never reachable at all.
    """
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    user_data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: user_data_dir.resolve())

    result = asyncio.run(
        fot.WriteFileTool().execute(
            file_path="search_history.db/note.txt",
            content="hello",
            create_directories=True,
        )
    )

    assert "error" in result
    assert not (user_data_dir / "search_history.db").exists()


def test_write_file_still_creates_legitimate_new_nested_directories(monkeypatch):
    """The TASK-849 fix must not block ordinary nested directory creation:
    an agent creating a brand-new subdirectory INSIDE an EXISTING container
    (``tool_sandbox``) must still succeed end to end, even under the same
    widened-root configuration the collision test above uses.
    """
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    sandbox_root = user_data_dir / "tool_sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: user_data_dir.resolve())

    result = asyncio.run(
        fot.WriteFileTool().execute(
            file_path="tool_sandbox/brand_new_subdir/note.txt",
            content="hello",
            create_directories=True,
        )
    )

    assert "error" not in result, result
    assert (sandbox_root / "brand_new_subdir" / "note.txt").read_text() == "hello"


# ---------------------------------------------------------------------------
# Finding 2's guardrail must not break the shipped DEFAULT configuration:
# the default sandbox root is `get_user_data_dir() / "tool_sandbox"`, a
# DIRECTORY nested directly inside the same user data directory whose other
# direct children are now refused. Every test above monkeypatches
# `_tool_sandbox_root` or `_resolve_sandbox_config`; this one does neither,
# so it exercises the REAL default resolution path end to end.
# ---------------------------------------------------------------------------


def test_default_sandbox_configuration_still_works_end_to_end():
    """No monkeypatching of the sandbox root/config at all -- proves the
    real shipped default (``get_user_data_dir() / "tool_sandbox"``) is
    still fully usable after Finding 2's "refuse direct-child files of the
    user data directory" rule. If that rule accidentally caught the
    sandbox root itself (a directory, not a file, so it should not), every
    one of these calls would fail instead.
    """
    from tldw_chatbook import config as app_config

    expected_root = (app_config.get_user_data_dir() / "tool_sandbox").resolve()

    write_result = asyncio.run(
        fot.WriteFileTool().execute(file_path="hello.txt", content="hi there")
    )
    assert "error" not in write_result, write_result
    assert write_result["file_path"] == str(expected_root / "hello.txt")

    read_result = asyncio.run(fot.ReadFileTool().execute(file_path="hello.txt"))
    assert "error" not in read_result, read_result
    assert read_result["content"] == "hi there"

    list_result = asyncio.run(fot.ListDirectoryTool().execute(directory_path="."))
    assert "error" not in list_result, list_result
    assert any(e["name"] == "hello.txt" for e in list_result["entries"])

    glob_result = asyncio.run(fot.GlobFiles().execute(pattern="*.txt"))
    assert "error" not in glob_result, glob_result
    assert any(Path(p).name == "hello.txt" for p in glob_result["matches"])

    grep_result = asyncio.run(fot.GrepFiles().execute(pattern="hi there"))
    assert "error" not in grep_result, grep_result
    assert any(m["path"].endswith("hello.txt") for m in grep_result["matches"])
