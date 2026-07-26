import asyncio
import os
from pathlib import Path

import pytest

from tldw_chatbook.Tools import file_operation_tools as fot


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
    """
    home = Path(os.environ["HOME"])
    sandbox = home / ".config" / "tldw_cli"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "config.toml").write_text("api_key = 'super-secret'\n")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    result = asyncio.run(fot.ReadFileTool().execute(file_path="config.toml"))

    assert "error" in result
    assert "super-secret" not in str(result)


def test_write_file_refuses_to_overwrite_sensitive_file(monkeypatch):
    """CRITICAL fix: write_file must not overwrite mcp_permissions.json --
    that would be a one-step bypass of the whole permission gate -- just
    because the configured sandbox root contains it.
    """
    home = Path(os.environ["HOME"])
    sandbox = home / ".config" / "tldw_cli"
    sandbox.mkdir(parents=True, exist_ok=True)
    target = sandbox / "mcp_permissions.json"
    target.write_text('{"version": 1}')

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    result = asyncio.run(
        fot.WriteFileTool().execute(
            file_path="mcp_permissions.json", content='{"pwned": true}'
        )
    )

    assert "error" in result
    assert target.read_text() == '{"version": 1}'  # untouched


def test_list_directory_refuses_sensitive_directory_as_its_own_target(monkeypatch):
    """CRITICAL fix: list_directory's own top-level target was never checked
    against the denylist -- only ``is_within``'s recursive-descent guard
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
    ``mcp_permissions.json`` under ``~/.config/tldw_cli`` and the
    ChaChaNotes DB (plus its ``-wal`` sidecar) under
    ``~/.local/share/tldw_cli/<user>``. Contents never leaked, only the
    listing row itself; this closes that.
    """
    home = Path(os.environ["HOME"])

    config_dir = home / ".config" / "tldw_cli"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "mcp_permissions.json").write_text('{"version": 1}')
    (config_dir / "ordinary.toml").write_text("fine = true\n")

    from tldw_chatbook import config as app_config

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
