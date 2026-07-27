from pathlib import Path

import pytest

from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path


@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/id_rsa",
        "~/.aws/credentials",
        "~/.gnupg/secring.gpg",
    ],
)
def test_credential_and_app_state_paths_are_refused(path):
    """read_file carries only the ``reads`` risk tag, which floors to
    ``ask`` rather than blocking outright -- this denylist is the backstop
    that keeps a private key from ever reaching a persisted transcript that
    may be sent to any provider, independent of the permission gate.
    """
    assert is_sensitive_path(Path(path).expanduser())


def test_ordinary_paths_are_allowed(tmp_path):
    assert not is_sensitive_path(tmp_path / "notes.md")


def test_config_toml_is_refused_via_the_actually_used_path(monkeypatch):
    """Finding 3 (substrate review): the app's own config.toml location
    honors ``TLDW_CONFIG_PATH`` (``config._get_effective_config_path()``),
    so a hardcoded ``~/.config/tldw_cli/config.toml`` literal misses the
    file actually holding the user's API keys whenever that override is
    set -- as it always is throughout this project's own test suite (see
    ``Tests/conftest.py``'s autouse environment-isolation fixture). This
    derives the path the SAME way the app does, then re-derives it again
    after retargeting the override, so drift in either direction would
    fail this test.
    """
    from tldw_chatbook import config as app_config

    # The path already in effect via this test's isolated TLDW_CONFIG_PATH.
    assert is_sensitive_path(app_config._get_effective_config_path())


def test_config_toml_override_is_followed_when_retargeted(tmp_path, monkeypatch):
    """A DIFFERENT ``TLDW_CONFIG_PATH`` must be tracked on the very next
    call, not just the one already active for this test process -- proving
    resolution happens at call time, not from a value baked in earlier.
    """
    from tldw_chatbook import config as app_config

    retargeted = tmp_path / "elsewhere" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(retargeted))

    assert app_config._get_effective_config_path() == retargeted.resolve()
    assert is_sensitive_path(retargeted)


def test_mcp_permission_store_is_refused_via_the_actually_used_path():
    """Finding 1 (CRITICAL, substrate review): the permission store's real
    path was never ``~/.config/tldw_cli/mcp_permissions.json`` on any real
    install -- the app builds it under ``get_user_data_dir()`` (see
    ``MCP.unified_control_plane_service``'s ``permission_store`` property:
    ``Path(store.path).with_name("mcp_permissions.json")``, where
    ``store.path`` is the ``LocalMCPStore`` path ``app.py`` constructs as
    ``get_user_data_dir() / "local_mcp_store.json"``). Derives the path the
    SAME way the app does rather than asserting a literal, so this can't
    silently drift back to matching nothing the way the original bug did.
    """
    from tldw_chatbook import config as app_config
    from tldw_chatbook.MCP.local_store import LocalMCPStore

    store = LocalMCPStore(app_config.get_user_data_dir() / "local_mcp_store.json")
    permissions_path = Path(store.path).with_name("mcp_permissions.json")

    assert is_sensitive_path(permissions_path)


def test_mcp_permission_store_companions_are_refused():
    """The execution-audit log and the server-definitions store are built
    the identical ``Path(...).with_name(...)`` way from the same base path
    as the permission store and carry the same class of gate-relevant
    state (server definitions + env, and the decision audit trail).
    """
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()

    assert is_sensitive_path(user_data_dir / "local_mcp_store.json")
    assert is_sensitive_path(user_data_dir / "mcp_execution_log.jsonl")


def test_matching_is_by_resolved_ancestry_not_substring(tmp_path):
    """`~/.sshfoo` is not `~/.ssh`."""
    assert not is_sensitive_path(Path("~/.sshfoo/file").expanduser())


def test_symlink_into_a_sensitive_dir_cannot_smuggle_a_path_past_the_check(
    tmp_path, monkeypatch
):
    """A symlink pointing INTO ``~/.ssh`` must still resolve as sensitive.

    An innocent-looking path is refused whenever following it lands inside a
    denylisted directory -- comparison happens after ``Path.resolve()``, not
    on the literal string the caller passed in.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    real_ssh = tmp_path / ".ssh"
    real_ssh.mkdir()
    (real_ssh / "id_rsa").write_text("fake-key-material")

    innocent_looking = tmp_path / "harmless.txt"
    innocent_looking.symlink_to(real_ssh / "id_rsa")

    assert is_sensitive_path(innocent_looking)


def test_this_apps_own_sqlite_dbs_are_refused():
    """Finding 2: the app's SQLite DBs live under
    ``config.get_user_data_dir()`` -- a SIBLING of ``~/.config/tldw_cli``
    (not beneath it) -- so the static ``_SENSITIVE_DIRS``/``_SENSITIVE_FILES``
    entries cannot express their location. Resolution goes through the
    app's own accessors (``config.get_*_db_path``), not a hardcoded path.

    `Tests/conftest.py`'s autouse environment-isolation fixture already
    redirects HOME (and hence ``get_user_data_dir()``) to a per-test tmp
    directory, so this does not touch the real user's data.
    """
    from tldw_chatbook import config as app_config

    for accessor_name in (
        "get_chachanotes_db_path",
        "get_prompts_db_path",
        "get_media_db_path",
    ):
        db_path = getattr(app_config, accessor_name)()
        assert is_sensitive_path(db_path), f"{accessor_name}() should be sensitive"


def test_ordinary_file_inside_the_default_sandbox_subdirectory_is_still_allowed():
    """Finding 2 (substrate review) denies files sitting DIRECTLY inside
    ``get_user_data_dir()``, but that must not blanket the whole tree:
    the default file-tool sandbox root nested inside it
    (``<user data dir>/tool_sandbox``) is a DIRECTORY, not a direct-child
    file, so an ordinary file placed inside IT stays reachable -- this is
    what keeps the shipped default sandbox usable at all.
    """
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    sandbox_root = user_data_dir / "tool_sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    nested = sandbox_root / "notes.md"

    assert not is_sensitive_path(nested)


def test_arbitrary_direct_child_file_of_user_data_dir_is_now_refused():
    """Finding 2 (substrate review): a live enumeration of
    ``get_user_data_dir()`` found several state files ALLOWED purely
    because nothing enumerated them by name (``agent_runs.db``,
    ``evals.db``, ``local_mcp_store.json``, ``tldw_cli_app.log``, ...).
    Rather than extend the enumeration (which will always trail whatever
    the app creates there next), any FILE sitting directly in the user
    data directory is refused -- proven here with a name that is
    deliberately NOT one of the enumerated DBs or MCP files, showing this
    is a rule, not a list.
    """
    from tldw_chatbook import config as app_config

    user_data_dir = app_config.get_user_data_dir()
    arbitrary_file = user_data_dir / "some_future_state_file.db"

    # Refused even before the file exists (a write_file target), and once
    # it does.
    assert is_sensitive_path(arbitrary_file)
    arbitrary_file.write_text("marker")
    assert is_sensitive_path(arbitrary_file)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_db_sidecar_files_are_refused(suffix):
    """Several of this app's DBs run WAL mode, which writes ``-wal``/``-shm``
    sidecars next to the ``.db`` file (``-journal`` under the default
    rollback-journal mode). Each carries the same class of recent data as
    the database itself, so exact-equality matching on the ``.db`` path
    alone left them unreached -- ``chachanotes.db-wal`` is not equal to
    ``chachanotes.db``. Under a sandbox root widened to contain the user
    data directory (the exact misconfiguration the DB denial guards
    against), an unfixed ``read_file`` could still recover recent rows from
    the sidecar even though the ``.db`` path itself was refused.
    """
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    sidecar = db_path.with_name(db_path.name + suffix)
    assert is_sensitive_path(sidecar)


def test_db_sidecar_matching_is_exact_not_a_loose_prefix(tmp_path, monkeypatch):
    """A file that merely *starts with* a DB's name is a different file.

    ``chachanotes.db.backup-2026`` and ``chachanotes.db2`` both begin with
    ``chachanotes.db`` but are not one of the three sidecar names this
    module constructs, and must stay allowed. Over-denying them would be
    harmless in itself, but it would signal the match is a loose prefix
    rather than the exact sidecar-name construction the design calls for.

    The DB accessor is monkeypatched to a ``tmp_path`` location OUTSIDE
    ``get_user_data_dir()`` so this stays a clean test of sidecar-name
    exactness alone -- inside the user data directory, Finding 2's
    direct-child-file rule would ALSO refuse these lookalikes (for an
    unrelated reason), which would mask a regression in the exact-match
    logic this test exists to pin.
    """
    from tldw_chatbook import config as app_config

    db_path = tmp_path / "elsewhere" / "tldw_chatbook_ChaChaNotes.db"
    monkeypatch.setattr(app_config, "get_chachanotes_db_path", lambda: db_path)

    lookalike_backup = db_path.with_name(db_path.name + ".backup-2026")
    lookalike_numbered = db_path.with_name(db_path.name + "2")

    assert not is_sensitive_path(lookalike_backup)
    assert not is_sensitive_path(lookalike_numbered)


def test_all_eleven_db_accessors_resolve_to_sensitive_paths():
    """Full coverage of dev's ``_DB_PATH_ACCESSOR_NAMES`` list -- every
    accessor `Utils.sensitive_paths` enumerates must itself be refused,
    not just the three ChaChaNotes/Prompts/Media DBs covered above.
    """
    from tldw_chatbook import config as app_config
    from tldw_chatbook.Utils.sensitive_paths import _DB_PATH_ACCESSOR_NAMES

    for accessor_name in _DB_PATH_ACCESSOR_NAMES:
        accessor = getattr(app_config, accessor_name)
        db_path = accessor()
        assert is_sensitive_path(db_path), f"{accessor_name}() should be sensitive"


def test_unresolvable_path_fails_closed(monkeypatch):
    """A path that cannot be resolved (e.g. ``Path.resolve`` raising) must
    be treated as sensitive rather than silently allowed."""
    import tldw_chatbook.Utils.sensitive_paths as sensitive_paths

    def _raise(_path_str):
        return None

    monkeypatch.setattr(sensitive_paths, "_resolved", _raise)
    assert is_sensitive_path(Path("/does/not/matter"))


def test_context_reuse_matches_fresh_resolution(tmp_path):
    """A caller-supplied ``SensitivePathContext`` must agree with the
    default (``context=None``) fresh-resolution path for the same candidate."""
    from tldw_chatbook.Utils.sensitive_paths import resolve_sensitive_context

    ctx = resolve_sensitive_context()
    ordinary = tmp_path / "notes.md"
    assert is_sensitive_path(ordinary, context=ctx) == is_sensitive_path(ordinary)

    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    assert is_sensitive_path(db_path, context=ctx) == is_sensitive_path(db_path)
