from pathlib import Path

import pytest

from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path


@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/id_rsa",
        "~/.aws/credentials",
        "~/.gnupg/secring.gpg",
        "~/.config/tldw_cli/config.toml",
        "~/.config/tldw_cli/mcp_permissions.json",
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


def test_ordinary_file_next_to_a_sensitive_db_is_still_allowed():
    """Coverage is per-file, not a directory-wide refusal.

    Marking the whole ``get_user_data_dir()`` tree sensitive would also
    blanket the default file-tool sandbox root nested inside it
    (``<user data dir>/tool_sandbox``), breaking ordinary reads/writes under
    the shipped default. Only the specific known database files are
    refused.
    """
    from tldw_chatbook import config as app_config

    sibling = app_config.get_chachanotes_db_path().parent / "notes.md"
    assert not is_sensitive_path(sibling)


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
    """
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()

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
