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
    """read_file is untagged and therefore silent.

    An unconfined read is a zero-prompt path from a credential file into a
    persisted transcript that may be sent to any provider. run_command
    reuses this same list (spec 8.4).
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
