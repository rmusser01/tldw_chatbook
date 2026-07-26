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
