from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Utils.input_validation import ValidationError


def _clone(repo_url, ref=None):
    return LocalMediaReadingService._clone_git_repository(
        repo_url, Path("/tmp/checkout_target"), ref=ref
    )


@pytest.mark.parametrize("repo_url", [
    "ext::sh -c 'touch /tmp/pwn'",
    "file:///etc/passwd",
    "-upload-pack=/bin/sh",
    "git@github.com:owner/repo.git",
])
def test_malicious_repo_url_rejected_before_subprocess(repo_url):
    with patch("subprocess.run") as mock_run:
        with pytest.raises((ValidationError, ValueError, RuntimeError)):
            _clone(repo_url)
        mock_run.assert_not_called()


def test_malicious_ref_rejected_before_subprocess():
    with patch("subprocess.run") as mock_run:
        with pytest.raises((ValidationError, ValueError, RuntimeError)):
            _clone("https://github.com/owner/repo.git", ref="--upload-pack=/bin/sh")
        mock_run.assert_not_called()


def test_valid_clone_uses_separator_and_restricted_env():
    ok = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=ok) as mock_run:
        _clone("https://github.com/owner/repo.git", ref="main")
    assert mock_run.call_count == 1
    argv = mock_run.call_args[0][0]
    # "--" separator immediately precedes the repo URL
    assert "--" in argv
    sep = argv.index("--")
    assert argv[sep + 1] == "https://github.com/owner/repo.git"
    assert "--branch" in argv and argv[argv.index("--branch") + 1] == "main"
    env = mock_run.call_args[1]["env"]
    assert env["GIT_ALLOW_PROTOCOL"] == "https:ssh"
    assert env["GIT_PROTOCOL_FROM_USER"] == "0"
