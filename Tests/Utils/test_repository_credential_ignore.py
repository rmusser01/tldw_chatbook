from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _is_ignored(relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative_path],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise RuntimeError(
        f"git check-ignore failed for {relative_path!r} with exit code "
        f"{completed.returncode}: {completed.stderr.strip()}"
    )


def test_exact_root_credential_filenames_are_ignored():
    assert _is_ignored("openai-api-key.txt")
    assert _is_ignored("moonshot-api-key.txt")


def test_same_filenames_in_subdirectories_are_not_covered_by_root_guard():
    assert not _is_ignored("nested/openai-api-key.txt")
    assert not _is_ignored("nested/moonshot-api-key.txt")


def test_check_ignore_fatal_error_cannot_satisfy_not_ignored(monkeypatch):
    def fatal_check_ignore(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=128,
            stdout="",
            stderr="fatal: simulated check-ignore failure",
        )

    monkeypatch.setattr(subprocess, "run", fatal_check_ignore)

    with pytest.raises(RuntimeError, match="exit code 128.*simulated check-ignore"):
        assert not _is_ignored("nested/sentinel.txt")
