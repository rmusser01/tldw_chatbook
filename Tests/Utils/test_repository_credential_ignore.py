from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _is_ignored(relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative_path],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def test_exact_root_credential_filenames_are_ignored():
    assert _is_ignored("openai-api-key.txt")
    assert _is_ignored("moonshot-api-key.txt")


def test_same_filenames_in_subdirectories_are_not_covered_by_root_guard():
    assert not _is_ignored("nested/openai-api-key.txt")
    assert not _is_ignored("nested/moonshot-api-key.txt")
