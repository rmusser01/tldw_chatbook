"""Compute the exact non-evidence source identity for speculative voice."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH_LIST = Path(__file__).with_name("speculative_voice_source_paths.txt")
_EXCLUDED_EXACT = {
    "Docs/Development/TTS/speculative-voice-qualification.md",
    "tldw_chatbook/Audio/voice_qualification_manifest.json",
    "tldw_chatbook/Audio/voice_build_identity.json",
}
_EXCLUDED_PREFIXES = (
    "Artifacts/voice_qualification/",
    ".git/",
)
_EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}


class VoiceSourceDigestError(RuntimeError):
    """Raised when source identity cannot be proven from a clean Git tree."""


def compute_voice_source_digest(*, root: Path, path_list: Path) -> str:
    """Hash sorted ``relative-path NUL file-sha256 LF`` records."""

    repository = root.resolve()
    list_path = path_list.resolve()
    try:
        list_relative = list_path.relative_to(repository).as_posix()
    except ValueError as exc:
        raise VoiceSourceDigestError("path list must be inside the repository") from exc
    try:
        lines = list_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise VoiceSourceDigestError(
            "source path list is missing or unreadable"
        ) from exc

    paths: list[str] = []
    for line_number, raw in enumerate(lines, start=1):
        candidate = raw.strip()
        if not candidate or candidate.startswith("#"):
            continue
        relative = _validated_relative_path(candidate, line_number=line_number)
        _reject_excluded_path(relative)
        paths.append(relative)
    if not paths:
        raise VoiceSourceDigestError("source path list is empty")
    if len(set(paths)) != len(paths):
        raise VoiceSourceDigestError("source path list contains a duplicate")
    if list_relative not in paths:
        raise VoiceSourceDigestError("source path list must include itself")

    sorted_paths = sorted(paths)
    sources: list[tuple[str, Path]] = []
    for relative in sorted_paths:
        source = repository / relative
        if not source.is_file() or source.is_symlink():
            raise VoiceSourceDigestError(f"listed source is missing: {relative}")
        sources.append((relative, source))

    _require_tracked_clean(repository, sorted_paths)
    records = [
        (relative, hashlib.sha256(source.read_bytes()).hexdigest())
        for relative, source in sources
    ]
    # Recheck after reading so an edit racing the digest cannot certify dirty bytes.
    _require_tracked_clean(repository, sorted_paths)

    digest = hashlib.sha256()
    for relative, file_digest in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_relative_path(candidate: str, *, line_number: int) -> str:
    path = PurePosixPath(candidate)
    windows_path = PureWindowsPath(candidate)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or "\\" in candidate
        or candidate != path.as_posix()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or "\x00" in candidate
    ):
        raise VoiceSourceDigestError(f"invalid source path on line {line_number}")
    return path.as_posix()


def _reject_excluded_path(relative: str) -> None:
    parts = PurePosixPath(relative).parts
    if (
        relative in _EXCLUDED_EXACT
        or relative.startswith(_EXCLUDED_PREFIXES)
        or any(part in _EXCLUDED_PARTS for part in parts)
        or relative.endswith((".spdx.json", ".intoto.jsonl", ".sigstore.json"))
    ):
        raise VoiceSourceDigestError(
            f"evidence or authority path is excluded from source identity: {relative}"
        )


def _git_output(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise VoiceSourceDigestError("could not inspect source repository state")
    return result.stdout


def _decode_git_path(value: bytes) -> str:
    return value.decode("utf-8", errors="surrogateescape")


def _dirty_paths(status: bytes) -> set[str]:
    fields = status.split(b"\0")
    dirty: set[str] = set()
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        state = field[:2]
        dirty.add(_decode_git_path(field[3:]))
        if b"R" in state or b"C" in state:
            if index < len(fields) and fields[index]:
                dirty.add(_decode_git_path(fields[index]))
            index += 1
    return dirty


def _require_tracked_clean(root: Path, paths: list[str]) -> None:
    top_level = Path(
        _git_output(root, "rev-parse", "--show-toplevel")
        .decode("utf-8", errors="strict")
        .strip()
    ).resolve()
    if top_level != root:
        raise VoiceSourceDigestError("source root must be the Git repository root")

    index_records = {
        _decode_git_path(value[2:]): value[:1]
        for value in _git_output(
            root,
            "ls-files",
            "--cached",
            "-v",
            "-z",
        ).split(b"\0")
        if len(value) >= 3 and value[1:2] == b" "
    }
    for relative in paths:
        tag = index_records.get(relative)
        if tag is None:
            raise VoiceSourceDigestError(f"listed source is untracked: {relative}")
        if tag != b"H":
            raise VoiceSourceDigestError(
                f"listed source has a hidden Git index flag: {relative}"
            )

    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    changed = set(paths) & _dirty_paths(status)
    if changed:
        raise VoiceSourceDigestError(f"listed source is dirty: {sorted(changed)[0]}")


def main(argv: list[str] | None = None) -> int:
    """Print one lowercase SHA-256 for the clean listed source tree."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--path-list", type=Path, default=_DEFAULT_PATH_LIST)
    args = parser.parse_args(argv)
    try:
        digest = compute_voice_source_digest(
            root=args.root,
            path_list=args.path_list,
        )
    except VoiceSourceDigestError as exc:
        parser.error(str(exc))
    print(digest)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by release commands
    raise SystemExit(main())
