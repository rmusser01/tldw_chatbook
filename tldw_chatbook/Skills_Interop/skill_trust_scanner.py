"""Deterministic local skill directory scanning for trust verification."""

from __future__ import annotations

from pathlib import Path

from ..tldw_api.skills_schemas import SUPPORTING_FILE_NAME_PATTERN
from .skill_trust_crypto import sha256_hex
from .skill_trust_models import SkillDirectorySnapshot, SkillFileFingerprint


_SKILL_FILENAME = "SKILL.md"
_TEMP_SUFFIXES = (".tmp", ".swp", ".part")


def _is_supported_filename(filename: str) -> bool:
    if filename == _SKILL_FILENAME:
        return True
    normalized_filename = filename.lower()
    if normalized_filename == _SKILL_FILENAME.lower():
        return False
    if normalized_filename.endswith(_TEMP_SUFFIXES):
        return False
    return bool(SUPPORTING_FILE_NAME_PATTERN.fullmatch(filename))


def scan_skill_directory(skill_name: str, skill_dir: Path) -> SkillDirectorySnapshot:
    """Return a deterministic trust snapshot for immediate files in a skill directory.

    Args:
        skill_name: Local skill name represented by the directory.
        skill_dir: Directory to scan. Only immediate child files are considered.

    Returns:
        Snapshot containing trusted file fingerprints, decoded text, and unsupported
        path names that must be excluded from trust material.
    """

    fingerprints: list[SkillFileFingerprint] = []
    text_files: dict[str, str] = {}
    unsupported_paths: list[str] = []

    for path in sorted(skill_dir.iterdir(), key=lambda child: child.name):
        relative_path = path.name
        if path.is_symlink() or not _is_supported_filename(relative_path):
            unsupported_paths.append(relative_path)
            continue

        try:
            if not path.is_file():
                unsupported_paths.append(relative_path)
                continue
            raw = path.read_bytes()
        except OSError:
            unsupported_paths.append(relative_path)
            continue

        if b"\x00" in raw:
            unsupported_paths.append(relative_path)
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            unsupported_paths.append(relative_path)
            continue

        fingerprints.append(
            SkillFileFingerprint(
                relative_path=relative_path,
                file_type="skill" if relative_path == _SKILL_FILENAME else "supporting_text",
                byte_length=len(raw),
                sha256=sha256_hex(raw),
            )
        )
        text_files[relative_path] = text

    return SkillDirectorySnapshot(
        skill_name=skill_name,
        fingerprints=tuple(fingerprints),
        text_files=text_files,
        unsupported_paths=tuple(unsupported_paths),
    )
