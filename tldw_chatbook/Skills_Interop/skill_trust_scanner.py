"""Deterministic local skill directory scanning for trust verification."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath

from .skill_trust_crypto import sha256_hex
from .skill_trust_models import SkillDirectorySnapshot, SkillFileFingerprint

_SKILL_FILENAME = "SKILL.md"
SUPPORTING_JUNK_DIRS = frozenset({".git", ".github", ".hg", ".svn", "node_modules", "__pycache__"})
SUPPORTING_JUNK_FILES = frozenset({".DS_Store", "Thumbs.db"})
SUPPORTING_JUNK_SUFFIXES = (".pyc", ".pyo", "~", ".tmp", ".swp", ".part")


def _is_junk(name: str) -> bool:
    return name in SUPPORTING_JUNK_FILES or name.lower().endswith(SUPPORTING_JUNK_SUFFIXES)


def _validate_relative_path(relative_path: str) -> bool:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from ..tldw_api.skills_schemas import validate_supporting_file_path

    try:
        validate_supporting_file_path(relative_path)
        return True
    except ValueError:
        return False


def scan_skill_directory(skill_name: str, skill_dir: Path) -> SkillDirectorySnapshot:
    """Return a deterministic trust snapshot for a skill directory tree.

    Walks ``skill_dir`` recursively (never following symlinks), pruning
    VCS/OS/build junk, sorting by POSIX relative path. Text files are decoded
    into ``text_files``; binaries are fingerprinted (``file_type=
    "supporting_binary"``) but not decoded. Symlinks and files failing path
    validation are recorded in ``unsupported_paths`` (they are NOT trust
    material) but never raise here.

    Args:
        skill_name: Local skill name represented by the directory.
        skill_dir: Directory to scan.

    Returns:
        Snapshot with fingerprints (sorted by relative path), decoded text
        files, and unsupported path names.
    """
    fingerprints: list[SkillFileFingerprint] = []
    text_files: dict[str, str] = {}
    unsupported_paths: list[str] = []

    collected: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(skill_dir, followlinks=False):
        # Prune junk directories in-place so os.walk does not descend them.
        # Symlinked directories must NOT be followed, but must still surface as
        # unsupported (never silently vanish): route them through ``collected``
        # so the shared ``path.is_symlink()`` check records them, and drop them
        # from ``dirs`` so the walk never descends into them.
        kept_dirs: list[str] = []
        for name in dirs:
            dir_abs = Path(root) / name
            # A symlink is suspicious regardless of its name: check is_symlink()
            # BEFORE the junk-name filter so a symlink named like a junk dir
            # (e.g. node_modules, .git) still surfaces in unsupported_paths
            # instead of being silently pruned. Junk-name pruning applies only
            # to REAL directories.
            if dir_abs.is_symlink():
                rel = PurePosixPath(dir_abs.relative_to(skill_dir).as_posix())
                collected.append((str(rel), dir_abs))
                continue
            if name in SUPPORTING_JUNK_DIRS:
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in files:
            if _is_junk(name):
                continue
            abs_path = Path(root) / name
            rel = PurePosixPath(abs_path.relative_to(skill_dir).as_posix())
            collected.append((str(rel), abs_path))
    collected.sort(key=lambda item: item[0])

    for relative_path, path in collected:
        is_body = relative_path == _SKILL_FILENAME
        # The reserved SKILL.md basename check inside validate_supporting_file_path
        # exists to reject *shadow* SKILL.md files nested under subdirectories; the
        # real top-level body file must bypass it rather than be misclassified as
        # unsupported.
        if path.is_symlink() or (not is_body and not _validate_relative_path(relative_path)):
            unsupported_paths.append(relative_path)
            continue
        try:
            if not path.is_file():
                unsupported_paths.append(relative_path)
                continue
            raw = path.read_bytes()
            # Read the exec bit inside the same guarded region so a concurrent
            # deletion between read_bytes() and stat() routes to unsupported_paths
            # rather than raising (honors the never-raise contract).
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
        except OSError:
            unsupported_paths.append(relative_path)
            continue

        is_binary = b"\x00" in raw
        text: str | None = None
        if not is_binary:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                is_binary = True

        if is_body:
            file_type = "skill"
        elif is_binary:
            file_type = "supporting_binary"
        else:
            file_type = "supporting_text"

        fingerprints.append(
            SkillFileFingerprint(
                relative_path=relative_path,
                file_type=file_type,
                byte_length=len(raw),
                sha256=sha256_hex(raw),
                executable=executable,
            )
        )
        if text is not None:
            text_files[relative_path] = text

    return SkillDirectorySnapshot(
        skill_name=skill_name,
        fingerprints=tuple(fingerprints),
        text_files=text_files,
        unsupported_paths=tuple(unsupported_paths),
    )
