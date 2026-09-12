"""Pure, bounded classification for local and archived skill packages."""

from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path, PurePosixPath

from ..tldw_api.skills_schemas import (
    MAX_SUPPORTING_FILE_BYTES,
    MAX_SUPPORTING_FILES_COUNT,
    MAX_SUPPORTING_FILES_TOTAL_BYTES,
    MAX_SUPPORTING_FILE_PATH_DEPTH,
    MAX_SUPPORTING_FILE_PATH_LEN,
)


class SkillPackageKind(StrEnum):
    """Stable package outcomes shared by local and remote import paths."""

    ROOT_SKILL = "root_skill"
    MULTI_SKILL_REPOSITORY = "multi_skill_repository"
    FRAMEWORK_REPOSITORY = "framework_repository"
    MALFORMED_OR_UNSUPPORTED = "malformed_or_unsupported"
    FETCH_OR_AUTH_FAILURE = "fetch_or_auth_failure"


@dataclass(frozen=True)
class SkillPackageInspection:
    """A display-safe package classification; it never owns package bytes."""

    kind: SkillPackageKind
    candidates: tuple[str, ...] = ()
    message: str = ""
    recovery_actions: tuple[str, ...] = ()


FRAMEWORK_MESSAGE = (
    "This repository is a framework, not an installable Codex skill."
)
FRAMEWORK_RECOVERY_ACTIONS = (
    "Choose a repository subdirectory that contains SKILL.md.",
    "Use its project instructions when that is the intended integration.",
    "Use the framework's external CLI outside Chatbook.",
    "Create a separately reviewed wrapper skill.",
)
_MALFORMED_MESSAGE = "That package is malformed or unsupported."
_CANDIDATE_SCAN_DEPTH = 3
_CANDIDATE_LIMIT = 20


def _outcome(
    candidates: set[str],
    *,
    repository_source: bool,
    nonempty: bool,
    requested_candidate: str | None = None,
) -> SkillPackageInspection:
    ordered = tuple(sorted(candidates))
    if requested_candidate is not None:
        if requested_candidate in candidates:
            return SkillPackageInspection(
                SkillPackageKind.ROOT_SKILL, (requested_candidate,)
            )
        return SkillPackageInspection(
            SkillPackageKind.MALFORMED_OR_UNSUPPORTED,
            message="No installable skill was found at that subdirectory.",
        )
    if "" in candidates:
        return SkillPackageInspection(SkillPackageKind.ROOT_SKILL, ("",))
    if len(ordered) == 1:
        return SkillPackageInspection(SkillPackageKind.ROOT_SKILL, ordered)
    if ordered:
        return SkillPackageInspection(
            SkillPackageKind.MULTI_SKILL_REPOSITORY,
            ordered[:_CANDIDATE_LIMIT],
            "Choose one installable skill.",
        )
    if repository_source and nonempty:
        return SkillPackageInspection(
            SkillPackageKind.FRAMEWORK_REPOSITORY,
            message=FRAMEWORK_MESSAGE,
            recovery_actions=FRAMEWORK_RECOVERY_ACTIONS,
        )
    return SkillPackageInspection(
        SkillPackageKind.MALFORMED_OR_UNSUPPORTED,
        message=_MALFORMED_MESSAGE,
    )


def _candidate_for_skill_path(relative: PurePosixPath) -> str | None:
    if relative.name != "SKILL.md":
        return None
    parent = relative.parent
    if str(parent) == ".":
        return ""
    if len(parent.parts) > _CANDIDATE_SCAN_DEPTH:
        return None
    return parent.as_posix()


def inspect_skill_directory(path: Path) -> SkillPackageInspection:
    """Classify one bounded local import candidate without importing it."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        return _outcome(set(), repository_source=False, nonempty=False)

    candidates: set[str] = set()
    nonempty = False
    seen = 0
    try:
        for entry in path.rglob("*"):
            seen += 1
            if seen > MAX_SUPPORTING_FILES_COUNT + 1:
                return _outcome(set(), repository_source=False, nonempty=False)
            nonempty = True
            relative = PurePosixPath(entry.relative_to(path).as_posix())
            if len(relative.parts) > MAX_SUPPORTING_FILE_PATH_DEPTH + 1:
                continue
            if entry.name == "SKILL.md" and entry.is_symlink():
                return _outcome(set(), repository_source=False, nonempty=False)
            if not entry.is_file():
                continue
            candidate = _candidate_for_skill_path(relative)
            if candidate is not None:
                candidates.add(candidate)
    except OSError:
        return _outcome(set(), repository_source=False, nonempty=False)
    return _outcome(candidates, repository_source=True, nonempty=nonempty)


def _safe_archive_path(name: str) -> PurePosixPath | None:
    if not name or "\\" in name or name.startswith("/"):
        return None
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    if len(name.encode("utf-8")) > MAX_SUPPORTING_FILE_PATH_LEN:
        return None
    return path


def inspect_skill_zip(
    data: bytes,
    *,
    repository_source: bool,
    requested_candidate: str | None = None,
) -> SkillPackageInspection:
    """Classify one bounded archive from central-directory metadata."""
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError):
        return _outcome(set(), repository_source=False, nonempty=False)

    with archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not members or len(members) > MAX_SUPPORTING_FILES_COUNT + 1:
            return _outcome(set(), repository_source=False, nonempty=False)
        paths: list[PurePosixPath] = []
        declared_total = 0
        for member in members:
            path = _safe_archive_path(member.filename)
            if path is None:
                return _outcome(set(), repository_source=False, nonempty=False)
            if member.file_size > MAX_SUPPORTING_FILE_BYTES:
                return _outcome(set(), repository_source=False, nonempty=False)
            declared_total += member.file_size
            if declared_total > MAX_SUPPORTING_FILES_TOTAL_BYTES:
                return _outcome(set(), repository_source=False, nonempty=False)
            mode = (member.external_attr >> 16) & 0xFFFF
            if path.name == "SKILL.md" and stat.S_ISLNK(mode):
                return _outcome(set(), repository_source=False, nonempty=False)
            paths.append(path)

    tops = {path.parts[0] for path in paths if len(path.parts) > 1}
    loose = [path for path in paths if len(path.parts) == 1]
    wrapper = next(iter(tops)) if len(tops) == 1 and not loose else ""
    candidates: set[str] = set()
    for path in paths:
        relative = PurePosixPath(*path.parts[1:]) if wrapper else path
        candidate = _candidate_for_skill_path(relative)
        if candidate is not None:
            candidates.add(candidate)
    return _outcome(
        candidates,
        repository_source=repository_source,
        nonempty=True,
        requested_candidate=requested_candidate,
    )
