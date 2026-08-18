"""Pure discovery of a project-local .SKILLS/ folder (spec 2026-08-17 §5).

No side effects, no execution, no trust decisions: this module only
enumerates candidate skills so a prompt can offer them. Hardened against
untrusted repos: symlink refusal, entry/read caps, top-level-only scan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_PROJECT_SKILLS_DIRNAMES = (".SKILLS", ".skills")
MAX_DISCOVERED_ENTRIES = 50
FRONTMATTER_READ_CAP_BYTES = 65536


@dataclass(frozen=True)
class ProjectSkillEntry:
    name: str
    kind: str  # "directory" | "file"
    path: Path
    description: str
    status: str  # "ok" | "invalid"
    reason: str = ""


@dataclass(frozen=True)
class ProjectSkillsDiscovery:
    root: Path
    skills_dir: Path
    entries: tuple[ProjectSkillEntry, ...]
    skipped: tuple[tuple[str, str], ...]  # (entry name, reason)
    truncated: int
    fingerprint: str


def find_project_skills_dir(root: Path) -> Path | None:
    """First non-symlinked, existing .SKILLS/.skills dir in root.

    ``.SKILLS`` is preferred over ``.skills`` purely by check order: the
    names are tried in that order and the function returns on the first
    candidate that exists, is a directory, and is not itself a symlink.
    """
    for name in _PROJECT_SKILLS_DIRNAMES:
        candidate = root / name
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
        except OSError:
            continue
        return candidate
    return None


def find_project_dir_with_skills(start: Path) -> Path | None:
    """cwd plus bounded ancestor walk (spec §5.4, decision #7).

    Checks each directory from ``start`` upward; a directory containing
    ``.git`` is the last one checked (the project root); ``$HOME`` and the
    filesystem root are never checked and end the walk. Traversal itself
    walks unresolved parents, but each stop-check resolves ``current``
    first, so a symlinked ancestor that points at (or through) ``$HOME``
    still stops the walk instead of silently crossing the boundary.
    """
    try:
        home = Path.home().resolve()
    except OSError:
        home = None
    current = start
    while True:
        try:
            resolved_current = current.resolve()
        except OSError:
            return None
        if resolved_current == home or resolved_current == Path(resolved_current.anchor):
            return None
        if find_project_skills_dir(current) is not None:
            return current
        if (current / ".git").exists():
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _entry_for(name: str, kind: str, path: Path, body: Path) -> ProjectSkillEntry:
    # Same normalization gate the importer applies -- pre-checking here turns
    # a late import failure into a labeled row (spec §5.2).
    from tldw_chatbook.tldw_api.skills_schemas import _normalize_skill_name

    try:
        normalized = _normalize_skill_name(name)
    except Exception:
        return ProjectSkillEntry(
            name=name,
            kind=kind,
            path=path,
            description="",
            status="invalid",
            reason="name must be lowercase-kebab",
        )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    try:
        with body.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(FRONTMATTER_READ_CAP_BYTES)
    except OSError:
        return ProjectSkillEntry(
            name=normalized,
            kind=kind,
            path=path,
            description="",
            status="invalid",
            reason="unreadable",
        )
    metadata, _ = LocalSkillsService._parse_front_matter(head)
    description = str(metadata.get("description") or "")[:200]
    return ProjectSkillEntry(
        name=normalized, kind=kind, path=path, description=description, status="ok"
    )


def _fingerprint(entries: list[ProjectSkillEntry]) -> str:
    # Stat the recognized skill FILE (spec §5.2), not the containing
    # directory: on POSIX a directory's mtime/size don't change when a file
    # inside it is edited in place, so stat'ing entry.path for a directory
    # kind would make in-place SKILL.md edits invisible to the fingerprint.
    digest = hashlib.sha256()
    for entry in entries:
        body = entry.path / "SKILL.md" if entry.kind == "directory" else entry.path
        try:
            stat = body.stat()
            digest.update(
                f"{entry.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
        except OSError:
            digest.update(f"{entry.name}|?\n".encode())
    return digest.hexdigest()


def discover_project_skills(root: Path) -> ProjectSkillsDiscovery | None:
    skills_dir = find_project_skills_dir(root)
    if skills_dir is None:
        return None
    entries: list[ProjectSkillEntry] = []
    skipped: list[tuple[str, str]] = []
    truncated = 0
    try:
        children = sorted(skills_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return None
    for child in children:
        if len(entries) >= MAX_DISCOVERED_ENTRIES:
            truncated += 1
            continue
        if child.is_symlink():
            skipped.append((child.name, "symlink"))
            continue
        if child.is_dir():
            body = child / "SKILL.md"
            if body.is_symlink() or not body.is_file():
                skipped.append((child.name, "no SKILL.md"))
                continue
            entries.append(_entry_for(child.name, "directory", child, body))
        elif child.is_file() and child.suffix.lower() == ".md":
            entries.append(_entry_for(child.stem, "file", child, child))
        else:
            skipped.append((child.name, "not a skill"))
    return ProjectSkillsDiscovery(
        root=root.resolve(),
        skills_dir=skills_dir,
        entries=tuple(entries),
        skipped=tuple(skipped),
        truncated=truncated,
        fingerprint=_fingerprint(entries),
    )
