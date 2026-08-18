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
    """First non-symlinked .SKILLS/.skills dir in root, deduped by resolved path."""
    seen: set[Path] = set()
    for name in _PROJECT_SKILLS_DIRNAMES:
        candidate = root / name
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        return candidate
    return None


def find_project_dir_with_skills(start: Path) -> Path | None:
    """cwd plus bounded ancestor walk (spec §5.4, decision #7).

    Checks each directory from ``start`` upward; a directory containing
    ``.git`` is the last one checked (the project root); ``$HOME`` and the
    filesystem root are never checked and end the walk.
    """
    try:
        home = Path.home().resolve()
    except OSError:
        home = None
    current = start
    while True:
        if current == home or current == Path(current.anchor):
            return None
        if find_project_skills_dir(current) is not None:
            return current
        if (current / ".git").exists():
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _raw_description(head: str) -> str:
    """Fallback description extraction for front matter that fails strict YAML.

    ``LocalSkillsService._parse_front_matter`` parses the whole front-matter
    block as YAML and silently returns no metadata on a ``YAMLError`` (e.g. a
    description containing ``[bracketed]`` text, which YAML reads as a flow
    sequence). That is the right behavior for real skill authoring, but a
    discovery scan over an untrusted project must not let a merely-odd
    description vanish into an empty string -- it should show up as plain
    preview text (escaping is the UI's job, not this module's). This walks
    the same front-matter block literally, line by line, with no YAML
    parsing at all.
    """
    from tldw_chatbook.Skills_Interop.local_skills_service import (
        _FRONT_MATTER_PATTERN,
    )

    match = _FRONT_MATTER_PATTERN.match(head)
    if match is None:
        return ""
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("description:"):
            continue
        value = stripped.split(":", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        return value
    return ""


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
    description = str(metadata.get("description") or "")
    if not description:
        description = _raw_description(head)
    description = description[:200]
    return ProjectSkillEntry(
        name=normalized, kind=kind, path=path, description=description, status="ok"
    )


def _fingerprint(entries: list[ProjectSkillEntry]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        try:
            stat = entry.path.stat()
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
