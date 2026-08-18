"""Pure discovery of a project-local .SKILLS/ folder (spec 2026-08-17 §5).

No side effects, no execution, no trust decisions: this module only
enumerates candidate skills so a prompt can offer them. Hardened against
untrusted repos: symlink refusal, entry/read caps, top-level-only scan.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_PROJECT_SKILLS_DIRNAMES = (".SKILLS", ".skills")
MAX_DISCOVERED_ENTRIES = 50
FRONTMATTER_READ_CAP_BYTES = 65536
#: Raw-enumeration bound (Qodo finding 8, PR #1810): a ``.SKILLS/`` full of
#: junk files must not be fully materialized and sorted just to discover it
#: has nothing recognizable past this many children.
MAX_SCANNED_CHILDREN = 500
#: Cap on real (non-synthetic) rows recorded in ``skipped`` -- past this,
#: overflow is folded into one synthetic ``"…"`` tail row instead of growing
#: unboundedly (Qodo finding 8).
MAX_RECORDED_SKIPPED = 20


@dataclass(frozen=True)
class ProjectSkillEntry:
    """One recognized (or recognized-but-invalid) top-level child of a
    ``.SKILLS``/``.skills`` folder.

    Attributes:
        name: The skill's would-be name -- the directory name for a
            ``kind="directory"`` entry, or the file stem (``.md`` suffix
            dropped) for a ``kind="file"`` entry. Normalized via the same
            ``_normalize_skill_name`` the importer applies, so an
            ``"ok"`` entry's name is already import-ready.
        kind: ``"directory"`` (a subfolder with its own ``SKILL.md``) or
            ``"file"`` (a loose top-level ``.md`` file).
        path: Filesystem path to the entry itself -- the skill's own
            directory for ``kind="directory"``, or the ``.md`` file for
            ``kind="file"``.
        description: The skill's frontmatter ``description``, truncated to
            200 characters; empty when unavailable (unparseable frontmatter
            degrades to empty, it is never treated as an error).
        status: ``"ok"`` (importable) or ``"invalid"`` (recognized as a
            skill-shaped entry but rejected, e.g. a bad name) -- distinct
            from ``skipped`` on the discovery, which is for children that
            were never even a skill candidate.
        reason: Human-readable explanation when ``status == "invalid"``;
            empty otherwise.
    """

    name: str
    kind: str  # "directory" | "file"
    path: Path
    description: str
    status: str  # "ok" | "invalid"
    reason: str = ""


@dataclass(frozen=True)
class ProjectSkillsDiscovery:
    """Result of scanning one project's ``.SKILLS``/``.skills`` folder.

    Attributes:
        root: The scanned project directory, resolved to an absolute path
            (this is the ledger key ``ProjectSkillsPromptLedger`` records
            decisions under).
        skills_dir: The specific ``.SKILLS``/``.skills`` folder that was
            scanned (see ``find_project_skills_dir``).
        entries: Recognized top-level children, in name-sorted order, up to
            ``MAX_DISCOVERED_ENTRIES``. Includes both ``"ok"`` and
            ``"invalid"`` ``ProjectSkillEntry`` rows.
        skipped: ``(entry name, reason)`` pairs for children that were
            never even skill candidates (symlinks, a directory missing
            ``SKILL.md``, an unrecognized file), bounded to
            ``MAX_RECORDED_SKIPPED`` real rows plus at most one synthetic
            ``("…", ...)`` summary row for whatever didn't fit.
        truncated: Count of children not reflected in ``entries``/
            ``skipped`` above their caps -- bumped once when the
            recognized-entry cap (``MAX_DISCOVERED_ENTRIES``) is hit, and
            again when the raw enumeration itself hit
            ``MAX_SCANNED_CHILDREN`` before every child could even be
            considered.
        fingerprint: A hash over every recognized entry's backing file
            (name + size + mtime), used to detect "this exact discovery has
            already been shown" -- see ``should_offer_project_skills_prompt``.
    """

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
    candidate that exists, is a directory, and is not itself a symlink. If
    a lower-priority candidate ALSO exists, that is logged (spec §5.1) --
    a project with both is an easy-to-miss footgun (e.g. a rename that left
    the old casing behind), so which one silently wins should be visible
    somewhere other than "reading this function's check order".

    Args:
        root: Directory to check for a ``.SKILLS``/``.skills`` subfolder.

    Returns:
        The winning candidate's ``Path`` (unresolved, ``root / name``), or
        ``None`` when neither name exists as a real, non-symlinked
        directory, or checking either raises ``OSError``.
    """
    found: Path | None = None
    for name in _PROJECT_SKILLS_DIRNAMES:
        candidate = root / name
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
        except OSError:
            continue
        if found is None:
            found = candidate
        else:
            logger.debug(
                "project-skills: both {} and {} exist in {}; using {}, "
                "ignoring {}",
                found.name,
                candidate.name,
                root,
                found.name,
                candidate.name,
            )
    return found


def find_project_dir_with_skills(start: Path) -> Path | None:
    """cwd plus bounded ancestor walk (spec §5.4, decision #7).

    Checks each directory from ``start`` upward; a directory containing
    ``.git`` is the last one checked (the project root); ``$HOME`` and the
    filesystem root are never checked and end the walk. Traversal itself
    walks unresolved parents, but each stop-check resolves ``current``
    first, so a symlinked ancestor that points at (or through) ``$HOME``
    still stops the walk instead of silently crossing the boundary.

    Args:
        start: Directory to begin the upward walk from (typically the
            app's cwd at startup).

    Returns:
        The first directory (``start`` itself, or an ancestor) containing
        a recognizable ``.SKILLS``/``.skills`` folder per
        ``find_project_skills_dir``, or ``None`` if none is found before
        the walk hits ``$HOME``, the filesystem root, or a ``.git``
        directory (the last directory checked, since a project root is
        never expected to have its skills folder above it).
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
    """Scan ``root``'s top-level ``.SKILLS``/``.skills`` folder, if any.

    Pure and side-effect-free (spec §5.2): this only enumerates and reads
    front matter, it never imports or executes anything. The scan is
    top-level only -- a subdirectory is either a skill (contains a
    non-symlinked ``SKILL.md``) or is skipped entirely, its own contents
    never recursed into.

    Hardened against an untrusted or hostile ``.SKILLS/``:

    * Symlinked entries (and a directory whose ``SKILL.md`` is a symlink)
      are refused outright -- recorded in ``skipped``, never followed.
    * The raw ``iterdir()`` enumeration is capped at
      ``MAX_SCANNED_CHILDREN`` (500): children beyond that bound are never
      materialized, sorted, or examined at all, so a folder seeded with
      thousands of junk files can't turn a discovery scan into an
      unbounded directory walk. Within that cap, results stay
      deterministic (sorted by name) exactly as for a normal-sized folder.
    * Recognized skills are capped at ``MAX_DISCOVERED_ENTRIES`` (50);
      further valid skills beyond the cap only bump ``truncated``.
    * The ``skipped`` list itself stays bounded to
      ``MAX_RECORDED_SKIPPED`` (20) real rows -- once exceeded (from
      per-child skip reasons, an unscanned raw-enumeration remainder past
      ``MAX_SCANNED_CHILDREN``, or both), one synthetic ``("…", ...)`` row
      summarizes the rest instead of growing without bound.
    * Each entry's frontmatter read is capped at
      ``FRONTMATTER_READ_CAP_BYTES``.

    Args:
        root: Project directory to scan for a ``.SKILLS``/``.skills``
            folder (see ``find_project_skills_dir``).

    Returns:
        A ``ProjectSkillsDiscovery`` describing what was found, or ``None``
        when ``root`` has no recognizable project-skills folder, or that
        folder's own top-level enumeration fails with an ``OSError``.
    """
    skills_dir = find_project_skills_dir(root)
    if skills_dir is None:
        return None
    entries: list[ProjectSkillEntry] = []
    skipped: list[tuple[str, str]] = []
    skipped_overflow = 0
    truncated = 0
    try:
        # `+1` distinguishes "exactly the bound" from "more than the
        # bound" without a second, unbounded iterdir() pass.
        scanned = list(
            itertools.islice(skills_dir.iterdir(), MAX_SCANNED_CHILDREN + 1)
        )
    except OSError:
        return None
    scan_bound_exceeded = len(scanned) > MAX_SCANNED_CHILDREN
    if scan_bound_exceeded:
        scanned = scanned[:MAX_SCANNED_CHILDREN]
    # Sorting only the (bounded) taken slice preserves determinism for
    # every normal-sized directory -- it's only a hostile/huge directory
    # whose raw enumeration order (not sort order) decides which children
    # even get considered.
    children = sorted(scanned, key=lambda p: p.name)

    def _skip(name: str, reason: str) -> None:
        nonlocal skipped_overflow
        if len(skipped) < MAX_RECORDED_SKIPPED:
            skipped.append((name, reason))
        else:
            skipped_overflow += 1

    for child in children:
        if len(entries) >= MAX_DISCOVERED_ENTRIES:
            truncated += 1
            continue
        if child.is_symlink():
            _skip(child.name, "symlink")
            continue
        if child.is_dir():
            body = child / "SKILL.md"
            if body.is_symlink() or not body.is_file():
                _skip(child.name, "no SKILL.md")
                continue
            entries.append(_entry_for(child.name, "directory", child, body))
        elif child.is_file() and child.suffix.lower() == ".md":
            entries.append(_entry_for(child.stem, "file", child, child))
        else:
            _skip(child.name, "not a skill")

    if scan_bound_exceeded:
        truncated += 1
    if skipped_overflow and scan_bound_exceeded:
        skipped.append((
            "…",
            f"{skipped_overflow} more skipped; directory has more entries "
            "than the scan bound",
        ))
    elif scan_bound_exceeded:
        skipped.append(("…", "directory has more entries than the scan bound"))
    elif skipped_overflow:
        skipped.append(("…", f"{skipped_overflow} more skipped"))

    return ProjectSkillsDiscovery(
        root=root.resolve(),
        skills_dir=skills_dir,
        entries=tuple(entries),
        skipped=tuple(skipped),
        truncated=truncated,
        fingerprint=_fingerprint(entries),
    )
