# Project Skills (.SKILLS/) Discovery + Import (PR B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A directory containing `.SKILLS/` triggers a prompt-driven (never silent) offer to import its skills — at app startup and after workspace creation with a bound folder containing one.

**Architecture:** A pure discovery module scans one `.SKILLS/` directory under strict caps and symlink refusal; a JSON prompt ledger with content fingerprints gates re-prompting; a `ModalScreen` presents the offer and runs imports through the existing quarantine-preserving importer (`trust_approved=False` always). Startup hooks into `app.py`'s `_post_mount_setup` beside the first-run wizard; the create-modal chaining reads discoveries off `WorkspaceCreateResult`.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest + Textual pilot; no new dependencies.

**Spec:** `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` (§5, §6, §7, §8, §9 PR B)

**Depends on:** PR A merged (`tldw_chatbook/Widgets/workspace_create_modal.py` with `WorkspaceCreateResult.project_skills` and the three surface handlers).

## Global Constraints

- Work in a worktree under `<repo>/.worktrees/` branched off `origin/dev` (PR A must be on dev first). Push after every task.
- pytest is venv-only: `.venv/bin/pytest …`.
- **Security posture (spec §3, §5.2):** never execute or trust anything at discovery time; all imports pass `trust_approved=False`; refuse symlinked `.SKILLS/` dirs and symlinked entries; caps: max 50 entries, 64 KiB frontmatter read; every repo-sourced string renders with `markup=False`.
- Convention: `.SKILLS/` preferred, `.skills/` accepted, deduplicated by resolved path.
- Config kill-switch: `[skills] project_skills_prompt_enabled` (default `true`) read via `get_cli_setting("skills", "project_skills_prompt_enabled", True)`.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- File one backlog task for PR B before starting (ID sweep per `backlog/docs/lessons-backlog-hygiene.md`).

---

### Task 1: Pure discovery module

**Files:**
- Create: `tldw_chatbook/Skills_Interop/project_skills_discovery.py`
- Test: `Tests/Skills/test_project_skills_discovery.py` (new)

**Interfaces:**
- Consumes: `_normalize_skill_name` from `tldw_chatbook.tldw_api.skills_schemas`; `LocalSkillsService._parse_front_matter` (static, `Skills_Interop/local_skills_service.py:330`).
- Produces (used by Tasks 2-5): `ProjectSkillEntry` (`name, kind: "directory"|"file", path: Path, description: str, status: "ok"|"invalid", reason: str`), `ProjectSkillsDiscovery` (`root, skills_dir, entries, skipped, truncated, fingerprint`), `find_project_skills_dir(root: Path) -> Path | None`, `find_project_dir_with_skills(start: Path) -> Path | None`, `discover_project_skills(root: Path) -> ProjectSkillsDiscovery | None`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Skills/test_project_skills_discovery.py
import os
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    MAX_DISCOVERED_ENTRIES,
    discover_project_skills,
    find_project_dir_with_skills,
    find_project_skills_dir,
)


def _skill_dir(root, name, description="Does a thing."):
    d = root / ".SKILLS" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody\n",
        encoding="utf-8",
    )
    return d


def test_no_skills_dir_returns_none(tmp_path):
    assert discover_project_skills(tmp_path) is None


def test_discovers_directory_and_loose_file_skills(tmp_path):
    _skill_dir(tmp_path, "alpha-skill")
    (tmp_path / ".SKILLS" / "beta-skill.md").write_text(
        "---\ndescription: Loose one.\n---\nBody\n", encoding="utf-8"
    )
    discovery = discover_project_skills(tmp_path)
    kinds = {(e.name, e.kind, e.status) for e in discovery.entries}
    assert ("alpha-skill", "directory", "ok") in kinds
    assert ("beta-skill", "file", "ok") in kinds
    assert discovery.truncated == 0


def test_subdir_without_skill_md_is_skipped_with_reason(tmp_path):
    (tmp_path / ".SKILLS" / "not-a-skill").mkdir(parents=True)
    discovery = discover_project_skills(tmp_path)
    assert ("not-a-skill", "no SKILL.md") in discovery.skipped


def test_invalid_name_flagged_not_failed(tmp_path):
    _skill_dir(tmp_path, "My_Skill")
    discovery = discover_project_skills(tmp_path)
    entry = discovery.entries[0]
    assert entry.status == "invalid"
    assert "lowercase" in entry.reason


def test_symlinked_skills_dir_refused(tmp_path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    os.symlink(real, tmp_path / ".SKILLS")
    assert find_project_skills_dir(tmp_path) is None


def test_symlinked_entry_skipped(tmp_path):
    _skill_dir(tmp_path, "alpha-skill")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / ".SKILLS" / "sneaky")
    discovery = discover_project_skills(tmp_path)
    assert ("sneaky", "symlink") in discovery.skipped
    assert [e.name for e in discovery.entries] == ["alpha-skill"]


def test_entry_cap_reports_truncation(tmp_path):
    for i in range(MAX_DISCOVERED_ENTRIES + 3):
        _skill_dir(tmp_path, f"skill-{i:03d}")
    discovery = discover_project_skills(tmp_path)
    assert len(discovery.entries) == MAX_DISCOVERED_ENTRIES
    assert discovery.truncated == 3


def test_fingerprint_changes_when_a_skill_is_added(tmp_path):
    _skill_dir(tmp_path, "alpha-skill")
    first = discover_project_skills(tmp_path).fingerprint
    _skill_dir(tmp_path, "gamma-skill")
    assert discover_project_skills(tmp_path).fingerprint != first


def test_hostile_description_survives_as_plain_data(tmp_path):
    _skill_dir(tmp_path, "alpha-skill", description="[red]evil[/red]")
    discovery = discover_project_skills(tmp_path)
    assert discovery.entries[0].description == "[red]evil[/red]"  # escaping is UI-side


def test_ancestor_walk_finds_project_root(tmp_path):
    _skill_dir(tmp_path / "repo", "alpha-skill")
    (tmp_path / "repo" / ".git").mkdir()
    sub = tmp_path / "repo" / "src" / "deep"
    sub.mkdir(parents=True)
    assert find_project_dir_with_skills(sub) == tmp_path / "repo"


def test_ancestor_walk_stops_at_git_root_without_skills(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    sub = tmp_path / "repo" / "src"
    sub.mkdir()
    _skill_dir(tmp_path, "above-the-repo")  # must NOT be found past the .git root
    assert find_project_dir_with_skills(sub) is None


def test_ancestor_walk_stops_at_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    start = tmp_path / "sub"
    start.mkdir()
    _skill_dir(tmp_path, "in-home-itself")
    assert find_project_dir_with_skills(start) is None
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Skills/test_project_skills_discovery.py -v` — ImportError.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Skills_Interop/project_skills_discovery.py
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
```

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Skills/test_project_skills_discovery.py -v` — all PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(skills): pure .SKILLS/ project skills discovery"` (+ trailer).

---

### Task 2: Prompt ledger + gating

**Files:**
- Create: `tldw_chatbook/Skills_Interop/project_skills_prompt.py`
- Test: `Tests/Skills/test_project_skills_prompt.py` (new)

**Interfaces:**
- Produces: `ProjectSkillsPromptLedger(user_data_dir)` with `decision_for(directory: Path) -> tuple[str, str] | None` (decision, fingerprint) and `record(directory: Path, decision: str, fingerprint: str) -> None`; `should_offer_project_skills_prompt(enabled: bool, entry: tuple[str, str] | None, fingerprint: str) -> bool`. Decisions: `"imported" | "declined" | "never"`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Skills/test_project_skills_prompt.py
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    ProjectSkillsPromptLedger,
    should_offer_project_skills_prompt,
)


def test_gating_truth_table():
    assert should_offer_project_skills_prompt(False, None, "f1") is False
    assert should_offer_project_skills_prompt(True, None, "f1") is True
    assert should_offer_project_skills_prompt(True, ("never", "f0"), "f1") is False
    assert should_offer_project_skills_prompt(True, ("declined", "f1"), "f1") is False
    assert should_offer_project_skills_prompt(True, ("declined", "f0"), "f1") is True
    assert should_offer_project_skills_prompt(True, ("imported", "f0"), "f1") is True


def test_ledger_roundtrip_and_missing(tmp_path):
    ledger = ProjectSkillsPromptLedger(tmp_path)
    directory = Path("/some/project")
    assert ledger.decision_for(directory) is None
    ledger.record(directory, "declined", "f1")
    assert ledger.decision_for(directory) == ("declined", "f1")
    ledger.record(directory, "never", "f2")
    assert ledger.decision_for(directory) == ("never", "f2")


def test_ledger_survives_corrupt_file(tmp_path):
    path = tmp_path / "skills" / "project_prompts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    ledger = ProjectSkillsPromptLedger(tmp_path)
    assert ledger.decision_for(Path("/x")) is None
    ledger.record(Path("/x"), "imported", "f1")
    assert ledger.decision_for(Path("/x")) == ("imported", "f1")
```

- [ ] **Step 2: Run to verify failure** — ImportError expected.
- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Skills_Interop/project_skills_prompt.py
"""Prompt ledger + gating for .SKILLS/ import offers (spec 2026-08-17 §5.3)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_LEDGER_FILENAME = "project_prompts.json"


class ProjectSkillsPromptLedger:
    """Atomic-replace JSON ledger under <user_data_dir>/skills/."""

    def __init__(self, user_data_dir: str | Path) -> None:
        self._path = Path(user_data_dir) / "skills" / _LEDGER_FILENAME

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "entries": {}}
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return data

    def decision_for(self, directory: Path) -> tuple[str, str] | None:
        entry = self._load()["entries"].get(str(directory))
        if not isinstance(entry, dict):
            return None
        return str(entry.get("decision", "")), str(entry.get("fingerprint", ""))

    def record(self, directory: Path, decision: str, fingerprint: str) -> None:
        data = self._load()
        data["entries"][str(directory)] = {
            "decision": decision,
            "fingerprint": fingerprint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp.replace(self._path)


def should_offer_project_skills_prompt(
    enabled: bool,
    entry: tuple[str, str] | None,
    fingerprint: str,
) -> bool:
    """Offer iff enabled AND (never seen OR changed-and-not-never) (spec §5.3)."""
    if not enabled:
        return False
    if entry is None:
        return True
    decision, recorded_fingerprint = entry
    if decision == "never":
        return False
    return recorded_fingerprint != fingerprint
```

- [ ] **Step 4: Run** — all PASS. **Step 5: Commit** — `git add -A && git commit -m "feat(skills): project-skills prompt ledger with fingerprint gating"` (+ trailer).

---

### Task 3: Import modal + shared offer helper

**Files:**
- Create: `tldw_chatbook/Widgets/project_skills_import_modal.py`
- Test: `Tests/Skills/test_project_skills_import_modal.py` (new)

**Interfaces:**
- Consumes: `ProjectSkillsDiscovery` (Task 1), `ProjectSkillsPromptLedger` (Task 2).
- Produces: `ProjectSkillsImportModal(ModalScreen)` with `__init__(self, *, discovery, installed_names: frozenset[str], importer)` where `importer` is `async (entry) -> None` (raises on failure); dismisses with `("imported", outcomes) | ("not_now", None) | ("never", None) | ("review", outcomes)` where `outcomes: tuple[tuple[str, str], ...]` of (name, "imported" | error text). Also `maybe_offer_project_skills_import(app, discoveries)` — the one entry point Tasks 4-5 call; it builds `installed_names` + `importer` from `app.skills_scope_service`, chains modals sequentially (one per discovery), writes the ledger on each dismissal (`imported`/`review` → "imported", `not_now` → "declined", `never` → "never"), and posts `NavigateToScreen("skills")` on `review`.

- [ ] **Step 1: Write the failing tests.** Import execution is injected, so tests use a recording fake importer — no skills store needed:

```python
# Tests/Skills/test_project_skills_import_modal.py
import pytest
from textual.app import App
from textual.widgets import Checkbox

from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    discover_project_skills,
)
from tldw_chatbook.Widgets.project_skills_import_modal import (
    ProjectSkillsImportModal,
)


def _discovery(tmp_path, names=("alpha-skill", "beta-skill")):
    for name in names:
        d = tmp_path / ".SKILLS" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\ndescription: [red]desc[/red] for {name}\n---\nBody\n",
            encoding="utf-8",
        )
    return discover_project_skills(tmp_path)


class _HarnessApp(App[None]):
    def __init__(self, modal):
        super().__init__()
        self._modal = modal
        self.result = "unset"

    def on_mount(self) -> None:
        def _done(result):
            self.result = result

        self.push_screen(self._modal, _done)


def _modal(tmp_path, installed=frozenset(), imported=None, fail=()):
    imported = imported if imported is not None else []

    async def importer(entry):
        if entry.name in fail:
            raise ValueError("import exploded")
        imported.append(entry.name)

    return (
        ProjectSkillsImportModal(
            discovery=_discovery(tmp_path),
            installed_names=installed,
            importer=importer,
        ),
        imported,
    )


@pytest.mark.asyncio
async def test_new_rows_checked_installed_rows_unchecked(tmp_path):
    modal, _ = _modal(tmp_path, installed=frozenset({"beta-skill"}))
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        boxes = {
            box.id: box.value for box in modal.query(Checkbox)
        }
        assert boxes["project-skill-row-0"] is True   # alpha-skill: new
        assert boxes["project-skill-row-1"] is False  # beta-skill: installed


@pytest.mark.asyncio
async def test_escape_means_not_now(tmp_path):
    modal, _ = _modal(tmp_path)
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result == ("not_now", None)


@pytest.mark.asyncio
async def test_never_button(tmp_path):
    modal, _ = _modal(tmp_path)
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-never")
        await pilot.pause()
    assert app.result == ("never", None)


@pytest.mark.asyncio
async def test_import_selected_runs_importer_and_reports(tmp_path):
    modal, imported = _modal(tmp_path, fail=("beta-skill",))
    app = _HarnessApp(modal)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#project-skills-import")
        await pilot.pause()
        # results phase: Close dismisses with outcomes
        await pilot.click("#project-skills-close")
        await pilot.pause()
    assert imported == ["alpha-skill"]
    decision, outcomes = app.result
    assert decision == "imported"
    assert ("alpha-skill", "imported") in outcomes
    assert any(name == "beta-skill" and "exploded" in msg for name, msg in outcomes)
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement.** Modal shape (follow `ConsoleWorkspaceSwitcherModal`'s DEFAULT_CSS/BINDINGS/escape pattern):
  - Header `Static`s (`markup=False`): title "Project skills found"; provenance `f"Found in {discovery.skills_dir}"`; the trust framing verbatim from spec §5.5: *"Imported skills require a one-time trust review in Library ▸ Skills before they can run."*
  - One row per entry: `status == "ok"` and not installed → `Checkbox(f"{name} — {description}", True, id=f"project-skill-row-{i}")`; installed → same, `False`, label suffix " (already installed)"; `invalid` → `Static(f"{name} — invalid: {reason}", markup=False)`. All checkbox labels built from repo strings must be passed through `rich.markup.escape()` (Checkbox labels render markup; Statics use `markup=False`).
  - Footer lines for `discovery.skipped` and `f"{discovery.truncated} more not shown"` when nonzero.
  - Buttons: `#project-skills-import` ("Import selected"), `#project-skills-not-now` ("Not now"), `#project-skills-never` ("Never for this folder"). `escape` → `dismiss(("not_now", None))`.
  - Import press: gather selected entries, run `self.run_worker(self._run_import(selected), exclusive=True)`; `_run_import` awaits `self._importer(entry)` per entry collecting `(name, "imported")` / `(name, str(exc))`, then swaps the modal body to a results phase (recompose on an internal `_outcomes` attribute): per-outcome `Static`s, the bootstrap-aware trust line ("Set up skill trust if this is your first skill, then approve each one."), and buttons `#project-skills-review` ("Review in Library ▸ Skills") → `dismiss(("review", outcomes))` and `#project-skills-close` ("Close") → `dismiss(("imported", outcomes))`.
  - `maybe_offer_project_skills_import(app, discoveries)` in the same module: builds `installed_names` from `app.skills_scope_service` (mirror the name-listing call `library_screen.py:3855` uses), an `importer` that mirrors the exact call shapes at `library_screen.py:9721-9811` (`import_skill_directory(entry.path, mode="local", name=entry.name, trust_approved=False)` for directories; the file-import call for loose files), and chains: push the modal for `discoveries[0]` with a callback that records the ledger decision (`ProjectSkillsPromptLedger(get_user_data_dir())`, keyed by `discovery.root`, with `discovery.fingerprint`), posts `NavigateToScreen("skills")` on `review`, then recurses on the remaining discoveries.

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Skills/test_project_skills_import_modal.py -v` — all PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(skills): project-skills import modal with quarantine-preserving importer"` (+ trailer).

---

### Task 4: Startup trigger in app.py

**Files:**
- Modify: `tldw_chatbook/app.py:7725-7744` (`_maybe_offer_first_run_wizard` — return bool) and `:7878-7882` (`_post_mount_setup` call site)
- Test: `Tests/Skills/test_project_skills_startup_gate.py` (new — pure pieces only; app-level behavior is covered by Task 6's live pass)

**Interfaces:**
- Consumes: Tasks 1-3 modules; `get_cli_setting`, `get_user_data_dir` from `tldw_chatbook.config`.
- Produces: `TldwCli._maybe_offer_project_skills_import()`; `_maybe_offer_first_run_wizard() -> bool` (True iff the wizard was pushed this launch).

- [ ] **Step 1: Write the failing test** for the seam that decides whether startup offers — extract it as a pure function in `project_skills_prompt.py` so app.py stays thin:

```python
# Tests/Skills/test_project_skills_startup_gate.py
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_prompt import (
    startup_discovery_for,
)


def _skill(root):
    d = root / ".SKILLS" / "alpha-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\ndescription: x\n---\nB\n", encoding="utf-8")


def test_startup_discovery_found(tmp_path):
    _skill(tmp_path / "repo")
    (tmp_path / "repo" / ".git").mkdir()
    sub = tmp_path / "repo" / "src"
    sub.mkdir()
    discovery = startup_discovery_for(sub, enabled=True, ledger_dir=tmp_path / "data")
    assert discovery is not None and discovery.entries


def test_startup_discovery_disabled(tmp_path):
    _skill(tmp_path)
    assert startup_discovery_for(tmp_path, enabled=False, ledger_dir=tmp_path / "d") is None


def test_startup_discovery_respects_never(tmp_path):
    _skill(tmp_path)
    from tldw_chatbook.Skills_Interop.project_skills_prompt import (
        ProjectSkillsPromptLedger,
    )
    ledger = ProjectSkillsPromptLedger(tmp_path / "data")
    ledger.record(tmp_path.resolve(), "never", "anything")
    assert startup_discovery_for(tmp_path, enabled=True, ledger_dir=tmp_path / "data") is None
```

- [ ] **Step 2: Run to verify failure**, then implement `startup_discovery_for(start: Path, *, enabled: bool, ledger_dir: Path) -> ProjectSkillsDiscovery | None` in `project_skills_prompt.py`: return None unless enabled; `find_project_dir_with_skills(start)`; `discover_project_skills(project_dir)`; None if no entries; None unless `should_offer_project_skills_prompt(True, ledger.decision_for(discovery.root), discovery.fingerprint)`.

- [ ] **Step 3: Wire app.py.** `_maybe_offer_first_run_wizard` returns True on the `should_offer_wizard` branch, False otherwise (update its docstring). At the `:7878` call site:

```python
        wizard_offered = self._maybe_offer_first_run_wizard()
        try:
            self._maybe_warn_second_instance()
        except Exception as e:
            logger.error(f"Second-instance warning failed: {e}")
        if not wizard_offered:
            # Spec 2026-08-17 §5.4: wizard wins; .SKILLS offer defers to next launch.
            self._maybe_offer_project_skills_import()
```

New methods on `TldwCli` (place after `_push_first_run_wizard`):

```python
    def _maybe_offer_project_skills_import(self) -> None:
        """Offer to import a project's .SKILLS/ folder (spec 2026-08-17 §5.4)."""
        try:
            self.run_worker(
                self._discover_project_skills_for_startup,
                thread=True,
                exclusive=True,
                group="project-skills-discovery",
            )
        except Exception:
            logger.opt(exception=True).debug("project-skills startup offer failed")

    def _discover_project_skills_for_startup(self) -> None:
        from tldw_chatbook.config import get_cli_setting, get_user_data_dir
        from tldw_chatbook.Skills_Interop.project_skills_prompt import (
            startup_discovery_for,
        )

        try:
            cwd = Path.cwd().resolve()
        except OSError:
            return  # launch directory deleted out from under the process
        discovery = startup_discovery_for(
            cwd,
            enabled=bool(
                get_cli_setting("skills", "project_skills_prompt_enabled", True)
            ),
            ledger_dir=get_user_data_dir(),
        )
        if discovery is None:
            return
        self.call_from_thread(self._push_project_skills_import_modal, discovery)

    def _push_project_skills_import_modal(self, discovery) -> None:
        from tldw_chatbook.Widgets.project_skills_import_modal import (
            maybe_offer_project_skills_import,
        )

        maybe_offer_project_skills_import(self, (discovery,))
```

(`startup_discovery_for` takes `ledger_dir=get_user_data_dir()` and builds the ledger itself, so the ledger path stays defined in exactly one module.)

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Skills/test_project_skills_startup_gate.py Tests/Skills/test_project_skills_prompt.py -v`; then `.venv/bin/pytest Tests/ --collect-only -q | tail -3` to prove app.py still imports.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(app): offer project .SKILLS import at startup behind wizard + ledger gates"` (+ trailer).

---

### Task 5: Create-modal chaining

**Files:**
- Modify: `tldw_chatbook/Widgets/workspace_create_modal.py` (`_add_folder`, `_create`, folder-row compose)
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py` (`_handle_workspace_create_result`), `tldw_chatbook/UI/Screens/settings_screen.py` (`handle_workspace_create._done`), `tldw_chatbook/UI/Screens/library_screen.py` (`create_local_workspace._done`)
- Test: extend `Tests/Workspaces/test_workspace_create_modal.py` and `Tests/Workspaces/test_console_workspace_create_handler.py`

**Interfaces:**
- Consumes: `discover_project_skills` (Task 1), `maybe_offer_project_skills_import` (Task 3).
- Produces: `WorkspaceCreateResult.project_skills: tuple[ProjectSkillsDiscovery, ...]` now populated for bound folders whose root contains `.SKILLS/`.

- [ ] **Step 1: Write the failing tests**

```python
# append to Tests/Workspaces/test_workspace_create_modal.py
@pytest.mark.asyncio
async def test_folder_with_skills_annotated_and_carried_on_result(tmp_path):
    registry = _registry(tmp_path)
    project = tmp_path / "project"
    skill = project / ".SKILLS" / "alpha-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: x\n---\nB\n", encoding="utf-8")
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()
        rows = [str(s.renderable) for s in modal.query(".workspace-create-folder-locator")]
        assert any("1 project skill" in row for row in rows)
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
    assert len(app.result.project_skills) == 1
    assert app.result.project_skills[0].entries[0].name == "alpha-skill"
```

and in the console handler test file, assert the handler calls the offer helper:

```python
def test_result_with_project_skills_offers_import(tmp_path, monkeypatch):
    offered = []
    import tldw_chatbook.UI.Console_Modules.workspace as ws_module

    monkeypatch.setattr(
        ws_module,
        "maybe_offer_project_skills_import",
        lambda app, discoveries: offered.append(discoveries),
    )
    stub = _Stub(_registry(tmp_path))
    result = WorkspaceCreateResult(
        workspace_id="workspace-local-1",
        name="Workspace 1",
        make_active=False,
        project_skills=("sentinel-discovery",),
    )
    ConsoleWorkspaceController._handle_workspace_create_result(stub, result)
    assert offered == [("sentinel-discovery",)]
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.**
  - In `_add_folder`, after appending the folder: run `discover_project_skills(Path(resolved))`; keep a parallel `self._folder_discoveries: dict[str, ProjectSkillsDiscovery]`; the folder row Static label becomes `f"{folder} — contains {n} project skill(s)"` when a discovery with entries exists.
  - In `_create`, set `project_skills=tuple(self._folder_discoveries[f] for f in bound if f in self._folder_discoveries)`.
  - In each of the three surface callbacks, after their existing post-create work add (module-level import so tests can monkeypatch it):

```python
        if result.project_skills:
            maybe_offer_project_skills_import(
                self.app_instance if hasattr(self, "app_instance") else self.app,
                result.project_skills,
            )
```

  (Console/Settings/Library each pass their app handle the way the surrounding code already does — `self.app_instance` in the Console controller and Library, `self.app` where the Settings callback uses it.)

- [ ] **Step 4: Run** — both extended test files + `.venv/bin/pytest Tests/Workspaces/ Tests/Skills/ -q`. All PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(workspaces): chain project-skills import offer after workspace creation"` (+ trailer).

---

### Task 6: ADR, config docs, User Guide, live verification, close-out

**Files:**
- Create: `backlog/decisions/0NN-project-skills-folder-convention.md` (`ls backlog/decisions/` for the next free number)
- Modify: `tldw_chatbook/config.py` (`:2588` comment block region — document the new key beside `workspace_root`'s doc line)
- Modify: `Docs/User_Guide/` skills + workspaces pages (locate via `grep -rln -i "skills" Docs/User_Guide/`)

- [ ] **Step 1: ADR.** Contents per spec §8: the `.SKILLS/`/`.skills/` layout (per-skill dirs with SKILL.md + loose `*.md`), both triggers, import-copy-not-live-load rationale, quarantine posture and its relation to ADR-009 (import-copy stays inside the boundary; live-load would not), the fingerprint ledger, and the `[skills] project_skills_prompt_enabled` kill-switch.
- [ ] **Step 2: Config docs.** Add `# project_skills_prompt_enabled = true  # offer .SKILLS/ import at startup; spec 2026-08-17` to the commented `[skills]`-area of the config template/docs near `config.py:2588`.
- [ ] **Step 3: User Guide.** Document the `.SKILLS/` convention on the skills page (layout, both prompts, "one-time trust review" expectation, kill-switch) and the create-modal chaining on the workspaces page; refresh "Verified against" stamps.
- [ ] **Step 4: Live verification** per the `verify` skill and `lessons-live-verification.md`: launch the TUI from a fixture project containing `.SKILLS/` (2 valid skills, 1 invalid name, 1 loose file) — confirm the startup modal, import 2, confirm they appear in Library ▸ Skills as trust-pending and are refused in Console with the Library pointer; relaunch to confirm no re-prompt; add a skill and relaunch to confirm the changed-fingerprint re-offer; create a workspace bound to the fixture to confirm the chained offer; confirm "Never" then relaunch.
- [ ] **Step 5: Gate + close-out** — `.venv/bin/pytest Tests/Skills/ Tests/Workspaces/ -q` plus `--collect-only` sweep; backlog task ACs checked, Implementation Notes added, status Done; commit docs (+ trailer); push and open the PR B branch PR.
