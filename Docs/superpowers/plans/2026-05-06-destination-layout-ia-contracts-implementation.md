# Destination Layout And IA Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 3.0 enforceable by aligning runtime route ownership, adding regression coverage, recording QA evidence, and preparing non-binding image-reference prompts for every destination.

**Architecture:** This plan treats Phase 3.0 as a contract gate, not a screen rewrite. The first slice fixes the only runtime inconsistency introduced by the contract: `study` must resolve as Library-owned. The remaining slices add doc/QA/test enforcement so later Phase 3+ work cannot drift from the approved destination layout contracts.

**Tech Stack:** Python 3.11+, pytest, Textual shell route metadata, Backlog.md task files, Markdown QA/spec/tracker docs.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Product roadmap: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- Product tracker: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Route inventory: `Docs/Design/master-shell-route-inventory.md`
- Design system contract: `Docs/Design/master-shell-design-system-contract.md`
- Backlog task: `backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md`

## File Structure

- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`
  - Owns master-shell destination metadata and legacy route resolution.
  - Add `study` as a Library-owned legacy route.

- Modify: `Tests/UI/test_shell_destinations.py`
  - Existing shell route contract tests.
  - Add regression that `study` resolves to `library`.

- Create: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
  - Markdown contract tests for Phase 3.0 spec, tracker, QA evidence, prompt manifest, and backlog task hygiene.
  - Keep this test file doc-focused and independent of Textual app startup.

- Modify: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
  - Update status after spec review.
  - Keep Study route and Study Dashboard Library-owned.

- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Add Phase 3.0 QA evidence row with `recorded; pending final closeout` status once the QA artifact exists.
  - Defer Phase 3.0 verified status until final TASK-10.0 closeout.

- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
  - Add Phase 3.0 evidence entry.

- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`
  - QA evidence for the design gate.

- Create: `Docs/Design/destination-layout-image-reference-prompts.md`
  - Consolidated prompt manifest for one non-binding image-reference per top-level destination.
  - Do not commit generated images in this plan unless the user explicitly asks during execution.

- Modify: `backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md`
  - Add implementation plan/notes.
  - Check ACs only after verification passes.
  - Mark Done only after all DoD evidence is present.

## Task 1: Align Runtime Study Route With Library Ownership

**Files:**
- Modify: `Tests/UI/test_shell_destinations.py`
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py`

- [ ] **Step 1: Write the failing route regression**

Add `study` to `test_legacy_routes_resolve_to_master_destinations()`:

```python
def test_legacy_routes_resolve_to_master_destinations():
    expectations = {
        "chat": ("console", "chat"),
        "home": ("home", "home"),
        "notes": ("library", "notes"),
        "media": ("library", "media"),
        "ingest": ("library", "ingest"),
        "search": ("library", "search"),
        "study": ("library", "study"),
        "chatbooks": ("artifacts", "chatbooks"),
        "ccp": ("personas", "ccp"),
        "conversation": ("library", "conversation"),
        "conversations_characters_prompts": ("personas", "ccp"),
        "subscriptions": ("watchlists_collections", "subscriptions"),
        "tools_settings": ("mcp", "tools_settings"),
        "settings": ("settings", "settings"),
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py::test_legacy_routes_resolve_to_master_destinations --tb=short
```

Expected before implementation: FAIL because `resolve_shell_route("study")` returns `("study", "study")`, not `("library", "study")`.

- [ ] **Step 3: Add `study` to Library route metadata**

In `tldw_chatbook/UI/Navigation/shell_destinations.py`, update the Library destination:

```python
ShellDestination(
    "library",
    "Library",
    "library",
    "Workspaces, source material, imports, notes, media, conversations, Study, and Search/RAG.",
    "Browse Workspaces, imports, notes, media, Study, search, and source material.",
    ("notes", "media", "ingest", "search", "conversation", "study"),
    navigation_priority=30,
),
```

Add `study` to `_ROUTABLE_LEGACY_ROUTES`:

```python
_ROUTABLE_LEGACY_ROUTES = {
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "study",
    "conversation",
    "chatbooks",
    "ccp",
    "subscriptions",
    "tools_settings",
    "customize",
}
```

- [ ] **Step 4: Run route tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Tests/UI/test_shell_destinations.py tldw_chatbook/UI/Navigation/shell_destinations.py
git commit -m "Align Study route with Library ownership"
```

## Task 2: Add Phase 3.0 Contract Regression Tests

**Files:**
- Create: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- Modify: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`

- [ ] **Step 1: Write failing contract status test**

Create `Tests/UI/test_product_maturity_phase3_layout_contracts.py`:

```python
"""Phase 3.0 destination layout contract documentation regressions."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = Path("Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
ROUTE_INVENTORY = Path("Docs/Design/master-shell-route-inventory.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_0_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md"
)
TASK_10_0 = Path(
    "backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md"
)
PROMPT_MANIFEST = Path("Docs/Design/destination-layout-image-reference-prompts.md")

DESTINATIONS = (
    "Home",
    "Console",
    "Library",
    "Artifacts",
    "Personas",
    "W+C",
    "Schedules",
    "Workflows",
    "MCP",
    "ACP",
    "Skills",
    "Settings",
)

SUBFLOWS = (
    "Library: Search/RAG",
    "Library: Import/Export",
    "Library: Workspaces",
    "Library: Study Dashboard",
    "Library: Flashcards",
    "Library: Quizzes",
    "Library: Notes, Media, Conversations, Source Detail",
    "Artifacts: Chatbooks",
    "Artifacts: Exports And Reuse",
    "Personas: Detail/Edit/Import/Export",
    "W+C: Watchlists",
    "W+C: Collections",
    "Schedules: Detail And History",
    "Workflows: Builder And Run Detail",
    "MCP: Tools/Resources/Readiness",
    "ACP: Agents/Sessions/Runtime",
    "Skills: Validation/Edit/Attach",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _status_line(text: str) -> str:
    match = re.search(r"^Status:\\s*(.+)$", text, re.MULTILINE)
    assert match is not None
    return match.group(1).strip()


def _markdown_table_row(text: str, first_column: str) -> list[str]:
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if columns and columns[0] == first_column:
            return columns
    raise AssertionError(f"row not found for {first_column!r}")


def test_phase30_spec_review_status_is_approved() -> None:
    spec = _text(SPEC)

    assert "spec review approved" in _status_line(spec)
    assert "pending spec review" not in spec
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_spec_review_status_is_approved --tb=short
```

Expected before implementation: FAIL because the spec status still says `pending spec review`.

- [ ] **Step 3: Update spec status**

In `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`, change:

```markdown
Status: User-approved design; pending spec review
```

to:

```markdown
Status: User-approved design; spec review approved; pending implementation planning
```

- [ ] **Step 4: Add contract completeness tests**

Append these tests to `Tests/UI/test_product_maturity_phase3_layout_contracts.py`:

```python
def test_phase30_spec_has_destination_contracts_and_image_briefs() -> None:
    spec = _text(SPEC)

    for destination in DESTINATIONS:
        assert f"### {destination}" in spec
        section_start = spec.index(f"### {destination}")
        next_heading = spec.find("\\n### ", section_start + 1)
        section = spec[section_start:] if next_heading == -1 else spec[section_start:next_heading]
        assert "User goal:" in section
        assert "Screen role:" in section
        assert "Binding regions:" in section
        assert "```text" in section
        assert "Primary actions:" in section
        assert "Focus path:" in section
        assert "Console handoff:" in section
        assert "Image reference brief:" in section
        assert "QA checks:" in section
        assert "Textual-native terminal UI concept" in section


def test_phase30_spec_has_major_subflow_contracts() -> None:
    spec = _text(SPEC)

    for subflow in SUBFLOWS:
        assert f"### {subflow}" in spec
    assert "`study`, Study Dashboard, Flashcards, and Quizzes" in spec
    assert "Study must expose the active source/workspace scope" in spec


def test_phase30_route_inventory_keeps_study_library_owned() -> None:
    inventory = _text(ROUTE_INVENTORY)

    library_row = _markdown_table_row(inventory, "Library")
    assert "`study`" in library_row[1]
    assert "Study Dashboard" in library_row[4]

    study_row = _markdown_table_row(inventory, "Study route")
    assert study_row[2] == "Library"
    assert "do not create a separate top-level Study destination" in study_row[3]


def test_phase30_tracker_records_layout_contract_gate() -> None:
    tracker = _text(TRACKER)

    assert "Layout Contract Spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`" in tracker
    assert "Phase 3.0: Destination Layout And IA Contracts - `TASK-10.0`" in tracker
    assert "destination layout and IA contracts must be approved" in tracker
```

- [ ] **Step 5: Run contract tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short
```

Expected after implementation: currently FAIL on missing Phase 3.0 QA evidence and prompt manifest only after Task 3 and Task 4 tests are added. The tests above should PASS.

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_product_maturity_phase3_layout_contracts.py Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
git commit -m "Add Phase 3.0 layout contract regressions"
```

## Task 3: Record Phase 3.0 QA Evidence And Tracker Pending-Closeout State

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`

- [ ] **Step 1: Add failing QA evidence tests**

Append these tests:

```python
def test_phase30_qa_evidence_is_recorded() -> None:
    evidence = _text(PHASE_3_0_EVIDENCE)
    readme = _text(PHASE_3_README)

    assert _status_line(evidence) == "recorded; pending final closeout"

    for section in (
        "## Scope",
        "## Evidence",
        "## Contract Coverage",
        "## Route Ownership Result",
        "## Terminal Size Gate",
        "## Image Reference Governance",
        "## Residual Risk",
        "## Result",
    ):
        assert section in evidence

    assert "TASK-10.0" in evidence
    assert "Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md" in evidence
    assert "spec review approved" in evidence
    assert "compact, default, and large terminal" in evidence
    assert "usable, not merely rendered" in evidence
    assert "pending final closeout" in evidence
    assert "Status: verified" not in evidence
    assert PHASE_3_0_EVIDENCE.name in readme
    assert "Phase 3.0 destination layout contract status: recorded; pending final closeout" in readme
    assert "Phase 3.0 destination layout contract status: verified" not in readme


def test_phase30_tracker_has_evidence_row() -> None:
    tracker = _text(TRACKER)

    row = _markdown_table_row(tracker, "Phase 3.0")
    assert "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md" in row[1]
    assert row[2] == "recorded; pending final closeout"
    phase_three_row = _markdown_table_row(tracker, "Phase 3: Knowledge And Study Workflows")
    assert "Phase 3.0 evidence recorded; pending final closeout" in phase_three_row[2]
    assert "Phase 3.0 verified" not in phase_three_row[2]
    assert "Phase 3.0 evidence pending" not in phase_three_row[4]
    assert "Phase 3.0 prerequisite planned" not in tracker
    assert "Status: Phase 1 verified; Phase 2 verified; Phase 3.0 verified; Phase 3.1 verified; Phase 3.2 verified" not in tracker
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_qa_evidence_is_recorded Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_tracker_has_evidence_row --tb=short
```

Expected before implementation: FAIL because the Phase 3.0 evidence file and tracker row do not exist yet.

- [ ] **Step 3: Create Phase 3.0 evidence file**

Create `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`:

```markdown
# Product Maturity Phase 3.0 Destination Layout Contracts

Date: 2026-05-06
Status: recorded; pending final closeout
Task: TASK-10.0

## Scope

Phase 3.0 records that destination layout and IA contracts exist before additional Phase 3 Knowledge/Study visual rewrites continue.

## Evidence

- Spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Route inventory: `Docs/Design/master-shell-route-inventory.md`
- Product tracker: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Backlog task: `backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md`
- spec review approved after fixing Study route and Study Dashboard ownership.

## Contract Coverage

All top-level destinations have user goal, screen role, binding regions, ASCII wireframe, primary actions, focus path, Console handoff behavior, image-reference brief, and QA checks.

Major subflows are assigned to owner destinations, including Library-owned Search/RAG, Import/Export, Workspaces, Study Dashboard, Flashcards, Quizzes, Notes, Media, Conversations, and source detail.

## Route Ownership Result

The `study` route, Study Dashboard, Flashcards, and Quizzes are Library-owned. Study is not a separate top-level destination.

## Terminal Size Gate

The contract requires compact, default, and large terminal verification for later implementation gates. No destination may require a large terminal to complete its primary workflow.

## Image Reference Governance

Generated references are non-binding inspiration only. Text and ASCII contracts are authoritative.

## Residual Risk

This gate verifies contracts, not runtime screen rewrites. Later implementation PRs must prove affected screens are usable, not merely rendered.

## Result

Phase 3.0 contract evidence is recorded for planning and remains pending final closeout until TASK-10.0 is prepared, checked, and marked Done.
```

- [ ] **Step 4: Update Phase 3 README**

Add a Phase 3.0 section to `Docs/superpowers/qa/product-maturity/phase-3/README.md` before Phase 3.1:

```markdown
Phase 3.0 evidence:

Phase 3.0 destination layout contract status: recorded; pending final closeout

- `2026-05-06-phase-3-0-destination-layout-contracts.md`

Phase 3.0 records destination layout and IA contracts before additional Phase 3 Knowledge/Study visual rewrites continue. It does not rewrite runtime screens or mark TASK-10.0 done.
```

- [ ] **Step 5: Update tracker evidence index**

In `Docs/superpowers/trackers/product-maturity-roadmap.md`, add:

```markdown
| Phase 3.0 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md` | recorded; pending final closeout |
```

Keep status as:

```markdown
Status: Phase 1 verified; Phase 2 verified; Phase 3.1 verified; Phase 3.2 verified
```

Do not add `Phase 3.0 verified` until final TASK-10.0 closeout.

Update the Phase Overview row from:

```markdown
| Phase 3: Knowledge And Study Workflows | Mature ingest, organize, retrieve, study, and reuse workflows. | in-progress; Phase 3.0 prerequisite planned; Phase 3.1 verified; Phase 3.2 verified | `TASK-10`, Phase 3.0 (`TASK-10.0`), Phase 3.1 (`TASK-10.1`), Phase 3.2 (`TASK-10.2`) | `phase-3/2026-05-06-phase-3-1-library-study-entry.md`; `phase-3/2026-05-06-phase-3-2-library-source-study-context.md`; Phase 3.0 evidence pending | Library Study entry and Library source context are verified; before additional Phase 3 visual rewrites, destination layout and IA contracts must be approved. Source-selected study generation, Workspaces, Collections, and deeper Import/Export/Search/RAG study flows remain. |
```

to:

```markdown
| Phase 3: Knowledge And Study Workflows | Mature ingest, organize, retrieve, study, and reuse workflows. | in-progress; Phase 3.0 evidence recorded; pending final closeout; Phase 3.1 verified; Phase 3.2 verified | `TASK-10`, Phase 3.0 (`TASK-10.0`), Phase 3.1 (`TASK-10.1`), Phase 3.2 (`TASK-10.2`) | `phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`; `phase-3/2026-05-06-phase-3-1-library-study-entry.md`; `phase-3/2026-05-06-phase-3-2-library-source-study-context.md` | Layout contracts are recorded, and Library Study entry and source context are verified; source-selected study generation, Workspaces runtime behavior, Collections, and deeper Import/Export/Search/RAG study flows remain. |
```

- [ ] **Step 6: Run QA evidence tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short
```

Expected: PASS for tests implemented so far.

- [ ] **Step 7: Commit**

```bash
git add Tests/UI/test_product_maturity_phase3_layout_contracts.py Docs/superpowers/qa/product-maturity/phase-3/README.md Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md Docs/superpowers/trackers/product-maturity-roadmap.md
git commit -m "Record Phase 3.0 layout contract QA evidence"
```

## Task 4: Create Non-Binding Image Reference Prompt Manifest

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- Create: `Docs/Design/destination-layout-image-reference-prompts.md`

- [ ] **Step 1: Add failing prompt manifest test**

Append:

```python
def test_phase30_image_prompt_manifest_has_one_prompt_per_destination() -> None:
    manifest = _text(PROMPT_MANIFEST)

    assert "# Destination Layout Image Reference Prompts" in manifest
    assert "Non-binding inspiration; text and ASCII contract are authoritative." in manifest

    for destination in DESTINATIONS:
        assert f"## {destination}" in manifest
        section_start = manifest.index(f"## {destination}")
        next_heading = manifest.find("\\n## ", section_start + 1)
        section = manifest[section_start:] if next_heading == -1 else manifest[section_start:next_heading]
        assert "Textual-native terminal UI concept" in section
        assert "Avoid" in section
        assert "Non-binding inspiration" in section
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_image_prompt_manifest_has_one_prompt_per_destination --tb=short
```

Expected before implementation: FAIL because `Docs/Design/destination-layout-image-reference-prompts.md` does not exist.

- [ ] **Step 3: Create prompt manifest**

Create `Docs/Design/destination-layout-image-reference-prompts.md`.

Use this structure:

```markdown
# Destination Layout Image Reference Prompts

Date: 2026-05-06
Status: Non-binding prompt manifest for Phase 3.0 destination references
Source Spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`

Every generated image must be captioned:

> Non-binding inspiration; text and ASCII contract are authoritative.

Generated images must not be used as 1:1 implementation requirements.

## Home

Caption: Non-binding inspiration; text and ASCII contract are authoritative.

Prompt:

```text
Create a Textual-native terminal UI concept for the Home destination...
```

## Console

...
```

Copy the 12 image-reference briefs from the spec into the manifest. Keep wording aligned with the spec.

- [ ] **Step 4: Run prompt manifest test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_image_prompt_manifest_has_one_prompt_per_destination --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Tests/UI/test_product_maturity_phase3_layout_contracts.py Docs/Design/destination-layout-image-reference-prompts.md
git commit -m "Add destination image reference prompt manifest"
```

## Task 5: Prepare TASK-10.0 For Verified Closeout

**Files:**
- Modify: `backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md`
- Modify: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`

- [ ] **Step 1: Add failing backlog preparation test**

Append:

```python
def test_phase30_backlog_task_has_plan_before_closeout() -> None:
    task = _text(TASK_10_0)

    assert "status: In Progress" in task
    assert "## Implementation Plan" in task
    assert "Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md" in task
    assert "Tests/UI/test_product_maturity_phase3_layout_contracts.py" in task
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_backlog_task_has_plan_before_closeout --tb=short
```

Expected before implementation: FAIL because TASK-10.0 is still `To Do` and does not have an implementation plan.

- [ ] **Step 3: Add implementation plan to TASK-10.0**

Use Backlog.md MCP or CLI equivalent to set status In Progress and add plan:

```markdown
1. Align runtime route metadata so `study` is Library-owned.
2. Add focused contract regressions for Phase 3.0 docs, route inventory, tracker, QA evidence, prompt manifest, and backlog task.
3. Record Phase 3.0 QA evidence and tracker pending-closeout state.
4. Add a non-binding destination image prompt manifest.
5. Run focused verification before checking ACs and closing the task.
```

- [ ] **Step 4: Run backlog preparation test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_backlog_task_has_plan_before_closeout --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Tests/UI/test_product_maturity_phase3_layout_contracts.py "backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md"
git commit -m "Prepare Phase 3.0 layout contract task"
```

## Task 6: Final Focused Verification And TASK-10.0 Closeout

**Files:**
- Modify: `backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md`
- Modify: `Tests/UI/test_product_maturity_phase3_layout_contracts.py`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`

- [ ] **Step 1: Add final backlog completion test**

Append:

```python
def test_phase30_backlog_task_is_closed_after_verification() -> None:
    task = _text(TASK_10_0)
    evidence = _text(PHASE_3_0_EVIDENCE)
    readme = _text(PHASE_3_README)
    tracker = _text(TRACKER)

    assert "status: Done" in task
    assert "- [x] #1" in task
    assert "- [x] #2" in task
    assert "- [x] #3" in task
    assert "- [x] #4" in task
    assert "- [x] #5" in task
    assert "## Implementation Plan" in task
    assert "## Implementation Notes" in task
    assert "Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md" in task
    assert "Tests/UI/test_product_maturity_phase3_layout_contracts.py" in task
    assert _status_line(evidence) == "verified"
    assert "Phase 3.0 destination layout contract status: verified" in readme
    assert "Status: Phase 1 verified; Phase 2 verified; Phase 3.0 verified; Phase 3.1 verified; Phase 3.2 verified" in tracker

    row = _markdown_table_row(tracker, "Phase 3.0")
    assert row[2] == "verified"
    phase_three_row = _markdown_table_row(tracker, "Phase 3: Knowledge And Study Workflows")
    assert "Phase 3.0 verified" in phase_three_row[2]
```

- [ ] **Step 2: Run final backlog completion test to verify failure**

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_backlog_task_is_closed_after_verification --tb=short
```

Expected before closeout: FAIL because TASK-10.0 is still In Progress and ACs are unchecked.

- [ ] **Step 3: Run focused Phase 3.0 test set before closing task**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short -k "not test_phase30_backlog_task_is_closed_after_verification"
```

Expected: PASS. The final backlog closeout test is intentionally excluded until TASK-10.0 is marked Done.

- [ ] **Step 4: Run related product maturity tests before closing task**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_shell_product_model_visibility.py Tests/UI/test_command_palette_shell_routes.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Check formatting before closing task**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Check ACs, add implementation notes, and promote docs to verified**

Use Backlog.md MCP or CLI equivalent to check all ACs and add implementation notes:

```markdown
Implemented Phase 3.0 as an enforceable layout contract gate. Runtime route metadata now treats `study` as Library-owned, matching the approved IA. Added doc-focused regressions for destination contracts, major subflows, Study ownership, QA evidence, route inventory, prompt governance, and backlog closure. Recorded Phase 3.0 QA evidence and created the non-binding image prompt manifest for later reference generation.
```

In this closeout step, promote Phase 3.0 QA evidence, README, and tracker status from `recorded; pending final closeout` to `verified`. This transition belongs here because TASK-10.0 is being checked and marked Done.

- [ ] **Step 7: Mark TASK-10.0 done**

Use Backlog.md MCP or CLI equivalent:

```bash
backlog task edit TASK-10.0 -s Done
```

- [ ] **Step 8: Run final backlog completion test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_layout_contracts.py::test_phase30_backlog_task_is_closed_after_verification --tb=short
```

Expected: PASS.

- [ ] **Step 9: Commit closeout**

```bash
git add Tests/UI/test_product_maturity_phase3_layout_contracts.py "backlog/tasks/task-10.0 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md" Docs/superpowers/qa/product-maturity/phase-3/README.md Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md Docs/superpowers/trackers/product-maturity-roadmap.md
git commit -m "Close Phase 3.0 layout contract gate"
```

## Task 7: Final PR Prep

**Files:**
- No planned source edits.
- Verify all touched areas.

- [ ] **Step 1: Re-run final focused Phase 3.0 test set**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_shell_destinations.py Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short
```

Expected: PASS.

- [ ] **Step 2: Re-run related product maturity tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_shell_product_model_visibility.py Tests/UI/test_command_palette_shell_routes.py --tb=short
```

Expected: PASS.

- [ ] **Step 3: Re-check formatting/trailing whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Review final diff**

Run:

```bash
git status --short
git diff --stat
git diff -- Docs/superpowers/ Docs/Design/ Tests/UI/ tldw_chatbook/UI/Navigation/ backlog/tasks/
```

Expected:

- Only Phase 3.0 route/docs/tests/task files are modified.
- Unrelated untracked files remain untouched unless intentionally included.

- [ ] **Step 5: Commit any final verification/doc tweaks**

Only if Step 4 shows intentional uncommitted changes:

```bash
git add <intentional-files>
git commit -m "Verify Phase 3.0 layout contract gate"
```

- [ ] **Step 6: Prepare PR against `dev`**

If working in a feature branch:

```bash
git push -u origin <branch-name>
gh pr create --base dev --head <branch-name> --title "Verify Phase 3.0 destination layout contracts" --body-file /tmp/phase30-layout-contract-pr.md
```

PR body should summarize:

- `study` route now resolves to Library.
- Phase 3.0 contract regressions added.
- QA evidence and tracker closeout recorded.
- Non-binding image prompt manifest added.
- Focused verification commands and results.

## Stop Conditions

- Stop if `origin/dev` changes the product-maturity tracker or route inventory in a conflicting way; rebase first and re-check the Phase 3.0 spec assumptions.
- Stop if generated images are requested as committed assets before prompt manifest review; confirm storage path and artifact expectations first.
- Stop if test startup hits local SQLite or optional dependency issues outside the touched doc/route tests; report the environment blocker and keep doc-only tests focused.
- Stop if implementing Phase 3.0 starts turning into runtime screen rewrites. Runtime destination redesign belongs in later PR-sized implementation slices.
