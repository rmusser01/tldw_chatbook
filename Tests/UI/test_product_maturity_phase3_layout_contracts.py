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
TASK_10_2 = Path(
    "backlog/tasks/task-10.2 - Product-Maturity-Phase-3.0-Destination-Layout-And-IA-Contracts.md"
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
    match = re.search(r"^Status:\s*(.+)$", text, re.MULTILINE)
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
    status = _status_line(spec)

    assert "spec review approved" in status
    assert "implementation verified" in status
    assert "pending spec review" not in spec
    assert "pending implementation planning" not in spec


def test_phase30_spec_has_destination_contracts_and_image_briefs() -> None:
    spec = _text(SPEC)

    for destination in DESTINATIONS:
        assert f"### {destination}" in spec
        section_start = spec.index(f"### {destination}")
        next_heading = spec.find("\n### ", section_start + 1)
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


def test_phase30_image_prompt_manifest_has_one_prompt_per_destination() -> None:
    manifest = _text(PROMPT_MANIFEST)
    caption = "Non-binding inspiration; text and ASCII contract are authoritative."

    assert "# Destination Layout Image Reference Prompts" in manifest
    assert caption in manifest

    destination_headings = list(re.finditer(r"^##\s+(.+?)\s*$", manifest, flags=re.MULTILINE))
    assert [heading.group(1) for heading in destination_headings] == list(DESTINATIONS)

    for index, heading in enumerate(destination_headings):
        section_start = heading.start()
        section_end = (
            destination_headings[index + 1].start()
            if index + 1 < len(destination_headings)
            else len(manifest)
        )
        section = manifest[section_start:section_end]

        assert re.search(rf"^Caption:\s*.*{re.escape(caption)}.*$", section, flags=re.MULTILINE)
        assert len(re.findall(r"^```text\s*\n.*?^```\s*$", section, flags=re.MULTILINE | re.DOTALL)) == 1
        assert "Textual-native terminal UI concept" in section
        assert "Avoid" in section
        assert "Non-binding inspiration" in section


def test_phase30_spec_has_major_subflow_contracts() -> None:
    spec = _text(SPEC)

    for subflow in SUBFLOWS:
        assert f"### {subflow}" in spec
        section_start = spec.index(f"### {subflow}")
        next_heading = spec.find("\n### ", section_start + 1)
        section = spec[section_start:] if next_heading == -1 else spec[section_start:next_heading]
        assert "```text" in section
        assert "Required behavior:" in section
    assert "`study`, Study Dashboard, Flashcards, and Quizzes" in spec
    assert "Study must expose the active source/workspace scope" in spec


def test_phase30_route_inventory_keeps_study_library_owned() -> None:
    inventory = _text(ROUTE_INVENTORY)

    library_row = _markdown_table_row(inventory, "Library")
    assert "`study`" in library_row[1]
    assert "`conversation`" in library_row[1]
    assert "conversation browsing" not in library_row[1]
    for route_token in [token.strip() for token in library_row[1].split(",")]:
        assert route_token.startswith("`")
        assert route_token.endswith("`")
    assert "Study Dashboard" in library_row[4]
    assert "conversation browsing" in library_row[4]

    study_row = _markdown_table_row(inventory, "Study route")
    assert study_row[2] == "Library"
    assert "do not create a separate top-level Study destination" in study_row[3]


def test_phase30_tracker_records_layout_contract_gate() -> None:
    tracker = _text(TRACKER)

    assert "Layout Contract Spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`" in tracker
    assert "Phase 3.0: Destination Layout And IA Contracts - `TASK-10.2`" in tracker
    assert "destination layout and IA contracts are approved" in tracker


def test_phase30_qa_evidence_is_verified() -> None:
    evidence = _text(PHASE_3_0_EVIDENCE)
    readme = _text(PHASE_3_README)

    assert _status_line(evidence) == "verified"

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

    assert "TASK-10.2" in evidence
    assert "Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md" in evidence
    assert "spec review approved" in evidence
    assert "compact, default, and large terminal" in evidence
    assert "usable, not merely rendered" in evidence
    assert PHASE_3_0_EVIDENCE.name in readme
    assert "Phase 3.0 destination layout contract status: verified" in readme


def test_phase30_tracker_has_evidence_row() -> None:
    tracker = _text(TRACKER)

    row = _markdown_table_row(tracker, "Phase 3.0")
    assert "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md" in row[1]
    assert row[2] == "verified"
    phase_three_row = _markdown_table_row(tracker, "Phase 3: Knowledge And Study Workflows")
    assert "Phase 3.0 verified" in phase_three_row[2]
    assert "Phase 3.0 evidence pending" not in phase_three_row[4]
    assert "Phase 3.0 prerequisite planned" not in tracker
    assert "Status: Phase 1 verified; Phase 2 verified; Phase 3.0 verified; Phase 3.1 verified" in tracker


def test_phase30_backlog_task_is_closed_after_verification() -> None:
    task = _text(TASK_10_2)

    assert "status: Done" in task
    for ac_number in range(1, 6):
        assert re.search(rf"^- \[x\] #{ac_number}\b", task, flags=re.MULTILINE)
    assert "## Implementation Plan" in task
    assert "## Implementation Notes" in task
    assert (REPO_ROOT / SPEC).is_file()
    assert (REPO_ROOT / Path("Tests/UI/test_product_maturity_phase3_layout_contracts.py")).is_file()
    assert "Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md" in task
    assert "Tests/UI/test_product_maturity_phase3_layout_contracts.py" in task
