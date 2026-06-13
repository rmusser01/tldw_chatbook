import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")

PHASES = {
    "Phase 2": {
        "qa_dir": "phase-2",
        "status": "verified",
        "parent_task": "TASK-4",
        "closeout_task": "TASK-4.8",
        "parent_task_file": Path("backlog/tasks/task-4 - Phase-2-Home-Operational-Control.md"),
        "task_file": Path(
            "backlog/tasks/task-4.8 - Phase-2.8-Replay-Home-operational-control-maturity-gate.md"
        ),
        "readme": Path("Docs/superpowers/qa/unified-shell/phase-2/README.md"),
        "closeout_doc": Path(
            "Docs/superpowers/qa/unified-shell/phase-2/2026-05-05-phase-2-home-operational-control-closeout.md"
        ),
        "overview_row": (
            "| Phase 2: Home Operational Control | Make Home a real dashboard/control surface. | "
            "verified |"
        ),
        "task_status": "Done",
    },
    "Phase 3": {
        "qa_dir": "phase-3",
        "status": "verified",
        "parent_task": "TASK-3",
        "closeout_task": "TASK-3.11",
        "parent_task_file": Path("backlog/tasks/task-3 - Phase-3-Console-Live-Work-Hub.md"),
        "task_file": Path(
            "backlog/tasks/task-3.11 - Phase-3.11-Replay-Console-live-work-maturity-gate.md"
        ),
        "readme": Path("Docs/superpowers/qa/unified-shell/phase-3/README.md"),
        "closeout_doc": Path(
            "Docs/superpowers/qa/unified-shell/phase-3/2026-05-05-phase-3-console-live-work-closeout.md"
        ),
        "overview_row": (
            "| Phase 3: Console Live Work Hub | Make Console the live-agent control surface. | "
            "verified |"
        ),
        "task_status": "Done",
    },
    "Phase 4": {
        "qa_dir": "phase-4",
        "status": "verified",
        "parent_task": "TASK-5",
        "closeout_task": "TASK-5.6",
        "parent_task_file": Path("backlog/tasks/task-5 - Phase-4-Destination-Service-Adoption.md"),
        "task_file": Path(
            "backlog/tasks/task-5.6 - Phase-4.6-Replay-destination-service-adoption-maturity-gate.md"
        ),
        "readme": Path("Docs/superpowers/qa/unified-shell/phase-4/README.md"),
        "closeout_doc": Path(
            "Docs/superpowers/qa/unified-shell/phase-4/2026-05-05-phase-4-destination-service-adoption-closeout.md"
        ),
        "overview_row": (
            "| Phase 4: Destination Service Adoption | Turn wrappers into useful product surfaces. | "
            "verified |"
        ),
        "task_status": "Done",
    },
}

def _text(path: Path) -> str:
    resolved_path = path if path.is_absolute() else REPO_ROOT / path
    return resolved_path.read_text(encoding="utf-8")


def _assert_roadmap_status_tracks_current_phase_progress(roadmap_text: str) -> None:
    status_match = re.search(r"^Status:\s*(.+)$", roadmap_text, re.MULTILINE)
    assert status_match is not None
    normalized_status = status_match.group(1).lower().replace("-", " ")

    assert "phase 2" in normalized_status
    assert "phase 3" in normalized_status
    assert "phase 4" in normalized_status
    assert "verified" in normalized_status
    assert re.search(r"phase\s+5\s+verified", normalized_status)
    assert re.search(r"phase\s+6\s+verified", normalized_status)


def _markdown_path(path: Path) -> str:
    relative_path = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    return relative_path.as_posix()


def _extract_phase_metadata(text: str, phase_number: int) -> dict:
    metadata_pattern = re.compile(
        rf"<!-- PHASE_{phase_number}_CLOSEOUT_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        rf"<!-- PHASE_{phase_number}_CLOSEOUT_METADATA:END -->",
        re.DOTALL,
    )
    metadata_match = metadata_pattern.search(text)
    assert metadata_match is not None, f"Metadata for Phase {phase_number} not found"
    return json.loads(metadata_match.group(1))


def test_phase_two_three_four_closeout_tasks_record_current_parent_status():
    for phase in PHASES.values():
        closeout_text = _text(phase["task_file"])
        assert f"id: {phase['closeout_task']}" in closeout_text
        assert f"status: {phase['task_status']}" in closeout_text
        assert f"parent_task_id: {phase['parent_task']}" in closeout_text
        assert "running-app QA walkthrough" in closeout_text
        assert "not render-only or click-only behavior" in closeout_text

        parent_text = _text(phase["parent_task_file"])
        expected_parent_status = "Done" if phase["status"] == "verified" else "In Progress"
        assert f"status: {expected_parent_status}" in parent_text
        assert phase["closeout_task"] in parent_text
        assert "maturity-gate QA" in parent_text


