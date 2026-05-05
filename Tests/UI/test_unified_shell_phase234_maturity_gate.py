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
    assert re.search(r"phase\s+5\s+in\s+progress", normalized_status)
    assert re.search(r"phase\s+6\s+not\s+started", normalized_status)


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


def test_phase_two_three_four_roadmap_and_indexes_record_current_gate_status():
    roadmap_text = _text(ROADMAP)

    _assert_roadmap_status_tracks_current_phase_progress(roadmap_text)

    for phase_name, phase in PHASES.items():
        qa_row = (
            f"| {phase_name} | `Docs/superpowers/qa/unified-shell/{phase['qa_dir']}/` | "
            f"{phase['status']} |"
        )
        assert qa_row in roadmap_text
        assert phase["overview_row"] in roadmap_text
        assert _markdown_path(phase["closeout_doc"]) in roadmap_text
        assert phase["closeout_task"] in roadmap_text

        readme_text = _text(phase["readme"])
        assert f"Status: {phase['status']}" in readme_text
        assert phase["closeout_task"] in readme_text
        assert phase["closeout_doc"].name in readme_text


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


def test_phase_two_closeout_doc_records_verified_workflows_and_task_completion():
    phase = PHASES["Phase 2"]
    closeout_text = _text(phase["closeout_doc"])
    metadata = _extract_phase_metadata(closeout_text, 2)

    assert "/Users/" not in closeout_text
    assert metadata["closeout_task"] == "TASK-4.8"
    assert metadata["parent_task"] == "TASK-4"
    assert metadata["decision"] == "verified"
    assert metadata["verified_workflows"] == [
        "approve",
        "reject",
        "pause",
        "resume",
        "retry",
        "open-detail",
        "open-in-console",
        "notification-review",
    ]
    assert metadata["unsupported_controls_policy"] == "explicitly_recoverable"
    assert metadata["baseline_replay_result"]["failed"] == 2
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0
    assert metadata["final_focused_replay_result"]["warnings"] >= 0

    parent_text = _text(phase["parent_task_file"])
    assert "status: Done" in parent_text
    assert "- [x] #1" in parent_text
    assert "- [x] #4" in parent_text

    task_text = _text(phase["task_file"])
    assert "status: Done" in task_text
    assert "- [x] #1" in task_text
    assert "- [x] #3" in task_text
    assert "Implementation Notes" in task_text


def test_phase_three_closeout_doc_records_verified_workflows_and_task_completion():
    phase = PHASES["Phase 3"]
    closeout_text = _text(phase["closeout_doc"])
    metadata = _extract_phase_metadata(closeout_text, 3)

    assert "/Users/" not in closeout_text
    assert metadata["closeout_task"] == "TASK-3.11"
    assert metadata["parent_task"] == "TASK-3"
    assert metadata["decision"] == "verified"
    assert metadata["verified_sources"] == [
        "home-active-work",
        "watchlists-collections",
        "schedules",
        "rag-search",
        "artifacts",
        "workflows",
    ]
    assert metadata["verified_recovery_sources"] == [
        "acp",
        "mcp",
        "event-streams",
    ]
    assert metadata["source_readiness"]["acp"] == "not_wired_recoverable"
    assert metadata["source_readiness"]["mcp"] == "not_wired_recoverable"
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0
    assert metadata["final_broader_replay_result"]["failed"] == 0
    assert metadata["final_broader_replay_result"]["passed"] >= metadata["final_focused_replay_result"]["passed"]

    parent_text = _text(phase["parent_task_file"])
    assert "status: Done" in parent_text
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in parent_text

    task_text = _text(phase["task_file"])
    assert "status: Done" in task_text
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in task_text
    assert "Implementation Notes" in task_text


def test_phase_four_closeout_doc_records_verified_destinations_and_task_completion():
    phase = PHASES["Phase 4"]
    closeout_text = _text(phase["closeout_doc"])
    metadata = _extract_phase_metadata(closeout_text, 4)

    assert "/Users/" not in closeout_text
    assert metadata["closeout_task"] == "TASK-5.6"
    assert metadata["parent_task"] == "TASK-5"
    assert metadata["decision"] == "verified"
    assert metadata["verified_destinations"] == [
        "mcp",
        "skills",
        "library",
        "personas",
        "watchlists-collections",
        "schedules",
        "workflows",
        "artifacts",
        "settings",
    ]
    assert metadata["verified_recovery_destinations"] == ["acp"]
    assert metadata["destination_readiness"]["mcp"] == "connected"
    assert metadata["destination_readiness"]["skills"] == "connected"
    assert metadata["destination_readiness"]["library"] == "connected"
    assert metadata["destination_readiness"]["personas"] == "connected"
    assert metadata["destination_readiness"]["watchlists-collections"] == "connected"
    assert metadata["destination_readiness"]["schedules"] == "connected_or_recoverable"
    assert metadata["destination_readiness"]["workflows"] == "connected_or_recoverable"
    assert metadata["destination_readiness"]["artifacts"] == "connected_or_recoverable"
    assert metadata["destination_readiness"]["settings"] == "connected"
    assert metadata["destination_readiness"]["acp"] == "runtime_not_configured_recoverable"
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0
    assert metadata["final_broader_replay_result"]["failed"] == 0
    assert metadata["final_broader_replay_result"]["passed"] >= metadata["final_focused_replay_result"]["passed"]

    parent_text = _text(phase["parent_task_file"])
    assert "status: Done" in parent_text
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in parent_text

    task_text = _text(phase["task_file"])
    assert "status: Done" in task_text
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in task_text
    assert "Implementation Notes" in task_text
