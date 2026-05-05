from pathlib import Path


ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")

PHASES = {
    "Phase 2": {
        "qa_dir": "phase-2",
        "status": "qa-needed",
        "parent_task": "TASK-4",
        "closeout_task": "TASK-4.8",
        "task_file": Path(
            "backlog/tasks/task-4.8 - Phase-2.8-Replay-Home-operational-control-maturity-gate.md"
        ),
        "readme": Path("Docs/superpowers/qa/unified-shell/phase-2/README.md"),
        "closeout_doc": Path(
            "Docs/superpowers/qa/unified-shell/phase-2/2026-05-05-phase-2-home-operational-control-closeout.md"
        ),
        "overview_row": (
            "| Phase 2: Home Operational Control | Make Home a real dashboard/control surface. | "
            "qa-needed |"
        ),
    },
    "Phase 3": {
        "qa_dir": "phase-3",
        "status": "qa-needed",
        "parent_task": "TASK-3",
        "closeout_task": "TASK-3.11",
        "task_file": Path(
            "backlog/tasks/task-3.11 - Phase-3.11-Replay-Console-live-work-maturity-gate.md"
        ),
        "readme": Path("Docs/superpowers/qa/unified-shell/phase-3/README.md"),
        "closeout_doc": Path(
            "Docs/superpowers/qa/unified-shell/phase-3/2026-05-05-phase-3-console-live-work-closeout.md"
        ),
        "overview_row": (
            "| Phase 3: Console Live Work Hub | Make Console the live-agent control surface. | "
            "qa-needed |"
        ),
    },
    "Phase 4": {
        "qa_dir": "phase-4",
        "status": "qa-needed",
        "parent_task": "TASK-5",
        "closeout_task": "TASK-5.6",
        "task_file": Path(
            "backlog/tasks/task-5.6 - Phase-4.6-Replay-destination-service-adoption-maturity-gate.md"
        ),
        "readme": Path("Docs/superpowers/qa/unified-shell/phase-4/README.md"),
        "closeout_doc": Path(
            "Docs/superpowers/qa/unified-shell/phase-4/2026-05-05-phase-4-destination-service-adoption-closeout.md"
        ),
        "overview_row": (
            "| Phase 4: Destination Service Adoption | Turn wrappers into useful product surfaces. | "
            "qa-needed |"
        ),
    },
}

PARENT_TASK_FILES = {
    "TASK-3": Path("backlog/tasks/task-3 - Phase-3-Console-Live-Work-Hub.md"),
    "TASK-4": Path("backlog/tasks/task-4 - Phase-2-Home-Operational-Control.md"),
    "TASK-5": Path("backlog/tasks/task-5 - Phase-4-Destination-Service-Adoption.md"),
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_phase_two_three_four_roadmap_and_indexes_mark_qa_needed():
    roadmap_text = _text(ROADMAP)

    assert "Status: Phase 2, Phase 3, and Phase 4 need maturity-gate QA" in roadmap_text

    for phase_name, phase in PHASES.items():
        qa_row = (
            f"| {phase_name} | `Docs/superpowers/qa/unified-shell/{phase['qa_dir']}/` | "
            f"{phase['status']} |"
        )
        assert qa_row in roadmap_text
        assert phase["overview_row"] in roadmap_text
        assert str(phase["closeout_doc"]) in roadmap_text
        assert phase["closeout_task"] in roadmap_text

        readme_text = _text(phase["readme"])
        assert "Status: qa-needed" in readme_text
        assert phase["closeout_task"] in readme_text
        assert phase["closeout_doc"].name in readme_text


def test_phase_two_three_four_closeout_tasks_keep_parent_phases_open():
    for phase in PHASES.values():
        closeout_text = _text(phase["task_file"])
        assert f"id: {phase['closeout_task']}" in closeout_text
        assert "status: To Do" in closeout_text
        assert f"parent_task_id: {phase['parent_task']}" in closeout_text
        assert "running-app QA walkthrough" in closeout_text
        assert "not render-only or click-only behavior" in closeout_text

        parent_text = _text(PARENT_TASK_FILES[phase["parent_task"]])
        assert "status: In Progress" in parent_text
        assert phase["closeout_task"] in parent_text
        assert "maturity-gate QA" in parent_text
