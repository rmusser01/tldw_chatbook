from pathlib import Path


QA_ROOT = Path("Docs/superpowers/qa/unified-shell")
PHASE_1_ROOT = QA_ROOT / "phase-1"
PROTOCOL = PHASE_1_ROOT / "walkthrough-protocol.md"
TEMPLATE = PHASE_1_ROOT / "walkthrough-template.md"
SMOKE_SUMMARY = PHASE_1_ROOT / "2026-05-03-phase-1-protocol-smoke.md"
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")

REQUIRED_TEMPLATE_FIELDS = {
    "Environment",
    "Entry Path",
    "Steps Attempted",
    "Visual Usability Notes",
    "Keyboard Path Result",
    "Mouse/Click Path Result",
    "Functional Result",
    "Defect Severity",
    "Evidence",
    "Residual Risk",
}

REQUIRED_SEVERITIES = {
    "blocker",
    "workflow-degradation",
    "recoverability",
    "polish",
}


def test_phase_one_qa_protocol_and_template_exist():
    assert PROTOCOL.exists()
    assert TEMPLATE.exists()
    assert SMOKE_SUMMARY.exists()


def test_phase_one_template_captures_required_walkthrough_fields():
    text = TEMPLATE.read_text(encoding="utf-8")

    for field in REQUIRED_TEMPLATE_FIELDS:
        assert f"## {field}" in text
    for severity in REQUIRED_SEVERITIES:
        assert severity in text


def test_phase_one_protocol_is_runnable_against_textual_app():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "python3 -m tldw_chatbook.app" in text
    assert "Tests/UI/test_master_shell_navigation.py" in text
    assert "actual Textual app" in text
    assert "render-only" in text
    assert "click-event-only" in text


def test_phase_one_smoke_summary_records_boundary_and_evidence():
    text = SMOKE_SUMMARY.read_text(encoding="utf-8")

    assert "TASK-2.1" in text
    assert "Phase 1.1" in text
    assert "Product QA Boundary" in text
    assert "Tests/UI/test_master_shell_navigation.py" in text
    assert "not a full destination workflow audit" in text


def test_roadmap_links_phase_one_protocol_evidence():
    text = ROADMAP.read_text(encoding="utf-8")

    assert "TASK-2.1" in text
    assert "walkthrough-protocol.md" in text
    assert "walkthrough-template.md" in text
    assert "2026-05-03-phase-1-protocol-smoke.md" in text
