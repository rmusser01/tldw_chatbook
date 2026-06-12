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


def test_roadmap_links_phase_one_protocol_evidence():
    text = ROADMAP.read_text(encoding="utf-8")

    assert "TASK-2.1" in text
    assert "walkthrough-protocol.md" in text
    assert "walkthrough-template.md" in text
    assert "2026-05-03-phase-1-protocol-smoke.md" in text
