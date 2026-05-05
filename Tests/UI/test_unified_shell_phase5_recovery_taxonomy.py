import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
PHASE_5_README = Path("Docs/superpowers/qa/unified-shell/phase-5/README.md")
PHASE_5_TAXONOMY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md"
)
PHASE_5_DESTINATION_RECOVERY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-destination-blocker-recovery.md"
)
PHASE_5_PARENT_TASK = Path("backlog/tasks/task-6 - Phase-5-Capability-And-Recovery-System.md")
PHASE_5_TAXONOMY_TASK = Path("backlog/tasks/task-6.1 - Phase-5.1-Create-shared-recovery-taxonomy.md")
PHASE_5_DESTINATION_RECOVERY_TASK = Path(
    "backlog/tasks/task-6.2 - Phase-5.2-Apply-recovery-taxonomy-to-shell-destination-blockers.md"
)


def _text(path: Path) -> str:
    resolved_path = path if path.is_absolute() else REPO_ROOT / path
    return resolved_path.read_text(encoding="utf-8")


def _status_line(text: str) -> str:
    status_match = re.search(r"^Status:\s*(.+)$", text, re.MULTILINE)
    assert status_match is not None
    return status_match.group(1).strip()


def _assert_roadmap_tracks_phase_five_progress(roadmap: str) -> None:
    normalized_status = _status_line(roadmap).lower().replace("-", " ")

    assert "phase 2" in normalized_status
    assert "phase 3" in normalized_status
    assert "phase 4" in normalized_status
    assert "verified" in normalized_status
    assert re.search(r"phase\s+5\s+in\s+progress", normalized_status)
    assert re.search(r"phase\s+6\s+not\s+started", normalized_status)


def _roadmap_phase_evidence_row(text: str, phase_name: str) -> list[str]:
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if columns and columns[0] == phase_name:
            return columns
    raise AssertionError(f"{phase_name} evidence row not found")


def _taxonomy_metadata(text: str) -> dict:
    metadata_match = re.search(
        r"<!-- PHASE_5_1_RECOVERY_TAXONOMY_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_5_1_RECOVERY_TAXONOMY_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert metadata_match is not None
    return json.loads(metadata_match.group(1))


def test_phase_five_recovery_taxonomy_is_tracked_from_roadmap_readme_and_tasks():
    roadmap = _text(ROADMAP)
    readme = _text(PHASE_5_README)
    parent_task = _text(PHASE_5_PARENT_TASK)
    child_task = _text(PHASE_5_TAXONOMY_TASK)
    destination_recovery_task = _text(PHASE_5_DESTINATION_RECOVERY_TASK)

    _assert_roadmap_tracks_phase_five_progress(roadmap)
    phase_five_row = _roadmap_phase_evidence_row(roadmap, "Phase 5")
    assert phase_five_row[1] == "`Docs/superpowers/qa/unified-shell/phase-5/`"
    assert phase_five_row[2] == "in-progress"
    assert re.search(r"Phase\s+5\.1:.*shared recovery taxonomy.*`TASK-6\.1`", roadmap, re.IGNORECASE)
    assert re.search(
        r"Phase\s+5\.2:.*shell destination blockers.*`TASK-6\.2`",
        roadmap,
        re.IGNORECASE,
    )
    assert "2026-05-05-shared-recovery-taxonomy.md" in roadmap
    assert "2026-05-05-destination-blocker-recovery.md" in roadmap

    assert _status_line(readme) == "in-progress"
    assert "`TASK-6.1`" in readme
    assert "`TASK-6.2`" in readme
    assert "2026-05-05-shared-recovery-taxonomy.md" in readme
    assert "2026-05-05-destination-blocker-recovery.md" in readme

    assert "status: In Progress" in parent_task
    assert "TASK-6.1" in parent_task
    assert "TASK-6.2" in parent_task
    assert "status: Done" in child_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in child_task
    assert "Implementation Notes" in child_task
    assert "status: Done" in destination_recovery_task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in destination_recovery_task
    assert "Implementation Notes" in destination_recovery_task


def test_phase_five_destination_recovery_evidence_records_applied_blockers():
    evidence = _text(PHASE_5_DESTINATION_RECOVERY)

    assert "/Users/" not in evidence
    assert "TASK-6.2" in evidence
    assert "ACP agent launch" in evidence
    assert "Console follow for Schedules" in evidence
    assert "Console launch for Workflows" in evidence
    assert "Console launch for Chatbook artifacts" in evidence
    assert "test_phase_five_destination_blockers_expose_taxonomy_recovery_fields" in evidence
    assert "13 passed" in evidence


def test_phase_five_recovery_taxonomy_defines_required_contract_and_reason_mappings():
    taxonomy = _text(PHASE_5_TAXONOMY)
    metadata = _taxonomy_metadata(taxonomy)

    assert "/Users/" not in taxonomy
    assert metadata["task"] == "TASK-6.1"
    assert metadata["parent_task"] == "TASK-6"
    assert metadata["decision"] == "foundation_defined"
    assert metadata["required_user_fields"] == [
        "status_label",
        "unavailable_what",
        "why",
        "next_action",
        "recovery_action",
        "authority_owner",
        "stable_selector",
        "disabled_tooltip",
    ]
    assert metadata["canonical_states"] == [
        "wrong_source",
        "server_not_configured",
        "server_unreachable",
        "server_auth_required",
        "server_session_invalid",
        "policy_denied",
        "capability_disabled",
        "runtime_not_configured",
        "service_unavailable",
        "dependency_missing",
        "empty_selection",
    ]
    assert metadata["runtime_policy_reason_codes"] == [
        "wrong_source",
        "server_not_configured",
        "server_profile_missing",
        "server_unreachable",
        "server_unavailable",
        "server_auth_required",
        "auth_required",
        "credential_store_unavailable",
        "server_credentials_unavailable",
        "server_session_invalid",
        "stale_authorization",
        "profile_no_longer_authorized",
        "authority_denied",
        "permission_denied",
        "capability_disabled",
    ]
    assert metadata["destination_recovery_sources"] == [
        "phase-1-destination-action-audit",
        "phase-3-console-live-work-closeout",
        "phase-4-destination-service-adoption-closeout",
        "runtime-policy-domain-edge-contracts",
    ]
