import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
PHASE_5_README = Path("Docs/superpowers/qa/unified-shell/phase-5/README.md")
PHASE_5_TAXONOMY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md"
)
PHASE_5_PARENT_TASK = Path("backlog/tasks/task-6 - Phase-5-Capability-And-Recovery-System.md")
PHASE_5_TAXONOMY_TASK = Path("backlog/tasks/task-6.1 - Phase-5.1-Create-shared-recovery-taxonomy.md")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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

    assert "Status: Phase 2, Phase 3, and Phase 4 verified; Phase 5 in progress; Phase 6 not started" in roadmap
    assert "| Phase 5 | `Docs/superpowers/qa/unified-shell/phase-5/` | in-progress |" in roadmap
    assert "Phase 5.1: Create shared recovery taxonomy - `TASK-6.1`" in roadmap
    assert "2026-05-05-shared-recovery-taxonomy.md" in roadmap

    assert "Status: in-progress" in readme
    assert "`TASK-6.1`" in readme
    assert "2026-05-05-shared-recovery-taxonomy.md" in readme

    assert "status: In Progress" in parent_task
    assert "TASK-6.1" in parent_task
    assert "status: Done" in child_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in child_task
    assert "Implementation Notes" in child_task


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
        "server_unreachable",
        "server_auth_required",
        "server_session_invalid",
        "authority_denied",
        "capability_disabled",
    ]
    assert metadata["destination_recovery_sources"] == [
        "phase-1-destination-action-audit",
        "phase-3-console-live-work-closeout",
        "phase-4-destination-service-adoption-closeout",
        "runtime-policy-domain-edge-contracts",
    ]
