import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.UI.Screens.destination_recovery import (
    optional_dependency_recovery_state,
    policy_denied_recovery_state,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md")
PHASE_5_README = Path("Docs/superpowers/qa/unified-shell/phase-5/README.md")
PHASE_5_TAXONOMY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md"
)
PHASE_5_DESTINATION_RECOVERY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-destination-blocker-recovery.md"
)
PHASE_5_RUNTIME_POLICY_RECOVERY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-runtime-policy-recovery.md"
)
PHASE_5_OPTIONAL_DEPENDENCY_RECOVERY = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-optional-dependency-recovery.md"
)
PHASE_5_CLOSEOUT = Path(
    "Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-phase-5-capability-recovery-closeout.md"
)
PHASE_5_PARENT_TASK = Path("backlog/tasks/task-6 - Phase-5-Capability-And-Recovery-System.md")
PHASE_5_TAXONOMY_TASK = Path("backlog/tasks/task-6.1 - Phase-5.1-Create-shared-recovery-taxonomy.md")
PHASE_5_DESTINATION_RECOVERY_TASK = Path(
    "backlog/tasks/task-6.2 - Phase-5.2-Apply-recovery-taxonomy-to-shell-destination-blockers.md"
)
PHASE_5_RUNTIME_POLICY_TASK = Path(
    "backlog/tasks/task-6.3 - Phase-5.3-Apply-recovery-taxonomy-to-runtime-policy-blockers.md"
)
PHASE_5_OPTIONAL_DEPENDENCY_TASK = Path(
    "backlog/tasks/task-6.4 - Phase-5.4-Apply-recovery-taxonomy-to-optional-dependency-blockers.md"
)
PHASE_5_CLOSEOUT_TASK = Path(
    "backlog/tasks/task-6.5 - Phase-5.5-Replay-capability-recovery-maturity-gate.md"
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
    assert re.search(r"phase\s+5\s+verified", normalized_status)
    assert re.search(r"phase\s+6\s+verified", normalized_status)


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


def _phase_five_closeout_metadata(text: str) -> dict:
    metadata_match = re.search(
        r"<!-- PHASE_5_CLOSEOUT_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_5_CLOSEOUT_METADATA:END -->",
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
    runtime_policy_task = _text(PHASE_5_RUNTIME_POLICY_TASK)
    optional_dependency_task = _text(PHASE_5_OPTIONAL_DEPENDENCY_TASK)

    _assert_roadmap_tracks_phase_five_progress(roadmap)
    phase_five_row = _roadmap_phase_evidence_row(roadmap, "Phase 5")
    assert phase_five_row[1] == "`Docs/superpowers/qa/unified-shell/phase-5/`"
    assert phase_five_row[2] == "verified"
    assert re.search(r"Phase\s+5\.1:.*shared recovery taxonomy.*`TASK-6\.1`", roadmap, re.IGNORECASE)
    assert re.search(
        r"Phase\s+5\.2:.*shell destination blockers.*`TASK-6\.2`",
        roadmap,
        re.IGNORECASE,
    )
    assert re.search(
        r"Phase\s+5\.3:.*runtime-policy blockers.*`TASK-6\.3`",
        roadmap,
        re.IGNORECASE,
    )
    assert re.search(
        r"Phase\s+5\.4:.*optional dependency blockers.*`TASK-6\.4`",
        roadmap,
        re.IGNORECASE,
    )
    assert re.search(
        r"Phase\s+5\.5:.*capability recovery maturity gate.*`TASK-6\.5`",
        roadmap,
        re.IGNORECASE,
    )
    assert "2026-05-05-shared-recovery-taxonomy.md" in roadmap
    assert "2026-05-05-destination-blocker-recovery.md" in roadmap
    assert "2026-05-05-runtime-policy-recovery.md" in roadmap
    assert "2026-05-05-optional-dependency-recovery.md" in roadmap
    assert "2026-05-05-phase-5-capability-recovery-closeout.md" in roadmap

    assert _status_line(readme) == "verified"
    assert "`TASK-6.1`" in readme
    assert "`TASK-6.2`" in readme
    assert "`TASK-6.3`" in readme
    assert "`TASK-6.4`" in readme
    assert "`TASK-6.5`" in readme
    assert "2026-05-05-shared-recovery-taxonomy.md" in readme
    assert "2026-05-05-destination-blocker-recovery.md" in readme
    assert "2026-05-05-runtime-policy-recovery.md" in readme
    assert "2026-05-05-optional-dependency-recovery.md" in readme
    assert "2026-05-05-phase-5-capability-recovery-closeout.md" in readme

    assert "status: Done" in parent_task
    assert "TASK-6.1" in parent_task
    assert "TASK-6.2" in parent_task
    assert "TASK-6.3" in parent_task
    assert "TASK-6.4" in parent_task
    assert "TASK-6.5" in parent_task
    assert "status: Done" in child_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in child_task
    assert "Implementation Notes" in child_task
    assert "status: Done" in destination_recovery_task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in destination_recovery_task
    assert "Implementation Notes" in destination_recovery_task
    assert "status: Done" in runtime_policy_task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in runtime_policy_task
    assert "Implementation Notes" in runtime_policy_task
    assert "status: Done" in optional_dependency_task
    for acceptance_criterion in range(1, 6):
        assert f"- [x] #{acceptance_criterion}" in optional_dependency_task
    assert "Implementation Notes" in optional_dependency_task


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


def test_phase_five_runtime_policy_recovery_evidence_records_applied_blockers():
    evidence = _text(PHASE_5_RUNTIME_POLICY_RECOVERY)

    assert "/Users/" not in evidence
    assert "TASK-6.3" in evidence
    assert "wrong_source" in evidence
    assert "server_auth_required" in evidence
    assert "server_session_invalid" in evidence
    assert "authority_denied" in evidence
    assert "Skills" in evidence
    assert "Library" in evidence
    assert "Personas" in evidence
    assert "W+C" in evidence
    assert "test_watchlists_collections_policy_denial_uses_runtime_recovery_taxonomy" in evidence
    assert "4 passed" in evidence


def test_phase_five_optional_dependency_recovery_evidence_records_applied_blockers():
    evidence = _text(PHASE_5_OPTIONAL_DEPENDENCY_RECOVERY)

    assert "/Users/" not in evidence
    assert "TASK-6.4" in evidence
    assert "Search/RAG" in evidence
    assert "Local speech" in evidence
    assert "dependency_missing" in evidence
    assert "test_search_rag_missing_embeddings_dependency_exposes_phase_five_recovery" in evidence
    assert "test_stts_missing_speech_dependencies_expose_phase_five_recovery" in evidence


def test_phase_five_closeout_doc_records_verified_recovery_paths_and_task_completion():
    closeout = _text(PHASE_5_CLOSEOUT)
    metadata = _phase_five_closeout_metadata(closeout)

    assert "/Users/" not in closeout
    assert metadata["closeout_task"] == "TASK-6.5"
    assert metadata["parent_task"] == "TASK-6"
    assert metadata["decision"] == "verified"
    assert metadata["verified_recovery_families"] == [
        "destination-blockers",
        "runtime-policy",
        "optional-dependencies",
    ]
    assert metadata["verified_blocked_states"] == [
        "runtime_not_configured",
        "empty_selection",
        "wrong_source",
        "server_auth_required",
        "server_session_invalid",
        "policy_denied",
        "dependency_missing",
    ]
    assert metadata["required_recovery_fields"] == [
        "status_label",
        "unavailable_what",
        "why",
        "next_action",
        "recovery_action",
        "authority_owner",
        "stable_selector",
        "disabled_tooltip",
    ]
    assert metadata["final_focused_replay_result"]["failed"] == 0
    assert metadata["final_focused_replay_result"]["passed"] > 0
    assert metadata["final_broader_replay_result"]["failed"] == 0
    assert metadata["final_broader_replay_result"]["passed"] >= metadata["final_focused_replay_result"]["passed"]

    readme = _text(PHASE_5_README)
    roadmap = _text(ROADMAP)
    parent_task = _text(PHASE_5_PARENT_TASK)
    closeout_task = _text(PHASE_5_CLOSEOUT_TASK)

    assert _status_line(readme) == "verified"
    assert PHASE_5_CLOSEOUT.name in readme
    assert _roadmap_phase_evidence_row(roadmap, "Phase 5")[2] == "verified"
    assert str(PHASE_5_CLOSEOUT).replace("\\", "/") in roadmap
    assert "TASK-6.5" in roadmap

    assert "status: Done" in parent_task
    assert "TASK-6.5" in parent_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in parent_task

    assert "status: Done" in closeout_task
    for acceptance_criterion in range(1, 5):
        assert f"- [x] #{acceptance_criterion}" in closeout_task
    assert "Implementation Notes" in closeout_task


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


@pytest.mark.parametrize(
    ("reason_code", "status_label", "next_action", "recovery_action"),
    [
        (
            "wrong_source",
            "Wrong source",
            "Switch to the required source, then retry this workflow.",
            "Source switch or Settings",
        ),
        (
            "server_not_configured",
            "Server not configured",
            "Add an active server profile in Settings before retrying.",
            "Settings",
        ),
        (
            "server_auth_required",
            "Server sign-in required",
            "Reconnect or configure server credentials in Settings before retrying.",
            "Settings",
        ),
        (
            "server_session_invalid",
            "Server session expired",
            "Re-authenticate the active server profile before retrying.",
            "Settings",
        ),
        (
            "capability_disabled",
            "Capability disabled",
            "Enable this capability in Settings or the governing policy before retrying.",
            "Settings or governing policy",
        ),
        (
            "authority_denied",
            "Policy denied",
            "Review workspace policy or ask the authority owner to allow this action.",
            "Workspace policy",
        ),
    ],
)
def test_phase_five_runtime_policy_recovery_helper_maps_reason_groups(
    reason_code,
    status_label,
    next_action,
    recovery_action,
):
    exc = PolicyDeniedError(
        action_id="test.action",
        reason_code=reason_code,
        user_message="Policy message from service.",
        effective_source="server",
        authority_owner="active server",
    )

    recovery_state = policy_denied_recovery_state(
        exc,
        unavailable_what="Test workflow",
        stable_selector="test-recovery",
    )

    assert recovery_state.status_label == status_label
    assert recovery_state.why == "Policy message from service."
    assert recovery_state.next_action == next_action
    assert recovery_state.recovery_action == recovery_action
    assert recovery_state.authority_owner == "active server"
    assert "Policy message from service." in recovery_state.disabled_tooltip
    assert next_action in recovery_state.disabled_tooltip


def test_phase_five_recovery_copy_preserves_policy_message_terminal_punctuation():
    exc = PolicyDeniedError(
        action_id="test.action",
        reason_code="authority_denied",
        user_message="Allow this action?",
        effective_source="server",
        authority_owner="workspace policy!",
    )

    recovery_state = policy_denied_recovery_state(
        exc,
        unavailable_what="Test workflow!",
        stable_selector="test-recovery",
    )

    assert recovery_state.why == "Allow this action?"
    assert recovery_state.authority_owner == "workspace policy!"
    assert "Unavailable: Test workflow!" in recovery_state.visible_copy
    assert "Why: Allow this action?" in recovery_state.visible_copy
    assert "Owner: workspace policy!" in recovery_state.visible_copy
    assert "Allow this action?" in recovery_state.disabled_tooltip


def test_phase_five_optional_dependency_recovery_helper_builds_required_fields():
    recovery_state = optional_dependency_recovery_state(
        unavailable_what="Search/RAG queries",
        missing_dependencies=("torch", "sentence-transformers"),
        install_target='pip install -e ".[embeddings_rag]"',
        stable_selector="search-rag-dependency-missing",
        recovery_action="Settings > RAG",
    )

    assert recovery_state.status_label == "Dependency missing"
    assert recovery_state.unavailable_what == "Search/RAG queries"
    assert recovery_state.why == "Missing optional dependencies: torch, sentence-transformers."
    assert recovery_state.next_action == 'Install with pip install -e ".[embeddings_rag]" and restart.'
    assert recovery_state.recovery_action == "Settings > RAG"
    assert recovery_state.authority_owner == "optional dependency"
    assert recovery_state.stable_selector == "search-rag-dependency-missing"
    assert "Unavailable: Search/RAG queries." in recovery_state.visible_copy
    assert "Why: Missing optional dependencies: torch, sentence-transformers." in recovery_state.visible_copy
    assert 'Install with pip install -e ".[embeddings_rag]" and restart.' in recovery_state.disabled_tooltip


def test_search_rag_window_imports_without_screens_recovery_cycle():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tldw_chatbook.UI.Views.RAGSearch.search_rag_window import SearchRAGWindow; "
            "print(SearchRAGWindow.__name__)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr
    assert "SearchRAGWindow" in result.stdout


def test_service_backed_policy_destinations_use_async_workers_without_asyncio_run():
    screen_paths = [
        Path("tldw_chatbook/UI/Screens/library_screen.py"),
        Path("tldw_chatbook/UI/Screens/personas_screen.py"),
        Path("tldw_chatbook/UI/Screens/skills_screen.py"),
    ]

    for screen_path in screen_paths:
        source = _text(screen_path)
        assert "thread=True" not in source, screen_path
        assert "asyncio.run" not in source, screen_path
        assert "_run_maybe_awaitable" not in source, screen_path
