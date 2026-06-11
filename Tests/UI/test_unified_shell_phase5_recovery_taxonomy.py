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
        install_targets=(
            'pip install -e ".[embeddings_rag]"',
            'pip install "tldw_chatbook[embeddings_rag]"',
        ),
        stable_selector="search-rag-dependency-missing",
        recovery_action="Settings > RAG",
    )

    assert recovery_state.status_label == "Dependency missing"
    assert recovery_state.unavailable_what == "Search/RAG queries"
    assert recovery_state.why == "Missing optional dependencies: torch, sentence-transformers."
    assert recovery_state.next_action == (
        'Install with pip install -e ".[embeddings_rag]" for source checkouts or '
        'pip install "tldw_chatbook[embeddings_rag]" for packaged installs, then restart.'
    )
    assert recovery_state.recovery_action == "Settings > RAG"
    assert recovery_state.authority_owner == "optional dependency"
    assert recovery_state.stable_selector == "search-rag-dependency-missing"
    assert "Unavailable: Search/RAG queries." in recovery_state.visible_copy
    assert "Why: Missing optional dependencies: torch, sentence-transformers." in recovery_state.visible_copy
    assert 'pip install -e ".[embeddings_rag]"' in recovery_state.disabled_tooltip
    assert 'pip install "tldw_chatbook[embeddings_rag]"' in recovery_state.disabled_tooltip


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

    # asyncio.run on the UI thread is banned. A worker-thread usage is
    # legitimate (no running loop there) but must carry an explicit
    # annotation AND appear in this allowlist with an exact count, so
    # exceptions cannot proliferate silently.
    allowed_annotated_asyncio_run = {
        Path("tldw_chatbook/UI/Screens/library_screen.py"): 1,
    }
    for screen_path in screen_paths:
        source = _text(screen_path)
        assert "thread=True" not in source, screen_path
        annotated = 0
        for line in source.splitlines():
            if "asyncio.run" in line:
                assert "policy-exception: worker-thread loop" in line, (
                    screen_path,
                    line.strip(),
                )
                annotated += 1
        assert annotated == allowed_annotated_asyncio_run.get(screen_path, 0), (
            screen_path,
            annotated,
        )
        assert "_run_maybe_awaitable" not in source, screen_path
