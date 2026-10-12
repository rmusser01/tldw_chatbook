"""Critical Personal Context regressions must execute in required PR lanes.

The full native/wheel qualification stays outside fast lanes. These explicit
memberships protect the introduced privacy/currentness/closed-route contracts;
adding an unrelated test cannot substitute for removing one of these targets.
"""

import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Separate source-bound admission suites from the ordinary sandboxed invocation.
CRITICAL = {
    "Run fast PR contract": [
        "Tests/Personal_Context/test_context_service.py::test_device_only_additions_cannot_change_context_selection_or_explanation",
        "Tests/Personal_Context/test_context_service.py::test_quarantine_hint_cannot_change_permitted_packing",
        "Tests/Personal_Context/test_context_service.py::test_explanation_uses_the_snapshot_pass_and_excludes_ineligible_records",
        "Tests/Personal_Context/test_context_service.py::test_explanation_expiry_invalidates_without_manifest_revision_change",
        "Tests/Personal_Context/test_context_service.py::test_explanation_rejects_record_version_reuse_of_snapshot_cache_key",
        "Tests/Personal_Context/test_context_service.py::test_explanation_rejects_replacement_profile_with_reused_revisions",
        "Tests/Agents/test_profile_tool_provider.py::test_device_only_reads_hide_content_and_preserve_manual_owner_access",
        "Tests/Personal_Context/test_native_profile_barrier.py::test_v2_manifest_blocks_readable_v1_before_content_authentication",
        "Tests/Personal_Context/test_native_profile_barrier.py::test_compiled_inventory_covers_every_protected_native_call_site",
        "Tests/Personal_Context/test_native_profile_barrier.py::test_reused_operation_rechecks_manifest_after_external_commit",
        "Tests/Personal_Context/test_native_profile_barrier_routes.py::test_real_v1_write_routes_then_current_v2_deny_without_side_effects[record]",
        "Tests/Personal_Context/test_native_profile_barrier_routes.py::test_real_v1_write_routes_then_current_v2_deny_without_side_effects[proposal]",
        "Tests/Personal_Context/test_native_profile_barrier_routes.py::test_exports_deny_before_destination_write",
        "packages/tldw_profile_core/tests/test_v2_aggregates.py::test_duplicate_aware_and_compiled_json_paths",
        "packages/tldw_profile_core/tests/test_v2_aggregates.py::test_every_serializer_revalidates_unsafe_root_state",
        "packages/tldw_profile_core/tests/test_v2_meaning.py::test_fixed_projection_and_json_round_trip",
        "packages/tldw_profile_core/tests/test_v2_meaning.py::test_fixed_digest_and_scope_cannot_transplant_meaning",
        "packages/tldw_profile_core/tests/test_evidence_binding.py::test_complete_identity_changes_when_one_bound_value_changes",
        "packages/tldw_profile_core/tests/test_evidence_binding.py::test_digest_and_instance_admission_revalidate_unsafe_copies",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[goal-canvas]",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[native-tight]",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[fenced-tight]",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[unreadable]",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[canvas-denied]",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_changes_during_project_preview_discard_snapshot",
        "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_changes_discard_late_profile_preview",
    ],
    "Run admission-sensitive suites": ["Tests/UI/test_app_factory_cleanup.py"],
    "UI census": [
        "Tests/UI/test_settings_personal_context.py::test_export_and_deleted_labels_keep_literal_metadata",
        "Tests/UI/test_personal_context_provenance.py",
        "Tests/UI/test_console_personal_context_selection_factory.py",
        "Tests/UI/test_personal_context_proposal_review.py::test_provenance_renewal_preserves_proposal_editor_and_acceptance",
        "Tests/UI/test_settings_personal_context.py::test_canonical_settings_pending_provenance_uses_cheap_current_owner",
        "Tests/UI/test_settings_personal_context.py::test_production_settings_inspects_private_and_deleted_metadata[size0]",
        "Tests/UI/test_console_conversation_inspector.py::test_detail_worker_creation_can_be_cancelled_before_coroutine_exists",
    ],
}


def lane_targets(lane):
    if lane == "UI census":
        return tuple(
            line.strip()
            for line in (ROOT / "scripts/ui_pr_gate_census.txt")
            .read_text()
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/derived-artifacts.yml").read_text()
    )
    step = next(
        s for s in workflow["jobs"]["pr-fast-lane"]["steps"] if s.get("name") == lane
    )
    return tuple(shlex.split(step["run"].replace("\\\n", " ")))


@pytest.mark.parametrize("lane", CRITICAL)
def test_critical_personal_context_regressions_remain_in_required_lane(lane):
    actual = lane_targets(lane)
    missing = set(CRITICAL[lane]) - set(actual)
    assert not missing, f"Restore critical regressions to {lane}: {sorted(missing)}"
    if lane == "Run fast PR contract":
        assert "Tests/CI" in actual, "This membership contract must itself run on PRs"


def test_admission_sensitive_original_owner_controls_do_not_join_sandbox_lane():
    assert not set(CRITICAL["Run admission-sensitive suites"]) & set(
        lane_targets("Run fast PR contract")
    )
