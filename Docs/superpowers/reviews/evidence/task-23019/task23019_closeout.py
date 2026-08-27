"""Task-local contracts for the TASK-23019 adaptive-reader closeout."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path


class CloseoutError(Exception):
    """A stable semantic failure from the closeout runner."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class Contract:
    automated_nodes: tuple[str, ...]
    live_cases: tuple[str, ...]


@dataclass(frozen=True)
class Subject:
    commit: str
    tree: str


@dataclass(frozen=True)
class ParentOptions:
    subject_revision: str | None
    development_run: bool
    live_case: str | None
    live_cases: tuple[str, ...]
    live_only: bool
    no_promote: bool
    promote: bool
    verify_evidence: Path | None


CATALOGUE: dict[str, Contract] = {
    "SH-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_shell.py::test_sync_layout_retains_every_mounted_child_identity",
            "Tests/UI/test_library_media_reader_shell.py::test_media_shell_mounts_library_items_reader_and_two_five_column_grips",
            "Tests/UI/test_library_conversation_reader.py::test_conversations_mount_three_retained_roles_once",
            "Tests/UI/test_library_notes_reader.py::test_database_notes_mount_three_retained_roles_once",
            "Tests/UI/test_library_prompts_reader.py::test_prompts_mount_three_retained_roles_once",
            "Tests/UI/test_library_skills_reader.py::test_skills_mount_three_retained_roles_and_default_to_overview",
        ),
        live_cases=("common_matrix", "single_app_route_cycle"),
    ),
    "SH-02": Contract(
        automated_nodes=(
            "Tests/Library/test_library_adaptive_reader_state.py::test_shared_resolution_uses_adaptive_width_classes",
            "Tests/UI/test_library_adaptive_reader_shell.py::test_all_five_regions_remain_inside_representative_media_widths",
        ),
        live_cases=("common_matrix",),
    ),
    "SH-03": Contract(
        automated_nodes=(
            "Tests/UI/test_library_media_reader_shell.py::test_shared_library_pane_choice_round_trips_between_media_and_conversations",
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_preferences_restore_in_fresh_screen",
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
        ),
        live_cases=("single_app_route_cycle",),
    ),
    "SH-04": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_shell.py::test_hiding_focused_pane_moves_focus_to_truthful_restore_grip",
            "Tests/UI/test_library_media_reader_flow.py::test_footer_advertises_only_working_current_actions",
            "Tests/UI/test_library_conversation_reader.py::test_conversations_global_f6_cycles_visible_destination_roles",
            "Tests/UI/test_library_skills_reader.py::test_skills_reader_f6_reaches_items_and_work_regions",
        ),
        live_cases=("common_matrix", "single_app_route_cycle"),
    ),
    "SH-05": Contract(
        automated_nodes=(
            "Tests/UI/test_library_media_reader_flow.py::test_late_completion_for_a_cannot_replace_loaded_b_or_show_error",
            "Tests/UI/test_library_conversation_reader.py::test_late_previous_selection_cannot_overwrite_current_reader",
            "Tests/Library/test_library_notes_session.py::test_stale_open_session_cannot_replace_a_newer_loaded_session",
            "Tests/UI/test_library_prompts_reader.py::test_same_prompt_older_detail_load_cannot_overwrite_newer_generation",
            "Tests/UI/test_library_skills_reader.py::test_same_skill_older_detail_result_cannot_replace_newer_generation",
        ),
        live_cases=(
            "media_capability",
            "conversations_capability",
            "notes_capability",
            "prompts_capability",
            "skills_capability",
        ),
    ),
    "SH-06": Contract(
        automated_nodes=(
            "Tests/Library/test_library_adaptive_reader_state.py::test_resolution_never_mutates_saved_preferences",
            "Tests/UI/test_library_media_reader_shell.py::test_media_shell_resize_uses_resolver_without_reads_or_recompose",
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_resize_is_presentation_only",
        ),
        live_cases=("resize_purity",),
    ),
    "SH-07": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
        ),
        live_cases=("single_app_route_cycle",),
    ),
    "ME-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_media_reader_flow.py::test_reader_defaults_to_read_and_keeps_mode_across_local_items",
            "Tests/UI/test_library_media_reader_flow.py::test_progress_restores_after_loaded_content_mounts",
        ),
        live_cases=("media_capability",),
    ),
    "ME-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_multiselect_media.py::test_confirming_bulk_delete_swaps_toolbar_for_confirm_row",
            "Tests/UI/test_library_multiselect_media.py::test_delete_selection_soft_deletes_via_real_db_and_updates_records_and_counts",
        ),
        live_cases=("media_capability",),
    ),
    "CO-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_conversation_reader.py::test_progressive_reader_paints_first_page_then_completes_find_off_loop",
            "Tests/UI/test_library_conversation_reader.py::test_reader_info_is_explicit_and_truthful",
        ),
        live_cases=("conversations_capability",),
    ),
    "CO-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_conversation_reader.py::test_open_console_requires_final_complete_error_free_match",
            "Tests/UI/test_library_conversation_reader.py::test_authoritative_refresh_marks_selected_conversation_deleted_without_fallback",
        ),
        live_cases=("conversations_capability",),
    ),
    "NO-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_notes_reader.py::test_list_and_work_identity_survive_open_preview_info_and_edit",
        ),
        live_cases=("notes_capability",),
    ),
    "NO-02": Contract(
        automated_nodes=(
            "Tests/Library/test_library_notes_session.py::test_conflict_stops_chaining_and_preserves_the_newest_draft",
            "Tests/UI/test_library_multiselect_notes.py::test_permanent_navigator_tasks_respect_dirty_draft_veto",
        ),
        live_cases=("notes_capability",),
    ),
    "PR-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_prompts_reader.py::test_basic_save_preserves_advanced_only_prompt_fields",
            "Tests/UI/test_library_prompts_reader.py::test_invalid_advanced_block_routes_save_focus_to_its_owner",
        ),
        live_cases=("prompts_capability",),
    ),
    "PR-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_prompts_reader.py::test_import_replaces_only_work_content_and_keeps_list_mounted",
            "Tests/UI/test_library_prompts_reader.py::test_detail_failure_keeps_prior_prompt_locked_and_retry_loads_selection",
        ),
        live_cases=("prompts_capability",),
    ),
    "SK-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_skills_reader.py::test_skill_modes_preserve_list_work_and_one_live_draft",
            "Tests/UI/test_library_skills_reader.py::test_skills_trust_mode_identifies_exact_review_snapshot",
        ),
        live_cases=("skills_capability",),
    ),
    "SK-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_skills_reader.py::test_skills_files_mode_is_read_only_and_labels_binary_files",
            "Tests/UI/test_library_skills_reader.py::test_same_skill_older_trust_review_cannot_patch_newer_generation",
            "Tests/UI/test_library_skills_reader.py::test_same_skill_older_delete_cannot_reset_a_newer_work_generation",
        ),
        live_cases=("skills_capability",),
    ),
}

EXPECTED_CONTRACT_IDS = frozenset(
    {
        "SH-01",
        "SH-02",
        "SH-03",
        "SH-04",
        "SH-05",
        "SH-06",
        "SH-07",
        "ME-01",
        "ME-02",
        "CO-01",
        "CO-02",
        "NO-01",
        "NO-02",
        "PR-01",
        "PR-02",
        "SK-01",
        "SK-02",
    }
)

SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))
DESTINATIONS = ("media", "conversations", "notes", "prompts", "skills")
CURATED_PYTEST_FILES = (
    "Tests/Library/test_library_adaptive_reader_state.py",
    "Tests/Library/test_library_media_reader_state.py",
    "Tests/Library/test_library_conversation_reader_state.py",
    "Tests/Library/test_library_notes_session.py",
    "Tests/Library/test_library_prompts_seam.py",
    "Tests/Library/test_library_skills_reader_state.py",
    "Tests/UI/test_library_adaptive_reader_shell.py",
    "Tests/UI/test_library_media_reader_shell.py",
    "Tests/UI/test_library_media_reader_flow.py",
    "Tests/UI/test_library_conversation_reader.py",
    "Tests/UI/test_library_notes_reader.py",
    "Tests/UI/test_library_prompts_reader.py",
    "Tests/UI/test_library_skills_reader.py",
    "Tests/UI/test_library_multiselect_media.py",
    "Tests/UI/test_library_multiselect_conversations.py",
    "Tests/UI/test_library_multiselect_notes.py",
    "Tests/UI/test_library_adaptive_reader_closeout.py",
    "Tests/Chat/test_chat_conversation_service.py",
    "Tests/Prompt_Management/test_prompt_preservation.py",
    "Tests/Skills/test_skills_library_flow.py",
    "Tests/Skills/test_skill_trust_service.py",
    "Tests/Skills/test_local_skills_bundle_io.py",
    "Tests/Skills/test_read_skill_file.py",
    "Tests/Skills/test_skill_file_trust_material.py",
)
DECLARED_LIVE_CASES = frozenset(
    case for contract in CATALOGUE.values() for case in contract.live_cases
)


def validate_catalogue(catalogue: Mapping[str, Contract]) -> None:
    """Reject an incomplete or ambiguous contract catalogue."""
    if set(catalogue) != EXPECTED_CONTRACT_IDS:
        raise CloseoutError("catalogue_ids_mismatch")
    for contract in catalogue.values():
        if not contract.automated_nodes:
            raise CloseoutError("automated_nodes_missing")
        if not contract.live_cases:
            raise CloseoutError("live_cases_missing")
        if len(contract.live_cases) != len(set(contract.live_cases)):
            raise CloseoutError("live_key_duplicate")


def matching_node_ids(selector: str, node_ids: Collection[str]) -> tuple[str, ...]:
    """Return exact or parameterized concrete node IDs for one selector."""
    parameterized_prefix = selector + "["
    return tuple(
        sorted(
            node_id
            for node_id in node_ids
            if node_id == selector or node_id.startswith(parameterized_prefix)
        )
    )


def validate_collected_selectors(
    catalogue: Mapping[str, Contract], collected_node_ids: Collection[str]
) -> None:
    """Require every declared selector to exist in a synthetic/real collection."""
    for contract in catalogue.values():
        for selector in contract.automated_nodes:
            if not matching_node_ids(selector, collected_node_ids):
                raise CloseoutError("pytest_selector_not_collected")


def validate_automated_results(
    catalogue: Mapping[str, Contract], results: Mapping[str, str]
) -> None:
    """Require all concrete nodes matching every selector to settle PASS."""
    validate_collected_selectors(catalogue, results.keys())
    for contract in catalogue.values():
        for selector in contract.automated_nodes:
            if any(results[node_id] != "PASS" for node_id in matching_node_ids(selector, results)):
                raise CloseoutError("pytest_node_not_pass")


def _git(repo: Path, *args: str) -> str:
    """Run one git query without a shell."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def admit_subject(repo: Path, requested: str) -> Subject:
    """Admit only an exact clean HEAD and record its source tree."""
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if head != requested:
        raise CloseoutError("subject_revision_mismatch")
    if _git(repo, "status", "--porcelain"):
        raise CloseoutError("subject_worktree_not_clean")
    return Subject(commit=head, tree=tree)


def verify_subject_tree(repo: Path, subject: Subject) -> None:
    """Reject a later HEAD whose tree differs from the admitted subject tree."""
    if _git(repo, "rev-parse", "HEAD^{tree}") != subject.tree:
        raise CloseoutError("subject_tree_mismatch")


def build_parser() -> argparse.ArgumentParser:
    """Build the parent CLI parser without running later-task behavior."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--subject-revision", metavar="REV")
    parser.add_argument("--development-run", action="store_true")
    parser.add_argument("--live-case", metavar="NAME")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--verify-evidence", metavar="PATH")
    return parser


def parse_options(arguments: list[str] | None = None) -> ParentOptions:
    """Parse syntax with argparse, then reject semantic misuse stably."""
    parsed = build_parser().parse_args(arguments)
    subject_revision_provided = parsed.subject_revision is not None
    live_case_provided = parsed.live_case is not None
    verify_evidence_provided = parsed.verify_evidence is not None
    if parsed.development_run and subject_revision_provided:
        raise CloseoutError("development_subject_revision_conflict")
    if parsed.development_run and parsed.promote:
        raise CloseoutError("development_promotion_conflict")
    if parsed.development_run and verify_evidence_provided:
        raise CloseoutError("development_verify_evidence_conflict")
    if live_case_provided and parsed.live_only:
        raise CloseoutError("live_selection_conflict")
    if subject_revision_provided and not parsed.subject_revision.strip():
        raise CloseoutError("subject_revision_empty")
    if verify_evidence_provided and not parsed.verify_evidence.strip():
        raise CloseoutError("verify_evidence_path_empty")
    if parsed.promote and not subject_revision_provided:
        raise CloseoutError("promotion_subject_required")
    if parsed.promote and parsed.no_promote:
        raise CloseoutError("promotion_mode_conflict")
    if live_case_provided and parsed.live_case not in DECLARED_LIVE_CASES:
        raise CloseoutError("scenario_not_defined")

    live_cases = (
        tuple(sorted(DECLARED_LIVE_CASES))
        if parsed.live_only
        else ((parsed.live_case,) if parsed.live_case else ())
    )
    return ParentOptions(
        subject_revision=parsed.subject_revision,
        development_run=parsed.development_run,
        live_case=parsed.live_case,
        live_cases=live_cases,
        live_only=parsed.live_only,
        no_promote=parsed.no_promote or parsed.development_run,
        promote=parsed.promote and not parsed.development_run,
        verify_evidence=Path(parsed.verify_evidence) if verify_evidence_provided else None,
    )


validate_catalogue(CATALOGUE)
