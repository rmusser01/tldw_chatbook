from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py"
)

EXPECTED_IDS = {
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

EXPECTED_CURATED_PYTEST_FILES = (
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


def _load_runner():
    spec = importlib.util.spec_from_file_location("task23019_closeout", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_git(*, head: str = "abc", tree: str = "tree-1", dirty: bool = False):
    def fake_git(_repo: Path, *args: str) -> str:
        outputs = {
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "HEAD^{tree}"): tree,
            ("status", "--porcelain"): " M changed.py" if dirty else "",
        }
        return outputs[args]

    return fake_git


def _error_category(error: pytest.ExceptionInfo[Exception]) -> str:
    return getattr(error.value, "category")


def test_catalogue_is_finite_unique_and_has_both_evidence_kinds():
    module = _load_runner()

    assert set(module.CATALOGUE) == EXPECTED_IDS
    assert all(entry.automated_nodes for entry in module.CATALOGUE.values())
    assert all(entry.live_cases for entry in module.CATALOGUE.values())
    assert all(isinstance(entry, module.Contract) for entry in module.CATALOGUE.values())
    assert module.Contract.__dataclass_params__.frozen is True


def test_catalogue_contains_the_exact_declared_contract_mapping():
    module = _load_runner()
    contract = module.Contract
    expected = {
        "SH-01": contract(
            (
                "Tests/UI/test_library_adaptive_reader_shell.py::test_sync_layout_retains_every_mounted_child_identity",
                "Tests/UI/test_library_media_reader_shell.py::test_media_shell_mounts_library_items_reader_and_two_five_column_grips",
                "Tests/UI/test_library_conversation_reader.py::test_conversations_mount_three_retained_roles_once",
                "Tests/UI/test_library_notes_reader.py::test_database_notes_mount_three_retained_roles_once",
                "Tests/UI/test_library_prompts_reader.py::test_prompts_mount_three_retained_roles_once",
                "Tests/UI/test_library_skills_reader.py::test_skills_mount_three_retained_roles_and_default_to_overview",
            ),
            ("common_matrix", "single_app_route_cycle"),
        ),
        "SH-02": contract(
            (
                "Tests/Library/test_library_adaptive_reader_state.py::test_shared_resolution_uses_adaptive_width_classes",
                "Tests/UI/test_library_adaptive_reader_shell.py::test_all_five_regions_remain_inside_representative_media_widths",
            ),
            ("common_matrix",),
        ),
        "SH-03": contract(
            (
                "Tests/UI/test_library_media_reader_shell.py::test_shared_library_pane_choice_round_trips_between_media_and_conversations",
                "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_preferences_restore_in_fresh_screen",
                "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
            ),
            ("single_app_route_cycle",),
        ),
        "SH-04": contract(
            (
                "Tests/UI/test_library_adaptive_reader_shell.py::test_hiding_focused_pane_moves_focus_to_truthful_restore_grip",
                "Tests/UI/test_library_media_reader_flow.py::test_footer_advertises_only_working_current_actions",
                "Tests/UI/test_library_conversation_reader.py::test_conversations_global_f6_cycles_visible_destination_roles",
                "Tests/UI/test_library_skills_reader.py::test_skills_reader_f6_reaches_items_and_work_regions",
            ),
            ("common_matrix", "single_app_route_cycle"),
        ),
        "SH-05": contract(
            (
                "Tests/UI/test_library_media_reader_flow.py::test_late_completion_for_a_cannot_replace_loaded_b_or_show_error",
                "Tests/UI/test_library_conversation_reader.py::test_late_previous_selection_cannot_overwrite_current_reader",
                "Tests/Library/test_library_notes_session.py::test_stale_open_session_cannot_replace_a_newer_loaded_session",
                "Tests/UI/test_library_prompts_reader.py::test_same_prompt_older_detail_load_cannot_overwrite_newer_generation",
                "Tests/UI/test_library_skills_reader.py::test_same_skill_older_detail_result_cannot_replace_newer_generation",
            ),
            (
                "media_capability",
                "conversations_capability",
                "notes_capability",
                "prompts_capability",
                "skills_capability",
            ),
        ),
        "SH-06": contract(
            (
                "Tests/Library/test_library_adaptive_reader_state.py::test_resolution_never_mutates_saved_preferences",
                "Tests/UI/test_library_media_reader_shell.py::test_media_shell_resize_uses_resolver_without_reads_or_recompose",
                "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_resize_is_presentation_only",
            ),
            ("resize_purity",),
        ),
        "SH-07": contract(
            (
                "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
            ),
            ("single_app_route_cycle",),
        ),
        "ME-01": contract(
            (
                "Tests/UI/test_library_media_reader_flow.py::test_reader_defaults_to_read_and_keeps_mode_across_local_items",
                "Tests/UI/test_library_media_reader_flow.py::test_progress_restores_after_loaded_content_mounts",
            ),
            ("media_capability",),
        ),
        "ME-02": contract(
            (
                "Tests/UI/test_library_multiselect_media.py::test_confirming_bulk_delete_swaps_toolbar_for_confirm_row",
                "Tests/UI/test_library_multiselect_media.py::test_delete_selection_soft_deletes_via_real_db_and_updates_records_and_counts",
            ),
            ("media_capability",),
        ),
        "CO-01": contract(
            (
                "Tests/UI/test_library_conversation_reader.py::test_progressive_reader_paints_first_page_then_completes_find_off_loop",
                "Tests/UI/test_library_conversation_reader.py::test_reader_info_is_explicit_and_truthful",
            ),
            ("conversations_capability",),
        ),
        "CO-02": contract(
            (
                "Tests/UI/test_library_conversation_reader.py::test_open_console_requires_final_complete_error_free_match",
                "Tests/UI/test_library_conversation_reader.py::test_authoritative_refresh_marks_selected_conversation_deleted_without_fallback",
            ),
            ("conversations_capability",),
        ),
        "NO-01": contract(
            (
                "Tests/UI/test_library_notes_reader.py::test_list_and_work_identity_survive_open_preview_info_and_edit",
            ),
            ("notes_capability",),
        ),
        "NO-02": contract(
            (
                "Tests/Library/test_library_notes_session.py::test_conflict_stops_chaining_and_preserves_the_newest_draft",
                "Tests/UI/test_library_multiselect_notes.py::test_permanent_navigator_tasks_respect_dirty_draft_veto",
            ),
            ("notes_capability",),
        ),
        "PR-01": contract(
            (
                "Tests/UI/test_library_prompts_reader.py::test_basic_save_preserves_advanced_only_prompt_fields",
                "Tests/UI/test_library_prompts_reader.py::test_invalid_advanced_block_routes_save_focus_to_its_owner",
            ),
            ("prompts_capability",),
        ),
        "PR-02": contract(
            (
                "Tests/UI/test_library_prompts_reader.py::test_import_replaces_only_work_content_and_keeps_list_mounted",
                "Tests/UI/test_library_prompts_reader.py::test_detail_failure_keeps_prior_prompt_locked_and_retry_loads_selection",
            ),
            ("prompts_capability",),
        ),
        "SK-01": contract(
            (
                "Tests/UI/test_library_skills_reader.py::test_skill_modes_preserve_list_work_and_one_live_draft",
                "Tests/UI/test_library_skills_reader.py::test_skills_trust_mode_identifies_exact_review_snapshot",
            ),
            ("skills_capability",),
        ),
        "SK-02": contract(
            (
                "Tests/UI/test_library_skills_reader.py::test_skills_files_mode_is_read_only_and_labels_binary_files",
                "Tests/UI/test_library_skills_reader.py::test_same_skill_older_trust_review_cannot_patch_newer_generation",
                "Tests/UI/test_library_skills_reader.py::test_same_skill_older_delete_cannot_reset_a_newer_work_generation",
            ),
            ("skills_capability",),
        ),
    }

    assert module.CATALOGUE == expected


def test_sizes_destinations_and_curated_files_are_exact():
    module = _load_runner()

    assert module.SIZES == ((160, 50), (120, 35), (100, 30), (80, 24))
    assert module.DESTINATIONS == (
        "media",
        "conversations",
        "notes",
        "prompts",
        "skills",
    )
    assert module.CURATED_PYTEST_FILES == EXPECTED_CURATED_PYTEST_FILES


def test_catalogue_validation_rejects_missing_ids():
    module = _load_runner()
    incomplete = dict(module.CATALOGUE)
    incomplete.pop("SK-02")

    with pytest.raises(module.CloseoutError) as error:
        module.validate_catalogue(incomplete)

    assert _error_category(error) == "catalogue_ids_mismatch"


def test_catalogue_validation_rejects_duplicate_live_keys_within_a_contract():
    module = _load_runner()
    invalid = dict(module.CATALOGUE)
    invalid["SK-02"] = module.Contract(("a::b",), ("same", "same"))

    with pytest.raises(module.CloseoutError) as error:
        module.validate_catalogue(invalid)

    assert _error_category(error) == "live_key_duplicate"


def test_collection_rejects_an_unknown_catalogue_selector():
    module = _load_runner()
    selector = module.CATALOGUE["SH-01"].automated_nodes[0]
    synthetic_collection = {selector + "[media]"}
    catalogue = {"SH-01": module.Contract((selector, "missing::node"), ("live",))}

    with pytest.raises(module.CloseoutError) as error:
        module.validate_collected_selectors(catalogue, synthetic_collection)

    assert _error_category(error) == "pytest_selector_not_collected"


def test_selector_matches_exact_or_one_or_more_parameterized_nodes():
    module = _load_runner()
    selector = "Tests/example.py::test_case"

    assert module.matching_node_ids(selector, {selector}) == (selector,)
    assert module.matching_node_ids(
        selector,
        {selector + "[two]", "Tests/other.py::test_case", selector + "[one]"},
    ) == (selector + "[one]", selector + "[two]")


def test_settlement_rejects_when_any_matching_concrete_node_is_not_pass():
    module = _load_runner()
    selector = "Tests/example.py::test_case"
    catalogue = {"SH-01": module.Contract((selector,), ("live",))}

    with pytest.raises(module.CloseoutError) as error:
        module.validate_automated_results(
            catalogue,
            {selector + "[one]": "PASS", selector + "[two]": "FAIL"},
        )

    assert _error_category(error) == "pytest_node_not_pass"


def test_subject_revision_requires_exact_requested_head(tmp_path, monkeypatch):
    module = _load_runner()
    monkeypatch.setattr(module, "_git", _fake_git(head="actual"))

    with pytest.raises(module.CloseoutError) as error:
        module.admit_subject(tmp_path, "requested")

    assert _error_category(error) == "subject_revision_mismatch"


def test_subject_revision_requires_clean_worktree(tmp_path, monkeypatch):
    module = _load_runner()
    monkeypatch.setattr(module, "_git", _fake_git(dirty=True))

    with pytest.raises(module.CloseoutError) as error:
        module.admit_subject(tmp_path, "abc")

    assert _error_category(error) == "subject_worktree_not_clean"


def test_subject_records_commit_and_head_tree(tmp_path, monkeypatch):
    module = _load_runner()
    monkeypatch.setattr(module, "_git", _fake_git(head="abc", tree="tree-7"))

    subject = module.admit_subject(tmp_path, "abc")

    assert subject == module.Subject(commit="abc", tree="tree-7")
    assert module.Subject.__dataclass_params__.frozen is True


def test_final_head_tree_must_match_admitted_subject(tmp_path, monkeypatch):
    module = _load_runner()
    subject = module.Subject(commit="abc", tree="tree-before")
    monkeypatch.setattr(module, "_git", _fake_git(tree="tree-after"))

    with pytest.raises(module.CloseoutError) as error:
        module.verify_subject_tree(tmp_path, subject)

    assert _error_category(error) == "subject_tree_mismatch"


def test_git_uses_argument_vector_without_shell(monkeypatch, tmp_path):
    module = _load_runner()
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="abc\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._git(tmp_path, "rev-parse", "HEAD") == "abc"
    assert observed["command"] == ["git", "rev-parse", "HEAD"]
    assert observed["kwargs"].get("shell", False) is False


@pytest.mark.parametrize(
    ("arguments", "category"),
    [
        (
            ["--development-run", "--subject-revision", "abc"],
            "development_subject_revision_conflict",
        ),
        (
            ["--development-run", "--promote"],
            "development_promotion_conflict",
        ),
        (
            ["--development-run", "--verify-evidence", "bundle"],
            "development_verify_evidence_conflict",
        ),
        (
            ["--live-case", "common_matrix", "--live-only"],
            "live_selection_conflict",
        ),
        (["--promote"], "promotion_subject_required"),
        (["--promote", "--subject-revision", "abc", "--no-promote"], "promotion_mode_conflict"),
        (["--live-case", "not-declared"], "scenario_not_defined"),
    ],
)
def test_cli_semantic_misuse_has_stable_closeout_categories(arguments, category):
    module = _load_runner()

    with pytest.raises(module.CloseoutError) as error:
        module.parse_options(arguments)

    assert _error_category(error) == category


@pytest.mark.parametrize(
    ("arguments", "category"),
    [
        (["--live-case", ""], "scenario_not_defined"),
        (["--live-case", "", "--live-only"], "live_selection_conflict"),
        (
            ["--development-run", "--subject-revision", ""],
            "development_subject_revision_conflict",
        ),
        (["--subject-revision", ""], "subject_revision_empty"),
        (["--promote", "--subject-revision", ""], "subject_revision_empty"),
        (["--verify-evidence", ""], "verify_evidence_path_empty"),
        (
            ["--development-run", "--verify-evidence", ""],
            "development_verify_evidence_conflict",
        ),
    ],
)
def test_cli_rejects_present_but_empty_values(arguments, category):
    module = _load_runner()

    with pytest.raises(module.CloseoutError) as error:
        module.parse_options(arguments)

    assert _error_category(error) == category


def test_development_run_is_always_non_promoting():
    module = _load_runner()

    options = module.parse_options(["--development-run"])

    assert options.development_run is True
    assert options.promote is False
    assert options.no_promote is True


def test_parent_cli_accepts_each_declared_option():
    module = _load_runner()

    promotable = module.parse_options(
        ["--subject-revision", "abc", "--promote", "--live-case", "common_matrix"]
    )
    assert promotable.subject_revision == "abc"
    assert promotable.promote is True
    assert promotable.live_cases == ("common_matrix",)

    live_only = module.parse_options(["--live-only", "--no-promote"])
    assert live_only.live_only is True
    assert live_only.no_promote is True
    assert set(live_only.live_cases) == module.DECLARED_LIVE_CASES

    evidence = module.parse_options(["--verify-evidence", "some/bundle"])
    assert evidence.verify_evidence == Path("some/bundle")


@pytest.mark.parametrize("arguments", [["--not-an-option"], ["--subject-revision"]])
def test_argparse_still_owns_unknown_flags_and_malformed_values(arguments):
    module = _load_runner()

    with pytest.raises(SystemExit) as error:
        module.parse_options(arguments)

    assert error.value.code == 2
