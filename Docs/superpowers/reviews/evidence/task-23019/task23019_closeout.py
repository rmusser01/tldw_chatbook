"""Task-local contracts for the TASK-23019 adaptive-reader closeout."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import html
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True


class CloseoutError(Exception):
    """A stable semantic failure from the closeout runner."""

    def __init__(
        self, category: str, details: Mapping[str, object] | None = None
    ) -> None:
        self.category = category
        self.details = dict(details or {})
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


@dataclass(frozen=True)
class ScratchEnvironment:
    env: dict[str, str]
    denied_roots: tuple[str, ...]


@dataclass(frozen=True)
class ChildRunResult:
    returncode: int
    error: str | None
    result_path: Path | None
    details: Mapping[str, object] | None = None


CONTAINMENT_EXIT_STATUS = 86
CHILD_TIMEOUT_SECONDS = 3600
PROCESS_CLEANUP_TIMEOUT_SECONDS = 5
READER_JOIN_TIMEOUT_SECONDS = 5
CHILD_OUTPUT_BYTE_LIMIT = 64 * 1024
CHILD_OUTPUT_CHUNK_BYTES = 16 * 1024
RAW_RESULT_BYTE_LIMIT = 1024 * 1024
JSON_ARTIFACT_BYTE_LIMIT = 256 * 1024
TEXT_ARTIFACT_BYTE_LIMIT = 128 * 1024
SVG_ARTIFACT_BYTE_LIMIT = 512 * 1024
PROMOTED_BUNDLE_BYTE_LIMIT = 16 * 1024 * 1024
SOURCE_ARTIFACT_BYTE_LIMIT = 2 * 1024 * 1024
MAX_FACT_ARTIFACTS = 256
MAX_CAPTURE_ARTIFACTS = 16
MAX_RAW_ARTIFACTS = 1 + MAX_FACT_ARTIFACTS + MAX_CAPTURE_ARTIFACTS
MAX_MANAGED_ARTIFACTS = 279
MAX_DIAGNOSTIC_TEXT = 512
MAX_ERROR_TYPE_TEXT = 80
MAX_JSON_DEPTH = 64
MAX_URI_DECODE_PASSES = 6
CHILD_PATH = Path(__file__).with_name("task23019_closeout_child.py")
SCENARIO_PATH = Path(__file__).with_name("task23019_scenarios.py")
CHILD_SEMANTIC_ERRORS = frozenset(
    {"result_too_large", "scenario_not_defined", "scenario_result_invalid"}
)
CREDENTIAL_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SSH_ASKPASS",
    }
)
CREDENTIAL_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_PASSWORD",
    "_SECRET",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PRIVATE_KEY",
)
PRESERVED_ENV_NAMES = frozenset(
    {"COLORTERM", "FORCE_COLOR", "LANG", "LANGUAGE", "NO_COLOR", "PATH", "TERM", "TZ"}
)
PRESERVED_ENV_PREFIXES = ("LC_",)
TASK_ID = "TASK-23019"
OWNERSHIP_MARKER = "task-23019-complete-evidence-v1"
SOURCE_ARTIFACTS = (
    "task23019_closeout.py",
    "task23019_closeout_child.py",
    "task23019_scenarios.py",
)
SOURCE_DIRECTORY = "Docs/superpowers/reviews/evidence/task-23019"
REQUIRED_MANAGED_ARTIFACTS = (
    "README.md",
    "manifest.json",
    "summary.json",
    "hashes.json",
)
INJECTABLE_PROMOTION_PHASES = frozenset(
    {
        "during_stage_build",
        "after_stage_validation",
        "after_target_to_backup",
        "after_stage_to_target",
        "before_backup_removal",
    }
)


CATALOGUE: dict[str, Contract] = {
    "SH-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_shell.py::test_sync_layout_retains_every_mounted_child_identity",
            "Tests/UI/test_library_media_reader_shell.py::test_media_shell_mounts_library_items_reader_and_its_two_grips",
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
        live_cases=("preferences_fresh_reload", "single_app_route_cycle"),
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
    "Tests/Library/test_collections_capture_scope_service.py",
    "Tests/UI/test_library_adaptive_reader_shell.py",
    "Tests/UI/test_library_media_reader_shell.py",
    "Tests/UI/test_library_media_reader_flow.py",
    "Tests/UI/test_library_conversation_reader.py",
    "Tests/UI/test_library_notes_reader.py",
    "Tests/UI/test_library_prompts_reader.py",
    "Tests/UI/test_library_skills_reader.py",
    "Tests/UI/test_library_collections_capture_controller.py",
    "Tests/UI/test_library_collections_capture_reader.py",
    "Tests/UI/test_library_collections_reader_geometry.py",
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
EXECUTABLE_LIVE_ROOTS = (
    "common_matrix",
    "media_capability",
    "conversations_capability",
    "notes_capability",
    "prompts_capability",
    "skills_capability",
    "resize_purity",
    "preferences_fresh_reload",
    "single_app_route_cycle",
)
EXPECTED_LIVE_RESULT_KEYS = {
    "common_matrix": frozenset(
        f"{destination}-{width}x{height}"
        for destination in DESTINATIONS
        for width, height in SIZES
    ),
    "resize_purity": frozenset(
        f"{destination}-resize-purity" for destination in DESTINATIONS
    ),
    "preferences_fresh_reload": frozenset({"preferences-fresh-reload"}),
    "single_app_route_cycle": frozenset({"single-app-route-cycle"}),
    **{
        root: frozenset({root})
        for root in EXECUTABLE_LIVE_ROOTS
        if root
        not in {
            "common_matrix",
            "resize_purity",
            "preferences_fresh_reload",
            "single_app_route_cycle",
        }
    },
}
EXPECTED_CONCRETE_LIVE_RESULTS = frozenset(
    result_name
    for root in EXECUTABLE_LIVE_ROOTS
    for result_name in EXPECTED_LIVE_RESULT_KEYS[root]
)
REPRESENTATIVE_CAPTURES = {
    "media-160x50": ("live", "media-160x50"),
    "conversations-120x35": ("live", "conversations-120x35"),
    "notes-100x30": ("live", "notes-100x30"),
    "prompts-80x24": ("live", "prompts-80x24"),
    "skills-80x24": ("live", "skills-80x24"),
    "conversations-capability": ("live", "conversations_capability"),
    "preferences-fresh-reload": ("live", "preferences-fresh-reload"),
    "single-app-route-cycle": ("live", "single-app-route-cycle"),
}
REPRESENTATIVE_CAPTURE_SOURCES = {
    **{
        stem: ("common_matrix", stem)
        for stem in (
            "media-160x50",
            "conversations-120x35",
            "notes-100x30",
            "prompts-80x24",
            "skills-80x24",
        )
    },
    "conversations-capability": (
        "conversations_capability",
        "conversations-capability",
    ),
    "preferences-fresh-reload": (
        "preferences_fresh_reload",
        "preferences-fresh-reload",
    ),
    "single-app-route-cycle": (
        "single_app_route_cycle",
        "single-app-route-cycle",
    ),
}

_PRODUCTION_IDENTITIES = {
    "media": {
        "shell": "library-media-reader-shell",
        "items": "library-canvas",
        "work": "library-media-viewer",
    },
    "conversations": {
        "shell": "library-conversations-reader-shell",
        "items": "library-canvas",
        "work": "library-conversation-reader",
    },
    "notes": {
        "shell": "library-notes-reader-shell",
        "items": "library-canvas",
        "work": "library-note-work-pane",
    },
    "prompts": {
        "shell": "library-prompts-reader-shell",
        "items": "library-canvas",
        "work": "library-prompt-work-pane",
    },
    "skills": {
        "shell": "library-skills-reader-shell",
        "items": "library-canvas",
        "work": "library-skill-work-pane",
    },
}
_CAPABILITY_SIZES = {
    "media_capability": (160, 50),
    "conversations_capability": (160, 50),
    "notes_capability": (120, 35),
    "prompts_capability": (100, 30),
    "skills_capability": (80, 24),
}
LIVE_RESULT_METADATA = {
    **{
        f"{destination}-{width}x{height}": {
            "destination": destination,
            "final_destination": destination,
            "terminal_size": (width, height),
            "identities": _PRODUCTION_IDENTITIES[destination],
        }
        for destination in DESTINATIONS
        for width, height in SIZES
    },
    **{
        name: {
            "destination": name.removesuffix("_capability"),
            "final_destination": name.removesuffix("_capability"),
            "terminal_size": size,
            "identities": _PRODUCTION_IDENTITIES[name.removesuffix("_capability")],
        }
        for name, size in _CAPABILITY_SIZES.items()
    },
    **{
        f"{destination}-resize-purity": {
            "destination": destination,
            "final_destination": destination,
            "terminal_size": (160, 50),
            "identities": _PRODUCTION_IDENTITIES[destination],
        }
        for destination in DESTINATIONS
    },
    "preferences-fresh-reload": {
        "destination": "all",
        "final_destination": "skills",
        "terminal_size": (160, 50),
        "identities": _PRODUCTION_IDENTITIES["skills"],
    },
    "single-app-route-cycle": {
        "destination": "all",
        "final_destination": "skills",
        "terminal_size": (160, 50),
        "identities": _PRODUCTION_IDENTITIES["skills"],
    },
}
assert set(LIVE_RESULT_METADATA) == EXPECTED_CONCRETE_LIVE_RESULTS

_ABSOLUTE_PATH = re.compile(r"(?<!>)(?:[A-Za-z]:[\\/]|/)[^\s\"']+")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|PASSWORD|SECRET|CREDENTIALS?|PRIVATE[_-]?KEY))"
    r"\s*[:=]\s*[^\s,;]+"
)
_HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9:.</])(?<!<checkout>)(?<!<runtime>)(?<!<scratch>)"
    r"(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|"
    r"/{2,}[^\s\"'<>]+|"
    r"/(?!/|(?i:&#(?:160|xa0);)[^/\s\"'<>]*(?:<|$))[^\s\"'<>]+)"
)
_WHITESPACE_HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9:.</])(?<!<checkout>)(?<!<runtime>)(?<!<scratch>)"
    r"/[^\S\r\n]+[^\"'<\r\n]*[\\/]"
)
_ANCHORED_WHITESPACE_HOST_PATH = re.compile(r"\A/\s+[^\s\"'<>]")
_HOST_URI = re.compile(
    r"(?i)\b(?:file|vscode-file|filesystem|local-file):(?://)?[^\s\"']*"
)
_CREDENTIAL_JSON_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "passphrase",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)
_CREDENTIAL_JSON_ENV_KEYS = frozenset(name.casefold() for name in CREDENTIAL_ENV_NAMES)
_CREDENTIAL_JSON_SUFFIXES = (
    "secret",
    "token",
    "api_key",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer_token",
    "client_secret",
    "private_key",
    "password",
    "passphrase",
    "credential",
    "credentials",
    "authorization",
)
_BENIGN_JSON_CREDENTIAL_KEYS = frozenset(
    {
        "continuation_token",
        "next_token",
        "page_token",
        "revision_token",
        "secret_recipe",
        "token_count",
        "tokenizer",
        "worker_token",
    }
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
        if any(key not in EXECUTABLE_LIVE_ROOTS for key in contract.live_cases):
            raise CloseoutError("live_case_not_mapped")
    if dict(catalogue) != CATALOGUE:
        raise CloseoutError("catalogue_mapping_mismatch")


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
            if any(
                results[node_id] != "PASS"
                for node_id in matching_node_ids(selector, results)
            ):
                raise CloseoutError("pytest_node_not_pass")


def _result_status(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        status = value.get("status")
        return status if isinstance(status, str) else None
    return None


def _live_result_names(live_case: str) -> tuple[str, ...]:
    expected = EXPECTED_LIVE_RESULT_KEYS.get(live_case)
    return tuple(sorted(expected)) if expected is not None else (live_case,)


def validate_complete_results(
    catalogue: Mapping[str, Contract],
    automated_results: Mapping[str, object],
    live_results: Mapping[str, object],
    *,
    not_applicable: Mapping[str, str] | None = None,
) -> None:
    """Require fresh automated and live evidence, or one declared catalogue reason."""
    selectors = tuple(
        selector
        for contract in catalogue.values()
        for selector in contract.automated_nodes
    )
    if any(
        not any(matching_node_ids(selector, (node_id,)) for selector in selectors)
        for node_id in automated_results
    ):
        raise CloseoutError("automated_result_unknown")
    if any(
        _result_status(result) not in {"PASS", "NOT_APPLICABLE"}
        for result in (*automated_results.values(), *live_results.values())
    ):
        raise CloseoutError("result_status_invalid")
    declared_na = dict(not_applicable or {})
    if not set(declared_na).issubset(catalogue):
        raise CloseoutError("not_applicable_catalogue_unknown")
    if any(
        not isinstance(reason, str) or not reason.strip()
        for reason in declared_na.values()
    ):
        raise CloseoutError("not_applicable_reason_missing")
    admitted_automated_na: set[str] = set()
    admitted_live_na: set[str] = set()
    for contract_id in declared_na:
        contract = catalogue[contract_id]
        for selector in contract.automated_nodes:
            matches = matching_node_ids(selector, automated_results)
            if not matches or any(
                _result_status(automated_results[node_id]) != "NOT_APPLICABLE"
                for node_id in matches
            ):
                raise CloseoutError("not_applicable_evidence_missing")
            admitted_automated_na.update(matches)
        for live_case in contract.live_cases:
            for result_name in _live_result_names(live_case):
                if _result_status(live_results.get(result_name)) != "NOT_APPLICABLE":
                    raise CloseoutError("not_applicable_evidence_missing")
                admitted_live_na.add(result_name)
    observed_automated_na = {
        node_id
        for node_id, value in automated_results.items()
        if _result_status(value) == "NOT_APPLICABLE"
    }
    observed_live_na = {
        live_case
        for live_case, value in live_results.items()
        if _result_status(value) == "NOT_APPLICABLE"
    }
    if (
        observed_automated_na != admitted_automated_na
        or observed_live_na != admitted_live_na
    ):
        raise CloseoutError("not_applicable_undeclared")

    active = {
        contract_id: contract
        for contract_id, contract in catalogue.items()
        if contract_id not in declared_na
    }
    validate_automated_results(
        active,
        {
            node_id: _result_status(result) or "INVALID"
            for node_id, result in automated_results.items()
        },
    )
    for contract in active.values():
        for live_case in contract.live_cases:
            for result_name in _live_result_names(live_case):
                if result_name not in live_results:
                    raise CloseoutError("live_evidence_missing")
                if _result_status(live_results[result_name]) != "PASS":
                    raise CloseoutError("live_evidence_not_pass")


def _artifact_allowed(relative: str, *, raw: bool) -> bool:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        return False
    if raw and relative == "summary.json":
        return True
    if not raw and relative in {*SOURCE_ARTIFACTS, *REQUIRED_MANAGED_ARTIFACTS}:
        return True
    if len(path.parts) != 2:
        return False
    folder, name = path.parts
    if folder == "facts":
        return name.endswith(".json") and bool(name.removesuffix(".json"))
    if folder == "captures":
        return name.endswith((".txt", ".svg")) and bool(name.rsplit(".", maxsplit=1)[0])
    return False


def _artifact_limit(relative: str) -> int:
    if relative == "README.md" or relative.endswith(".txt"):
        return TEXT_ARTIFACT_BYTE_LIMIT
    if relative.endswith(".svg"):
        return SVG_ARTIFACT_BYTE_LIMIT
    if relative.endswith(".json"):
        return JSON_ARTIFACT_BYTE_LIMIT
    return PROMOTED_BUNDLE_BYTE_LIMIT


def _read_regular_file(path: Path, *, limit: int) -> bytes:
    """Read one unchanged regular file without following its final symlink."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise CloseoutError("artifact_symlink")
    if not stat.S_ISREG(before.st_mode):
        raise CloseoutError("artifact_not_regular")
    if before.st_size > limit:
        raise CloseoutError("artifact_too_large")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise CloseoutError("artifact_read_failed") from None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise CloseoutError("artifact_changed")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise CloseoutError("artifact_too_large")
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise CloseoutError("artifact_changed")
        return payload
    finally:
        os.close(descriptor)


def _receipt(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)


def _open_directory_nofollow(path: Path) -> tuple[int, tuple[int, int, int]]:
    absolute = path.absolute()
    try:
        canonical = path.resolve(strict=True)
    except OSError:
        raise CloseoutError("artifact_root_missing") from None
    macos_var_alias = False
    if sys.platform == "darwin" and (
        str(absolute) == "/var" or str(absolute).startswith("/var/")
    ):
        expected = Path("/private") / absolute.relative_to("/")
        macos_var_alias = canonical == expected
    if absolute != canonical and not macos_var_alias:
        raise CloseoutError("artifact_symlink")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(canonical.anchor, flags)
    except OSError:
        raise CloseoutError("artifact_root_missing") from None
    try:
        receipt = _receipt(os.fstat(descriptor))
        for part in canonical.parts[1:]:
            child_fd, receipt = _open_child_directory(descriptor, part)
            os.close(descriptor)
            descriptor = child_fd
        return descriptor, receipt
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(
    parent_fd: int, name: str
) -> tuple[int, tuple[int, int, int]]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise CloseoutError("artifact_symlink")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        raise CloseoutError("artifact_symlink") from None
    opened = os.fstat(descriptor)
    if _receipt(opened) != _receipt(before):
        os.close(descriptor)
        raise CloseoutError("artifact_changed")
    return descriptor, _receipt(opened)


def _read_regular_at(parent_fd: int, name: str, *, limit: int) -> bytes:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode):
        raise CloseoutError("artifact_symlink")
    if not stat.S_ISREG(before.st_mode):
        raise CloseoutError("artifact_not_regular")
    if before.st_size > limit:
        raise CloseoutError("artifact_too_large")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        raise CloseoutError("artifact_read_failed") from None
    try:
        opened = os.fstat(descriptor)
        if _receipt(opened) != _receipt(before):
            raise CloseoutError("artifact_changed")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(payload) > limit:
            raise CloseoutError("artifact_too_large")
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise CloseoutError("artifact_changed")
        return payload
    finally:
        os.close(descriptor)


def _read_artifact_tree_fd(
    root_fd: int, root_receipt: tuple[int, int, int], *, raw: bool
) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    total = 0
    root_entries = sorted(os.listdir(root_fd))
    for name in root_entries:
        status = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISLNK(status.st_mode):
            raise CloseoutError("artifact_symlink")
        if stat.S_ISDIR(status.st_mode):
            if name not in {"facts", "captures"}:
                raise CloseoutError("artifact_path_not_allowed")
            child_fd, child_receipt = _open_child_directory(root_fd, name)
            try:
                filenames = sorted(os.listdir(child_fd))
                limit = MAX_FACT_ARTIFACTS if name == "facts" else MAX_CAPTURE_ARTIFACTS
                if len(filenames) > limit:
                    raise CloseoutError("artifact_count_exceeded")
                for filename in filenames:
                    relative = f"{name}/{filename}"
                    if not _artifact_allowed(relative, raw=raw):
                        raise CloseoutError("artifact_path_not_allowed")
                    payload = _read_regular_at(
                        child_fd, filename, limit=_artifact_limit(relative)
                    )
                    artifacts[relative] = payload
                    total += len(payload)
                if _receipt(os.fstat(child_fd)) != child_receipt:
                    raise CloseoutError("artifact_changed")
            finally:
                os.close(child_fd)
        else:
            if not _artifact_allowed(name, raw=raw):
                raise CloseoutError("artifact_path_not_allowed")
            artifacts[name] = _read_regular_at(
                root_fd, name, limit=_artifact_limit(name)
            )
            total += len(artifacts[name])
        if total > PROMOTED_BUNDLE_BYTE_LIMIT:
            raise CloseoutError("bundle_too_large")
        count_limit = MAX_RAW_ARTIFACTS if raw else MAX_MANAGED_ARTIFACTS
        if len(artifacts) > count_limit:
            raise CloseoutError("artifact_count_exceeded")
    if _receipt(os.fstat(root_fd)) != root_receipt:
        raise CloseoutError("artifact_changed")
    return artifacts


def _read_artifact_tree(
    root: Path, *, raw: bool, with_receipt: bool = False
) -> dict[str, bytes] | tuple[dict[str, bytes], tuple[int, int, int]]:
    root_fd, root_receipt = _open_directory_nofollow(root)
    try:
        artifacts = _read_artifact_tree_fd(root_fd, root_receipt, raw=raw)
        try:
            if _receipt(root.lstat()) != root_receipt:
                raise CloseoutError("artifact_changed")
        except OSError:
            raise CloseoutError("artifact_changed") from None
    finally:
        os.close(root_fd)
    return (artifacts, root_receipt) if with_receipt else artifacts


def _read_artifact_tree_at(
    parent_fd: int, name: str, *, raw: bool
) -> tuple[dict[str, bytes], tuple[int, int, int]]:
    root_fd, root_receipt = _open_child_directory(parent_fd, name)
    try:
        return _read_artifact_tree_fd(root_fd, root_receipt, raw=raw), root_receipt
    finally:
        os.close(root_fd)


def collect_raw_artifacts(raw_root: Path) -> dict[str, bytes]:
    """Admit only bounded allowlisted raw outputs into memory."""
    artifacts = _read_artifact_tree(raw_root, raw=True)
    assert isinstance(artifacts, dict)
    return artifacts


def _is_json_credential_key(key: str) -> bool:
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
    if normalized in _BENIGN_JSON_CREDENTIAL_KEYS:
        return False
    return (
        normalized in _CREDENTIAL_JSON_KEYS
        or normalized in _CREDENTIAL_JSON_ENV_KEYS
        or any(
            normalized.endswith("_" + suffix) for suffix in _CREDENTIAL_JSON_SUFFIXES
        )
    )


def _json_contains_credential_key(value: object) -> bool:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise CloseoutError("artifact_json_invalid")
        if isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise CloseoutError("artifact_json_invalid")
                if _is_json_credential_key(key):
                    return True
                pending.append((child, depth + 1))
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return False


def _decoded_text_candidates(value: str) -> list[str]:
    inspected = [value]
    for _ in range(MAX_URI_DECODE_PASSES):
        changed = False
        decoded = urllib.parse.unquote(inspected[-1])
        if decoded != inspected[-1]:
            inspected.append(decoded)
            changed = True
        decoded = html.unescape(inspected[-1])
        if decoded != inspected[-1]:
            inspected.append(decoded)
            changed = True
        if not changed:
            return inspected
    raise CloseoutError("host_path_present")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, child in pairs:
        if key in result:
            raise CloseoutError("artifact_json_invalid")
        result[key] = child
    return result


def _strict_json_loads(payload: str | bytes) -> object:
    return json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)


def _validate_json_text_values(
    value: object,
    *,
    replacements: Collection[tuple[str, str]],
    credential_values: Collection[str],
) -> None:
    pending = [(value, 0, False)]
    while pending:
        current, depth, is_key = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise CloseoutError("artifact_json_invalid")
        if isinstance(current, str):
            for candidate in _decoded_text_candidates(current):
                if is_key and _is_json_credential_key(candidate):
                    raise CloseoutError("credential_material")
                if any(secret and secret in candidate for secret in credential_values):
                    raise CloseoutError("credential_material")
                if _CREDENTIAL_ASSIGNMENT.search(candidate):
                    raise CloseoutError("credential_material")
                if any(
                    re.search(
                        re.escape(host_root) + r"(?=$|[\\/\s\"'?,#])",
                        candidate,
                    )
                    for host_root, _replacement in replacements
                ):
                    raise CloseoutError("host_path_present")
                if (
                    _HOST_URI.search(candidate)
                    or _HOST_PATH.search(candidate)
                    or _WHITESPACE_HOST_PATH.search(candidate)
                    or _ANCHORED_WHITESPACE_HOST_PATH.search(candidate)
                ):
                    raise CloseoutError("host_path_present")
        elif isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise CloseoutError("artifact_json_invalid")
                pending.extend(((key, depth + 1, True), (child, depth + 1, False)))
        elif isinstance(current, list):
            pending.extend((child, depth + 1, False) for child in current)


def _validate_json_artifacts(artifacts: Mapping[str, bytes]) -> None:
    for relative, payload in artifacts.items():
        if not relative.endswith(".json"):
            continue
        try:
            parsed = _strict_json_loads(payload)
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise CloseoutError("artifact_json_invalid") from None
        if not isinstance(parsed, (dict, list)):
            raise CloseoutError("artifact_json_invalid")
        if _json_contains_credential_key(parsed):
            raise CloseoutError("credential_material")


def _normalize_text_payload(
    payload: bytes,
    *,
    replacements: Collection[tuple[str, str]],
    credential_values: Collection[str],
) -> bytes:
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        raise CloseoutError("artifact_encoding_invalid") from None
    inspected = _decoded_text_candidates(text)
    for candidate in inspected:
        if any(value and value in candidate for value in credential_values):
            raise CloseoutError("credential_material")
        if candidate != text and any(
            re.search(
                re.escape(host_root) + r"(?=$|[\\/\s\"'?,#])",
                candidate,
            )
            for host_root, _replacement in replacements
        ):
            raise CloseoutError("host_path_present")
        if _CREDENTIAL_ASSIGNMENT.search(candidate):
            raise CloseoutError("credential_material")
        inspected_normalized = candidate
        for host_root, replacement in replacements:
            inspected_normalized = re.sub(
                re.escape(host_root) + r"(?=$|[\\/\s\"'?,#])",
                replacement,
                inspected_normalized,
            )
        try:
            decoded_json = _strict_json_loads(inspected_normalized)
        except (UnicodeError, json.JSONDecodeError):
            decoded_json = None
        except RecursionError:
            raise CloseoutError("artifact_json_invalid") from None
        decoded_json_container = isinstance(decoded_json, (dict, list))
        if decoded_json_container:
            _validate_json_text_values(
                decoded_json,
                replacements=replacements,
                credential_values=credential_values,
            )
        if not decoded_json_container and (
            _HOST_URI.search(inspected_normalized)
            or _HOST_PATH.search(inspected_normalized)
            or _WHITESPACE_HOST_PATH.search(inspected_normalized)
            or _ANCHORED_WHITESPACE_HOST_PATH.search(inspected_normalized)
        ):
            raise CloseoutError("host_path_present")
    normalized = text
    for host_root, replacement in replacements:
        normalized = re.sub(
            re.escape(host_root) + r"(?=$|[\\/\s\"'?,#])",
            replacement,
            normalized,
        )
    if any(value and value in normalized for value in credential_values):
        raise CloseoutError("credential_material")
    if _CREDENTIAL_ASSIGNMENT.search(normalized):
        raise CloseoutError("credential_material")
    try:
        normalized_json = _strict_json_loads(normalized)
    except (UnicodeError, json.JSONDecodeError):
        normalized_json = None
    except RecursionError:
        raise CloseoutError("artifact_json_invalid") from None
    normalized_json_container = isinstance(normalized_json, (dict, list))
    if normalized_json_container:
        _validate_json_text_values(
            normalized_json,
            replacements=replacements,
            credential_values=credential_values,
        )
    if not normalized_json_container and (
        _HOST_URI.search(normalized)
        or _HOST_PATH.search(normalized)
        or _WHITESPACE_HOST_PATH.search(normalized)
        or _ANCHORED_WHITESPACE_HOST_PATH.search(normalized)
    ):
        raise CloseoutError("host_path_present")
    return normalized.encode("utf-8")


def normalize_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    roots: Mapping[str, Path],
    credential_values: Collection[str] = (),
) -> dict[str, bytes]:
    """Normalize declared roots in memory and reject retained host authority."""
    if set(roots) != {"checkout", "runtime", "scratch"}:
        raise CloseoutError("normalization_roots_invalid")
    replacements = sorted(
        ((str(Path(path).resolve()), f"<{name}>") for name, path in roots.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    normalized: dict[str, bytes] = {}
    for relative, payload in artifacts.items():
        if not _artifact_allowed(relative, raw=True):
            raise CloseoutError("artifact_path_not_allowed")
        normalized[relative] = _normalize_text_payload(
            payload,
            replacements=replacements,
            credential_values=credential_values,
        )
    _validate_json_artifacts(normalized)
    return normalized


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _result_inventory(
    automated_results: Mapping[str, object], live_results: Mapping[str, object]
) -> dict[tuple[str, str], str]:
    return {
        **{
            ("automated", name): _result_status(result) or "INVALID"
            for name, result in automated_results.items()
        },
        **{
            ("live", name): _result_status(result) or "INVALID"
            for name, result in live_results.items()
        },
    }


_LIVE_ORACLE_KEYS = frozenset(
    {
        "status",
        "destination",
        "final_destination",
        "terminal_size",
        "contained",
        "regions",
        "identities",
        "focus_owner",
        "record",
        "preferences",
        "host_worker_groups",
        "visible_controls",
        "compositor_text",
        "cleanup_owner_counts",
    }
)
_SCENARIO_ORACLE_KEYS = frozenset(
    {
        "regions_do_not_intersect",
        "selected_row",
        "grips",
        "primary_actions",
        "footer_shortcuts",
        "f6_route",
    }
)
_CAPABILITY_OBSERVATION_KEYS = {
    "media_capability": frozenset(
        {
            "catalogue_ids",
            "find_status",
            "selected_loaded_identity",
            "mode_after_round_trip",
            "bulk_preview_copy",
            "bulk_selected_count",
            "item_count_after_cancel",
            "destructive_boundary",
        }
    ),
    "conversations_capability": frozenset(
        {
            "catalogue_ids",
            "progressive_page_offsets",
            "progressive_message_total",
            "find_match_ids",
            "stale_first_id",
            "settled_target_id",
            "retry_error",
            "retry_recovered_id",
            "retry_message_ids",
            "info_copy",
            "handoff_source_id",
            "handoff_action_label",
        }
    ),
    "notes_capability": frozenset(
        {
            "catalogue_ids",
            "draft_note_id",
            "preview_info_draft",
            "save_calls_before_conflict",
            "dirty_navigation_veto",
            "conflict_copy",
            "recovered_title",
            "recovered_body",
            "bulk_preview_copy",
        }
    ),
    "prompts_capability": frozenset(
        {
            "catalogue_ids",
            "import_boundary",
            "bulk_preview_copy",
            "retry_selected_id",
            "retry_prior_loaded_id",
            "structured_prompt_id",
            "basic_draft",
            "preserved_advanced_author",
            "preserved_advanced_details",
            "preserved_advanced_keywords",
            "preserved_advanced_title_before_validation",
            "preserved_advanced_mapping_hint",
            "validation_status",
            "validation_focus_owner",
            "history_title",
        }
    ),
    "skills_capability": frozenset(
        {
            "catalogue_ids",
            "draft_after_overview_round_trip",
            "review_id",
            "review_manifest_generation",
            "review_digest",
            "review_identity_copy",
            "stale_review_rejection",
            "files_copy",
            "delete_preview_copy",
            "destructive_boundary",
        }
    ),
}
_DURABLE_OBSERVATION_KEYS = {
    **{
        f"{destination}-resize-purity": frozenset(
            {
                "resize_sequence",
                "widget_identity_retained",
                "semantic_state_retained",
                "service_calls_during_resize",
                "config_write_worker_poll_calls",
            }
        )
        for destination in DESTINATIONS
    },
    "preferences-fresh-reload": frozenset(
        {
            "fresh_screen",
            "requested_library_open",
            "requested_items_open",
            "first_host_cleanup_owner_counts",
        }
    ),
    "single-app-route-cycle": frozenset(
        {
            "route_order",
            "shared_library_open",
            "destination_items_open",
            "focus_regions",
            "notes_draft_retained_without_save",
            "prompt_draft_retained_without_save",
            "late_conversation_worker_fenced",
            "revisit_receipts",
        }
    ),
}


def _validate_route_receipts(
    value: object,
    *,
    shared_library_open: object,
    destination_items_open: object,
) -> bool:
    if (
        not isinstance(shared_library_open, bool)
        or not isinstance(destination_items_open, Mapping)
        or set(destination_items_open) != set(DESTINATIONS)
        or any(not isinstance(state, bool) for state in destination_items_open.values())
        or not isinstance(value, Mapping)
        or set(value) != set(DESTINATIONS)
    ):
        return False
    for destination, receipt in value.items():
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "preferences",
            "record",
            "focus",
            "identities",
            "draft",
            "worker_fenced",
        }:
            return False
        if receipt.get("worker_fenced") is not True:
            return False
        if receipt.get("identities") != _PRODUCTION_IDENTITIES[destination]:
            return False
        focus = receipt.get("focus")
        draft = receipt.get("draft")
        preferences = receipt.get("preferences")
        record = receipt.get("record")
        if (
            not isinstance(preferences, Mapping)
            or set(preferences)
            != {
                "requested_library_open",
                "requested_items_open",
                "effective_library_open",
                "effective_items_open",
            }
            or any(not isinstance(state, bool) for state in preferences.values())
            or preferences["requested_library_open"] is not shared_library_open
            or preferences["effective_library_open"] is not shared_library_open
            or preferences["requested_items_open"]
            is not destination_items_open[destination]
            or preferences["effective_items_open"]
            is not destination_items_open[destination]
            or not isinstance(record, Mapping)
            or set(record) != {"selected", "pending", "loaded", "mode"}
            or record["pending"] is not None
            or record["selected"] != record["loaded"]
            or not isinstance(record["mode"], str)
            or not record["mode"]
            or not isinstance(focus, Mapping)
            or set(focus) != {"region", "owner"}
            or focus.get("region") != "work"
            or not isinstance(focus.get("owner"), str)
            or not isinstance(draft, Mapping)
            or set(draft) != {"dirty", "retained_without_save", "value"}
        ):
            return False
        identity = record["selected"]
        if destination == "prompts":
            if (
                isinstance(identity, bool)
                or not isinstance(identity, int)
                or identity <= 0
            ):
                return False
        elif not isinstance(identity, str) or not identity:
            return False
        should_retain = destination in {"notes", "prompts"}
        if (
            draft.get("dirty") is not should_retain
            or draft.get("retained_without_save") is not should_retain
            or (should_retain and not isinstance(draft.get("value"), str))
            or (not should_retain and draft.get("value") is not None)
        ):
            return False
    return True


def _validate_live_oracle(name: str, value: object) -> None:
    """Require one complete, identity-bearing structured live oracle."""
    if not isinstance(value, Mapping):
        raise CloseoutError("evidence_inventory_invalid")
    if value.get("status") == "NOT_APPLICABLE":
        return
    if value.get("status") != "PASS":
        raise CloseoutError("evidence_inventory_invalid")
    destination = value.get("destination")
    size = value.get("terminal_size")
    regions = value.get("regions")
    identities = value.get("identities")
    record = value.get("record")
    preferences = value.get("preferences")
    cleanup = value.get("cleanup_owner_counts")
    metadata = LIVE_RESULT_METADATA.get(name)
    if metadata is None:
        raise CloseoutError("evidence_inventory_invalid")
    if name in _CAPABILITY_OBSERVATION_KEYS:
        expected_keys = _LIVE_ORACLE_KEYS | _SCENARIO_ORACLE_KEYS | {"observations"}
    elif name in _DURABLE_OBSERVATION_KEYS:
        expected_keys = _LIVE_ORACLE_KEYS | {"observations"}
    else:
        expected_keys = (
            _LIVE_ORACLE_KEYS
            | _SCENARIO_ORACLE_KEYS
            | {
                "items_comfort_expansion",
                "restoration_paths",
            }
        )
    required_regions = {"library", "items", "work"}
    required_region_fields = {"x", "y", "width", "height"}
    region_truth = (
        isinstance(regions, Mapping)
        and set(regions)
        in (
            {"library", "items", "work"},
            {"library", "library_grip", "items", "items_grip", "work"},
        )
        and all(
            isinstance(region, Mapping)
            and set(region)
            in (required_region_fields, required_region_fields | {"right", "bottom"})
            and all(
                isinstance(region[field], int) and region[field] >= 0
                for field in required_region_fields
            )
            and region["x"] + region["width"] <= size[0]
            and region["y"] + region["height"] <= size[1]
            for region in (regions[key] for key in required_regions)
        )
        if isinstance(size, list)
        and len(size) == 2
        and all(isinstance(part, int) and part > 0 for part in size)
        else False
    )
    if (
        set(value) != expected_keys
        or destination != metadata["destination"]
        or value.get("final_destination") != metadata["final_destination"]
        or not isinstance(size, list)
        or len(size) != 2
        or any(not isinstance(part, int) or part <= 0 for part in size)
        or tuple(size) != metadata["terminal_size"]
        or value.get("contained") is not True
        or not region_truth
        or not isinstance(identities, Mapping)
        or identities != metadata["identities"]
        or not isinstance(value.get("focus_owner"), str)
        or not value["focus_owner"]
        or not isinstance(record, Mapping)
        or set(record) != {"selected", "pending", "loaded", "mode"}
        or any(
            record[key] is not None
            and (
                isinstance(record[key], bool) or not isinstance(record[key], (int, str))
            )
            for key in ("selected", "pending", "loaded")
        )
        or not isinstance(record.get("mode"), str)
        or not record["mode"]
        or not isinstance(preferences, Mapping)
        or set(preferences)
        not in (
            {
                "requested_library_open",
                "requested_items_open",
                "effective_library_open",
                "effective_items_open",
            },
            {
                "requested_library_open",
                "requested_items_open",
                "requested_custom_widths_enabled",
                "requested_library_width",
                "requested_items_width",
                "effective_library_open",
                "effective_items_open",
                "effective_library_width",
                "effective_items_width",
                "effective_reader_width",
                "effective_priority_pane",
            },
        )
        or any(
            not isinstance(preferences[key], bool)
            for key in (
                "requested_library_open",
                "requested_items_open",
                "effective_library_open",
                "effective_items_open",
            )
        )
        or not isinstance(value.get("host_worker_groups"), list)
        or any(
            not isinstance(group, str) or not group
            for group in value["host_worker_groups"]
        )
        or not isinstance(value.get("visible_controls"), list)
        or not value["visible_controls"]
        or any(
            not isinstance(control, str) or not control
            for control in value["visible_controls"]
        )
        or not isinstance(value.get("compositor_text"), str)
        or not value["compositor_text"].strip()
        or not isinstance(cleanup, Mapping)
        or not {
            "host_worker_leaks",
            "host_task_leaks",
            "host_thread_worker_leaks",
        }.issubset(cleanup)
        or any(
            cleanup.get(key) != 0
            for key in (
                "host_worker_leaks",
                "host_task_leaks",
                "host_thread_worker_leaks",
            )
        )
        or set(cleanup)
        not in (
            {"host_worker_leaks", "host_task_leaks", "host_thread_worker_leaks"},
            {
                "host_workers_before",
                "host_workers_owned",
                "host_worker_leaks",
                "host_task_leaks",
                "host_thread_worker_leaks",
            },
        )
    ):
        raise CloseoutError("evidence_inventory_invalid")
    observations = value.get("observations")
    expected_observation_keys = _CAPABILITY_OBSERVATION_KEYS.get(
        name
    ) or _DURABLE_OBSERVATION_KEYS.get(name)
    if expected_observation_keys is not None and (
        not isinstance(observations, Mapping)
        or set(observations) != expected_observation_keys
    ):
        raise CloseoutError("evidence_inventory_invalid")
    if name.endswith("-resize-purity") and observations.get("resize_sequence") != [
        [120, 35],
        [100, 30],
        [80, 24],
        [160, 50],
    ]:
        raise CloseoutError("evidence_inventory_invalid")
    if name == "preferences-fresh-reload":
        first_cleanup = observations.get("first_host_cleanup_owner_counts")
        if (
            not isinstance(first_cleanup, Mapping)
            or set(first_cleanup)
            != {
                "host_workers_before",
                "host_workers_owned",
                "host_worker_leaks",
                "host_task_leaks",
                "host_thread_worker_leaks",
            }
            or any(
                first_cleanup[key] != 0
                for key in (
                    "host_worker_leaks",
                    "host_task_leaks",
                    "host_thread_worker_leaks",
                )
            )
        ):
            raise CloseoutError("evidence_inventory_invalid")
    if name == "single-app-route-cycle" and not _validate_route_receipts(
        observations.get("revisit_receipts"),
        shared_library_open=observations.get("shared_library_open"),
        destination_items_open=observations.get("destination_items_open"),
    ):
        raise CloseoutError("evidence_inventory_invalid")


def _validate_evidence_inventory(
    artifacts: Mapping[str, bytes],
    *,
    automated_results: Mapping[str, object],
    live_results: Mapping[str, object],
) -> None:
    expected = _result_inventory(automated_results, live_results)
    observed: dict[tuple[str, str], str] = {}
    retained_facts: dict[tuple[str, str], Mapping[str, object]] = {}
    for relative, payload in artifacts.items():
        if not relative.startswith("facts/"):
            continue
        try:
            fact = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise CloseoutError("evidence_inventory_invalid") from None
        if not isinstance(fact, dict):
            raise CloseoutError("evidence_inventory_invalid")
        kind, name, status_value = (
            fact.get("kind"),
            fact.get("result_name"),
            fact.get("status"),
        )
        if (
            kind not in {"automated", "live"}
            or not isinstance(name, str)
            or not isinstance(status_value, str)
            or (kind, name) in observed
        ):
            raise CloseoutError("evidence_inventory_invalid")
        observed[(kind, name)] = status_value
        retained_facts[(kind, name)] = fact
        if kind == "live":
            expected_live = live_results.get(name)
            _validate_live_oracle(name, expected_live)
            if fact != {"kind": "live", "result_name": name, **expected_live}:
                raise CloseoutError("evidence_inventory_invalid")
        elif set(fact) != {"kind", "result_name", "status"}:
            raise CloseoutError("evidence_inventory_invalid")
    if observed != expected or set(live_results) != EXPECTED_CONCRETE_LIVE_RESULTS:
        raise CloseoutError("evidence_inventory_invalid")

    captures: dict[str, set[str]] = {}
    for relative, payload in artifacts.items():
        if not relative.startswith("captures/"):
            continue
        path = Path(relative)
        stem = path.stem
        try:
            text = payload.decode("utf-8")
        except UnicodeError:
            raise CloseoutError("evidence_inventory_invalid") from None
        expected_identity = REPRESENTATIVE_CAPTURES.get(stem)
        if expected_identity is None:
            raise CloseoutError("evidence_inventory_invalid")
        kind, result_name = expected_identity
        expected_status = expected.get((kind, result_name))
        if expected_status is None:
            raise CloseoutError("evidence_inventory_invalid")
        if path.suffix == ".txt":
            prefix = f"result_name: {result_name}\nstatus: {expected_status}\n"
            fact = retained_facts.get((kind, result_name), {})
            identity_ok = text == prefix + str(fact.get("compositor_text", ""))
        else:
            try:
                root = ET.fromstring(text)
            except (ET.ParseError, RecursionError):
                identity_ok = False
            else:
                identity_ok = (
                    root.tag.rpartition("}")[2] == "svg"
                    and root.attrib.get("data-result-name") == result_name
                    and root.attrib.get("data-status") == expected_status
                )
        if not identity_ok:
            raise CloseoutError("evidence_inventory_invalid")
        captures.setdefault(stem, set()).add(path.suffix)
    if set(captures) != set(REPRESENTATIVE_CAPTURES) or any(
        suffixes != {".txt", ".svg"} for suffixes in captures.values()
    ):
        raise CloseoutError("evidence_inventory_invalid")


def _canonical_summary(
    automated_results: Mapping[str, object], live_results: Mapping[str, object]
) -> dict[str, object]:
    statuses = [
        *map(_result_status, automated_results.values()),
        *map(_result_status, live_results.values()),
    ]
    return {
        "status": "PASS",
        "automated_results": len(automated_results),
        "live_results": len(live_results),
        "not_applicable_results": statuses.count("NOT_APPLICABLE"),
    }


def _canonical_readme(subject: Subject, summary: Mapping[str, object]) -> bytes:
    """Build the bounded, literal closeout runbook retained with evidence."""
    return (
        f"# {TASK_ID} adaptive-reader closeout evidence\n\n"
        "## Subject and result\n\n"
        f"Subject commit: `{subject.commit}`\n\n"
        f"Subject tree: `{subject.tree}`\n\n"
        f"Result: PASS — {summary['automated_results']} automated results, "
        f"{summary['live_results']} live results, and "
        f"{summary['not_applicable_results']} NOT_APPLICABLE results.\n\n"
        "## Exact environment and commands\n\n"
        "Run these commands from a clean detached subject worktree. The commands "
        "resolve that worktree once, change to its repository root, and use the "
        "repository-adjacent virtual environment interpreter.\n\n"
        "The child runs with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, explicit "
        "`pytest_asyncio.plugin`, scratch-owned HOME/XDG/config/database/temp paths, "
        "read-only subject-checkout/runtime authority, and network/process denial.\n\n"
        "```bash\n"
        'SUBJECT_ROOT="$(git rev-parse --show-toplevel)"\n'
        'cd "$SUBJECT_ROOT"\n'
        'test -z "$(git status --porcelain)"\n'
        'test -z "$(git symbolic-ref -q HEAD)"\n'
        "PYTHONDONTWRITEBYTECODE=1 "
        f'TASK23019_SUBJECT_REVISION="{subject.commit}" '
        "../../.venv/bin/python "
        "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py "
        f'--subject-revision "{subject.commit}" --promote\n'
        "PYTHONDONTWRITEBYTECODE=1 ../../.venv/bin/python "
        "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py "
        "--verify-evidence Docs/superpowers/reviews/evidence/task-23019\n"
        "```\n\n"
        "## Repair history\n\n"
        "- `a92000229b`: retained-reader route contracts (focused product RED/GREEN).\n"
        "- `471da9f9db`: production dispatch and cleanup boundary (harness RED/GREEN).\n"
        "- `d81e231f26`: explicit hermetic async plugin (harness RED/GREEN).\n"
        "- `c9b8a7e002`: scratch descriptor metadata hardening (harness RED/GREEN).\n"
        "- `04c5c55c73`: stronger retained-evidence and containment assertions "
        "(harness RED/GREEN).\n"
        "- `f44970a1b5`: bounded child-failure context retained through the parent "
        "runner (harness RED/GREEN).\n"
        "- `79e70364e4`: Media capability state settled before capture "
        "(live scenario RED/GREEN).\n"
        "- `38021d064c`: bounded layered evidence normalization and structural JSON "
        "scanning (harness RED/GREEN).\n"
        "- `77c05aeb5c`: real Work focus settled before capability capture "
        "(live scenario RED/GREEN).\n"
        "- `fb465fede6`: mounted, displayed identity-row settling and named live-root "
        "diagnostics (live scenario RED/GREEN).\n"
        f"- `{subject.commit[:10]}`: frozen subject including all current final "
        "hardening (focused and production-matrix RED/GREEN).\n\n"
        "## Promotion and cleanup proof\n\n"
        "All facts and captures were normalized and validated in memory; the raw "
        "TemporaryDirectory exited before repository promotion. Promotion then "
        "validated the subject bytes, canonical catalogue, hashes, limits, and "
        "complete sibling transaction before the atomic destination swap.\n"
    ).encode()


def _normalization_replacements(
    roots: Mapping[str, Path],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            ((str(Path(path).resolve()), f"<{name}>") for name, path in roots.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )


def _sanitize_not_applicable(
    reasons: Mapping[str, str] | None,
    *,
    replacements: Collection[tuple[str, str]],
    credential_values: Collection[str],
) -> dict[str, str]:
    sanitized = {}
    for contract_id, reason in (reasons or {}).items():
        if len(reason.encode("utf-8")) > MAX_DIAGNOSTIC_TEXT:
            raise CloseoutError("not_applicable_reason_too_large")
        payload = _normalize_text_payload(
            reason.strip().encode(),
            replacements=replacements,
            credential_values=credential_values,
        )
        sanitized[contract_id] = payload.decode()
    return sanitized


def _build_bundle(
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
    subject_hashes: Mapping[str, str],
    raw_artifacts: Mapping[str, bytes],
    catalogue: Mapping[str, Contract],
    automated_results: Mapping[str, object],
    live_results: Mapping[str, object],
    normalization_roots: Mapping[str, Path],
    not_applicable: Mapping[str, str] | None,
    credential_values: Collection[str],
) -> dict[str, bytes]:
    validate_catalogue(catalogue)
    validate_complete_results(
        catalogue,
        automated_results,
        live_results,
        not_applicable=not_applicable,
    )
    _validate_subject_sources(subject_sources, subject_hashes)

    replacements = _normalization_replacements(normalization_roots)
    normalized_raw = normalize_artifacts(
        raw_artifacts,
        roots=normalization_roots,
        credential_values=credential_values,
    )
    expected_summary = _canonical_summary(automated_results, live_results)
    try:
        supplied_summary = json.loads(normalized_raw["summary.json"])
    except (KeyError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise CloseoutError("summary_mismatch") from None
    if supplied_summary != expected_summary:
        raise CloseoutError("summary_mismatch")
    normalized_raw["summary.json"] = _json_bytes(expected_summary)
    _validate_evidence_inventory(
        normalized_raw,
        automated_results=automated_results,
        live_results=live_results,
    )
    sanitized_na = _sanitize_not_applicable(
        not_applicable,
        replacements=replacements,
        credential_values=credential_values,
    )
    artifacts = {**subject_sources, **normalized_raw}
    manifest = {
        "task": TASK_ID,
        "ownership_marker": OWNERSHIP_MARKER,
        "bundle_role": "complete-evidence",
        "subject_commit": subject.commit,
        "subject_tree": subject.tree,
        "hashes_excluded": ["hashes.json"],
        "retained_paths": [
            *SOURCE_ARTIFACTS,
            *REQUIRED_MANAGED_ARTIFACTS,
            "facts/*.json",
            "captures/*.txt",
            "captures/*.svg",
        ],
        "catalogue": {
            contract_id: {
                "automated": list(contract.automated_nodes),
                "live": list(contract.live_cases),
                **(
                    {"not_applicable": sanitized_na[contract_id]}
                    if contract_id in sanitized_na
                    else {}
                ),
            }
            for contract_id, contract in sorted(catalogue.items())
        },
    }
    artifacts["README.md"] = _canonical_readme(subject, expected_summary)
    artifacts["manifest.json"] = _json_bytes(manifest)
    for relative in tuple(artifacts):
        if relative in SOURCE_ARTIFACTS:
            continue
        artifacts[relative] = _normalize_text_payload(
            artifacts[relative],
            replacements=replacements,
            credential_values=credential_values,
        )
    _validate_json_artifacts(artifacts)
    hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in sorted(artifacts.items())
    }
    artifacts["hashes.json"] = _json_bytes(
        {"algorithm": "sha256", "excluded": ["hashes.json"], "files": hashes}
    )
    artifacts["hashes.json"] = _normalize_text_payload(
        artifacts["hashes.json"],
        replacements=replacements,
        credential_values=credential_values,
    )
    _validate_json_artifacts({"hashes.json": artifacts["hashes.json"]})
    for relative, payload in artifacts.items():
        if len(payload) > _artifact_limit(relative):
            raise CloseoutError("artifact_too_large")
    if sum(map(len, artifacts.values())) > PROMOTED_BUNDLE_BYTE_LIMIT:
        raise CloseoutError("bundle_too_large")
    if len(artifacts) > MAX_MANAGED_ARTIFACTS:
        raise CloseoutError("artifact_count_exceeded")
    return artifacts


def _validate_subject_sources(
    subject_sources: Mapping[str, bytes], subject_hashes: Mapping[str, str]
) -> None:
    if set(subject_sources) != set(SOURCE_ARTIFACTS) or set(subject_hashes) != set(
        SOURCE_ARTIFACTS
    ):
        raise CloseoutError("subject_source_mapping_missing")
    for relative in SOURCE_ARTIFACTS:
        if (
            hashlib.sha256(subject_sources[relative]).hexdigest()
            != subject_hashes[relative]
        ):
            raise CloseoutError("subject_source_hash_mismatch")


def _validate_bundle_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
) -> None:
    required = {*SOURCE_ARTIFACTS, *REQUIRED_MANAGED_ARTIFACTS}
    if not required.issubset(artifacts):
        raise CloseoutError("promotion_collision", {"reason": "artifact_missing"})
    for relative, payload in artifacts.items():
        if relative in SOURCE_ARTIFACTS:
            continue
        if (
            _normalize_text_payload(payload, replacements=(), credential_values=())
            != payload
        ):
            raise CloseoutError("promotion_collision", {"reason": "content_changed"})
    _validate_json_artifacts(artifacts)
    try:
        manifest = json.loads(artifacts["manifest.json"])
        hashes = json.loads(artifacts["hashes.json"])
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise CloseoutError(
            "promotion_collision", {"reason": "metadata_invalid"}
        ) from None
    expected_identity = {
        "task": TASK_ID,
        "ownership_marker": OWNERSHIP_MARKER,
        "bundle_role": "complete-evidence",
        "subject_commit": subject.commit,
        "subject_tree": subject.tree,
    }
    if not isinstance(manifest, dict) or set(manifest) != {
        *expected_identity,
        "hashes_excluded",
        "retained_paths",
        "catalogue",
    }:
        raise CloseoutError("promotion_collision", {"reason": "metadata_invalid"})
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise CloseoutError("promotion_collision", {"reason": "identity_mismatch"})
    if manifest.get("hashes_excluded") != ["hashes.json"]:
        raise CloseoutError("promotion_collision", {"reason": "hash_exclusion_invalid"})
    if manifest.get("retained_paths") != [
        *SOURCE_ARTIFACTS,
        *REQUIRED_MANAGED_ARTIFACTS,
        "facts/*.json",
        "captures/*.txt",
        "captures/*.svg",
    ]:
        raise CloseoutError("promotion_collision", {"reason": "paths_invalid"})
    catalogue_payload = manifest.get("catalogue")
    if not isinstance(catalogue_payload, dict) or set(catalogue_payload) != set(
        CATALOGUE
    ):
        raise CloseoutError("promotion_collision", {"reason": "catalogue_invalid"})
    not_applicable: dict[str, str] = {}
    for contract_id, contract in CATALOGUE.items():
        declared = catalogue_payload.get(contract_id)
        if not isinstance(declared, dict) or set(declared) - {
            "automated",
            "live",
            "not_applicable",
        }:
            raise CloseoutError("promotion_collision", {"reason": "catalogue_invalid"})
        if declared.get("automated") != list(contract.automated_nodes) or declared.get(
            "live"
        ) != list(contract.live_cases):
            raise CloseoutError("promotion_collision", {"reason": "catalogue_invalid"})
        if "not_applicable" in declared:
            reason = declared["not_applicable"]
            if (
                not isinstance(reason, str)
                or not reason.strip()
                or len(reason.encode("utf-8")) > MAX_DIAGNOSTIC_TEXT
            ):
                raise CloseoutError(
                    "promotion_collision", {"reason": "catalogue_invalid"}
                )
            not_applicable[contract_id] = reason
    expected_hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in artifacts.items()
        if relative != "hashes.json"
    }
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {"algorithm", "excluded", "files"}
        or hashes.get("algorithm") != "sha256"
        or hashes.get("excluded") != ["hashes.json"]
        or hashes.get("files") != dict(sorted(expected_hashes.items()))
    ):
        raise CloseoutError("promotion_collision", {"reason": "hashes_invalid"})
    for relative in SOURCE_ARTIFACTS:
        if artifacts[relative] != subject_sources[relative]:
            raise CloseoutError(
                "promotion_collision", {"reason": "subject_source_changed"}
            )
    automated: dict[str, str] = {}
    live: dict[str, object] = {}
    for relative, payload in artifacts.items():
        if not relative.startswith("facts/"):
            continue
        try:
            fact = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise CloseoutError(
                "promotion_collision", {"reason": "evidence_inventory_invalid"}
            ) from None
        if not isinstance(fact, dict) or not {
            "kind",
            "result_name",
            "status",
        }.issubset(fact):
            raise CloseoutError(
                "promotion_collision", {"reason": "evidence_inventory_invalid"}
            )
        kind, name, status_value = fact["kind"], fact["result_name"], fact["status"]
        target = automated if kind == "automated" else live if kind == "live" else None
        if (
            target is None
            or not isinstance(name, str)
            or not isinstance(status_value, str)
            or name in target
        ):
            raise CloseoutError(
                "promotion_collision", {"reason": "evidence_inventory_invalid"}
            )
        target[name] = (
            status_value
            if kind == "automated"
            else {
                key: value
                for key, value in fact.items()
                if key not in {"kind", "result_name"}
            }
        )
    try:
        validate_complete_results(
            CATALOGUE, automated, live, not_applicable=not_applicable
        )
        _validate_evidence_inventory(
            artifacts, automated_results=automated, live_results=live
        )
        summary = json.loads(artifacts["summary.json"])
    except (CloseoutError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        reason = (
            error.category if isinstance(error, CloseoutError) else "metadata_invalid"
        )
        raise CloseoutError("promotion_collision", {"reason": reason}) from None
    if summary != _canonical_summary(automated, live):
        raise CloseoutError("promotion_collision", {"reason": "summary_invalid"})
    if artifacts["README.md"] != _canonical_readme(subject, summary):
        raise CloseoutError("promotion_collision", {"reason": "readme_invalid"})


def _validate_bundle_at(
    parent_fd: int,
    name: str,
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
) -> tuple[dict[str, bytes], tuple[int, int, int, str]]:
    try:
        artifacts, _ = _read_artifact_tree_at(parent_fd, name, raw=False)
        _validate_bundle_artifacts(
            artifacts, subject=subject, subject_sources=subject_sources
        )
        first = _content_receipt_at(parent_fd, name)
        confirmed, _ = _read_artifact_tree_at(parent_fd, name, raw=False)
        _validate_bundle_artifacts(
            confirmed, subject=subject, subject_sources=subject_sources
        )
        second = _content_receipt_at(parent_fd, name)
    except CloseoutError as error:
        raise CloseoutError("promotion_collision", {"reason": error.category}) from None
    if first != second:
        raise CloseoutError("promotion_collision", {"reason": "artifact_changed"})
    return confirmed, second


def _validate_bundle(
    root: Path,
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
) -> tuple[dict[str, bytes], tuple[int, int, int, str]]:
    parent_fd, _ = _open_directory_nofollow(root.parent)
    try:
        return _validate_bundle_at(
            parent_fd,
            root.name,
            subject=subject,
            subject_sources=subject_sources,
        )
    finally:
        os.close(parent_fd)


def _validate_source_bootstrap_at(
    parent_fd: int, name: str, *, subject_sources: Mapping[str, bytes]
) -> tuple[int, int, int, str]:
    """Admit only the exact pre-promotion three-source repository state."""
    try:
        descriptor, _ = _open_child_directory(parent_fd, name)
        try:
            if set(os.listdir(descriptor)) != set(SOURCE_ARTIFACTS):
                raise CloseoutError(
                    "promotion_collision", {"reason": "bootstrap_paths_invalid"}
                )
        finally:
            os.close(descriptor)
    except OSError:
        raise CloseoutError("promotion_collision") from None
    try:
        artifacts, _ = _read_artifact_tree_at(parent_fd, name, raw=False)
    except CloseoutError as error:
        raise CloseoutError("promotion_collision", {"reason": error.category}) from None
    if set(artifacts) != set(SOURCE_ARTIFACTS):
        raise CloseoutError(
            "promotion_collision", {"reason": "bootstrap_paths_invalid"}
        )
    if any(
        artifacts[relative] != subject_sources[relative]
        for relative in SOURCE_ARTIFACTS
    ):
        raise CloseoutError(
            "promotion_collision", {"reason": "bootstrap_source_changed"}
        )
    first = _content_receipt_at(parent_fd, name)
    confirmed, _ = _read_artifact_tree_at(parent_fd, name, raw=False)
    if confirmed != artifacts or _content_receipt_at(parent_fd, name) != first:
        raise CloseoutError("promotion_collision", {"reason": "artifact_changed"})
    return first


def _validate_recovery_candidate_at(
    parent_fd: int,
    name: str,
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
    allow_bootstrap: bool,
) -> tuple[int, int, int, str]:
    try:
        _, receipt = _validate_bundle_at(
            parent_fd,
            name,
            subject=subject,
            subject_sources=subject_sources,
        )
    except CloseoutError:
        if not allow_bootstrap:
            raise
        receipt = _validate_source_bootstrap_at(
            parent_fd, name, subject_sources=subject_sources
        )
    return receipt


def _entry_receipt(parent_fd: int, name: str) -> tuple[int, int, int]:
    status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise CloseoutError("promotion_collision")
    return _receipt(status)


def _content_receipt_at(parent_fd: int, name: str) -> tuple[int, int, int, str]:
    """Hash one no-follow directory tree while its parent descriptor stays open."""
    root_fd, opened = _open_child_directory(parent_fd, name)
    digest = hashlib.sha256()
    total = 0

    def visit(descriptor: int, prefix: str) -> None:
        nonlocal total
        directory_receipt = _receipt(os.fstat(descriptor))
        for child in sorted(os.listdir(descriptor)):
            status = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}{child}"
            if stat.S_ISLNK(status.st_mode):
                raise CloseoutError("promotion_collision")
            if stat.S_ISDIR(status.st_mode):
                digest.update(f"d\0{relative}\0".encode())
                child_fd, child_receipt = _open_child_directory(descriptor, child)
                try:
                    visit(child_fd, relative + "/")
                    if _receipt(os.fstat(child_fd)) != child_receipt:
                        raise CloseoutError("promotion_collision")
                finally:
                    os.close(child_fd)
                continue
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_size > SOURCE_ARTIFACT_BYTE_LIMIT
            ):
                raise CloseoutError("promotion_collision")
            payload = _read_regular_at(
                descriptor, child, limit=SOURCE_ARTIFACT_BYTE_LIMIT
            )
            total += len(payload)
            if total > 2 * PROMOTED_BUNDLE_BYTE_LIMIT + JSON_ARTIFACT_BYTE_LIMIT:
                raise CloseoutError("promotion_collision")
            digest.update(f"f\0{relative}\0{len(payload)}\0".encode())
            digest.update(hashlib.sha256(payload).digest())
        if _receipt(os.fstat(descriptor)) != directory_receipt:
            raise CloseoutError("promotion_collision")

    try:
        visit(root_fd, "")
        if _receipt(os.fstat(root_fd)) != opened:
            raise CloseoutError("promotion_collision")
        return (*opened, digest.hexdigest())
    finally:
        os.close(root_fd)


def _recheck_entry_receipt(
    parent_fd: int, name: str, expected: tuple[int, int, int]
) -> None:
    try:
        actual = _entry_receipt(parent_fd, name)
    except (OSError, CloseoutError):
        raise CloseoutError("promotion_collision") from None
    if actual != expected:
        raise CloseoutError("promotion_collision")


def _rename_noreplace(
    source_fd: int, source: str, destination_fd: int, destination: str
) -> None:
    """Atomically rename one entry without replacing a raced destination."""
    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        result = library.renameatx_np(
            source_fd,
            encoded_source,
            destination_fd,
            encoded_destination,
            0x00000004,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        result = library.renameat2(
            source_fd,
            encoded_source,
            destination_fd,
            encoded_destination,
            1,
        )
    else:
        raise CloseoutError("promotion_unsupported_platform")
    if result == 0:
        return
    failure = ctypes.get_errno()
    if failure in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CloseoutError("promotion_collision")
    raise OSError(failure, os.strerror(failure))


def _rename_entry(
    parent_fd: int,
    source: str,
    destination: str,
    expected: tuple[int, int, int, str],
) -> None:
    if _content_receipt_at(parent_fd, source) != expected:
        raise CloseoutError("promotion_collision")
    try:
        _rename_noreplace(parent_fd, source, parent_fd, destination)
        os.fsync(parent_fd)
    except CloseoutError:
        raise
    except OSError:
        raise CloseoutError("promotion_io_failed") from None


def _rename_between(
    source_fd: int,
    source: str,
    destination_fd: int,
    destination: str,
    expected: tuple[int, int, int, str],
) -> None:
    if _content_receipt_at(source_fd, source) != expected:
        raise CloseoutError("promotion_collision")
    try:
        _rename_noreplace(source_fd, source, destination_fd, destination)
        if destination_fd != source_fd:
            os.fsync(destination_fd)
        os.fsync(source_fd)
    except CloseoutError:
        raise
    except OSError:
        raise CloseoutError("promotion_io_failed") from None


def _delete_directory_contents(descriptor: int, *, marker_last: str | None) -> None:
    names = sorted(os.listdir(descriptor), key=lambda name: name == marker_last)
    for name in names:
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        expected = _receipt(before)
        if stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode):
            child_fd, opened = _open_child_directory(descriptor, name)
            try:
                _delete_directory_contents(child_fd, marker_last=None)
                os.fsync(child_fd)
            finally:
                os.close(child_fd)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if _receipt(current) != opened:
                raise CloseoutError("promotion_collision")
            os.rmdir(name, dir_fd=descriptor)
        else:
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if _receipt(current) != expected:
                raise CloseoutError("promotion_collision")
            os.unlink(name, dir_fd=descriptor)


_TRANSACTION_MARKER = "transaction.json"
_TRANSACTION_MARKER_TEMP = "transaction.json.tmp"
_ACTIVE_TRANSACTION = re.compile(r"^\.task-23019\.txn-([0-9a-f]{32})$")
_RETIRED_TRANSACTION = re.compile(r"^\.task-23019\.txn-retired-([0-9a-f]{32})$")
_PENDING_TRANSACTION = re.compile(r"^\.task-23019\.txn-pending-([0-9a-f]{32})$")
_PREACTIVATION_TRANSACTION = re.compile(r"^\.task-23019\.preactivation-([0-9a-f]{32})$")
_AUTHORITY = re.compile(r"^\.task-23019\.txn-authority-([0-9a-f]{32})\.json$")
_AUTHORITY_TEMP = re.compile(r"^\.task-23019\.txn-authority-([0-9a-f]{32})\.json\.tmp$")


def _transaction_nonce(name: str) -> str:
    for pattern in (
        _ACTIVE_TRANSACTION,
        _RETIRED_TRANSACTION,
        _PENDING_TRANSACTION,
        _PREACTIVATION_TRANSACTION,
    ):
        match = pattern.fullmatch(name)
        if match is not None:
            return match.group(1)
    raise CloseoutError("promotion_collision")


def _authority_name(nonce: str) -> str:
    return f".task-23019.txn-authority-{nonce}.json"


def _transaction_payload(subject: Subject, role: str, nonce: str = "") -> bytes:
    return _json_bytes(
        {
            "task": TASK_ID,
            "ownership_marker": OWNERSHIP_MARKER,
            "subject_commit": subject.commit,
            "subject_tree": subject.tree,
            "role": role,
            "nonce": nonce,
            "active_name": f".task-23019.txn-{nonce}",
        }
    )


def _authority_payload(
    subject: Subject, nonce: str, directory_receipt: tuple[int, int, int]
) -> bytes:
    return _json_bytes(
        {
            "task": TASK_ID,
            "ownership_marker": OWNERSHIP_MARKER,
            "subject_commit": subject.commit,
            "subject_tree": subject.tree,
            "nonce": nonce,
            "active_name": f".task-23019.txn-{nonce}",
            "directory_receipt": list(directory_receipt),
        }
    )


def _write_atomic_regular(
    parent_fd: int, final: str, payload: bytes, *, temporary: str | None = None
) -> None:
    temporary = temporary or final + ".tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    _rename_noreplace(parent_fd, temporary, parent_fd, final)
    os.fsync(parent_fd)


def _regular_content_receipt_at(
    parent_fd: int, name: str
) -> tuple[int, int, int, int, str]:
    payload = _read_regular_at(parent_fd, name, limit=JSON_ARTIFACT_BYTE_LIMIT)
    status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode):
        raise CloseoutError("promotion_collision")
    return (*_receipt(status), len(payload), hashlib.sha256(payload).hexdigest())


def _read_authority(
    parent_fd: int, nonce: str, subject: Subject
) -> tuple[str, tuple[int, int, int], tuple[int, int, int, int, str]]:
    name = _authority_name(nonce)
    try:
        payload = _read_regular_at(parent_fd, name, limit=JSON_ARTIFACT_BYTE_LIMIT)
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise CloseoutError("promotion_collision")
        receipt_value = parsed.get("directory_receipt")
        if (
            set(parsed)
            != {
                "task",
                "ownership_marker",
                "subject_commit",
                "subject_tree",
                "nonce",
                "active_name",
                "directory_receipt",
            }
            or not isinstance(receipt_value, list)
            or len(receipt_value) != 3
            or any(not isinstance(value, int) for value in receipt_value)
        ):
            raise CloseoutError("promotion_collision")
        directory_receipt = tuple(receipt_value)
        if payload != _authority_payload(subject, nonce, directory_receipt):
            raise CloseoutError("promotion_collision")
        authority_receipt = _regular_content_receipt_at(parent_fd, name)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, CloseoutError):
        raise CloseoutError("promotion_collision") from None
    return name, directory_receipt, authority_receipt


def _validate_authority(
    parent_fd: int, nonce: str, subject: Subject, transaction_name: str
) -> tuple[str, tuple[int, int, int, int, str]]:
    name, expected_directory, authority_receipt = _read_authority(
        parent_fd, nonce, subject
    )
    if _entry_receipt(parent_fd, transaction_name) != expected_directory:
        raise CloseoutError("promotion_collision")
    return name, authority_receipt


def _unlink_authority(
    parent_fd: int, name: str, expected: tuple[int, int, int, int, str]
) -> None:
    if _regular_content_receipt_at(parent_fd, name) != expected:
        raise CloseoutError("promotion_collision")
    os.unlink(name, dir_fd=parent_fd)


def _write_transaction_marker(
    transaction_fd: int, subject: Subject, role: str, nonce: str
) -> None:
    if role not in {"active", "retirement"}:
        raise CloseoutError("promotion_collision")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    marker_fd = os.open(_TRANSACTION_MARKER_TEMP, flags, 0o600, dir_fd=transaction_fd)
    try:
        view = memoryview(_transaction_payload(subject, role, nonce))
        while view:
            view = view[os.write(marker_fd, view) :]
        os.fsync(marker_fd)
    finally:
        os.close(marker_fd)
    os.replace(
        _TRANSACTION_MARKER_TEMP,
        _TRANSACTION_MARKER,
        src_dir_fd=transaction_fd,
        dst_dir_fd=transaction_fd,
    )
    os.fsync(transaction_fd)


def _read_transaction_marker(
    parent_fd: int, name: str, subject: Subject
) -> tuple[str, int]:
    nonce = _transaction_nonce(name)
    _validate_authority(parent_fd, nonce, subject, name)
    transaction_fd, _ = _open_child_directory(parent_fd, name)
    try:
        try:
            marker_payload = _read_regular_at(
                transaction_fd,
                _TRANSACTION_MARKER,
                limit=JSON_ARTIFACT_BYTE_LIMIT,
            )
            marker = json.loads(marker_payload)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise CloseoutError("promotion_collision") from None
        expected = {
            "task": TASK_ID,
            "ownership_marker": OWNERSHIP_MARKER,
            "subject_commit": subject.commit,
            "subject_tree": subject.tree,
            "nonce": nonce,
            "active_name": f".task-23019.txn-{nonce}",
        }
        if not isinstance(marker, dict) or any(
            marker.get(key) != value for key, value in expected.items()
        ):
            raise CloseoutError("promotion_collision")
        role = marker.get("role")
        if role not in {"active", "retirement"}:
            raise CloseoutError("promotion_collision")
        name_role = "retirement" if _RETIRED_TRANSACTION.fullmatch(name) else "active"
        if role != name_role and not (role == "retirement" and name_role == "active"):
            raise CloseoutError("promotion_collision")
        if marker_payload != _transaction_payload(subject, role, nonce):
            raise CloseoutError("promotion_collision")
        try:
            temporary = os.stat(
                _TRANSACTION_MARKER_TEMP,
                dir_fd=transaction_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(temporary.st_mode):
                raise CloseoutError("promotion_collision")
            os.unlink(_TRANSACTION_MARKER_TEMP, dir_fd=transaction_fd)
            os.fsync(transaction_fd)
        return role, transaction_fd
    except BaseException:
        os.close(transaction_fd)
        raise


def _create_transaction(parent_fd: int, subject: Subject) -> str:
    nonce = secrets.token_hex(16)
    active_name = f".task-23019.txn-{nonce}"
    pending_name = f".task-23019.txn-pending-{nonce}"
    preactivation_name = f".task-23019.preactivation-{nonce}"
    authority_name = _authority_name(nonce)
    authority_temporary = authority_name + ".tmp"
    os.mkdir(preactivation_name, 0o700, dir_fd=parent_fd)
    transaction_fd, opened = _open_child_directory(parent_fd, preactivation_name)
    try:
        _write_atomic_regular(
            parent_fd,
            authority_name,
            _authority_payload(subject, nonce, opened),
            temporary=authority_temporary,
        )
        receipt = _content_receipt_at(parent_fd, preactivation_name)
        _rename_entry(parent_fd, preactivation_name, pending_name, receipt)
        _write_transaction_marker(transaction_fd, subject, "active", nonce)
        receipt = _content_receipt_at(parent_fd, pending_name)
        _rename_entry(parent_fd, pending_name, active_name, receipt)
        os.fsync(parent_fd)
    except BaseException as transaction_error:
        try:
            cleanup_name = next(
                candidate
                for candidate in (active_name, pending_name, preactivation_name)
                if candidate in os.listdir(parent_fd)
                and _entry_receipt(parent_fd, candidate) == opened
            )
            _recheck_entry_receipt(parent_fd, cleanup_name, opened)
            _delete_directory_contents(transaction_fd, marker_last=None)
            os.fsync(transaction_fd)
            os.close(transaction_fd)
            transaction_fd = -1
            _recheck_entry_receipt(parent_fd, cleanup_name, opened)
            os.rmdir(cleanup_name, dir_fd=parent_fd)
            if authority_name in os.listdir(parent_fd):
                _name, _directory, authority_receipt = _read_authority(
                    parent_fd, nonce, subject
                )
                _unlink_authority(parent_fd, authority_name, authority_receipt)
            os.fsync(parent_fd)
        except BaseException as cleanup_error:
            if transaction_fd >= 0:
                os.close(transaction_fd)
            raise cleanup_error from transaction_error
        raise
    os.close(transaction_fd)
    return active_name


def _validate_partial_stage_at(transaction_fd: int) -> bool:
    try:
        stage_fd, _ = _open_child_directory(transaction_fd, "stage")
    except (OSError, CloseoutError):
        return False
    try:
        if os.listdir(stage_fd) != ["partial"]:
            return False
        status = os.stat("partial", dir_fd=stage_fd, follow_symlinks=False)
        return stat.S_ISREG(status.st_mode) and _read_regular_at(
            stage_fd, "partial", limit=len(b"partial")
        ) in {b"", b"partial"}
    finally:
        os.close(stage_fd)


def _delete_transaction(
    parent_fd: int,
    name: str,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
) -> None:
    nonce = _transaction_nonce(name)
    authority_name, authority_receipt = _validate_authority(
        parent_fd, nonce, subject, name
    )
    transaction_fd, _ = _open_child_directory(parent_fd, name)
    if not _RETIRED_TRANSACTION.fullmatch(name):
        os.close(transaction_fd)
        raise CloseoutError("promotion_collision")
    expected = _entry_receipt(parent_fd, name)
    try:
        entries = set(os.listdir(transaction_fd))
        if not entries.issubset(
            {_TRANSACTION_MARKER, _TRANSACTION_MARKER_TEMP, "stage", "backup"}
        ):
            raise CloseoutError("promotion_collision")
        if _TRANSACTION_MARKER in entries:
            marker = _read_regular_at(
                transaction_fd,
                _TRANSACTION_MARKER,
                limit=JSON_ARTIFACT_BYTE_LIMIT,
            )
            if marker != _transaction_payload(subject, "retirement", nonce):
                raise CloseoutError("promotion_collision")
        elif entries:
            raise CloseoutError("promotion_collision")
        if "stage" in entries:
            try:
                _validate_recovery_candidate_at(
                    transaction_fd,
                    "stage",
                    subject=subject,
                    subject_sources=subject_sources,
                    allow_bootstrap=False,
                )
            except CloseoutError:
                if not _validate_partial_stage_at(transaction_fd):
                    raise CloseoutError("promotion_collision") from None
        if "backup" in entries:
            _validate_recovery_candidate_at(
                transaction_fd,
                "backup",
                subject=subject,
                subject_sources=subject_sources,
                allow_bootstrap=True,
            )
        _delete_directory_contents(transaction_fd, marker_last=_TRANSACTION_MARKER)
        os.fsync(transaction_fd)
        os.close(transaction_fd)
        transaction_fd = -1
        _recheck_entry_receipt(parent_fd, name, expected)
        os.rmdir(name, dir_fd=parent_fd)
        _unlink_authority(parent_fd, authority_name, authority_receipt)
        os.fsync(parent_fd)
    except BaseException:
        if transaction_fd >= 0:
            os.close(transaction_fd)
        raise


def _retire_transaction(parent_fd: int, name: str, subject: Subject) -> str:
    nonce = _transaction_nonce(name)
    role, transaction_fd = _read_transaction_marker(parent_fd, name, subject)
    try:
        if role == "active":
            _write_transaction_marker(transaction_fd, subject, "retirement", nonce)
    finally:
        os.close(transaction_fd)
    match = _ACTIVE_TRANSACTION.fullmatch(name)
    if match is None:
        retired_name = name
    else:
        retired_name = f".task-23019.txn-retired-{match.group(1)}"
        receipt = _content_receipt_at(parent_fd, name)
        _rename_entry(parent_fd, name, retired_name, receipt)
    return retired_name


def _recover_interrupted_promotion(
    destination: Path,
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
) -> None:
    parent = destination.parent
    if not parent.exists():
        return
    parent_fd, _ = _open_directory_nofollow(parent)
    try:
        names = set(os.listdir(parent_fd))
        transaction_names = sorted(
            name
            for name in names
            if _ACTIVE_TRANSACTION.fullmatch(name)
            or _RETIRED_TRANSACTION.fullmatch(name)
            or _PENDING_TRANSACTION.fullmatch(name)
            or _PREACTIVATION_TRANSACTION.fullmatch(name)
        )
        authority_names = sorted(name for name in names if _AUTHORITY.fullmatch(name))
        recognized = set(transaction_names) | set(authority_names)
        prefixed = {
            name
            for name in names
            if name.startswith(".task-23019.txn-")
            and _AUTHORITY_TEMP.fullmatch(name) is None
        } | {name for name in names if _PREACTIVATION_TRANSACTION.fullmatch(name)}
        if (
            prefixed != recognized
            or len(transaction_names) > 1
            or len(authority_names) > 1
        ):
            raise CloseoutError("promotion_collision")
        if not transaction_names and authority_names:
            authority_match = _AUTHORITY.fullmatch(authority_names[0])
            assert authority_match is not None
            authority_name, _directory, authority_receipt = _read_authority(
                parent_fd, authority_match.group(1), subject
            )
            _unlink_authority(parent_fd, authority_name, authority_receipt)
            os.fsync(parent_fd)
            return
        for name in transaction_names:
            nonce = _transaction_nonce(name)
            if _PREACTIVATION_TRANSACTION.fullmatch(name) and not authority_names:
                transaction_fd, opened = _open_child_directory(parent_fd, name)
                try:
                    if os.listdir(transaction_fd):
                        raise CloseoutError("promotion_collision")
                    os.fsync(transaction_fd)
                finally:
                    os.close(transaction_fd)
                _recheck_entry_receipt(parent_fd, name, opened)
                os.rmdir(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                continue
            if authority_names != [_authority_name(nonce)]:
                raise CloseoutError("promotion_collision")
            if _PREACTIVATION_TRANSACTION.fullmatch(name):
                authority_name, authority_receipt = _validate_authority(
                    parent_fd, nonce, subject, name
                )
                transaction_fd, opened = _open_child_directory(parent_fd, name)
                try:
                    if os.listdir(transaction_fd):
                        raise CloseoutError("promotion_collision")
                    os.fsync(transaction_fd)
                finally:
                    os.close(transaction_fd)
                _recheck_entry_receipt(parent_fd, name, opened)
                os.rmdir(name, dir_fd=parent_fd)
                _unlink_authority(parent_fd, authority_name, authority_receipt)
                os.fsync(parent_fd)
                continue
            if _PENDING_TRANSACTION.fullmatch(name):
                authority_name, authority_receipt = _validate_authority(
                    parent_fd, nonce, subject, name
                )
                transaction_fd, opened = _open_child_directory(parent_fd, name)
                try:
                    if not set(os.listdir(transaction_fd)).issubset(
                        {_TRANSACTION_MARKER, _TRANSACTION_MARKER_TEMP}
                    ):
                        raise CloseoutError("promotion_collision")
                    _delete_directory_contents(transaction_fd, marker_last=None)
                    os.fsync(transaction_fd)
                finally:
                    os.close(transaction_fd)
                _recheck_entry_receipt(parent_fd, name, opened)
                os.rmdir(name, dir_fd=parent_fd)
                _unlink_authority(parent_fd, authority_name, authority_receipt)
                os.fsync(parent_fd)
                continue
            transaction_fd, _opened = _open_child_directory(parent_fd, name)
            empty_retired = bool(
                _RETIRED_TRANSACTION.fullmatch(name) and not os.listdir(transaction_fd)
            )
            os.close(transaction_fd)
            if empty_retired:
                _delete_transaction(parent_fd, name, subject, subject_sources)
                continue
            role, transaction_fd = _read_transaction_marker(parent_fd, name, subject)
            try:
                if role == "retirement":
                    os.close(transaction_fd)
                    transaction_fd = -1
                    try:
                        retired = _retire_transaction(parent_fd, name, subject)
                        _delete_transaction(
                            parent_fd, retired, subject, subject_sources
                        )
                    except OSError:
                        raise CloseoutError("promotion_cleanup_failed") from None
                    continue
                entries = set(os.listdir(transaction_fd)) - {
                    _TRANSACTION_MARKER,
                    _TRANSACTION_MARKER_TEMP,
                }
                if not entries.issubset({"stage", "backup"}):
                    raise CloseoutError("promotion_collision")
                backup_present = "backup" in entries
                stage_present = "stage" in entries
                destination_present = destination.name in os.listdir(parent_fd)
                if backup_present:
                    backup_receipt = _validate_recovery_candidate_at(
                        transaction_fd,
                        "backup",
                        subject=subject,
                        subject_sources=subject_sources,
                        allow_bootstrap=True,
                    )
                    if destination_present:
                        _validate_recovery_candidate_at(
                            parent_fd,
                            destination.name,
                            subject=subject,
                            subject_sources=subject_sources,
                            allow_bootstrap=False,
                        )
                        if stage_present:
                            raise CloseoutError("promotion_collision")
                    else:
                        _rename_between(
                            transaction_fd,
                            "backup",
                            parent_fd,
                            destination.name,
                            backup_receipt,
                        )
                elif destination_present:
                    _validate_recovery_candidate_at(
                        parent_fd,
                        destination.name,
                        subject=subject,
                        subject_sources=subject_sources,
                        allow_bootstrap=True,
                    )
                os.close(transaction_fd)
                transaction_fd = -1
                retired = _retire_transaction(parent_fd, name, subject)
                try:
                    _delete_transaction(parent_fd, retired, subject, subject_sources)
                except OSError:
                    raise CloseoutError("promotion_cleanup_failed") from None
            finally:
                if transaction_fd >= 0:
                    os.close(transaction_fd)
    finally:
        os.close(parent_fd)


def _write_stage_contents(root_fd: int, artifacts: Mapping[str, bytes]) -> None:
    child_fds: dict[str, int] = {}
    try:
        for folder in ("facts", "captures"):
            if any(relative.startswith(folder + "/") for relative in artifacts):
                os.mkdir(folder, 0o700, dir_fd=root_fd)
                child_fds[folder], _ = _open_child_directory(root_fd, folder)
        for relative, payload in sorted(artifacts.items()):
            path = Path(relative)
            descriptor = root_fd if len(path.parts) == 1 else child_fds[path.parts[0]]
            name = path.name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            target_fd = os.open(name, flags, 0o600, dir_fd=descriptor)
            try:
                view = memoryview(payload)
                while view:
                    view = view[os.write(target_fd, view) :]
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
        for descriptor in child_fds.values():
            os.fsync(descriptor)
        os.fsync(root_fd)
    finally:
        for descriptor in child_fds.values():
            os.close(descriptor)


def _write_stage_at(transaction_fd: int, artifacts: Mapping[str, bytes]) -> None:
    os.mkdir("stage", 0o700, dir_fd=transaction_fd)
    stage_fd, _ = _open_child_directory(transaction_fd, "stage")
    try:
        _write_stage_contents(stage_fd, artifacts)
    finally:
        os.close(stage_fd)


def _rollback_owned_transaction(
    parent_fd: int,
    transaction_fd: int,
    destination_name: str,
    *,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
) -> None:
    entries = set(os.listdir(transaction_fd))
    if "backup" not in entries:
        return
    backup_receipt = _validate_recovery_candidate_at(
        transaction_fd,
        "backup",
        subject=subject,
        subject_sources=subject_sources,
        allow_bootstrap=True,
    )
    if destination_name in os.listdir(parent_fd):
        if "stage" in entries:
            raise CloseoutError("promotion_collision")
        target_receipt = _validate_recovery_candidate_at(
            parent_fd,
            destination_name,
            subject=subject,
            subject_sources=subject_sources,
            allow_bootstrap=False,
        )
        _rename_between(
            parent_fd,
            destination_name,
            transaction_fd,
            "stage",
            target_receipt,
        )
    _rename_between(
        transaction_fd,
        "backup",
        parent_fd,
        destination_name,
        backup_receipt,
    )


def promote_evidence(
    *,
    destination: Path,
    raw_root: Path,
    subject: Subject,
    subject_sources: Mapping[str, bytes],
    subject_hashes: Mapping[str, str],
    raw_artifacts: Mapping[str, bytes],
    catalogue: Mapping[str, Contract],
    automated_results: Mapping[str, object],
    live_results: Mapping[str, object],
    normalization_roots: Mapping[str, Path],
    not_applicable: Mapping[str, str] | None = None,
    credential_values: Collection[str] | None = None,
    inject_failure: str | None = None,
) -> None:
    """Validate a complete bundle and atomically replace one owned destination."""
    if os.name != "posix":
        raise CloseoutError("promotion_unsupported_platform")
    if inject_failure is not None and inject_failure not in INJECTABLE_PROMOTION_PHASES:
        raise CloseoutError("promotion_phase_invalid")
    if raw_root.exists() or raw_root.is_symlink():
        raise CloseoutError("raw_root_still_exists")
    _validate_subject_sources(subject_sources, subject_hashes)
    admitted_credentials = (
        tuple(
            value
            for name, value in os.environ.items()
            if value and _is_credential_environment_name(name)
        )
        if credential_values is None
        else credential_values
    )
    parent = destination.parent
    _recover_interrupted_promotion(
        destination, subject=subject, subject_sources=subject_sources
    )
    artifacts = _build_bundle(
        subject=subject,
        subject_sources=subject_sources,
        subject_hashes=subject_hashes,
        raw_artifacts=raw_artifacts,
        catalogue=catalogue,
        automated_results=automated_results,
        live_results=live_results,
        normalization_roots=normalization_roots,
        not_applicable=not_applicable,
        credential_values=admitted_credentials,
    )
    parent.mkdir(parents=True, exist_ok=True)
    parent_fd, _ = _open_directory_nofollow(parent)
    transaction_name: str | None = None
    transaction_fd = -1
    quarantined = False
    try:
        transaction_name = _create_transaction(parent_fd, subject)
        _role, transaction_fd = _read_transaction_marker(
            parent_fd, transaction_name, subject
        )
        if inject_failure == "during_stage_build":
            os.mkdir("stage", 0o700, dir_fd=transaction_fd)
            stage_fd, _ = _open_child_directory(transaction_fd, "stage")
            try:
                partial_fd = os.open(
                    "partial",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=stage_fd,
                )
                os.close(partial_fd)
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            raise CloseoutError("injected_promotion_failure")
        _write_stage_at(transaction_fd, artifacts)
        stage_receipt = _validate_recovery_candidate_at(
            transaction_fd,
            "stage",
            subject=subject,
            subject_sources=subject_sources,
            allow_bootstrap=False,
        )
        if inject_failure == "after_stage_validation":
            raise CloseoutError("injected_promotion_failure")
        if destination.name in os.listdir(parent_fd):
            destination_receipt = _validate_recovery_candidate_at(
                parent_fd,
                destination.name,
                subject=subject,
                subject_sources=subject_sources,
                allow_bootstrap=True,
            )
            try:
                _rename_between(
                    parent_fd,
                    destination.name,
                    transaction_fd,
                    "backup",
                    destination_receipt,
                )
            except BaseException:
                if "backup" in os.listdir(
                    transaction_fd
                ) and destination.name not in os.listdir(parent_fd):
                    quarantined = True
                raise
            quarantined = True
        if inject_failure == "after_target_to_backup":
            raise CloseoutError("injected_promotion_failure")
        _rename_between(
            transaction_fd,
            "stage",
            parent_fd,
            destination.name,
            stage_receipt,
        )
        if inject_failure == "after_stage_to_target":
            raise CloseoutError("injected_promotion_failure")
        _validate_bundle_at(
            parent_fd,
            destination.name,
            subject=subject,
            subject_sources=subject_sources,
        )
        if inject_failure == "before_backup_removal":
            raise CloseoutError("injected_promotion_failure")
        os.close(transaction_fd)
        transaction_fd = -1
        retired = _retire_transaction(parent_fd, transaction_name, subject)
        try:
            _delete_transaction(parent_fd, retired, subject, subject_sources)
        except OSError:
            raise CloseoutError("promotion_cleanup_failed") from None
    except CloseoutError as error:
        if quarantined and transaction_fd >= 0:
            try:
                _rollback_owned_transaction(
                    parent_fd,
                    transaction_fd,
                    destination.name,
                    subject=subject,
                    subject_sources=subject_sources,
                )
                os.close(transaction_fd)
                transaction_fd = -1
                assert transaction_name is not None
                retired = _retire_transaction(parent_fd, transaction_name, subject)
                _delete_transaction(parent_fd, retired, subject, subject_sources)
            except OSError:
                raise CloseoutError("promotion_cleanup_failed") from None
        elif (
            transaction_name is not None
            and transaction_fd >= 0
            and error.category != "injected_promotion_failure"
        ):
            os.close(transaction_fd)
            transaction_fd = -1
            try:
                retired = _retire_transaction(parent_fd, transaction_name, subject)
                _delete_transaction(parent_fd, retired, subject, subject_sources)
            except OSError:
                raise CloseoutError("promotion_cleanup_failed") from None
        raise error
    except (OSError, NotImplementedError):
        if quarantined and transaction_fd >= 0:
            try:
                _rollback_owned_transaction(
                    parent_fd,
                    transaction_fd,
                    destination.name,
                    subject=subject,
                    subject_sources=subject_sources,
                )
            except (OSError, CloseoutError):
                raise CloseoutError("promotion_collision") from None
        if transaction_name is not None and transaction_fd >= 0:
            os.close(transaction_fd)
            transaction_fd = -1
            try:
                retired = _retire_transaction(parent_fd, transaction_name, subject)
                _delete_transaction(parent_fd, retired, subject, subject_sources)
            except (OSError, CloseoutError):
                raise CloseoutError("promotion_cleanup_failed") from None
        raise CloseoutError("promotion_io_failed") from None
    finally:
        if transaction_fd >= 0:
            os.close(transaction_fd)
        os.close(parent_fd)


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


def load_subject_sources(
    repo: Path, subject: Subject
) -> tuple[dict[str, bytes], dict[str, str]]:
    """Read the three executable sources from the admitted Git object tree."""
    try:
        resolved_tree = _git(repo, "rev-parse", f"{subject.commit}^{{tree}}")
    except subprocess.CalledProcessError:
        raise CloseoutError("subject_tree_mismatch") from None
    if resolved_tree != subject.tree:
        raise CloseoutError("subject_tree_mismatch")
    sources: dict[str, bytes] = {}
    for filename in SOURCE_ARTIFACTS:
        object_name = f"{subject.commit}:{SOURCE_DIRECTORY}/{filename}"
        try:
            object_size = int(_git(repo, "cat-file", "-s", object_name))
        except (ValueError, subprocess.CalledProcessError):
            raise CloseoutError("subject_source_missing") from None
        if object_size > SOURCE_ARTIFACT_BYTE_LIMIT:
            raise CloseoutError("subject_source_too_large")
        completed = subprocess.run(
            [
                "git",
                "show",
                object_name,
            ],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            raise CloseoutError("subject_source_missing")
        if len(completed.stdout) != object_size:
            raise CloseoutError("subject_source_changed")
        sources[filename] = completed.stdout
    return sources, {
        filename: hashlib.sha256(payload).hexdigest()
        for filename, payload in sources.items()
    }


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


def _is_credential_environment_name(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in CREDENTIAL_ENV_NAMES or upper_name.endswith(
        CREDENTIAL_ENV_SUFFIXES
    )


def _is_preserved_environment_name(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in PRESERVED_ENV_NAMES or upper_name.startswith(
        PRESERVED_ENV_PREFIXES
    )


def prepare_scratch_environment(
    scratch: Path, *, environ: Mapping[str, str] | None = None
) -> ScratchEnvironment:
    """Validate and create all child-owned filesystem roots under scratch."""
    source = dict(os.environ if environ is None else environ)
    original_home = Path(source.get("HOME", os.path.expanduser("~"))).resolve()
    original_roots = (
        original_home,
        Path(source.get("XDG_CONFIG_HOME", original_home / ".config")).resolve(),
        Path(source.get("XDG_DATA_HOME", original_home / ".local/share")).resolve(),
        Path(source.get("XDG_CACHE_HOME", original_home / ".cache")).resolve(),
        Path(source.get("XDG_STATE_HOME", original_home / ".local/state")).resolve(),
    )

    scratch_root = scratch.resolve()
    owned_paths = {
        "HOME": scratch_root / "home",
        "XDG_CONFIG_HOME": scratch_root / "xdg-config",
        "XDG_DATA_HOME": scratch_root / "xdg-data",
        "XDG_CACHE_HOME": scratch_root / "xdg-cache",
        "XDG_STATE_HOME": scratch_root / "xdg-state",
        "TLDW_CONFIG_PATH": scratch_root / "xdg-config/tldw_cli/config.toml",
        "TMPDIR": scratch_root / "tmp",
        "TEMP": scratch_root / "tmp",
        "TMP": scratch_root / "tmp",
        "PATH": scratch_root / "bin",
    }
    resolved_owned = {name: path.resolve() for name, path in owned_paths.items()}
    child_stdin = (scratch_root / "stdin").resolve()
    if any(
        not path.is_relative_to(scratch_root) for path in resolved_owned.values()
    ) or not child_stdin.is_relative_to(scratch_root):
        raise CloseoutError("scratch_owner_escape")

    scratch_root.mkdir(parents=True, exist_ok=True)
    for name, path in resolved_owned.items():
        (path.parent if name == "TLDW_CONFIG_PATH" else path).mkdir(
            parents=True, exist_ok=True
        )
    child_stdin.touch(exist_ok=True)

    child_environment = {
        name: value
        for name, value in source.items()
        if _is_preserved_environment_name(name)
        and not _is_credential_environment_name(name)
    }
    child_environment.update({name: str(path) for name, path in resolved_owned.items()})
    child_environment.update(
        {
            "TLDW_TEST_MODE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    return ScratchEnvironment(
        env=child_environment,
        denied_roots=tuple(sorted({str(path) for path in original_roots})),
    )


def _bounded_diagnostic(
    value: object,
    *,
    secrets: Collection[str] = (),
    roots: Collection[str] = (),
    retain_tail: bool = False,
) -> str:
    """Normalize one diagnostic without retaining credentials or host paths."""
    text = "" if value is None else str(value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    for root in sorted((item for item in roots if item), key=len, reverse=True):
        text = text.replace(root, "<path>")
    text = _CREDENTIAL_ASSIGNMENT.sub(r"\1=<redacted>", text)
    text = _ABSOLUTE_PATH.sub("<path>", text)
    normalized = " ".join(text.split()).encode("utf-8")
    if len(normalized) <= MAX_DIAGNOSTIC_TEXT:
        return normalized.decode("utf-8")
    if not retain_tail:
        return normalized[:MAX_DIAGNOSTIC_TEXT].decode("utf-8", errors="ignore")
    separator = b" ... "
    head_size = (MAX_DIAGNOSTIC_TEXT - len(separator)) // 2
    tail_size = MAX_DIAGNOSTIC_TEXT - len(separator) - head_size
    return (
        normalized[:head_size].decode("utf-8", errors="ignore")
        + separator.decode()
        + normalized[-tail_size:].decode("utf-8", errors="ignore")
    )


def _child_failure_details(
    completed: subprocess.CompletedProcess[str],
    *,
    secrets: Collection[str],
    roots: Collection[str],
) -> dict[str, object]:
    details: dict[str, object] = {"returncode": completed.returncode}
    for stream in ("stdout", "stderr"):
        diagnostic = _bounded_diagnostic(
            getattr(completed, stream, ""),
            secrets=secrets,
            roots=roots,
            retain_tail=stream == "stderr",
        )
        if diagnostic:
            details[stream] = diagnostic
    return details


def _live_failure_details(
    payload: Mapping[str, object], *, roots: Collection[str]
) -> dict[str, object]:
    failures = []
    for name, cell in sorted(payload.items()):
        if isinstance(cell, dict) and cell.get("status") == "PASS":
            continue
        error_type = cell.get("error_type") if isinstance(cell, dict) else None
        message = cell.get("error") if isinstance(cell, dict) else None
        failures.append(
            {
                "cell": name,
                "error_type": _bounded_diagnostic(error_type or "InvalidResult")[
                    :MAX_ERROR_TYPE_TEXT
                ],
                "error": _bounded_diagnostic(
                    message or "Live cell did not pass", roots=roots
                ),
            }
        )
    return {"failures": failures}


def _with_live_case_details(
    live_case: str, details: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Add one bounded, sanitized live-root identity to failure details."""
    combined = dict(details or {})
    combined["live_case"] = _bounded_diagnostic(live_case) or "unknown"
    return combined


def _read_json_object(path: Path) -> tuple[dict[str, object] | None, str | None]:
    try:
        status = path.stat()
    except OSError:
        return None, "missing"
    if not stat.S_ISREG(status.st_mode):
        return None, "missing"
    if status.st_size > RAW_RESULT_BYTE_LIMIT:
        return None, "result_too_large"
    try:
        with path.open("rb") as handle:
            raw_payload = handle.read(RAW_RESULT_BYTE_LIMIT + 1)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "malformed_json"
    if len(raw_payload) > RAW_RESULT_BYTE_LIMIT:
        return None, "result_too_large"
    try:
        payload = json.loads(raw_payload)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        return None, "malformed_json"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def _write_capped(handle: object, chunk: bytes) -> None:
    remaining = CHILD_OUTPUT_BYTE_LIMIT - handle.tell()
    if remaining > 0:
        handle.write(chunk[:remaining])


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdin: object,
    scratch: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Drain both child streams while retaining at most a fixed file cap."""
    stdout_path = scratch / "child-stdout.log"
    stderr_path = scratch / "child-stderr.log"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process_pipes = (process.stdout, process.stderr)
    stream_specs = []
    outputs = []
    threads = []
    reader_errors = []
    cleanup_errors = []
    failure: BaseException | None = None
    returncode: int | None = None

    def close_endpoints() -> None:
        for stream in process_pipes:
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as error:
                cleanup_errors.append(error)
        for output in outputs:
            try:
                output.close()
            except BaseException as error:
                cleanup_errors.append(error)

    def drain(stream: object, output: object) -> None:
        try:
            while chunk := stream.read(CHILD_OUTPUT_CHUNK_BYTES):
                _write_capped(output, chunk)
        except BaseException as error:
            reader_errors.append(error)
            try:
                process.kill()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)

    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("child_pipe_missing")
        stream_specs.extend(
            (
                ("stdout", process.stdout, stdout_path),
                ("stderr", process.stderr, stderr_path),
            )
        )
        for _name, _stream, output_path in stream_specs:
            outputs.append(output_path.open("wb"))
        for (name, stream, _output_path), output in zip(
            stream_specs, outputs, strict=True
        ):
            thread = threading.Thread(
                target=drain,
                args=(stream, output),
                name=f"task23019-pipe-{name}",
            )
            thread.start()
            threads.append(thread)
        returncode = process.wait(timeout=timeout)
        if reader_errors:
            raise reader_errors[0]
    except BaseException as error:
        failure = error
    finally:
        if failure is not None:
            try:
                process.kill()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                process.wait(timeout=PROCESS_CLEANUP_TIMEOUT_SECONDS)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            close_endpoints()
        for thread in threads:
            try:
                thread.join(timeout=READER_JOIN_TIMEOUT_SECONDS)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        close_endpoints()
        for thread in threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=READER_JOIN_TIMEOUT_SECONDS)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
    if failure is not None:
        raise failure
    if reader_errors:
        raise reader_errors[0]
    if cleanup_errors:
        raise cleanup_errors[0]
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("pipe_reader_cleanup_failed")
    assert returncode is not None
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=stdout_path.read_bytes().decode("utf-8", errors="replace"),
        stderr=stderr_path.read_bytes().decode("utf-8", errors="replace"),
    )


def run_closeout_child(
    *,
    checkout: Path,
    scratch: Path,
    mode: str,
    target: Path,
    scenario: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ChildRunResult:
    """Run one bounded child using only explicit arguments, cwd, and environment."""
    if mode not in {"pytest", "live"}:
        raise CloseoutError("child_mode_not_defined")
    if mode == "live" and scenario is None:
        raise CloseoutError("scenario_not_defined")

    checkout_root = checkout.resolve()
    scratch_root = scratch.resolve()
    resolved_target = target.resolve()
    try:
        target_identity = resolved_target.relative_to(checkout_root).as_posix()
    except ValueError:
        target_identity = "<outside-checkout>"
    source_environment = os.environ if environ is None else environ
    credential_values = tuple(
        value
        for name, value in source_environment.items()
        if value and _is_credential_environment_name(name)
    )
    prepared = prepare_scratch_environment(scratch_root, environ=environ)
    command = [
        sys.executable,
        str(CHILD_PATH),
        "--checkout",
        str(checkout_root),
        "--scratch",
        str(scratch_root),
        "--mode",
        mode,
        "--target",
        str(resolved_target),
    ]
    for denied_root in prepared.denied_roots:
        command.extend(("--denied-root", denied_root))
    if scenario is not None:
        command.extend(("--scenario", scenario))

    try:
        with (scratch_root / "stdin").open("rb") as child_stdin:
            completed = _run_bounded_process(
                command,
                cwd=checkout_root,
                env=prepared.env,
                stdin=child_stdin,
                scratch=scratch_root,
                timeout=CHILD_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise CloseoutError("child_timeout") from None
    except Exception as error:
        diagnostic = _bounded_diagnostic(
            f"{type(error).__name__}: {error}",
            secrets=credential_values,
            roots=(str(checkout_root), str(scratch_root), *prepared.denied_roots),
        )
        raise CloseoutError(
            "child_failed",
            {
                "target": target_identity,
                "process": diagnostic or "process_failure",
            },
        ) from None
    if completed.returncode == CONTAINMENT_EXIT_STATUS:
        return ChildRunResult(
            returncode=completed.returncode,
            error="containment_failure",
            result_path=None,
            details={"returncode": completed.returncode, "target": target_identity},
        )

    result_path = scratch_root / (
        "automated-results.json" if mode == "pytest" else "live-results.json"
    )
    payload, result_problem = _read_json_object(result_path)
    details = (
        _child_failure_details(
            completed,
            secrets=credential_values,
            roots=(str(checkout_root), str(scratch_root), *prepared.denied_roots),
        )
        if completed.returncode or result_problem
        else None
    )
    if details is not None:
        details["target"] = target_identity
    child_error: str | None = None
    if result_problem is not None:
        assert details is not None
        details["result_parse"] = result_problem
        child_error = "child_failed"
    elif completed.returncode:
        assert payload is not None and details is not None
        raw_error = payload.get("error")
        if "error" in payload and not isinstance(raw_error, str):
            details["result_parse"] = "error_not_string"
            child_error = "child_failed"
        elif raw_error in CHILD_SEMANTIC_ERRORS:
            child_error = raw_error
        elif isinstance(raw_error, str):
            details["child_error"] = _bounded_diagnostic(
                raw_error,
                secrets=credential_values,
                roots=(
                    str(checkout_root),
                    str(scratch_root),
                    *prepared.denied_roots,
                ),
            )
            child_error = "child_failed"
        elif mode == "live":
            details["result_parse"] = "error_missing"
            child_error = "child_failed"
        else:
            child_error = "pytest_failed"
    return ChildRunResult(
        returncode=completed.returncode,
        error=child_error,
        result_path=result_path if result_path.is_file() else None,
        details=details,
    )


def run_development_live_cases(
    *, checkout: Path, scratch: Path, live_cases: tuple[str, ...]
) -> dict[str, object]:
    """Run declared live roots through isolated children and merge their cells."""
    if not SCENARIO_PATH.is_file():
        raise CloseoutError("scenario_not_defined")
    combined: dict[str, object] = {}
    for live_case in live_cases:
        expected_keys = EXPECTED_LIVE_RESULT_KEYS.get(live_case)
        if expected_keys is None:
            raise CloseoutError(
                "scenario_not_defined", _with_live_case_details(live_case)
            )
        case_scratch = scratch / "raw-results" / live_case
        try:
            result = run_closeout_child(
                checkout=checkout,
                scratch=case_scratch,
                mode="live",
                target=SCENARIO_PATH,
                scenario=live_case,
            )
        except CloseoutError as error:
            raise CloseoutError(
                error.category,
                _with_live_case_details(live_case, error.details),
            ) from error
        if result.error is not None:
            raise CloseoutError(
                result.error, _with_live_case_details(live_case, result.details)
            )
        if result.result_path is None:
            raise CloseoutError("child_failed", _with_live_case_details(live_case))
        payload, result_problem = _read_json_object(result.result_path)
        if result_problem is not None or payload is None:
            raise CloseoutError(
                "child_failed",
                _with_live_case_details(
                    live_case, {"result_parse": result_problem or "missing"}
                ),
            )
        if set(payload) != expected_keys:
            raise CloseoutError(
                "live_result_keys_mismatch", _with_live_case_details(live_case)
            )
        failure_details = _live_failure_details(
            payload, roots=(str(checkout.resolve()), str(scratch.resolve()))
        )
        if failure_details["failures"]:
            raise CloseoutError(
                "live_case_failed",
                _with_live_case_details(live_case, failure_details),
            )
        overlap = combined.keys() & payload.keys()
        if overlap:
            raise CloseoutError(
                "live_result_duplicate", _with_live_case_details(live_case)
            )
        combined.update(payload)
    return combined


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
    """Parse syntax with argparse, then reject semantic misuse stably.

    Args:
        arguments: Optional argument vector. Uses ``sys.argv`` when omitted.

    Returns:
        The validated parent-runner options.

    Raises:
        CloseoutError: The parsed options select an invalid mode or an
            incompatible combination of modes.
    """
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
    if verify_evidence_provided and (
        subject_revision_provided
        or parsed.promote
        or parsed.no_promote
        or live_case_provided
        or parsed.live_only
    ):
        raise CloseoutError("verify_evidence_mode_conflict")
    if live_case_provided and parsed.live_only:
        raise CloseoutError("live_selection_conflict")
    if live_case_provided and parsed.live_case not in EXECUTABLE_LIVE_ROOTS:
        raise CloseoutError("scenario_not_defined")
    if not parsed.development_run and (live_case_provided or parsed.live_only):
        raise CloseoutError("production_live_selection_conflict")
    if subject_revision_provided and not parsed.subject_revision.strip():
        raise CloseoutError("subject_revision_empty")
    if verify_evidence_provided and not parsed.verify_evidence.strip():
        raise CloseoutError("verify_evidence_path_empty")
    if parsed.promote and not subject_revision_provided:
        raise CloseoutError("promotion_subject_required")
    if parsed.promote and parsed.no_promote:
        raise CloseoutError("promotion_mode_conflict")
    live_cases = (
        EXECUTABLE_LIVE_ROOTS
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
        verify_evidence=Path(parsed.verify_evidence)
        if verify_evidence_provided
        else None,
    )


validate_catalogue(CATALOGUE)


def main(arguments: list[str] | None = None) -> int:
    """Execute development, production, or retained-evidence verification."""
    production_raw_active = False
    try:
        options = parse_options(arguments)
        checkout = Path(__file__).resolve().parents[5]
        if options.development_run:
            if not options.live_cases:
                raise CloseoutError("live_selection_required")
            with tempfile.TemporaryDirectory(prefix="task23019-") as raw:
                results = run_development_live_cases(
                    checkout=checkout,
                    scratch=Path(raw),
                    live_cases=options.live_cases,
                )
            print(
                json.dumps(
                    {"live_count": len(results), "results": results}, sort_keys=True
                )
            )
            return 0

        if options.verify_evidence is not None:
            artifacts = _read_artifact_tree(options.verify_evidence, raw=False)
            assert isinstance(artifacts, dict)
            try:
                manifest = json.loads(artifacts["manifest.json"])
            except (KeyError, UnicodeError, json.JSONDecodeError, RecursionError):
                raise CloseoutError("promotion_collision") from None
            if (
                not isinstance(manifest, dict)
                or not isinstance(manifest.get("subject_commit"), str)
                or not isinstance(manifest.get("subject_tree"), str)
            ):
                raise CloseoutError("promotion_collision")
            subject = Subject(
                commit=manifest["subject_commit"], tree=manifest["subject_tree"]
            )
            subject_sources, _subject_hashes = load_subject_sources(checkout, subject)
            _validate_bundle(
                options.verify_evidence,
                subject=subject,
                subject_sources=subject_sources,
            )
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "subject_commit": subject.commit,
                        "subject_tree": subject.tree,
                        "verified_evidence": str(options.verify_evidence),
                    },
                    sort_keys=True,
                )
            )
            return 0

        if not options.promote and not options.no_promote:
            raise CloseoutError("run_mode_not_implemented")
        if options.subject_revision is None:
            raise CloseoutError("subject_revision_required")
        environment_revision = os.environ.get("TASK23019_SUBJECT_REVISION")
        if not environment_revision or not environment_revision.strip():
            raise CloseoutError("subject_revision_environment_required")
        if environment_revision != options.subject_revision:
            raise CloseoutError("subject_revision_environment_mismatch")

        subject = admit_subject(checkout, options.subject_revision)
        subject_sources, subject_hashes = load_subject_sources(checkout, subject)
        live_cases = options.live_cases or EXECUTABLE_LIVE_ROOTS
        production_raw_active = True
        with tempfile.TemporaryDirectory(prefix="task23019-") as raw:
            raw_root = Path(raw)
            automated_results: dict[str, object] = {}
            declared_selectors = tuple(
                selector
                for contract in CATALOGUE.values()
                for selector in contract.automated_nodes
            )
            if not options.live_only:
                for index, relative in enumerate(CURATED_PYTEST_FILES):
                    result = run_closeout_child(
                        checkout=checkout,
                        scratch=raw_root / f"raw-results/automated/{index:02}",
                        mode="pytest",
                        target=checkout / relative,
                    )
                    if result.error is not None:
                        raise CloseoutError(result.error, result.details)
                    if result.result_path is None:
                        raise CloseoutError("child_failed")
                    payload, result_problem = _read_json_object(result.result_path)
                    if result_problem is not None or payload is None:
                        raise CloseoutError(
                            "child_failed",
                            {"result_parse": result_problem or "missing"},
                        )
                    retained_payload = {
                        node_id: value
                        for node_id, value in payload.items()
                        if any(
                            matching_node_ids(selector, (node_id,))
                            for selector in declared_selectors
                        )
                    }
                    if automated_results.keys() & retained_payload.keys():
                        raise CloseoutError("automated_result_duplicate")
                    automated_results.update(retained_payload)

            live_results = run_development_live_cases(
                checkout=checkout,
                scratch=raw_root,
                live_cases=live_cases,
            )
            validate_complete_results(CATALOGUE, automated_results, live_results)

            retained = raw_root / "retained"
            facts = retained / "facts"
            captures = retained / "captures"
            facts.mkdir(parents=True)
            captures.mkdir()
            (retained / "summary.json").write_bytes(
                _json_bytes(_canonical_summary(automated_results, live_results))
            )
            inventory = [
                *(
                    ("automated", name, value)
                    for name, value in automated_results.items()
                ),
                *(("live", name, value) for name, value in live_results.items()),
            ]
            for index, (kind, name, value) in enumerate(sorted(inventory)):
                fact = {
                    "kind": kind,
                    "result_name": name,
                    "status": _result_status(value),
                }
                if kind == "live":
                    if not isinstance(value, Mapping):
                        raise CloseoutError("evidence_inventory_invalid")
                    fact.update(value)
                (facts / f"result-{index:03}.json").write_bytes(_json_bytes(fact))
            results_by_kind = {
                "automated": automated_results,
                "live": live_results,
            }
            for stem, (kind, name) in REPRESENTATIVE_CAPTURES.items():
                source_root, source_stem = REPRESENTATIVE_CAPTURE_SOURCES[stem]
                raw_capture_root = (
                    raw_root / "raw-results" / source_root / "raw-evidence/captures"
                )
                try:
                    body = _read_regular_file(
                        raw_capture_root / f"{source_stem}.txt",
                        limit=TEXT_ARTIFACT_BYTE_LIMIT,
                    )
                    svg = _read_regular_file(
                        raw_capture_root / f"{source_stem}.svg",
                        limit=SVG_ARTIFACT_BYTE_LIMIT,
                    ).decode("utf-8")
                except (OSError, UnicodeError):
                    raise CloseoutError("representative_capture_missing") from None
                status = _result_status(results_by_kind[kind][name])
                (captures / f"{stem}.txt").write_bytes(
                    (f"result_name: {name}\nstatus: {status}\n").encode() + body
                )
                svg_index = svg.find("<svg")
                if svg_index < 0:
                    raise CloseoutError("representative_capture_invalid")
                svg = (
                    svg[: svg_index + 4]
                    + f' data-result-name="{name}" data-status="{status}"'
                    + svg[svg_index + 4 :]
                )
                (captures / f"{stem}.svg").write_text(svg, encoding="utf-8")
            raw_artifacts = collect_raw_artifacts(retained)
            normalization_roots = {
                "checkout": checkout,
                "runtime": Path(sys.prefix),
                "scratch": raw_root,
            }
        production_raw_active = False

        verify_subject_tree(checkout, subject)
        if options.promote:
            promote_evidence(
                destination=checkout / SOURCE_DIRECTORY,
                raw_root=raw_root,
                subject=subject,
                subject_sources=subject_sources,
                subject_hashes=subject_hashes,
                raw_artifacts=raw_artifacts,
                catalogue=CATALOGUE,
                automated_results=automated_results,
                live_results=live_results,
                normalization_roots=normalization_roots,
            )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "subject_commit": subject.commit,
                    "subject_tree": subject.tree,
                    "automated_count": len(automated_results),
                    "live_count": len(live_results),
                    "promoted": options.promote,
                },
                sort_keys=True,
            )
        )
        return 0
    except CloseoutError as error:
        failure = {"error": error.category}
        if error.details:
            failure["details"] = error.details
    except OSError as error:
        if not production_raw_active:
            raise
        primary = error.__context__
        if isinstance(primary, CloseoutError):
            failure = {"error": primary.category}
            if primary.details:
                failure["details"] = primary.details
        else:
            failure = {"error": "production_evidence_io_failed"}
    print(json.dumps(failure, sort_keys=True), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
