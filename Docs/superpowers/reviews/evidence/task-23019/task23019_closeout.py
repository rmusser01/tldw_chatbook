"""Task-local contracts for the TASK-23019 adaptive-reader closeout."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path


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
MAX_DIAGNOSTIC_TEXT = 512
MAX_ERROR_TYPE_TEXT = 80
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
EXECUTABLE_LIVE_ROOTS = (
    "common_matrix",
    "media_capability",
    "conversations_capability",
    "notes_capability",
    "prompts_capability",
    "skills_capability",
)
EXPECTED_LIVE_RESULT_KEYS = {
    "common_matrix": frozenset(
        f"{destination}-{width}x{height}"
        for destination in DESTINATIONS
        for width, height in SIZES
    ),
    **{
        root: frozenset({root})
        for root in EXECUTABLE_LIVE_ROOTS
        if root != "common_matrix"
    },
}
DURABLE_PYTEST_SELECTORS = (
    "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_resize_is_presentation_only",
    "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_preferences_restore_in_fresh_screen",
    "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
)
DURABLE_EVIDENCE_ALIASES = {
    "resize_purity": (DURABLE_PYTEST_SELECTORS[0],),
    "single_app_route_cycle": (DURABLE_PYTEST_SELECTORS[2],),
}

_ABSOLUTE_PATH = re.compile(r"(?<!>)(?:[A-Za-z]:[\\/]|/)[^\s\"']+")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|PASSWORD|SECRET|CREDENTIALS?|PRIVATE[_-]?KEY))"
    r"\s*[:=]\s*[^\s,;]+"
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
        if any(
            key not in EXECUTABLE_LIVE_ROOTS and key not in DURABLE_EVIDENCE_ALIASES
            for key in contract.live_cases
        ):
            raise CloseoutError("live_case_not_mapped")


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
) -> str:
    """Normalize one diagnostic without retaining credentials or host paths."""
    text = "" if value is None else str(value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    for root in sorted((item for item in roots if item), key=len, reverse=True):
        text = text.replace(root, "<path>")
    text = _CREDENTIAL_ASSIGNMENT.sub(r"\1=<redacted>", text)
    text = _ABSOLUTE_PATH.sub("<path>", text)
    return " ".join(text.split())[:MAX_DIAGNOSTIC_TEXT]


def _child_failure_details(
    completed: subprocess.CompletedProcess[str],
    *,
    secrets: Collection[str],
    roots: Collection[str],
) -> dict[str, object]:
    details: dict[str, object] = {"returncode": completed.returncode}
    for stream in ("stdout", "stderr"):
        diagnostic = _bounded_diagnostic(
            getattr(completed, stream, ""), secrets=secrets, roots=roots
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
    except (UnicodeError, json.JSONDecodeError):
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
        str(target.resolve()),
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
            "child_failed", {"process": diagnostic or "process_failure"}
        ) from None
    if completed.returncode == CONTAINMENT_EXIT_STATUS:
        return ChildRunResult(
            returncode=completed.returncode,
            error="containment_failure",
            result_path=None,
            details={"returncode": completed.returncode},
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
            raise CloseoutError("scenario_not_defined")
        case_scratch = scratch / "raw-results" / live_case
        result = run_closeout_child(
            checkout=checkout,
            scratch=case_scratch,
            mode="live",
            target=SCENARIO_PATH,
            scenario=live_case,
        )
        if result.error is not None:
            raise CloseoutError(result.error, result.details)
        if result.result_path is None:
            raise CloseoutError("child_failed")
        payload, result_problem = _read_json_object(result.result_path)
        if result_problem is not None or payload is None:
            raise CloseoutError(
                "child_failed", {"result_parse": result_problem or "missing"}
            )
        if set(payload) != expected_keys:
            raise CloseoutError("live_result_keys_mismatch")
        failure_details = _live_failure_details(
            payload, roots=(str(checkout.resolve()), str(scratch.resolve()))
        )
        if failure_details["failures"]:
            raise CloseoutError("live_case_failed", failure_details)
        overlap = combined.keys() & payload.keys()
        if overlap:
            raise CloseoutError("live_result_duplicate")
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
    if live_case_provided and parsed.live_case not in EXECUTABLE_LIVE_ROOTS:
        raise CloseoutError("scenario_not_defined")

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
    """Execute the development-only Task 4 live path."""
    try:
        options = parse_options(arguments)
        if not options.development_run:
            raise CloseoutError("run_mode_not_implemented")
        if not options.live_cases:
            raise CloseoutError("live_selection_required")
        checkout = Path(__file__).resolve().parents[5]
        with tempfile.TemporaryDirectory(prefix="task23019-") as raw:
            results = run_development_live_cases(
                checkout=checkout,
                scratch=Path(raw),
                live_cases=options.live_cases,
            )
        print(
            json.dumps({"live_count": len(results), "results": results}, sort_keys=True)
        )
        return 0
    except CloseoutError as error:
        failure = {"error": error.category}
        if error.details:
            failure["details"] = error.details
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
