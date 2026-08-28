from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.parse
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py"
)
CHILD_PATH = (
    REPO_ROOT
    / "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py"
)
SCENARIO_PATH = (
    REPO_ROOT / "Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py"
)

NETWORK_ATTEMPTS = (
    ("socket.connect", 'socket.socket().connect(("127.0.0.1", 9))'),
    ("socket.sendto", 'socket.socket().sendto(b"x", ("127.0.0.1", 9))'),
    *(
        (("socket.sendmsg", 'socket.socket().sendmsg([b"x"])'),)
        if hasattr(socket.socket, "sendmsg")
        else ()
    ),
    ("socket.bind", 'socket.socket().bind(("127.0.0.1", 0))'),
    ("socket.listen", "socket.socket().listen()"),
    ("socket.getaddrinfo", 'socket.getaddrinfo("example.invalid", 80)'),
    ("socket.gethostbyname", 'socket.gethostbyname("example.invalid")'),
    ("socket.gethostbyname_ex", 'socket.gethostbyname_ex("example.invalid")'),
    ("socket.gethostbyaddr", 'socket.gethostbyaddr("127.0.0.1")'),
    ("socket.getnameinfo", 'socket.getnameinfo(("127.0.0.1", 80), 0)'),
)


def _available_process_apis() -> tuple[str, ...]:
    names = ["subprocess.Popen", "os.system"]
    names.extend(
        f"os.{name}"
        for name in sorted(dir(os))
        if callable(getattr(os, name, None))
        and (
            name.startswith("spawn")
            or name.startswith("exec")
            or name in {"posix_spawn", "posix_spawnp"}
        )
    )
    return tuple(names)


PROCESS_APIS = _available_process_apis()
METADATA_READ_APIS = (
    *tuple(
        name
        for name in ("stat", "lstat", "access", "readlink", "statvfs", "pathconf")
        if callable(getattr(os, name, None))
    ),
    "path.realpath",
)
FILESYSTEM_MUTATOR_APIS = tuple(
    name
    for name in ("mkfifo", "mknod", "chflags", "lchflags")
    if callable(getattr(os, name, None))
)
FD_MUTATION_APIS = tuple(
    name for name in ("fchmod", "ftruncate") if callable(getattr(os, name, None))
)
DUPLICATION_APIS = tuple(
    name for name in ("dup", "dup2", "dup3") if callable(getattr(os, name, None))
)
HAS_DIRECTORY_FD_TRAVERSAL = hasattr(os, "O_DIRECTORY") and os.open in getattr(
    os, "supports_dir_fd", ()
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


def test_parent_module_does_not_require_selectors_for_pipe_drain(monkeypatch):
    real_import = builtins.__import__

    def without_selectors(name, *args, **kwargs):
        if name == "selectors":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_selectors)

    module = _load_runner()

    assert callable(module._run_bounded_process)


def _load_child():
    spec = importlib.util.spec_from_file_location(
        "task23019_closeout_child", CHILD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scenarios():
    spec = importlib.util.spec_from_file_location("task23019_scenarios", SCENARIO_PATH)
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


def _expected_live_keys(root: str) -> set[str]:
    if root == "common_matrix":
        return {
            f"{destination}-{width}x{height}"
            for destination in ("media", "conversations", "notes", "prompts", "skills")
            for width, height in ((160, 50), (120, 35), (100, 30), (80, 24))
        }
    if root == "resize_purity":
        return {
            f"{destination}-resize-purity"
            for destination in ("media", "conversations", "notes", "prompts", "skills")
        }
    if root == "preferences_fresh_reload":
        return {"preferences-fresh-reload"}
    if root == "single_app_route_cycle":
        return {"single-app-route-cycle"}
    return {root}


def _passing_live_payload(root: str) -> dict[str, dict[str, str]]:
    return {key: {"status": "PASS"} for key in _expected_live_keys(root)}


def test_child_module_imports_when_fcntl_and_msvcrt_are_unavailable(monkeypatch):
    real_import = builtins.__import__

    def without_platform_fd_modules(name, *args, **kwargs):
        if name in {"fcntl", "msvcrt"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_platform_fd_modules)

    module = _load_child()

    assert module._REAL_FCNTL is None
    assert module._REAL_GET_OSFHANDLE is None


def test_child_platform_descriptor_token_invalidates_reopened_handle(
    monkeypatch, tmp_path
):
    module = _load_child()
    tracked = tmp_path / "tracked.bin"
    handles = {}
    module._SCRATCH = str(tmp_path.resolve())
    module._READ_ROOTS = (str(tmp_path.resolve()),)
    descriptor = None
    reopened = None
    try:
        monkeypatch.setattr(module, "_REAL_FCNTL", None)
        monkeypatch.setattr(
            module,
            "_REAL_GET_OSFHANDLE",
            lambda candidate: handles.setdefault(candidate, 101),
            raising=False,
        )
        descriptor = module._guarded_os_open(tracked, os.O_RDWR | os.O_CREAT, 0o600)

        authority = module._fd_authority(descriptor)
        assert authority is not None
        assert authority[2] is True
        os.fdopen(descriptor, "r+b").close()
        reopened = tracked.open("rb")
        assert reopened.fileno() == descriptor
        handles[descriptor] = 202
        assert module._fd_authority(descriptor) is None
        assert descriptor not in module._OPEN_FDS
    finally:
        if reopened is not None:
            reopened.close()
        elif descriptor is not None and module._fd_identity(descriptor) is not None:
            os.close(descriptor)


def test_child_platform_write_probe_invalidates_same_handle_read_only_reuse(
    monkeypatch, tmp_path
):
    module = _load_child()
    tracked = tmp_path / "tracked.bin"
    tracked.write_bytes(b"payload")
    module._SCRATCH = str(tmp_path.resolve())
    module._READ_ROOTS = (str(tmp_path.resolve()),)
    descriptor = None
    reopened = None
    try:
        monkeypatch.setattr(module, "_REAL_FCNTL", None)
        monkeypatch.setattr(
            module, "_REAL_GET_OSFHANDLE", lambda _candidate: 101, raising=False
        )
        descriptor = module._guarded_os_open(tracked, os.O_RDWR)
        assert module._fd_authority(descriptor) is not None

        os.fdopen(descriptor, "r+b").close()
        reopened = tracked.open("rb")
        assert reopened.fileno() == descriptor
        assert module._fd_authority(descriptor) is None
        assert descriptor not in module._OPEN_FDS
    finally:
        if reopened is not None:
            reopened.close()
        elif descriptor is not None and module._fd_identity(descriptor) is not None:
            os.close(descriptor)


def test_child_platform_write_probe_preserves_same_handle_without_side_effects(
    monkeypatch, tmp_path
):
    module = _load_child()
    tracked = tmp_path / "tracked.bin"
    tracked.write_bytes(b"payload")
    before = tracked.stat()
    module._SCRATCH = str(tmp_path.resolve())
    module._READ_ROOTS = (str(tmp_path.resolve()),)
    monkeypatch.setattr(module, "_REAL_FCNTL", None)
    monkeypatch.setattr(
        module, "_REAL_GET_OSFHANDLE", lambda _candidate: 101, raising=False
    )

    descriptor = module._guarded_os_open(tracked, os.O_RDWR)
    try:
        assert module._fd_authority(descriptor) is not None
        assert module._fd_authority(descriptor) is not None
    finally:
        os.close(descriptor)

    after = tracked.stat()
    assert tracked.read_bytes() == b"payload"
    assert after.st_atime_ns == before.st_atime_ns
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns == before.st_ctime_ns


def test_child_descriptor_authority_fails_closed_without_a_platform_token(
    monkeypatch, tmp_path
):
    module = _load_child()
    tracked = tmp_path / "tracked.bin"
    module._SCRATCH = str(tmp_path.resolve())
    module._READ_ROOTS = (str(tmp_path.resolve()),)
    monkeypatch.setattr(module, "_REAL_FCNTL", None)
    monkeypatch.setattr(module, "_REAL_GET_OSFHANDLE", None)

    descriptor = module._guarded_os_open(tracked, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert module._fd_identity(descriptor) is None
        assert descriptor not in module._OPEN_FDS
        assert module._fd_authority(descriptor) is None
    finally:
        os.close(descriptor)


def test_runtime_authority_uses_injected_present_and_missing_paths(tmp_path):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    runtime = tmp_path / "runtime"
    present = runtime / "site-packages"
    missing_zip = runtime / "python-test.zip"
    missing_locale = tmp_path / "base/share/locale"
    checkout.mkdir()
    scratch.mkdir()
    present.mkdir(parents=True)

    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(str(present), str(missing_zip)),
        locale_root=str(missing_locale),
        language="en_US.UTF-8",
    )

    assert str(present.resolve()) in roots
    assert str(missing_zip.resolve()) in files
    assert str(runtime.resolve()) not in roots
    assert str(missing_zip.with_name("sibling.zip").resolve()) not in files
    assert str(missing_locale.resolve()) in files
    assert str(missing_locale.resolve()) not in roots
    assert (
        str((missing_locale / "en_US.UTF-8/LC_MESSAGES/messages.mo").resolve()) in files
    )
    assert str((missing_locale / "undeclared").resolve()) not in files

    module._READ_ROOTS = roots
    module._READ_FILES = files
    module._SCRATCH = str(scratch.resolve())
    assert module._read_allowed(missing_zip)
    assert not module._read_allowed(missing_zip.parent)
    assert not module._read_allowed(missing_zip.with_name("sibling.zip"))
    assert not module._write_allowed(missing_zip)


def test_catalogue_is_finite_unique_and_has_both_evidence_kinds():
    module = _load_runner()

    assert set(module.CATALOGUE) == EXPECTED_IDS
    assert all(entry.automated_nodes for entry in module.CATALOGUE.values())
    assert all(entry.live_cases for entry in module.CATALOGUE.values())
    assert all(
        isinstance(entry, module.Contract) for entry in module.CATALOGUE.values()
    )
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
            ("preferences_fresh_reload", "single_app_route_cycle"),
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


def test_actual_child_results_use_parent_settlement_vocabulary(tmp_path):
    module = _load_runner()
    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, "assert True", must_not_continue=False
    )
    node_id = "test_boundary_attempt.py::test_attempt"
    recorded = json.loads((scratch / "automated-results.json").read_text())
    catalogue = {"synthetic": module.Contract((node_id,), ("live",))}

    assert result.returncode == 0
    module.validate_automated_results(catalogue, recorded)


@pytest.mark.parametrize("resolver", ("_resolved", "_guarded_realpath"))
def test_nested_resolution_restores_an_outer_active_guard(resolver):
    child = _load_child()
    child._READ_ROOTS = (str(REPO_ROOT.resolve()),)

    class NestedPath:
        def __fspath__(self):
            child._resolved(REPO_ROOT)
            return str(REPO_ROOT)

    child._RESOLVING_PATH.active = True
    try:
        getattr(child, resolver)(NestedPath())
        assert child._RESOLVING_PATH.active is True
    finally:
        child._RESOLVING_PATH.active = False


def test_child_read_authority_fails_closed_for_an_untracked_descriptor():
    child = _load_child()
    child._READ_ROOTS = (str(REPO_ROOT.resolve()),)

    assert child._read_allowed(123456789) is False


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
        (
            ["--promote", "--subject-revision", "abc", "--no-promote"],
            "promotion_mode_conflict",
        ),
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

    promotable = module.parse_options(["--subject-revision", "abc", "--promote"])
    assert promotable.subject_revision == "abc"
    assert promotable.promote is True
    assert promotable.live_cases == ()

    selected = module.parse_options(
        ["--development-run", "--live-case", "common_matrix"]
    )
    assert selected.live_cases == ("common_matrix",)

    live_only = module.parse_options(["--development-run", "--live-only"])
    assert live_only.live_only is True
    assert live_only.no_promote is True
    assert live_only.live_cases == module.EXECUTABLE_LIVE_ROOTS

    evidence = module.parse_options(["--verify-evidence", "some/bundle"])
    assert evidence.verify_evidence == Path("some/bundle")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--subject-revision", "abc", "--promote", "--live-case", "common_matrix"],
        ["--subject-revision", "abc", "--promote", "--live-only"],
        ["--subject-revision", "abc", "--no-promote", "--live-case", "common_matrix"],
        ["--subject-revision", "abc", "--no-promote", "--live-only"],
    ],
)
def test_production_rejects_partial_live_selection(arguments):
    module = _load_runner()

    with pytest.raises(module.CloseoutError) as error:
        module.parse_options(arguments)

    assert error.value.category == "production_live_selection_conflict"


@pytest.mark.parametrize(
    "conflicting",
    [
        ["--subject-revision", "abc"],
        ["--promote"],
        ["--no-promote"],
        ["--live-case", "common_matrix"],
        ["--live-only"],
    ],
)
def test_verify_evidence_is_exclusive_with_execution_options(conflicting):
    module = _load_runner()

    with pytest.raises(module.CloseoutError) as error:
        module.parse_options(["--verify-evidence", "bundle", *conflicting])

    assert error.value.category == "verify_evidence_mode_conflict"


@pytest.mark.parametrize(
    "durable_key",
    ("resize_purity", "preferences_fresh_reload", "single_app_route_cycle"),
)
def test_catalogue_durable_key_is_a_concrete_live_scenario(durable_key):
    module = _load_runner()

    options = module.parse_options(["--development-run", "--live-case", durable_key])

    assert options.live_cases == (durable_key,)
    assert module.EXPECTED_LIVE_RESULT_KEYS[durable_key] == frozenset(
        _expected_live_keys(durable_key)
    )


def test_complete_live_inventory_includes_durable_oracles():
    module = _load_runner()
    expected = {
        *(_expected_live_keys("common_matrix")),
        *(_expected_live_keys("media_capability")),
        *(_expected_live_keys("conversations_capability")),
        *(_expected_live_keys("notes_capability")),
        *(_expected_live_keys("prompts_capability")),
        *(_expected_live_keys("skills_capability")),
        *(_expected_live_keys("resize_purity")),
        *(_expected_live_keys("preferences_fresh_reload")),
        *(_expected_live_keys("single_app_route_cycle")),
    }

    assert module.EXPECTED_CONCRETE_LIVE_RESULTS == frozenset(expected)
    assert len(expected) == 32
    for contract_id in ("SH-03", "SH-06", "SH-07"):
        assert all(
            module._live_result_names(root)
            for root in module.CATALOGUE[contract_id].live_cases
        )


@pytest.mark.parametrize("arguments", [["--not-an-option"], ["--subject-revision"]])
def test_argparse_still_owns_unknown_flags_and_malformed_values(arguments):
    module = _load_runner()

    with pytest.raises(SystemExit) as error:
        module.parse_options(arguments)

    assert error.value.code == 2


def _process_attempt(api: str) -> str:
    if api == "subprocess.Popen":
        return 'subprocess.Popen(["/definitely/not/a/task23019-program"])'
    if api == "os.system":
        return 'os.system("exit 0")'
    name = api.removeprefix("os.")
    if name in {"posix_spawn", "posix_spawnp"}:
        return (
            f'os.{name}("/definitely/not/a/task23019-program", '
            '["task23019-program"], os.environ.copy())'
        )
    if name.startswith("spawnl"):
        suffix = ", os.environ.copy()" if name.endswith("e") else ""
        return (
            f'os.{name}(os.P_NOWAIT, "/definitely/not/a/task23019-program", '
            f'"task23019-program"{suffix})'
        )
    if name.startswith("spawnv"):
        suffix = ", os.environ.copy()" if name.endswith("e") else ""
        return (
            f'os.{name}(os.P_NOWAIT, "/definitely/not/a/task23019-program", '
            f'["task23019-program"]{suffix})'
        )
    if name.startswith("execl"):
        suffix = ", os.environ.copy()" if name.endswith("e") else ""
        return (
            f'os.{name}("/definitely/not/a/task23019-program", '
            f'"task23019-program"{suffix})'
        )
    if name.startswith("execv"):
        suffix = ", os.environ.copy()" if name.endswith("e") else ""
        return (
            f'os.{name}("/definitely/not/a/task23019-program", '
            f'["task23019-program"]{suffix})'
        )
    raise AssertionError(f"unmapped process API: {api}")


def _metadata_read_attempt(api: str, path: Path) -> str:
    if api == "path.realpath":
        return f"os.path.realpath({str(path)!r})"
    if api == "access":
        return f"os.access({str(path)!r}, os.R_OK)"
    if api == "pathconf":
        return f"os.pathconf({str(path)!r}, 'PC_NAME_MAX')"
    return f"os.{api}({str(path)!r})"


def _filesystem_mutator_attempt(api: str, path: Path) -> str:
    if api in {"chflags", "lchflags"}:
        return f"os.{api}({str(path)!r}, 0)"
    return f"os.{api}({str(path)!r})"


def _duplicate_descriptor(api: str, source: str, destination: str) -> str:
    if api == "dup":
        return f"duplicated_fd = os.dup({source})"
    flags = ", 0" if api == "dup3" else ""
    return f"duplicated_fd = os.{api}({source}, {destination}{flags})"


def _run_pytest_child(
    module,
    tmp_path: Path,
    body: str,
    *,
    environ=None,
    must_not_continue: bool = True,
    support_file: Path | None = None,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if support_file is not None:
        (checkout / support_file.name).write_bytes(support_file.read_bytes())
    test_file = checkout / "test_boundary_attempt.py"
    test_file.write_text(
        "import os\nimport socket\nimport subprocess\n\n"
        "def test_attempt():\n"
        "    try:\n"
        + textwrap.indent(body, "        ")
        + "\n    except BaseException:\n"
        "        pass\n"
        + (
            '    raise AssertionError("prohibited attempt continued")\n'
            if must_not_continue
            else ""
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="pytest",
        target=test_file,
        environ=environ,
    )
    return result, scratch, checkout


def _assert_containment_attempt(module, result, scratch: Path, category: str) -> None:
    assert result.returncode == module.CONTAINMENT_EXIT_STATUS
    assert result.error == "containment_failure"
    assert result.result_path is None
    assert (scratch / "attempts.jsonl").read_bytes() == (
        b'{"category":"' + category.encode() + b'"}\n'
    )


@pytest.mark.parametrize(("api", "body"), NETWORK_ATTEMPTS, ids=lambda value: value)
def test_child_immediately_exits_for_each_prohibited_network_api(api, body, tmp_path):
    module = _load_runner()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    _assert_containment_attempt(module, result, scratch, "network_denied")


@pytest.mark.parametrize("api", PROCESS_APIS)
def test_child_immediately_exits_for_each_prohibited_process_api(api, tmp_path):
    module = _load_runner()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, _process_attempt(api)
    )

    _assert_containment_attempt(module, result, scratch, "process_denied")


@pytest.mark.parametrize("api", METADATA_READ_APIS)
def test_child_immediately_exits_for_each_prohibited_metadata_read_api(api, tmp_path):
    module = _load_runner()
    denied_home = tmp_path / "fake-real-home"
    denied_entry = denied_home / "profile-entry"
    environment = {**os.environ, "HOME": str(denied_home)}

    result, scratch, _checkout = _run_pytest_child(
        module,
        tmp_path,
        _metadata_read_attempt(api, denied_entry),
        environ=environment,
    )

    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.parametrize("api", ("stat", "path.realpath"))
def test_child_pathlike_conversion_cannot_bypass_metadata_tripwire(api, tmp_path):
    module = _load_runner()
    denied_home = tmp_path / "fake-real-home"
    denied_entry = denied_home / "profile-entry"
    environment = {**os.environ, "HOME": str(denied_home)}
    call = "os.stat(path)" if api == "stat" else "os.path.realpath(path)"
    body = textwrap.dedent(
        f"""
        class ReentrantPath:
            attempted = False

            def __fspath__(self):
                if not self.attempted:
                    self.attempted = True
                    try:
                        os.stat({str(denied_entry)!r})
                    except BaseException:
                        pass
                return __file__

        path = ReentrantPath()
        {call}
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module,
        tmp_path,
        body,
        environ=environment,
    )

    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.parametrize("api", FILESYSTEM_MUTATOR_APIS)
def test_child_immediately_exits_for_each_prohibited_filesystem_mutator(api, tmp_path):
    module = _load_runner()
    outside_scratch = (
        tmp_path
        / "checkout"
        / (
            "test_boundary_attempt.py"
            if api in {"chflags", "lchflags"}
            else f"{api}-outside-scratch"
        )
    )

    result, scratch, _checkout = _run_pytest_child(
        module,
        tmp_path,
        _filesystem_mutator_attempt(api, outside_scratch),
    )

    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")
    if api in {"mkfifo", "mknod"}:
        assert not outside_scratch.exists()


def test_child_records_and_blocks_read_from_real_profile(tmp_path):
    module = _load_runner()
    denied_home = tmp_path / "fake-real-home"
    attempted_path = denied_home / "credentials.txt"
    environment = {**os.environ, "HOME": str(denied_home)}

    result, scratch, _checkout = _run_pytest_child(
        module,
        tmp_path,
        f'open({str(attempted_path)!r}, encoding="utf-8").read()',
        environ=environment,
    )

    assert not denied_home.exists()
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


def test_child_records_and_blocks_write_to_checkout(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    body = f'open({str(checkout / "mutation.txt")!r}, "w").write("no")'

    result, scratch, actual_checkout = _run_pytest_child(module, tmp_path, body)

    assert actual_checkout == checkout
    assert not (checkout / "mutation.txt").exists()
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


def test_child_allows_checkout_and_runtime_reads_but_only_scratch_writes(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        import json
        import sqlite3
        from pathlib import Path

        assert Path(__file__).read_text(encoding="utf-8")
        assert Path(json.__file__).read_text(encoding="utf-8")
        assert os.stat(__file__)
        assert os.lstat(__file__)
        assert os.access(__file__, os.R_OK)
        if hasattr(os, "statvfs"):
            assert os.statvfs(__file__)
        if hasattr(os, "pathconf"):
            assert os.pathconf(__file__, "PC_NAME_MAX") > 0
        scratch_file = Path(os.environ["TMPDIR"]) / "allowed.txt"
        scratch_file.write_text("allowed", encoding="utf-8")
        scratch_link = Path(os.environ["TMPDIR"]) / "allowed-link"
        scratch_link.symlink_to(scratch_file)
        assert os.readlink(scratch_link) == str(scratch_file)
        if hasattr(os, "mkfifo"):
            scratch_fifo = Path(os.environ["TMPDIR"]) / "allowed-fifo"
            os.mkfifo(scratch_fifo)
            scratch_fifo.unlink()
        database = Path(os.environ["XDG_DATA_HOME"]) / "allowed.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE allowed (value TEXT)")
        connection.close()
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert result.result_path == scratch / "automated-results.json"
    assert (scratch / "tmp/allowed.txt").read_text(encoding="utf-8") == "allowed"
    assert (scratch / "xdg-data/allowed.sqlite3").is_file()
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.parametrize("use_dir_fds", (False, True), ids=("paths", "dir_fds"))
def test_child_hard_link_cannot_alias_checkout_into_scratch(use_dir_fds, tmp_path):
    module = _load_runner()
    if use_dir_fds:
        if not HAS_DIRECTORY_FD_TRAVERSAL:
            pytest.skip("directory-fd link coverage requires POSIX directory fds")
        body = textwrap.dedent(
            """
            flags = os.O_RDONLY | os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            checkout_fd = os.open(os.path.dirname(__file__), flags)
            scratch_fd = os.open(os.environ["TMPDIR"], flags)
            os.link(
                os.path.basename(__file__),
                "checkout-alias.py",
                src_dir_fd=checkout_fd,
                dst_dir_fd=scratch_fd,
            )
            alias_fd = os.open(
                "checkout-alias.py", os.O_WRONLY, dir_fd=scratch_fd
            )
            os.write(alias_fd, b"X")
            """
        ).strip()
    else:
        body = textwrap.dedent(
            """
            alias = os.path.join(os.environ["TMPDIR"], "checkout-alias.py")
            os.link(__file__, alias)
            alias_fd = os.open(alias, os.O_WRONLY)
            os.write(alias_fd, b"X")
            """
        ).strip()

    result, scratch, checkout = _run_pytest_child(module, tmp_path, body)

    assert (checkout / "test_boundary_attempt.py").read_bytes().startswith(b"import os")
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


def test_child_allows_hard_links_wholly_inside_scratch(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        source = os.path.join(os.environ["TMPDIR"], "source.bin")
        destination = os.path.join(os.environ["TMPDIR"], "destination.bin")
        open(source, "wb").write(b"scratch")
        os.link(source, destination)
        assert open(destination, "rb").read() == b"scratch"
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/destination.bin").read_bytes() == b"scratch"
    assert (scratch / "attempts.jsonl").read_bytes() == b""


def test_child_allows_exact_configured_missing_runtime_probe(tmp_path):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_probe = tmp_path / "runtime/python-test.zip"
    checkout.mkdir()
    scratch.mkdir()

    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(str(missing_probe),),
        locale_root=str(tmp_path / "base/share/locale"),
        language="C",
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files

    assert not missing_probe.exists()
    assert module._read_allowed(missing_probe)


def test_child_allows_exact_configured_missing_locale_probe(tmp_path):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_probe = tmp_path / "base/share/locale"
    checkout.mkdir()
    scratch.mkdir()

    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(),
        locale_root=str(missing_probe),
        language="C",
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files

    assert not missing_probe.exists()
    assert module._read_allowed(missing_probe)


@pytest.mark.parametrize("relation", ("descendant", "sibling"))
def test_child_missing_locale_probe_does_not_authorize_related_paths(
    relation, tmp_path
):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_probe = tmp_path / "base/share/locale"
    checkout.mkdir()
    scratch.mkdir()
    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(),
        locale_root=str(missing_probe),
        language="C",
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files
    prohibited = (
        missing_probe / "undeclared"
        if relation == "descendant"
        else missing_probe.with_name("task23019-locale-sibling")
    )

    assert not module._read_allowed(prohibited)


def test_child_locale_environment_cannot_inject_an_exact_read_probe(tmp_path):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_locale = tmp_path / "base/share/locale"
    fake_profile = tmp_path / "never-created-profile"
    target = fake_profile / "LC_MESSAGES/messages.mo"
    locale_escape = os.path.relpath(fake_profile, missing_locale)
    checkout.mkdir()
    scratch.mkdir()

    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(),
        locale_root=str(missing_locale),
        language=locale_escape,
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files

    assert not fake_profile.exists()
    assert not module._read_allowed(target)


@pytest.mark.parametrize("component", (":", ".", ".."), ids=("empty", "dot", "dot_dot"))
def test_child_locale_dot_components_cannot_normalize_outside_exact_shape(
    component, tmp_path
):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_locale = tmp_path / "base/share/locale"
    checkout.mkdir()
    scratch.mkdir()
    target = (
        missing_locale
        / (component if component != ":" else "")
        / "LC_MESSAGES/messages.mo"
    ).resolve()

    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(),
        locale_root=str(missing_locale),
        language=component,
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files

    assert not module._read_allowed(target)


@pytest.mark.parametrize(
    "language", ("C", "C.UTF-8", "en_US.UTF-8", "en_US.UTF-8@modifier")
)
def test_child_accepts_inert_locale_name_shapes(language, tmp_path):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_locale = tmp_path / "base/share/locale"
    checkout.mkdir()
    scratch.mkdir()

    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(),
        locale_root=str(missing_locale),
        language=language,
    )

    assert str(missing_locale.resolve()) in files
    assert str(missing_locale.resolve()) not in roots
    assert all(
        Path(probe) == missing_locale
        or (
            Path(probe).parent.name == "LC_MESSAGES"
            and Path(probe).name == "messages.mo"
            and missing_locale.resolve() in Path(probe).parents
        )
        for probe in files
        if probe.startswith(str(missing_locale.resolve()))
    )


def test_child_preserves_os_capability_sets_for_wrapped_functions(tmp_path):
    module = _load_runner()
    wrapped_names = tuple(
        dict.fromkeys(
            (
                "open",
                "close",
                "stat",
                "lstat",
                "access",
                "readlink",
                "statvfs",
                "pathconf",
                "mkfifo",
                "mknod",
                "listdir",
                "scandir",
                "chdir",
                *DUPLICATION_APIS,
                *(
                    api.removeprefix("os.")
                    for api in PROCESS_APIS
                    if api.startswith("os.")
                ),
            )
        )
    )
    expected = {
        set_name: {
            name: getattr(os, name) in getattr(os, set_name, frozenset())
            for name in wrapped_names
            if hasattr(os, name)
        }
        for set_name in (
            "supports_dir_fd",
            "supports_effective_ids",
            "supports_fd",
            "supports_follow_symlinks",
        )
    }
    assertions = "\n".join(
        f"assert (os.{name} in os.{set_name}) is {supported!r}"
        for set_name, support in expected.items()
        for name, supported in support.items()
    )
    process_names = tuple(
        api.removeprefix("os.") for api in PROCESS_APIS if api.startswith("os.")
    )
    assertions += (
        f"\nassert len({{id(getattr(os, name)) for name in {process_names!r}}}) "
        f"== {len(process_names)}"
    )
    assertions += (
        '\nopen(os.path.join(os.environ["TMPDIR"], "capabilities.txt"), '
        '"w", encoding="utf-8").write("preserved")'
    )

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, assertions, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/capabilities.txt").read_text(encoding="utf-8") == "preserved"


@pytest.mark.parametrize(
    "attempt",
    (
        'open(PARENT / "task23019-parent-file", "rb")',
        "os.stat(PARENT)",
        "os.access(PARENT, os.R_OK)",
        "os.readlink(PARENT)",
        "os.listdir(PARENT)",
        "list(os.scandir(PARENT))",
    ),
)
def test_child_ancestor_traversal_does_not_grant_content_or_metadata_read(
    attempt, tmp_path
):
    module = _load_runner()
    parent = tmp_path
    body = textwrap.dedent(
        f"""
        from pathlib import Path
        PARENT = Path({str(parent)!r})
        reached = os.path.join(os.environ["TMPDIR"], "reached.txt")
        open(reached, "w", encoding="utf-8").write("yes")
        {attempt}
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS /private traversal")
def test_macos_runtime_ancestors_are_not_content_read_files(monkeypatch):
    module = _load_child()
    scratch = "/private/var/folders/aa/bb/T/task23019/scratch"
    monkeypatch.setattr(module.sys, "platform", "darwin")

    _roots, files = module._runtime_authority(
        "/checkout",
        scratch,
        configured_paths=(),
        locale_root="/missing-locale",
        language="C",
    )

    assert "/private" not in files
    assert "/private/var/folders/aa/bb/T" not in files


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS /private traversal")
def test_child_allows_named_nofollow_metadata_for_macos_ancestor(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        os.lstat("/private")
        os.stat("/private", follow_symlinks=False)
        open(os.path.join(os.environ["TMPDIR"], "metadata.txt"), "w").write("ok")
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert (scratch / "tmp/metadata.txt").read_text() == "ok"
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS /private traversal")
@pytest.mark.parametrize(
    "attempt",
    (
        'os.open("/private", os.O_RDONLY)',
        'os.listdir("/private")',
        'glob.glob("/private/*")',
    ),
)
def test_child_macos_metadata_ancestor_denies_content_and_enumeration(
    attempt, tmp_path
):
    module = _load_runner()
    body = textwrap.dedent(
        f"""
        import glob
        reached = os.path.join(os.environ["TMPDIR"], "reached.txt")
        open(reached, "w", encoding="utf-8").write("yes")
        {attempt}
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text() == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="ancestor traversal coverage requires POSIX directory fds",
)
def test_child_traversal_descriptor_does_not_grant_absolute_metadata_read(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        root_fd = os.open(os.sep, flags)
        reached = os.path.join(os.environ["TMPDIR"], "reached.txt")
        open(reached, "w", encoding="utf-8").write("yes")
        os.stat(os.sep, dir_fd=root_fd, follow_symlinks=False)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="ancestor traversal coverage requires POSIX directory fds",
)
def test_child_ancestor_descriptor_cannot_traverse_a_sibling(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        root_fd = os.open(os.sep, flags)
        reached = os.path.join(os.environ["TMPDIR"], "reached.txt")
        open(reached, "w", encoding="utf-8").write("yes")
        os.open("etc", flags, dir_fd=root_fd)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="relative directory-fd coverage requires POSIX directory fds",
)
def test_child_directory_descriptor_rejects_dot_dot_escape(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        scratch_fd = os.open(os.environ["TMPDIR"], flags)
        reached = os.path.join(os.environ["TMPDIR"], "reached.txt")
        open(reached, "w", encoding="utf-8").write("yes")
        os.open("..", flags, dir_fd=scratch_fd)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="directory-fd reuse coverage requires POSIX directory fds",
)
def test_child_closed_directory_descriptor_cannot_reuse_stale_authority(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        stale_fd = os.open(os.environ["TMPDIR"], flags)
        os.close(stale_fd)
        read_fd, write_fd = os.pipe()
        if read_fd != stale_fd:
            os.dup2(read_fd, stale_fd)
            os.close(read_fd)
        reached = os.path.join(os.environ["TMPDIR"], "reached.txt")
        open(reached, "w", encoding="utf-8").write("yes")
        os.stat("anything", dir_fd=stale_fd)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_read_denied")


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="directory-fd replacement coverage requires POSIX directory fds",
)
def test_child_dup2_cannot_retain_replaced_scratch_directory_authority(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        scratch_fd = os.open(os.environ["TMPDIR"], flags)
        checkout_fd = os.open(os.path.dirname(__file__), flags)
        os.dup2(checkout_fd, scratch_fd)
        escaped_fd = os.open(
            "escaped.txt", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=scratch_fd
        )
        os.close(escaped_fd)
        """
    ).strip()

    result, scratch, checkout = _run_pytest_child(module, tmp_path, body)

    assert not (checkout / "escaped.txt").exists()
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.parametrize("api", DUPLICATION_APIS)
@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="directory-fd duplication coverage requires POSIX directory fds",
)
def test_child_allows_duplicated_scratch_directory_authority(api, tmp_path):
    module = _load_runner()
    destination_setup = (
        "destination_fd = os.dup(source_fd); os.close(destination_fd)"
        if api != "dup"
        else "destination_fd = -1"
    )
    duplication = _duplicate_descriptor(api, "source_fd", "destination_fd")
    body = textwrap.dedent(
        f"""
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        source_fd = os.open(os.environ["TMPDIR"], flags)
        {destination_setup}
        {duplication}
        allowed_fd = os.open(
            "duplicated.txt", os.O_WRONLY | os.O_CREAT, 0o600,
            dir_fd=duplicated_fd,
        )
        os.close(allowed_fd)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/duplicated.txt").is_file()
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.parametrize("api", DUPLICATION_APIS)
@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="directory-fd duplication coverage requires POSIX directory fds",
)
def test_child_duplicated_checkout_descriptor_remains_read_only(api, tmp_path):
    module = _load_runner()
    destination_setup = (
        "destination_fd = os.dup(source_fd); os.close(destination_fd)"
        if api != "dup"
        else "destination_fd = -1"
    )
    duplication = _duplicate_descriptor(api, "source_fd", "destination_fd")
    body = textwrap.dedent(
        f"""
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        source_fd = os.open(os.path.dirname(__file__), flags)
        {destination_setup}
        {duplication}
        os.open(
            "escaped.txt", os.O_WRONLY | os.O_CREAT, 0o600,
            dir_fd=duplicated_fd,
        )
        """
    ).strip()

    result, scratch, checkout = _run_pytest_child(module, tmp_path, body)

    assert not (checkout / "escaped.txt").exists()
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.parametrize("api", DUPLICATION_APIS)
@pytest.mark.skipif(
    not hasattr(os, "fchmod"),
    reason="untracked descriptor mutation coverage requires os.fchmod",
)
def test_child_untracked_duplicate_source_grants_no_authority(api, tmp_path):
    module = _load_runner()
    destination_setup = (
        'destination_fd = os.open(os.path.join(os.environ["TMPDIR"], '
        '"destination.bin"), os.O_RDWR | os.O_CREAT, 0o600)'
        if api != "dup"
        else "destination_fd = -1"
    )
    duplication = _duplicate_descriptor(api, "source_fd", "destination_fd")
    body = textwrap.dedent(
        f"""
        source_fd, write_fd = os.pipe()
        {destination_setup}
        {duplication}
        open(os.path.join(os.environ["TMPDIR"], "reached.txt"),
             "w", encoding="utf-8").write("yes")
        os.fchmod(duplicated_fd, 0o600)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="same descriptor coverage requires POSIX directory fds",
)
def test_child_dup2_same_descriptor_preserves_scratch_authority(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        scratch_fd = os.open(os.environ["TMPDIR"], flags)
        assert os.dup2(scratch_fd, scratch_fd) == scratch_fd
        allowed_fd = os.open(
            "same-fd.txt", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=scratch_fd
        )
        os.close(allowed_fd)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/same-fd.txt").is_file()
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.skipif(
    not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="directory-fd reuse coverage requires POSIX directory fds",
)
def test_child_closed_duplicate_descriptor_cannot_reuse_stale_authority(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        scratch_fd = os.open(os.environ["TMPDIR"], flags)
        stale_fd = os.dup(scratch_fd)
        os.close(stale_fd)
        read_fd, write_fd = os.pipe()
        if read_fd != stale_fd:
            os.dup2(read_fd, stale_fd)
            os.close(read_fd)
        open(os.path.join(os.environ["TMPDIR"], "reached.txt"),
             "w", encoding="utf-8").write("yes")
        os.open(
            "escaped.txt", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=stale_fd
        )
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.parametrize("api", FD_MUTATION_APIS)
def test_child_allows_tracked_scratch_file_descriptor_mutation(api, tmp_path):
    module = _load_runner()
    operation = (
        "os.fchmod(file_fd, 0o600)" if api == "fchmod" else "os.ftruncate(file_fd, 0)"
    )
    body = textwrap.dedent(
        f"""
        scratch_file = os.path.join(os.environ["TMPDIR"], "tracked.bin")
        file_fd = os.open(scratch_file, os.O_RDWR | os.O_CREAT, 0o600)
        {operation}
        os.close(file_fd)
        open(os.path.join(os.environ["TMPDIR"], "fd-mutation.txt"),
             "w", encoding="utf-8").write("allowed")
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/fd-mutation.txt").read_text(encoding="utf-8") == "allowed"
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.skipif(
    not hasattr(os, "fchmod"),
    reason="read-only descriptor metadata coverage requires os.fchmod",
)
def test_child_allows_scratch_metadata_mutation_through_read_only_descriptor(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        scratch_file = os.path.join(os.environ["TMPDIR"], "read-only.bin")
        created_fd = os.open(scratch_file, os.O_WRONLY | os.O_CREAT, 0o400)
        os.close(created_fd)
        read_only_fd = os.open(scratch_file, os.O_RDONLY)
        os.fchmod(read_only_fd, 0o600)
        os.close(read_only_fd)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/read-only.bin").stat().st_mode & 0o777 == 0o600
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.skipif(
    not hasattr(os, "ftruncate"),
    reason="read-only descriptor data-mutation coverage requires os.ftruncate",
)
def test_child_blocks_scratch_data_mutation_through_read_only_descriptor(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        scratch_file = os.path.join(os.environ["TMPDIR"], "read-only.bin")
        created_fd = os.open(scratch_file, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(created_fd)
        read_only_fd = os.open(scratch_file, os.O_RDONLY)
        os.ftruncate(read_only_fd, 0)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.skipif(
    not hasattr(os, "fchmod"),
    reason="runtime descriptor metadata coverage requires os.fchmod",
)
def test_child_blocks_runtime_metadata_mutation_through_read_only_descriptor(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        runtime_fd = os.open(os.path.realpath(os.sys.executable), os.O_RDONLY)
        os.fchmod(runtime_fd, 0o600)
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.parametrize("api", FD_MUTATION_APIS)
def test_child_blocks_file_descriptor_mutation_for_checkout_read(api, tmp_path):
    module = _load_runner()
    operation = (
        "os.fchmod(file_fd, 0o600)" if api == "fchmod" else "os.ftruncate(file_fd, 0)"
    )
    body = textwrap.dedent(
        f"""
        file_fd = os.open(__file__, os.O_RDONLY)
        open(os.path.join(os.environ["TMPDIR"], "reached.txt"),
             "w", encoding="utf-8").write("yes")
        {operation}
        """
    ).strip()

    result, scratch, checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    assert (checkout / "test_boundary_attempt.py").stat().st_size > 0
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.parametrize("api", FD_MUTATION_APIS)
def test_child_closed_file_descriptor_cannot_reuse_stale_write_authority(api, tmp_path):
    module = _load_runner()
    operation = (
        "os.fchmod(stale_fd, 0o600)" if api == "fchmod" else "os.ftruncate(stale_fd, 0)"
    )
    body = textwrap.dedent(
        f"""
        scratch_file = os.path.join(os.environ["TMPDIR"], "tracked.bin")
        stale_fd = os.open(scratch_file, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(stale_fd)
        read_fd, write_fd = os.pipe()
        if read_fd != stale_fd:
            os.dup2(read_fd, stale_fd)
            os.close(read_fd)
        open(os.path.join(os.environ["TMPDIR"], "reached.txt"),
             "w", encoding="utf-8").write("yes")
        {operation}
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    assert (scratch / "tmp/reached.txt").read_text(encoding="utf-8") == "yes"
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.parametrize(
    "close_expression",
    (
        'os.fdopen(stale_fd, "r+b").close()',
        'open(stale_fd, "r+b", closefd=True).close()',
    ),
    ids=("fdopen", "file_object"),
)
@pytest.mark.skipif(
    not hasattr(os, "fchmod"),
    reason="object-close mutation coverage requires os.fchmod",
)
def test_child_object_close_cannot_leave_stale_scratch_authority(
    close_expression, tmp_path
):
    module = _load_runner()
    body = textwrap.dedent(
        f"""
        scratch_file = os.path.join(os.environ["TMPDIR"], "tracked.bin")
        stale_fd = os.open(scratch_file, os.O_RDWR | os.O_CREAT, 0o600)
        {close_expression}
        checkout_file = open(__file__, "rb")
        assert checkout_file.fileno() == stale_fd
        checkout_mode = os.stat(__file__).st_mode & 0o777
        os.fchmod(checkout_file.fileno(), checkout_mode ^ 0o100)
        """
    ).strip()

    result, scratch, checkout = _run_pytest_child(module, tmp_path, body)

    assert (checkout / "test_boundary_attempt.py").stat().st_mode & 0o100 == 0
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("fchmod", "ftruncate")),
    reason="descriptor mutation coverage requires fchmod and ftruncate",
)
def test_child_identity_validation_preserves_still_open_scratch_authority(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        scratch_file = os.path.join(os.environ["TMPDIR"], "tracked.bin")
        file_fd = os.open(scratch_file, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(file_fd, 0o600)
        os.ftruncate(file_fd, 0)
        open(os.path.join(os.environ["TMPDIR"], "identity-valid.txt"),
             "w", encoding="utf-8").write("allowed")
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module, tmp_path, body, must_not_continue=False
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/identity-valid.txt").read_text(encoding="utf-8") == "allowed"
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.parametrize("api", FD_MUTATION_APIS)
def test_child_same_inode_read_only_reuse_drops_write_authority(api, tmp_path):
    module = _load_runner()
    operation = (
        "os.fchmod(read_only.fileno(), 0o400)"
        if api == "fchmod"
        else "os.ftruncate(read_only.fileno(), 0)"
    )
    body = textwrap.dedent(
        f"""
        scratch_file = os.path.join(os.environ["TMPDIR"], "same-inode.bin")
        stale_fd = os.open(scratch_file, os.O_RDWR | os.O_CREAT, 0o600)
        os.write(stale_fd, b"payload")
        os.fdopen(stale_fd, "r+b").close()
        read_only = open(scratch_file, "rb")
        assert read_only.fileno() == stale_fd
        {operation}
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(module, tmp_path, body)

    tracked = scratch / "tmp/same-inode.bin"
    assert tracked.read_bytes() == b"payload"
    assert tracked.stat().st_mode & 0o777 == 0o600
    _assert_containment_attempt(module, result, scratch, "filesystem_write_denied")


def test_child_allows_private_binary_read_inside_scratch(tmp_path):
    module = _load_runner()
    body = textwrap.dedent(
        """
        from pathlib import Path
        from private_paths import open_private_binary

        private_file = Path(os.environ["TMPDIR"]) / "private.bin"
        file_fd = os.open(private_file, os.O_WRONLY | os.O_CREAT, 0o600)
        os.write(file_fd, b"private")
        os.close(file_fd)
        with open_private_binary(private_file) as opened:
            assert opened.stream.read() == b"private"
        Path(os.environ["TMPDIR"]).joinpath("private-read.txt").write_text(
            "read", encoding="utf-8"
        )
        """
    ).strip()

    result, scratch, _checkout = _run_pytest_child(
        module,
        tmp_path,
        body,
        must_not_continue=False,
        support_file=REPO_ROOT / "tldw_chatbook/Utils/private_paths.py",
    )

    assert result.returncode == 0
    assert result.error is None
    assert (scratch / "tmp/private-read.txt").read_text(encoding="utf-8") == "read"
    assert (scratch / "attempts.jsonl").read_bytes() == b""


@pytest.mark.parametrize("relation", ("parent", "sibling"))
def test_child_missing_runtime_probe_does_not_authorize_related_paths(
    relation, tmp_path
):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_probe = tmp_path / "runtime/python-test.zip"
    checkout.mkdir()
    scratch.mkdir()
    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(str(missing_probe),),
        locale_root=str(tmp_path / "base/share/locale"),
        language="C",
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files
    prohibited = (
        missing_probe.parent
        if relation == "parent"
        else missing_probe.with_name("task23019-arbitrary-sibling.zip")
    )

    assert not module._read_allowed(prohibited)


def test_child_missing_runtime_probe_remains_read_only(tmp_path):
    module = _load_child()
    checkout = tmp_path / "checkout"
    scratch = tmp_path / "scratch"
    missing_probe = tmp_path / "runtime/python-test.zip"
    checkout.mkdir()
    scratch.mkdir()
    roots, files = module._runtime_authority(
        str(checkout),
        str(scratch),
        configured_paths=(str(missing_probe),),
        locale_root=str(tmp_path / "base/share/locale"),
        language="C",
    )
    module._READ_ROOTS = roots
    module._READ_FILES = files
    module._SCRATCH = str(scratch.resolve())

    assert not missing_probe.exists()
    assert not module._write_allowed(missing_probe)


@pytest.mark.skipif(
    os.name != "posix" or not HAS_DIRECTORY_FD_TRAVERSAL,
    reason="real catalogue boundary smoke currently requires POSIX directory fds",
)
def test_child_settles_one_real_catalogue_selector(tmp_path):
    module = _load_runner()
    selector = module.CATALOGUE["SH-02"].automated_nodes[0]

    result = module.run_closeout_child(
        checkout=REPO_ROOT,
        scratch=tmp_path / "scratch",
        mode="pytest",
        target=Path(f"{REPO_ROOT / selector}"),
    )

    assert result.returncode == 0
    assert result.error is None


def test_child_environment_redirects_every_writable_owner_before_import(tmp_path):
    module = _load_runner()
    scratch = tmp_path / "scratch"
    expected = {
        "HOME": scratch / "home",
        "XDG_CONFIG_HOME": scratch / "xdg-config",
        "XDG_DATA_HOME": scratch / "xdg-data",
        "XDG_CACHE_HOME": scratch / "xdg-cache",
        "XDG_STATE_HOME": scratch / "xdg-state",
        "TLDW_CONFIG_PATH": scratch / "xdg-config/tldw_cli/config.toml",
        "TMPDIR": scratch / "tmp",
        "TEMP": scratch / "tmp",
        "TMP": scratch / "tmp",
    }
    import_checks = "\n".join(
        f"assert os.environ[{name!r}] == {str(path)!r}"
        for name, path in expected.items()
    )
    source = (
        "import os\nimport sys\n"
        + import_checks
        + "\nassert os.environ['TLDW_TEST_MODE'] == '1'\n"
        + "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
        + "assert 'TASK23019_FAKE_API_KEY' not in os.environ\n"
        + "assert 'TASK23019_FAKE_TOKEN' not in os.environ\n"
        + "assert 'TASK23019_FAKE_PASSWORD' not in os.environ\n\n"
        + "assert 'TASK23019_NONSECRET_NOT_NEEDED' not in os.environ\n\n"
        + "assert not any(value.startswith('__editable__') for value in sys.path)\n\n"
        + "def test_attempt():\n    pass\n"
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    test_file = checkout / "test_import_environment.py"
    test_file.write_text(source, encoding="utf-8")
    environment = {
        **os.environ,
        "TASK23019_FAKE_API_KEY": "secret-api-key",
        "TASK23019_FAKE_TOKEN": "secret-token",
        "TASK23019_FAKE_PASSWORD": "secret-password",
        "TASK23019_NONSECRET_NOT_NEEDED": "drop-me",
    }

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="pytest",
        target=test_file,
        environ=environment,
    )

    assert result.returncode == 0
    assert result.error is None
    assert all(
        path.parent.exists() if path.suffix else path.is_dir()
        for path in expected.values()
    )
    assert (scratch / "attempts.jsonl").read_bytes() == b""


def test_environment_rejects_symlink_escape_before_creating_owned_directories(tmp_path):
    module = _load_runner()
    scratch = tmp_path / "scratch"
    outside = tmp_path / "outside"
    scratch.mkdir()
    outside.mkdir()
    (scratch / "home").symlink_to(outside, target_is_directory=True)

    with pytest.raises(module.CloseoutError) as error:
        module.prepare_scratch_environment(scratch, environ=os.environ)

    assert _error_category(error) == "scratch_owner_escape"
    assert not (scratch / "xdg-config").exists()


def test_parent_spawns_child_with_argument_vector_and_explicit_boundary(
    tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "test_empty.py"
    target.write_text("def test_empty(): pass\n", encoding="utf-8")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, module.CONTAINMENT_EXIT_STATUS, "", ""
        )

    monkeypatch.setattr(module, "_run_bounded_process", fake_run)

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=tmp_path / "scratch",
        mode="pytest",
        target=target,
    )

    assert result.error == "containment_failure"
    assert isinstance(observed["command"], list)
    assert observed["command"][0] == sys.executable
    assert observed["command"][1] == str(CHILD_PATH)
    assert observed["kwargs"]["cwd"] == checkout.resolve()
    assert isinstance(observed["kwargs"]["env"], dict)
    assert observed["kwargs"].get("shell", False) is False


def test_parent_child_timeout_is_stable_and_rejects_existing_results(
    tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "test_empty.py"
    target.write_text("def test_empty(): pass\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    observed = {}

    def fake_run(command, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        (scratch / "automated-results.json").write_text(
            '{"test_empty.py::test_empty": "PASS"}\n', encoding="utf-8"
        )
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(module, "_run_bounded_process", fake_run)

    with pytest.raises(module.CloseoutError) as error:
        module.run_closeout_child(
            checkout=checkout,
            scratch=scratch,
            mode="pytest",
            target=target,
        )

    assert _error_category(error) == "child_timeout"
    assert observed["timeout"] == module.CHILD_TIMEOUT_SECONDS == 3600


def test_child_result_names_every_collected_and_settled_node(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    test_file = checkout / "test_results.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest

            def test_passes():
                pass

            def test_fails():
                assert False

            @pytest.fixture
            def broken_setup():
                raise RuntimeError("setup failed")

            def test_setup_failure(broken_setup):
                pass
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="pytest",
        target=test_file,
    )

    assert result.returncode == 1
    assert result.error == "pytest_failed"
    recorded = json.loads((scratch / "automated-results.json").read_text())
    assert recorded == {
        "test_results.py::test_fails": "FAIL",
        "test_results.py::test_passes": "PASS",
        "test_results.py::test_setup_failure": "FAIL",
    }


def test_child_teardown_failure_overrides_a_passing_call(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    test_file = checkout / "test_teardown.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.fixture
            def broken_teardown():
                yield
                raise RuntimeError("teardown failed")

            def test_passes_call(broken_teardown):
                pass
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="pytest",
        target=test_file,
    )

    assert result.returncode == 1
    assert result.error == "pytest_failed"
    recorded = json.loads((scratch / "automated-results.json").read_text())
    assert recorded == {"test_teardown.py::test_passes_call": "FAIL"}


@pytest.mark.parametrize(
    ("selector", "expected_node"),
    (
        ("test_one", "test_selected.py::test_one"),
        ("test_parameterized[named]", "test_selected.py::test_parameterized[named]"),
    ),
)
def test_child_runs_only_the_explicit_pytest_node(selector, expected_node, tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    test_file = checkout / "test_selected.py"
    test_file.write_text(
        textwrap.dedent(
            """
            import pytest

            def test_one():
                pass

            def test_two():
                raise AssertionError("test_two must not run")

            @pytest.mark.parametrize(
                "value",
                (
                    pytest.param("selected", id="named"),
                    pytest.param("not-selected", id="other"),
                ),
            )
            def test_parameterized(value):
                assert value == "selected"
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="pytest",
        target=Path(f"{test_file}::{selector}"),
    )

    assert result.returncode == 0
    assert result.error is None
    recorded = json.loads((scratch / "automated-results.json").read_text())
    assert recorded == {expected_node: "PASS"}


def test_real_hermetic_child_runs_declared_async_curated_selector(tmp_path):
    module = _load_runner()
    selector = (
        "Tests/UI/test_library_adaptive_reader_shell.py::"
        "test_sync_layout_retains_every_mounted_child_identity"
    )
    assert selector in module.CATALOGUE["SH-01"].automated_nodes
    scratch = tmp_path / "async-child"
    prepared = module.prepare_scratch_environment(scratch)

    result = module.run_closeout_child(
        checkout=REPO_ROOT,
        scratch=scratch,
        mode="pytest",
        target=Path(f"{REPO_ROOT / selector}"),
    )

    assert prepared.env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert result.returncode == 0
    assert result.error is None
    assert json.loads(result.result_path.read_text()) == {selector: "PASS"}


def test_real_hermetic_child_still_runs_declared_synchronous_selector(tmp_path):
    module = _load_runner()
    selector = (
        "Tests/Library/test_library_adaptive_reader_state.py::"
        "test_shared_resolution_uses_adaptive_width_classes"
    )
    assert selector in module.CATALOGUE["SH-02"].automated_nodes
    scratch = tmp_path / "sync-child"

    result = module.run_closeout_child(
        checkout=REPO_ROOT,
        scratch=scratch,
        mode="pytest",
        target=Path(f"{REPO_ROOT / selector}"),
    )

    assert result.returncode == 0
    assert result.error is None
    recorded = json.loads(result.result_path.read_text())
    matching = module.matching_node_ids(selector, recorded)
    assert matching
    assert {recorded[node_id] for node_id in matching} == {"PASS"}


def test_real_hermetic_child_allows_legacy_config_private_read(tmp_path):
    module = _load_runner()
    selector = (
        "Tests/UI/test_library_media_reader_shell.py::"
        "test_persisted_shared_library_read_honors_real_legacy_config"
    )
    scratch = tmp_path / "legacy-config-child"

    result = module.run_closeout_child(
        checkout=REPO_ROOT,
        scratch=scratch,
        mode="pytest",
        target=Path(f"{REPO_ROOT / selector}"),
    )

    assert result.returncode == 0
    assert result.error is None
    assert json.loads(result.result_path.read_text()) == {selector: "PASS"}


def test_child_pytest_explicitly_loads_only_required_async_plugin(
    monkeypatch, tmp_path
):
    child = _load_child()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "test_plugin_contract.py"
    target.write_text("def test_plugin_contract(): pass\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    observed = {}

    def fake_main(arguments, *, plugins):
        observed["arguments"] = arguments
        observed["plugins"] = plugins
        observed["autoload"] = os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
        return 0

    monkeypatch.chdir(checkout)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setattr(pytest, "main", fake_main)

    assert child._run_pytest(target, scratch) == 0

    explicitly_loaded = [
        arguments
        for flag, arguments in zip(
            observed["arguments"], observed["arguments"][1:], strict=False
        )
        if flag == "-p" and not arguments.startswith("no:")
    ]
    assert observed["autoload"] == "1"
    assert explicitly_loaded == ["pytest_asyncio.plugin"]
    assert len(observed["plugins"]) == 1
    assert isinstance(observed["plugins"][0], child.ResultRecorder)


def test_missing_explicit_async_plugin_is_a_stable_child_failure(monkeypatch, tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "test_sync.py"
    target.write_text("def test_sync(): pass\n", encoding="utf-8")
    broken_child = tmp_path / "task23019_broken_child.py"
    source = CHILD_PATH.read_text(encoding="utf-8")
    broken_source = source.replace(
        '"pytest_asyncio.plugin"', '"task23019_missing_async_plugin"', 1
    )
    assert broken_source != source
    broken_child.write_text(broken_source, encoding="utf-8")
    monkeypatch.setattr(module, "CHILD_PATH", broken_child)

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=tmp_path / "scratch",
        mode="pytest",
        target=target,
    )

    assert result.returncode != 0
    assert result.error == "child_failed"
    assert result.result_path is None


def test_child_live_mode_imports_only_supplied_scenario_after_boundary(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scenario_file = checkout / "synthetic_scenarios.py"
    scenario_file.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            assert Path(os.environ["HOME"]).name == "home"

            async def contained_case():
                result = Path(os.environ["TMPDIR"]) / "live.txt"
                result.write_text("settled", encoding="utf-8")
                return {"status": "PASS"}

            SCENARIOS = {"contained_case": contained_case}
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=scenario_file,
        scenario="contained_case",
    )

    assert result.returncode == 0
    assert result.error is None
    assert json.loads((scratch / "live-results.json").read_text()) == {
        "contained_case": {"status": "PASS"}
    }
    assert (scratch / "tmp/live.txt").read_text(encoding="utf-8") == "settled"


def test_child_live_mode_keeps_independently_named_results(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scenario_file = checkout / "synthetic_scenarios.py"
    scenario_file.write_text(
        textwrap.dedent(
            """
            async def matrix():
                return {
                    "media-160x50": {"status": "PASS"},
                    "notes-80x24": {"status": "PASS"},
                }

            SCENARIOS = {"common_matrix": matrix}
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=scenario_file,
        scenario="common_matrix",
    )

    assert result.returncode == 0
    assert result.error is None
    assert json.loads((scratch / "live-results.json").read_text()) == {
        "media-160x50": {"status": "PASS"},
        "notes-80x24": {"status": "PASS"},
    }


def test_parent_development_live_case_uses_task_scenario_module(monkeypatch, tmp_path):
    module = _load_runner()
    calls = []

    def fake_child(**kwargs):
        calls.append(kwargs)
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(_passing_live_payload(kwargs["scenario"])),
            encoding="utf-8",
        )
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    result = module.run_development_live_cases(
        checkout=REPO_ROOT,
        scratch=tmp_path,
        live_cases=("common_matrix",),
    )

    assert set(result) == _expected_live_keys("common_matrix")
    assert len(calls) == 1
    assert calls[0]["mode"] == "live"
    assert calls[0]["scenario"] == "common_matrix"
    assert calls[0]["target"] == SCENARIO_PATH


@pytest.mark.parametrize("payload_kind", ("empty", "partial", "extra", "wrong-root"))
def test_parent_rejects_nonexact_common_matrix_result_keys(
    payload_kind, monkeypatch, tmp_path
):
    module = _load_runner()
    payload = _passing_live_payload("common_matrix")
    if payload_kind == "empty":
        payload = {}
    elif payload_kind == "partial":
        payload.pop("media-160x50")
    elif payload_kind == "extra":
        payload["unexpected-999x999"] = {"status": "PASS"}
    else:
        payload = _passing_live_payload("media_capability")

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    with pytest.raises(module.CloseoutError) as error:
        module.run_development_live_cases(
            checkout=REPO_ROOT,
            scratch=tmp_path,
            live_cases=("common_matrix",),
        )

    assert _error_category(error) == "live_result_keys_mismatch"


@pytest.mark.parametrize(
    "root",
    (
        "media_capability",
        "conversations_capability",
        "notes_capability",
        "prompts_capability",
        "skills_capability",
    ),
)
def test_parent_requires_each_capability_root_canonical_result_name(
    root, monkeypatch, tmp_path
):
    module = _load_runner()

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({f"{root}-wrong": {"status": "PASS"}}), encoding="utf-8"
        )
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    with pytest.raises(module.CloseoutError) as error:
        module.run_development_live_cases(
            checkout=REPO_ROOT,
            scratch=tmp_path,
            live_cases=(root,),
        )

    assert _error_category(error) == "live_result_keys_mismatch"


def test_parent_rejects_duplicate_live_roots(monkeypatch, tmp_path):
    module = _load_runner()

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(_passing_live_payload(kwargs["scenario"])), encoding="utf-8"
        )
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    with pytest.raises(module.CloseoutError) as error:
        module.run_development_live_cases(
            checkout=REPO_ROOT,
            scratch=tmp_path,
            live_cases=("media_capability", "media_capability"),
        )

    assert _error_category(error) == "live_result_duplicate"


def test_parent_live_only_requires_and_returns_exact_32_results(monkeypatch, tmp_path):
    module = _load_runner()

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(_passing_live_payload(kwargs["scenario"])), encoding="utf-8"
        )
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    results = module.run_development_live_cases(
        checkout=REPO_ROOT,
        scratch=tmp_path,
        live_cases=module.EXECUTABLE_LIVE_ROOTS,
    )

    expected = set().union(
        *(_expected_live_keys(root) for root in module.EXECUTABLE_LIVE_ROOTS)
    )
    assert len(results) == len(expected) == 32
    assert set(results) == expected


@pytest.mark.asyncio
async def test_common_matrix_records_one_failed_cell_and_continues(
    monkeypatch, tmp_path
):
    module = _load_scenarios()
    monkeypatch.setenv("TASK23019_RAW_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "DESTINATIONS", ("media", "notes"))
    monkeypatch.setattr(module, "SIZES", ((160, 50),))
    calls = []

    async def fake_cell(destination, terminal_size, context):
        calls.append((destination, terminal_size, context.root))
        if destination == "media":
            raise AssertionError("media contract")
        return {"status": "PASS"}

    monkeypatch.setattr(module, "run_common_cell", fake_cell)

    results = await module.run_common_matrix()

    assert list(results) == ["media-160x50", "notes-160x50"]
    assert results["media-160x50"] == {
        "status": "FAIL",
        "error_type": "AssertionError",
        "error": "media contract",
    }
    assert results["notes-160x50"] == {"status": "PASS"}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_media_capability_captures_settled_visible_selected_row(
    monkeypatch, tmp_path
):
    module = _load_scenarios()
    monkeypatch.setenv("TASK23019_RAW_ROOT", str(tmp_path))

    facts = await module.run_media_capability()

    assert facts["status"] == "PASS"
    assert facts["record"]["selected"] == facts["record"]["loaded"]
    assert facts["record"]["pending"] is None
    assert facts["selected_row"]["selected"] is True
    assert facts["selected_row"]["region"]["width"] > 0
    assert facts["selected_row"]["region"]["height"] > 0


def test_scenario_cleanup_failure_is_not_suppressed(tmp_path):
    module = _load_scenarios()
    context = module.ScenarioContext(tmp_path)

    def fail_cleanup():
        raise RuntimeError("cleanup failed")

    context.add_cleanup(fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        context.close()


def test_parent_rejects_a_named_failed_live_result(monkeypatch, tmp_path):
    module = _load_runner()
    payload = _passing_live_payload("common_matrix")
    payload["media-160x50"] = {
        "status": "FAIL",
        "error_type": "AssertionError",
        "error": "media contract",
    }

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    with pytest.raises(module.CloseoutError, match="live_case_failed") as error:
        module.run_development_live_cases(
            checkout=REPO_ROOT,
            scratch=tmp_path,
            live_cases=("common_matrix",),
        )

    assert error.value.category == "live_case_failed"
    assert error.value.details == {
        "failures": [
            {
                "cell": "media-160x50",
                "error_type": "AssertionError",
                "error": "media contract",
            }
        ]
    }


def test_live_cell_failure_details_are_bounded_and_sanitized(monkeypatch, tmp_path):
    module = _load_runner()
    payload = _passing_live_payload("common_matrix")
    secret = "should-not-appear"
    payload["skills-80x24"] = {
        "status": "FAIL",
        "error_type": "AssertionError",
        "error": (
            f"failed at /Users/person/private.py API_TOKEN={secret} "
            + "x" * (module.MAX_DIAGNOSTIC_TEXT * 2)
        ),
    }

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    with pytest.raises(module.CloseoutError) as error:
        module.run_development_live_cases(
            checkout=REPO_ROOT,
            scratch=tmp_path,
            live_cases=("common_matrix",),
        )

    failure = error.value.details["failures"][0]
    assert failure["cell"] == "skills-80x24"
    assert failure["error_type"] == "AssertionError"
    assert len(failure["error"]) <= module.MAX_DIAGNOSTIC_TEXT
    assert "/Users/person" not in failure["error"]
    assert secret not in failure["error"]
    assert "<path>" in failure["error"]
    assert "<redacted>" in failure["error"]


def test_parent_emits_bounded_live_failure_details_as_json(monkeypatch, capsys):
    module = _load_runner()

    def fail_live_cases(**_kwargs):
        raise module.CloseoutError(
            "live_case_failed",
            {
                "failures": [
                    {
                        "cell": "media-160x50",
                        "error_type": "AssertionError",
                        "error": "media contract",
                    }
                ]
            },
        )

    monkeypatch.setattr(module, "run_development_live_cases", fail_live_cases)

    assert module.main(["--development-run", "--live-case", "common_matrix"]) == 2
    emitted = json.loads(capsys.readouterr().err)
    assert emitted == {
        "error": "live_case_failed",
        "details": {
            "failures": [
                {
                    "cell": "media-160x50",
                    "error_type": "AssertionError",
                    "error": "media contract",
                }
            ]
        },
    }


def test_child_nonzero_diagnostics_are_bounded_and_sanitized(tmp_path, monkeypatch):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "synthetic_scenarios.py"
    target.write_text("SCENARIOS = {}\n", encoding="utf-8")
    secret = "should-never-escape"

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="partial output\n",
            stderr=(
                f"RuntimeError: {checkout}/private.py API_TOKEN={secret} "
                + "x" * (module.MAX_DIAGNOSTIC_TEXT * 2)
            ),
        )

    monkeypatch.setattr(module, "_run_bounded_process", fake_run)

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=tmp_path / "scratch",
        mode="live",
        target=target,
        scenario="common_matrix",
        environ={"HOME": str(tmp_path / "home"), "API_TOKEN": secret},
    )

    assert result.error == "child_failed"
    assert result.details["returncode"] == 7
    assert result.details["target"] == "synthetic_scenarios.py"
    assert result.details["stdout"] == "partial output"
    assert len(result.details["stderr"]) <= module.MAX_DIAGNOSTIC_TEXT
    assert str(checkout) not in result.details["stderr"]
    assert secret not in result.details["stderr"]
    assert "<path>" in result.details["stderr"]
    assert "<redacted>" in result.details["stderr"]


def test_child_failure_keeps_sanitized_traceback_tail_within_existing_cap(
    tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "Checkout With Spaces"
    target = checkout / "Tests/Chat/test_failure.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_failure(): pass\n", encoding="utf-8")
    secret = "tail-secret"
    stderr = (
        "config startup "
        + "x" * module.MAX_DIAGNOSTIC_TEXT
        + f"\n  File {checkout}/private.py\nAPI_TOKEN={secret}\n"
        + "RuntimeError: terminal traceback"
    )

    monkeypatch.setattr(
        module,
        "_run_bounded_process",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr=stderr
        ),
    )

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=tmp_path / "scratch",
        mode="pytest",
        target=target,
        environ={"HOME": str(tmp_path / "home"), "API_TOKEN": secret},
    )

    assert result.error == "child_failed"
    assert result.details["target"] == "Tests/Chat/test_failure.py"
    assert "config startup" in result.details["stderr"]
    assert "RuntimeError: terminal traceback" in result.details["stderr"]
    assert "<path>" in result.details["stderr"]
    assert "<redacted>" in result.details["stderr"]
    assert str(checkout) not in result.details["stderr"]
    assert secret not in result.details["stderr"]
    assert len(result.details["stderr"].encode("utf-8")) <= (module.MAX_DIAGNOSTIC_TEXT)


@pytest.mark.parametrize(
    ("name", "returncode", "result_text", "parse_detail"),
    (
        ("missing", 7, None, "missing"),
        ("partial", 7, '{"status":"PASS"}', "error_missing"),
        ("malformed", 7, '{"error":', "malformed_json"),
        ("non-dict", 7, "[]", "not_object"),
        ("non-string-error", 7, '{"error":{}}', "error_not_string"),
        ("zero-malformed", 0, "{", "malformed_json"),
    ),
)
def test_child_result_failures_are_stably_classified(
    name, returncode, result_text, parse_detail, tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "synthetic_scenarios.py"
    target.write_text("SCENARIOS = {}\n", encoding="utf-8")
    scratch = tmp_path / "scratch"

    def fake_run(command, **_kwargs):
        if result_text is not None:
            (scratch / "live-results.json").write_text(result_text, encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=f"{name} stdout",
            stderr=f"{name} stderr",
        )

    monkeypatch.setattr(module, "_run_bounded_process", fake_run)

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=target,
        scenario="common_matrix",
    )

    assert result.error == "child_failed"
    assert result.details == {
        "returncode": returncode,
        "target": "synthetic_scenarios.py",
        "stdout": f"{name} stdout",
        "stderr": f"{name} stderr",
        "result_parse": parse_detail,
    }


def test_unknown_child_error_is_demoted_and_sanitized(tmp_path, monkeypatch):
    module = _load_runner()
    checkout = tmp_path / "Checkout With Spaces"
    checkout.mkdir()
    target = checkout / "synthetic_scenarios.py"
    target.write_text("SCENARIOS = {}\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    secret = "first-secret-line\nsecond-secret-line"
    unknown_error = (
        f"unknown category\n{checkout}/private file.py\nAPI_TOKEN={secret}\n"
        + "x" * (module.MAX_DIAGNOSTIC_TEXT * 2)
    )

    def fake_run(command, **_kwargs):
        (scratch / "live-results.json").write_text(
            json.dumps({"error": unknown_error}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_bounded_process", fake_run)

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=target,
        scenario="common_matrix",
        environ={"HOME": str(tmp_path / "home"), "API_TOKEN": secret},
    )

    assert result.error == "child_failed"
    assert len(result.details["child_error"]) <= module.MAX_DIAGNOSTIC_TEXT
    assert "Checkout With Spaces" not in result.details["child_error"]
    assert "first-secret-line" not in result.details["child_error"]
    assert "second-secret-line" not in result.details["child_error"]
    assert "<path>" in result.details["child_error"]
    assert "<redacted>" in result.details["child_error"]


@pytest.mark.parametrize("returncode", (0, 7))
def test_parent_rejects_oversized_child_result_before_json_parse(
    returncode, tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = checkout / "synthetic_scenarios.py"
    target.write_text("SCENARIOS = {}\n", encoding="utf-8")
    scratch = tmp_path / "scratch"

    def fake_run(command, **_kwargs):
        (scratch / "live-results.json").write_bytes(
            b'{"oversized":"' + b"x" * module.RAW_RESULT_BYTE_LIMIT + b'"}'
        )
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_bounded_process", fake_run)

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=target,
        scenario="common_matrix",
    )

    assert result.error == "child_failed"
    assert result.details["result_parse"] == "result_too_large"


def test_child_result_writer_is_atomic_and_fails_closed_at_raw_ceiling(
    tmp_path, monkeypatch
):
    module = _load_child()
    result_path = tmp_path / "live-results.json"
    replacements = []
    real_replace = os.replace

    def observed_replace(source, destination, *args, **kwargs):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.os, "replace", observed_replace)

    written = module._write_json(
        result_path, {"fact": "x" * module.RAW_RESULT_BYTE_LIMIT}
    )

    assert written is False
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "error": "result_too_large"
    }
    assert result_path.stat().st_size <= module.RAW_RESULT_BYTE_LIMIT
    assert not result_path.with_name(result_path.name + ".tmp").exists()
    assert replacements[-1][1] == result_path


def test_parent_malformed_result_emits_stable_json_not_traceback(monkeypatch, capsys):
    module = _load_runner()

    def fake_child(**kwargs):
        result_path = kwargs["scratch"] / "live-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{", encoding="utf-8")
        return module.ChildRunResult(0, None, result_path)

    monkeypatch.setattr(module, "run_closeout_child", fake_child)

    assert module.main(["--development-run", "--live-case", "common_matrix"]) == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "child_failed",
        "details": {"result_parse": "malformed_json"},
    }


def test_noisy_child_output_is_capped_on_disk_during_execution(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scenario_file = checkout / "synthetic_scenarios.py"
    scenario_file.write_text(
        textwrap.dedent(
            """
            import sys

            async def noisy():
                sys.stdout.write("o" * 262144)
                sys.stdout.flush()
                sys.stderr.write("e" * 262144)
                sys.stderr.flush()
                raise RuntimeError("noisy child failure")

            SCENARIOS = {"noisy": noisy}
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=scenario_file,
        scenario="noisy",
    )

    assert result.error == "child_failed"
    assert result.details["returncode"] != 0
    assert len(result.details["stdout"]) <= module.MAX_DIAGNOSTIC_TEXT
    assert len(result.details["stderr"]) <= module.MAX_DIAGNOSTIC_TEXT
    for name in ("child-stdout.log", "child-stderr.log"):
        output_path = scratch / name
        assert 0 < output_path.stat().st_size <= module.CHILD_OUTPUT_BYTE_LIMIT


def test_noisy_child_timeout_kills_process_and_joins_pipe_readers(tmp_path):
    module = _load_runner()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    command = [
        sys.executable,
        "-c",
        (
            "import sys,time;"
            "sys.stdout.write('o'*262144);sys.stdout.flush();"
            "sys.stderr.write('e'*262144);sys.stderr.flush();"
            "time.sleep(30)"
        ),
    ]

    with pytest.raises(subprocess.TimeoutExpired):
        module._run_bounded_process(
            command,
            cwd=tmp_path,
            env=os.environ,
            stdin=subprocess.DEVNULL,
            scratch=scratch,
            timeout=1,
        )

    for name in ("child-stdout.log", "child-stderr.log"):
        output_path = scratch / name
        assert output_path.stat().st_size == module.CHILD_OUTPUT_BYTE_LIMIT
    assert not any(
        thread.name.startswith("task23019-pipe-")
        for thread in module.threading.enumerate()
    )


@pytest.mark.parametrize(
    ("boundary", "failure_index"),
    (("log_open", 1), ("log_open", 2), ("thread_start", 1), ("thread_start", 2)),
)
def test_post_popen_setup_failure_reaps_child_and_closes_every_endpoint(
    boundary, failure_index, tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "Checkout With Spaces"
    checkout.mkdir()
    target = checkout / "synthetic_scenarios.py"
    target.write_text("SCENARIOS = {}\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    secret = "post-popen-secret"
    diagnostic = f"setup failed at {checkout}/private file.py API_TOKEN={secret}"
    processes = []
    log_handles = []
    log_open_count = 0
    thread_start_count = 0
    real_popen = module.subprocess.Popen
    real_path_open = module.Path.open
    real_thread_start = module.threading.Thread.start

    def observed_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def failing_path_open(path, *args, **kwargs):
        nonlocal log_open_count
        if path.name in {"child-stdout.log", "child-stderr.log"}:
            log_open_count += 1
            if boundary == "log_open" and log_open_count == failure_index:
                raise OSError(diagnostic)
            handle = real_path_open(path, *args, **kwargs)
            log_handles.append(handle)
            return handle
        return real_path_open(path, *args, **kwargs)

    def failing_thread_start(thread, *args, **kwargs):
        nonlocal thread_start_count
        if thread.name.startswith("task23019-pipe-"):
            thread_start_count += 1
            if boundary == "thread_start" and thread_start_count == failure_index:
                raise RuntimeError(diagnostic)
        return real_thread_start(thread, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", observed_popen)
    monkeypatch.setattr(module.Path, "open", failing_path_open)
    monkeypatch.setattr(module.threading.Thread, "start", failing_thread_start)

    errors = []
    finished = module.threading.Event()

    def invoke() -> None:
        try:
            module.run_closeout_child(
                checkout=checkout,
                scratch=scratch,
                mode="live",
                target=target,
                scenario="common_matrix",
                environ={"HOME": str(tmp_path / "home"), "API_TOKEN": secret},
            )
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = module.threading.Thread(target=invoke, name="post-popen-test")
    worker.start()
    completed_without_rescue = finished.wait(1)
    process = processes[0]
    alive_before_rescue = process.poll() is None
    pipes_closed_before_rescue = process.stdout.closed and process.stderr.closed
    logs_closed_before_rescue = all(handle.closed for handle in log_handles)
    if alive_before_rescue:
        process.kill()
        process.wait(timeout=2)
    worker.join(timeout=2)

    assert completed_without_rescue
    assert not alive_before_rescue
    assert pipes_closed_before_rescue
    assert logs_closed_before_rescue
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], module.CloseoutError)
    assert errors[0].category == "child_failed"
    assert len(errors[0].details["process"]) <= module.MAX_DIAGNOSTIC_TEXT
    assert str(checkout) not in errors[0].details["process"]
    assert secret not in errors[0].details["process"]
    assert "<path>" in errors[0].details["process"]
    assert "<redacted>" in errors[0].details["process"]


def test_reader_failure_kills_child_without_waiting_for_global_timeout(
    tmp_path, monkeypatch
):
    module = _load_runner()
    checkout = tmp_path / "Checkout With Spaces"
    checkout.mkdir()
    target = checkout / "synthetic_scenarios.py"
    target.write_text(
        textwrap.dedent(
            """
            import sys

            async def noisy():
                sys.stdout.write("o" * 2097152)
                sys.stdout.flush()
                return {"status": "PASS"}

            SCENARIOS = {"noisy": noisy}
            """
        ),
        encoding="utf-8",
    )
    secret = "reader-secret"
    diagnostic = f"reader failed at {checkout}/private file.py API_TOKEN={secret}"
    processes = []
    real_popen = module.subprocess.Popen

    def observed_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail_write(_handle, _chunk):
        raise OSError(diagnostic)

    monkeypatch.setattr(module.subprocess, "Popen", observed_popen)
    monkeypatch.setattr(module, "_write_capped", fail_write)
    monkeypatch.setattr(module, "CHILD_TIMEOUT_SECONDS", 2)

    started = time.monotonic()
    with pytest.raises(module.CloseoutError) as error:
        module.run_closeout_child(
            checkout=checkout,
            scratch=tmp_path / "scratch",
            mode="live",
            target=target,
            scenario="noisy",
            environ={"HOME": str(tmp_path / "home"), "API_TOKEN": secret},
        )
    elapsed = time.monotonic() - started

    process = processes[0]
    assert elapsed < 1.5
    assert process.poll() is not None
    assert process.stdout.closed and process.stderr.closed
    assert error.value.category == "child_failed"
    assert str(checkout) not in error.value.details["process"]
    assert secret not in error.value.details["process"]
    assert "<path>" in error.value.details["process"]
    assert "<redacted>" in error.value.details["process"]
    assert not any(
        thread.name.startswith("task23019-pipe-")
        for thread in module.threading.enumerate()
    )


def test_diagnostic_redacts_multiline_secret_and_spaced_root_before_normalizing():
    module = _load_runner()
    root = "/Users/Example User/Private Checkout"
    secret = "first-secret-line\nsecond-secret-line"
    raw = f"RuntimeError: {root}/private.py API_TOKEN={secret} trailing"

    diagnostic = module._bounded_diagnostic(raw, secrets=(secret,), roots=(root,))
    emitted = json.dumps({"error": "child_failed", "details": diagnostic})

    assert diagnostic == (
        "RuntimeError: <path>/private.py API_TOKEN=<redacted> trailing"
    )
    assert "Example User" not in emitted
    assert "first-secret-line" not in emitted
    assert "second-secret-line" not in emitted


def test_parent_preserves_child_failure_diagnostics(monkeypatch, tmp_path):
    module = _load_runner()
    diagnostics = {
        "returncode": 7,
        "stderr": "RuntimeError: contained child failure",
    }

    monkeypatch.setattr(
        module,
        "run_closeout_child",
        lambda **_kwargs: module.ChildRunResult(7, "child_failed", None, diagnostics),
    )

    with pytest.raises(module.CloseoutError) as error:
        module.run_development_live_cases(
            checkout=REPO_ROOT,
            scratch=tmp_path,
            live_cases=("common_matrix",),
        )

    assert error.value.category == "child_failed"
    assert error.value.details == diagnostics


def test_capability_scenarios_are_distinct_journeys_not_a_shared_stub():
    source = SCENARIO_PATH.read_text(encoding="utf-8")

    assert "async def _run_capability" not in source
    required_contracts = {
        "run_media_capability": (
            "library-media-content-search",
            "library-media-bulk-delete-cancel",
        ),
        "run_conversations_capability": (
            "_ProgressiveConversationService",
            "_GatedFindRetryConversationService",
            "library-conversation-reader-retry",
            "library-conversation-open-console",
        ),
        "run_notes_capability": (
            "dirty_navigation_veto",
            ".library-notes-row",
            "library-note-conflict-reload",
            "library-note-bulk-status",
        ),
        "run_prompts_capability": (
            "prompt-block-title-delivery",
            "library-prompt-history-collapsible",
        ),
        "run_skills_capability": (
            "library-skill-trust-review",
            "library-skill-delete-cancel",
        ),
    }
    for function_name, markers in required_contracts.items():
        function_source = source.split(f"async def {function_name}", 1)[1].split(
            "\n\nasync def ", 1
        )[0]
        assert all(marker in function_source for marker in markers)


def test_common_matrix_comfort_oracle_requires_strict_expansion():
    source = SCENARIO_PATH.read_text(encoding="utf-8")

    assert "collapsed_items_width > initial_items_width" in source
    assert "collapsed_items_width >= initial_items_width" not in source


def test_child_live_mode_rejects_unknown_scenario_with_stable_error(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scenario_file = checkout / "synthetic_scenarios.py"
    scenario_file.write_text(
        "async def declared(): pass\nSCENARIOS = {'declared': declared}\n",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=scenario_file,
        scenario="not-declared",
    )

    assert result.returncode == 2
    assert result.error == "scenario_not_defined"
    assert json.loads((scratch / "live-results.json").read_text()) == {
        "error": "scenario_not_defined"
    }


def test_child_tripwire_is_installed_before_scenario_module_import(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    scenario_file = checkout / "synthetic_scenarios.py"
    scenario_file.write_text(
        textwrap.dedent(
            """
            import socket
            try:
                socket.getaddrinfo("example.invalid", 80)
            except BaseException:
                pass

            async def should_not_run():
                return {"status": "WRONG"}

            SCENARIOS = {"should_not_run": should_not_run}
            """
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"

    result = module.run_closeout_child(
        checkout=checkout,
        scratch=scratch,
        mode="live",
        target=scenario_file,
        scenario="should_not_run",
    )

    _assert_containment_attempt(module, result, scratch, "network_denied")


def _promotion_subject(module):
    return module.Subject(commit="a" * 40, tree="b" * 40)


def _promotion_sources(module):
    sources = {
        "task23019_closeout.py": b"# parent\n",
        "task23019_closeout_child.py": b"# child\n",
        "task23019_scenarios.py": b"# scenarios\n",
    }
    hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in sources.items()
    }
    return sources, hashes


def _canonical_promotion_results(module):
    automated = {
        selector: "PASS"
        for contract in module.CATALOGUE.values()
        for selector in contract.automated_nodes
    }
    live = {
        name: _rich_live_result(module, name)
        for name in module.EXPECTED_CONCRETE_LIVE_RESULTS
    }
    return automated, live


def _rich_live_result(module, name: str) -> dict[str, object]:
    metadata = module.LIVE_RESULT_METADATA[name]
    destination = metadata["destination"]
    width, height = metadata["terminal_size"]
    result = {
        "status": "PASS",
        "destination": destination,
        "final_destination": metadata["final_destination"],
        "terminal_size": list(metadata["terminal_size"]),
        "contained": True,
        "regions": {
            "library": {"x": 0, "y": 0, "width": 20, "height": height},
            "items": {"x": 20, "y": 0, "width": 20, "height": height},
            "work": {"x": 40, "y": 0, "width": width - 40, "height": height},
        },
        "identities": dict(metadata["identities"]),
        "focus_owner": "work",
        "record": {
            "selected": "record-2",
            "pending": None,
            "loaded": "record-2",
            "mode": "read",
        },
        "preferences": {
            "requested_library_open": True,
            "requested_items_open": True,
            "effective_library_open": True,
            "effective_items_open": True,
        },
        "host_worker_groups": [],
        "visible_controls": ["library-grip", "items-grip"],
        "compositor_text": f"structured compositor for {name}",
        "cleanup_owner_counts": {
            "host_worker_leaks": 0,
            "host_task_leaks": 0,
            "host_thread_worker_leaks": 0,
        },
    }
    if name in module._CAPABILITY_OBSERVATION_KEYS:
        result.update(
            {
                "regions_do_not_intersect": True,
                "selected_row": {},
                "grips": {},
                "primary_actions": {},
                "footer_shortcuts": [],
                "f6_route": [],
                "observations": {
                    key: ([] if key == "catalogue_ids" else "fixture")
                    for key in module._CAPABILITY_OBSERVATION_KEYS[name]
                },
            }
        )
    elif name in module._DURABLE_OBSERVATION_KEYS:
        observations = {key: True for key in module._DURABLE_OBSERVATION_KEYS[name]}
        if name.endswith("-resize-purity"):
            observations["resize_sequence"] = [
                [120, 35],
                [100, 30],
                [80, 24],
                [160, 50],
            ]
        elif name == "preferences-fresh-reload":
            observations["first_host_cleanup_owner_counts"] = {
                "host_workers_before": 0,
                "host_workers_owned": 0,
                "host_worker_leaks": 0,
                "host_task_leaks": 0,
                "host_thread_worker_leaks": 0,
            }
        else:
            selected = {
                "media": "media-2",
                "conversations": "chat-b",
                "notes": "note-2",
                "prompts": 2,
                "skills": "review-skill",
            }
            modes = {
                "media": "info",
                "conversations": "read",
                "notes": "preview",
                "prompts": "info",
                "skills": "edit",
            }
            item_states = {
                destination: destination != "notes"
                for destination in module.DESTINATIONS
            }
            observations["shared_library_open"] = False
            observations["destination_items_open"] = item_states
            observations["revisit_receipts"] = {
                destination: {
                    "preferences": {
                        "requested_library_open": False,
                        "requested_items_open": item_states[destination],
                        "effective_library_open": False,
                        "effective_items_open": item_states[destination],
                    },
                    "record": {
                        "selected": selected[destination],
                        "pending": None,
                        "loaded": selected[destination],
                        "mode": modes[destination],
                    },
                    "focus": {"region": "work", "owner": "work"},
                    "identities": dict(module._PRODUCTION_IDENTITIES[destination]),
                    "draft": {
                        "dirty": destination in {"notes", "prompts"},
                        "retained_without_save": destination in {"notes", "prompts"},
                        "value": "draft"
                        if destination in {"notes", "prompts"}
                        else None,
                    },
                    "worker_fenced": True,
                }
                for destination in module.DESTINATIONS
            }
        result["observations"] = observations
    else:
        result.update(
            {
                "regions_do_not_intersect": True,
                "selected_row": {},
                "grips": {},
                "primary_actions": {},
                "footer_shortcuts": [],
                "f6_route": [],
                "items_comfort_expansion": {},
                "restoration_paths": {},
            }
        )
    return result


def _raw_promotion_artifacts(
    module,
    root: Path,
    *,
    summary: str = "first",
    automated_results=None,
    live_results=None,
):
    if automated_results is None or live_results is None:
        automated_results, live_results = _canonical_promotion_results(module)
    (root / "facts").mkdir(parents=True)
    (root / "captures").mkdir()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "automated_results": len(automated_results),
                "live_results": len(live_results),
                "not_applicable_results": sum(
                    (value if isinstance(value, str) else value["status"])
                    == "NOT_APPLICABLE"
                    for value in (*automated_results.values(), *live_results.values())
                ),
            }
        ),
        encoding="utf-8",
    )
    inventory = [
        *(("automated", name, value) for name, value in automated_results.items()),
        *(("live", name, value) for name, value in live_results.items()),
    ]
    for index, (kind, name, value) in enumerate(sorted(inventory)):
        status = value if isinstance(value, str) else value["status"]
        fact = {"kind": kind, "result_name": name, "status": status}
        if kind == "live" and isinstance(value, dict):
            fact.update(value)
        (root / f"facts/result-{index:03}.json").write_text(
            json.dumps(fact),
            encoding="utf-8",
        )
    expected = {
        **{("automated", name): value for name, value in automated_results.items()},
        **{("live", name): value for name, value in live_results.items()},
    }
    for stem, identity in module.REPRESENTATIVE_CAPTURES.items():
        status = expected[identity]
        if not isinstance(status, str):
            status = status["status"]
        (root / f"captures/{stem}.txt").write_text(
            f"result_name: {identity[1]}\nstatus: {status}\n"
            f"{live_results[identity[1]]['compositor_text']}",
            encoding="utf-8",
        )
        (root / f"captures/{stem}.svg").write_text(
            f'<svg data-result-name="{identity[1]}" data-status="{status}">'
            f"<text>representative frame {summary}</text></svg>",
            encoding="utf-8",
        )
    return module.collect_raw_artifacts(root)


def _rich_raw_promotion_artifacts(module, root: Path, automated, live):
    artifacts = _raw_promotion_artifacts(
        module,
        root,
        automated_results=automated,
        live_results=live,
    )
    for relative, payload in tuple(artifacts.items()):
        if not relative.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if fact["kind"] == "live":
            fact.update(live[fact["result_name"]])
            artifacts[relative] = module._json_bytes(fact)
    for stem, (kind, name) in module.REPRESENTATIVE_CAPTURES.items():
        status = automated[name] if kind == "automated" else live[name]["status"]
        artifacts[f"captures/{stem}.svg"] = (
            f'<svg data-result-name="{name}" data-status="{status}">'
            f"<text>{stem} structured frame</text></svg>"
        ).encode()
    return artifacts


def _valid_contract_results(module):
    catalogue = {"X-01": module.Contract(("tests/test_x.py::test_x",), ("live-x",))}
    automated = {"tests/test_x.py::test_x": "PASS"}
    live = {"live-x": "PASS"}
    return catalogue, automated, live


def _promote_fixture(module, tmp_path: Path, *, summary: str = "first"):
    raw = tmp_path / f"raw-{summary}"
    catalogue = module.CATALOGUE
    automated, live = _canonical_promotion_results(module)
    live = {
        name: {
            **result,
            "compositor_text": f"{result['compositor_text']} {summary}",
        }
        for name, result in live.items()
    }
    artifacts = _raw_promotion_artifacts(
        module,
        raw,
        summary=summary,
        automated_results=automated,
        live_results=live,
    )
    shutil.rmtree(raw)
    sources, source_hashes = _promotion_sources(module)
    return {
        "destination": tmp_path / "evidence/task-23019",
        "raw_root": raw,
        "subject": _promotion_subject(module),
        "subject_sources": sources,
        "subject_hashes": source_hashes,
        "raw_artifacts": artifacts,
        "catalogue": catalogue,
        "automated_results": automated,
        "live_results": live,
        "normalization_roots": {
            "checkout": tmp_path / "checkout",
            "runtime": tmp_path / "runtime",
            "scratch": raw,
        },
    }


def test_status_only_live_facts_are_rejected(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["raw_artifacts"] = dict(kwargs["raw_artifacts"])
    for relative, payload in tuple(kwargs["raw_artifacts"].items()):
        if not relative.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if fact["kind"] == "live":
            kwargs["raw_artifacts"][relative] = module._json_bytes(
                {
                    "kind": "live",
                    "result_name": fact["result_name"],
                    "status": "PASS",
                }
            )

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"


def test_live_oracle_rejects_undeclared_top_level_and_nested_keys(tmp_path):
    module = _load_runner()
    for field, value in (
        ("undeclared", "extension"),
        (
            "preferences",
            {
                "requested_library_open": True,
                "requested_items_open": True,
                "effective_library_open": True,
                "effective_items_open": True,
                "undeclared": True,
            },
        ),
    ):
        kwargs = _promote_fixture(module, tmp_path / field)
        name = "media-160x50"
        kwargs["live_results"][name] = {
            **kwargs["live_results"][name],
            field: value,
        }
        for relative, payload in tuple(kwargs["raw_artifacts"].items()):
            if not relative.startswith("facts/"):
                continue
            fact = json.loads(payload)
            if fact.get("kind") == "live" and fact.get("result_name") == name:
                fact[field] = value
                kwargs["raw_artifacts"][relative] = module._json_bytes(fact)
                break

        with pytest.raises(module.CloseoutError) as error:
            module.promote_evidence(**kwargs)

        assert error.value.category == "evidence_inventory_invalid"


@pytest.mark.parametrize("mutation", ("wrong-size", "fake-identities"))
def test_live_oracle_must_match_canonical_result_metadata(tmp_path, mutation):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    name = "notes_capability"
    field, value = (
        ("terminal_size", [160, 50])
        if mutation == "wrong-size"
        else (
            "identities",
            {"shell": "fake-shell", "items": "fake-items", "work": "fake-work"},
        )
    )
    kwargs["live_results"][name] = {
        **kwargs["live_results"][name],
        field: value,
    }
    for relative, payload in tuple(kwargs["raw_artifacts"].items()):
        if not relative.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if fact.get("kind") == "live" and fact.get("result_name") == name:
            fact[field] = value
            kwargs["raw_artifacts"][relative] = module._json_bytes(fact)
            break

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"


@pytest.mark.parametrize(
    "field,value",
    (
        ("preferences", {}),
        (
            "preferences",
            {
                "requested_library_open": False,
                "requested_items_open": "yes",
                "effective_library_open": False,
                "effective_items_open": True,
            },
        ),
        ("record", {}),
        (
            "record",
            {
                "selected": "record-a",
                "pending": "record-a",
                "loaded": "record-b",
                "mode": "",
            },
        ),
    ),
)
def test_route_receipts_reject_empty_or_malformed_truth(tmp_path, field, value):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    name = "single-app-route-cycle"
    result = kwargs["live_results"][name]
    result["observations"]["revisit_receipts"]["media"][field] = value
    for relative, payload in tuple(kwargs["raw_artifacts"].items()):
        if not relative.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if fact.get("kind") == "live" and fact.get("result_name") == name:
            fact["observations"] = result["observations"]
            kwargs["raw_artifacts"][relative] = module._json_bytes(fact)
            break

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"


@pytest.mark.parametrize("identity", ("", 2))
def test_route_receipts_reject_non_string_notes_identity(tmp_path, identity):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    name = "single-app-route-cycle"
    result = kwargs["live_results"][name]
    record = result["observations"]["revisit_receipts"]["notes"]["record"]
    record["selected"] = identity
    record["loaded"] = identity
    for relative, payload in tuple(kwargs["raw_artifacts"].items()):
        if not relative.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if fact.get("kind") == "live" and fact.get("result_name") == name:
            fact["observations"] = result["observations"]
            kwargs["raw_artifacts"][relative] = module._json_bytes(fact)
            break

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"


@pytest.mark.parametrize(
    "field,value",
    (
        ("destination", "notes"),
        ("identities", {"shell": "same", "items": "same", "work": "same"}),
        (
            "regions",
            {
                "library": {"x": -1, "y": 0, "width": 20, "height": 50},
                "items": {"x": 21, "y": 0, "width": 40, "height": 50},
                "work": {"x": 62, "y": 0, "width": 98, "height": 50},
            },
        ),
        (
            "preferences",
            {
                "requested_library_open": "yes",
                "requested_items_open": True,
                "effective_library_open": True,
                "effective_items_open": True,
            },
        ),
        ("focus_owner", None),
        ("visible_controls", [1]),
        (
            "cleanup_owner_counts",
            {
                "host_worker_leaks": 1,
                "host_task_leaks": 0,
                "host_thread_worker_leaks": 0,
            },
        ),
    ),
)
def test_live_oracle_schema_rejects_invalid_structured_truth(tmp_path, field, value):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    name = "media-160x50"
    kwargs["live_results"][name] = {
        **kwargs["live_results"][name],
        field: value,
    }
    for relative, payload in tuple(kwargs["raw_artifacts"].items()):
        if not relative.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if fact.get("kind") == "live" and fact.get("result_name") == name:
            fact[field] = value
            kwargs["raw_artifacts"][relative] = module._json_bytes(fact)
            break

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"


def test_rich_live_facts_and_paired_text_svg_captures_are_retained(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    automated = kwargs["automated_results"]
    live = {
        name: _rich_live_result(module, name)
        for name in module.EXPECTED_CONCRETE_LIVE_RESULTS
    }
    raw = tmp_path / "rich-raw"
    kwargs["live_results"] = live
    kwargs["raw_artifacts"] = _rich_raw_promotion_artifacts(
        module, raw, automated, live
    )

    module.promote_evidence(**kwargs)

    destination = kwargs["destination"]
    retained_live = []
    for path in (destination / "facts").glob("*.json"):
        fact = json.loads(path.read_text(encoding="utf-8"))
        if fact["kind"] == "live":
            retained_live.append(fact)
    assert len(retained_live) == 32
    required = {
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
    assert all(required.issubset(fact) for fact in retained_live)
    assert {
        path.suffix
        for path in (destination / "captures").iterdir()
        if path.stem == "media-160x50"
    } == {".txt", ".svg"}


@pytest.mark.parametrize("mutation", ("unrelated-body", "non-svg"))
def test_representative_capture_must_match_live_oracle_and_be_svg_root(
    tmp_path, mutation
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    if mutation == "unrelated-body":
        kwargs["raw_artifacts"]["captures/media-160x50.txt"] = (
            b"result_name: media-160x50\nstatus: PASS\nunrelated body"
        )
    else:
        kwargs["raw_artifacts"]["captures/media-160x50.svg"] = (
            b'<html data-result-name="media-160x50" data-status="PASS"></html>'
        )

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"


def test_generated_readme_is_the_canonical_closeout_runbook(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)

    module.promote_evidence(**kwargs)

    readme = (kwargs["destination"] / "README.md").read_text(encoding="utf-8")
    assert (
        f'TASK23019_SUBJECT_REVISION="{kwargs["subject"].commit}" '
        "../../.venv/bin/python "
        "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py "
        f'--subject-revision "{kwargs["subject"].commit}" --promote'
    ) in readme
    assert 'SUBJECT_ROOT="$(git rev-parse --show-toplevel)"' in readme
    assert 'cd "$SUBJECT_ROOT"' in readme
    assert "clean detached subject worktree" in readme
    assert 'TASK23019_SUBJECT_REVISION="$(git rev-parse HEAD)"' not in readme
    assert (
        "../../.venv/bin/python "
        "Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py "
        "--verify-evidence Docs/superpowers/reviews/evidence/task-23019"
    ) in readme
    for literal in (
        kwargs["subject"].commit,
        kwargs["subject"].tree,
        f"{len(kwargs['automated_results'])} automated results",
        "32 live results",
        "a92000229b",
        "471da9f9db",
        "d81e231f26",
        "c9b8a7e002",
        "raw TemporaryDirectory exited before repository promotion",
    ):
        assert literal in readme


def _write_source_bootstrap(destination: Path, sources: dict[str, bytes]) -> None:
    destination.mkdir(parents=True)
    for relative, payload in sources.items():
        (destination / relative).write_bytes(payload)


def _snapshot_directory(directory: Path) -> dict[str, tuple[str, object]]:
    snapshot = {}
    for entry in directory.iterdir():
        if entry.is_symlink():
            snapshot[entry.name] = ("symlink", os.readlink(entry))
        elif entry.is_dir():
            snapshot[entry.name] = ("directory", None)
        else:
            snapshot[entry.name] = ("file", entry.read_bytes())
    return snapshot


def test_success_requires_every_catalogue_id_to_have_fresh_automated_and_live_pass():
    module = _load_runner()
    catalogue, automated, live = _valid_contract_results(module)

    module.validate_complete_results(catalogue, automated, live)
    for missing_automated, missing_live in ((True, False), (False, True)):
        with pytest.raises(module.CloseoutError) as error:
            module.validate_complete_results(
                catalogue,
                {} if missing_automated else automated,
                {} if missing_live else live,
            )
        assert error.value.category in {
            "pytest_selector_not_collected",
            "live_evidence_missing",
        }


def test_not_applicable_requires_catalogue_level_reason():
    module = _load_runner()
    catalogue, _automated, _live = _valid_contract_results(module)

    with pytest.raises(module.CloseoutError) as error:
        module.validate_complete_results(
            catalogue,
            {"tests/test_x.py::test_x": "NOT_APPLICABLE"},
            {"live-x": "NOT_APPLICABLE"},
        )
    assert error.value.category == "not_applicable_undeclared"

    with pytest.raises(module.CloseoutError) as error:
        module.validate_complete_results(
            catalogue, {}, {}, not_applicable={"X-01": " "}
        )
    assert error.value.category == "not_applicable_reason_missing"

    module.validate_complete_results(
        catalogue,
        {"tests/test_x.py::test_x": "NOT_APPLICABLE"},
        {"live-x": "NOT_APPLICABLE"},
        not_applicable={"X-01": "Optional provider is outside this contained run."},
    )


def test_normalization_replaces_scratch_checkout_and_runtime_paths(tmp_path):
    module = _load_runner()
    roots = {
        "checkout": tmp_path / "checkout",
        "runtime": tmp_path / "runtime",
        "scratch": tmp_path / "scratch",
    }
    artifacts = {
        "summary.json": json.dumps(
            {"paths": [str(path / "one") for path in roots.values()]}
        ).encode()
    }

    normalized = module.normalize_artifacts(artifacts, roots=roots)
    text = normalized["summary.json"].decode()

    assert "<checkout>/one" in text
    assert "<runtime>/one" in text
    assert "<scratch>/one" in text
    assert all(str(path) not in text for path in roots.values())


@pytest.mark.parametrize(
    "leak",
    ("OPENAI_API_KEY=sk-live-secret", "/Users/alice/private/database.sqlite"),
)
def test_secret_or_user_path_rejects_the_whole_bundle(tmp_path, leak):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    module.promote_evidence(**kwargs)
    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    kwargs["raw_artifacts"] = {
        **kwargs["raw_artifacts"],
        "facts/leak.json": json.dumps({"value": leak}).encode(),
    }

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category in {"credential_material", "host_path_present"}
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == before
    assert sorted(path.name for path in destination.parent.iterdir()) == ["task-23019"]


def test_only_allowlisted_relative_artifacts_are_promoted(tmp_path):
    module = _load_runner()
    raw = tmp_path / "raw"
    _raw_promotion_artifacts(module, raw)
    (raw / "unexpected.log").write_text("not retained", encoding="utf-8")

    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(raw)

    assert error.value.category == "artifact_path_not_allowed"
    assert not (tmp_path / "evidence").exists()


def test_symlink_or_oversized_capture_is_rejected(tmp_path):
    module = _load_runner()
    raw = tmp_path / "raw-link"
    raw.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (raw / "capture.txt").symlink_to(outside)

    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(raw)
    assert error.value.category == "artifact_symlink"

    for index, (relative, size) in enumerate(
        (
            ("captures/large.txt", 128 * 1024 + 1),
            ("captures/large.svg", 512 * 1024 + 1),
            ("facts/large.json", 256 * 1024 + 1),
        )
    ):
        raw = tmp_path / f"raw-large-{index}"
        target = raw / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * size)
        with pytest.raises(module.CloseoutError) as error:
            module.collect_raw_artifacts(raw)
        assert error.value.category == "artifact_too_large"


def test_raw_root_is_absent_before_repository_promotion(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["raw_root"].mkdir()

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "raw_root_still_exists"
    assert not kwargs["destination"].parent.exists()


def test_existing_unrelated_destination_is_never_replaced(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("unrelated", encoding="utf-8")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_collision"
    assert sentinel.read_text(encoding="utf-8") == "unrelated"


def test_first_promotion_accepts_exact_subject_source_bootstrap(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    _write_source_bootstrap(destination, kwargs["subject_sources"])

    module.promote_evidence(**kwargs)

    assert (destination / "manifest.json").is_file()
    for relative, payload in kwargs["subject_sources"].items():
        assert (destination / relative).read_bytes() == payload
    assert not list(destination.parent.glob(".task-23019.*-*"))


@pytest.mark.parametrize(
    "defect", ("altered", "extra", "extra-directory", "missing", "symlink")
)
def test_source_bootstrap_defects_fail_closed_and_remain_untouched(tmp_path, defect):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    _write_source_bootstrap(destination, kwargs["subject_sources"])
    if defect == "altered":
        (destination / "task23019_closeout.py").write_bytes(b"altered")
    elif defect == "extra":
        (destination / "extra.txt").write_text("extra", encoding="utf-8")
    elif defect == "extra-directory":
        (destination / "facts").mkdir()
    elif defect == "missing":
        (destination / "task23019_scenarios.py").unlink()
    else:
        source = destination / "task23019_scenarios.py"
        source.unlink()
        source.symlink_to(tmp_path / "outside.py")
    before = _snapshot_directory(destination)

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_collision"
    assert _snapshot_directory(destination) == before
    assert not list(destination.parent.glob(".task-23019.*-*"))


@pytest.mark.parametrize(
    "phase",
    tuple(
        sorted(
            (
                "after_stage_validation",
                "after_target_to_backup",
                "after_stage_to_target",
                "before_backup_removal",
            )
        )
    ),
)
def test_source_bootstrap_crash_recovers_on_next_invocation(tmp_path, phase):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    _write_source_bootstrap(destination, kwargs["subject_sources"])

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs, inject_failure=phase)
    assert error.value.category == "injected_promotion_failure"

    module.promote_evidence(**kwargs)
    assert (destination / "manifest.json").is_file()
    assert not list(destination.parent.glob(".task-23019.*-*"))


@pytest.mark.parametrize(
    "phase",
    (
        "after_stage_validation",
        "after_target_to_backup",
        "after_stage_to_target",
        "before_backup_removal",
    ),
)
def test_owned_destination_replace_rolls_back_on_injected_failure(tmp_path, phase):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    second = _promote_fixture(module, tmp_path, summary="second")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second, inject_failure=phase)
    assert error.value.category == "injected_promotion_failure"

    module.promote_evidence(**second)
    destination = second["destination"]
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == "PASS"
    capture = next(iter(module.REPRESENTATIVE_CAPTURES))
    assert b"second" in (destination / f"captures/{capture}.txt").read_bytes()
    assert not list(destination.parent.glob(".task-23019.txn-*"))
    for name, payload in second["subject_sources"].items():
        assert (destination / name).read_bytes() == payload


@pytest.mark.parametrize("role", ("stage", "backup"))
def test_recovery_leaves_unrelated_lookalike_residues_untouched(tmp_path, role):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    residue = parent / f".task-23019.{role}-lookalike"
    residue.mkdir(parents=True)
    marker = residue / "keep.txt"
    marker.write_text("unrelated", encoding="utf-8")

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert marker.read_text(encoding="utf-8") == "unrelated"


def test_promoted_hashes_cover_every_retained_artifact_except_hashes_itself(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)

    module.promote_evidence(**kwargs)

    destination = kwargs["destination"]
    retained = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    hashes = json.loads((destination / "hashes.json").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())
    assert set(hashes["files"]) == retained - {"hashes.json"}
    assert hashes["excluded"] == ["hashes.json"]
    assert manifest["hashes_excluded"] == ["hashes.json"]
    for relative, digest in hashes["files"].items():
        assert (
            hashlib.sha256((destination / relative).read_bytes()).hexdigest() == digest
        )


def test_subject_source_mapping_and_hashes_are_exact(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["subject_sources"] = dict(kwargs["subject_sources"])
    kwargs["subject_sources"].pop("task23019_scenarios.py")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "subject_source_mapping_missing"

    kwargs = _promote_fixture(module, tmp_path / "mismatch")
    kwargs["subject_hashes"] = dict(kwargs["subject_hashes"])
    kwargs["subject_hashes"]["task23019_scenarios.py"] = "0" * 64
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "subject_source_hash_mismatch"


def test_promotion_rejects_a_missing_catalogue_mapping(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["catalogue"] = dict(kwargs["catalogue"])
    kwargs["catalogue"].pop("SK-02")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "catalogue_ids_mismatch"
    assert not kwargs["destination"].parent.exists()


def test_subject_sources_are_read_from_the_recorded_commit_tree():
    module = _load_runner()
    subject = module.Subject(
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        tree=subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )

    sources, hashes = module.load_subject_sources(REPO_ROOT, subject)

    for filename in module.SOURCE_ARTIFACTS:
        expected = subprocess.run(
            [
                "git",
                "show",
                f"{subject.commit}:{module.SOURCE_DIRECTORY}/{filename}",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert sources[filename] == expected
        assert hashlib.sha256(expected).hexdigest() == hashes[filename]


def test_declared_credential_value_rejects_the_whole_bundle(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    opaque_secret = "opaque-value-not-identifiable-by-shape"
    kwargs["raw_artifacts"] = {
        **kwargs["raw_artifacts"],
        "facts/value.json": json.dumps({"value": opaque_secret}).encode(),
    }

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs, credential_values=(opaque_secret,))

    assert error.value.category == "credential_material"
    assert not kwargs["destination"].parent.exists()


def test_environment_credential_value_is_rejected_without_caller_wiring(
    tmp_path, monkeypatch
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    opaque_secret = "opaque-environment-value"
    monkeypatch.setenv("SYNTHETIC_API_KEY", opaque_secret)
    kwargs["raw_artifacts"] = {
        **kwargs["raw_artifacts"],
        "facts/value.json": json.dumps({"value": opaque_secret}).encode(),
    }

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "credential_material"


@pytest.mark.parametrize(
    "payload",
    (
        {"api_key": "not-from-env-secret"},
        {"apiKey": "not-from-env-secret"},
        {"outer": {"token": "not-from-env-secret"}},
        {"outer": [{"authorization": "not-from-env-secret"}]},
        {"AWS_SECRET_ACCESS_KEY": "not-from-env-secret"},
        {"clientSecret": "not-from-env-secret"},
        {"client_secret": "not-from-env-secret"},
        {"client-secret": "not-from-env-secret"},
        {"outer": {"accessToken": "not-from-env-secret"}},
        {"outer": {"access_token": "not-from-env-secret"}},
        {"outer": {"access-token": "not-from-env-secret"}},
        {"outer": [{"privateKey": "not-from-env-secret"}]},
        {"outer": [{"private_key": "not-from-env-secret"}]},
        {"outer": [{"private-key": "not-from-env-secret"}]},
        {"refreshToken": "not-from-env-secret"},
        {"auth-token": "not-from-env-secret"},
        {"bearer_token": "not-from-env-secret"},
        {"passphrase": "not-from-env-secret"},
        {"credentials": "not-from-env-secret"},
    ),
)
def test_json_credential_keys_reject_the_whole_bundle(tmp_path, payload):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["raw_artifacts"] = {
        **kwargs["raw_artifacts"],
        "facts/credential-key.json": json.dumps(payload).encode(),
    }

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "credential_material"
    assert not kwargs["destination"].parent.exists()


def test_json_credential_key_detection_allows_benign_prose_keys(tmp_path):
    module = _load_runner()
    payload = {
        "facts/benign.json": module._json_bytes(
            {
                "kind": "diagnostic",
                "description": "Explains API key handling without retaining a value.",
                "state": {
                    "revision_token": "revision-3",
                    "continuation_token": "continue-4",
                    "worker_token": "worker-5",
                    "page_token": "page-6",
                    "next_token": "next-7",
                },
                "analysis": [
                    {
                        "tokenizer": "standard",
                        "token_count": 42,
                        "secret_recipe": "ordinary prose key",
                        "summary": "The API key and access token were not retained.",
                    }
                ],
            }
        )
    }

    normalized = module.normalize_artifacts(
        payload,
        roots={
            "checkout": tmp_path / "checkout",
            "runtime": tmp_path / "runtime",
            "scratch": tmp_path / "scratch",
        },
    )

    assert normalized == payload


def test_promotion_rejects_a_substituted_canonical_catalogue_selector(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    injected = "Tests/unowned.py::test_injected"
    altered = dict(kwargs["catalogue"])
    original = altered["SH-01"]
    altered["SH-01"] = module.Contract(
        (injected, *original.automated_nodes[1:]), original.live_cases
    )
    kwargs["catalogue"] = altered
    kwargs["automated_results"] = {
        **kwargs["automated_results"],
        injected: "PASS",
    }

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "catalogue_mapping_mismatch"
    assert not kwargs["destination"].parent.exists()


@pytest.mark.parametrize("reason_kind", ("secret", "path"))
def test_not_applicable_reason_is_scanned_before_manifest_generation(
    tmp_path, reason_kind
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    contract = kwargs["catalogue"]["SH-06"]
    for selector in contract.automated_nodes:
        kwargs["automated_results"][selector] = "NOT_APPLICABLE"
    for live_case in contract.live_cases:
        for result_name in module.EXPECTED_LIVE_RESULT_KEYS[live_case]:
            kwargs["live_results"][result_name] = {
                **kwargs["live_results"][result_name],
                "status": "NOT_APPLICABLE",
            }
    for name, payload in tuple(kwargs["raw_artifacts"].items()):
        if not name.startswith("facts/"):
            continue
        fact = json.loads(payload)
        if (
            fact["kind"] == "automated"
            and fact["result_name"] in contract.automated_nodes
        ):
            fact["status"] = "NOT_APPLICABLE"
            kwargs["raw_artifacts"][name] = json.dumps(fact).encode()
        elif (
            fact["kind"] == "live"
            and fact["result_name"] in kwargs["live_results"]
            and kwargs["live_results"][fact["result_name"]]["status"]
            == "NOT_APPLICABLE"
        ):
            fact["status"] = "NOT_APPLICABLE"
            kwargs["raw_artifacts"][name] = json.dumps(fact).encode()
    kwargs["raw_artifacts"]["summary.json"] = module._json_bytes(
        module._canonical_summary(kwargs["automated_results"], kwargs["live_results"])
    )
    secret = "opaque-not-applicable-secret"
    reason = secret if reason_kind == "secret" else str(tmp_path / "checkout/private")

    if reason_kind == "secret":
        with pytest.raises(module.CloseoutError) as error:
            module.promote_evidence(
                **kwargs,
                not_applicable={"SH-06": reason},
                credential_values=(secret,),
            )
        assert error.value.category == "credential_material"
        assert not kwargs["destination"].parent.exists()
        return

    module.promote_evidence(
        **kwargs,
        not_applicable={"SH-06": reason},
        credential_values=(),
    )
    manifest = json.loads(
        (kwargs["destination"] / "manifest.json").read_text(encoding="utf-8")
    )
    retained_reason = manifest["catalogue"]["SH-06"]["not_applicable"]
    assert retained_reason == "<checkout>/private"
    assert str(tmp_path) not in retained_reason


def test_partial_stage_write_never_creates_managed_residue_or_blocks_retry(
    tmp_path, monkeypatch
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    original = module._write_stage_at

    def fail_partial(transaction_fd, artifacts):
        os.mkdir("stage", 0o700, dir_fd=transaction_fd)
        stage_fd, _ = module._open_child_directory(transaction_fd, "stage")
        partial_fd = os.open(
            "partial", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=stage_fd
        )
        os.write(partial_fd, b"partial")
        os.close(partial_fd)
        os.close(stage_fd)
        raise OSError("disk full")

    monkeypatch.setattr(module, "_write_stage_at", fail_partial)
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "promotion_io_failed"
    assert not list(kwargs["destination"].parent.glob(".task-23019.txn-*"))
    assert not list(kwargs["destination"].parent.glob(".task23019-init-*"))
    assert not list(kwargs["destination"].parent.glob(".task23019-init-*"))

    monkeypatch.setattr(module, "_write_stage_at", original)
    module.promote_evidence(**kwargs)
    assert (kwargs["destination"] / "manifest.json").is_file()


def test_stage_fsync_failure_never_creates_managed_residue_or_blocks_retry(
    tmp_path, monkeypatch
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    original = os.fsync
    failed = False

    def fail_once(descriptor):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("fsync failed")
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", fail_once)
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "promotion_io_failed"
    assert not list(kwargs["destination"].parent.glob(".task-23019.txn-*"))

    monkeypatch.setattr(os, "fsync", original)
    module.promote_evidence(**kwargs)
    assert (kwargs["destination"] / "manifest.json").is_file()


def test_partial_retirement_is_resumed_without_permanent_collision(
    tmp_path, monkeypatch
):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    second = _promote_fixture(module, tmp_path, summary="second")
    original = module._delete_directory_contents
    failed = False

    def fail_partial(descriptor, *, marker_last):
        nonlocal failed
        if not failed and marker_last == module._TRANSACTION_MARKER:
            failed = True
            victim = next(
                name
                for name in os.listdir(descriptor)
                if name != module._TRANSACTION_MARKER
            )
            victim_fd, _ = module._open_child_directory(descriptor, victim)
            original(victim_fd, marker_last=None)
            os.close(victim_fd)
            os.rmdir(victim, dir_fd=descriptor)
            raise OSError("partial retirement")
        return original(descriptor, marker_last=marker_last)

    monkeypatch.setattr(module, "_delete_directory_contents", fail_partial)
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)
    assert error.value.category == "promotion_cleanup_failed"

    monkeypatch.setattr(module, "_delete_directory_contents", original)
    module.promote_evidence(**second)
    capture = next(iter(module.REPRESENTATIVE_CAPTURES))
    assert b"second" in (second["destination"] / f"captures/{capture}.txt").read_bytes()


@pytest.mark.parametrize("failure_kind", ("rename", "marker"))
def test_retirement_transition_failure_is_resumed_on_next_invocation(
    tmp_path, monkeypatch, failure_kind
):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    second = _promote_fixture(module, tmp_path, summary="second")

    if failure_kind == "rename":
        original = module._rename_entry
        failed = False

        def fail_retirement_rename(parent_fd, source, destination, expected):
            nonlocal failed
            if not failed and destination.startswith(".task-23019.txn-retired-"):
                failed = True
                raise OSError("retirement rename failed")
            return original(parent_fd, source, destination, expected)

        monkeypatch.setattr(module, "_rename_entry", fail_retirement_rename)
    else:
        original = module._write_transaction_marker
        failed = False

        def fail_retirement_marker(transaction_fd, subject, role, nonce):
            nonlocal failed
            if not failed and role == "retirement":
                failed = True
                raise OSError("retirement marker write failed")
            return original(transaction_fd, subject, role, nonce)

        monkeypatch.setattr(module, "_write_transaction_marker", fail_retirement_marker)

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)
    assert error.value.category == "promotion_io_failed"

    if failure_kind == "rename":
        monkeypatch.setattr(module, "_rename_entry", original)
    else:
        monkeypatch.setattr(module, "_write_transaction_marker", original)
    module.promote_evidence(**second)
    capture = next(iter(module.REPRESENTATIVE_CAPTURES))
    assert b"second" in (second["destination"] / f"captures/{capture}.txt").read_bytes()
    assert not list(second["destination"].parent.glob(".task-23019.*-*"))


def test_backup_residue_identity_swap_before_retirement_is_rejected_and_untouched(
    tmp_path, monkeypatch
):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    second = _promote_fixture(module, tmp_path, summary="second")
    parent_fd, _ = module._open_directory_nofollow(second["destination"].parent)
    transaction_name = module._create_transaction(parent_fd, second["subject"])
    _role, transaction_fd = module._read_transaction_marker(
        parent_fd, transaction_name, second["subject"]
    )
    destination_receipt = module._validate_recovery_candidate_at(
        parent_fd,
        second["destination"].name,
        subject=second["subject"],
        subject_sources=second["subject_sources"],
        allow_bootstrap=False,
    )
    module._rename_between(
        parent_fd,
        second["destination"].name,
        transaction_fd,
        "backup",
        destination_receipt,
    )
    os.close(transaction_fd)
    os.close(parent_fd)

    transaction = second["destination"].parent / transaction_name
    original = module._validate_recovery_candidate_at
    replacement = second["destination"].parent / "replacement-backup"
    swapped = False

    def swap_after_validation(parent_fd, name, **validation):
        nonlocal swapped
        result = original(parent_fd, name, **validation)
        if not swapped and name == "backup":
            swapped = True
            backup = transaction / "backup"
            backup.rename(transaction / "validated-backup")
            backup.mkdir()
            (backup / "keep.txt").write_text("replacement", encoding="utf-8")
            replacement.write_text("unrelated", encoding="utf-8")
        return result

    monkeypatch.setattr(
        module, "_validate_recovery_candidate_at", swap_after_validation
    )
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)

    assert error.value.category == "promotion_collision"
    backup = transaction / "backup"
    assert (backup / "keep.txt").read_text(encoding="utf-8") == "replacement"
    assert replacement.read_text(encoding="utf-8") == "unrelated"


def test_retirement_residue_identity_swap_before_delete_is_rejected_and_untouched(
    tmp_path, monkeypatch
):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    second = _promote_fixture(module, tmp_path, summary="second")
    original_delete = module._delete_transaction
    monkeypatch.setattr(
        module,
        "_delete_transaction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop retirement")),
    )
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)
    assert error.value.category == "promotion_cleanup_failed"
    monkeypatch.setattr(module, "_delete_transaction", original_delete)
    retired = next(second["destination"].parent.glob(".task-23019.txn-retired-*"))

    original_remove = module._delete_directory_contents
    swapped = False

    def swap_before_delete(descriptor, *, marker_last):
        nonlocal swapped
        if not swapped and marker_last == module._TRANSACTION_MARKER:
            swapped = True
            retired.rename(retired.parent / "validated-retirement")
            retired.mkdir()
            (retired / "keep.txt").write_text("replacement", encoding="utf-8")
        return original_remove(descriptor, marker_last=marker_last)

    monkeypatch.setattr(module, "_delete_directory_contents", swap_before_delete)
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)

    assert error.value.category == "promotion_collision"
    assert (retired / "keep.txt").read_text(encoding="utf-8") == "replacement"


def test_intermediate_artifact_directory_symlink_swap_fails_closed(
    tmp_path, monkeypatch
):
    module = _load_runner()
    raw = tmp_path / "raw-swap"
    _raw_promotion_artifacts(module, raw)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "matrix.json").write_text('{"status":"PASS"}', encoding="utf-8")
    original = module._open_child_directory
    swapped = False

    def swap_then_open(parent_fd, name):
        nonlocal swapped
        if not swapped and name == "facts":
            swapped = True
            facts = raw / "facts"
            facts.rename(raw / "facts-original")
            facts.symlink_to(outside, target_is_directory=True)
        return original(parent_fd, name)

    monkeypatch.setattr(module, "_open_child_directory", swap_then_open)
    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(raw)
    assert error.value.category in {"artifact_symlink", "artifact_changed"}
    assert (outside / "matrix.json").read_text(encoding="utf-8") == '{"status":"PASS"}'


def test_root_component_symlink_swap_is_blocked_by_descriptor_walk(
    tmp_path, monkeypatch
):
    module = _load_runner()
    trusted = tmp_path / "trusted"
    raw = trusted / "raw"
    _raw_promotion_artifacts(module, raw)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = module._open_child_directory
    swapped = False

    def swap_component(parent_fd, name):
        nonlocal swapped
        if not swapped and name == trusted.name:
            swapped = True
            trusted.rename(tmp_path / "trusted-original")
            trusted.symlink_to(outside, target_is_directory=True)
        return original(parent_fd, name)

    monkeypatch.setattr(module, "_open_child_directory", swap_component)
    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(raw)

    assert swapped
    assert error.value.category in {"artifact_symlink", "artifact_root_missing"}


def test_destination_identity_swap_before_rename_is_rejected_and_untouched(
    tmp_path, monkeypatch
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    _write_source_bootstrap(destination, kwargs["subject_sources"])
    original = module._validate_recovery_candidate_at
    swapped = False

    def swap_after_validation(parent_fd, name, **validation):
        nonlocal swapped
        result = original(parent_fd, name, **validation)
        if name == destination.name and not swapped:
            swapped = True
            destination.rename(destination.parent / "validated-original")
            destination.mkdir()
            (destination / "keep.txt").write_text("replacement", encoding="utf-8")
        return result

    monkeypatch.setattr(
        module, "_validate_recovery_candidate_at", swap_after_validation
    )
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "promotion_collision"
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "replacement"


def test_summary_only_bundle_and_mismatched_fact_inventory_are_rejected(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["raw_artifacts"] = {"summary.json": kwargs["raw_artifacts"]["summary.json"]}
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "evidence_inventory_invalid"


@pytest.mark.parametrize("defect", ("missing", "extra"))
def test_missing_or_extra_fact_inventory_is_rejected(tmp_path, defect):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    facts = [name for name in kwargs["raw_artifacts"] if name.startswith("facts/")]
    kwargs["raw_artifacts"] = dict(kwargs["raw_artifacts"])
    if defect == "missing":
        del kwargs["raw_artifacts"][facts[0]]
    else:
        fact = json.loads(kwargs["raw_artifacts"][facts[0]])
        kwargs["raw_artifacts"]["facts/extra.json"] = json.dumps(fact).encode()

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "evidence_inventory_invalid"

    kwargs = _promote_fixture(module, tmp_path / "mismatch")
    fact = next(name for name in kwargs["raw_artifacts"] if name.startswith("facts/"))
    kwargs["raw_artifacts"] = dict(kwargs["raw_artifacts"])
    kwargs["raw_artifacts"][fact] = json.dumps(
        {"kind": "live", "result_name": "not-declared", "status": "PASS"}
    ).encode()
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "evidence_inventory_invalid"


def test_path_normalization_is_component_aware_and_rejects_host_uris(tmp_path):
    module = _load_runner()
    checkout = tmp_path / "checkout with spaces"
    roots = {
        "checkout": checkout,
        "runtime": tmp_path / "runtime",
        "scratch": tmp_path / "scratch",
    }
    normalized = module.normalize_artifacts(
        {"summary.json": json.dumps({"path": str(checkout / "child file")}).encode()},
        roots=roots,
    )
    assert "<checkout>/child file" in normalized["summary.json"].decode()

    for leaked in (
        str(checkout) + "-sibling/private",
        f"file://{checkout}/private",
        "vscode-file://vscode-app/Users/alice/private.txt",
        "file:///Users/alice/My%20Private/file.txt",
    ):
        with pytest.raises(module.CloseoutError) as error:
            module.normalize_artifacts(
                {"summary.json": json.dumps({"path": leaked}).encode()}, roots=roots
            )
        assert error.value.category == "host_path_present"


def test_artifact_count_limits_reject_many_zero_byte_files(tmp_path):
    module = _load_runner()
    raw = tmp_path / "many"
    facts = raw / "facts"
    facts.mkdir(parents=True)
    for index in range(module.MAX_FACT_ARTIFACTS + 1):
        (facts / f"{index:04}.json").write_bytes(b"")

    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(raw)

    assert error.value.category == "artifact_count_exceeded"


def test_oversized_git_object_is_rejected_before_show_materialization(monkeypatch):
    module = _load_runner()
    subject = module.Subject(commit="a" * 40, tree="b" * 40)
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(tuple(arguments))
        if arguments[1] == "rev-parse":
            return subprocess.CompletedProcess(arguments, 0, subject.tree + "\n", "")
        if arguments[1:3] == ["cat-file", "-s"]:
            return subprocess.CompletedProcess(
                arguments, 0, str(module.SOURCE_ARTIFACT_BYTE_LIMIT + 1) + "\n", ""
            )
        pytest.fail("oversized Git object was materialized")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(module.CloseoutError) as error:
        module.load_subject_sources(Path("/unused"), subject)

    assert error.value.category == "subject_source_too_large"
    assert not any(call[1] == "show" for call in calls)


def test_total_promoted_bundle_limit_is_enforced(tmp_path):
    module = _load_runner()
    raw = tmp_path / "raw-total"
    facts = raw / "facts"
    facts.mkdir(parents=True)
    payload = json.dumps({"value": "x" * (256 * 1024 - 32)}).encode()
    for index in range(65):
        (facts / f"{index:02}.json").write_bytes(payload)

    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(raw)

    assert error.value.category == "bundle_too_large"


def test_promotion_io_failure_is_bounded_and_does_not_expose_host_paths(
    tmp_path, monkeypatch
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    monkeypatch.setattr(
        module,
        "_write_stage_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("/Users/alice/private/evidence")
        ),
    )

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_io_failed"
    assert error.value.details == {}


def test_hard_crash_during_stage_build_is_recovered_from_one_marked_transaction(
    tmp_path,
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs, inject_failure="during_stage_build")
    assert error.value.category == "injected_promotion_failure"
    transactions = [
        path
        for path in kwargs["destination"].parent.glob(".task-23019.txn-*")
        if path.is_dir()
    ]
    assert len(transactions) == 1
    marker = json.loads((transactions[0] / "transaction.json").read_text())
    assert marker["task"] == module.TASK_ID
    assert marker["subject_commit"] == kwargs["subject"].commit

    module.promote_evidence(**kwargs)
    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not list(kwargs["destination"].parent.glob(".task-23019.txn-*"))


def test_partial_atomic_marker_temp_is_recovered_only_inside_owned_transaction(
    tmp_path,
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    with pytest.raises(module.CloseoutError):
        module.promote_evidence(**kwargs, inject_failure="during_stage_build")
    transaction = next(
        path
        for path in kwargs["destination"].parent.glob(".task-23019.txn-*")
        if path.is_dir()
    )
    (transaction / "transaction.json.tmp").write_bytes(b'{"partial":')

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not transaction.exists()


def test_transaction_cleanup_failure_is_surfaced_and_resumed(tmp_path, monkeypatch):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    original = module._delete_transaction
    failed = False

    def fail_once(*args, **validation):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("cleanup failed")
        return original(*args, **validation)

    monkeypatch.setattr(module, "_delete_transaction", fail_once)
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)
    assert error.value.category == "promotion_cleanup_failed"
    assert list(kwargs["destination"].parent.glob(".task-23019.txn-*"))

    monkeypatch.setattr(module, "_delete_transaction", original)
    module.promote_evidence(**kwargs)
    assert not list(kwargs["destination"].parent.glob(".task-23019.txn-*"))


def test_unmarked_transaction_lookalike_is_left_untouched(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    lookalike = kwargs["destination"].parent / (".task-23019.txn-" + "0" * 32)
    lookalike.mkdir(parents=True)
    sentinel = lookalike / "keep.txt"
    sentinel.write_text("unrelated", encoding="utf-8")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_collision"
    assert sentinel.read_text(encoding="utf-8") == "unrelated"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS canonical /var alias")
def test_canonical_root_accepts_macos_var_alias(tmp_path):
    module = _load_runner()
    raw = tmp_path / "raw"
    _raw_promotion_artifacts(module, raw)
    alias = Path(str(raw).replace("/private/var/", "/var/", 1))

    artifacts = module.collect_raw_artifacts(alias)

    assert "summary.json" in artifacts


def test_windows_promotion_fails_stably_before_repository_write(tmp_path, monkeypatch):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    monkeypatch.setattr(module.os, "name", "nt")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_unsupported_platform"
    assert not kwargs["destination"].parent.exists()


@pytest.mark.parametrize("mutation", ("added-file", "modified-bytes"))
def test_content_change_between_validation_and_rename_fails_closed(
    tmp_path, monkeypatch, mutation
):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    second = _promote_fixture(module, tmp_path, summary="second")
    destination = second["destination"]
    original = module._validate_recovery_candidate_at
    target_validations = 0

    def mutate_after_validation(parent_fd, name, **validation):
        nonlocal target_validations
        receipt = original(parent_fd, name, **validation)
        if name == destination.name:
            target_validations += 1
        if name == destination.name and target_validations == 1:
            if mutation == "added-file":
                (destination / "added.txt").write_text("added", encoding="utf-8")
            else:
                (destination / "summary.json").write_text(
                    '{"status":"PASS","tampered":true}', encoding="utf-8"
                )
        return receipt

    monkeypatch.setattr(
        module, "_validate_recovery_candidate_at", mutate_after_validation
    )
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)

    assert error.value.category == "promotion_collision"
    if mutation == "added-file":
        assert (destination / "added.txt").read_text(encoding="utf-8") == "added"
    else:
        assert "tampered" in (destination / "summary.json").read_text()


@pytest.mark.parametrize("phase", ("after_target_to_backup", "after_stage_to_target"))
def test_post_quarantine_failure_restores_original_before_return(tmp_path, phase):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    destination = first["destination"]
    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    second = _promote_fixture(module, tmp_path, summary="second")

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second, inject_failure=phase)

    assert error.value.category == "injected_promotion_failure"
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == before


def test_unknown_automated_result_is_rejected_even_with_a_matching_fact(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    injected = "Tests/unowned.py::test_injected"
    kwargs["automated_results"] = {**kwargs["automated_results"], injected: "PASS"}
    kwargs["raw_artifacts"] = dict(kwargs["raw_artifacts"])
    kwargs["raw_artifacts"]["facts/injected.json"] = json.dumps(
        {"kind": "automated", "result_name": injected, "status": "PASS"}
    ).encode()

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "automated_result_unknown"


def test_supplied_summary_must_match_canonical_validated_counts(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    kwargs["raw_artifacts"] = dict(kwargs["raw_artifacts"])
    kwargs["raw_artifacts"]["summary.json"] = json.dumps(
        {"status": "PASS", "automated_results": 1, "live_results": 1}
    ).encode()

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "summary_mismatch"


def test_http_url_with_home_path_is_allowed_but_file_uri_is_rejected(tmp_path):
    module = _load_runner()
    roots = {
        "checkout": tmp_path / "checkout",
        "runtime": tmp_path / "runtime",
        "scratch": tmp_path / "scratch",
    }
    normalized = module.normalize_artifacts(
        {
            "summary.json": json.dumps(
                {"url": "https://example.test/home/library"}
            ).encode()
        },
        roots=roots,
    )
    assert b"https://example.test/home/library" in normalized["summary.json"]

    for uri in ("file:///opt/private.txt", "file://host/share/private.txt"):
        with pytest.raises(module.CloseoutError) as error:
            module.normalize_artifacts(
                {"summary.json": json.dumps({"url": uri}).encode()}, roots=roots
            )
        assert error.value.category == "host_path_present"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS canonical /var alias")
def test_canonical_var_alias_rejects_a_nested_symlink(tmp_path):
    module = _load_runner()
    actual = tmp_path / "actual"
    raw = actual / "raw"
    _raw_promotion_artifacts(module, raw)
    nested = tmp_path / "nested-link"
    nested.symlink_to(actual, target_is_directory=True)
    alias = Path(str(nested / "raw").replace("/private/var/", "/var/", 1))

    with pytest.raises(module.CloseoutError) as error:
        module.collect_raw_artifacts(alias)

    assert error.value.category == "artifact_symlink"


def test_uri_inspection_preserves_safe_encoded_bytes_and_rejects_double_encoding(
    tmp_path,
):
    module = _load_runner()
    roots = {
        "checkout": tmp_path / "checkout",
        "runtime": tmp_path / "runtime",
        "scratch": tmp_path / "scratch",
    }
    safe = b'{"label":"representative%20capture"}'

    assert (
        module.normalize_artifacts({"summary.json": safe}, roots=roots)["summary.json"]
        == safe
    )

    for encoded_local_uri in (
        "file%253A%252F%252F%252FUsers%252Falice%252Fsecret.txt",
        "%252Fhome%252Falice%252Fsecret.txt",
    ):
        with pytest.raises(module.CloseoutError) as error:
            module.normalize_artifacts(
                {"summary.json": json.dumps({"path": encoded_local_uri}).encode()},
                roots=roots,
            )
        assert error.value.category == "host_path_present"


@pytest.mark.parametrize(
    "credential_key",
    ("primary_api_key", "myAccessToken", "vendor-client-secret", "backupPrivateKey"),
)
def test_compound_json_credential_suffixes_are_rejected(tmp_path, credential_key):
    module = _load_runner()
    roots = {
        "checkout": tmp_path / "checkout",
        "runtime": tmp_path / "runtime",
        "scratch": tmp_path / "scratch",
    }

    with pytest.raises(module.CloseoutError) as error:
        module.normalize_artifacts(
            {"summary.json": json.dumps({credential_key: "opaque"}).encode()},
            roots=roots,
        )

    assert error.value.category == "credential_material"


def test_deep_json_is_rejected_with_a_stable_closeout_error(tmp_path):
    module = _load_runner()
    roots = {
        "checkout": tmp_path / "checkout",
        "runtime": tmp_path / "runtime",
        "scratch": tmp_path / "scratch",
    }
    deeply_nested = b"[" * 1500 + b"{}" + b"]" * 1500

    with pytest.raises(module.CloseoutError) as error:
        module.normalize_artifacts({"summary.json": deeply_nested}, roots=roots)

    assert error.value.category == "artifact_json_invalid"


def test_destination_appearing_at_stage_install_is_never_replaced(
    tmp_path, monkeypatch
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    destination = kwargs["destination"]
    original = module._rename_between
    raced = False

    def race_destination(source_fd, source, destination_fd, destination_name, expected):
        nonlocal raced
        if source == "stage" and destination_name == destination.name and not raced:
            raced = True
            os.mkdir(destination_name, 0o700, dir_fd=destination_fd)
        return original(source_fd, source, destination_fd, destination_name, expected)

    monkeypatch.setattr(module, "_rename_between", race_destination)
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert raced
    assert error.value.category == "promotion_collision"
    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_self_contained_transaction_marker_cannot_authorize_deletion(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "7" * 32
    spoof = parent / f".task-23019.txn-retired-{nonce}"
    spoof.mkdir()
    marker = module._transaction_payload(kwargs["subject"], "retirement")
    (spoof / module._TRANSACTION_MARKER).write_bytes(marker)

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_collision"
    assert spoof.is_dir()
    assert (spoof / module._TRANSACTION_MARKER).read_bytes() == marker


def test_existing_bundle_requires_full_canonical_manifest_semantics(tmp_path):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    destination = first["destination"]
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["catalogue"] = {"SPOOF-01": {"automated_nodes": [], "live_cases": []}}
    manifest_path.write_bytes(module._json_bytes(manifest))
    hashes_path = destination / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["files"]["manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    hashes_path.write_bytes(module._json_bytes(hashes))
    before = _snapshot_directory(destination)

    second = _promote_fixture(module, tmp_path, summary="second")
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)

    assert error.value.category == "promotion_collision"
    assert _snapshot_directory(destination) == before


def test_hard_crash_with_installed_target_and_backup_keeps_valid_target(tmp_path):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    backup_source = tmp_path / "backup-source"
    shutil.copytree(first["destination"], backup_source)

    second = _promote_fixture(module, tmp_path, summary="second")
    module.promote_evidence(**second)
    target_before = {
        path.relative_to(second["destination"]).as_posix(): path.read_bytes()
        for path in second["destination"].rglob("*")
        if path.is_file()
    }
    parent_fd, _ = module._open_directory_nofollow(second["destination"].parent)
    try:
        transaction_name = module._create_transaction(parent_fd, second["subject"])
    finally:
        os.close(parent_fd)
    transaction = second["destination"].parent / transaction_name
    backup_source.rename(transaction / "backup")
    assert transaction.is_dir() and (transaction / "backup").is_dir()

    module._recover_interrupted_promotion(
        second["destination"],
        subject=second["subject"],
        subject_sources=second["subject_sources"],
    )

    assert {
        path.relative_to(second["destination"]).as_posix(): path.read_bytes()
        for path in second["destination"].rglob("*")
        if path.is_file()
    } == target_before
    assert not transaction.exists()


@pytest.mark.parametrize("marker_bytes", (b'{"partial":', None))
def test_hard_crash_pending_marker_state_is_recovered_without_collision(
    tmp_path, marker_bytes
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "8" * 32
    pending = parent / f".task-23019.txn-pending-{nonce}"
    pending.mkdir()
    authority = parent / module._authority_name(nonce)
    authority.write_bytes(
        module._authority_payload(
            kwargs["subject"], nonce, module._receipt(pending.lstat())
        )
    )
    marker = pending / module._TRANSACTION_MARKER_TEMP
    marker.write_bytes(
        module._transaction_payload(kwargs["subject"], "active", nonce)
        if marker_bytes is None
        else marker_bytes
    )
    assert authority.is_file() and pending.is_dir() and marker.is_file()

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not authority.exists()
    assert not pending.exists()


def test_hard_crash_stage_only_state_is_discarded_before_fresh_promotion(tmp_path):
    module = _load_runner()
    seed = _promote_fixture(module, tmp_path / "seed")
    module.promote_evidence(**seed)
    kwargs = _promote_fixture(module, tmp_path / "main")
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    parent_fd, _ = module._open_directory_nofollow(parent)
    try:
        transaction_name = module._create_transaction(parent_fd, kwargs["subject"])
    finally:
        os.close(parent_fd)
    transaction = parent / transaction_name
    seed["destination"].rename(transaction / "stage")
    assert (transaction / "stage" / "manifest.json").is_file()
    assert not kwargs["destination"].exists()

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not transaction.exists()


def test_hard_crash_backup_and_stage_without_target_restores_backup(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    module.promote_evidence(**kwargs)
    before = {
        path.relative_to(kwargs["destination"]).as_posix(): path.read_bytes()
        for path in kwargs["destination"].rglob("*")
        if path.is_file()
    }
    parent_fd, _ = module._open_directory_nofollow(kwargs["destination"].parent)
    try:
        transaction_name = module._create_transaction(parent_fd, kwargs["subject"])
    finally:
        os.close(parent_fd)
    transaction = kwargs["destination"].parent / transaction_name
    kwargs["destination"].rename(transaction / "backup")
    shutil.copytree(transaction / "backup", transaction / "stage")
    assert not kwargs["destination"].exists()
    assert (transaction / "backup").is_dir() and (transaction / "stage").is_dir()

    module._recover_interrupted_promotion(
        kwargs["destination"],
        subject=kwargs["subject"],
        subject_sources=kwargs["subject_sources"],
    )

    assert {
        path.relative_to(kwargs["destination"]).as_posix(): path.read_bytes()
        for path in kwargs["destination"].rglob("*")
        if path.is_file()
    } == before
    assert not transaction.exists()


def test_post_quarantine_rename_fsync_failure_restores_original(tmp_path, monkeypatch):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    before = (first["destination"] / "summary.json").read_bytes()
    second = _promote_fixture(module, tmp_path, summary="second")
    original = module._rename_between
    failed = False

    def fail_after_rename(source_fd, source, destination_fd, destination, expected):
        nonlocal failed
        if (
            source == second["destination"].name
            and destination == "backup"
            and not failed
        ):
            failed = True
            module._rename_noreplace(source_fd, source, destination_fd, destination)
            raise OSError("fsync after rename")
        return original(source_fd, source, destination_fd, destination, expected)

    monkeypatch.setattr(module, "_rename_between", fail_after_rename)
    with pytest.raises(module.CloseoutError):
        module.promote_evidence(**second)

    assert failed
    assert (second["destination"] / "summary.json").read_bytes() == before


def test_spoofed_retired_transaction_with_unknown_stage_is_never_deleted(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "9" * 32
    retired = parent / f".task-23019.txn-retired-{nonce}"
    retired.mkdir()
    (retired / module._TRANSACTION_MARKER).write_bytes(
        module._transaction_payload(kwargs["subject"], "retirement", nonce)
    )
    stage = retired / "stage"
    stage.mkdir()
    keep = stage / "keep.txt"
    keep.write_text("unrelated", encoding="utf-8")
    (parent / module._authority_name(nonce)).write_bytes(
        module._authority_payload(
            kwargs["subject"], nonce, module._receipt(retired.lstat())
        )
    )

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_collision"
    assert keep.read_text(encoding="utf-8") == "unrelated"


def test_partial_authority_temp_is_nonblocking_and_left_untouched(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    partial = parent / (".task-23019.txn-authority-" + "a" * 32 + ".json.tmp")
    partial.write_bytes(b'{"partial":')

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert partial.read_bytes() == b'{"partial":'


def test_empty_retired_directory_after_marker_removal_is_resumed(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    parent_fd, _ = module._open_directory_nofollow(parent)
    try:
        active = module._create_transaction(parent_fd, kwargs["subject"])
        retired_name = module._retire_transaction(parent_fd, active, kwargs["subject"])
    finally:
        os.close(parent_fd)
    retired = parent / retired_name
    (retired / module._TRANSACTION_MARKER).unlink()
    assert retired.is_dir() and not any(retired.iterdir())

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not retired.exists()


def test_authority_identity_swap_before_unlink_is_rejected(tmp_path, monkeypatch):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    parent_fd, _ = module._open_directory_nofollow(parent)
    try:
        active = module._create_transaction(parent_fd, kwargs["subject"])
        retired = module._retire_transaction(parent_fd, active, kwargs["subject"])
    finally:
        os.close(parent_fd)
    nonce = module._transaction_nonce(retired)
    authority = parent / module._authority_name(nonce)
    original = module._validate_authority
    swapped = False

    def swap_after_validation(parent_fd, nonce, subject, transaction_name):
        nonlocal swapped
        result = original(parent_fd, nonce, subject, transaction_name)
        if not swapped:
            swapped = True
            authority.rename(parent / "validated-authority")
            authority.write_text("replacement", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "_validate_authority", swap_after_validation)
    parent_fd, _ = module._open_directory_nofollow(parent)
    try:
        with pytest.raises(module.CloseoutError):
            module._delete_transaction(
                parent_fd,
                retired,
                kwargs["subject"],
                kwargs["subject_sources"],
            )
    finally:
        os.close(parent_fd)
    assert authority.read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize(
    "key", ("primary_secret", "buildSecret", "session_token", "myToken")
)
def test_generic_compound_secret_and_token_keys_are_rejected(tmp_path, key):
    module = _load_runner()
    roots = {name: tmp_path / name for name in ("checkout", "runtime", "scratch")}
    with pytest.raises(module.CloseoutError) as error:
        module.normalize_artifacts(
            {"summary.json": json.dumps({key: "opaque"}).encode()}, roots=roots
        )
    assert error.value.category == "credential_material"


def test_encoded_declared_root_is_rejected_without_decoding_safe_evidence(tmp_path):
    module = _load_runner()
    roots = {name: tmp_path / name for name in ("checkout", "runtime", "scratch")}
    encoded_root = urllib.parse.quote(str(roots["checkout"]), safe="")
    with pytest.raises(module.CloseoutError) as error:
        module.normalize_artifacts(
            {"summary.json": json.dumps({"path": encoded_root}).encode()}, roots=roots
        )
    assert error.value.category == "host_path_present"
    safe = b'{"url":"https://example.test/a%20b"}'
    assert (
        module.normalize_artifacts({"summary.json": safe}, roots=roots)["summary.json"]
        == safe
    )


@pytest.mark.parametrize(
    "encoded", ("%7B%22primary_api_key%22%3A%22opaque%22%7D", "API_KEY%3Dopaque")
)
def test_encoded_credential_material_is_rejected(tmp_path, encoded):
    module = _load_runner()
    roots = {name: tmp_path / name for name in ("checkout", "runtime", "scratch")}
    with pytest.raises(module.CloseoutError) as error:
        module.normalize_artifacts(
            {"captures/frame.txt": encoded.encode()}, roots=roots
        )
    assert error.value.category == "credential_material"


@pytest.mark.parametrize("metadata", ("manifest", "hashes"))
def test_bundle_metadata_rejects_top_level_extensions(tmp_path, metadata):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    destination = first["destination"]
    target = destination / f"{metadata}.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["extension"] = "not canonical"
    target.write_bytes(module._json_bytes(payload))
    if metadata == "manifest":
        hashes = json.loads((destination / "hashes.json").read_text(encoding="utf-8"))
        hashes["files"]["manifest.json"] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
        (destination / "hashes.json").write_bytes(module._json_bytes(hashes))
    second = _promote_fixture(module, tmp_path, summary="second")
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)
    assert error.value.category == "promotion_collision"


def test_pending_name_collision_never_publishes_authority(tmp_path, monkeypatch):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "b" * 32
    pending = parent / f".task-23019.txn-pending-{nonce}"
    pending.mkdir()
    keep = pending / "keep.txt"
    keep.write_text("unrelated", encoding="utf-8")
    monkeypatch.setattr(module.secrets, "token_hex", lambda _length: nonce)
    parent_fd, _ = module._open_directory_nofollow(parent)
    try:
        with pytest.raises((OSError, module.CloseoutError)):
            module._create_transaction(parent_fd, kwargs["subject"])
    finally:
        os.close(parent_fd)
    assert not (parent / module._authority_name(nonce)).exists()
    assert keep.read_text(encoding="utf-8") == "unrelated"


def test_hard_crash_after_bound_authority_before_preactivation_activation_recovers(
    tmp_path,
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "c" * 32
    preactivation = parent / f".task-23019.preactivation-{nonce}"
    preactivation.mkdir()
    authority = parent / module._authority_name(nonce)
    authority.write_bytes(
        module._authority_payload(
            kwargs["subject"], nonce, module._receipt(preactivation.lstat())
        )
    )
    assert preactivation.is_dir() and authority.is_file()

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not preactivation.exists()
    assert not authority.exists()


def test_hard_crash_during_authority_publication_empty_preactivation_is_nonblocking(
    tmp_path,
):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "d" * 32
    preactivation = parent / f".task-23019.preactivation-{nonce}"
    preactivation.mkdir()
    authority_temp = parent / (module._authority_name(nonce) + ".tmp")
    authority_temp.write_bytes(b'{"partial":')
    before = module._receipt(preactivation.lstat()), authority_temp.read_bytes()
    assert preactivation.is_dir() and authority_temp.is_file()

    module.promote_evidence(**kwargs)

    assert (kwargs["destination"] / "manifest.json").is_file()
    assert not preactivation.exists()
    assert authority_temp.read_bytes() == before[1]


def test_incomplete_preactivation_with_unknown_content_fails_closed_untouched(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "e" * 32
    preactivation = parent / f".task-23019.preactivation-{nonce}"
    preactivation.mkdir()
    keep = preactivation / "keep.txt"
    keep.write_text("unrelated", encoding="utf-8")
    authority_temp = parent / (module._authority_name(nonce) + ".tmp")
    authority_temp.write_bytes(b'{"partial":')

    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**kwargs)

    assert error.value.category == "promotion_collision"
    assert keep.read_text(encoding="utf-8") == "unrelated"
    assert authority_temp.read_bytes() == b'{"partial":'
    assert not kwargs["destination"].exists()


def test_cross_directory_quarantine_fsyncs_destination_before_source_and_rolls_back(
    tmp_path, monkeypatch
):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    before = (first["destination"] / "summary.json").read_bytes()
    second = _promote_fixture(module, tmp_path, summary="second")
    original_rename = module._rename_noreplace
    original_fsync = module.os.fsync
    tracking = False
    observed: list[int] = []
    expected: list[int] = []

    def track_quarantine(source_fd, source, destination_fd, destination):
        nonlocal tracking
        result = original_rename(source_fd, source, destination_fd, destination)
        if source == second["destination"].name and destination == "backup":
            tracking = True
            expected[:] = [destination_fd, source_fd]
        return result

    def fail_source_fsync(descriptor):
        nonlocal tracking
        if tracking:
            observed.append(descriptor)
            if len(observed) == 2:
                tracking = False
                raise OSError("source parent fsync failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(module, "_rename_noreplace", track_quarantine)
    monkeypatch.setattr(module.os, "fsync", fail_source_fsync)
    with pytest.raises(module.CloseoutError):
        module.promote_evidence(**second)

    assert observed == expected
    assert (second["destination"] / "summary.json").read_bytes() == before


def test_non_object_authority_json_is_a_stable_collision(tmp_path):
    module = _load_runner()
    kwargs = _promote_fixture(module, tmp_path)
    parent = kwargs["destination"].parent
    parent.mkdir(parents=True)
    nonce = "f" * 32
    (parent / module._authority_name(nonce)).write_bytes(b"[]")
    parent_fd, _ = module._open_directory_nofollow(parent)
    try:
        with pytest.raises(module.CloseoutError) as error:
            module._read_authority(parent_fd, nonce, kwargs["subject"])
    finally:
        os.close(parent_fd)
    assert error.value.category == "promotion_collision"


def test_non_object_manifest_json_is_a_stable_promotion_collision(tmp_path):
    module = _load_runner()
    first = _promote_fixture(module, tmp_path, summary="first")
    module.promote_evidence(**first)
    destination = first["destination"]
    manifest_path = destination / "manifest.json"
    manifest_path.write_bytes(b'[["not", "a", "mapping"]]')
    hashes_path = destination / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["files"]["manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    hashes_path.write_bytes(module._json_bytes(hashes))

    second = _promote_fixture(module, tmp_path, summary="second")
    with pytest.raises(module.CloseoutError) as error:
        module.promote_evidence(**second)

    assert error.value.category == "promotion_collision"


def _install_passing_production_main_fakes(module, monkeypatch):
    subject = _promotion_subject(module)
    sources, source_hashes = _promotion_sources(module)
    automated, live = _canonical_promotion_results(module)
    observed = {
        "admissions": [],
        "automated_targets": [],
        "live_cases": [],
        "live_scratch": [],
        "tree_checks": [],
    }

    def admit(repo, requested):
        observed["admissions"].append((repo, requested))
        return subject

    def load_sources(repo, admitted):
        assert admitted == subject
        return sources, source_hashes

    def run_child(*, checkout, scratch, mode, target, **_kwargs):
        assert mode == "pytest"
        relative = target.relative_to(checkout).as_posix()
        observed["automated_targets"].append(relative)
        payload = {
            name: value
            for name, value in automated.items()
            if name.partition("::")[0] == relative
        }
        payload[f"{relative}::test_unlisted_curated_smoke"] = "PASS"
        result_path = scratch / "automated-results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return module.ChildRunResult(0, None, result_path)

    def run_live(*, checkout, scratch, live_cases):
        observed["live_cases"].append(live_cases)
        observed["live_scratch"].append(scratch)
        for stem, (kind, _name) in module.REPRESENTATIVE_CAPTURES.items():
            if kind == "live":
                root, source_stem = module.REPRESENTATIVE_CAPTURE_SOURCES[stem]
                captures = scratch / f"raw-results/{root}/raw-evidence/captures"
                captures.mkdir(parents=True, exist_ok=True)
                (captures / f"{source_stem}.txt").write_text(
                    f"production compositor for {stem}\n", encoding="utf-8"
                )
                (captures / f"{source_stem}.svg").write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
                    encoding="utf-8",
                )
        return live

    def verify_tree(repo, admitted):
        observed["tree_checks"].append((repo, admitted))

    monkeypatch.setattr(module, "admit_subject", admit)
    monkeypatch.setattr(module, "load_subject_sources", load_sources)
    monkeypatch.setattr(module, "run_closeout_child", run_child)
    monkeypatch.setattr(module, "run_development_live_cases", run_live)
    monkeypatch.setattr(module, "verify_subject_tree", verify_tree)
    return subject, sources, source_hashes, automated, live, observed


def test_production_main_runs_complete_subject_and_promotes_after_raw_cleanup(
    monkeypatch, capsys
):
    module = _load_runner()
    subject, sources, source_hashes, automated, live, observed = (
        _install_passing_production_main_fakes(module, monkeypatch)
    )
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", subject.commit)
    validations = []
    promotions = []
    validate_complete = module.validate_complete_results

    def validate(*args, **kwargs):
        validations.append((args, kwargs))
        return validate_complete(*args, **kwargs)

    def promote(**kwargs):
        assert not kwargs["raw_root"].exists()
        promotions.append(kwargs)

    monkeypatch.setattr(module, "validate_complete_results", validate)
    monkeypatch.setattr(module, "promote_evidence", promote)

    assert module.main(["--subject-revision", subject.commit, "--promote"]) == 0

    assert observed["admissions"] == [(REPO_ROOT, subject.commit)]
    assert observed["automated_targets"] == list(module.CURATED_PYTEST_FILES)
    assert observed["live_cases"] == [module.EXECUTABLE_LIVE_ROOTS]
    assert observed["tree_checks"] == [(REPO_ROOT, subject)]
    assert len(validations) == 1
    assert len(promotions) == 1
    promoted = promotions[0]
    assert promoted["destination"] == REPO_ROOT / module.SOURCE_DIRECTORY
    assert promoted["subject"] == subject
    assert promoted["subject_sources"] == sources
    assert promoted["subject_hashes"] == source_hashes
    assert promoted["automated_results"] == automated
    assert promoted["live_results"] == live
    assert json.loads(promoted["raw_artifacts"]["summary.json"]) == (
        module._canonical_summary(automated, live)
    )
    assert {
        path for path in promoted["raw_artifacts"] if path.startswith("captures/")
    } == {
        f"captures/{stem}{suffix}"
        for stem in module.REPRESENTATIVE_CAPTURES
        for suffix in (".txt", ".svg")
    }
    assert all(not scratch.exists() for scratch in observed["live_scratch"])
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_production_no_promote_executes_and_validates_without_repository_writes(
    monkeypatch, capsys
):
    module = _load_runner()
    subject, _sources, _hashes, _automated, _live, observed = (
        _install_passing_production_main_fakes(module, monkeypatch)
    )
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", subject.commit)
    validations = []
    validate_complete = module.validate_complete_results

    def validate(*args, **kwargs):
        validations.append((args, kwargs))
        return validate_complete(*args, **kwargs)

    monkeypatch.setattr(module, "validate_complete_results", validate)
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("no-promote attempted a repository write"),
    )

    assert module.main(["--subject-revision", subject.commit, "--no-promote"]) == 0

    assert observed["automated_targets"] == list(module.CURATED_PYTEST_FILES)
    assert observed["live_cases"] == [module.EXECUTABLE_LIVE_ROOTS]
    assert observed["tree_checks"] == [(REPO_ROOT, subject)]
    assert len(validations) == 1
    assert all(not scratch.exists() for scratch in observed["live_scratch"])
    assert json.loads(capsys.readouterr().out)["promoted"] is False


def test_verify_evidence_validates_bundle_without_executing_tests(
    monkeypatch, tmp_path, capsys
):
    module = _load_runner()
    subject = _promotion_subject(module)
    sources, source_hashes = _promotion_sources(module)
    evidence = tmp_path / "task-23019"
    manifest = {
        "subject_commit": subject.commit,
        "subject_tree": subject.tree,
    }
    monkeypatch.setattr(
        module,
        "_read_artifact_tree",
        lambda root, **_kwargs: {"manifest.json": json.dumps(manifest).encode("utf-8")},
    )
    monkeypatch.setattr(
        module,
        "load_subject_sources",
        lambda repo, admitted: (sources, source_hashes),
    )
    validations = []

    def validate_bundle(root, *, subject, subject_sources):
        validations.append((root, subject, subject_sources))
        return {}, (1, 2, 3, "receipt")

    monkeypatch.setattr(module, "_validate_bundle", validate_bundle)
    monkeypatch.setattr(
        module,
        "run_closeout_child",
        lambda **_kwargs: pytest.fail("verification executed automated tests"),
    )
    monkeypatch.setattr(
        module,
        "run_development_live_cases",
        lambda **_kwargs: pytest.fail("verification executed live tests"),
    )
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("verification attempted promotion"),
    )

    assert module.main(["--verify-evidence", str(evidence)]) == 0

    assert validations == [(evidence, subject, sources)]
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == {
        "status": "PASS",
        "subject_commit": subject.commit,
        "subject_tree": subject.tree,
        "verified_evidence": str(evidence),
    }


@pytest.mark.parametrize(
    "category", ("subject_revision_mismatch", "subject_worktree_not_clean")
)
def test_production_subject_admission_failure_blocks_execution_and_promotion(
    monkeypatch, category, capsys
):
    module = _load_runner()
    requested = "a" * 40
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", requested)
    monkeypatch.setattr(
        module,
        "admit_subject",
        lambda *_args: (_ for _ in ()).throw(module.CloseoutError(category)),
    )
    monkeypatch.setattr(
        module,
        "run_closeout_child",
        lambda **_kwargs: pytest.fail("failed subject executed tests"),
    )
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("failed subject attempted promotion"),
    )

    assert module.main(["--subject-revision", requested, "--promote"]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": category}


@pytest.mark.parametrize("failed_kind", ("automated_result", "live_result"))
def test_failed_production_result_blocks_promotion(monkeypatch, failed_kind, capsys):
    module = _load_runner()
    subject, _sources, _hashes, automated, live, _observed = (
        _install_passing_production_main_fakes(module, monkeypatch)
    )
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", subject.commit)
    if failed_kind == "automated_result":
        failed = next(iter(automated))
        automated[failed] = "FAIL"
    else:
        failed = next(iter(live))
        live[failed] = "FAIL"
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("failed result attempted promotion"),
    )

    assert module.main(["--subject-revision", subject.commit, "--promote"]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": "result_status_invalid"}


@pytest.mark.parametrize(
    ("environment_revision", "arguments", "category"),
    [
        (
            None,
            ["--subject-revision", "a" * 40, "--promote"],
            "subject_revision_environment_required",
        ),
        (
            "b" * 40,
            ["--subject-revision", "a" * 40, "--promote"],
            "subject_revision_environment_mismatch",
        ),
        ("a" * 40, ["--no-promote"], "subject_revision_required"),
    ],
)
def test_production_subject_argument_and_environment_must_agree_stably(
    monkeypatch, environment_revision, arguments, category, capsys
):
    module = _load_runner()
    if environment_revision is None:
        monkeypatch.delenv("TASK23019_SUBJECT_REVISION", raising=False)
    else:
        monkeypatch.setenv("TASK23019_SUBJECT_REVISION", environment_revision)
    monkeypatch.setattr(
        module,
        "admit_subject",
        lambda *_args: pytest.fail("invalid revision agreement admitted subject"),
    )

    assert module.main(arguments) == 2
    assert json.loads(capsys.readouterr().err) == {"error": category}


def test_production_raw_materialization_failure_is_stable_and_does_not_promote(
    monkeypatch, capsys
):
    module = _load_runner()
    subject, *_rest = _install_passing_production_main_fakes(module, monkeypatch)
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", subject.commit)
    original_write_bytes = Path.write_bytes

    def fail_summary_write(path, payload):
        if path.name == "summary.json":
            raise OSError("injected raw write failure")
        return original_write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", fail_summary_write)
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("failed raw materialization promoted evidence"),
    )

    assert module.main(["--subject-revision", subject.commit, "--promote"]) == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "production_evidence_io_failed"
    }


def test_production_raw_cleanup_failure_is_stable_and_does_not_promote(
    monkeypatch, tmp_path, capsys
):
    module = _load_runner()
    subject, *_rest = _install_passing_production_main_fakes(module, monkeypatch)
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", subject.commit)
    raw_root = tmp_path / "raw-root"

    class FailingCleanup:
        def __enter__(self):
            raw_root.mkdir()
            return str(raw_root)

        def __exit__(self, *_args):
            raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        module.tempfile, "TemporaryDirectory", lambda **_kwargs: FailingCleanup()
    )
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("failed raw cleanup promoted evidence"),
    )

    assert module.main(["--subject-revision", subject.commit, "--promote"]) == 2
    assert raw_root.exists()
    assert json.loads(capsys.readouterr().err) == {
        "error": "production_evidence_io_failed"
    }


def test_production_cleanup_failure_preserves_primary_closeout_error(
    monkeypatch, tmp_path, capsys
):
    module = _load_runner()
    subject, *_rest = _install_passing_production_main_fakes(module, monkeypatch)
    monkeypatch.setenv("TASK23019_SUBJECT_REVISION", subject.commit)
    raw_root = tmp_path / "raw-root"

    class FailingCleanup:
        def __enter__(self):
            raw_root.mkdir()
            return str(raw_root)

        def __exit__(self, *_args):
            raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        module.tempfile, "TemporaryDirectory", lambda **_kwargs: FailingCleanup()
    )
    monkeypatch.setattr(
        module,
        "run_closeout_child",
        lambda **_kwargs: (_ for _ in ()).throw(
            module.CloseoutError("primary_child_failure")
        ),
    )
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("failed run promoted evidence"),
    )

    assert module.main(["--subject-revision", subject.commit, "--promote"]) == 2
    assert raw_root.exists()
    assert json.loads(capsys.readouterr().err) == {"error": "primary_child_failure"}


def test_verify_evidence_unresolvable_manifest_subject_fails_stably(
    monkeypatch, tmp_path, capsys
):
    module = _load_runner()
    evidence = tmp_path / "bundle"
    monkeypatch.setattr(
        module,
        "_read_artifact_tree",
        lambda *_args, **_kwargs: {
            "manifest.json": json.dumps(
                {"subject_commit": "not-a-revision", "subject_tree": "not-a-tree"}
            ).encode()
        },
    )
    monkeypatch.setattr(
        module,
        "_git",
        lambda *_args: (_ for _ in ()).throw(
            subprocess.CalledProcessError(128, ["git", "rev-parse"])
        ),
    )
    monkeypatch.setattr(
        module,
        "run_closeout_child",
        lambda **_kwargs: pytest.fail("invalid evidence subject executed tests"),
    )
    monkeypatch.setattr(
        module,
        "promote_evidence",
        lambda **_kwargs: pytest.fail("invalid evidence subject promoted"),
    )

    assert module.main(["--verify-evidence", str(evidence)]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": "subject_tree_mismatch"}
