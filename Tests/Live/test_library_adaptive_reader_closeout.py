from __future__ import annotations

import builtins
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
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


def _load_child():
    spec = importlib.util.spec_from_file_location(
        "task23019_closeout_child", CHILD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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

    monkeypatch.setattr(module.subprocess, "run", fake_run)

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

    monkeypatch.setattr(module.subprocess, "run", fake_run)

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
