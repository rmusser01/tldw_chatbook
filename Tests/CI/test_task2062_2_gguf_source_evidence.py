"""Semantic contract for TASK-2062.2's bounded native evidence workflow."""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/task-2062-2-gguf-source-evidence.yml"
REQUIRED_NODES = (
    "Tests/Model_Artifacts/test_gguf_admission.py::test_open_local_gguf_rejects_symlink",
    "Tests/Model_Artifacts/test_gguf_admission.py::test_open_local_gguf_rejects_windows_reparse_point",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_external_identity_change_after_inspection_fails_final_recheck_before_popen",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_external_source_validation_is_worker_thread_store_free_and_read_only",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_handlers_reject_source_overrides_before_reserving_claim",
    "Tests/LLM_Management/test_gguf_source_modes.py::test_acquire_managed_gguf_returns_exact_declared_payload_and_open_lease",
    "Tests/Model_Artifacts/test_service.py::test_acquire_preserves_retryable_shared_lease_timeout_from_real_contention",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_managed_transfer_precedes_popen_and_spawn_failure_closes_once",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_cancel_before_managed_preparation_releases_claim_without_acquire",
    "Tests/LLM_Management/test_server_lifecycle_resources.py::test_process_exit_closes_resource_only_after_exact_death",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_real_managed_lease_blocks_delete_until_exact_claim_and_process_death",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_real_stop_terminates_reaps_and_releases_managed_lease_for_delete",
    "Tests/LLM_Management/test_server_lifecycle_resources.py::test_stale_release_and_clear_cannot_close_any_generation_resource",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_source_command_matrix_uses_only_active_authority",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_gguf_nonzero_exit_presents_sanitized_runtime_compatibility_copy",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_real_managed_contention_delivers_busy_recovery_without_spawning",
    "Tests/LLM_Management/test_vllm_setup.py::test_vllm_command_snapshot_is_unchanged",
    "Tests/LLM_Management/test_gguf_server_sources.py::test_mlx_command_snapshot_is_unchanged",
)
REQUIRED_UI_NODES = (
    "Tests/UI/test_llm_gguf_source_modes.py::test_lazy_mount_cached_inventory_preserves_handoff_awaiting_fresh_read",
    "Tests/UI/test_llm_gguf_source_modes.py::test_inventory_during_lazy_selector_mount_is_replayed_after_leaving_pane[False]",
    "Tests/UI/test_llm_gguf_source_modes.py::test_inventory_during_lazy_selector_mount_is_replayed_after_leaving_pane[True]",
    "Tests/UI/test_llm_gguf_source_modes.py::test_claim_authority_survives_screen_recompose_and_not_window_selection",
    "Tests/UI/test_llm_gguf_source_modes.py::test_external_copy_keyboard_geometry_and_unrelated_views_stay_stable",
    "Tests/UI/test_llm_gguf_source_modes.py::test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp]",
    "Tests/UI/test_llm_gguf_source_modes.py::test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamafile-llamafile]",
)
EXPECTED_OSES = ("ubuntu-latest", "macos-latest", "windows-latest")
EXPECTED_PULL_REQUEST_PATHS = (
    ".github/workflows/task-2062-2-gguf-source-evidence.yml",
    "pyproject.toml",
    "tldw_chatbook/app.py",
    "tldw_chatbook/config.py",
    "tldw_chatbook/Event_Handlers/LLM_Management_Events/**",
    "tldw_chatbook/Model_Artifacts/**",
    "tldw_chatbook/UI/LLM_Management_Window.py",
    "tldw_chatbook/UI/Screens/llm_screen.py",
    "Tests/conftest.py",
    "Tests/private_profile.py",
    "Tests/LLM_Management/**",
    "Tests/Model_Artifacts/**",
    "Tests/UI/app_factory.py",
    "Tests/UI/conftest.py",
    "Tests/UI/test_llm_gguf_source_modes.py",
)
EXPECTED_STEP_NAMES = (
    "Enable long checkout paths on Windows",
    "Check out the exact tested commit",
    "Set up Python 3.12",
    "Install bounded test dependencies",
    "Prepare Windows package files beneath a trusted private ancestor",
    "Run exact GGUF source evidence nodes",
    "Run each full-app GGUF case with its profile selected before imports",
    "Compare Windows keyboard waits with the dev source baseline",
)


def _workflow() -> tuple[str, dict[str, object]]:
    assert WORKFLOW_PATH.is_file(), "TASK-2062.2 native evidence workflow is missing"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def _only_job(workflow: dict[str, object]) -> dict[str, object]:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and len(jobs) == 1
    job = next(iter(jobs.values()))
    assert isinstance(job, dict)
    return job


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    matches = [
        step for step in steps if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def _command_tokens(command: str) -> tuple[str, ...]:
    return tuple(shlex.split(command.replace("\\\n", " ")))


def test_workflow_is_one_read_only_exact_three_os_matrix() -> None:
    text, workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))
    assert triggers == {
        "pull_request": {
            "branches": ["dev"],
            "paths": list(EXPECTED_PULL_REQUEST_PATHS),
        },
        "workflow_dispatch": None,
    }
    assert workflow.get("permissions") == {"contents": "read"}
    assert set(workflow) == {"name", "on", "permissions", "jobs"} or set(workflow) == {
        "name",
        True,
        "permissions",
        "jobs",
    }

    job = _only_job(workflow)
    assert job.get("timeout-minutes") == 20
    assert job.get("runs-on") == "${{ matrix.os }}"
    assert job.get("defaults") == {"run": {"shell": "bash"}}
    assert "permissions" not in job and "env" not in job
    strategy = job.get("strategy")
    assert strategy == {
        "fail-fast": False,
        "matrix": {"os": list(EXPECTED_OSES)},
    }

    steps = job.get("steps")
    assert isinstance(steps, list)
    assert tuple(step.get("name") for step in steps) == EXPECTED_STEP_NAMES
    # Only the final diagnostic may ignore a failure; evidence remains required.
    expected_step_keys = (
        {"name", "if", "shell", "run"},
        {"name", "uses", "with"},
        {"name", "uses", "with"},
        {"name", "run"},
        {"name", "if", "run"},
        {"name", "working-directory", "run"},
        {"name", "working-directory", "env", "run"},
    )
    assert tuple(set(step) for step in steps[:-1]) == expected_step_keys
    assert set(steps[-1]) == {"name", "if", "continue-on-error", "run"}
    assert [step.get("uses") for step in steps if "uses" in step] == [
        "actions/checkout@v4",
        "actions/setup-python@v5",
    ]
    checkout = _named_step(job, "Check out the exact tested commit")
    assert checkout.get("with") == {
        "ref": "${{ github.event.pull_request.head.sha || github.sha }}"
    }
    python = _named_step(job, "Set up Python 3.12")
    assert python.get("with") == {"python-version": "3.12"}
    assert steps[0] == {
        "name": "Enable long checkout paths on Windows",
        "if": "runner.os == 'Windows'",
        "shell": "pwsh",
        "run": "git config --global core.longpaths true",
    }

    lowered = text.casefold()
    assert lowered.count("continue-on-error") == 1
    for forbidden in (
        "actions/cache",
        "upload-artifact",
        "download-artifact",
        "cache:",
        "secrets.",
        "contents: write",
        "pull-requests: write",
        "curl ",
        "wget ",
        "docker ",
        "llama-server",
        "ollama serve",
    ):
        assert forbidden not in lowered


def test_windows_failure_diagnostic_keeps_exact_read_only_commands() -> None:
    _text, workflow = _workflow()
    diagnostic = _named_step(_only_job(workflow), EXPECTED_STEP_NAMES[-1])
    assert diagnostic.get("if") == "failure() && runner.os == 'Windows'"
    assert diagnostic.get("continue-on-error") is True
    baseline = "4631b60f8dd9623fc55bf16f4a37e29fcb1240c7"
    assert _command_tokens(str(diagnostic.get("run", ""))) == (
        "git",
        "fetch",
        "--depth=1",
        "origin",
        baseline,
        "git",
        "worktree",
        "add",
        "--detach",
        "$RUNNER_TEMP/backup-startup-dev",
        baseline,
        "diagnostic=$GITHUB_WORKSPACE/Tests/Backup_Recovery/startup_timing_diagnostic.py",
        "failed=0",
        "for",
        "node",
        "in",
        *REQUIRED_UI_NODES[-2:],
        "do",
        "python",
        "$diagnostic",
        "--source",
        "$RUNNER_TEMP/backup-startup-dev",
        "--node",
        "$node",
        "--label",
        "dev",
        "--no-profile",
        "||",
        "failed=1",
        "python",
        "$diagnostic",
        "--source",
        "$GGUF_EVIDENCE_SOURCE",
        "--node",
        "$node",
        "--label",
        "candidate",
        "--no-profile",
        "||",
        "failed=1",
        "done",
        "exit",
        "$failed",
    )


def test_workflow_installs_only_editable_package_and_bounded_test_dependencies() -> (
    None
):
    _text, workflow = _workflow()
    install = _named_step(_only_job(workflow), "Install bounded test dependencies")
    assert install.get("run") == (
        "python -m pip install -e . pytest pytest-asyncio pytest-timeout"
    )


def test_workflow_runs_every_exact_node_once_with_only_bounded_flags() -> None:
    _text, workflow = _workflow()
    test_step = _named_step(
        _only_job(workflow),
        "Run exact GGUF source evidence nodes",
    )
    command = str(test_step.get("run", ""))
    assert (
        test_step.get("working-directory")
        == "${{ env.GGUF_EVIDENCE_SOURCE || github.workspace }}"
    )
    assert _command_tokens(command) == (
        "python",
        "-m",
        "pytest",
        *REQUIRED_NODES,
        "--timeout=60",
        "-q",
    )
    selected = tuple(re.findall(r"Tests/\S+?::\S+?(?=\s|\\|$)", command))
    assert selected == REQUIRED_NODES
    assert len(selected) == len(set(selected))

    ui_step = _named_step(
        _only_job(workflow),
        "Run each full-app GGUF case with its profile selected before imports",
    )
    ui_tokens = _command_tokens(str(ui_step.get("run", "")))
    assert ui_step.get("working-directory") == test_step.get("working-directory")
    assert ui_step.get("env") == {
        "GGUF_APP_TIMEOUT": "${{ runner.os == 'Windows' && '180' || '60' }}"
    }
    assert ui_tokens == (
        "failed=0",
        "for",
        "node",
        "in",
        *REQUIRED_UI_NODES,
        "do",
        "TLDW_TEST_PRIVATE_PROFILE_NODE=$node",
        "python",
        "-m",
        "pytest",
        "$node",
        "--timeout=$GGUF_APP_TIMEOUT",
        "-v",
        "||",
        "failed=1",
        "done",
        "exit",
        "$failed",
    )
    selected += tuple(token for token in ui_tokens if token.startswith("Tests/"))
    assert selected == REQUIRED_NODES + REQUIRED_UI_NODES
    assert len(selected) == len(set(selected))


def test_every_workflow_node_names_one_existing_test_function() -> None:
    by_file: dict[str, set[str]] = {}
    for node in REQUIRED_NODES + REQUIRED_UI_NODES:
        relative_path, function_name = node.split("::", 1)
        by_file.setdefault(relative_path, set()).add(function_name.split("[", 1)[0])

    for relative_path, expected_names in by_file.items():
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert expected_names <= functions
