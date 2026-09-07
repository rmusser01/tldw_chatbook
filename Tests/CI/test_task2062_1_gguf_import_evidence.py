"""Semantic contract for TASK-2062.1's bounded native evidence workflow."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/task-2062-1-gguf-import-evidence.yml"
REQUIRED_NODES = (
    "Tests/Model_Artifacts/test_gguf_admission.py::test_open_local_gguf_rejects_windows_reparse_point",
    "Tests/Model_Artifacts/test_gguf_admission.py::test_validate_local_gguf_rejects_replacement_between_lstat_and_open",
    "Tests/Model_Artifacts/test_service.py::test_import_local_gguf_promotes_path_private_full_digest_artifact",
    "Tests/Model_Artifacts/test_service.py::test_import_cancel_during_copy_removes_only_its_stage",
    "Tests/Model_Artifacts/test_service.py::test_import_cancel_immediately_before_promotion_never_publishes",
    "Tests/Model_Artifacts/test_service.py::test_import_finalizing_is_point_of_no_return",
    "Tests/Model_Artifacts/test_service.py::test_reconcile_removes_only_abandoned_import_stage",
    "Tests/Model_Artifacts/test_service.py::test_concurrent_identical_imports_converge_on_one_manifest",
    "Tests/Model_Artifacts/test_service.py::test_import_local_gguf_preserves_retryable_lease_timeout",
    "Tests/UI/test_model_installed_view.py::test_tldwcli_css_finish_slice_restores_terminal_import_focus",
    "Tests/UI/test_model_installed_view.py::test_import_progress_updates_without_replacing_focused_cancel",
    "Tests/UI/test_model_installed_view.py::test_physical_cancel_sets_service_probe_and_preserves_source",
    "Tests/UI/test_model_installed_view.py::test_attached_queued_cancel_settles_without_entering_service",
    "Tests/UI/test_model_installed_view.py::test_finalizing_disables_cancel_before_promotion",
    "Tests/UI/test_model_installed_view.py::test_activation_failure_keeps_installed_row_and_offers_activate",
    "Tests/UI/test_model_installed_view.py::test_import_failure_logs_only_stable_category_and_never_selected_path",
    "Tests/UI/test_model_installed_view.py::test_real_import_lease_timeout_offers_busy_retry_without_publication",
    "Tests/UI/test_model_installed_view.py::test_import_lane_disables_every_lifecycle_action_at_80_columns",
)
EXPECTED_OSES = ("ubuntu-latest", "macos-latest", "windows-latest")
EXPECTED_PULL_REQUEST_PATHS = (
    ".github/workflows/task-2062-1-gguf-import-evidence.yml",
    "pyproject.toml",
    "tldw_chatbook/app.py",
    "tldw_chatbook/css/**",
    "tldw_chatbook/Model_Artifacts/**",
    "tldw_chatbook/UI/Screens/model_installed_view.py",
    "Tests/conftest.py",
    "Tests/Model_Artifacts/**",
    "Tests/UI/conftest.py",
    "Tests/UI/consolidated_css.py",
    "Tests/UI/test_model_installed_view.py",
)
ASYNC_EVIDENCE_FUNCTIONS = {
    "Tests/UI/test_model_installed_view.py": {
        "test_tldwcli_css_finish_slice_restores_terminal_import_focus",
        "test_import_progress_updates_without_replacing_focused_cancel",
        "test_physical_cancel_sets_service_probe_and_preserves_source",
        "test_attached_queued_cancel_settles_without_entering_service",
        "test_finalizing_disables_cancel_before_promotion",
        "test_activation_failure_keeps_installed_row_and_offers_activate",
        "test_import_failure_logs_only_stable_category_and_never_selected_path",
        "test_real_import_lease_timeout_offers_busy_retry_without_publication",
        "test_import_lane_disables_every_lifecycle_action_at_80_columns",
    },
}


def _workflow() -> tuple[str, dict[str, object]]:
    assert WORKFLOW_PATH.is_file(), "TASK-2062.1 native evidence workflow is missing"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    matches = [
        step for step in steps if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_workflow_is_one_read_only_exact_three_os_matrix() -> None:
    text, workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert triggers["pull_request"] == {
        "branches": ["dev"],
        "paths": list(EXPECTED_PULL_REQUEST_PATHS),
    }
    assert workflow.get("permissions") == {"contents": "read"}

    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and len(jobs) == 1
    job = next(iter(jobs.values()))
    assert isinstance(job, dict)
    assert job.get("timeout-minutes") == 20
    strategy = job.get("strategy")
    assert isinstance(strategy, dict)
    assert strategy.get("fail-fast") is False
    assert strategy.get("matrix") == {"os": list(EXPECTED_OSES)}
    assert job.get("runs-on") == "${{ matrix.os }}"

    uses = [
        step["uses"]
        for step in job["steps"]
        if isinstance(step, dict) and "uses" in step
    ]
    assert uses == ["actions/checkout@v4", "actions/setup-python@v5"]
    checkout = job["steps"][0]
    assert checkout.get("with") == {
        "ref": "${{ github.event.pull_request.head.sha || github.sha }}"
    }
    setup = job["steps"][1]
    assert setup.get("with") == {"python-version": "3.12"}

    lowered = text.casefold()
    for forbidden in (
        "continue-on-error",
        "actions/cache",
        "upload-artifact",
        "download-artifact",
        "cache:",
        "secrets.",
        "contents: write",
    ):
        assert forbidden not in lowered


def test_workflow_installs_only_core_and_bounded_test_dependencies() -> None:
    _text, workflow = _workflow()
    job = next(iter(workflow["jobs"].values()))
    install = _named_step(job, "Install bounded test dependencies")
    assert install.get("run") == (
        "python -m pip install -e . pytest pytest-asyncio pytest-timeout"
    )


def test_workflow_runs_every_exact_node_once_and_nothing_broader() -> None:
    _text, workflow = _workflow()
    job = next(iter(workflow["jobs"].values()))
    test_step = _named_step(job, "Run exact GGUF import evidence nodes")
    command = str(test_step.get("run", ""))
    selected = tuple(re.findall(r"Tests/\S+?::\S+?(?=\s|\\|$)", command))
    assert selected == REQUIRED_NODES
    assert command.startswith("python -m pytest ")
    assert "--timeout=60" in command
    assert " -q" in command
    assert "--ignore" not in command
    assert "-k " not in command


def test_windows_async_nodes_explicitly_allow_only_the_proactor_socketpair() -> None:
    for relative_path, expected_names in ASYNC_EVIDENCE_FUNCTIONS.items():
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in expected_names:
            node = functions[name]
            decorators = {ast.unparse(decorator) for decorator in node.decorator_list}
            assert "pytest.mark.allow_network" in decorators, (
                f"{relative_path}::{name} needs the documented Windows Proactor "
                "socket-pair exception"
            )
