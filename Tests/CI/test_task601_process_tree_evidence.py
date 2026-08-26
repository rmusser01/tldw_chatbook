from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / ".github/scripts/task601_process_tree_evidence.py"
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/task-601-platform-evidence.yml"
REQUIRED_NODES = (
    "Tests/STT/test_executor_process_tree.py::test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup",
    "Tests/STT/test_executor_process_tree.py::test_native_crashed_leader_reaps_descendant_before_scratch_cleanup",
    "Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch",
)
EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}

EXPECTED_STEP_NAMES = (
    "Check out the exact tested commit",
    "Set up Python",
    "Initialize failure evidence",
    "Install test dependencies",
    "Record dependency installation failure",
    "Run bounded platform tests",
    "Normalize platform evidence",
    "Validate platform evidence",
    "Upload platform evidence",
)


def _load_evidence() -> ModuleType:
    assert EVIDENCE_PATH.is_file(), "TASK-601 evidence normalizer is missing"
    spec = importlib.util.spec_from_file_location("task601_evidence", EVIDENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow_text() -> str:
    assert WORKFLOW_PATH.is_file(), "TASK-601 evidence workflow is missing"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _yaml_block(text: str, header: str) -> str:
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line == header]
    assert len(matches) == 1, f"expected one YAML header: {header!r}"
    start = matches[0]
    indentation = len(header) - len(header.lstrip())
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and len(line) - len(line.lstrip()) <= indentation:
            end = index
            break
    return "\n".join(lines[start + 1 : end])


def _nonempty_lines(block: str) -> list[str]:
    return [line for line in block.splitlines() if line.strip()]


def _workflow_step(workflow: str, name: str) -> str:
    return _yaml_block(workflow, f"      - name: {name}")


def _yaml_scalar(block: str, indentation: int, key: str) -> str:
    prefix = " " * indentation + key + ":"
    matches = [
        line.removeprefix(prefix).strip()
        for line in block.splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1, f"expected one YAML scalar: {key!r}"
    return matches[0]


def _step_command(step: str) -> str:
    lines = step.splitlines()
    run_lines = [
        index for index, line in enumerate(lines) if line.startswith("        run:")
    ]
    assert len(run_lines) == 1
    run_index = run_lines[0]
    first = lines[run_index].removeprefix("        run:").strip()
    if first != "|":
        return first
    command_lines: list[str] = []
    for line in lines[run_index + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= 8:
            break
        if line.strip():
            command_lines.append(line.strip().removesuffix("\\").rstrip())
    return " ".join(command_lines)


def _run_identity(
    *, commit: str = "a" * 40, run_id: str = "123", attempt: str = "1"
) -> dict[str, str]:
    return {
        "tested_commit": commit,
        "workflow_run_id": run_id,
        "workflow_run_attempt": attempt,
        "workflow_run_url": (
            "https://github.com/rmusser01/tldw_chatbook/actions/runs/" + run_id
        ),
    }


def _valid_result(evidence_name: str) -> dict[str, object]:
    system, architecture = EXPECTED_PLATFORMS[evidence_name]
    return {
        "schema_version": 1,
        "evidence_label": "task601_native_process_tree",
        "evidence_name": evidence_name,
        "status": "passed",
        "failure_code": None,
        "failure_stage": None,
        "run": _run_identity(),
        "host": {
            "system": system,
            "architecture": architecture,
            "python": "3.12.9",
        },
        "pytest": {
            "outcome": "success",
            "duration_seconds": 0.75,
            "required_nodes": dict.fromkeys(REQUIRED_NODES, "passed"),
        },
    }


def _junit_case(node: str, body: str = "", *, file_value: str = "ignored.py") -> str:
    path, name = node.split("::", 1)
    classname = path.removesuffix(".py").replace("/", ".")
    return (
        f'<testcase classname="{classname}" name="{name}" '
        f'file="{file_value}" time="0.25">{body}</testcase>'
    )


def _junit(*cases: str, duration: str = "0.75") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuite name="pytest" tests="{len(cases)}" time="{duration}">'
        + "".join(cases)
        + "</testsuite>"
    )


def _write_junit(tmp_path: Path, xml: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _passing_junit(tmp_path: Path, *, file_value: str = "ignored.py") -> Path:
    return _write_junit(
        tmp_path,
        _junit(*(_junit_case(node, file_value=file_value) for node in REQUIRED_NODES)),
    )


def _stub_linux_evidence(evidence: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evidence, "current_run_identity", lambda: _run_identity())
    monkeypatch.setattr(
        evidence,
        "_host_result",
        lambda _name: {
            "system": "Linux",
            "architecture": "x86_64",
            "python": "3.12.9",
        },
    )


def _main_code(evidence: ModuleType, args: list[str]) -> int:
    try:
        return evidence.main(args)
    except SystemExit as error:
        return int(error.code or 0)


def test_workflow_has_only_explicit_triggers_and_read_only_permissions() -> None:
    workflow = _workflow_text()

    assert _nonempty_lines(_yaml_block(workflow, "on:")) == [
        "  pull_request:",
        "    types: [labeled]",
        "  workflow_dispatch:",
    ]
    assert _nonempty_lines(_yaml_block(workflow, "permissions:")) == [
        "  contents: read"
    ]
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "concurrency:" not in workflow
    assert "secrets." not in workflow
    assert workflow.count("permissions:") == 1
    assert ": write" not in workflow


def test_workflow_gates_one_job_and_checks_out_the_selected_commit() -> None:
    workflow = _workflow_text()
    jobs = _yaml_block(workflow, "jobs:")
    checkout = _workflow_step(workflow, EXPECTED_STEP_NAMES[0])

    assert re.findall(r"^  ([a-z0-9-]+):$", jobs, re.MULTILINE) == ["platform-evidence"]
    assert _yaml_scalar(jobs, 4, "if") == (
        "github.event_name == 'workflow_dispatch' || "
        "github.event.label.name == 'task-601-platform-evidence'"
    )
    assert _yaml_scalar(checkout, 8, "uses") == "actions/checkout@v4"
    assert _nonempty_lines(_yaml_block(checkout, "        with:")) == [
        "          ref: ${{ github.event.pull_request.head.sha || github.sha }}"
    ]


def test_workflow_uses_only_the_exact_bounded_three_os_matrix() -> None:
    workflow = _workflow_text()
    strategy = _yaml_block(workflow, "    strategy:")
    defaults = _yaml_block(workflow, "    defaults:")
    setup = _workflow_step(workflow, EXPECTED_STEP_NAMES[1])

    assert _nonempty_lines(strategy) == [
        "      fail-fast: false",
        "      matrix:",
        "        include:",
        "          - evidence_name: linux-x86_64",
        "            os: ubuntu-24.04",
        "          - evidence_name: windows-x86_64",
        "            os: windows-2022",
        "          - evidence_name: macos-x86_64",
        "            os: macos-15-intel",
    ]
    jobs = _yaml_block(workflow, "jobs:")
    assert _yaml_scalar(jobs, 4, "runs-on") == "${{ matrix.os }}"
    assert _yaml_scalar(jobs, 4, "timeout-minutes") == "20"
    assert _nonempty_lines(defaults) == ["      run:", "        shell: bash"]
    assert workflow.count("shell:") == 1
    assert _yaml_scalar(setup, 8, "uses") == "actions/setup-python@v5"
    assert _nonempty_lines(_yaml_block(setup, "        with:")) == [
        '          python-version: "3.12"',
        "          cache: pip",
    ]
    assert "actions/cache" not in workflow
    assert "HF_HOME" not in workflow
    assert "TRANSFORMERS_CACHE" not in workflow


def test_workflow_runs_only_the_exact_dependency_and_pytest_commands() -> None:
    workflow = _workflow_text()
    step_names = tuple(re.findall(r"^      - name: (.+)$", workflow, re.MULTILINE))
    initialize = _workflow_step(workflow, EXPECTED_STEP_NAMES[2])
    dependencies = _workflow_step(workflow, EXPECTED_STEP_NAMES[3])
    platform_tests = _workflow_step(workflow, EXPECTED_STEP_NAMES[5])

    assert step_names == EXPECTED_STEP_NAMES
    assert _step_command(initialize) == (
        "python .github/scripts/task601_process_tree_evidence.py --initialize "
        '--evidence-name "${{ matrix.evidence_name }}" '
        '--output "$RUNNER_TEMP/task-601-platform-evidence.json"'
    )
    assert _yaml_scalar(dependencies, 8, "id") == "dependencies"
    assert _yaml_scalar(dependencies, 8, "continue-on-error") == "true"
    assert _step_command(dependencies) == "python -m pip install -e '.[dev]'"
    assert workflow.count("pip install") == 1
    assert re.findall(r"\.\[([^]]+)]", workflow) == ["dev"]
    assert _yaml_scalar(platform_tests, 8, "id") == "platform_tests"
    assert _yaml_scalar(platform_tests, 8, "continue-on-error") == "true"
    assert _yaml_scalar(platform_tests, 8, "if") == (
        "steps.dependencies.outcome == 'success'"
    )
    assert _step_command(platform_tests) == (
        "python -m pytest Tests/STT/test_executor_process_tree.py "
        "Tests/STT/test_local_stt_executor.py::"
        "test_force_stop_detaches_before_kill_and_cleans_generation_scratch "
        "Tests/CI/test_task601_process_tree_evidence.py --timeout=60 "
        '--junitxml="$RUNNER_TEMP/task-601-junit.xml" -q'
    )


def test_workflow_preserves_failures_and_always_uploads_only_the_json() -> None:
    workflow = _workflow_text()
    dependency_failure = _workflow_step(workflow, EXPECTED_STEP_NAMES[4])
    normalize = _workflow_step(workflow, EXPECTED_STEP_NAMES[6])
    validate = _workflow_step(workflow, EXPECTED_STEP_NAMES[7])
    upload = _workflow_step(workflow, EXPECTED_STEP_NAMES[8])

    assert _yaml_scalar(dependency_failure, 8, "if") == (
        "steps.dependencies.outcome != 'success'"
    )
    assert _step_command(dependency_failure) == (
        "python .github/scripts/task601_process_tree_evidence.py "
        "--record-failure dependency_install --failure-stage dependency_install "
        '--evidence-name "${{ matrix.evidence_name }}" '
        '--output "$RUNNER_TEMP/task-601-platform-evidence.json"'
    )
    assert _yaml_scalar(normalize, 8, "if") == (
        "always() && steps.dependencies.outcome == 'success'"
    )
    assert _step_command(normalize) == (
        "python .github/scripts/task601_process_tree_evidence.py "
        '--from-junit "$RUNNER_TEMP/task-601-junit.xml" '
        '--pytest-outcome "${{ steps.platform_tests.outcome }}" '
        '--evidence-name "${{ matrix.evidence_name }}" '
        '--output "$RUNNER_TEMP/task-601-platform-evidence.json"'
    )
    assert _yaml_scalar(validate, 8, "if") == "always()"
    assert "continue-on-error" not in validate
    assert _step_command(validate) == (
        "python .github/scripts/task601_process_tree_evidence.py "
        '--validate "$RUNNER_TEMP/task-601-platform-evidence.json"'
    )
    assert _yaml_scalar(upload, 8, "if") == "always()"
    assert _yaml_scalar(upload, 8, "uses") == "actions/upload-artifact@v4"
    assert _nonempty_lines(_yaml_block(upload, "        with:")) == [
        "          name: task-601-platform-${{ matrix.evidence_name }}",
        "          path: ${{ runner.temp }}/task-601-platform-evidence.json",
        "          if-no-files-found: error",
    ]


def test_evidence_script_exists_and_loads() -> None:
    evidence = _load_evidence()

    assert evidence.REQUIRED_NODES == REQUIRED_NODES
    assert evidence.EXPECTED_PLATFORMS == EXPECTED_PLATFORMS


def test_current_run_identity_uses_checked_out_commit_not_github_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_evidence()
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "987")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")

    result = evidence.current_run_identity()

    assert result == _run_identity(commit=expected, run_id="987", attempt="3")
    assert re.fullmatch(r"[0-9a-f]{40}", result["tested_commit"])
    assert result["tested_commit"] != "b" * 40


@pytest.mark.parametrize(
    ("evidence_name", "system", "machine", "expected_architecture"),
    (
        ("linux-x86_64", "Linux", "x86_64", "x86_64"),
        ("windows-x86_64", "Windows", "AMD64", "x86_64"),
        ("windows-x86_64", "Windows", "x86_64", "x86_64"),
        ("macos-x86_64", "Darwin", "x86_64", "x86_64"),
    ),
)
def test_evidence_names_derive_only_the_exact_expected_host(
    monkeypatch: pytest.MonkeyPatch,
    evidence_name: str,
    system: str,
    machine: str,
    expected_architecture: str,
) -> None:
    evidence = _load_evidence()
    monkeypatch.setattr(evidence.platform, "system", lambda: system)
    monkeypatch.setattr(evidence.platform, "machine", lambda: machine)
    monkeypatch.setattr(evidence.platform, "python_version", lambda: "3.12.9")

    assert evidence._host_result(evidence_name) == {
        "system": system,
        "architecture": expected_architecture,
        "python": "3.12.9",
    }


@pytest.mark.parametrize(
    ("evidence_name", "system", "machine"),
    (
        ("linux-x86_64", "Windows", "AMD64"),
        ("windows-x86_64", "Linux", "x86_64"),
        ("macos-x86_64", "Darwin", "arm64"),
        ("linux-x86_64", "Linux", "AMD64"),
        ("unknown-x86_64", "Linux", "x86_64"),
    ),
)
def test_evidence_name_rejects_host_or_architecture_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    evidence_name: str,
    system: str,
    machine: str,
) -> None:
    evidence = _load_evidence()
    monkeypatch.setattr(evidence.platform, "system", lambda: system)
    monkeypatch.setattr(evidence.platform, "machine", lambda: machine)

    with pytest.raises(ValueError, match="host|evidence_name"):
        evidence._host_result(evidence_name)


def test_passing_junit_and_successful_pytest_outcome_produce_passed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)

    result = evidence.result_from_junit(
        _passing_junit(tmp_path),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result == _valid_result("linux-x86_64")
    evidence.validate_result(result)


@pytest.mark.parametrize("pytest_outcome", ("failure", "cancelled", "skipped"))
def test_non_success_pytest_outcome_never_passes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytest_outcome: str,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)

    result = evidence.result_from_junit(
        _passing_junit(tmp_path),
        pytest_outcome=pytest_outcome,
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "test_execution"
    assert result["failure_stage"] == "test_execution"
    assert result["pytest"]["outcome"] == pytest_outcome
    with pytest.raises(ValueError, match="did not pass"):
        evidence.validate_result(result)


@pytest.mark.parametrize("failure_element", ("failure", "error"))
def test_any_selected_test_failure_prevents_passing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_element: str,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    non_required_failure = (
        '<testcase classname="Tests.CI.test_task601_process_tree_evidence" '
        'name="test_unrelated_selected_contract" time="0.1">'
        f"<{failure_element}>private traceback</{failure_element}></testcase>"
    )
    junit = _write_junit(
        tmp_path,
        _junit(
            *(_junit_case(node) for node in REQUIRED_NODES),
            non_required_failure,
            duration="0.85",
        ),
    )

    result = evidence.result_from_junit(
        junit,
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "test_execution"
    assert "private traceback" not in json.dumps(result)


@pytest.mark.parametrize(
    ("replacement", "expected_node_outcome"),
    (
        (None, None),
        ("<skipped />", "skipped"),
        ("<failure>private path /tmp/failure</failure>", "failed"),
    ),
)
def test_missing_skipped_or_failed_required_node_is_test_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str | None,
    expected_node_outcome: str | None,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    cases = [_junit_case(node) for node in REQUIRED_NODES[1:]]
    if replacement is not None:
        cases.insert(0, _junit_case(REQUIRED_NODES[0], replacement))

    result = evidence.result_from_junit(
        _write_junit(tmp_path, _junit(*cases)),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "test_execution"
    if expected_node_outcome is None:
        assert REQUIRED_NODES[0] not in result["pytest"]["required_nodes"]
    else:
        assert (
            result["pytest"]["required_nodes"][REQUIRED_NODES[0]]
            == expected_node_outcome
        )


def test_duplicate_or_parameterized_required_node_is_test_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    exact_duplicate = _write_junit(
        tmp_path,
        _junit(
            *(_junit_case(node) for node in REQUIRED_NODES),
            _junit_case(REQUIRED_NODES[0]),
            duration="1.0",
        ),
    )
    duplicate_result = evidence.result_from_junit(
        exact_duplicate,
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )
    parameterized = REQUIRED_NODES[0] + "[unexpected]"
    parameterized_cases = [_junit_case(parameterized)] + [
        _junit_case(node) for node in REQUIRED_NODES[1:]
    ]

    parameterized_result = evidence.result_from_junit(
        _write_junit(tmp_path, _junit(*parameterized_cases)),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert duplicate_result["status"] == "failed"
    assert duplicate_result["failure_code"] == "test_execution"
    assert parameterized_result["status"] == "failed"
    assert parameterized_result["failure_code"] == "test_execution"


@pytest.mark.parametrize("alias_kind", ("wrong_module", "parameterized"))
def test_required_node_alias_cannot_hide_beside_the_exact_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    _, required_name = REQUIRED_NODES[0].split("::", 1)
    if alias_kind == "wrong_module":
        alias = (
            '<testcase classname="Tests.Other.test_executor_process_tree" '
            f'name="{required_name}" time="0.1" />'
        )
    else:
        alias = _junit_case(REQUIRED_NODES[0] + "[unexpected]")

    result = evidence.result_from_junit(
        _write_junit(
            tmp_path,
            _junit(*(_junit_case(node) for node in REQUIRED_NODES), alias),
        ),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "test_execution"


@pytest.mark.parametrize(
    "file_value",
    (
        "/Users/runner/work/private/test_executor_process_tree.py",
        r"C:\a\private\test_executor_process_tree.py",
    ),
    ids=("posix-private-path", "windows-private-path"),
)
def test_junit_file_is_never_authority_and_wrong_module_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_value: str,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    passing = evidence.result_from_junit(
        _passing_junit(tmp_path, file_value=file_value),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )
    _, required_name = REQUIRED_NODES[0].split("::", 1)
    wrong_module_case = (
        f'<testcase classname="Tests.Other.test_executor_process_tree" '
        f'name="{required_name}" file="Tests/STT/test_executor_process_tree.py" '
        'time="0.25" />'
    )
    wrong_module = evidence.result_from_junit(
        _write_junit(
            tmp_path,
            _junit(
                wrong_module_case,
                *(_junit_case(node) for node in REQUIRED_NODES[1:]),
            ),
        ),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    serialized = json.dumps(passing)
    assert passing["status"] == "passed"
    assert file_value not in serialized
    assert "file" not in serialized
    assert wrong_module["status"] == "failed"
    assert wrong_module["failure_code"] == "test_execution"


@pytest.mark.parametrize(
    "xml",
    (
        "<not-closed",
        "<testsuite><testcase",
    ),
)
def test_malformed_junit_produces_bounded_test_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    xml: str,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)

    result = evidence.result_from_junit(
        _write_junit(tmp_path, xml),
        pytest_outcome="failure",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "test_execution"
    assert result["pytest"]["required_nodes"] == {}
    assert str(tmp_path) not in json.dumps(result)


def test_initialized_and_dependency_failure_documents_are_bounded_but_validate_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    output = tmp_path / "result.json"

    assert (
        evidence.main(
            [
                "--initialize",
                "--evidence-name",
                "linux-x86_64",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    initialized = json.loads(output.read_text(encoding="utf-8"))
    evidence.validate_result(initialized, require_pass=False)
    assert evidence.main(["--validate", str(output)]) == 1
    assert initialized["failure_code"] == "not_run"
    assert initialized["pytest"]["required_nodes"] == {}

    assert (
        evidence.main(
            [
                "--record-failure",
                "dependency_install",
                "--failure-stage",
                "dependency_install",
                "--evidence-name",
                "linux-x86_64",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    failed = json.loads(output.read_text(encoding="utf-8"))
    evidence.validate_result(failed, require_pass=False)
    assert evidence.main(["--validate", str(output)]) == 1
    assert failed["failure_code"] == "dependency_install"
    assert str(tmp_path) not in json.dumps(failed)


@pytest.mark.parametrize(
    ("failure_code", "failure_stage"),
    (("runneradmin", "initialize"), ("dependency_install", "runneradmin")),
)
def test_failure_documents_accept_only_stable_codes_and_stages(
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
    failure_stage: str,
) -> None:
    evidence = _load_evidence()
    monkeypatch.setattr(
        evidence,
        "_host_result",
        lambda _name: {
            "system": "Linux",
            "architecture": "x86_64",
            "python": "3.12.9",
        },
    )

    with pytest.raises(ValueError, match="failure_code|failure_stage"):
        evidence.failure_result(
            _run_identity(),
            evidence_name="linux-x86_64",
            failure_code=failure_code,
            failure_stage=failure_stage,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("debug", "/Users/runner/private.xml"),
        ("debug", "opened /opt/private/report.xml"),
        ("debug", r"C:\Users\runner\private.xml"),
        ("debug", r"\\server\share\private.xml"),
        ("pid", "4242"),
        ("handle", "0x1af"),
        ("username", "runneradmin"),
        ("command", "python -m pytest"),
    ),
    ids=(
        "posix-home-path",
        "posix-opt-path",
        "windows-user-path",
        "unc-path",
        "pid",
        "handle",
        "username",
        "command",
    ),
)
def test_result_validation_rejects_path_process_and_user_material(
    key: str, value: str
) -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")
    result[key] = value

    with pytest.raises(ValueError, match="private|path|field"):
        evidence.validate_result(result)


def test_result_validation_rejects_off_repository_run_url() -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")
    result["run"]["workflow_run_url"] = (
        "https://github.com/another-owner/another-repo/actions/runs/123"
    )

    with pytest.raises(ValueError, match="workflow_run_url|slash"):
        evidence.validate_result(result)


def test_exact_three_matching_platform_documents_aggregate() -> None:
    evidence = _load_evidence()
    results = [_valid_result(name) for name in EXPECTED_PLATFORMS]
    for attempt, result in enumerate(results, start=1):
        result["run"]["workflow_run_attempt"] = str(attempt)

    aggregate = evidence.aggregate_results(results)

    assert aggregate == {
        "schema_version": 1,
        "evidence_label": "task601_native_process_tree_matrix",
        "status": "passed",
        "run": {
            "tested_commit": "a" * 40,
            "workflow_run_id": "123",
            "workflow_run_url": (
                "https://github.com/rmusser01/tldw_chatbook/actions/runs/123"
            ),
        },
        "platforms": {result["evidence_name"]: result for result in results},
    }
    evidence.validate_aggregate(aggregate)


@pytest.mark.parametrize(
    "identity_field",
    ("tested_commit", "workflow_run_id", "workflow_run_url"),
    ids=("commit", "run", "url"),
)
def test_aggregate_rejects_commit_or_run_mismatch(identity_field: str) -> None:
    evidence = _load_evidence()
    results = [_valid_result(name) for name in EXPECTED_PLATFORMS]
    replacement = (
        "b" * 40
        if identity_field == "tested_commit"
        else (
            "https://github.com/rmusser01/tldw_chatbook/actions/runs/456"
            if identity_field == "workflow_run_url"
            else "456"
        )
    )
    results[1]["run"][identity_field] = replacement
    if identity_field == "workflow_run_id":
        results[1]["run"]["workflow_run_url"] = (
            "https://github.com/rmusser01/tldw_chatbook/actions/runs/456"
        )

    with pytest.raises(ValueError, match=r"commit|workflow.run"):
        evidence.aggregate_results(results)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda aggregate: aggregate["platforms"]["linux-x86_64"]["host"].update(
                system="Darwin"
            ),
            "host",
        ),
        (
            lambda aggregate: aggregate["platforms"].pop("linux-x86_64"),
            "platform",
        ),
        (
            lambda aggregate: aggregate["platforms"].update(
                {"extra-x86_64": _valid_result("linux-x86_64")}
            ),
            "platform",
        ),
        (
            lambda aggregate: aggregate["platforms"]["linux-x86_64"]["pytest"][
                "required_nodes"
            ].update({REQUIRED_NODES[0]: "failed"}),
            "passed",
        ),
        (
            lambda aggregate: aggregate["platforms"]["linux-x86_64"]["pytest"][
                "required_nodes"
            ].update({REQUIRED_NODES[0]: "skipped"}),
            "passed",
        ),
        (lambda aggregate: aggregate.update(unreviewed=True), "field"),
        (
            lambda aggregate: aggregate.update(debug="/private/platform/report"),
            "path|slash",
        ),
    ),
)
def test_validate_aggregate_rejects_invalid_platform_matrix(
    mutation: object, expected: str
) -> None:
    evidence = _load_evidence()
    aggregate = evidence.aggregate_results(
        [_valid_result(name) for name in EXPECTED_PLATFORMS]
    )
    mutation(aggregate)

    with pytest.raises(ValueError, match=expected):
        evidence.validate_aggregate(aggregate)


def test_validate_aggregate_rejects_aggregate_identity_mismatch() -> None:
    evidence = _load_evidence()
    aggregate = evidence.aggregate_results(
        [_valid_result(name) for name in EXPECTED_PLATFORMS]
    )
    aggregate["run"]["workflow_run_id"] = "456"
    aggregate["run"]["workflow_run_url"] = (
        "https://github.com/rmusser01/tldw_chatbook/actions/runs/456"
    )

    with pytest.raises(ValueError, match="workflow run"):
        evidence.validate_aggregate(aggregate)


def test_validate_aggregate_rejects_unknown_nested_key() -> None:
    evidence = _load_evidence()
    aggregate = evidence.aggregate_results(
        [_valid_result(name) for name in EXPECTED_PLATFORMS]
    )
    aggregate["platforms"]["linux-x86_64"]["host"]["runner"] = "hosted"

    with pytest.raises(ValueError, match="field"):
        evidence.validate_aggregate(aggregate)


def test_aggregate_cli_writes_sorted_atomic_json(tmp_path: Path) -> None:
    evidence = _load_evidence()
    inputs: list[str] = []
    for name in EXPECTED_PLATFORMS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_valid_result(name)), encoding="utf-8")
        inputs.append(str(path))
    output = tmp_path / "aggregate.json"

    assert evidence.main(["--aggregate", *inputs, "--output", str(output)]) == 0
    assert evidence.main(["--validate-aggregate", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert not output.with_name(f".{output.name}.tmp").exists()
    assert output.read_text(encoding="utf-8").startswith('{\n  "evidence_label"')


def test_cli_process_rejects_multiple_operation_modes(tmp_path: Path) -> None:
    input_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(_valid_result("linux-x86_64")), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(EVIDENCE_PATH),
            "--validate",
            str(input_path),
            "--validate-aggregate",
            str(input_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0


@pytest.mark.parametrize("destination_exists", (True, False))
def test_cli_rejects_multiple_modes_without_writing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_exists: bool,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    destination = tmp_path / "result.json"
    if destination_exists:
        destination.write_text("sentinel", encoding="utf-8")
    junit = _passing_junit(tmp_path)
    extra_mode = (
        ["--from-junit", str(junit), "--pytest-outcome", "success"]
        if destination_exists
        else ["--record-failure", "dependency_install"]
    )

    code = _main_code(
        evidence,
        [
            "--initialize",
            "--evidence-name",
            "linux-x86_64",
            "--output",
            str(destination),
            *extra_mode,
        ],
    )

    assert code != 0
    if destination_exists:
        assert destination.read_text(encoding="utf-8") == "sentinel"
    else:
        assert not destination.exists()


@pytest.mark.parametrize("mode", ("initialize", "validate"))
def test_cli_rejects_irrelevant_mode_arguments_without_writing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    destination = tmp_path / "destination.json"
    if mode == "initialize":
        destination.write_text("sentinel", encoding="utf-8")
        args = [
            "--initialize",
            "--pytest-outcome",
            "success",
            "--evidence-name",
            "linux-x86_64",
            "--output",
            str(destination),
        ]
    else:
        input_path = tmp_path / "result.json"
        input_path.write_text(
            json.dumps(_valid_result("linux-x86_64")), encoding="utf-8"
        )
        args = ["--validate", str(input_path), "--output", str(destination)]

    code = _main_code(evidence, args)

    assert code != 0
    if mode == "initialize":
        assert destination.read_text(encoding="utf-8") == "sentinel"
    else:
        assert not destination.exists()


def test_cli_accepts_each_exact_documented_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _load_evidence()
    _stub_linux_evidence(evidence, monkeypatch)
    platform_output = tmp_path / "platform.json"
    junit = _passing_junit(tmp_path)

    assert (
        evidence.main(
            [
                "--initialize",
                "--evidence-name",
                "linux-x86_64",
                "--output",
                str(platform_output),
            ]
        )
        == 0
    )
    assert (
        evidence.main(
            [
                "--record-failure",
                "dependency_install",
                "--failure-stage",
                "dependency_install",
                "--evidence-name",
                "linux-x86_64",
                "--output",
                str(platform_output),
            ]
        )
        == 0
    )
    assert (
        evidence.main(
            [
                "--from-junit",
                str(junit),
                "--pytest-outcome",
                "success",
                "--evidence-name",
                "linux-x86_64",
                "--output",
                str(platform_output),
            ]
        )
        == 0
    )
    assert evidence.main(["--validate", str(platform_output)]) == 0

    inputs: list[str] = []
    for name in EXPECTED_PLATFORMS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_valid_result(name)), encoding="utf-8")
        inputs.append(str(path))
    aggregate_output = tmp_path / "aggregate.json"
    assert (
        evidence.main(["--aggregate", *inputs, "--output", str(aggregate_output)]) == 0
    )
    assert evidence.main(["--validate-aggregate", str(aggregate_output)]) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 2),
        ("evidence_label", "wrong_label"),
        ("evidence_name", "unknown"),
        ("status", "skipped"),
    ),
)
def test_result_schema_rejects_unknown_fixed_values(field: str, value: object) -> None:
    evidence = _load_evidence()
    result = copy.deepcopy(_valid_result("linux-x86_64"))
    result[field] = value

    with pytest.raises(ValueError):
        evidence.validate_result(result)


def test_schema_version_rejects_boolean_alias_for_one() -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")
    result["schema_version"] = True

    with pytest.raises(ValueError, match="schema_version"):
        evidence.validate_result(result)


@pytest.mark.parametrize(
    "duration",
    (
        True,
        None,
        "0.75",
        -1,
        1_200.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        10**309,
    ),
    ids=(
        "bool",
        "null",
        "string",
        "negative",
        "too-large",
        "nan",
        "positive-inf",
        "negative-inf",
        "huge-int",
    ),
)
@pytest.mark.parametrize("document_kind", ("result", "aggregate"))
def test_duration_validation_rejects_every_non_bounded_number(
    duration: object, document_kind: str
) -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")
    result["pytest"]["duration_seconds"] = duration
    if document_kind == "result":
        document = result
        validator = evidence.validate_result
    else:
        document = evidence.aggregate_results(
            [_valid_result(name) for name in EXPECTED_PLATFORMS]
        )
        document["platforms"]["linux-x86_64"] = result
        validator = evidence.validate_aggregate

    with pytest.raises(ValueError, match="duration_seconds"):
        validator(document)


@pytest.mark.parametrize(
    ("flag", "document_kind"),
    (("--validate", "result"), ("--validate-aggregate", "aggregate")),
)
def test_cli_huge_duration_is_bounded_and_does_not_modify_files(
    tmp_path: Path, flag: str, document_kind: str
) -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")
    result["pytest"]["duration_seconds"] = 10**309
    if document_kind == "result":
        document = result
    else:
        document = evidence.aggregate_results(
            [_valid_result(name) for name in EXPECTED_PLATFORMS]
        )
        document["platforms"]["linux-x86_64"] = result
    input_path = tmp_path / f"{document_kind}.json"
    original = json.dumps(document)
    input_path.write_text(original, encoding="utf-8")
    destination = tmp_path / "output.json"

    completed = subprocess.run(
        [sys.executable, str(EVIDENCE_PATH), flag, str(input_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    assert str(PROJECT_ROOT) not in completed.stderr
    assert str(tmp_path) not in completed.stderr
    assert input_path.read_text(encoding="utf-8") == original
    assert not destination.exists()
