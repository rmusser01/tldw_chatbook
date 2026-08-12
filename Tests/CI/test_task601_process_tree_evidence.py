from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / ".github/scripts/task601_process_tree_evidence.py"
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


def _load_evidence() -> ModuleType:
    assert EVIDENCE_PATH.is_file(), "TASK-601 evidence normalizer is missing"
    spec = importlib.util.spec_from_file_location("task601_evidence", EVIDENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
