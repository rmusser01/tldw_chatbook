from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / ".github/scripts/task603_dictation_evidence.py"
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/task-603-platform-evidence.yml"
REQUIRED_NODES = (
    "Tests/STT/test_dispatch_coordinator.py::test_pcm_byte_limit_is_derived_from_the_single_sixty_second_ceiling",
    "Tests/STT/test_dispatch_coordinator.py::test_waiting_segments_coalesce_into_one_source_with_ordered_boundaries",
    "Tests/STT/test_dispatch_coordinator.py::test_pending_cancel_clears_gate_once_without_preempting_batch",
    "Tests/STT/test_dispatch_coordinator.py::test_processing_thread_exits_within_join_bound_while_dictation_waits",
    "Tests/Library/test_library_ingest_runner.py::test_dictation_reservation_gates_only_heavy_library_work",
    "Tests/Library/test_library_ingest_runner.py::test_library_terminal_hands_executor_to_pending_dictation_before_top_up",
    "Tests/Library/test_library_ingest_runner.py::test_shutdown_cooperatively_cancels_active_dictation_before_executor_close",
    "Tests/STT/test_transcription_service_facade.py::test_parakeet_streaming_reports_unsupported_without_consulting_the_bridge",
    "Tests/UI/test_console_dictation.py::test_console_mic_has_strict_wall_timer_and_visible_limit_transition",
    "Tests/UI/test_console_hands_free_wiring.py::test_hands_free_limit_exits_without_reopen_until_a_physical_mic_press",
)
EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "linux-aarch64": ("Linux", "aarch64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-arm64": ("Darwin", "arm64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}


def _load_evidence() -> ModuleType:
    assert EVIDENCE_PATH.is_file(), "TASK-603 evidence normalizer is missing"
    spec = importlib.util.spec_from_file_location("task603_evidence", EVIDENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_identity(
    *, commit: str = "a" * 40, run_id: str = "603", attempt: str = "1"
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
        "evidence_label": "task603_bounded_dictation",
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
            "duration_seconds": 1.25,
            "required_nodes": dict.fromkeys(REQUIRED_NODES, "passed"),
        },
    }


def _junit_case(node: str, body: str = "") -> str:
    path, name = node.split("::", 1)
    classname = path.removesuffix(".py").replace("/", ".")
    return (
        f'<testcase classname="{classname}" name="{name}" time="0.1">{body}</testcase>'
    )


def _junit(*cases: str, duration: str = "1.0") -> str:
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


def _passing_junit(tmp_path: Path) -> Path:
    return _write_junit(
        tmp_path,
        _junit(*(_junit_case(node) for node in REQUIRED_NODES)),
    )


def _stub_host(evidence: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_constants_pin_exact_required_nodes_and_platforms() -> None:
    evidence = _load_evidence()

    assert evidence.REQUIRED_NODES == REQUIRED_NODES
    assert evidence.EXPECTED_PLATFORMS == EXPECTED_PLATFORMS


def test_initialize_is_a_valid_bounded_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_evidence()
    _stub_host(evidence, monkeypatch)

    result = evidence.failure_result(
        _run_identity(),
        evidence_name="linux-x86_64",
        failure_code="not_run",
        failure_stage="initialize",
    )

    evidence.validate_result(result, require_pass=False)
    assert result["status"] == "failed"
    assert result["pytest"]["required_nodes"] == {}


def test_passing_junit_requires_every_exact_node_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_evidence()
    _stub_host(evidence, monkeypatch)

    result = evidence.result_from_junit(
        _passing_junit(tmp_path),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    evidence.validate_result(result, require_pass=True)
    assert result["status"] == "passed"
    assert result["pytest"]["required_nodes"] == dict.fromkeys(REQUIRED_NODES, "passed")


@pytest.mark.parametrize("problem", ["missing", "duplicate", "failed", "skipped"])
def test_junit_non_vacuity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    evidence = _load_evidence()
    _stub_host(evidence, monkeypatch)
    cases = [_junit_case(node) for node in REQUIRED_NODES]
    if problem == "missing":
        cases.pop()
    elif problem == "duplicate":
        cases.append(cases[-1])
    elif problem == "failed":
        cases[-1] = _junit_case(
            REQUIRED_NODES[-1], "<failure>no details retained</failure>"
        )
    else:
        cases[-1] = _junit_case(REQUIRED_NODES[-1], "<skipped />")

    result = evidence.result_from_junit(
        _write_junit(tmp_path, _junit(*cases)),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    evidence.validate_result(result, require_pass=False)
    assert result["status"] == "failed"
    assert result["failure_code"] == "test_execution"


def test_wrong_module_with_required_name_is_not_trusted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_evidence()
    _stub_host(evidence, monkeypatch)
    cases = [_junit_case(node) for node in REQUIRED_NODES[:-1]]
    _, required_name = REQUIRED_NODES[-1].split("::", 1)
    cases.append(
        f'<testcase classname="Tests.wrong_module" name="{required_name}" time="0.1" />'
    )

    result = evidence.result_from_junit(
        _write_junit(tmp_path, _junit(*cases)),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"


@pytest.mark.parametrize("pytest_outcome", ["failure", "cancelled", "skipped"])
def test_successful_nodes_cannot_override_non_successful_pytest_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytest_outcome: str,
) -> None:
    evidence = _load_evidence()
    _stub_host(evidence, monkeypatch)

    result = evidence.result_from_junit(
        _passing_junit(tmp_path),
        pytest_outcome=pytest_outcome,
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"


@pytest.mark.parametrize("duration", ["-1", "1801", "nan", "inf", "9" * 310])
def test_junit_duration_is_finite_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration: str,
) -> None:
    evidence = _load_evidence()
    _stub_host(evidence, monkeypatch)
    xml = _junit(*(_junit_case(node) for node in REQUIRED_NODES), duration=duration)

    result = evidence.result_from_junit(
        _write_junit(tmp_path, xml),
        pytest_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("path", "private/model"),
        ("traceback", "secret"),
        ("note", "/Users/example/private"),
        ("note", "C:\\Users\\example\\private"),
    ],
)
def test_validation_rejects_recursive_private_keys_and_paths(
    key: str,
    value: str,
) -> None:
    evidence = _load_evidence()

    with pytest.raises(ValueError):
        evidence._validate_privacy({"nested": {"deeper": {key: value}}})


def test_validation_allows_only_canonical_url_and_required_node_slashes() -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")

    evidence.validate_result(result, require_pass=True)


def test_canonical_run_url_prefix_cannot_hide_a_private_suffix() -> None:
    evidence = _load_evidence()

    with pytest.raises(ValueError):
        evidence._validate_privacy(
            {
                "note": (
                    "https://github.com/rmusser01/tldw_chatbook/actions/runs/603 "
                    "/Users/example/private"
                )
            }
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["host"].update(architecture="arm64"),
        lambda result: result["run"].update(tested_commit="bad"),
        lambda result: result["pytest"].update(outcome="failure"),
        lambda result: result["pytest"].update(duration_seconds=True),
        lambda result: result.update(failure_code="test_execution"),
    ],
)
def test_passed_result_schema_rejects_mismatched_fields(mutation) -> None:
    evidence = _load_evidence()
    result = _valid_result("linux-x86_64")
    mutation(result)

    with pytest.raises(ValueError):
        evidence.validate_result(result, require_pass=True)


def test_aggregate_requires_all_five_platforms_and_same_run(tmp_path: Path) -> None:
    evidence = _load_evidence()
    results = []
    for name in EXPECTED_PLATFORMS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_valid_result(name)), encoding="utf-8")
        results.append(path)

    aggregate = evidence.aggregate_results(results)

    evidence.validate_aggregate(aggregate)
    assert aggregate["status"] == "passed"
    assert list(aggregate["platforms"]) == list(EXPECTED_PLATFORMS)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "commit", "run", "failed"])
def test_aggregate_fails_closed_on_matrix_mismatch(
    tmp_path: Path,
    problem: str,
) -> None:
    evidence = _load_evidence()
    documents = [_valid_result(name) for name in EXPECTED_PLATFORMS]
    if problem == "missing":
        documents.pop()
    elif problem == "duplicate":
        documents[-1] = copy.deepcopy(documents[0])
    elif problem == "commit":
        documents[-1]["run"]["tested_commit"] = "b" * 40
    elif problem == "run":
        documents[-1]["run"]["workflow_run_id"] = "604"
        documents[-1]["run"]["workflow_run_url"] = (
            "https://github.com/rmusser01/tldw_chatbook/actions/runs/604"
        )
    else:
        documents[-1]["status"] = "failed"
        documents[-1]["failure_code"] = "test_execution"
        documents[-1]["failure_stage"] = "test_execution"
        documents[-1]["pytest"]["outcome"] = "failure"
    paths = []
    for index, document in enumerate(documents):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError):
        evidence.aggregate_results(paths)


def test_cli_modes_are_mutually_exclusive_and_preserve_destination(
    tmp_path: Path,
) -> None:
    _load_evidence()
    destination = tmp_path / "sentinel.json"
    destination.write_text("sentinel", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(EVIDENCE_PATH),
            "--initialize",
            "--validate",
            str(destination),
            "--evidence-name",
            "linux-x86_64",
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert destination.read_text(encoding="utf-8") == "sentinel"


def test_cli_validate_is_silent_and_does_not_rewrite_input(tmp_path: Path) -> None:
    evidence = _load_evidence()
    path = tmp_path / "result.json"
    before = json.dumps(_valid_result("linux-x86_64"), indent=2) + "\n"
    path.write_text(before, encoding="utf-8")

    assert _main_code(evidence, ["--validate", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == before


def test_cli_aggregate_writes_only_a_validated_matrix(tmp_path: Path) -> None:
    evidence = _load_evidence()
    inputs = []
    for name in EXPECTED_PLATFORMS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_valid_result(name)), encoding="utf-8")
        inputs.append(path)
    output = tmp_path / "aggregate.json"

    code = _main_code(
        evidence,
        ["--aggregate", *(str(path) for path in inputs), "--output", str(output)],
    )

    assert code == 0
    evidence.validate_aggregate(json.loads(output.read_text(encoding="utf-8")))


def test_workflow_file_is_deferred_to_the_workflow_tdd_phase() -> None:
    assert not WORKFLOW_PATH.exists()
