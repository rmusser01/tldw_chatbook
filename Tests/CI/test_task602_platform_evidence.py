from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / ".github/scripts/task602_platform_evidence.py"
SMOKE_PATH = PROJECT_ROOT / ".github/scripts/task602_platform_smoke.py"
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/task-602-platform-evidence.yml"

EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "linux-aarch64": ("Linux", "aarch64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-arm64": ("Darwin", "arm64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}
REQUIRED_CHECKS = (
    "package_resolution",
    "runtime_probe",
    "v2_int8_cpu",
    "v3_int8_cpu",
    "long_form_vad",
    "cancellation",
    "batch_reuse",
    "retry_wiring",
)
V2_REF = {
    "artifact_id": "parakeet-v2",
    "revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
    "variant": "int8",
}
V3_REF = {
    "artifact_id": "parakeet-v3",
    "revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
    "variant": "int8",
}
VAD_REF = {
    "artifact_id": "silero-vad",
    "revision": "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    "variant": "f32",
}

EXPECTED_STEPS = (
    "Check out the exact tested commit",
    "Set up Python",
    "Initialize failure evidence",
    "Install Parakeet evidence dependencies",
    "Record dependency installation failure",
    "Run bounded native Parakeet smoke",
    "Normalize native smoke evidence",
    "Validate platform evidence",
    "Upload platform evidence",
)


def _load(path: Path, name: str) -> ModuleType:
    assert path.is_file(), f"TASK-602 {name} is missing"
    spec = importlib.util.spec_from_file_location(f"task602_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow_text() -> str:
    assert WORKFLOW_PATH.is_file(), "TASK-602 platform workflow is missing"
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


def _valid_smoke_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "passed",
        "failure_code": None,
        "failure_stage": None,
        "packages": {
            "onnx-asr": "0.12.0",
            "onnxruntime": "1.27.0",
            "faster-whisper": "1.2.1",
            "ctranslate2": "4.8.1",
        },
        "execution_provider": "CPUExecutionProvider",
        "artifacts": {
            "v2_int8": {
                "reference": V2_REF,
                "closure_fingerprint": "1" * 64,
            },
            "v3_int8": {
                "reference": V3_REF,
                "closure_fingerprint": "2" * 64,
            },
            "vad": VAD_REF,
        },
        "checks": dict.fromkeys(REQUIRED_CHECKS, "passed"),
        "durations_seconds": {
            "acquisition": 10.0,
            "v2_int8_cpu": 1.0,
            "v3_int8_cpu": 1.1,
            "long_form_vad": 2.0,
            "total": 15.0,
        },
        "cleanup": "passed",
    }


def _valid_result(evidence_name: str = "linux-x86_64") -> dict[str, object]:
    system, architecture = EXPECTED_PLATFORMS[evidence_name]
    return {
        "schema_version": 1,
        "evidence_label": "task602_native_parakeet",
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
        **{
            key: copy.deepcopy(value)
            for key, value in _valid_smoke_payload().items()
            if key not in {"schema_version", "status", "failure_code", "failure_stage"}
        },
    }


def _valid_failure() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_label": "task602_native_parakeet",
        "evidence_name": "linux-x86_64",
        "status": "failed",
        "failure_code": "dependency_install",
        "failure_stage": "dependency_install",
        "run": _run_identity(),
        "host": {
            "system": "Linux",
            "architecture": "x86_64",
            "python": "3.12.9",
        },
    }


def _stub_linux(evidence: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_scripts_and_workflow_exist() -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    smoke = _load(SMOKE_PATH, "smoke")

    assert evidence.EXPECTED_PLATFORMS == EXPECTED_PLATFORMS
    assert evidence.REQUIRED_CHECKS == REQUIRED_CHECKS
    assert smoke.FIXTURE_SHA256 == (
        "c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14"
    )
    assert _workflow_text()


@pytest.mark.parametrize(
    ("evidence_name", "system", "machine", "architecture"),
    (
        ("linux-x86_64", "Linux", "x86_64", "x86_64"),
        ("linux-aarch64", "Linux", "aarch64", "aarch64"),
        ("windows-x86_64", "Windows", "AMD64", "x86_64"),
        ("macos-arm64", "Darwin", "arm64", "arm64"),
        ("macos-x86_64", "Darwin", "x86_64", "x86_64"),
    ),
)
def test_evidence_name_binds_to_exact_native_host(
    monkeypatch: pytest.MonkeyPatch,
    evidence_name: str,
    system: str,
    machine: str,
    architecture: str,
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    monkeypatch.setattr(evidence.platform, "system", lambda: system)
    monkeypatch.setattr(evidence.platform, "machine", lambda: machine)
    monkeypatch.setattr(evidence.platform, "python_version", lambda: "3.12.9")

    assert evidence._host_result(evidence_name) == {
        "system": system,
        "architecture": architecture,
        "python": "3.12.9",
    }


@pytest.mark.parametrize(
    ("evidence_name", "system", "machine"),
    (
        ("linux-x86_64", "Linux", "aarch64"),
        ("linux-aarch64", "Linux", "x86_64"),
        ("windows-x86_64", "Linux", "x86_64"),
        ("macos-arm64", "Darwin", "x86_64"),
        ("macos-x86_64", "Darwin", "arm64"),
        ("unknown", "Linux", "x86_64"),
    ),
)
def test_evidence_name_rejects_host_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    evidence_name: str,
    system: str,
    machine: str,
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    monkeypatch.setattr(evidence.platform, "system", lambda: system)
    monkeypatch.setattr(evidence.platform, "machine", lambda: machine)

    with pytest.raises(ValueError, match="host|evidence"):
        evidence._host_result(evidence_name)


def test_successful_smoke_payload_normalizes_to_passed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps(_valid_smoke_payload()), encoding="utf-8")

    result = evidence.result_from_smoke(
        source,
        smoke_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result == _valid_result()
    evidence.validate_result(result)


@pytest.mark.parametrize("outcome", ("failure", "cancelled", "skipped"))
def test_external_smoke_outcome_must_be_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps(_valid_smoke_payload()), encoding="utf-8")

    result = evidence.result_from_smoke(
        source,
        smoke_outcome=outcome,
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "smoke_execution"
    with pytest.raises(ValueError, match="did not pass"):
        evidence.validate_result(result)


@pytest.mark.parametrize(
    ("failure_code", "failure_stage"),
    (
        ("fixture_download", "fixture_download"),
        ("artifact_acquisition", "artifact_acquisition"),
        ("smoke_execution", "runtime_smoke"),
        ("cleanup", "cleanup"),
    ),
)
def test_smoke_failure_preserves_only_its_stable_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
    failure_stage: str,
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)
    source = tmp_path / "smoke.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "failure_code": failure_code,
                "failure_stage": failure_stage,
            }
        ),
        encoding="utf-8",
    )

    result = evidence.result_from_smoke(
        source,
        smoke_outcome="failure",
        evidence_name="linux-x86_64",
    )

    assert result["failure_code"] == failure_code
    assert result["failure_stage"] == failure_stage
    assert set(result) == {
        "schema_version",
        "evidence_label",
        "evidence_name",
        "status",
        "failure_code",
        "failure_stage",
        "run",
        "host",
    }


def test_malformed_smoke_failure_is_reduced_to_generic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)
    source = tmp_path / "smoke.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "failure_code": ["not-hashable"],
                "failure_stage": "runtime_smoke",
            }
        ),
        encoding="utf-8",
    )

    result = evidence.result_from_smoke(
        source,
        smoke_outcome="failure",
        evidence_name="linux-x86_64",
    )

    assert result["failure_code"] == "smoke_execution"
    assert result["failure_stage"] == "runtime_smoke"


@pytest.mark.parametrize("check", REQUIRED_CHECKS)
def test_every_required_check_must_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check: str,
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)
    payload = _valid_smoke_payload()
    payload["checks"][check] = "failed"  # type: ignore[index]
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    result = evidence.result_from_smoke(
        source,
        smoke_outcome="success",
        evidence_name="linux-x86_64",
    )

    assert result["status"] == "failed"
    assert result["checks"][check] == "failed"  # type: ignore[index]


@pytest.mark.parametrize("change", ("missing", "skipped", "extra"))
def test_missing_skipped_or_extra_check_is_rejected(change: str) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    result = _valid_result()
    checks = result["checks"]
    assert isinstance(checks, dict)
    if change == "missing":
        checks.pop(REQUIRED_CHECKS[0])
    elif change == "skipped":
        checks[REQUIRED_CHECKS[0]] = "skipped"
    else:
        checks["unapproved"] = "passed"

    with pytest.raises(ValueError, match="check"):
        evidence.validate_result(result, require_pass=False)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result: result["packages"].__setitem__("onnx-asr", "0.11.0"),
        lambda result: result["packages"].__setitem__("onnxruntime-gpu", "1.27.0"),
        lambda result: result.__setitem__(
            "execution_provider", "CUDAExecutionProvider"
        ),
        lambda result: result["artifacts"]["v2_int8"].__setitem__("reference", V3_REF),
        lambda result: result["artifacts"].__setitem__("vad", V2_REF),
        lambda result: result["durations_seconds"].__setitem__("total", 99_999),
    ),
)
def test_package_provider_artifact_and_duration_contracts_are_strict(mutation) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    result = _valid_result()
    mutation(result)

    with pytest.raises(ValueError):
        evidence.validate_result(result, require_pass=False)


@pytest.mark.parametrize(
    "duration", (True, None, "1", -1, float("nan"), float("inf"), 10**310)
)
def test_duration_validation_rejects_non_finite_or_unbounded_values(duration) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    result = _valid_result()
    result["durations_seconds"]["total"] = duration

    with pytest.raises(ValueError, match="duration|numeric"):
        evidence.validate_result(result, require_pass=False)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("path", "private"),
        ("exception", "private"),
        ("token", "private"),
        ("detail", "/home/runner/work/private"),
        ("detail", r"C:\\Users\\runner\\private"),
        ("detail", "hf_secret_credential"),
    ),
    ids=(
        "forbidden-path-key",
        "forbidden-exception-key",
        "forbidden-token-key",
        "posix-home-value",
        "windows-user-value",
        "credential-like-value",
    ),
)
def test_nested_private_content_is_rejected(key: str, value: str) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    result = _valid_result()
    result["nested"] = {"deeper": {key: value}}

    with pytest.raises(ValueError, match="private|key|schema"):
        evidence.validate_result(result, require_pass=False)


def test_failure_documents_are_bounded_and_never_validate_as_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)

    result = evidence.failure_result(
        _run_identity(),
        evidence_name="linux-x86_64",
        failure_code="dependency_install",
        failure_stage="dependency_install",
    )

    assert result["status"] == "failed"
    assert "packages" not in result
    with pytest.raises(ValueError, match="did not pass"):
        evidence.validate_result(result)
    evidence.validate_result(result, require_pass=False)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evidence_name", ["linux-x86_64"]),
        ("status", ["passed"]),
        ("failure_code", ["dependency_install"]),
        ("failure_stage", ["dependency_install"]),
        ("checks", {**dict.fromkeys(REQUIRED_CHECKS, "passed"), "cancellation": []}),
    ),
)
def test_unhashable_json_values_fail_as_validation_errors(
    field: str, value: object
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    result = _valid_result()
    if field in {"failure_code", "failure_stage"}:
        result = _valid_failure()
    result[field] = value

    with pytest.raises(ValueError):
        evidence.validate_result(result, require_pass=False)


def test_cli_rejects_irrelevant_companion_arguments_without_writing(
    tmp_path: Path,
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_valid_result()), encoding="utf-8")
    destination = tmp_path / "sentinel.json"
    destination.write_text("sentinel", encoding="utf-8")

    code = evidence.main(
        [
            "--validate",
            str(source),
            "--evidence-name",
            "linux-x86_64",
            "--output",
            str(destination),
        ]
    )

    assert code == 1
    assert destination.read_text(encoding="utf-8") == "sentinel"


def test_cli_catches_huge_duration_without_traceback_or_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    _stub_linux(evidence, monkeypatch)
    payload = _valid_smoke_payload()
    payload["durations_seconds"]["total"] = 10**310
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    destination = tmp_path / "result.json"

    code = evidence.main(
        [
            "--from-smoke",
            str(source),
            "--smoke-outcome",
            "success",
            "--evidence-name",
            "linux-x86_64",
            "--output",
            str(destination),
        ]
    )

    assert code == 0
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "Traceback" not in capsys.readouterr().err


def test_aggregate_requires_exact_five_same_run_platforms(tmp_path: Path) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    paths = []
    for name in EXPECTED_PLATFORMS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_valid_result(name)), encoding="utf-8")
        paths.append(path)

    result = evidence.aggregate_results(paths)

    assert result["schema_version"] == 1
    assert result["evidence_label"] == "task602_native_parakeet_matrix"
    assert result["status"] == "passed"
    assert result["tested_commit"] == "a" * 40
    assert set(result["platforms"]) == set(EXPECTED_PLATFORMS)
    evidence.validate_aggregate(result)


@pytest.mark.parametrize("mismatch", ("commit", "run", "missing", "duplicate"))
def test_aggregate_rejects_identity_or_platform_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    evidence = _load(EVIDENCE_PATH, "evidence")
    results = [_valid_result(name) for name in EXPECTED_PLATFORMS]
    if mismatch == "commit":
        results[-1]["run"]["tested_commit"] = "b" * 40  # type: ignore[index]
    elif mismatch == "run":
        results[-1]["run"]["workflow_run_id"] = "456"  # type: ignore[index]
        results[-1]["run"]["workflow_run_url"] = (  # type: ignore[index]
            "https://github.com/rmusser01/tldw_chatbook/actions/runs/456"
        )
    elif mismatch == "missing":
        results.pop()
    else:
        results[-1] = copy.deepcopy(results[0])
    paths = []
    for index, result in enumerate(results):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="commit|run|platform|exact"):
        evidence.aggregate_results(paths)


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
    assert "secrets." not in workflow
    assert ": write" not in workflow


def test_workflow_uses_exact_five_platform_matrix_and_selected_commit() -> None:
    workflow = _workflow_text()
    jobs = _yaml_block(workflow, "jobs:")
    strategy = _yaml_block(workflow, "    strategy:")
    checkout = _workflow_step(workflow, EXPECTED_STEPS[0])

    assert re.findall(r"^  ([a-z0-9-]+):$", jobs, re.MULTILINE) == ["platform-evidence"]
    assert _yaml_scalar(jobs, 4, "if") == (
        "github.event_name == 'workflow_dispatch' || "
        "github.event.label.name == 'task-602-platform-evidence'"
    )
    assert _yaml_scalar(jobs, 4, "timeout-minutes") == "45"
    assert _nonempty_lines(strategy) == [
        "      fail-fast: false",
        "      matrix:",
        "        include:",
        "          - evidence_name: linux-x86_64",
        "            os: ubuntu-24.04",
        "          - evidence_name: linux-aarch64",
        "            os: ubuntu-24.04-arm",
        "          - evidence_name: windows-x86_64",
        "            os: windows-2022",
        "          - evidence_name: macos-arm64",
        "            os: macos-15",
        "          - evidence_name: macos-x86_64",
        "            os: macos-15-intel",
    ]
    assert _yaml_scalar(checkout, 8, "uses") == "actions/checkout@v4"
    assert _nonempty_lines(_yaml_block(checkout, "        with:")) == [
        "          ref: ${{ github.event.pull_request.head.sha || github.sha }}"
    ]


def test_workflow_runs_only_bounded_evidence_steps() -> None:
    workflow = _workflow_text()
    names = tuple(re.findall(r"^      - name: (.+)$", workflow, re.MULTILINE))
    dependencies = _workflow_step(workflow, EXPECTED_STEPS[3])
    smoke = _workflow_step(workflow, EXPECTED_STEPS[5])

    assert names == EXPECTED_STEPS
    assert _yaml_scalar(dependencies, 8, "continue-on-error") == "true"
    assert _step_command(dependencies) == (
        "python -m pip install -e "
        "'.[transcription_parakeet_onnx,transcription_faster_whisper]'"
    )
    assert _yaml_scalar(smoke, 8, "continue-on-error") == "true"
    assert "task602_platform_smoke.py" in _step_command(smoke)
    assert '--evidence-name "${{ matrix.evidence_name }}"' in _step_command(smoke)
    assert "pytest" not in _step_command(smoke)
    assert "actions/cache" not in workflow


def test_workflow_normalizes_failures_validates_and_uploads_only_json() -> None:
    workflow = _workflow_text()
    dependency_failure = _workflow_step(workflow, EXPECTED_STEPS[4])
    normalize = _workflow_step(workflow, EXPECTED_STEPS[6])
    validate = _workflow_step(workflow, EXPECTED_STEPS[7])
    upload = _workflow_step(workflow, EXPECTED_STEPS[8])

    assert _yaml_scalar(dependency_failure, 8, "if") == (
        "steps.dependencies.outcome != 'success'"
    )
    assert "--record-failure dependency_install" in _step_command(dependency_failure)
    assert _yaml_scalar(normalize, 8, "if") == (
        "always() && steps.dependencies.outcome == 'success'"
    )
    assert '--smoke-outcome "${{ steps.native_smoke.outcome }}"' in _step_command(
        normalize
    )
    assert _yaml_scalar(validate, 8, "if") == "always()"
    assert "continue-on-error" not in validate
    assert _yaml_scalar(upload, 8, "if") == "always()"
    assert _yaml_scalar(upload, 8, "uses") == "actions/upload-artifact@v4"
    assert _nonempty_lines(_yaml_block(upload, "        with:")) == [
        "          name: task-602-platform-${{ matrix.evidence_name }}",
        "          path: ${{ runner.temp }}/task-602-platform-evidence.json",
        "          if-no-files-found: error",
    ]
