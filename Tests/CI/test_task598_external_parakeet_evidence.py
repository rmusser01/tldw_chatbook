from __future__ import annotations

import importlib.util
import importlib
import json
import os
import re
import signal
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/task-598-platform-evidence.yml"
PROBE_PATH = PROJECT_ROOT / ".github/scripts/task598_external_parakeet_evidence.py"
MODEL_IDS = (
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
)
EXPECTED_REFERENCES = {
    MODEL_IDS[0]: {
        "artifact_id": "parakeet-v2",
        "revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
        "variant": "int8",
    },
    MODEL_IDS[1]: {
        "artifact_id": "parakeet-v3",
        "revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
        "variant": "int8",
    },
}
VAD_REFERENCE = {
    "artifact_id": "silero-vad",
    "revision": "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    "variant": "f32",
}


def _load_probe() -> ModuleType:
    assert PROBE_PATH.is_file(), "TASK-598 evidence probe is missing"
    spec = importlib.util.spec_from_file_location("task598_evidence", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow_text() -> str:
    assert WORKFLOW_PATH.is_file(), "TASK-598 evidence workflow is missing"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _run_identity() -> dict[str, str]:
    return {
        "tested_commit": "a" * 40,
        "workflow_run_id": "123",
        "workflow_run_attempt": "1",
    }


def _valid_result() -> dict[str, object]:
    model_result = {
        "descriptor_verified": True,
        "managed_copy_deleted": True,
        "external_unchanged": True,
        "cache_unchanged": True,
        "store_unchanged": True,
        "source_preference_unchanged": True,
        "execution_provider": "CPUExecutionProvider",
        "artifact_root": None,
        "artifact_dependencies": [VAD_REFERENCE],
        "shutdown_completed": True,
        "timings": {
            "inference_seconds": 1.0,
            "model_total_seconds": 2.0,
        },
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "failure_code": None,
        "failure_stage": None,
        "run": _run_identity(),
        "host": {
            "system": "Linux",
            "architecture": "x86_64",
            "python": "3.12.10",
            "packages": {
                "onnx-asr": "0.12.0",
                "onnxruntime": "1.23.2",
            },
            "available_providers": ["CPUExecutionProvider"],
        },
        "models": {
            model_id: {**model_result, "reference": EXPECTED_REFERENCES[model_id]}
            for model_id in MODEL_IDS
        },
        "final_store": {
            "vad_only": True,
            "no_parakeet_roots": True,
            "no_readiness": True,
            "no_active_selector": True,
        },
    }


def test_workflow_is_explicitly_triggered_and_checks_out_pr_head() -> None:
    workflow = _workflow_text()

    trigger = workflow[: workflow.index("jobs:")]
    assert "pull_request:" in trigger
    assert "types: [labeled]" in trigger
    assert "workflow_dispatch:" in trigger
    assert "push:" not in trigger
    assert "schedule:" not in trigger
    assert (
        "github.event_name == 'workflow_dispatch' || "
        "github.event.label.name == 'task-598-platform-evidence'" in workflow
    )
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "permissions:\n  contents: read" in workflow


def test_workflow_uses_the_exact_bounded_native_matrix() -> None:
    workflow = _workflow_text()

    assert set(re.findall(r"^\s+os: ([-\w.]+)$", workflow, re.MULTILINE)) == {
        "ubuntu-24.04",
        "ubuntu-24.04-arm",
        "windows-2022",
        "macos-15-intel",
    }
    assert "runs-on: ${{ matrix.os }}" in workflow
    assert "fail-fast: false" in workflow
    assert "max-parallel: 2" in workflow
    assert 'python-version: "3.12"' in workflow
    assert 'pip install -e ".[transcription_parakeet_onnx]"' in workflow
    assert "onnxruntime" not in workflow.lower()

    job_timeout = int(re.search(r"timeout-minutes: (\d+)", workflow).group(1))
    worker_timeout = int(re.search(r"--timeout-seconds (\d+)", workflow).group(1))
    assert worker_timeout < job_timeout * 60


def test_workflow_records_install_failure_and_always_uploads_json() -> None:
    workflow = _workflow_text()

    initialize = workflow.index("--initialize")
    install = workflow.index("id: dependencies")
    failure = workflow.index("--record-failure dependency_install")
    probe = workflow.index("--timeout-seconds")
    assert initialize < install < failure < probe
    assert "continue-on-error: true" in workflow[install:failure]
    assert "steps.dependencies.outcome != 'success'" in workflow[failure:probe]
    assert "steps.dependencies.outcome == 'success'" in workflow[probe:]
    assert workflow.count("if: always()") >= 2
    assert "uses: actions/upload-artifact@v4" in workflow
    assert "task-598-platform-${{ matrix.evidence_name }}" in workflow


def test_workflow_uses_runner_temp_only_after_runner_assignment() -> None:
    workflow = _workflow_text()
    job_preamble = workflow[
        workflow.index("platform-evidence:") : workflow.index("steps:")
    ]

    assert "runner.temp" not in job_preamble
    assert "EVIDENCE_PATH" not in workflow
    assert workflow.count("$RUNNER_TEMP/task-598-platform-evidence.json") == 4
    assert "path: ${{ runner.temp }}/task-598-platform-evidence.json" in workflow


def test_supervisor_records_a_path_private_timeout(tmp_path: Path) -> None:
    evidence = _load_probe()
    output = tmp_path / "result.json"

    result = evidence.supervise(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        output=output,
        timeout_seconds=0.05,
        run_identity=_run_identity(),
        forbidden_roots=(tmp_path,),
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "timeout"
    serialized = output.read_text(encoding="utf-8")
    assert json.loads(serialized) == result
    assert str(tmp_path) not in serialized


def test_supervisor_records_a_path_private_start_failure(tmp_path: Path) -> None:
    evidence = _load_probe()
    output = tmp_path / "result.json"

    result = evidence.supervise(
        [str(tmp_path / "missing-python")],
        output=output,
        timeout_seconds=1,
        run_identity=_run_identity(),
        forbidden_roots=(tmp_path,),
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "worker_start_failed"
    serialized = output.read_text(encoding="utf-8")
    assert json.loads(serialized) == result
    assert str(tmp_path) not in serialized


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
def test_supervisor_terminates_a_descendant_in_a_new_session(tmp_path: Path) -> None:
    evidence = _load_probe()
    output = tmp_path / "result.json"
    control = tmp_path / "control.json"
    grandchild = (
        "import json, os, sys, time; "
        "open(sys.argv[1], 'w', encoding='utf-8').write("
        "json.dumps({'native_pid': os.getpid(), "
        "'native_process_group_id': os.getpgrp()})); "
        "time.sleep(30)"
    )
    parent = (
        "import signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "start_new_session=True); "
        "time.sleep(30)"
    )

    try:
        result = evidence.supervise(
            [sys.executable, "-c", parent, grandchild, str(control)],
            output=output,
            timeout_seconds=0.2,
            run_identity=_run_identity(),
            forbidden_roots=(tmp_path,),
            control=control,
            cleanup_seconds=0.2,
        )

        identity = json.loads(control.read_text(encoding="utf-8"))
        assert result["failure_code"] == "timeout"
        assert not evidence._posix_group_exists(identity["native_process_group_id"])
    finally:
        if control.exists():
            identity = json.loads(control.read_text(encoding="utf-8"))
            try:
                os.killpg(identity["native_process_group_id"], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-kill contract")
def test_supervisor_removes_owned_native_temp_after_forced_timeout(
    tmp_path: Path,
) -> None:
    evidence = _load_probe()
    output = tmp_path / "result.json"
    environment = evidence._isolated_environment(tmp_path)
    native_temp = Path(environment["TMPDIR"])
    child = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "root = pathlib.Path(os.environ['TMPDIR']); "
        "root.mkdir(parents=True, exist_ok=True); "
        "(root / 'executor-scratch').mkdir(); "
        "(root / 'executor-scratch' / 'live').write_text('open'); "
        "time.sleep(30)"
    )

    result = evidence.supervise(
        [sys.executable, "-c", child],
        output=output,
        timeout_seconds=0.2,
        run_identity=_run_identity(),
        forbidden_roots=(tmp_path,),
        env=environment,
        cleanup_root=native_temp,
        cleanup_parent=tmp_path,
        cleanup_seconds=0.2,
    )

    assert result["failure_code"] == "timeout"
    assert not native_temp.exists()


def test_parent_isolates_all_profile_roots_and_uses_zero_pcm(tmp_path: Path) -> None:
    evidence = _load_probe()

    environment = evidence._isolated_environment(tmp_path)

    assert environment["HOME"].startswith(str(tmp_path))
    assert environment["XDG_CONFIG_HOME"].startswith(str(tmp_path))
    assert environment["XDG_DATA_HOME"].startswith(str(tmp_path))
    assert environment["XDG_CACHE_HOME"].startswith(str(tmp_path))
    assert environment["HF_HOME"].startswith(str(tmp_path))
    assert environment["TMPDIR"].startswith(str(tmp_path))
    assert environment["TMP"] == environment["TMPDIR"]
    assert environment["TEMP"] == environment["TMPDIR"]
    pcm = evidence._pcm_fixture()
    assert len(pcm) == 16_000 * 4 * 2
    assert not any(pcm)


def test_isolated_root_check_canonicalizes_a_directory_alias(tmp_path: Path) -> None:
    evidence = _load_probe()
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(canonical, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    managed = canonical / "data" / "task598" / "models" / "managed"

    assert evidence._path_is_within(managed, alias)


def test_parent_passes_a_canonical_scratch_root_to_the_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_probe()
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(canonical, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    observed: dict[str, object] = {}

    class _TemporaryDirectory:
        def __enter__(self) -> str:
            return str(alias)

        def __exit__(self, *_args: object) -> None:
            return None

    def supervise(command: object, **kwargs: object) -> dict[str, object]:
        observed["command"] = command
        observed["cleanup_parent"] = kwargs["cleanup_parent"]
        return evidence.failure_result(
            kwargs["run_identity"],
            failure_code="probe_failed",
            failure_stage="test",
        )

    monkeypatch.setattr(
        evidence.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: _TemporaryDirectory(),
    )
    monkeypatch.setattr(evidence, "supervise", supervise)

    assert evidence.run_parent(tmp_path / "result.json", 1) == 1
    command = observed["command"]
    scratch_index = command.index("--scratch") + 1
    assert Path(command[scratch_index]) == canonical.resolve(strict=True)
    assert observed["cleanup_parent"] == canonical.resolve(strict=True)


def test_cache_token_covers_xdg_writes_outside_hf_home(tmp_path: Path) -> None:
    evidence = _load_probe()
    xdg_cache = tmp_path / "cache"
    hf_home = xdg_cache / "huggingface"
    hf_home.mkdir(parents=True)
    before = evidence._cache_token((xdg_cache, hf_home))

    (xdg_cache / "native-runtime.cache").write_bytes(b"changed")

    assert evidence._cache_token((xdg_cache, hf_home)) != before


def test_shutdown_proof_rejects_an_unproven_native_tree(tmp_path: Path) -> None:
    evidence = _load_probe()

    class _Resource:
        def close(self) -> None:
            return None

    class _Tree:
        def close(self) -> bool:
            return False

    class _Executor(_Resource):
        _tree = _Tree()
        _scratch_path = tmp_path / "native-scratch"

    _Executor._scratch_path.mkdir()

    assert not evidence._close_runtime_resources(
        _Resource(),
        _Resource(),
        _Executor(),
    )

    class _ProvenTree:
        def close(self) -> bool:
            return True

    class _ProvenExecutor(_Resource):
        _tree = _ProvenTree()
        _scratch_path = tmp_path / "proven-scratch"

        def close(self) -> None:
            self._scratch_path.rmdir()

    _ProvenExecutor._scratch_path.mkdir()
    assert evidence._close_runtime_resources(
        _Resource(),
        _Resource(),
        _ProvenExecutor(),
    )


def test_dependency_install_failure_is_valid_but_not_successful() -> None:
    evidence = _load_probe()
    result = evidence.failure_result(
        _run_identity(),
        failure_code="dependency_install",
        failure_stage="install",
    )

    evidence.validate_result(result, require_success=False)
    with pytest.raises(ValueError, match="did not pass"):
        evidence.validate_result(result)


def test_failure_envelope_never_probes_the_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_probe()

    def fail_if_called() -> list[str]:
        pytest.fail("failure-result generation imported the native runtime")

    monkeypatch.setattr(evidence, "_runtime_providers", fail_if_called)

    result = evidence.failure_result(
        _run_identity(),
        failure_code="dependency_install",
        failure_stage="install",
    )

    assert result["host"]["available_providers"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result: result["host"].update(unreviewed="runneradmin"),
        lambda result: result["models"].update(unreviewed=True),
        lambda result: result["final_store"].update(unreviewed=True),
        lambda result: result["run"].update(workflow_run_id="runneradmin"),
        lambda result: result.update(failure_type="bad failure type"),
    ),
)
def test_failure_envelope_is_exact_and_bounded(mutation: object) -> None:
    evidence = _load_probe()
    result = evidence.failure_result(
        _run_identity(),
        failure_code="dependency_install",
        failure_stage="install",
    )
    mutation(result)

    with pytest.raises(ValueError):
        evidence.validate_result(result, require_success=False)


def test_success_evidence_rejects_failure_type() -> None:
    evidence = _load_probe()
    result = _valid_result()
    result["failure_type"] = "RuntimeError"

    with pytest.raises(ValueError, match="failure_type"):
        evidence.validate_result(result)


def test_probe_reads_the_production_transcription_provenance_field() -> None:
    evidence = _load_probe()
    provenance = {"artifact_root": None}

    assert (
        evidence._transcription_provenance({"transcription_provenance": provenance})
        is provenance
    )
    with pytest.raises(RuntimeError, match="omitted provenance"):
        evidence._transcription_provenance({"provenance": provenance})


@pytest.mark.parametrize(
    ("error_message", "expected_code"),
    (
        ("provider_unavailable", "provider_unavailable"),
        ("native loader opened a private location", "probe_failed"),
    ),
)
def test_worker_records_only_bounded_model_substage_and_failure_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_message: str,
    expected_code: str,
) -> None:
    evidence = _load_probe()
    artifact_module = importlib.import_module(
        "tldw_chatbook.Local_Ingestion.parakeet_v2_artifact"
    )
    store_module = importlib.import_module("tldw_chatbook.Model_Artifacts.store")
    managed_root = tmp_path / "data" / "managed"

    class _Service:
        artifacts_path = managed_root / "artifacts"

    def fail_model(*_args: object, report_stage: object, **_kwargs: object) -> None:
        report_stage("transcription")
        raise RuntimeError(error_message)

    monkeypatch.setattr(artifact_module, "parakeet_v2_managed_service", _Service)
    monkeypatch.setattr(
        store_module, "managed_model_artifact_root", lambda: managed_root
    )
    monkeypatch.setattr(evidence, "_run_one_model", fail_model)
    monkeypatch.setattr(evidence.signal, "signal", lambda *_args: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache" / "huggingface"))
    output = tmp_path / "result.json"

    assert evidence.run_worker(output, tmp_path, tmp_path / "control.json") == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["failure_code"] == expected_code
    assert result["failure_stage"] == "model_0_transcription"
    if expected_code == "probe_failed":
        assert error_message not in output.read_text(encoding="utf-8")


def test_normalized_failure_code_never_coerces_a_string_subclass() -> None:
    evidence = _load_probe()

    class _Secret(str):
        def __str__(self) -> str:
            return "private_exception_text"

    error = RuntimeError(_Secret("provider_unavailable"))

    assert evidence._normalized_failure_code(error) == "probe_failed"


@pytest.mark.parametrize(
    "code_value",
    (
        "unsupported_descriptor",
        "missing_file",
        "irregular_file",
        "changed_file",
        "corrupt_file",
        "cancelled",
    ),
)
def test_normalized_failure_code_reports_only_bounded_external_verifier_codes(
    code_value: str,
) -> None:
    evidence = _load_probe()
    external = importlib.import_module("tldw_chatbook.STT.parakeet_external")
    code = external.ExternalParakeetErrorCode(code_value)
    error = external.ExternalParakeetVerificationError(code)

    assert evidence._normalized_failure_code(error) == f"external_{code_value}"


@pytest.mark.parametrize(
    "diagnostic_code",
    (
        "ancestor_identity",
        "file_path_identity",
        "open_file_identity",
        "post_read_file_identity",
        "file_read",
        "snapshot_identity",
    ),
)
def test_normalized_failure_code_reports_bounded_changed_file_diagnostics(
    diagnostic_code: str,
) -> None:
    evidence = _load_probe()
    external = importlib.import_module("tldw_chatbook.STT.parakeet_external")
    error = external.ExternalParakeetVerificationError(
        external.ExternalParakeetErrorCode.CHANGED,
        diagnostic_code=diagnostic_code,
    )

    assert evidence._normalized_failure_code(error) == (
        f"external_changed_{diagnostic_code}"
    )


def test_facade_cleanup_failure_still_closes_native_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _load_probe()
    closed: list[tuple[object, object, object]] = []

    class _Facade:
        def cleanup(self) -> None:
            raise RuntimeError("private_exception_text")

    resources = (object(), object(), object())

    def close_resources(*values: object) -> bool:
        closed.append(values)
        return True

    monkeypatch.setattr(evidence, "_close_runtime_resources", close_resources)

    assert not evidence._cleanup_model_runtime(_Facade(), *resources)
    assert closed == [resources]


def test_validate_cli_rejects_failure_without_a_path_bearing_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = _load_probe()
    output = tmp_path / "result.json"
    evidence._write_result(
        output,
        evidence.failure_result(
            _run_identity(),
            failure_code="dependency_install",
            failure_stage="install",
        ),
    )

    assert evidence.main(["--validate", str(output)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert str(tmp_path) not in captured.out + captured.err


def test_main_docstring_documents_cli_arguments_and_exit_code() -> None:
    evidence = _load_probe()
    docstring = evidence.main.__doc__ or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring


def test_validator_accepts_only_complete_two_model_cpu_evidence() -> None:
    evidence = _load_probe()

    evidence.validate_result(_valid_result())


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda result: result["models"].pop(MODEL_IDS[1]), "models"),
        (
            lambda result: result["models"][MODEL_IDS[0]].update(
                execution_provider="CUDAExecutionProvider"
            ),
            "CPUExecutionProvider",
        ),
        (
            lambda result: result["models"][MODEL_IDS[0]].update(
                artifact_root=VAD_REFERENCE
            ),
            "artifact_root",
        ),
        (
            lambda result: result["models"][MODEL_IDS[0]].update(cache_unchanged=False),
            "cache_unchanged",
        ),
        (
            lambda result: result["host"]["packages"].pop("onnxruntime"),
            "onnxruntime",
        ),
        (
            lambda result: result["models"][MODEL_IDS[0]].pop("timings"),
            "timings",
        ),
    ),
)
def test_validator_rejects_incomplete_or_non_cpu_evidence(
    mutation: object,
    expected: str,
) -> None:
    evidence = _load_probe()
    result = _valid_result()
    mutation(result)

    with pytest.raises(ValueError, match=expected):
        evidence.validate_result(result)


def test_validator_rejects_paths_anywhere_in_result(tmp_path: Path) -> None:
    evidence = _load_probe()
    result = _valid_result()
    result["models"][MODEL_IDS[0]]["debug"] = str(tmp_path / "external")

    with pytest.raises(ValueError, match="local path"):
        evidence.validate_result(result, forbidden_roots=(tmp_path,))


def test_validator_rejects_an_embedded_absolute_path() -> None:
    evidence = _load_probe()
    result = _valid_result()
    result["debug"] = "native loader opened /opt/private/model"

    with pytest.raises(ValueError, match="local path"):
        evidence.validate_result(result)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        ("username", "runneradmin", "username"),
        ("access_token", "credential-value", "credential"),
    ),
)
def test_validator_rejects_sensitive_identity_fields(
    field: str,
    value: str,
    expected: str,
) -> None:
    evidence = _load_probe()
    result = _valid_result()
    result[field] = value

    with pytest.raises(ValueError, match=expected):
        evidence.validate_result(result)


def test_validator_rejects_unreviewed_result_fields() -> None:
    evidence = _load_probe()
    result = _valid_result()
    result["debug"] = "harmless-but-unreviewed"

    with pytest.raises(ValueError, match="fields"):
        evidence.validate_result(result)
