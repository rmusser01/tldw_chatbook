from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = PROJECT_ROOT / ".github/scripts/task602_platform_smoke.py"


def _load_smoke() -> ModuleType:
    assert SMOKE_PATH.is_file()
    spec = importlib.util.spec_from_file_location("task602_smoke", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wav_bytes(*, frames: int = 1_600) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x01\x00" * frames)
    return stream.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_fixture_download_is_bounded_hash_pinned_and_pcm16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    payload = _wav_bytes()
    monkeypatch.setattr(smoke, "FIXTURE_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def open_url(request, *, timeout):
        calls.append((request.full_url, timeout, request.headers))
        return _Response(payload)

    destination = tmp_path / "fixture.wav"
    smoke._download_fixture(destination, open_url=open_url)

    assert destination.read_bytes() == payload
    assert calls == [
        (smoke.FIXTURE_URL, 30.0, {"User-agent": "tldw-task602-evidence/1"})
    ]


@pytest.mark.parametrize("mode", ("digest", "oversize", "format"))
def test_fixture_download_rejects_untrusted_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    smoke = _load_smoke()
    payload = _wav_bytes()
    if mode == "digest":
        monkeypatch.setattr(smoke, "FIXTURE_SHA256", "0" * 64)
    elif mode == "oversize":
        payload = b"x" * (smoke.MAX_FIXTURE_BYTES + 1)
    else:
        payload = b"not a wave"
        monkeypatch.setattr(
            smoke, "FIXTURE_SHA256", hashlib.sha256(payload).hexdigest()
        )

    with pytest.raises(ValueError, match="fixture"):
        smoke._download_fixture(
            tmp_path / "fixture.wav",
            open_url=lambda *_args, **_kwargs: _Response(payload),
        )

    assert not (tmp_path / "fixture.wav").exists()


def test_long_fixture_has_two_speech_regions_and_exceeds_threshold(
    tmp_path: Path,
) -> None:
    smoke = _load_smoke()
    source = tmp_path / "source.wav"
    source.write_bytes(_wav_bytes(frames=16_000))
    destination = tmp_path / "long.wav"

    smoke._build_long_fixture(source, destination)

    with wave.open(str(destination), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16_000
        frames = audio.readframes(audio.getnframes())
        assert audio.getnframes() / audio.getframerate() > 30
    speech = b"\x01\x00" * 16_000
    assert frames.startswith(speech)
    assert frames.endswith(speech)
    assert b"\x00\x00" * (30 * 16_000) in frames


def test_package_observation_requires_exact_runtime_and_cpu_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load_smoke()
    versions = {
        "onnx-asr": "0.12.0",
        "onnxruntime": "1.27.0",
        "faster-whisper": "1.2.1",
        "ctranslate2": "4.8.1",
    }
    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"]
    )

    def package_version(name: str) -> str:
        try:
            return versions[name]
        except KeyError:
            raise smoke.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(smoke.metadata, "version", package_version)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)

    packages, provider = smoke._package_observation()

    assert packages == versions
    assert provider == "CPUExecutionProvider"


def test_package_observation_rejects_accelerator_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load_smoke()

    def package_version(name: str) -> str:
        if name == "onnxruntime-gpu":
            return "1.27.0"
        raise smoke.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(smoke.metadata, "version", package_version)

    with pytest.raises(ValueError, match="accelerator"):
        smoke._package_observation()


def test_managed_dispatch_uses_public_three_string_artifact_reference() -> None:
    smoke = _load_smoke()
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("parakeet-v2", "revision", "int8")
    leased = SimpleNamespace(
        handle=SimpleNamespace(
            root=reference,
            closure_fingerprint="1" * 64,
        )
    )

    dispatch = smoke._managed_dispatch(leased, Path("store"), smoke.V2_MODEL)

    assert dispatch.managed_artifact_ref == ("parakeet-v2", "revision", "int8")


def test_runtime_probe_does_not_import_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load_smoke()
    optional = SimpleNamespace(parakeet_onnx_deps_installed=lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "tldw_chatbook.Utils.optional_deps",
        optional,
    )
    sys.modules.pop("onnx_asr", None)
    sys.modules.pop("onnxruntime", None)

    smoke._probe_runtime()

    assert "onnx_asr" not in sys.modules
    assert "onnxruntime" not in sys.modules


def test_run_smoke_returns_only_bounded_allowlisted_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    offline_names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    for name in offline_names:
        monkeypatch.delenv(name, raising=False)
    setup_order: list[str] = []
    packages = {
        "onnx-asr": "0.12.0",
        "onnxruntime": "1.27.0",
        "faster-whisper": "1.2.1",
        "ctranslate2": "4.8.1",
    }
    artifacts = {
        "v2_int8": {
            "reference": smoke.V2_REFERENCE,
            "closure_fingerprint": "1" * 64,
        },
        "v3_int8": {
            "reference": smoke.V3_REFERENCE,
            "closure_fingerprint": "2" * 64,
        },
        "vad": smoke.VAD_REFERENCE,
    }
    executor_runtime = {
        "checks": {
            "v2_int8_cpu": "passed",
            "v3_int8_cpu": "passed",
            "batch_reuse": "passed",
        },
        "durations": {
            "v2_int8_cpu": 1.0,
            "v3_int8_cpu": 1.1,
        },
    }
    runtime = {
        "checks": {
            "long_form_vad": "passed",
            "cancellation": "passed",
            "retry_wiring": "passed",
        },
        "durations": {
            "long_form_vad": 2.0,
        },
    }

    def observe_packages():
        setup_order.append("packages")
        return packages, "CPUExecutionProvider"

    def probe_runtime():
        setup_order.append("probe")

    monkeypatch.setattr(smoke, "_package_observation", observe_packages)
    monkeypatch.setattr(smoke, "_probe_runtime", probe_runtime)
    monkeypatch.setattr(
        smoke, "_download_fixture", lambda path: path.write_bytes(_wav_bytes())
    )
    monkeypatch.setattr(
        smoke,
        "_build_long_fixture",
        lambda _source, path: path.write_bytes(_wav_bytes()),
    )
    monkeypatch.setattr(
        smoke, "_provision_artifacts", lambda _root: (artifacts, object())
    )

    def observe_executor(*_args):
        setup_order.append("executor")
        assert {name: smoke.os.environ.get(name) for name in offline_names} == {
            name: "1" for name in offline_names
        }
        return executor_runtime

    monkeypatch.setattr(smoke, "_executor_observations", observe_executor)
    monkeypatch.setattr(smoke, "_bounded_runtime_observations", lambda *_args: runtime)
    monkeypatch.setattr(smoke, "_close_resources", lambda _resources: None)
    monkeypatch.setattr(smoke.time, "monotonic", iter((0.0, 3.0, 5.0)).__next__)

    result = smoke.run_smoke("macos-arm64", tmp_path)

    assert result == {
        "schema_version": 1,
        "status": "passed",
        "failure_code": None,
        "failure_stage": None,
        "packages": packages,
        "execution_provider": "CPUExecutionProvider",
        "artifacts": artifacts,
        "checks": {
            "package_resolution": "passed",
            "runtime_probe": "passed",
            **executor_runtime["checks"],
            **runtime["checks"],
        },
        "durations_seconds": {
            "acquisition": 3.0,
            **executor_runtime["durations"],
            **runtime["durations"],
            "total": 5.0,
        },
        "cleanup": "passed",
    }
    assert "/" not in json.dumps(result)
    assert setup_order == ["probe", "packages", "executor"]
    assert all(name not in smoke.os.environ for name in offline_names)


def test_runtime_observations_require_second_segment_cancellation_and_retry_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    from tldw_chatbook.STT.contracts import ExecutionDevice, TranscriptionFailureCode
    from tldw_chatbook.STT.parakeet_onnx import (
        ParakeetOnnxCancelled,
        ParakeetOnnxFailure,
    )

    class Runtime:
        def __init__(self) -> None:
            self._vad = object()

        def transcribe(self, *, attempt_id: str, is_cancelled=None, **_kwargs):
            if attempt_id == "task602-long":
                return SimpleNamespace(
                    produced_capabilities=SimpleNamespace(vad=True),
                    segments=(object(), object()),
                )
            if attempt_id == "task602-cancel":
                assert is_cancelled is not None
                assert is_cancelled() is False
                if is_cancelled():
                    raise ParakeetOnnxCancelled
                raise AssertionError("second segment was not cancelled")
            if attempt_id == "task602-retry" and self._vad is None:
                raise ParakeetOnnxFailure(
                    TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                    "bounded",
                    attempt_id=attempt_id,
                    batch_id=None,
                    job_id=None,
                    model_id=smoke.V2_MODEL,
                    artifact_root=None,
                    artifact_dependencies=(),
                    precision="int8",
                    requested_language="en",
                    effective_language="en",
                    effective_device=ExecutionDevice.CPU,
                )
            raise AssertionError("unexpected runtime call")

        def close(self) -> None:
            pass

    monkeypatch.setattr(smoke, "_load_runtime", lambda *_args: Runtime())

    result = smoke._runtime_observations(
        {"v2": object(), "vad_ref": object()},
        tmp_path / "long.wav",
    )

    assert result["checks"] == {
        "long_form_vad": "passed",
        "cancellation": "passed",
        "retry_wiring": "passed",
    }


def test_executor_reuse_proves_resident_root_and_vad_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _load_smoke()
    from tldw_chatbook.Model_Artifacts import service as service_module
    from tldw_chatbook.STT import dispatch_coordinator, executor as executor_module

    calls: list[tuple[str, object]] = []

    class Lease:
        def __init__(self, reference):
            self.handle = SimpleNamespace(root=reference)

        def close(self):
            calls.append(("close", self.handle.root))

    class Service:
        def __init__(self, root, *, lease_timeout_seconds):
            assert root == Path("store")
            assert lease_timeout_seconds == 0.1

        def delete(self, reference):
            calls.append(("delete", reference))
            raise service_module.ArtifactInUseError("held")

        def acquire(self, reference):
            calls.append(("acquire", reference))
            return Lease(reference)

    class Executor:
        generation = 1
        _unavailable = False

        def __init__(self, **_kwargs):
            pass

        def close(self):
            calls.append(("executor.close", None))

    class Coordinator:
        def __init__(self, _executor):
            pass

        def close(self):
            calls.append(("coordinator.close", None))

    results = iter(
        (
            ({"text": "one"}, 1),
            ({"text": "two"}, 1),
            (
                {
                    "text": "trois",
                    "transcription_provenance": {
                        "requested_language": "fr",
                        "effective_language": "auto",
                        "detected_language": None,
                        "warnings": ["requested_language_not_enforced"],
                    },
                },
                2,
            ),
        )
    )
    monkeypatch.setattr(service_module, "ModelArtifactService", Service)
    monkeypatch.setattr(executor_module, "LocalSTTExecutor", Executor)
    monkeypatch.setattr(
        dispatch_coordinator, "LocalSTTDispatchCoordinator", Coordinator
    )
    monkeypatch.setattr(smoke, "_pcm_source", lambda _path: object())
    monkeypatch.setattr(smoke, "_managed_dispatch", lambda *_args: object())
    monkeypatch.setattr(
        smoke, "_submit_buffer", lambda *_args, **_kwargs: next(results)
    )
    resources = {
        "v2": Lease("v2"),
        "v3": Lease("v3"),
        "vad_ref": "vad",
        "store_root": Path("store"),
    }

    result = smoke._executor_observations(resources, Path("fixture.wav"))

    assert result["checks"]["batch_reuse"] == "passed"
    assert calls[:6] == [
        ("close", "v2"),
        ("close", "v3"),
        ("delete", "v2"),
        ("delete", "vad"),
        ("acquire", "v2"),
        ("acquire", "v3"),
    ]
    assert resources["v2"].handle.root == "v2"
    assert resources["v3"].handle.root == "v3"


def test_runtime_observation_child_returns_only_bounded_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    sent: list[object] = []
    connection = SimpleNamespace(send=sent.append, close=lambda: None)
    expected = {"checks": {"long_form_vad": "passed"}, "durations": {}}
    monkeypatch.setattr(
        smoke,
        "_runtime_observations",
        lambda resources, fixture: (
            expected
            if resources["vad_ref"] == "vad" and fixture == tmp_path / "long.wav"
            else None
        ),
    )

    smoke._runtime_observation_child(
        connection,
        tmp_path / "model",
        tmp_path / "vad",
        "root",
        "vad",
        tmp_path / "long.wav",
    )

    assert sent == [("passed", expected)]


def test_bounded_runtime_observation_terminates_timed_out_child(tmp_path: Path) -> None:
    smoke = _load_smoke()
    calls: list[str] = []

    class Receive:
        def poll(self, _timeout):
            return False

        def close(self):
            calls.append("receive.close")

    class Send:
        def close(self):
            calls.append("send.close")

    class Process:
        alive = True

        def start(self):
            calls.append("start")

        def terminate(self):
            calls.append("terminate")
            self.alive = False

        def join(self, _timeout):
            calls.append("join")

        def kill(self):
            calls.append("kill")
            self.alive = False

        def is_alive(self):
            return self.alive

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is False
            return Receive(), Send()

        def Process(self, *, target, args):
            assert target is smoke._runtime_observation_child
            assert args[-1] == tmp_path / "long.wav"
            return Process()

    handle = SimpleNamespace(
        root="root",
        closure=("root", "vad"),
        paths=(("root", tmp_path / "model"), ("vad", tmp_path / "vad")),
    )
    resources = {"v2": SimpleNamespace(handle=handle), "vad_ref": "vad"}

    with pytest.raises(TimeoutError):
        smoke._bounded_runtime_observations(
            resources,
            tmp_path / "long.wav",
            context=Context(),
            timeout=0.0,
        )

    assert calls == [
        "start",
        "send.close",
        "receive.close",
        "join",
        "terminate",
        "join",
    ]


def test_bounded_runtime_observation_retires_child_after_ipc_failure(
    tmp_path: Path,
) -> None:
    smoke = _load_smoke()
    calls: list[str] = []
    alive = [True]
    receive = SimpleNamespace(
        poll=lambda _timeout: (_ for _ in ()).throw(OSError("private pipe detail")),
        close=lambda: calls.append("receive.close"),
    )
    send = SimpleNamespace(close=lambda: calls.append("send.close"))
    process = SimpleNamespace(
        start=lambda: calls.append("start"),
        join=lambda _timeout: calls.append("join"),
        is_alive=lambda: alive[0],
        terminate=lambda: (calls.append("terminate"), alive.__setitem__(0, False)),
        kill=lambda: (calls.append("kill"), alive.__setitem__(0, False)),
    )
    context = SimpleNamespace(
        Pipe=lambda **_kwargs: (receive, send),
        Process=lambda **_kwargs: process,
    )
    handle = SimpleNamespace(
        root="root",
        closure=("root", "vad"),
        paths=(("root", tmp_path / "model"), ("vad", tmp_path / "vad")),
    )

    with pytest.raises(OSError):
        smoke._bounded_runtime_observations(
            {"v2": SimpleNamespace(handle=handle), "vad_ref": "vad"},
            tmp_path / "long.wav",
            context=context,
        )

    assert "terminate" in calls
    assert alive == [False]


def test_runtime_child_retirement_uses_kill_when_terminate_fails() -> None:
    smoke = _load_smoke()
    calls: list[str] = []
    alive = [True]
    process = SimpleNamespace(
        is_alive=lambda: alive[0],
        terminate=lambda: (
            calls.append("terminate"),
            (_ for _ in ()).throw(OSError("private termination detail")),
        )[-1],
        kill=lambda: (calls.append("kill"), alive.__setitem__(0, False)),
        join=lambda _timeout: calls.append("join"),
    )

    assert smoke._terminate_process(process) is True
    assert calls == ["terminate", "kill", "join"]


def test_main_suppresses_exception_details_and_removes_owned_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    prior_home = smoke.os.environ.get("HOME")
    output = tmp_path / "result.json"
    workspace = tmp_path / "owned-workspace"
    workspace.mkdir()
    (workspace / "sentinel").write_text("private", encoding="utf-8")
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda prefix: str(workspace))
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private /Users/person/model")
        ),
    )

    code = smoke.main(["--evidence-name", "macos-arm64", "--output", str(output)])

    assert code == 1
    assert not workspace.exists()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result == {
        "schema_version": 1,
        "status": "failed",
        "failure_code": "smoke_execution",
        "failure_stage": "runtime_smoke",
    }
    assert "Users" not in output.read_text(encoding="utf-8")
    assert smoke.os.environ.get("HOME") == prior_home


def test_cleanup_failure_replaces_nominal_success_and_keeps_cli_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    output = tmp_path / "result.json"
    workspace = tmp_path / "owned-workspace"
    workspace.mkdir()
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda prefix: str(workspace))
    monkeypatch.setattr(smoke, "run_smoke", lambda *_args: {"status": "passed"})
    monkeypatch.setattr(
        smoke.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("private cleanup detail")),
    )

    code = smoke.main(["--evidence-name", "macos-arm64", "--output", str(output)])

    assert code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "failed",
        "failure_code": "cleanup",
        "failure_stage": "cleanup",
    }


def test_executor_containment_failure_remains_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    monkeypatch.setattr(smoke, "_probe_runtime", lambda: None)
    monkeypatch.setattr(
        smoke,
        "_package_observation",
        lambda: (
            {
                "onnx-asr": "0.12.0",
                "onnxruntime": "1.27.0",
                "faster-whisper": "1.2.1",
                "ctranslate2": "4.8.1",
            },
            "CPUExecutionProvider",
        ),
    )
    monkeypatch.setattr(
        smoke, "_download_fixture", lambda path: path.write_bytes(_wav_bytes())
    )
    monkeypatch.setattr(
        smoke,
        "_build_long_fixture",
        lambda _source, path: path.write_bytes(_wav_bytes()),
    )
    monkeypatch.setattr(smoke, "_provision_artifacts", lambda _root: ({}, object()))
    monkeypatch.setattr(
        smoke,
        "_executor_observations",
        lambda *_args: (_ for _ in ()).throw(smoke.SmokeFailure("cleanup", "cleanup")),
    )
    monkeypatch.setattr(smoke, "_close_resources", lambda _resources: None)

    with pytest.raises(smoke.SmokeFailure) as failure:
        smoke.run_smoke("macos-arm64", tmp_path)

    assert (failure.value.code, failure.value.stage) == ("cleanup", "cleanup")


def test_unproven_executor_cleanup_quarantines_owned_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    output = tmp_path / "result.json"
    workspace = tmp_path / "owned-workspace"
    workspace.mkdir()
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda prefix: str(workspace))
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda *_args: (_ for _ in ()).throw(smoke.SmokeFailure("cleanup", "cleanup")),
    )
    removed: list[Path] = []
    monkeypatch.setattr(smoke.shutil, "rmtree", removed.append)

    code = smoke.main(["--evidence-name", "macos-arm64", "--output", str(output)])

    assert code == 1
    assert workspace.is_dir()
    assert removed == []
    assert json.loads(output.read_text(encoding="utf-8"))["failure_code"] == "cleanup"
