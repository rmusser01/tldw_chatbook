from __future__ import annotations

import importlib
import sys
import tomllib
import traceback
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.STT.contracts import (
    DeviceFailureOrigin,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
    TranscriptionWarningCode,
)
from tldw_chatbook.STT.executor import ExecutorRequest, ModelIdentity
from tldw_chatbook.STT.executor_worker import (
    _ProviderLoadFailure,
    _failure_from_exception,
    _transcribe_cpp_provider,
)


def _write_pcm_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 1_600)


class _FakeSession:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self._calls = calls

    def __enter__(self) -> "_FakeSession":
        self._calls.append(("session_enter", None))
        return self

    def __exit__(self, *_args: object) -> None:
        self._calls.append(("session_exit", None))

    def run(self, pcm: object, **kwargs: object) -> object:
        self._calls.append(("run", (pcm, kwargs)))
        return SimpleNamespace(
            text="hello world",
            language="en",
            segments=(SimpleNamespace(text="hello world", t0_ms=0, t1_ms=100),),
            timings=SimpleNamespace(
                mel_ms=1.0,
                encode_ms=2.0,
                decode_ms=3.0,
            ),
        )


class _FakeModel:
    def __init__(
        self,
        path: str,
        calls: list[tuple[str, object]],
        *,
        backend: str = "auto",
    ) -> None:
        self._calls = calls
        calls.append(("model", (path, backend)))
        self.arch = "whisper"
        self.variant = "base"
        self.backend = backend
        self.device = SimpleNamespace(kind="cpu" if backend == "auto" else backend)
        self.capabilities = SimpleNamespace(
            native_sample_rate=16_000,
            languages=("en", "fr"),
            max_timestamp_kind="segment",
            supports_language_detect=True,
            supports_translate=True,
            supports_streaming=False,
            supports_spec_decode=False,
            max_audio_ms=None,
            translate_target_languages=("en",),
        )

    def session(self) -> _FakeSession:
        self._calls.append(("session", None))
        return _FakeSession(self._calls)

    def close(self) -> None:
        self._calls.append(("close", None))


def _fake_runtime(calls: list[tuple[str, object]]) -> object:
    class Model:
        def __new__(cls, path: str, *, backend: str = "auto") -> _FakeModel:
            return _FakeModel(path, calls, backend=backend)

    def set_log_callback(callback: object) -> None:
        calls.append(("set_log_callback", callback))

    return SimpleNamespace(Model=Model, set_log_callback=set_log_callback)


def test_module_import_does_not_import_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "transcribe_cpp", raising=False)

    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    importlib.reload(module)

    assert "transcribe_cpp" not in sys.modules


def test_optional_extra_pins_exact_runtime_without_platform_marker() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["optional-dependencies"][
        "transcription_transcribe_cpp"
    ] == ["transcribe-cpp==0.1.3"]


def test_transcribe_file_revalidates_then_loads_once_and_normalizes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    audio_path = tmp_path / "audio.wav"
    model_path = tmp_path / "private-model.gguf"
    _write_pcm_wav(audio_path)
    model_path.write_bytes(b"fixture")
    calls: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "transcribe_cpp", _fake_runtime(calls))

    def validate(path: Path) -> object:
        calls.append(("validate", path))
        return SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        )

    monkeypatch.setattr(module, "validate_local_gguf", validate)

    result = module.transcribe_file(
        audio_path=audio_path,
        model_path=model_path,
        attempt_id="attempt-1",
        job_id="ingest-job-1",
        language="en",
        timestamps=True,
    )

    event_names = [name for name, _value in calls]
    assert event_names.index("validate") < event_names.index("model")
    assert event_names.count("model") == 1
    assert event_names.count("run") == 1
    assert event_names[-1] == "close"
    pcm, run_kwargs = next(value for name, value in calls if name == "run")
    assert len(pcm) == 1_600
    assert run_kwargs == {
        "task": "transcribe",
        "language": "en",
        "timestamps": "segment",
    }
    assert result.text == "hello world"
    assert result.segments[0].start_seconds == 0
    assert result.segments[0].end_seconds == pytest.approx(0.1)
    assert result.duration_seconds == pytest.approx(0.1)
    assert result.provenance.provider_id == "transcribe-cpp"
    assert result.provenance.model_id == "local-gguf:whisper"
    assert result.provenance.artifact_root is None
    assert result.provenance.artifact_dependencies == ()
    assert result.provenance.precision == "native"
    assert result.provenance.effective_device.value == "cpu"
    assert result.timings.total_seconds == pytest.approx(
        result.timings.model_load_seconds + result.timings.inference_seconds
    )


def test_reusable_runtime_passes_explicit_cpu_backend_to_native_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    model_path = tmp_path / "private-model.gguf"
    model_path.write_bytes(b"fixture")
    calls: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "transcribe_cpp", _fake_runtime(calls))
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )

    runtime = module.TranscribeCppRuntime.load(
        model_path=model_path,
        attempt_id="attempt-cpu",
        device=module.ExecutionDevice.CPU,
    )
    runtime.close()

    assert ("model", (str(model_path), "cpu")) in calls


def test_native_backend_load_failure_is_typed_for_one_cpu_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    model_path = tmp_path / "private-model.gguf"
    model_path.write_bytes(b"fixture")

    class BackendError(RuntimeError):
        pass

    class FailingModel:
        def __init__(self, _path: str, *, backend: str = "auto") -> None:
            assert backend == "auto"
            raise BackendError("private native backend detail")

    monkeypatch.setitem(
        sys.modules,
        "transcribe_cpp",
        SimpleNamespace(
            BackendError=BackendError,
            Model=FailingModel,
            backends=lambda: (SimpleNamespace(kind="metal"),),
            set_log_callback=lambda _callback: None,
        ),
    )
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )

    with pytest.raises(module.TranscribeCppFailure) as raised:
        module.TranscribeCppRuntime.load(
            model_path=model_path,
            attempt_id="attempt-metal",
            device=ExecutionDevice.AUTO,
        )

    assert raised.value.code.value == "provider_unavailable"
    assert (
        raised.value.device_failure_origin
        is DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION
    )
    assert raised.value.failed_device is ExecutionDevice.METAL
    assert "private native backend detail" not in str(raised.value)
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-metal",
        job_id="job-metal",
        source=FileAudioSource(tmp_path / "audio.wav"),
        identity=ModelIdentity(
            provider_id="transcribe-cpp",
            model_id="local-gguf:whisper",
            root_revision=None,
            closure_fingerprint=None,
            precision="native",
            device=ExecutionDevice.AUTO,
        ),
        options={},
    )
    worker_failure = _failure_from_exception(request, raised.value)
    assert (
        worker_failure.device_failure_origin
        is DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION
    )
    assert worker_failure.failed_device is ExecutionDevice.METAL


@pytest.mark.parametrize(
    ("backend_override", "registered_kinds"),
    [
        ("cpu", ("metal",)),
        ("not-a-backend", ("metal",)),
        (None, ("metal", "cuda")),
    ],
)
def test_auto_failure_candidate_fails_closed_when_device_is_not_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
    backend_override: str | None,
    registered_kinds: tuple[str, ...],
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    runtime = SimpleNamespace(
        backends=lambda: tuple(SimpleNamespace(kind=kind) for kind in registered_kinds)
    )
    if backend_override is None:
        monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("TRANSCRIBE_BACKEND", backend_override)

    assert module._failed_accelerator_candidate(runtime, ExecutionDevice.AUTO) is None


def test_successful_cpu_retry_records_warning_and_original_requested_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    audio_path = tmp_path / "audio.wav"
    model_path = tmp_path / "private-model.gguf"
    _write_pcm_wav(audio_path)
    model_path.write_bytes(b"fixture")
    calls: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "transcribe_cpp", _fake_runtime(calls))
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )
    request = ExecutorRequest(
        generation=2,
        attempt_id="attempt-cpu-retry",
        job_id="job-cpu-retry",
        source=FileAudioSource(audio_path),
        identity=ModelIdentity(
            provider_id="transcribe-cpp",
            model_id="local-gguf:whisper",
            root_revision=None,
            closure_fingerprint=None,
            precision="native",
            device=ExecutionDevice.CPU,
        ),
        options={"_local_stt_cpu_fallback_requested_device": "auto"},
    )

    provider = _transcribe_cpp_provider(request, model_path, None, lambda: False)
    try:
        payload = provider.runner(str(audio_path), timestamps=True)
    finally:
        provider.close()

    provenance = payload["transcription_provenance"]
    assert provenance["requested_device"] == "auto"
    assert provenance["effective_device"] == "cpu"
    assert provenance["warnings"] == [
        TranscriptionWarningCode.DEVICE_FALLBACK_TO_CPU.value
    ]


@pytest.mark.parametrize(
    "relative_value",
    (
        r"C:\private\evil.gguf",
        r"\\server\share\evil.gguf",
        r"nested\..\evil.gguf",
    ),
)
def test_managed_gguf_rejects_windows_path_escape_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_value: str,
) -> None:
    model_root = tmp_path / "managed"
    model_root.mkdir()
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-managed-path",
        job_id="job-managed-path",
        source=FileAudioSource(tmp_path / "audio.wav"),
        identity=ModelIdentity(
            provider_id="transcribe-cpp",
            model_id="managed-gguf",
            root_revision="revision-a",
            closure_fingerprint="closure-a",
            precision="native",
            device=ExecutionDevice.CPU,
        ),
        options={"managed_model_relative_path": relative_value},
    )
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    monkeypatch.setattr(
        module.TranscribeCppRuntime,
        "load",
        lambda **_kwargs: SimpleNamespace(close=lambda: None),
    )

    with pytest.raises(_ProviderLoadFailure) as raised:
        _transcribe_cpp_provider(request, model_root, None, lambda: False)

    assert raised.value.code is TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE


def test_managed_gguf_rejects_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "managed"
    model_root.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"fixture")
    link = model_root / "model.gguf"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-managed-link",
        job_id="job-managed-link",
        source=FileAudioSource(tmp_path / "audio.wav"),
        identity=ModelIdentity(
            provider_id="transcribe-cpp",
            model_id="managed-gguf",
            root_revision="revision-a",
            closure_fingerprint="closure-a",
            precision="native",
            device=ExecutionDevice.CPU,
        ),
        options={"managed_model_relative_path": "model.gguf"},
    )
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    monkeypatch.setattr(
        module.TranscribeCppRuntime,
        "load",
        lambda **_kwargs: SimpleNamespace(close=lambda: None),
    )

    with pytest.raises(_ProviderLoadFailure) as raised:
        _transcribe_cpp_provider(request, model_root, None, lambda: False)

    assert raised.value.code is TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE


def test_reusable_runtime_loads_once_transcribes_twice_and_closes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    model_path = tmp_path / "private-model.gguf"
    _write_pcm_wav(first_audio)
    _write_pcm_wav(second_audio)
    model_path.write_bytes(b"fixture")
    calls: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "transcribe_cpp", _fake_runtime(calls))
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )

    runtime = module.TranscribeCppRuntime.load(
        model_path=model_path,
        attempt_id="attempt-1",
        language="en",
    )
    try:
        first = runtime.transcribe(
            audio_path=first_audio,
            attempt_id="attempt-1",
            language="en",
        )
        second = runtime.transcribe(
            audio_path=second_audio,
            attempt_id="attempt-2",
            language="en",
        )
    finally:
        runtime.close()
        runtime.close()

    event_names = [name for name, _value in calls]
    assert event_names.count("model") == 1
    assert event_names.count("run") == 2
    assert event_names.count("close") == 1
    assert first.provenance.attempt_id == "attempt-1"
    assert second.provenance.attempt_id == "attempt-2"
    assert first.timings.model_load_seconds >= 0
    assert second.timings.model_load_seconds == 0


def test_loaded_capabilities_are_identical_in_declaration_and_probe() -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    calls: list[tuple[str, object]] = []
    model = _FakeModel("ignored.gguf", calls)

    adapter = module.TranscribeCppAdapter(
        model=model,
        architecture="whisper",
        model_load_seconds=0.01,
    )
    described = adapter.describe()[0]
    observed = adapter.probe(described.model_id)

    assert observed.available is True
    assert observed.capabilities == described.capabilities
    assert described.semantic_default_eligible is False


@pytest.mark.parametrize(
    ("native_maximum", "expected"),
    [
        ("none", {"none"}),
        ("segment", {"none", "segment"}),
        ("word", {"none", "segment", "word"}),
        ("token", {"none", "segment", "word"}),
    ],
)
def test_native_timestamp_maximum_maps_to_supported_contract_granularities(
    native_maximum: str,
    expected: set[str],
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")

    mapped = module._timestamp_capabilities(native_maximum)

    assert {granularity.value for granularity in mapped} == expected


@pytest.mark.parametrize(
    ("native_kind", "expected"),
    [
        ("cpu", "cpu"),
        ("accel", "cpu"),
        ("cpu_accel", "cpu"),
        ("metal", "metal"),
        ("mps", "metal"),
        ("vulkan", "vulkan"),
        ("cuda", "cuda"),
    ],
)
def test_native_device_kind_maps_to_supported_contract_device(
    native_kind: str,
    expected: str,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    model = SimpleNamespace(device=SimpleNamespace(kind=native_kind))

    mapped = module._device_from_model(model)

    assert mapped.value == expected


def test_missing_configured_model_fails_with_picker_action_before_runtime_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    audio_path = tmp_path / "audio.wav"
    _write_pcm_wav(audio_path)
    imports: list[str] = []

    def record_import(name: str) -> object:
        imports.append(name)
        return object()

    monkeypatch.setattr(module.importlib, "import_module", record_import)

    with pytest.raises(module.TranscribeCppFailure) as raised:
        module.transcribe_file(
            audio_path=audio_path,
            model_path=None,
            attempt_id="attempt-1",
            language="en",
        )

    assert raised.value.code.value == "model_not_installed"
    assert raised.value.actions == ("choose_another_gguf", "retry_faster_whisper")
    assert imports == []


def test_invalid_model_is_rejected_before_runtime_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    audio_path = tmp_path / "audio.wav"
    model_path = tmp_path / "changed.gguf"
    _write_pcm_wav(audio_path)
    imports: list[str] = []
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda _path: (_ for _ in ()).throw(ValueError("private path detail")),
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(module.TranscribeCppFailure) as raised:
        module.transcribe_file(
            audio_path=audio_path,
            model_path=model_path,
            attempt_id="attempt-1",
            language="en",
        )

    assert raised.value.code.value == "artifact_incompatible"
    assert raised.value.actions == ("choose_another_gguf", "retry_faster_whisper")
    assert imports == []


def test_native_import_failure_is_sanitized_and_path_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    secret = tmp_path / "secret-model.gguf"
    audio_path = tmp_path / "audio.wav"
    _write_pcm_wav(audio_path)
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )

    def unavailable(_name: str) -> object:
        raise ImportError(f"bad ABI near {secret}")

    monkeypatch.setattr(module.importlib, "import_module", unavailable)

    with pytest.raises(module.TranscribeCppFailure) as raised:
        module.transcribe_file(
            audio_path=audio_path,
            model_path=secret,
            attempt_id="attempt-1",
            language="en",
        )

    assert raised.value.code.value == "provider_unavailable"
    assert str(secret) not in str(raised.value)
    assert str(secret) not in repr(raised.value)
    assert "bad ABI" not in str(raised.value)
    rendered = "".join(traceback.format_exception(raised.type, raised.value, raised.tb))
    assert str(secret) not in rendered
    assert "bad ABI" not in rendered


def test_native_model_load_failure_closes_nothing_and_never_leaks_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tldw_chatbook.STT.transcribe_cpp")
    secret = tmp_path / "secret-model.gguf"
    audio_path = tmp_path / "audio.wav"
    _write_pcm_wav(audio_path)
    calls: list[tuple[str, object]] = []

    class FailingModel:
        def __init__(self, path: str, *, backend: str = "auto") -> None:
            calls.append(("model", path))
            raise RuntimeError(f"native load failed for {path}")

    runtime = SimpleNamespace(Model=FailingModel, set_log_callback=lambda _cb: None)
    monkeypatch.setitem(sys.modules, "transcribe_cpp", runtime)
    monkeypatch.setattr(
        module,
        "validate_local_gguf",
        lambda path: SimpleNamespace(
            path=path,
            metadata=SimpleNamespace(architecture="whisper"),
        ),
    )

    with pytest.raises(module.TranscribeCppFailure) as raised:
        module.transcribe_file(
            audio_path=audio_path,
            model_path=secret,
            attempt_id="attempt-1",
            language="en",
        )

    assert raised.value.code.value == "artifact_incompatible"
    assert raised.value.actions == ("choose_another_gguf", "retry_faster_whisper")
    assert str(secret) not in str(raised.value)
    assert str(secret) not in repr(raised.value)
    rendered = "".join(traceback.format_exception(raised.type, raised.value, raised.tb))
    assert str(secret) not in rendered
    assert "native load failed" not in rendered
    assert [name for name, _value in calls] == ["model"]
