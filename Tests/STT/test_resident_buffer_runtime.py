"""Public buffer-owner tests with real source/lease checks and fake native STT."""

from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest


def installed_sources(tmp_path: Path, precision="int8"):
    from Tests.Model_Artifacts.test_service import (
        install_descriptor_payload,
        single_file_descriptor,
    )
    from Tests.STT.test_parakeet_sources import _descriptor
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
        parakeet_vad_reference,
    )
    from tldw_chatbook.Model_Artifacts import ArtifactRole, ModelArtifactService
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    key = ParakeetSourceKey.from_values("nemo-parakeet-tdt-0.6b-v2", precision)
    root = replace(
        _descriptor(key),
        reference=parakeet_reference(key.model_id, precision),
        dependencies=(parakeet_vad_reference(),),
    )
    dependency = single_file_descriptor(
        parakeet_vad_reference(), ArtifactRole.DEPENDENCY, b"dependency"
    )
    artifacts = ModelArtifactService(tmp_path / "store", lease_timeout_seconds=0.01)
    install_descriptor_payload(artifacts, tmp_path, dependency, b"dependency")
    install_descriptor_payload(artifacts, tmp_path, root, b"model")
    artifacts.activate(root.reference)
    external = tmp_path / "external"
    external.mkdir()
    (external / "model.onnx").write_bytes(b"model")
    return artifacts, root, dependency, external


def source_service(artifacts, root, external, preferred="managed"):
    from tldw_chatbook.STT.contracts import ExecutionDevice
    from tldw_chatbook.STT.executor import ModelIdentity
    from tldw_chatbook.STT.parakeet_dispatch import ParakeetDispatch
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceKey,
        ParakeetSourceService,
    )

    key = ParakeetSourceKey.from_values(root.model_id, root.precision)
    table = {
        key.value: {
            "model_id": root.model_id,
            "precision": root.precision,
            "directory": str(external),
            "preferred_source": preferred,
        }
    }

    def resolve(*, model_id, precision, model_dir):
        assert (model_id, precision, model_dir) == (root.model_id, root.precision, None)
        with artifacts.acquire(root.reference) as leased:
            fingerprint = leased.handle.closure_fingerprint
        return ParakeetDispatch(
            identity=ModelIdentity(
                "parakeet-onnx",
                model_id,
                root.reference.revision,
                fingerprint,
                precision,
                ExecutionDevice.CPU,
            ),
            local_source=None,
            managed_store_root=artifacts.artifacts_path.parent,
            managed_artifact_ref=(
                root.reference.artifact_id,
                root.reference.revision,
                root.reference.variant,
            ),
            option_updates={
                "transcription_context": {"batch_id": "buffer-batch"},
                "source_option": "preserved",
            },
        )

    return ParakeetSourceService(
        read_setting=lambda section, name, default: (
            table
            if (section, name) == ("transcription", "parakeet_external_sources")
            else default
        ),
        write_settings=lambda _: pytest.fail(
            "source lookup must not write configuration"
        ),
        descriptor_for=lambda model, precision: root,
        active_managed=lambda model, precision: artifacts.artifact_path(root.reference),
        dispatch_resolver=resolve,
        managed_service=artifacts,
        vad_ready=lambda: True,
    )


def fake_native_loader(record, *, mode="normal", entered=None, release=None):
    """Keep the real resident/provider/provenance path below a fake native model."""
    from tldw_chatbook.STT.contracts import (
        ExecutionDevice,
        ProducedCapabilities,
        TimestampGranularity,
        TranscriptionProvenance,
        TranscriptionResult,
        TranscriptionTask,
        TranscriptionTimings,
    )
    from tldw_chatbook.STT.parakeet_onnx import ParakeetBufferResult

    def load(**loaded):
        record(("load", os.getpid(), loaded))

        class Native:
            def transcribe_buffer(self, **kwargs):
                record(("infer", os.getpid(), kwargs))
                if entered is not None and mode != "close-hang":
                    entered.set()
                if release is not None and mode != "close-hang":
                    assert release.wait(5), "test did not release fake inference"
                return ParakeetBufferResult(
                    normalized=TranscriptionResult(
                        text="onnx result",
                        segments=(),
                        provenance=TranscriptionProvenance(
                            schema_version=1,
                            attempt_id=kwargs["attempt_id"],
                            batch_id=None,
                            job_id=kwargs["job_id"],
                            retry_of_attempt_id=None,
                            retry_of_job_id=None,
                            provider_id="parakeet-onnx",
                            model_id=loaded["model_id"],
                            artifact_root=loaded["artifact_root"],
                            artifact_dependencies=loaded["artifact_dependencies"],
                            precision=loaded["precision"],
                            requested_device=ExecutionDevice.CPU,
                            effective_device=ExecutionDevice.CPU,
                            requested_language=kwargs["language"],
                            effective_language="en",
                            detected_language=None,
                            task=TranscriptionTask.TRANSCRIBE,
                        ),
                        produced_capabilities=ProducedCapabilities(
                            timestamps=TimestampGranularity.NONE,
                            punctuation=True,
                            capitalization=True,
                            vad=False,
                            diarization=False,
                        ),
                        duration_seconds=0.01,
                        timings=TranscriptionTimings(total_seconds=0.001),
                    ),
                    logical_segments=("onnx result",),
                )

            def close(self):
                record(("native-close", os.getpid()))
                if mode == "close-error":
                    raise RuntimeError("private failed native cleanup")
                if mode == "close-hang":
                    import time

                    if entered is not None:
                        entered.set()
                    # Never terminate a child while it owns a shared Event's
                    # condition: that can strand the parent's test teardown.
                    time.sleep(30)

        return Native()

    return load


def real_facade_process_runtime(
    events, store, root, external, options, mode, entered, release, model, language
):
    """Child-only guarded native replacement; production builds facade and owner."""
    import multiprocessing.process
    import subprocess
    import sys
    from Tests.Audio.test_local_voice_stt_process import install_child_guards

    install_child_guards()
    # The facade's optional import probes remain unavailable; they never import
    # model packages. Its real ONNX load boundary is replaced below.
    for name in (
        "faster_whisper",
        "onnx_asr",
        "torch",
        "transformers",
        "nemo",
        "parakeet_mlx",
        "lightning_whisper_mlx",
    ):
        sys.modules[name] = None

    def no_extra_process(*args, **kwargs):
        raise AssertionError("the model owner must not spawn a fourth process")

    multiprocessing.process.BaseProcess.start = no_extra_process
    subprocess.Popen = no_extra_process
    from tldw_chatbook.Audio.parakeet_voice_worker import _load_runtime
    from tldw_chatbook.Local_Ingestion import transcription_service as facade
    from tldw_chatbook.Model_Artifacts import ModelArtifactService
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    class Legacy:
        config = {
            "default_provider": "parakeet-onnx",
            "device": "cpu",
            "compute_type": "int8",
        }

        def cleanup(self):
            events.send(("facade-cleanup", os.getpid()))

    def record(event):
        if len(event) > 2:
            details = dict(event[2])
            details.pop("is_cancelled", None)
            event = (*event[:2], details)
        events.send(event)

    artifacts = ModelArtifactService(store, lease_timeout_seconds=0.01)
    facade._LegacyTranscriptionBackend = Legacy
    facade.ParakeetSourceService = lambda: source_service(artifacts, root, external)
    ParakeetOnnxRuntime.load = staticmethod(
        fake_native_loader(record, mode=mode, entered=entered, release=release)
    )
    return _load_runtime(model, language, provider="parakeet-onnx", options=options)


def retiring_facade_process_runtime(events, control, wire_failure, *args):
    """Expose the real failed-close retirement interval without shared locks."""
    import gc
    import multiprocessing.util

    runtime = real_facade_process_runtime(events, *args)
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    connection_type = type(control)
    actual_send, actual_close = connection_type.send, connection_type.close
    rpc_connection = None

    def send(connection, payload):
        nonlocal rpc_connection
        if type(payload) is tuple and payload[0] == "cleanup":
            rpc_connection = connection
            if wire_failure == "send":
                raise KeyboardInterrupt()
        return actual_send(connection, payload)

    def close(connection):
        if connection is rpc_connection and wire_failure == "close":
            raise KeyboardInterrupt()
        return actual_close(connection)

    connection_type.send, connection_type.close = send, close
    native_owners = []
    native_load = ParakeetOnnxRuntime.load

    def retained_load(**kwargs):
        native = native_load(**kwargs)
        native_owners.append(native)
        return native

    ParakeetOnnxRuntime.load = staticmethod(retained_load)
    actual_exit = os._exit

    def pre_exit(code):
        try:
            events.send(("pre-exit", os.getpid(), code))
            if control.poll(5):
                control.recv()
        finally:
            actual_exit(code)

    def late_finalizer():
        # Retain a native/background owner independently after _serve locals
        # disappear. The protecting lease must not disappear first.
        gc.collect()
        events.send(("late-finalizer", os.getpid(), len(native_owners)))
        if control.poll(5):
            control.recv()

    os._exit = pre_exit
    multiprocessing.util.Finalize(None, late_finalizer, exitpriority=1)
    return runtime


@pytest.fixture
def native(monkeypatch):
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    calls = []
    monkeypatch.setattr(ParakeetOnnxRuntime, "load", fake_native_loader(calls.append))
    return calls


def pcm():
    from tldw_chatbook.STT.contracts import BufferAudioSource

    return BufferAudioSource(bytes(960), 48_000)


def test_real_facade_owner_keeps_managed_root_and_vad_leased_across_idle_reuse(
    tmp_path, native
):
    from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService
    from tldw_chatbook.Model_Artifacts import ArtifactInUseError
    from tldw_chatbook.STT.executor_worker import ResidentBufferRuntime

    artifacts, root, dependency, external = installed_sources(tmp_path)
    sources = source_service(artifacts, root, external)
    owner = ResidentBufferRuntime()
    facade = TranscriptionService(
        local_buffer_owner=owner, parakeet_source_service=sources
    )
    try:
        for language in ("en", "fr"):
            result = facade.transcribe_buffer(
                bytes(960), 48_000, provider="parakeet-onnx", language=language
            )
            assert result["text"] == "onnx result"
            assert result["logical_segments"] == ("onnx result",)
            provenance = result["transcription_provenance"]
            assert provenance["requested_language"] == language
            assert provenance["batch_id"] == "buffer-batch"
            assert (
                provenance["artifact_root"]["artifact_id"] == root.reference.artifact_id
            )
            assert (
                provenance["artifact_dependencies"][0]["artifact_id"]
                == dependency.reference.artifact_id
            )
            for reference in (root.reference, dependency.reference):
                with pytest.raises(ArtifactInUseError):
                    artifacts.delete(reference)
        assert [call[0] for call in native] == ["load", "infer", "infer"]
        assert native[1][2]["attempt_id"] != native[2][2]["attempt_id"]
        assert native[1][2]["segment_end_frames"] == (480,)
        assert native[2][2]["language"] == "fr"
    finally:
        facade.cleanup()
        sources.close()
    facade.cleanup()
    assert [call[0] for call in native].count("native-close") == 1
    artifacts.delete(root.reference)
    artifacts.delete(dependency.reference)


@pytest.mark.parametrize("precision", ["int8", "f32"])
@pytest.mark.parametrize(
    "preferred, override, expect_external",
    [
        ("external", False, True),
        ("managed", False, False),
        ("managed", True, True),
    ],
)
def test_facade_retains_real_source_precedence_precision_and_omitted_defaults(
    tmp_path, monkeypatch, native, precision, preferred, override, expect_external
):
    import tldw_chatbook.Local_Ingestion.transcription_service as module
    from tldw_chatbook.STT.executor_worker import ResidentBufferRuntime

    artifacts, root, dependency, external = installed_sources(tmp_path, precision)
    sources = source_service(artifacts, root, external, preferred)
    monkeypatch.setattr(
        module,
        "get_cli_setting",
        lambda key, default=None: (
            precision if key == "transcription.default_precision" else default
        ),
    )
    facade = module.TranscriptionService(
        local_buffer_owner=ResidentBufferRuntime(), parakeet_source_service=sources
    )
    facade.config["default_provider"] = "parakeet-onnx"
    try:
        result = facade.transcribe_buffer(
            bytes(960), 48_000, **({"model_dir": str(external)} if override else {})
        )
        loaded = native[0][2]
        assert loaded["model_root"] == (
            external if expect_external else artifacts.artifact_path(root.reference)
        )
        assert loaded["vad_root"] == artifacts.artifact_path(dependency.reference)
        assert loaded["precision"] == precision
        assert loaded["model_id"] == "nemo-parakeet-tdt-0.6b-v2"
        assert result["transcription_provenance"]["requested_language"] == "en"
    finally:
        facade.cleanup()
        sources.close()


@pytest.mark.parametrize("mutation", ["identity", "closure", "external-file", "vad"])
def test_resident_rejects_changed_identity_source_or_dependency_without_hot_swap(
    tmp_path, native, mutation
):
    from tldw_chatbook.STT.executor_worker import ResidentBufferRuntime
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    artifacts, root, dependency, external = installed_sources(tmp_path)
    sources = source_service(artifacts, root, external, "external")
    dispatch = sources.resolve(ParakeetSourceKey.V2_INT8)
    owner = ResidentBufferRuntime()
    try:
        owner.transcribe_buffer(source=pcm(), dispatch=dispatch, language="en")
        if mutation == "identity":
            dispatch = replace(
                dispatch,
                identity=replace(
                    dispatch.identity, model_id="nemo-parakeet-tdt-0.6b-v3"
                ),
            )
        elif mutation == "closure":
            dispatch = replace(
                dispatch,
                identity=replace(dispatch.identity, closure_fingerprint="changed"),
            )
        elif mutation == "external-file":
            replacement = external / "replacement"
            replacement.write_bytes(b"model")
            replacement.replace(external / "model.onnx")
        else:
            (
                artifacts.artifact_path(dependency.reference) / dependency.files[0].path
            ).write_bytes(b"corrupted!")
        with pytest.raises((RuntimeError, ValueError)):
            owner.transcribe_buffer(source=pcm(), dispatch=dispatch, language="fr")
        assert [call[0] for call in native] == ["load", "infer"]
    finally:
        owner.close()
        sources.close()


def test_provider_options_and_per_request_metadata_use_the_existing_request_path(
    tmp_path, monkeypatch
):
    from tldw_chatbook.STT import executor_worker as module
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    artifacts, root, dependency, external = installed_sources(tmp_path)
    sources = source_service(artifacts, root, external)
    dispatch = sources.resolve(ParakeetSourceKey.V2_INT8)
    seen = []
    payload = {"text": "owned", "transcription_provenance": {"retained": True}}

    def builder(request, model_root, handle, cancelled):
        seen.append(request)
        return module.ProviderRuntime(
            runner=lambda *_args: pytest.fail("file runner must not execute"),
            close=lambda: None,
            buffer_runner=lambda source, **kwargs: (
                seen.append((source, kwargs)) or payload
            ),
        )

    monkeypatch.setattr(module, "_default_provider_builder", builder)
    owner = module.ResidentBufferRuntime()
    try:
        assert (
            owner.transcribe_buffer(source=pcm(), dispatch=dispatch, language="fr")
            is payload
        )
        request = seen[0]
        assert request.identity is dispatch.identity
        assert request.options == {
            "transcription_context": {"batch_id": "buffer-batch"},
            "source_option": "preserved",
            "language": "fr",
        }
        assert request.managed_artifact_ref == dispatch.managed_artifact_ref
        assert request.managed_store_root == dispatch.managed_store_root
        assert seen[1][1]["transcription_context"] == {"batch_id": "buffer-batch"}
        assert seen[1][1]["attempt_id"] == request.attempt_id
    finally:
        owner.close()
        sources.close()


def test_inference_keeps_root_and_vad_deletion_blocked_until_actual_completion(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.Model_Artifacts import ArtifactInUseError
    from tldw_chatbook.STT.executor_worker import ResidentBufferRuntime
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    artifacts, root, dependency, external = installed_sources(tmp_path)
    sources = source_service(artifacts, root, external)
    dispatch = sources.resolve(ParakeetSourceKey.V2_INT8)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        ParakeetOnnxRuntime,
        "load",
        fake_native_loader(lambda _: None, entered=entered, release=release),
    )
    owner = ResidentBufferRuntime()
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(
            owner.transcribe_buffer, source=pcm(), dispatch=dispatch, language="en"
        )
        try:
            assert entered.wait(2)
            for reference in (root.reference, dependency.reference):
                with pytest.raises(ArtifactInUseError):
                    artifacts.delete(reference)
        finally:
            release.set()
            future.result(timeout=3)
            owner.close()
            sources.close()


@pytest.mark.parametrize("secondary_failure", [False, True])
def test_failed_native_close_keeps_original_provider_and_leases_without_retry(
    tmp_path, monkeypatch, secondary_failure
):
    from tldw_chatbook.Local_Ingestion import transcription_service as module
    from tldw_chatbook.Model_Artifacts import ArtifactInUseError
    from tldw_chatbook.STT.executor_worker import ResidentBufferRuntime
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    artifacts, root, dependency, external = installed_sources(tmp_path)
    sources = source_service(artifacts, root, external)
    calls = []
    monkeypatch.setattr(
        ParakeetOnnxRuntime,
        "load",
        fake_native_loader(calls.append, mode="close-error"),
    )
    owner = ResidentBufferRuntime()
    monkeypatch.setattr(module, "ParakeetSourceService", lambda: sources)
    facade = module.TranscriptionService(local_buffer_owner=owner)
    source_close = sources.close
    source_closes = []

    def close_source():
        source_closes.append(True)
        source_close()
        if secondary_failure:
            raise OSError("secondary source cleanup failed")

    monkeypatch.setattr(sources, "close", close_source)
    try:
        facade.transcribe_buffer(bytes(960), 48_000, provider="parakeet-onnx")
        for _ in range(2):
            with pytest.raises(RuntimeError):
                facade.cleanup()
        for reference in (root.reference, dependency.reference):
            with pytest.raises(ArtifactInUseError):
                artifacts.delete(reference)
        assert [call[0] for call in calls].count("native-close") == 1
        assert source_closes == []
    finally:
        # This test owns the fake; explicitly release test-held leases only after
        # assertions. Production retains them until the model process exits.
        if owner._resident is not None and owner._resident.lease is not None:
            owner._resident.lease.close()
        source_close()
