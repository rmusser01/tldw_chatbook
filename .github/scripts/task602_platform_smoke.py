#!/usr/bin/env python3
"""Run the bounded native TASK-602 Parakeet smoke."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
from importlib import metadata
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Sequence


SCHEMA_VERSION = 1
FIXTURE_URL = (
    "https://download.pytorch.org/torchaudio/tutorial-assets/"
    "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)
FIXTURE_SHA256 = "c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14"
MAX_FIXTURE_BYTES = 1_000_000
V2_MODEL = "nemo-parakeet-tdt-0.6b-v2"
V3_MODEL = "nemo-parakeet-tdt-0.6b-v3"
V2_REFERENCE = {
    "artifact_id": "parakeet-v2",
    "revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
    "variant": "int8",
}
V3_REFERENCE = {
    "artifact_id": "parakeet-v3",
    "revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
    "variant": "int8",
}
VAD_REFERENCE = {
    "artifact_id": "silero-vad",
    "revision": "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    "variant": "f32",
}
EVIDENCE_NAMES = (
    "linux-x86_64",
    "linux-aarch64",
    "windows-x86_64",
    "macos-arm64",
    "macos-x86_64",
)


class SmokeFailure(RuntimeError):
    """Path-private stage identity for a native smoke failure."""

    def __init__(self, code: str, stage: str) -> None:
        self.code = code
        self.stage = stage
        super().__init__(code)


def _download_fixture(
    destination: Path,
    *,
    open_url: Callable[..., object] = urllib.request.urlopen,
) -> None:
    """Download, bound, hash, and structurally validate the speech fixture."""

    request = urllib.request.Request(
        FIXTURE_URL,
        headers={"User-Agent": "tldw-task602-evidence/1"},
    )
    try:
        with open_url(request, timeout=30.0) as response:
            payload = response.read(MAX_FIXTURE_BYTES + 1)
        if len(payload) > MAX_FIXTURE_BYTES:
            raise ValueError("fixture exceeds its byte bound")
        if hashlib.sha256(payload).hexdigest() != FIXTURE_SHA256:
            raise ValueError("fixture digest does not match")
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(payload)
        try:
            with wave.open(str(temporary), "rb") as audio:
                if (
                    audio.getnchannels() != 1
                    or audio.getsampwidth() != 2
                    or audio.getframerate() != 16_000
                    or audio.getnframes() <= 0
                ):
                    raise ValueError("fixture is not expected PCM16 audio")
        except (EOFError, wave.Error) as error:
            raise ValueError("fixture is not a valid WAVE file") from error
        temporary.replace(destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        destination.with_name(destination.name + ".tmp").unlink(missing_ok=True)
        raise


def _build_long_fixture(source: Path, destination: Path) -> None:
    """Create two speech regions separated by thirty seconds of silence."""

    try:
        with wave.open(str(source), "rb") as input_audio:
            channels = input_audio.getnchannels()
            sample_width = input_audio.getsampwidth()
            sample_rate = input_audio.getframerate()
            frames = input_audio.readframes(input_audio.getnframes())
        if (channels, sample_width, sample_rate) != (1, 2, 16_000) or not frames:
            raise ValueError("fixture has an unsupported WAVE format")
        silence = b"\0" * (30 * sample_rate * sample_width)
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(sample_rate)
            output.writeframes(frames + silence + frames)
    except (EOFError, wave.Error) as error:
        destination.unlink(missing_ok=True)
        raise ValueError("fixture could not produce long-form audio") from error


def _package_observation() -> tuple[dict[str, str], str]:
    """Record exact resolved packages and require the CPU provider."""

    for accelerator in ("onnxruntime-gpu", "onnxruntime-directml"):
        try:
            metadata.version(accelerator)
        except metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("an accelerator ONNX Runtime distribution resolved")
    packages = {
        name: metadata.version(name)
        for name in ("onnx-asr", "onnxruntime", "faster-whisper", "ctranslate2")
    }
    if packages["onnx-asr"] != "0.12.0":
        raise ValueError("the pinned onnx-asr runtime did not resolve")
    runtime = importlib.import_module("onnxruntime")
    providers = runtime.get_available_providers()
    if "CPUExecutionProvider" not in providers:
        raise ValueError("CPUExecutionProvider is unavailable")
    return packages, "CPUExecutionProvider"


def _probe_runtime() -> None:
    """Exercise the cheap optional-dependency probe without native imports."""

    native = {"onnx_asr", "onnxruntime"}
    before = native.intersection(sys.modules)
    optional_deps = importlib.import_module("tldw_chatbook.Utils.optional_deps")
    if not optional_deps.parakeet_onnx_deps_installed():
        raise ValueError("Parakeet ONNX dependency probe failed")
    if native.intersection(sys.modules) != before:
        raise ValueError("cheap runtime probe imported native modules")


async def _install_artifacts(root: Path):
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        run_parakeet_preflight,
        run_parakeet_provision,
    )
    from tldw_chatbook.Model_Artifacts.service import ModelArtifactService

    service = ModelArtifactService(root)
    for model in (V2_MODEL, V3_MODEL):
        report = await run_parakeet_preflight(model, "int8", core=service)
        if report.gating_errors:
            raise ValueError("artifact preflight is not grantable")
        await run_parakeet_provision(model, "int8", report, core=service)
    return service


def _provision_artifacts(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Provision exact roots through the production service and hold leases."""

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
        parakeet_vad_reference,
    )

    service = asyncio.run(asyncio.wait_for(_install_artifacts(root), timeout=1_800))
    v2_ref = parakeet_reference(V2_MODEL, "int8")
    v3_ref = parakeet_reference(V3_MODEL, "int8")
    vad_ref = parakeet_vad_reference()
    v2 = service.acquire(v2_ref)
    try:
        v3 = service.acquire(v3_ref)
    except BaseException:
        v2.close()
        raise
    artifacts = {
        "v2_int8": {
            "reference": v2_ref.to_dict(),
            "closure_fingerprint": v2.handle.closure_fingerprint,
        },
        "v3_int8": {
            "reference": v3_ref.to_dict(),
            "closure_fingerprint": v3.handle.closure_fingerprint,
        },
        "vad": vad_ref.to_dict(),
    }
    if (
        artifacts["v2_int8"]["reference"] != V2_REFERENCE
        or artifacts["v3_int8"]["reference"] != V3_REFERENCE
        or artifacts["vad"] != VAD_REFERENCE
    ):
        v3.close()
        v2.close()
        raise ValueError("provisioned artifact identity is not exact")
    return artifacts, {
        "v2": v2,
        "v3": v3,
        "vad_ref": vad_ref,
        "store_root": service.artifacts_path.parent,
    }


def _managed_dispatch(leased: object, store_root: Path, model_id: str):
    """Build the exact production executor dispatch for one held closure."""

    from tldw_chatbook.STT.contracts import ExecutionDevice
    from tldw_chatbook.STT.executor import ModelIdentity
    from tldw_chatbook.STT.parakeet_dispatch import ParakeetDispatch

    handle = leased.handle
    root = handle.root
    return ParakeetDispatch(
        identity=ModelIdentity(
            provider_id="parakeet-onnx",
            model_id=model_id,
            root_revision=root.revision,
            closure_fingerprint=handle.closure_fingerprint,
            precision="int8",
            device=ExecutionDevice.CPU,
        ),
        local_source=None,
        managed_store_root=store_root,
        managed_artifact_ref=(root.artifact_id, root.revision, root.variant),
        option_updates=MappingProxyType({}),
    )


def _pcm_source(path: Path):
    from tldw_chatbook.STT.contracts import BufferAudioSource

    with wave.open(str(path), "rb") as audio:
        return BufferAudioSource(
            audio.readframes(audio.getnframes()),
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
        )


def _submit_buffer(
    coordinator: object,
    executor: object,
    *,
    dispatch: object,
    source: object,
    language: str,
    attempt_id: str,
) -> tuple[dict[str, object], int]:
    """Submit one bounded buffer request through the production coordinator."""

    from tldw_chatbook.STT.executor import ExecutorFailure, ExecutorResult

    done = threading.Event()
    terminal: list[ExecutorResult | ExecutorFailure] = []

    def finish(value: ExecutorResult | ExecutorFailure) -> None:
        terminal.append(value)
        done.set()

    generation = coordinator.submit_library(
        attempt_id=attempt_id,
        job_id=None,
        source=source,
        identity=dispatch.identity,
        options={"language": language},
        local_source=dispatch.local_source,
        managed_store_root=dispatch.managed_store_root,
        managed_artifact_ref=dispatch.managed_artifact_ref,
        managed_dependency_refs=dispatch.managed_dependency_refs,
        on_result=finish,
        on_failure=finish,
    )
    if not done.wait(300.0):
        executor.force_stop(attempt_id)
        executor.wait_for_retirement(30.0)
        raise TimeoutError("native executor request timed out")
    result = terminal[0]
    if type(result) is ExecutorFailure:
        raise RuntimeError(result.code.value)
    if type(result) is not ExecutorResult:
        raise RuntimeError("native executor returned an invalid result")
    return result.payload, generation


def _executor_observations(
    resources: dict[str, object], fixture: Path
) -> dict[str, object]:
    """Exercise v2/v3 through the production coordinator and resident worker."""

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactInUseError,
        ModelArtifactService,
    )
    from tldw_chatbook.STT.dispatch_coordinator import LocalSTTDispatchCoordinator
    from tldw_chatbook.STT.executor import LocalSTTExecutor

    source = _pcm_source(fixture)
    store_root = resources["store_root"]
    v2_lease = resources["v2"]
    v3_lease = resources["v3"]
    v2_ref = v2_lease.handle.root
    v3_ref = v3_lease.handle.root
    vad_ref = resources["vad_ref"]
    v2 = _managed_dispatch(v2_lease, store_root, V2_MODEL)
    v3 = _managed_dispatch(v3_lease, store_root, V3_MODEL)
    executor = LocalSTTExecutor(
        startup_timeout=60.0,
        graceful_shutdown_timeout=10.0,
        force_stop_timeout=10.0,
    )
    coordinator = LocalSTTDispatchCoordinator(executor)
    try:
        v2_lease.close()
        v3_lease.close()
        lease_service = ModelArtifactService(store_root, lease_timeout_seconds=0.1)
        started = time.monotonic()
        first, first_generation = _submit_buffer(
            coordinator,
            executor,
            dispatch=v2,
            source=source,
            language="en",
            attempt_id="task602-v2-first",
        )
        second, second_generation = _submit_buffer(
            coordinator,
            executor,
            dispatch=v2,
            source=source,
            language="en",
            attempt_id="task602-v2-second",
        )
        v2_seconds = time.monotonic() - started
        if (
            not str(first.get("text", "")).strip()
            or not str(second.get("text", "")).strip()
            or first_generation != second_generation
            or executor.generation != first_generation
        ):
            raise ValueError("v2 resident executor reuse failed")
        for reference in (v2_ref, vad_ref):
            try:
                lease_service.delete(reference)
            except ArtifactInUseError:
                pass
            else:
                raise ValueError("resident artifact lease was not retained")

        started = time.monotonic()
        third, third_generation = _submit_buffer(
            coordinator,
            executor,
            dispatch=v3,
            source=source,
            language="fr",
            attempt_id="task602-v3",
        )
        v3_seconds = time.monotonic() - started
        provenance = third.get("transcription_provenance")
        if (
            not str(third.get("text", "")).strip()
            or third_generation <= second_generation
            or type(provenance) is not dict
            or provenance.get("requested_language") != "fr"
            or provenance.get("effective_language") != "auto"
            or provenance.get("detected_language") is not None
            or provenance.get("warnings") != ["requested_language_not_enforced"]
        ):
            raise ValueError("v3 resident executor semantics failed")
        resources["v2"] = lease_service.acquire(v2_ref)
        try:
            resources["v3"] = lease_service.acquire(v3_ref)
        except BaseException:
            resources["v2"].close()
            raise
        return {
            "checks": {
                "v2_int8_cpu": "passed",
                "v3_int8_cpu": "passed",
                "batch_reuse": "passed",
            },
            "durations": {
                "v2_int8_cpu": v2_seconds,
                "v3_int8_cpu": v3_seconds,
            },
        }
    finally:
        cleanup_failed = False
        try:
            coordinator.close()
        except BaseException:
            cleanup_failed = True
        try:
            executor.close()
        except BaseException:
            cleanup_failed = True
        if executor._unavailable:  # noqa: SLF001 - no public containment proof.
            cleanup_failed = True
        if cleanup_failed:
            raise SmokeFailure("cleanup", "cleanup")


def _close_resources(resources: Mapping[str, object]) -> None:
    cleanup_failed = False
    for name in ("v3", "v2"):
        try:
            resources[name].close()
        except BaseException:
            cleanup_failed = True
    if cleanup_failed:
        raise SmokeFailure("cleanup", "cleanup")


def _load_runtime(model_id: str, leased, vad_ref):
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    paths = dict(leased.handle.paths)
    root_ref = leased.handle.root
    dependencies = tuple(
        reference.lease_key()
        for reference in leased.handle.closure
        if reference != root_ref
    )
    return ParakeetOnnxRuntime.load(
        model_root=paths[root_ref],
        vad_root=paths[vad_ref],
        model_id=model_id,
        precision="int8",
        artifact_root=root_ref.lease_key(),
        artifact_dependencies=dependencies,
    )


def _runtime_observations(
    resources: Mapping[str, object],
    long_fixture: Path,
) -> dict[str, object]:
    """Exercise real long-form VAD, cancellation, and retry wiring."""

    from tldw_chatbook.STT.parakeet_onnx import (
        ParakeetOnnxCancelled,
        ParakeetOnnxFailure,
    )

    v2_handle = resources["v2"]
    vad_ref = resources["vad_ref"]
    v2_runtime = None
    timings: dict[str, float] = {}
    try:
        v2_runtime = _load_runtime(V2_MODEL, v2_handle, vad_ref)
        started = time.monotonic()
        long_result = v2_runtime.transcribe(
            audio_path=long_fixture,
            attempt_id="task602-long",
            language="en",
            timestamps=True,
        )
        timings["long_form_vad"] = time.monotonic() - started
        if not long_result.produced_capabilities.vad or len(long_result.segments) < 2:
            raise ValueError("long-form VAD did not produce two speech segments")

        cancellation_checks = 0

        def cancelled() -> bool:
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 2

        try:
            v2_runtime.transcribe(
                audio_path=long_fixture,
                attempt_id="task602-cancel",
                language="en",
                timestamps=True,
                is_cancelled=cancelled,
            )
        except ParakeetOnnxCancelled:
            pass
        else:
            raise ValueError("cancellation was not observed before a second batch")
        if cancellation_checks != 2:
            raise ValueError("cancellation did not stop at the second segment")

        managed_vad = v2_runtime._vad
        v2_runtime._vad = None
        try:
            with_retry = None
            try:
                v2_runtime.transcribe(
                    audio_path=long_fixture,
                    attempt_id="task602-retry",
                    language="en",
                    timestamps=True,
                )
            except ParakeetOnnxFailure as error:
                with_retry = error
            if with_retry is None or with_retry.error_detail.get("actions") != [
                "retry_faster_whisper"
            ]:
                raise ValueError("eligible failure did not expose exact retry wiring")
            provenance = with_retry.stt_failure_provenance
            if (
                provenance.get("provider_id") != "parakeet-onnx"
                or provenance.get("model_id") != V2_MODEL
                or provenance.get("error_code") != "artifact_incompatible"
            ):
                raise ValueError("eligible failure provenance is invalid")
        finally:
            v2_runtime._vad = managed_vad

        return {
            "checks": {
                "long_form_vad": "passed",
                "cancellation": "passed",
                "retry_wiring": "passed",
            },
            "durations": timings,
        }
    finally:
        if v2_runtime is not None:
            v2_runtime.close()


def _runtime_observation_child(
    connection: object,
    model_root: Path,
    vad_root: Path,
    root_ref: object,
    vad_ref: object,
    long_fixture: Path,
) -> None:
    """Run the direct native checks in one disposable spawned process."""

    outcome: tuple[str, object | None] = ("smoke_execution", None)
    try:
        handle = SimpleNamespace(
            root=root_ref,
            closure=(root_ref, vad_ref),
            paths=((root_ref, model_root), (vad_ref, vad_root)),
        )
        result = _runtime_observations(
            {"v2": SimpleNamespace(handle=handle), "vad_ref": vad_ref},
            long_fixture,
        )
        outcome = ("passed", result)
    except SmokeFailure as error:
        outcome = (error.code, None)
    except BaseException:
        pass
    try:
        connection.send(outcome)
    except BaseException:
        pass
    finally:
        connection.close()


def _terminate_process(process: object) -> bool:
    """Boundedly stop one exact child and prove it is no longer alive."""

    try:
        if not process.is_alive():
            return True
    except BaseException:
        return False
    try:
        process.terminate()
        process.join(10.0)
    except BaseException:
        pass
    try:
        if process.is_alive():
            process.kill()
            process.join(10.0)
        return not process.is_alive()
    except BaseException:
        return False


def _bounded_runtime_observations(
    resources: Mapping[str, object],
    long_fixture: Path,
    *,
    context: object | None = None,
    timeout: float = 300.0,
) -> dict[str, object]:
    """Run direct native checks behind a bounded, disposable process."""

    handle = resources["v2"].handle
    paths = dict(handle.paths)
    root_ref = handle.root
    vad_ref = resources["vad_ref"]
    context = context or multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_runtime_observation_child,
        args=(
            send,
            paths[root_ref],
            paths[vad_ref],
            root_ref,
            vad_ref,
            long_fixture,
        ),
    )
    process.start()
    send.close()
    forced_termination = False
    try:
        if not receive.poll(timeout):
            raise TimeoutError("native runtime observation timed out")
        try:
            outcome, result = receive.recv()
        except (EOFError, OSError, TypeError, ValueError) as error:
            raise RuntimeError("native runtime observation failed") from error
    finally:
        receive.close()
        try:
            process.join(10.0)
            alive = process.is_alive()
        except BaseException as error:
            raise SmokeFailure("cleanup", "cleanup") from error
        if alive:
            if not _terminate_process(process):
                raise SmokeFailure("cleanup", "cleanup")
            forced_termination = True
    if forced_termination:
        raise SmokeFailure("cleanup", "cleanup")
    if outcome == "cleanup":
        raise SmokeFailure("cleanup", "cleanup")
    if outcome != "passed" or type(result) is not dict:
        raise RuntimeError("native runtime observation failed")
    return result


def _enable_runtime_offline() -> dict[str, str | None]:
    """Disable common model-hub acquisition paths after provisioning."""

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    return previous


def run_smoke(evidence_name: str, workspace: Path) -> dict[str, object]:
    """Run the production-path smoke and return allowlisted observations.

    Args:
        evidence_name: Expected native platform evidence identity.
        workspace: Lane-owned temporary workspace for acquisition and runtime.

    Returns:
        The bounded observations accepted by the evidence normalizer.

    Raises:
        SmokeFailure: If acquisition, execution, or cleanup cannot be proven.
        ValueError: If the evidence identity is unknown.
    """

    if evidence_name not in EVIDENCE_NAMES:
        raise ValueError("unknown evidence name")
    started = time.monotonic()
    try:
        _probe_runtime()
        packages, provider = _package_observation()
    except BaseException as error:
        raise SmokeFailure("smoke_execution", "runtime_smoke") from error
    fixture = workspace / "fixture.wav"
    long_fixture = workspace / "long.wav"
    try:
        _download_fixture(fixture)
        _build_long_fixture(fixture, long_fixture)
    except BaseException as error:
        raise SmokeFailure("fixture_download", "fixture_download") from error
    try:
        artifacts, resources = _provision_artifacts(workspace / "artifacts")
    except BaseException as error:
        raise SmokeFailure("artifact_acquisition", "artifact_acquisition") from error
    acquisition_finished = time.monotonic()
    previous_offline = _enable_runtime_offline()
    try:
        executor_runtime = _executor_observations(resources, fixture)
        runtime = _bounded_runtime_observations(resources, long_fixture)
    except SmokeFailure:
        raise
    except BaseException as error:
        raise SmokeFailure("smoke_execution", "runtime_smoke") from error
    finally:
        try:
            _close_resources(resources)
        finally:
            _restore_process(previous_offline)
    finished = time.monotonic()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "failure_code": None,
        "failure_stage": None,
        "packages": packages,
        "execution_provider": provider,
        "artifacts": artifacts,
        "checks": {
            "package_resolution": "passed",
            "runtime_probe": "passed",
            **executor_runtime["checks"],
            **runtime["checks"],
        },
        "durations_seconds": {
            "acquisition": acquisition_finished - started,
            **executor_runtime["durations"],
            **runtime["durations"],
            "total": finished - started,
        },
        "cleanup": "passed",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _failure(code: str, stage: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "failure_code": code,
        "failure_stage": stage,
    }


def _isolate_process(workspace: Path) -> dict[str, str | None]:
    """Confine application config and cache writes to the owned workspace."""

    home = workspace / "home"
    config = workspace / "config"
    cache = workspace / "cache"
    for directory in (home, config, cache):
        directory.mkdir(parents=True, exist_ok=True)
    values = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(config),
        "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"),
        "TLDW_CONFIG_PATH": str(config / "config.toml"),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    return previous


def _restore_process(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded smoke without emitting path-bearing exception details."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-name", required=True, choices=EVIDENCE_NAMES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    workspace = Path(tempfile.mkdtemp(prefix="task602-native-"))
    previous_environment = _isolate_process(workspace)
    code = 0
    cleanup_allowed = True
    try:
        try:
            result = run_smoke(args.evidence_name, workspace)
        except SmokeFailure as error:
            result = _failure(error.code, error.stage)
            code = 1
            cleanup_allowed = error.code != "cleanup"
        except BaseException:
            result = _failure("smoke_execution", "runtime_smoke")
            code = 1
        if cleanup_allowed:
            try:
                shutil.rmtree(workspace)
            except BaseException:
                result = _failure("cleanup", "cleanup")
                code = 1
        _write_json(args.output, result)
        return code
    except (OSError, ValueError):
        return 1
    finally:
        _restore_process(previous_environment)


if __name__ == "__main__":
    raise SystemExit(main())
