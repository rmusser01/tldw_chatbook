from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import socket
import struct
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from Tests.TTS.fixtures.fake_audiocpp_server import write_executable_wrapper
from tldw_chatbook.TTS.adapters import audio_cpp as audio_cpp_adapter_module
from tldw_chatbook.TTS import audio_cpp_supervisor as supervisor_module
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS._async_lifecycle import current_shutdown_deadline
from tldw_chatbook.TTS.adapter_types import (
    _AdmittedAudioCppCloneRequest,
    _new_admitted_audio_cpp_clone_request,
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSRegistryClosedError,
    TTSRequest,
)
from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppAcceptedPackage,
    AudioCppSettingsConfig,
)
from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY
from tldw_chatbook.TTS.audio_cpp_package_scanner import (
    scan_audio_cpp_package_root,
)
from tldw_chatbook.TTS.audio_cpp_guided_launch import (
    materialize_audio_cpp_guided_launch,
)
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    AudioCppGenerationHooks,
    AudioCppProcessAdmissionSnapshot,
    AudioCppProcessSnapshot,
    AudioCppReadyEndpoint,
    AudioCppSupervisor,
    _AudioCppGenerationChanged,
    _OwnedAudioCppProcess,
)
from tldw_chatbook.TTS.effective_settings import (
    TTSCharacterProfileSelection,
    TTSSelectionOverrides,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneReferenceMaterializer,
)
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.TTS_Generation import (
    AudioCppRuntimeObservation,
    TTSService,
    TTSSettingsPersistenceOutcome,
)

# Network opt-in (task-15111): this module talks to `fake_audiocpp_server`,
# an in-process HTTP server on an ephemeral loopback port.
# The autouse guard in Tests/conftest.py denies egress by default; every address
# these tests reach is a port this process itself is listening on.
pytestmark = pytest.mark.allow_network


def _wav() -> bytes:
    data = b"\x00\x00\x00\x00"
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        1,
        24_000,
        48_000,
        2,
        16,
    )
    payload = b"WAVE" + fmt + struct.pack("<4sI", b"data", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


class _BytesStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body

    async def aclose(self) -> None:
        return None


def _response(
    body: bytes,
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(200, headers=headers, stream=_BytesStream(body))


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return _response(
            json.dumps({"status": "ok", "backend": "cpu", "models": 1}).encode(),
        )
    if request.url.path == "/v1/models":
        return _response(
            json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "model",
                            "object": "model",
                            "owned_by": "engine",
                            "family": "pocket_tts",
                            "task": "tts",
                            "mode": "native",
                        }
                    ],
                }
            ).encode(),
        )
    if request.url.path == "/v1/audio/voices":
        return _response(b'{"voices":["default"]}')
    if request.url.path == "/v1/audio/speech":
        return _response(
            _wav(),
            headers={"Content-Type": "audio/wav"},
        )
    raise AssertionError(f"unexpected path: {request.url.path}")


def _no_model_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return _response(
            json.dumps({"status": "ok", "backend": "cpu", "models": 0}).encode(),
        )
    if request.url.path == "/v1/models":
        return _response(b'{"object":"list","data":[]}')
    raise AssertionError(f"unexpected path: {request.url.path}")


class _PreparationSupervisor:
    def __init__(self) -> None:
        self._application_owner_token = (
            supervisor_module._AUDIO_CPP_SUPERVISOR_OWNER_TOKEN
        )
        self.state = "stopped"
        self.lifecycle_epoch = 0
        self.process_generation = 0
        self.observation_version = 0
        self.ensure_calls = 0
        self.ensure_requirements: list[AudioCppProcessAdmissionSnapshot | None] = []
        self.launches = 0
        self.stop_calls = 0
        self.close_calls = 0
        self.hooks: AudioCppGenerationHooks | None = None
        self.change_before_ensure = False
        self.admission_observer: Callable[[int], None] | None = None
        self.admission_calls = 0
        self.draining_started = asyncio.Event()
        self.events: list[str] = []
        self.inflight_probe_gate: asyncio.Event | None = None
        self.stop_failures_remaining = 0
        self.deadline_observations: list[float | None] = []
        self.wait_closed_calls = 0
        self.tts_capability = "unknown"
        self.suppressed_clone_generations: list[int] = []

    def admission_snapshot(self) -> AudioCppProcessAdmissionSnapshot:
        self.admission_calls += 1
        if self.admission_observer is not None:
            self.admission_observer(self.admission_calls)
        return AudioCppProcessAdmissionSnapshot(
            lifecycle_epoch=self.lifecycle_epoch,
            process_generation=self.process_generation,
            state=self.state,  # type: ignore[arg-type]
            stage_application_eligible=(
                self.state in {"stopped", "unavailable"} and self.hooks is None
            ),
        )

    def snapshot(self) -> AudioCppProcessSnapshot:
        return AudioCppProcessSnapshot(
            state=self.state,  # type: ignore[arg-type]
            process_generation=self.process_generation,
            observation_version=self.observation_version,
            endpoint=(
                f"http://127.0.0.1:{19_000 + self.process_generation}"
                if self.state in {"running", "unhealthy"}
                else None
            ),
            tts_capability=(
                self.tts_capability if self.state == "running" else "unknown"
            ),
            consecutive_health_failures=0,
            last_failure=None,
            diagnostics=(),
            dropped_diagnostic_lines=0,
        )

    def suppress_clone_diagnostics(self, process_generation: int) -> bool:
        if process_generation != self.process_generation or self.state not in {
            "running",
            "draining",
        }:
            return False
        self.suppressed_clone_generations.append(process_generation)
        return True

    async def ensure_running(
        self,
        launch: Any,
        *,
        generation_hooks_factory: Callable[[int], Awaitable[AudioCppGenerationHooks]],
        require_existing: AudioCppProcessAdmissionSnapshot | None = None,
    ) -> AudioCppReadyEndpoint:
        self.ensure_calls += 1
        self.ensure_requirements.append(require_existing)
        if self.change_before_ensure:
            self.change_before_ensure = False
            if self.hooks is not None:
                hooks, self.hooks = self.hooks, None
                await hooks.cleanup()
            self.state = "unavailable"
            self.lifecycle_epoch += 1
            raise _AudioCppGenerationChanged
        if require_existing is not None and (
            require_existing.lifecycle_epoch != self.lifecycle_epoch
            or require_existing.process_generation != self.process_generation
            or require_existing.state not in {"starting", "running", "unhealthy"}
            or (require_existing.state != self.state and self.state != "draining")
        ):
            raise _AudioCppGenerationChanged
        if self.state == "draining" and require_existing is not None:
            return self._endpoint(launch.base_url)
        if self.state in {"draining", "stopping"}:
            raise TTSProviderReconfiguringError(
                "The audio.cpp provider is reconfiguring"
            )
        if self.state == "running" and self.hooks is not None:
            return self._endpoint(launch.base_url)
        self.state = "starting"
        self.process_generation += 1
        self.observation_version += 1
        self.launches += 1
        self.hooks = await generation_hooks_factory(self.process_generation)
        assert await self.hooks.health_probe()
        self.tts_capability = await self.hooks.contract_probe()
        self.state = "running"
        return self._endpoint(launch.base_url)

    async def begin_draining(self) -> None:
        self.events.append("draining")
        if self.hooks is not None:
            self.state = "draining"
        self.draining_started.set()

    async def stop(
        self,
        *,
        application_shutdown: bool = False,
        expected_process_generation: int | None = None,
    ) -> None:
        self.deadline_observations.append(current_shutdown_deadline())
        self.events.append(
            "terminal_stop" if application_shutdown else "generation_stop"
        )
        if self.stop_failures_remaining:
            self.stop_failures_remaining -= 1
            raise RuntimeError("simulated lifecycle failure")
        if expected_process_generation is not None and (
            expected_process_generation != self.process_generation
        ):
            return
        self.stop_calls += 1
        self.lifecycle_epoch += 1
        self.state = "stopping"
        if self.inflight_probe_gate is not None:
            await self.inflight_probe_gate.wait()
            self.events.append("probe_joined")
            self.inflight_probe_gate = None
        if self.hooks is not None:
            hooks, self.hooks = self.hooks, None
            await hooks.cleanup()
        self.state = "stopped"

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("terminal_close")
        await self.stop(application_shutdown=True)

    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1
        self.events.append("terminal_wait_closed")
        return None

    async def force_exit(self) -> None:
        if self.hooks is not None:
            hooks, self.hooks = self.hooks, None
            await hooks.cleanup()
        self.lifecycle_epoch += 1
        self.state = "unavailable"

    def _endpoint(self, base_url: str) -> AudioCppReadyEndpoint:
        return AudioCppReadyEndpoint(
            base_url=base_url,
            process_generation=self.process_generation,
            observation_version=self.observation_version,
        )


class _LifecycleReader:
    def __init__(self) -> None:
        self._items: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    async def read(self, _size: int = -1) -> bytes:
        item = await self._items.get()
        if isinstance(item, BaseException):
            raise item
        return b"" if item is None else item

    def finish(self) -> None:
        self._items.put_nowait(None)

    def fail(self, error: BaseException) -> None:
        self._items.put_nowait(error)


class _LifecycleProcess:
    def __init__(self, *, exit_on_terminate: bool = True) -> None:
        self.returncode: int | None = None
        self.stdout = _LifecycleReader()
        self.stderr = _LifecycleReader()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self._exit_on_terminate = exit_on_terminate
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        self.wait_calls += 1
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._exit_on_terminate:
            self.exit(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self.exit(-9)

    def exit(self, returncode: int = 0) -> None:
        if self.returncode is not None:
            return
        self.returncode = returncode
        self.stdout.finish()
        self.stderr.finish()
        self._exited.set()

    def close_parent_pipes(self) -> None:
        self.stdout.finish()
        self.stderr.finish()


class _LifecycleLauncher:
    def __init__(
        self,
        processes: list[_LifecycleProcess],
        *,
        block_call: int | None = None,
    ) -> None:
        self._processes = processes
        self._block_call = block_call
        self.release_blocked_call = asyncio.Event()
        self.blocked_call_started = asyncio.Event()
        self.calls = 0
        self.launches: list[Any] = []

    async def __call__(
        self,
        _launch: Any,
        _environment: dict[str, str],
    ) -> _OwnedAudioCppProcess:
        self.calls += 1
        self.launches.append(_launch)
        if self.calls == self._block_call:
            self.blocked_call_started.set()
            await self.release_blocked_call.wait()
        process = self._processes[self.calls - 1]
        return _OwnedAudioCppProcess(
            process=process,
            close_parent_pipes=process.close_parent_pipes,
        )


async def _available_port(_port: int, _timeout: float) -> str:
    return "available"


def _external_config() -> dict[str, Any]:
    return AudioCppConfig().to_mapping()


def _managed_config(tmp_path: Path, label: str, port: int) -> dict[str, Any]:
    directory = tmp_path / label
    directory.mkdir()
    binary = directory / "audiocpp_server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    server_json = directory / "server.json"
    server_json.write_text(
        json.dumps({"host": "127.0.0.1", "port": port}),
        encoding="utf-8",
    )
    return AudioCppConfig(
        mode="managed",
        managed_binary_path=str(binary),
        managed_server_json_path=str(server_json),
    ).to_mapping()


def _guided_settings(
    tmp_path: Path,
    *,
    filename: str = "supertonic-3-orig.gguf",
    package_variant: str = "supertonic_3_orig",
    public_model_id: str = "model",
) -> AudioCppSettingsConfig:
    accepted = _accepted_guided_package(
        tmp_path / "guided-model",
        filename=filename,
        package_variant=package_variant,
        public_model_id=public_model_id,
    )
    binary = tmp_path / "audiocpp_server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    return AudioCppSettingsConfig.from_mapping(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_path": str(binary),
            "guided_packages": [accepted.model_dump(mode="json")],
            "guided_default_model_id": public_model_id,
        }
    )


def _accepted_guided_package(
    root: Path,
    *,
    filename: str,
    package_variant: str,
    public_model_id: str,
) -> Any:
    root.mkdir(parents=True)
    (root / filename).write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
    scan = scan_audio_cpp_package_root(root)
    candidates = tuple(
        candidate
        for discovery in scan.discoveries
        for candidate in discovery.match.candidates
        if candidate.recipe.package_variant == package_variant
    )
    assert len(candidates) == 1
    return candidates[0].accept(public_model_id=public_model_id)


def _guided_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return _response(b'{"status":"ok","backend":"cpu","models":1}')
    if request.url.path == "/v1/models":
        return _response(
            b'{"object":"list","data":[{"id":"model","object":"model",'
            b'"owned_by":"engine","family":"supertonic","task":"tts",'
            b'"mode":"offline"}]}'
        )
    raise AssertionError(f"unexpected path: {request.url.path}")


def _guided_models_handler(
    models: list[dict[str, str]],
) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _response(
                json.dumps(
                    {"status": "ok", "backend": "cpu", "models": len(models)}
                ).encode()
            )
        if request.url.path == "/v1/models":
            return _response(json.dumps({"object": "list", "data": models}).encode())
        raise AssertionError(f"unexpected path: {request.url.path}")

    return respond


def _upstream_model(**updates: str) -> dict[str, str]:
    model = {
        "id": "model",
        "object": "model",
        "owned_by": "engine",
        "family": "supertonic",
        "task": "tts",
        "mode": "offline",
    }
    model.update(updates)
    return model


def _preferences() -> TTSPreferencesSnapshot:
    return TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )


def _request() -> TTSRequest:
    return TTSRequest(
        provider_id="audio_cpp",
        model_id="model",
        text="hello",
        voice=None,
        response_format="wav",
    )


def _service(
    initial_config: Mapping[str, Any],
    supervisor: Any,
    *,
    shutdown_timeout_seconds: float = 10.0,
    transport_handler: Callable[[httpx.Request], httpx.Response] | None = _handler,
) -> tuple[TTSService, list[dict[str, Any]]]:
    factory_configs: list[dict[str, Any]] = []

    def factory(config: Mapping[str, Any]) -> AudioCppAdapter:
        factory_configs.append(dict(config))
        settings = AudioCppSettingsConfig.from_mapping(config)
        return AudioCppAdapter(
            AudioCppConfig.from_mapping(config),
            guided_settings=(
                settings
                if settings.mode == "managed"
                and settings.managed_setup_source.value == "guided"
                else None
            ),
            transport=(
                httpx.MockTransport(transport_handler)
                if transport_handler is not None
                else None
            ),
            supervisor=supervisor,  # type: ignore[arg-type]
        )

    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config=initial_config,
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )
    return (
        TTSService(
            registry,
            preferences_snapshot=_preferences(),
            audio_cpp_supervisor=supervisor,  # type: ignore[arg-type]
        ),
        factory_configs,
    )


async def _stage(
    service: TTSService,
    config: Mapping[str, Any],
    *,
    generation: int = 1,
) -> None:
    await service.registry.stage_provider_configuration(
        "audio_cpp",
        config,
        generation=generation,
    )
    service._settings_persisted_provider_generations["audio_cpp"] = generation


@pytest.mark.asyncio
async def test_runtime_observation_is_passive_and_keeps_paths_out_of_repr(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "passive-observation", 19_121)
    service, factory_configs = _service(managed, supervisor)

    try:
        observation = await service.audio_cpp_runtime_observation()

        assert isinstance(observation, AudioCppRuntimeObservation)
        assert observation.saved_mode == "managed"
        assert observation.applied_mode == "managed"
        assert observation.saved_configuration_generation == 0
        assert observation.applied_configuration_generation == 0
        assert observation.pending_configuration is False
        assert observation.process.state == "stopped"
        assert observation.catalog_revision is None
        assert observation.catalog_fresh is False
        assert observation.service_closed is False
        assert observation.saved_managed_binary_path == managed["managed_binary_path"]
        assert (
            observation.applied_managed_server_json_path
            == managed["managed_server_json_path"]
        )
        rendered = repr(observation)
        assert str(managed["managed_binary_path"]) not in rendered
        assert str(managed["managed_server_json_path"]) not in rendered
        assert factory_configs == []
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_guided_runtime_observation_exposes_only_safe_sample_facts(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    settings = _guided_settings(tmp_path, public_model_id="safe-model")
    service, factory_configs = _service(settings.to_mapping(), supervisor)

    try:
        observation = await service.audio_cpp_runtime_observation()

        assert observation.saved_managed_setup_source == "guided"
        assert observation.applied_managed_setup_source == "guided"
        assert observation.saved_guided_model_ids == ("safe-model",)
        assert observation.applied_guided_model_ids == ("safe-model",)
        assert observation.saved_guided_default_model_id == "safe-model"
        assert observation.applied_guided_default_model_id == "safe-model"
        assert observation.saved_guided_text_ready is True
        assert observation.applied_guided_text_ready is True
        assert observation.saved_managed_binary_path is None
        assert observation.saved_managed_server_json_path is None
        assert observation.applied_managed_binary_path is None
        assert observation.applied_managed_server_json_path is None
        assert str(tmp_path) not in repr(observation)
        assert factory_configs == []
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_guided_runtime_sample_readiness_belongs_to_exact_default(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    supertonic = _accepted_guided_package(
        tmp_path / "models" / "supertonic",
        filename="supertonic-3-orig.gguf",
        package_variant="supertonic_3_orig",
        public_model_id="narrator",
    )
    pocket = _accepted_guided_package(
        tmp_path / "models" / "pocket",
        filename="pocket-tts-english-bf16.gguf",
        package_variant="pocket_tts_english_bf16",
        public_model_id="clone-voice",
    )
    binary = tmp_path / "audiocpp_server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    settings = AudioCppSettingsConfig.from_mapping(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_path": str(binary),
            "guided_packages": [
                supertonic.model_dump(mode="json"),
                pocket.model_dump(mode="json"),
            ],
            "guided_default_model_id": "clone-voice",
        }
    )
    service, _ = _service(settings.to_mapping(), supervisor)

    try:
        observation = await service.audio_cpp_runtime_observation(
            selected_model_id="clone-voice"
        )

        assert observation.saved_guided_model_ids == ("narrator", "clone-voice")
        assert observation.saved_guided_default_model_id == "clone-voice"
        assert observation.saved_guided_text_ready is False
        assert observation.applied_guided_text_ready is False
        assert observation.clone_setup is not None
        assert observation.clone_setup.model_id == "clone-voice"
        assert observation.clone_setup.family_label == "Pocket TTS"
        assert observation.clone_setup.recipe_revision == 2
        assert observation.clone_setup.reference_requirement == "required"
        assert observation.clone_setup.voice_reference_policy == "reference_only"
        assert str(tmp_path) not in repr(observation.clone_setup)
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_optional_reference_recipe_keeps_normal_tts_action(
    tmp_path: Path,
) -> None:
    """Task 13205 projects setup only for recipes that require a reference."""

    supervisor = _PreparationSupervisor()
    # TASK-18609: the reviewed catalog classified
    # pocket_tts_english_safetensors as voice_or_reference_required
    # (e206d0882), so it is NOT text-ready and cannot carry this test's
    # "an optional-reference recipe still permits normal TTS" intent.
    # dramabox_q8_0 is the catalog's optional_reference_only TTS recipe.
    recipe = next(
        candidate
        for candidate in AUDIO_CPP_RECIPE_REGISTRY.recipes
        if candidate.package_variant == "dramabox_q8_0"
    )
    accepted = AudioCppAcceptedPackage(
        package_uuid="d3f6d610-6fd9-4cde-9ea7-cc5175ca445b",
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.recipe_revision,
        package_variant=recipe.package_variant,
        public_model_id="optional-clone",
        canonical_root=str(tmp_path / "pocket-safetensors"),
        canonical_root_identity="1" * 64,
        configuration_identity="2" * 64,
        weight_identity="3" * 64,
        projection=recipe.projection,
    )
    binary = tmp_path / "audiocpp_server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    settings = AudioCppSettingsConfig.from_mapping(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_path": str(binary),
            "guided_packages": [accepted.model_dump(mode="json")],
            "guided_default_model_id": "optional-clone",
        }
    )
    service, _ = _service(settings.to_mapping(), supervisor)

    try:
        observation = await service.audio_cpp_runtime_observation(
            selected_model_id="optional-clone"
        )

        assert observation.clone_setup is None
        assert observation.applied_guided_text_ready is True
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_runtime_observation_reports_staged_managed_over_applied_external(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "staged-managed-observation", 19_122)
    service, factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed, generation=7)

    try:
        observation = await service.audio_cpp_runtime_observation()

        assert observation.saved_mode == "managed"
        assert observation.applied_mode == "external"
        assert observation.saved_configuration_generation == 7
        assert observation.applied_configuration_generation == 0
        assert observation.pending_configuration is True
        assert observation.process.state == "stopped"
        assert observation.saved_managed_binary_path == managed["managed_binary_path"]
        assert observation.applied_managed_binary_path is None
        assert factory_configs == []
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_first_deliberate_start_projects_staged_guided_clone_setup(
    tmp_path: Path,
) -> None:
    """A first-time staged clone model explains the setup action before launch."""

    supervisor = _PreparationSupervisor()
    package = _accepted_guided_package(
        tmp_path / "models" / "pocket",
        filename="pocket-tts-english-bf16.gguf",
        package_variant="pocket_tts_english_bf16",
        public_model_id="clone-voice",
    )
    binary = tmp_path / "audiocpp_server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    managed = AudioCppSettingsConfig.from_mapping(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_path": str(binary),
            "guided_packages": [package.model_dump(mode="json")],
            "guided_default_model_id": "clone-voice",
        }
    )
    service, factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed.to_mapping(), generation=7)

    try:
        observation = await service.audio_cpp_runtime_observation(
            selected_model_id="clone-voice"
        )

        assert observation.saved_mode == "managed"
        assert observation.applied_mode == "external"
        assert observation.pending_configuration is True
        assert observation.process.state == "stopped"
        assert observation.clone_setup is not None
        assert observation.clone_setup.model_id == "clone-voice"
        assert observation.clone_setup.reference_requirement == "required"
        assert factory_configs == []
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_external_runtime_capability_reflects_tested_catalog() -> None:
    supervisor = _PreparationSupervisor()
    service, factory_configs = _service(_external_config(), supervisor)

    try:
        before = await service.audio_cpp_runtime_observation()

        assert before.tts_capability == "unknown"
        assert before.catalog_revision is None

        await service.start_and_test_audio_cpp()
        after = await service.audio_cpp_runtime_observation()

        assert after.applied_mode == "external"
        assert after.tts_capability == "available"
        assert after.catalog_revision is not None
        assert after.catalog_fresh is True
        assert after.active_endpoint == _external_config()["base_url"]
        assert factory_configs == [_external_config()]
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_runtime_observation_keeps_live_managed_visible_when_external_staged(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "active-managed-observation", 19_123)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    await _stage(service, _external_config(), generation=9)

    try:
        observation = await service.audio_cpp_runtime_observation()

        assert observation.saved_mode == "external"
        assert observation.applied_mode == "managed"
        assert observation.saved_configuration_generation == 9
        assert observation.applied_configuration_generation == 0
        assert observation.pending_configuration is True
        assert observation.process.state == "running"
        assert observation.process.process_generation == 1
        assert observation.process.tts_capability == "available"
        assert observation.catalog_revision is not None
        assert observation.catalog_fresh is True
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_runtime_observation_marks_catalog_stale_after_managed_exit(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "crashed-observation", 19_124)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    before_exit = await service.audio_cpp_runtime_observation()
    await supervisor.force_exit()

    try:
        after_exit = await service.audio_cpp_runtime_observation()

        assert before_exit.catalog_fresh is True
        assert after_exit.process.state == "unavailable"
        assert after_exit.catalog_revision == before_exit.catalog_revision
        assert after_exit.catalog_fresh is False
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_runtime_observation_keeps_catalog_stale_after_health_recovers(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "recovered-observation", 19_126)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()

    try:
        assert (await service.audio_cpp_runtime_observation()).catalog_fresh is True

        # The first failed probe leaves the process Running, but invalidates
        # generation-bound catalog evidence immediately.
        supervisor.observation_version += 1
        assert (await service.audio_cpp_runtime_observation()).catalog_fresh is False

        # A later successful probe must not resurrect the old catalog.
        supervisor.observation_version += 1
        recovered = await service.audio_cpp_runtime_observation()

        assert recovered.catalog_fresh is False

        await service.get_catalog("audio_cpp", refresh=True)
        refreshed = await service.audio_cpp_runtime_observation()
        assert refreshed.catalog_fresh is True
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_health_failure_between_catalog_read_and_publication_stays_stale(
    tmp_path: Path,
) -> None:
    fail_health = False

    def mutable_handler(request: httpx.Request) -> httpx.Response:
        if fail_health and request.url.path == "/health":
            return httpx.Response(503, request=request)
        return _handler(request)

    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "publication-health-race", 19_129)
    service, _factory_configs = _service(
        managed,
        supervisor,
        transport_handler=mutable_handler,
    )
    await service.start_and_test_audio_cpp()
    record = service.registry._slots["audio_cpp"].active
    assert record is not None
    adapter = record.adapter
    assert isinstance(adapter, AudioCppAdapter)
    original_get_catalog = adapter.get_catalog

    async def get_catalog_then_fail_health(
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        nonlocal fail_health
        catalog = await original_get_catalog(refresh=refresh)
        fail_health = True
        assert supervisor.hooks is not None
        assert await supervisor.hooks.health_probe() is False
        supervisor.observation_version += 1
        return catalog

    adapter.get_catalog = get_catalog_then_fail_health  # type: ignore[method-assign]

    try:
        returned = await service.get_catalog("audio_cpp", refresh=False)
        observation = await service.audio_cpp_runtime_observation()

        assert returned.health.fresh is True
        assert observation.catalog_fresh is False
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_synthesis_revalidates_after_managed_health_observation_changes(
    tmp_path: Path,
) -> None:
    request_paths: list[str] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        request_paths.append(request.url.path)
        return _handler(request)

    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "synthesis-revalidation", 19_128)
    service, _factory_configs = _service(
        managed,
        supervisor,
        transport_handler=recording_handler,
    )
    await service.start_and_test_audio_cpp()
    request_paths.clear()

    try:
        supervisor.observation_version += 1
        assert (await service.audio_cpp_runtime_observation()).catalog_fresh is False

        response = await service.synthesize(_request())
        assert [chunk async for chunk in response.byte_stream] == [_wav()]
        await response.aclose()

        assert request_paths == ["/health", "/v1/models", "/v1/audio/speech"]
        assert (await service.audio_cpp_runtime_observation()).catalog_fresh is True
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_runtime_observation_reports_fresh_zero_model_managed_catalog(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "zero-model-observation", 19_127)
    service, _factory_configs = _service(
        managed,
        supervisor,
        transport_handler=_no_model_handler,
    )

    try:
        catalog = await service.start_and_test_audio_cpp()
        observation = await service.audio_cpp_runtime_observation()

        assert catalog.health.state == "not_configured"
        assert catalog.health.fresh is True
        assert observation.process.tts_capability == "not_configured"
        assert observation.tts_capability == "not_configured"
        assert observation.catalog_fresh is True
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_runtime_observation_reports_closed_service_without_provider_work(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "closed-observation", 19_125)
    service, factory_configs = _service(managed, supervisor)
    await service.close()
    await service.wait_closed()

    observation = await service.audio_cpp_runtime_observation()

    assert observation.service_closed is True
    assert observation.process.state == "stopped"
    assert observation.catalog_fresh is False
    assert factory_configs == []
    assert supervisor.launches == 0


@pytest.mark.asyncio
async def test_catalog_refresh_applies_latest_stage_before_adapter_lease(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "managed", 19_101)
    service, factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)

    try:
        catalog = await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog.models[0].model_id == "model"
        assert dict(snapshot.applied_config) == managed
        assert snapshot.staged_config is None
        assert factory_configs == [managed]
        assert supervisor.launches == 1
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_direct_admit_prepares_stage_before_returning_operation(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "admit", 19_102)
    service, _factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)

    try:
        operation = await service.admit(_request())
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert snapshot.staged_config is None
        assert supervisor.launches == 1
        await operation.close()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_console_and_roleplay_admission_apply_stage_before_read_gate(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "roleplay", 19_103)
    service, _factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)
    character = TTSCharacterProfileSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        repository_generation=1,
        profile_revision=1,
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
    )

    try:
        response, selection = await service.synthesize_effective(
            text="roleplay line",
            character_profile=character,
        )
        await response.aclose()

        assert selection.provider_id == "audio_cpp"
        assert supervisor.launches == 1
        assert (
            await service.registry.provider_configuration_snapshot("audio_cpp")
        ).staged_config is None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_default_first_available_launches_applied_managed_first_use(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "default-first-use", 19_120)
    service, _factory_configs = _service(managed, supervisor)

    try:
        response = await service.synthesize_default(text="first managed line")
        audio = [chunk async for chunk in response.byte_stream]
        await response.aclose()

        assert response.model_id == "model"
        assert audio == [_wav()]
        assert response.metadata["process_generation"] == 1
        assert supervisor.launches == 1
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_managed_authorization_uses_exact_launched_outbound_origin(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    port = 19_152
    managed = _managed_config(tmp_path, "authorized-origin", port)
    requested_urls: list[str] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return _handler(request)

    service, _factory_configs = _service(
        managed,
        supervisor,
        transport_handler=capture,
    )
    authorized: list[tuple[str, str]] = []

    def authorize(provider_id: str, endpoint: str) -> bool:
        authorized.append((provider_id, endpoint))
        return endpoint == f"http://127.0.0.1:{port}"

    try:
        resolved_endpoint = await service.resolve_provider_outbound_endpoint(
            "audio_cpp"
        )
        response = await service.synthesize_default(
            text="Authorized managed reply.",
            admission_authorizer=authorize,
        )
        await response.byte_stream.__anext__()
        await response.aclose()

        assert resolved_endpoint == f"http://127.0.0.1:{port}"
        assert authorized == [("audio_cpp", f"http://127.0.0.1:{port}")]
        speech_url = next(url for url in requested_urls if "/v1/audio/speech" in url)
        assert speech_url == f"http://127.0.0.1:{port}/v1/audio/speech"
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_complete_wav_keeps_the_admitted_process_generation_if_exit_follows(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "generation-provenance", 19_129)
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(managed),
        transport=httpx.MockTransport(_handler),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    original_post = adapter._post_speech

    async def post_then_invalidate(payload: dict[str, str]):
        outcome = await original_post(payload)
        adapter._managed_process_generation = None
        return outcome

    adapter._post_speech = post_then_invalidate  # type: ignore[method-assign]

    try:
        response = await adapter.synthesize(
            TTSRequest(
                provider_id="audio_cpp",
                model_id="model",
                text="retain exact generation provenance",
                voice=None,
                response_format="wav",
            )
        )
        await response.aclose()

        assert response.metadata["process_generation"] == 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_passive_service_capability_paths_neither_apply_nor_launch(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "passive", 19_104)
    service, _factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)

    try:
        process_snapshot = service.audio_cpp_process_snapshot()
        await service.get_catalog("audio_cpp", refresh=False)
        await service.get_voices("audio_cpp", "model", refresh=False)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert process_snapshot.state == "stopped"
        assert process_snapshot.process_generation == 0
        assert dict(snapshot.staged_config or {}) == managed
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_profile_capability_validation_stays_unverified_without_launch(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "profile-passive", 19_116)
    service, _factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)

    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "unverified"
        assert snapshot.catalog is None
        assert supervisor.ensure_calls == 0
        assert (
            await service.registry.provider_configuration_snapshot("audio_cpp")
        ).staged_config is not None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_applied_managed_capability_read_never_launches_stopped_child(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "applied-passive", 19_118)
    service, _factory_configs = _service(managed, supervisor)

    try:
        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "unverified"
        assert snapshot.catalog is None
        assert supervisor.launches == 0
        assert supervisor.state == "stopped"
        assert supervisor.ensure_requirements[-1] is not None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_passive_capability_read_never_restarts_changed_managed_generation(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "passive-generation-race", 19_119)
    service, _factory_configs = _service(managed, supervisor)

    try:
        await service.get_catalog("audio_cpp", refresh=True)
        assert supervisor.launches == 1
        supervisor.change_before_ensure = True

        snapshot = await service.get_native_capability_snapshot(
            "audio_cpp",
            ("model",),
        )

        assert snapshot.state == "unverified"
        assert supervisor.launches == 1
        assert supervisor.state == "unavailable"
        assert supervisor.ensure_requirements[-1] is not None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_settings_capability_observation_stays_passive_while_stopped(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "settings-passive", 19_117)
    service, _factory_configs = _service(managed, supervisor)

    try:
        catalog = await service.get_catalog("audio_cpp", refresh=False)
        observation = service.latest_native_capability_observation("audio_cpp")

        assert catalog.health.fresh is False
        assert observation is not None
        assert observation.snapshot.state == "unverified"
        assert supervisor.ensure_calls == 0
        assert supervisor.state == "stopped"
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_concurrent_preparation_applies_one_latest_generation(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "concurrent", 19_105)
    service, factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)

    try:
        catalogs = await asyncio.gather(
            service.get_catalog("audio_cpp", refresh=True),
            service.get_catalog("audio_cpp", refresh=True),
        )

        assert all(catalog.models[0].model_id == "model" for catalog in catalogs)
        assert factory_configs == [managed]
        assert supervisor.launches == 1
        assert supervisor.stop_calls == 1
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_concurrent_guided_first_use_materializes_and_launches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _guided_settings(tmp_path)
    runtime_root = tmp_path / "guided-runtime"
    materialize_calls = 0

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        nonlocal materialize_calls
        materialize_calls += 1
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
            port_selector=lambda: 54_330,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    process = _LifecycleProcess()
    launcher = _LifecycleLauncher([process])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(_guided_handler),
        supervisor=supervisor,
    )

    try:
        catalogs = await asyncio.gather(
            adapter.get_catalog(refresh=True),
            adapter.get_catalog(refresh=True),
        )

        assert materialize_calls == 1
        assert launcher.calls == 1
        assert all(catalog.models[0].model_id == "model" for catalog in catalogs)
        assert catalogs[0].models[0].speech_capabilities == ("tts",)
    finally:
        await adapter.close()
        await supervisor.close()
        await supervisor.wait_closed()

    assert process.wait_calls == 1
    assert runtime_root.exists()
    assert tuple(runtime_root.iterdir()) == ()


def _clone_reference() -> TTSCloneReference:
    wav_bytes = b"PRIVATE CLONE WAV"
    now = datetime(2026, 8, 10, tzinfo=UTC)
    return TTSCloneReference(
        summary=TTSCloneReferenceSummary(
            reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            byte_length=len(wav_bytes),
            duration_ms=250,
            sample_rate_hz=24_000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=now,
            updated_at=now,
        ),
        reference_text="PRIVATE REFERENCE TRANSCRIPT",
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )


@pytest.mark.asyncio
async def test_clone_source_preflight_rejects_external_before_http() -> None:
    requests: list[httpx.Request] = []

    def forbidden(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("clone preflight must not contact External audio.cpp")

    adapter = AudioCppAdapter(
        AudioCppConfig(),
        transport=httpx.MockTransport(forbidden),
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            adapter.preflight_clone_source()
        assert caught.value.code == "configuration_invalid"
        assert requests == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_clone_source_preflight_rejects_user_json_before_launch_or_http(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(
            _managed_config(tmp_path, "clone-user-json", 19_411)
        ),
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("clone preflight must not use HTTP")
            )
        ),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            adapter.preflight_clone_source()
        assert caught.value.code == "configuration_invalid"
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_clone_source_preflight_requires_the_application_process_owner(
    tmp_path: Path,
) -> None:
    settings = _guided_settings(
        tmp_path,
        filename="pocket-tts-english-q8_0.gguf",
        package_variant="pocket_tts_english_q8_0",
        public_model_id="clone-model",
    )
    supervisor = _PreparationSupervisor()
    del supervisor._application_owner_token
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            adapter.preflight_clone_source()
        assert caught.value.code == "configuration_invalid"
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_clone_dependency_preflight_rejects_config_drift_without_readiness(
    tmp_path: Path,
) -> None:
    settings = _guided_settings(
        tmp_path,
        filename="pocket-tts-english-q8_0.gguf",
        package_variant="pocket_tts_english_q8_0",
        public_model_id="clone-model",
    )
    accepted = settings.guided_packages[0]
    supervisor = _PreparationSupervisor()
    requests: list[httpx.Request] = []
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(
            lambda request: requests.append(request)  # type: ignore[arg-type,return-value]
        ),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    wrong = TTSCloneRecipeRequirement(
        recipe_id=accepted.recipe_id,
        recipe_revision=accepted.recipe_revision + 1,
        model_id="clone-model",
    )
    try:
        with pytest.raises(TTSOperationError) as caught:
            adapter.preflight_clone_dependency(wrong)

        assert caught.value.code == "dependency_changed"
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
        assert requests == []
        assert adapter._managed_bundle is None
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_clone_request_policy_preflight_uses_exact_resolved_voice_without_readiness(
    tmp_path: Path,
) -> None:
    settings = _guided_settings(
        tmp_path,
        filename="pocket-tts-english-q8_0.gguf",
        package_variant="pocket_tts_english_q8_0",
        public_model_id="clone-model",
    )
    accepted = settings.guided_packages[0]
    supervisor = _PreparationSupervisor()
    requests: list[httpx.Request] = []
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(
            lambda request: requests.append(request)  # type: ignore[arg-type,return-value]
        ),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    requirement = TTSCloneRecipeRequirement(
        recipe_id=accepted.recipe_id,
        recipe_revision=accepted.recipe_revision,
        model_id="clone-model",
    )
    request = TTSRequest(
        provider_id="audio_cpp",
        model_id="clone-model",
        text="Dependency preflight",
        voice="native-voice",
        response_format="wav",
    )
    try:
        adapter.preflight_clone_dependency(requirement)
        with pytest.raises(TTSOperationError) as caught:
            adapter.preflight_clone_request_dependency(request, requirement)

        assert caught.value.code == "dependency_changed"
        assert supervisor.ensure_calls == 0
        assert supervisor.launches == 0
        assert requests == []
        assert adapter._managed_bundle is None
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_guided_clone_admission_sends_only_typed_live_reference_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _guided_settings(
        tmp_path,
        filename="pocket-tts-english-bf16.gguf",
        package_variant="pocket_tts_english_bf16",
        public_model_id="clone-model",
    )
    supervisor = _PreparationSupervisor()
    payloads: list[dict[str, str]] = []
    caplog.set_level(logging.DEBUG, logger="httpcore.http11")

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=tmp_path / "guided-runtime",
            port_selector=lambda: 54_332,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _response(b'{"status":"ok","backend":"cpu","models":1}')
        if request.url.path == "/v1/models":
            return _response(
                b'{"object":"list","data":[{"id":"clone-model",'
                b'"object":"model","owned_by":"engine",'
                b'"family":"pocket_tts","task":"tts","mode":"offline"}]}'
            )
        if request.url.path == "/v1/audio/speech":
            logging.getLogger("httpcore.http11").debug(
                "PRIVATE CLONE BODY %s",
                request.content,
            )
            payloads.append(json.loads(request.content))
            return _response(_wav(), headers={"content-type": "audio/wav"})
        raise AssertionError(f"unexpected path: {request.url.path}")

    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(respond),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    materializer = TTSCloneReferenceMaterializer(tmp_path / "clone-runtime")
    request = TTSRequest(
        provider_id="audio_cpp",
        model_id="clone-model",
        text="speak as the character",
        voice=None,
        response_format="wav",
    )
    try:
        adapter.preflight_clone_source()
        assert supervisor.ensure_calls == 0
        await adapter.ensure_ready()
        with pytest.raises(TTSOperationError) as omitted:
            await adapter.synthesize(request)
        assert omitted.value.code == "request_invalid"
        assert payloads == []
        with pytest.raises(TTSOperationError) as conflicting:
            adapter.admit_clone_capability(
                TTSRequest(
                    provider_id="audio_cpp",
                    model_id="clone-model",
                    text="invalid combined request",
                    voice="native-voice",
                    response_format="wav",
                )
            )
        assert conflicting.value.code == "request_invalid"

        forged_internal = object.__new__(_AdmittedAudioCppCloneRequest)
        with pytest.raises(TTSOperationError) as forged_error:
            await adapter.synthesize_clone(forged_internal)
        assert forged_error.value.code == "request_invalid"
        assert payloads == []

        capability = adapter.admit_clone_capability(request)
        with pytest.raises(AttributeError):
            capability._model_id = "forged-model"
        owner = await materializer.materialize(_clone_reference())
        altered_request = TTSRequest(
            provider_id=request.provider_id,
            model_id=request.model_id,
            text="altered after capability admission",
            voice=request.voice,
            response_format=request.response_format,
        )
        with pytest.raises(ValueError):
            _new_admitted_audio_cpp_clone_request(
                request=altered_request,
                materialization=owner,
                capability=capability,
                provider_revision=0,
                applied_provider_generation=0,
            )
        copied_capability = copy.copy(capability)
        copied_admission = _new_admitted_audio_cpp_clone_request(
            request=request,
            materialization=owner,
            capability=copied_capability,
            provider_revision=0,
            applied_provider_generation=0,
        )
        with pytest.raises(TTSOperationError) as copied:
            await adapter.synthesize_clone(copied_admission)
        assert copied.value.code == "request_invalid"
        swapped_owner = await materializer.materialize(_clone_reference())
        with pytest.raises(ValueError):
            _new_admitted_audio_cpp_clone_request(
                request=request,
                materialization=swapped_owner,
                capability=copied_capability,
                provider_revision=999,
                applied_provider_generation=999,
            )
        await swapped_owner.aclose()

        with pytest.raises(TypeError):
            _AdmittedAudioCppCloneRequest(
                request=request,
                materialization=owner,
                capability=capability,
                provider_revision=0,
                applied_provider_generation=0,
                recipe_id="wrong-recipe",
                recipe_revision=capability.recipe_revision,
                process_generation=capability.process_generation,
            )
        with pytest.raises(TypeError):
            _AdmittedAudioCppCloneRequest(
                request=request,
                materialization=Path("/private/forged-reference.wav"),  # type: ignore[arg-type]
                capability=capability,
                provider_revision=0,
                applied_provider_generation=0,
                recipe_id=capability.recipe_id,
                recipe_revision=capability.recipe_revision,
                process_generation=capability.process_generation,
            )
        copied_owner = copy.copy(owner)
        copied_owner_admission = _new_admitted_audio_cpp_clone_request(
            request=request,
            materialization=copied_owner,
            capability=capability,
            provider_revision=0,
            applied_provider_generation=0,
        )
        with pytest.raises(TTSOperationError) as copied_owner_error:
            await adapter.synthesize_clone(copied_owner_admission)
        assert copied_owner_error.value.code == "request_invalid"
        capability = adapter.admit_clone_capability(request)
        admitted = _new_admitted_audio_cpp_clone_request(
            request=request,
            materialization=owner,
            capability=capability,
            provider_revision=0,
            applied_provider_generation=0,
        )
        with pytest.raises(TypeError):
            copy.copy(admitted)
        with pytest.raises(TTSOperationError) as public_forgery:
            await adapter.synthesize(admitted)  # type: ignore[arg-type]
        assert public_forgery.value.code == "request_invalid"
        assert payloads == []

        response = await adapter.synthesize_clone(admitted)
        await response.aclose()
        assert supervisor.suppressed_clone_generations == [admitted.process_generation]
        assert "voice_ref" not in response.metadata
        assert "reference_text" not in response.metadata

        assert payloads == [
            {
                "model": "clone-model",
                "input": "speak as the character",
                "response_format": "wav",
                "voice_ref": str(owner.voice_ref),
                "reference_text": "PRIVATE REFERENCE TRANSCRIPT",
            }
        ]
        assert "PRIVATE" not in repr(admitted)
        assert str(owner.voice_ref) not in repr(admitted)
        assert "PRIVATE REFERENCE TRANSCRIPT" not in caplog.text
        assert str(owner.voice_ref) not in caplog.text
        with pytest.raises(TTSOperationError) as consumed:
            await adapter.synthesize_clone(admitted)
        assert consumed.value.code == "request_invalid"
        assert len(payloads) == 1

        closed_owner = await materializer.materialize(_clone_reference())
        closed_capability = adapter.admit_clone_capability(request)
        closed_admission = _new_admitted_audio_cpp_clone_request(
            request=request,
            materialization=closed_owner,
            capability=closed_capability,
            provider_revision=0,
            applied_provider_generation=0,
        )
        await closed_owner.aclose()
        with pytest.raises(TTSOperationError) as stale_owner:
            await adapter.synthesize_clone(closed_admission)
        assert stale_owner.value.code == "request_invalid"
        assert len(payloads) == 1
    finally:
        await materializer.close()
        await adapter.close()


@pytest.mark.asyncio
async def test_guided_clone_capability_is_process_generation_fenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _guided_settings(
        tmp_path,
        filename="pocket-tts-english-q8_0.gguf",
        package_variant="pocket_tts_english_q8_0",
        public_model_id="clone-model",
    )

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=tmp_path / "guided-runtime",
            port_selector=lambda: 54_333,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    speech_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal speech_requests
        if request.url.path == "/health":
            return _response(b'{"status":"ok","backend":"cpu","models":1}')
        if request.url.path == "/v1/models":
            return _response(
                b'{"object":"list","data":[{"id":"clone-model",'
                b'"object":"model","owned_by":"engine",'
                b'"family":"pocket_tts","task":"tts","mode":"offline"}]}'
            )
        if request.url.path == "/v1/audio/speech":
            speech_requests += 1
            return _response(_wav(), headers={"content-type": "audio/wav"})
        raise AssertionError

    supervisor = _PreparationSupervisor()
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(respond),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    materializer = TTSCloneReferenceMaterializer(tmp_path / "clone-runtime")
    request = TTSRequest(
        provider_id="audio_cpp",
        model_id="clone-model",
        text="stale generation",
        voice=None,
        response_format="wav",
    )
    try:
        await adapter.ensure_ready()
        capability = adapter.admit_clone_capability(request)
        owner = await materializer.materialize(_clone_reference())
        admitted = _new_admitted_audio_cpp_clone_request(
            request=request,
            materialization=owner,
            capability=capability,
            provider_revision=0,
            applied_provider_generation=0,
        )

        async def replace_during_progress(_progress: object) -> None:
            await supervisor.force_exit()

        with pytest.raises(TTSOperationError) as stale:
            await adapter.synthesize_clone(admitted, replace_during_progress)

        assert stale.value.code == "connection_unavailable"
        assert speech_requests == 0
    finally:
        await materializer.close()
        await adapter.close()


@pytest.mark.asyncio
async def test_guided_catalog_preserves_recipe_declared_clone_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _guided_settings(
        tmp_path,
        filename="pocket-tts-english-q8_0.gguf",
        package_variant="pocket_tts_english_q8_0",
    )
    runtime_root = tmp_path / "guided-runtime"

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
            port_selector=lambda: 54_331,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    process = _LifecycleProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_LifecycleLauncher([process]),
        port_preflight=_available_port,
    )
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(
            _guided_models_handler([_upstream_model(family="pocket_tts")])
        ),
        supervisor=supervisor,
    )

    try:
        catalog = await adapter.get_catalog(refresh=True)

        assert catalog.models[0].speech_capabilities == ("tts", "clone")
    finally:
        await adapter.close()
        await supervisor.close()
        await supervisor.wait_closed()

    assert process.wait_calls == 1
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models",
    [
        [],
        [_upstream_model(), _upstream_model(id="unexpected")],
        [_upstream_model(family="pocket_tts")],
        [_upstream_model(task="clone")],
        [_upstream_model(mode="online")],
        [_upstream_model(), _upstream_model(id="asr", task="asr")],
    ],
    ids=(
        "missing",
        "extra-speech",
        "wrong-family",
        "wrong-task",
        "wrong-mode",
        "extra-non-speech",
    ),
)
async def test_guided_catalog_requires_exact_generated_model_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    models: list[dict[str, str]],
) -> None:
    settings = _guided_settings(tmp_path)
    runtime_root = tmp_path / "guided-runtime"

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
            port_selector=lambda: 54_332,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    process = _LifecycleProcess()
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=_LifecycleLauncher([process]),
        port_preflight=_available_port,
    )
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(_guided_models_handler(models)),
        supervisor=supervisor,
    )

    try:
        with pytest.raises(TTSOperationError) as caught:
            await adapter.get_catalog(refresh=True)

        assert caught.value.code == "contract_incompatible"
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    finally:
        await adapter.close()
        await supervisor.close()
        await supervisor.wait_closed()

    assert process.wait_calls == 1
    assert runtime_root.exists()
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_running_guided_generation_keeps_snapshot_and_next_start_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _guided_settings(tmp_path)
    runtime_root = tmp_path / "guided-runtime"
    materialize_calls = 0

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        nonlocal materialize_calls
        materialize_calls += 1
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
            port_selector=lambda: 54_333,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    process = _LifecycleProcess()
    launcher = _LifecycleLauncher([process])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        transport=httpx.MockTransport(_guided_handler),
        supervisor=supervisor,
    )

    try:
        await adapter.get_catalog(refresh=True)
        model_path = (
            Path(settings.guided_packages[0].canonical_root) / "supertonic-3-orig.gguf"
        )
        model_path.write_bytes(b"changed after launch")

        still_running = await adapter.get_catalog(refresh=True)
        assert still_running.models[0].model_id == "model"
        assert materialize_calls == 1
        assert launcher.calls == 1

        await supervisor.stop()
        with pytest.raises(TTSOperationError) as caught:
            await adapter.get_catalog(refresh=True)

        assert caught.value.code == "configuration_invalid"
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert materialize_calls == 2
        assert launcher.calls == 1
    finally:
        await adapter.close()
        await supervisor.close()
        await supervisor.wait_closed()

    assert process.wait_calls == 1
    assert runtime_root.exists()
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_saved_guided_settings_stage_until_explicit_restart_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_settings = _guided_settings(tmp_path / "first")
    second_settings = _guided_settings(tmp_path / "second")
    runtime_root = tmp_path / "guided-runtime"
    materialized_roots: list[str] = []

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        port = 54_340 + len(materialized_roots)
        materialized_roots.append(current.guided_packages[0].canonical_root)
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
            port_selector=lambda: port,
            system="darwin",
            architecture="arm64",
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    first_process = _LifecycleProcess()
    second_process = _LifecycleProcess()
    launcher = _LifecycleLauncher([first_process, second_process])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    service, factory_configs = _service(
        first_settings.to_mapping(),
        supervisor,
        transport_handler=_guided_handler,
    )

    try:
        await service.start_and_test_audio_cpp()
        assert len(materialized_roots) == 1
        assert len(tuple(runtime_root.iterdir())) == 1

        await _stage(service, second_settings.to_mapping(), generation=1)
        assert len(materialized_roots) == 1
        assert launcher.calls == 1
        assert len(tuple(runtime_root.iterdir())) == 1

        catalog = await service.restart_audio_cpp()

        assert catalog is not None
        assert catalog.models[0].model_id == "model"
        assert materialized_roots == [
            first_settings.guided_packages[0].canonical_root,
            second_settings.guided_packages[0].canonical_root,
        ]
        assert launcher.calls == 2
        assert first_process.wait_calls == 1
        assert len(tuple(runtime_root.iterdir())) == 1
        assert factory_configs == [
            first_settings.to_mapping(),
            second_settings.to_mapping(),
        ]
    finally:
        await service.close()
        await service.wait_closed()

    assert second_process.wait_calls == 1
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_direct_service_launch_paths_prepare_before_adapter_lease(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "direct-paths", 19_106)
    service, _factory_configs = _service(_external_config(), supervisor)
    await _stage(service, managed)

    try:
        voices = await service.get_voices(
            "audio_cpp",
            "model",
            refresh=True,
        )
        response = await service.synthesize(_request())
        audio = [chunk async for chunk in response.byte_stream]
        await response.aclose()

        assert voices == ("default",)
        assert audio == [_wav()]
        assert supervisor.launches == 1
        assert (
            await service.registry.provider_configuration_snapshot("audio_cpp")
        ).staged_config is None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_crash_with_external_staged_applies_external_without_child(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "managed-a", 19_107)
    external_b = AudioCppConfig(base_url="http://127.0.0.1:19108").to_mapping()
    service, factory_configs = _service(managed_a, supervisor)
    await service.get_catalog("audio_cpp", refresh=True)
    await _stage(service, external_b)
    await supervisor.force_exit()

    try:
        catalog = await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog.models[0].model_id == "model"
        assert dict(snapshot.applied_config) == external_b
        assert snapshot.staged_config is None
        assert factory_configs == [managed_a, external_b]
        assert supervisor.launches == 1
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_crash_with_managed_staged_starts_one_latest_replacement(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "managed-a", 19_109)
    managed_b = _managed_config(tmp_path, "managed-b", 19_110)
    service, factory_configs = _service(managed_a, supervisor)
    await service.get_catalog("audio_cpp", refresh=True)
    await _stage(service, managed_b)
    await supervisor.force_exit()

    try:
        catalog = await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog.models[0].model_id == "model"
        assert dict(snapshot.applied_config) == managed_b
        assert snapshot.staged_config is None
        assert factory_configs == [managed_a, managed_b]
        assert supervisor.launches == 2
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_exit_between_live_check_and_ensure_retries_before_old_launch(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "managed-a", 19_111)
    managed_b = _managed_config(tmp_path, "managed-b", 19_112)
    service, factory_configs = _service(managed_a, supervisor)
    await service.get_catalog("audio_cpp", refresh=True)
    await _stage(service, managed_b)
    supervisor.change_before_ensure = True

    try:
        catalog = await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog.models[0].model_id == "model"
        assert dict(snapshot.applied_config) == managed_b
        assert snapshot.staged_config is None
        assert factory_configs == [managed_a, managed_b]
        assert supervisor.launches == 2
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["starting", "stopping"])
async def test_pre_spawn_starting_and_post_exit_stopping_do_not_apply_stage(
    tmp_path: Path,
    state: str,
) -> None:
    supervisor = _PreparationSupervisor()
    supervisor.state = state
    managed_a = _managed_config(tmp_path, "managed-a", 19_113)
    managed_b = _managed_config(tmp_path, "managed-b", 19_114)
    service, factory_configs = _service(managed_a, supervisor)
    await _stage(service, managed_b)

    try:
        if state == "stopping":
            with pytest.raises(TTSProviderReconfiguringError):
                await service.get_catalog("audio_cpp", refresh=True)
        else:
            await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert dict(snapshot.applied_config) == managed_a
        assert dict(snapshot.staged_config or {}) == managed_b
        assert factory_configs == [managed_a]
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_stage_eligibility_is_rechecked_if_starting_appears_before_writer(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()

    def move_to_starting(admission_call: int) -> None:
        if admission_call == 2:
            supervisor.state = "starting"

    supervisor.admission_observer = move_to_starting
    managed = _managed_config(tmp_path, "managed", 19_115)
    external = _external_config()
    service, factory_configs = _service(external, supervisor)
    await _stage(service, managed)

    try:
        await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert dict(snapshot.applied_config) == external
        assert dict(snapshot.staged_config or {}) == managed
        assert factory_configs == [external]
        assert supervisor.stop_calls == 0
        assert supervisor.launches == 0
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_start_and_test_managed_starts_and_refreshes_catalog(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "start-test", 19_118)
    service, _factory_configs = _service(managed, supervisor)

    try:
        catalog = await service.start_and_test_audio_cpp()

        assert catalog.models[0].model_id == "model"
        assert supervisor.launches == 1
        assert supervisor.state == "running"
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_restart_drains_work_stops_old_generation_and_starts_one_new_generation(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "restart", 19_119)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    operation = await service.admit(_request())

    restart = asyncio.create_task(service.restart_audio_cpp())
    try:
        await asyncio.wait_for(supervisor.draining_started.wait(), timeout=1)
        assert restart.done() is False
        assert supervisor.launches == 1

        await operation.close()
        catalog = await asyncio.wait_for(restart, timeout=1)

        assert catalog is not None
        assert catalog.models[0].model_id == "model"
        assert supervisor.launches == 2
        assert supervisor.state == "running"
    finally:
        if not restart.done():
            await operation.close()
            await restart
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_restart_applies_latest_stage_not_earlier_stage(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "managed-a", 19_120)
    managed_b = _managed_config(tmp_path, "managed-b", 19_121)
    managed_c = _managed_config(tmp_path, "managed-c", 19_122)
    service, factory_configs = _service(managed_a, supervisor)
    await service.start_and_test_audio_cpp()
    await _stage(service, managed_b, generation=1)
    await _stage(service, managed_c, generation=2)

    try:
        catalog = await service.restart_audio_cpp()
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog is not None
        assert dict(snapshot.applied_config) == managed_c
        assert snapshot.staged_config is None
        assert factory_configs == [managed_a, managed_c]
        assert supervisor.launches == 2
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_restart_launches_chosen_generation_and_leaves_later_save_staged(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "restart-a", 19_133)
    managed_b = _managed_config(tmp_path, "restart-b", 19_134)
    managed_c = _managed_config(tmp_path, "restart-c", 19_135)
    service, factory_configs = _service(managed_a, supervisor)
    await service.start_and_test_audio_cpp()
    await _stage(service, managed_b, generation=1)
    supervisor.inflight_probe_gate = asyncio.Event()

    restart = asyncio.create_task(service.restart_audio_cpp())
    publication = None
    try:
        await asyncio.wait_for(supervisor.draining_started.wait(), timeout=1)
        while "generation_stop" not in supervisor.events:
            await asyncio.sleep(0)

        publication = service.begin_preferences_publication(
            _preferences(),
            {"audio_cpp": managed_c},
            lambda: TTSSettingsPersistenceOutcome(
                file_replaced=True,
                caches_reloaded=True,
                failure_phase=None,
            ),
            foreground_timeout_seconds=0,
        )
        await asyncio.sleep(0)
        assert publication.completion.done() is False

        supervisor.inflight_probe_gate.set()
        catalog = await asyncio.wait_for(restart, timeout=1)
        await asyncio.wait_for(publication.completion, timeout=1)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog is not None
        assert catalog.models[0].model_id == "model"
        assert dict(snapshot.applied_config) == managed_b
        assert dict(snapshot.staged_config or {}) == managed_c
        assert factory_configs == [managed_a, managed_b]
        assert supervisor.launches == 2
    finally:
        if supervisor.inflight_probe_gate is not None:
            supervisor.inflight_probe_gate.set()
        await asyncio.gather(restart, return_exceptions=True)
        if publication is not None:
            await asyncio.gather(publication.completion, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["restart", "shutdown"])
async def test_admitted_synthesis_finishes_after_draining_begins(
    tmp_path: Path,
    lifecycle: str,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, f"admitted-{lifecycle}", 19_136)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    operation = await service.admit(_request())
    transition = asyncio.create_task(
        service.restart_audio_cpp()
        if lifecycle == "restart"
        else service.shutdown_audio_cpp()
    )
    response = None

    try:
        await asyncio.wait_for(supervisor.draining_started.wait(), timeout=1)
        response = await operation.synthesize()
        assert [chunk async for chunk in response.byte_stream] == [_wav()]
        assert transition.done() is False

        await response.aclose()
        response = None
        await asyncio.wait_for(transition, timeout=1)
    finally:
        if response is not None:
            await response.aclose()
        await asyncio.gather(transition, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_admitted_synthesis_reports_safe_error_after_generation_exit(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "admitted-exit", 19_139)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    operation = await service.admit(_request())
    await supervisor.force_exit()

    try:
        with pytest.raises(TTSOperationError) as raised:
            await operation.synthesize()

        assert raised.value.code == "connection_unavailable"
        assert str(raised.value) == "The audio.cpp server is unavailable"
        assert raised.value.retryable is True
        assert raised.value.recovery_action == "retry"
        assert raised.value.__context__ is None
        assert raised.value.__cause__ is None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_passive_capability_save_race_stays_unverified(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "capability-a", 19_137)
    managed_b = _managed_config(tmp_path, "capability-b", 19_138)
    service, _factory_configs = _service(managed_a, supervisor)
    await service.start_and_test_audio_cpp()
    lease = await service.registry.acquire("audio_cpp")
    adapter = lease.adapter
    await lease.release()
    original_get_catalog = adapter.get_catalog
    observation_started = asyncio.Event()
    release_observation = asyncio.Event()
    catalog_calls = 0

    async def controlled_get_catalog(refresh: bool = False) -> TTSProviderCatalog:
        nonlocal catalog_calls
        catalog_calls += 1
        if catalog_calls == 1:
            observation_started.set()
            await release_observation.wait()
        return await original_get_catalog(refresh=refresh)

    adapter.get_catalog = controlled_get_catalog  # type: ignore[method-assign]
    observation = asyncio.create_task(
        service.get_native_capability_snapshot("audio_cpp", ("model",))
    )
    publication = None
    try:
        await asyncio.wait_for(observation_started.wait(), timeout=1)
        publication = service.begin_preferences_publication(
            _preferences(),
            {"audio_cpp": managed_b},
            lambda: TTSSettingsPersistenceOutcome(
                file_replaced=True,
                caches_reloaded=True,
                failure_phase=None,
            ),
            foreground_timeout_seconds=0,
        )
        await asyncio.wait_for(publication.completion, timeout=1)
        release_observation.set()
        result = await asyncio.wait_for(observation, timeout=1)

        assert service.saved_configuration_revision("audio_cpp") == 1
        assert service.applied_configuration_revision("audio_cpp") == 0
        assert result.state == "unverified"
        assert result.catalog is None
        assert result.voice_results == {}
    finally:
        release_observation.set()
        await asyncio.gather(observation, return_exceptions=True)
        if publication is not None:
            await asyncio.gather(publication.completion, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_passive_capability_apply_race_stays_unverified(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "capability-apply-a", 19_142)
    managed_b = _managed_config(tmp_path, "capability-apply-b", 19_143)
    service, _factory_configs = _service(managed_a, supervisor)
    await service.start_and_test_audio_cpp()
    lease = await service.registry.acquire("audio_cpp")
    adapter = lease.adapter
    await lease.release()
    original_get_catalog = adapter.get_catalog
    observation_started = asyncio.Event()
    release_observation = asyncio.Event()
    catalog_calls = 0

    async def controlled_get_catalog(refresh: bool = False) -> TTSProviderCatalog:
        nonlocal catalog_calls
        catalog_calls += 1
        if catalog_calls == 1:
            observation_started.set()
            await release_observation.wait()
        return await original_get_catalog(refresh=refresh)

    adapter.get_catalog = controlled_get_catalog  # type: ignore[method-assign]
    observation = asyncio.create_task(
        service.get_native_capability_snapshot("audio_cpp", ("model",))
    )
    restart = None
    try:
        await asyncio.wait_for(observation_started.wait(), timeout=1)
        await _stage(service, managed_b, generation=1)
        restart = asyncio.create_task(service.restart_audio_cpp())
        await asyncio.wait_for(supervisor.draining_started.wait(), timeout=1)

        release_observation.set()
        await asyncio.wait_for(restart, timeout=1)
        result = await asyncio.wait_for(observation, timeout=1)

        assert service.saved_configuration_revision("audio_cpp") == 1
        assert service.applied_configuration_revision("audio_cpp") == 1
        assert result.state == "unverified"
        assert result.catalog is None
        assert result.voice_results == {}
    finally:
        release_observation.set()
        await asyncio.gather(observation, return_exceptions=True)
        if restart is not None:
            await asyncio.gather(restart, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_passive_capability_exit_before_publication_stays_unverified(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "capability-exit", 19_144)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    original_acquire = service.registry.acquire
    release_started = asyncio.Event()
    release_allowed = asyncio.Event()

    async def controlled_acquire(*args: Any, **kwargs: Any) -> Any:
        lease = await original_acquire(*args, **kwargs)
        original_release = lease.release

        async def controlled_release() -> None:
            release_started.set()
            await release_allowed.wait()
            await original_release()

        lease.release = controlled_release
        return lease

    service.registry.acquire = controlled_acquire  # type: ignore[method-assign]
    observation = asyncio.create_task(
        service.get_native_capability_snapshot("audio_cpp", ("model",))
    )
    try:
        await asyncio.wait_for(release_started.wait(), timeout=1)
        await supervisor.force_exit()
        release_allowed.set()
        result = await asyncio.wait_for(observation, timeout=1)

        assert result.state == "unverified"
        assert result.catalog is None
        assert result.voice_results == {}
    finally:
        service.registry.acquire = original_acquire  # type: ignore[method-assign]
        release_allowed.set()
        await asyncio.gather(observation, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_shutdown_drains_work_stops_child_and_promotes_stage_without_launch(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "managed-a", 19_123)
    managed_b = _managed_config(tmp_path, "managed-b", 19_124)
    service, factory_configs = _service(managed_a, supervisor)
    await service.start_and_test_audio_cpp()
    await _stage(service, managed_b)

    try:
        await service.shutdown_audio_cpp()
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert dict(snapshot.applied_config) == managed_b
        assert snapshot.staged_config is None
        assert factory_configs == [managed_a]
        assert supervisor.launches == 1
        assert supervisor.state == "stopped"
        assert service.registry._slots["audio_cpp"].active is None
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_applying_external_stops_child_and_never_relaunches_managed(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "managed", 19_125)
    external = AudioCppConfig(base_url="http://127.0.0.1:19126").to_mapping()
    service, factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    await _stage(service, external)

    try:
        assert await service.restart_audio_cpp() is None
        catalog = await service.get_catalog("audio_cpp", refresh=True)
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert catalog.models[0].model_id == "model"
        assert dict(snapshot.applied_config) == external
        assert factory_configs == [managed, external]
        assert supervisor.launches == 1
        assert supervisor.state == "stopped"
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_cancelled_lifecycle_waiter_does_not_abandon_accepted_transition(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed_a = _managed_config(tmp_path, "managed-a", 19_127)
    managed_b = _managed_config(tmp_path, "managed-b", 19_128)
    service, _factory_configs = _service(managed_a, supervisor)
    await service.start_and_test_audio_cpp()
    await _stage(service, managed_b)
    operation = await service.admit(_request())
    shutdown = asyncio.create_task(service.shutdown_audio_cpp())

    try:
        await asyncio.wait_for(supervisor.draining_started.wait(), timeout=1)
        shutdown.cancel("caller cancelled")
        await asyncio.sleep(0)
        assert shutdown.done() is False

        await operation.close()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await shutdown
        snapshot = await service.registry.provider_configuration_snapshot("audio_cpp")

        assert dict(snapshot.applied_config) == managed_b
        assert snapshot.staged_config is None
        assert supervisor.state == "stopped"
    finally:
        if not shutdown.done():
            await operation.close()
            await shutdown
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_service_close_keeps_child_alive_until_admitted_lease_releases(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "close-drain", 19_129)
    service, _factory_configs = _service(managed, supervisor)
    response = await service.synthesize(_request())
    close = asyncio.create_task(service.close())

    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert close.done() is False
        assert supervisor.state == "running"
        assert supervisor.close_calls == 0

        await response.aclose()
        await asyncio.wait_for(close, timeout=1)
        await asyncio.wait_for(service.wait_closed(), timeout=1)

        assert supervisor.state == "stopped"
        assert supervisor.close_calls == 1
        assert supervisor.wait_closed_calls == 1
    finally:
        await response.aclose()
        await asyncio.gather(close, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_service_close_joins_inflight_probe_before_client_and_terminal_close(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "close-order", 19_130)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    record = service.registry._slots["audio_cpp"].active
    assert record is not None
    adapter = record.adapter
    assert isinstance(adapter, AudioCppAdapter)
    bundle = adapter._managed_bundle
    assert bundle is not None

    class _TrackedClient:
        def __init__(self, client: httpx.AsyncClient, event: str) -> None:
            self._client = client
            self._event = event

        async def aclose(self) -> None:
            supervisor.events.append(self._event)
            await self._client.aclose()

    bundle.request_client = _TrackedClient(  # type: ignore[assignment]
        bundle.request_client,
        "request_client_close",
    )
    bundle.health_client = _TrackedClient(  # type: ignore[assignment]
        bundle.health_client,
        "health_client_close",
    )
    supervisor.inflight_probe_gate = asyncio.Event()
    close = asyncio.create_task(service.close())

    try:
        for _ in range(20):
            if "generation_stop" in supervisor.events:
                break
            await asyncio.sleep(0)

        assert "generation_stop" in supervisor.events
        assert "probe_joined" not in supervisor.events
        assert "request_client_close" not in supervisor.events
        assert "terminal_close" not in supervisor.events

        supervisor.inflight_probe_gate.set()
        await asyncio.wait_for(close, timeout=1)
        await asyncio.wait_for(service.wait_closed(), timeout=1)

        expected = (
            "generation_stop",
            "probe_joined",
            "request_client_close",
            "health_client_close",
            "terminal_close",
            "terminal_stop",
            "terminal_wait_closed",
        )
        positions = [supervisor.events.index(event) for event in expected]
        assert positions == sorted(positions)
    finally:
        if supervisor.inflight_probe_gate is not None:
            supervisor.inflight_probe_gate.set()
        await asyncio.gather(close, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_service_close_uses_one_deadline_for_registry_then_supervisor(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "close-deadline", 19_131)
    service, _factory_configs = _service(
        managed,
        supervisor,
        shutdown_timeout_seconds=0.25,
    )
    await service.start_and_test_audio_cpp()

    await service.close()
    await service.wait_closed()

    assert supervisor.close_calls == 1
    assert len(supervisor.deadline_observations) == 2
    first, terminal = supervisor.deadline_observations
    assert first is not None
    assert terminal == first


@pytest.mark.asyncio
async def test_close_deadline_reaches_lifecycle_transition_accepted_before_close(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "close-draining-deadline", 19_145)
    service, _factory_configs = _service(
        managed,
        supervisor,
        shutdown_timeout_seconds=0.25,
    )
    await service.start_and_test_audio_cpp()
    response = await service.synthesize(_request())
    restart = asyncio.create_task(service.restart_audio_cpp())
    close = None

    try:
        await asyncio.wait_for(supervisor.draining_started.wait(), timeout=1)
        close = asyncio.create_task(service.close())
        await asyncio.wait_for(service._close_signal.wait(), timeout=1)
        while service.registry._shutdown_deadline is None:
            await asyncio.sleep(0)
        expected_deadline = service.registry._shutdown_deadline

        await response.aclose()
        await asyncio.gather(restart, return_exceptions=True)
        await asyncio.gather(close, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)

        assert expected_deadline is not None
        assert supervisor.deadline_observations[0] == expected_deadline
    finally:
        await response.aclose()
        await asyncio.gather(restart, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_close_deadline_interrupts_lifecycle_stop_already_waiting_on_grace(
    tmp_path: Path,
) -> None:
    process = _LifecycleProcess(exit_on_terminate=False)
    launcher = _LifecycleLauncher([process])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    managed = _managed_config(tmp_path, "close-active-grace", 19_146)
    managed["managed_termination_grace_seconds"] = 1.0
    service, _factory_configs = _service(
        managed,
        supervisor,  # type: ignore[arg-type]
        shutdown_timeout_seconds=0.05,
    )
    await service.start_and_test_audio_cpp()
    lifecycle = asyncio.create_task(service.shutdown_audio_cpp())
    close = None

    try:
        while process.terminate_calls == 0:
            await asyncio.sleep(0)
        close = asyncio.create_task(service.close())
        await asyncio.wait_for(service._close_signal.wait(), timeout=1)

        for _ in range(30):
            if process.kill_calls:
                break
            await asyncio.sleep(0.01)

        assert process.kill_calls == 1
        await asyncio.gather(lifecycle, close, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)
    finally:
        process.exit(0)
        await asyncio.gather(lifecycle, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_close_cancels_replacement_spawn_before_startup_timeout(
    tmp_path: Path,
) -> None:
    first = _LifecycleProcess()
    replacement = _LifecycleProcess()
    launcher = _LifecycleLauncher([first, replacement], block_call=2)
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    managed = _managed_config(tmp_path, "close-replacement-spawn", 19_147)
    managed["managed_startup_timeout_seconds"] = 30.0
    service, _factory_configs = _service(
        managed,
        supervisor,  # type: ignore[arg-type]
        shutdown_timeout_seconds=0.05,
    )
    await service.start_and_test_audio_cpp()
    restart = asyncio.create_task(service.restart_audio_cpp())
    close = None
    waiter = None

    try:
        await asyncio.wait_for(launcher.blocked_call_started.wait(), timeout=1)
        close = asyncio.create_task(service.close())
        waiter = asyncio.create_task(service.wait_closed())

        await asyncio.wait_for(
            asyncio.gather(close, waiter, return_exceptions=True),
            timeout=0.5,
        )
        assert supervisor.snapshot().state == "stopped"
        assert restart.done() is True
    finally:
        launcher.release_blocked_call.set()
        first.exit(0)
        replacement.exit(0)
        await asyncio.gather(restart, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_first_use_after_exit_waits_cleanup_then_applies_latest_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _LifecycleProcess()
    replacement = _LifecycleProcess()
    launcher = _LifecycleLauncher([first, replacement])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    managed_a = _managed_config(tmp_path, "exit-stage-a", 19_148)
    managed_b = _managed_config(tmp_path, "exit-stage-b", 19_149)
    service, _factory_configs = _service(
        managed_a,
        supervisor,  # type: ignore[arg-type]
    )
    await service.start_and_test_audio_cpp()
    await _stage(service, managed_b, generation=1)
    output_join_started = asyncio.Event()
    release_output_join = asyncio.Event()

    async def blocked_output_join(_record: Any) -> None:
        output_join_started.set()
        await release_output_join.wait()

    monkeypatch.setattr(supervisor, "_join_output_drains", blocked_output_join)
    first.exit(12)
    await asyncio.wait_for(output_join_started.wait(), timeout=1)
    first_use = asyncio.create_task(service.start_and_test_audio_cpp())

    try:
        await asyncio.sleep(0)
        assert launcher.calls == 1
        assert first_use.done() is False

        release_output_join.set()
        catalog = await asyncio.wait_for(first_use, timeout=1)
        configuration = await service.registry.provider_configuration_snapshot(
            "audio_cpp"
        )

        assert catalog.health.fresh is True
        assert launcher.calls == 2
        assert launcher.launches[1].base_url == "http://127.0.0.1:19149"
        assert dict(configuration.applied_config) == managed_b
        assert configuration.staged_config is None
    finally:
        release_output_join.set()
        await asyncio.gather(first_use, return_exceptions=True)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_output_failure_invalidates_passive_catalog_before_child_exits(
    tmp_path: Path,
) -> None:
    process = _LifecycleProcess(exit_on_terminate=False)
    launcher = _LifecycleLauncher([process])
    supervisor = AudioCppSupervisor(
        source_environment={},
        process_launcher=launcher,
        port_preflight=_available_port,
    )
    managed = _managed_config(tmp_path, "output-failure-passive", 19_150)
    managed["managed_termination_grace_seconds"] = 1.0
    service, _factory_configs = _service(
        managed,
        supervisor,  # type: ignore[arg-type]
    )
    await service.start_and_test_audio_cpp()

    try:
        process.stderr.fail(RuntimeError("private output failure"))
        while process.terminate_calls == 0:
            await asyncio.sleep(0)

        catalog = await service.get_catalog("audio_cpp", refresh=False)
        observation = service.latest_native_capability_observation("audio_cpp")

        assert supervisor.snapshot().state == "unavailable"
        assert catalog.health.fresh is False
        assert catalog.health.state == "unavailable"
        assert observation is not None
        assert observation.snapshot.catalog is not None
        assert observation.snapshot.catalog.health.fresh is False
    finally:
        process.exit(7)
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["restart", "shutdown"])
async def test_lifecycle_commands_reject_after_service_close(
    tmp_path: Path,
    lifecycle: str,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, f"closed-{lifecycle}", 19_140)
    service, _factory_configs = _service(managed, supervisor)
    await service.close()
    await service.wait_closed()
    await service._audio_cpp_lifecycle_lock.acquire()

    try:
        with pytest.raises(TTSRegistryClosedError):
            async with asyncio.timeout(0.1):
                if lifecycle == "restart":
                    await service.restart_audio_cpp()
                else:
                    await service.shutdown_audio_cpp()
    finally:
        service._audio_cpp_lifecycle_lock.release()

    assert service._audio_cpp_lifecycle_tasks == set()


@pytest.mark.asyncio
async def test_wait_closed_joins_lifecycle_command_accepted_before_close(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "close-race", 19_141)
    service, _factory_configs = _service(managed, supervisor)
    await service._audio_cpp_lifecycle_lock.acquire()
    restart = asyncio.create_task(service.restart_audio_cpp())
    waiter = None

    try:
        while not service._audio_cpp_lifecycle_tasks:
            await asyncio.sleep(0)
        await service.close()
        waiter = asyncio.create_task(service.wait_closed())
        await asyncio.sleep(0)
        assert waiter.done() is False

        service._audio_cpp_lifecycle_lock.release()
        with pytest.raises(TTSRegistryClosedError):
            await restart
        with pytest.raises(RuntimeError, match="TTS shutdown cleanup failed"):
            await asyncio.wait_for(waiter, timeout=1)
        assert service._audio_cpp_lifecycle_tasks == set()
    finally:
        if service._audio_cpp_lifecycle_lock.locked():
            service._audio_cpp_lifecycle_lock.release()
        await asyncio.gather(restart, return_exceptions=True)
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.gather(service.wait_closed(), return_exceptions=True)


@pytest.mark.asyncio
async def test_lifecycle_retry_clears_prior_transition_unavailable_state(
    tmp_path: Path,
) -> None:
    supervisor = _PreparationSupervisor()
    managed = _managed_config(tmp_path, "retry", 19_132)
    service, _factory_configs = _service(managed, supervisor)
    await service.start_and_test_audio_cpp()
    supervisor.stop_failures_remaining = 1

    try:
        with pytest.raises(RuntimeError, match="simulated lifecycle failure"):
            await service.shutdown_audio_cpp()
        assert service.registry._slots["audio_cpp"].unavailable is True

        catalog = await service.restart_audio_cpp()

        assert catalog is not None
        assert catalog.models[0].model_id == "model"
        assert service.registry._slots["audio_cpp"].unavailable is False
        assert supervisor.state == "running"
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_real_child_argv_cwd_environment_readiness_and_cleanup(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("direct executable shebang wrappers require a POSIX host")
    wrapper = write_executable_wrapper(tmp_path / "fake_audiocpp_server")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    server_json = tmp_path / "server.json"
    server_json.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": port,
                "test_behavior": {
                    "readiness_delay_seconds": 0.05,
                    "stdout_chunks": ["fixture stdout line\n"],
                    "stderr_chunks": ["fixture stderr line\n"],
                    "observe_environment_names": [
                        "PATH",
                        "LANG",
                        "OPENAI_API_KEY",
                        "UNRELATED_SECRET",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    managed = AudioCppConfig(
        mode="managed",
        managed_binary_path=str(wrapper),
        managed_server_json_path=str(server_json),
        managed_startup_timeout_seconds=10.0,
        managed_health_check_interval_seconds=2.0,
        managed_termination_grace_seconds=0.1,
    ).to_mapping()
    launched: list[asyncio.subprocess.Process] = []

    async def capture_launch(launch: Any, environment: dict[str, str]) -> Any:
        owned = await supervisor_module._default_process_launcher(
            launch,
            environment,
        )
        launched.append(owned.process)
        return owned

    supervisor = AudioCppSupervisor(
        source_environment={
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C",
            "OPENAI_API_KEY": "SYNTHETIC_PRIVATE_PROVIDER_SECRET",
            "UNRELATED_SECRET": "SYNTHETIC_PRIVATE_UNRELATED_SECRET",
        },
        process_launcher=capture_launch,
    )

    def factory(config: Mapping[str, Any]) -> AudioCppAdapter:
        return AudioCppAdapter(
            AudioCppConfig.from_mapping(config),
            supervisor=supervisor,
        )

    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=factory,
                initial_config=managed,
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=_preferences(),
        audio_cpp_supervisor=supervisor,
    )

    try:
        catalog = await service.start_and_test_audio_cpp()
        snapshot = service.audio_cpp_process_snapshot()
        assert snapshot.endpoint is not None
        async with httpx.AsyncClient(
            base_url=snapshot.endpoint,
            trust_env=False,
            follow_redirects=False,
            timeout=1.0,
        ) as client:
            state_response = await client.get("/test/state")
            state_response.raise_for_status()
            state = state_response.json()

        response = await service.synthesize(
            TTSRequest(
                provider_id="audio_cpp",
                model_id="fixture-model",
                text="character roleplay response",
                voice=None,
                response_format="wav",
            )
        )
        audio = [chunk async for chunk in response.byte_stream]
        await response.aclose()

        assert [model.model_id for model in catalog.models] == ["fixture-model"]
        assert audio == [_wav()]
        assert state["pid"] == launched[0].pid
        assert state["argv"] == [
            str(wrapper),
            "--config",
            str(server_json),
        ]
        assert state["cwd"] == str(tmp_path)
        assert state["environment_present"] == {
            "PATH": True,
            "LANG": True,
            "OPENAI_API_KEY": False,
            "UNRELATED_SECRET": False,
        }
        assert {line.text for line in snapshot.diagnostics} == {
            "fixture stdout line",
            "fixture stderr line",
        }
    finally:
        try:
            await service.close()
            await service.wait_closed()
        finally:
            for process in launched:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    assert launched[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(launched[0].pid, 0)
    assert supervisor._generation is None
    assert supervisor._startup_task is None
    assert supervisor._stop_task is None


@pytest.mark.asyncio
async def test_real_child_uses_generated_multi_model_config_and_cleans_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("direct executable shebang wrappers require a POSIX host")
    wrapper = write_executable_wrapper(tmp_path / "fake_audiocpp_server")
    packages = [
        _accepted_guided_package(
            tmp_path / "models" / "supertonic",
            filename="supertonic-3-orig.gguf",
            package_variant="supertonic_3_orig",
            public_model_id="narrator",
        ),
        _accepted_guided_package(
            tmp_path / "models" / "pocket",
            filename="pocket-tts-english-q8_0.gguf",
            package_variant="pocket_tts_english_q8_0",
            public_model_id="clone-voice",
        ),
    ]
    settings = AudioCppSettingsConfig.from_mapping(
        {
            "mode": "managed",
            "managed_setup_source": "guided",
            "guided_binary_path": str(wrapper),
            "guided_packages": [
                package.model_dump(mode="json") for package in packages
            ],
            "guided_default_model_id": "narrator",
            "managed_startup_timeout_seconds": 10.0,
            "managed_health_check_interval_seconds": 2.0,
            "managed_termination_grace_seconds": 0.1,
        }
    )
    runtime_root = tmp_path / "generated"

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    launched: list[asyncio.subprocess.Process] = []
    launches: list[Any] = []

    async def capture_launch(launch: Any, environment: dict[str, str]) -> Any:
        owned = await supervisor_module._default_process_launcher(
            launch,
            environment,
        )
        launches.append(launch)
        launched.append(owned.process)
        return owned

    supervisor = AudioCppSupervisor(
        source_environment={"PATH": os.environ.get("PATH", "")},
        process_launcher=capture_launch,
    )
    adapter = AudioCppAdapter(
        AudioCppConfig.from_mapping(settings.to_mapping()),
        guided_settings=settings,
        supervisor=supervisor,
    )
    clone_materializer = TTSCloneReferenceMaterializer(
        tmp_path / "clone-materializations"
    )
    artifact_directory: Path | None = None

    try:
        catalog = await adapter.get_catalog(refresh=True)
        snapshot = supervisor.snapshot()
        assert snapshot.endpoint is not None
        artifact_directory = launches[0].working_directory
        document = json.loads(launches[0].server_json_path.read_text())
        async with httpx.AsyncClient(
            base_url=snapshot.endpoint,
            trust_env=False,
            follow_redirects=False,
            timeout=1.0,
        ) as client:
            state_response = await client.get("/test/state")
            state_response.raise_for_status()
            state = state_response.json()

        response = await adapter.synthesize(
            TTSRequest(
                provider_id="audio_cpp",
                model_id="narrator",
                text="character roleplay response",
                voice=None,
                response_format="wav",
            )
        )
        audio = [chunk async for chunk in response.byte_stream]
        await response.aclose()
        first_process_generation = response.metadata["process_generation"]

        second_request = TTSRequest(
            provider_id="audio_cpp",
            model_id="clone-voice",
            text="a second model in the same managed child",
            voice=None,
            response_format="wav",
        )
        second_capability = adapter.admit_clone_capability(second_request)
        second_owner = await clone_materializer.materialize(_clone_reference())
        second_response = await adapter.synthesize_clone(
            _new_admitted_audio_cpp_clone_request(
                request=second_request,
                materialization=second_owner,
                capability=second_capability,
                provider_revision=0,
                applied_provider_generation=0,
            )
        )
        second_audio = [chunk async for chunk in second_response.byte_stream]
        await second_response.aclose()

        assert [model.model_id for model in catalog.models] == [
            "narrator",
            "clone-voice",
        ]
        assert [model.speech_capabilities for model in catalog.models] == [
            ("tts",),
            ("tts", "clone"),
        ]
        assert audio == [_wav()]
        assert second_audio == [_wav()]
        assert second_response.metadata["process_generation"] == (
            first_process_generation
        )
        assert len(launched) == 1
        assert state["pid"] == launched[0].pid
        assert state["argv"] == [
            str(wrapper),
            "--config",
            str(launches[0].server_json_path),
        ]
        assert state["cwd"] == str(artifact_directory)
        assert "test_behavior" not in document
        assert len(tuple(runtime_root.iterdir())) == 1
    finally:
        try:
            await clone_materializer.close()
            await adapter.close()
        finally:
            await supervisor.close()
            await supervisor.wait_closed()
            for process in launched:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    assert len(launched) == 1
    assert launched[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(launched[0].pid, 0)
    assert artifact_directory is not None
    assert not artifact_directory.exists()
    assert runtime_root.exists()
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_real_generated_child_replaces_recovers_and_leaves_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("direct executable shebang wrappers require a POSIX host")
    wrapper = write_executable_wrapper(tmp_path / "fake_audiocpp_server")
    first = _guided_settings(
        tmp_path / "first",
        public_model_id="first-model",
    )
    second = _guided_settings(
        tmp_path / "second",
        filename="pocket-tts-english-q8_0.gguf",
        package_variant="pocket_tts_english_q8_0",
        public_model_id="second-model",
    )
    first = AudioCppSettingsConfig.from_mapping(
        {
            **first.to_mapping(),
            "guided_binary_path": str(wrapper),
            "managed_startup_timeout_seconds": 10.0,
            "managed_termination_grace_seconds": 0.1,
        }
    )
    second = AudioCppSettingsConfig.from_mapping(
        {
            **second.to_mapping(),
            "guided_binary_path": str(wrapper),
            "managed_startup_timeout_seconds": 10.0,
            "managed_termination_grace_seconds": 0.1,
        }
    )
    runtime_root = tmp_path / "generated"

    async def deterministic_materialize(current: AudioCppSettingsConfig):
        return await materialize_audio_cpp_guided_launch(
            current,
            artifact_root=runtime_root,
        )

    monkeypatch.setattr(
        audio_cpp_adapter_module,
        "materialize_audio_cpp_guided_launch",
        deterministic_materialize,
    )
    launched: list[asyncio.subprocess.Process] = []
    launches: list[Any] = []

    async def capture_launch(launch: Any, environment: dict[str, str]) -> Any:
        owned = await supervisor_module._default_process_launcher(
            launch,
            environment,
        )
        launches.append(launch)
        launched.append(owned.process)
        return owned

    supervisor = AudioCppSupervisor(
        source_environment={"PATH": os.environ.get("PATH", "")},
        process_launcher=capture_launch,
    )
    service, _factory_configs = _service(
        first.to_mapping(),
        supervisor,
        transport_handler=None,
    )

    try:
        initial = await service.start_and_test_audio_cpp()
        assert [model.model_id for model in initial.models] == ["first-model"]
        first_artifact = launches[0].working_directory

        await _stage(service, second.to_mapping(), generation=1)
        replacement = await service.restart_audio_cpp()
        assert replacement is not None
        assert [model.model_id for model in replacement.models] == ["second-model"]
        assert replacement.models[0].speech_capabilities == ("tts", "clone")
        second_artifact = launches[1].working_directory
        assert launched[0].returncode is not None
        assert not first_artifact.exists()
        assert second_artifact.exists()
        assert len(tuple(runtime_root.iterdir())) == 1

        launched[1].kill()
        async with asyncio.timeout(3.0):
            while supervisor._generation is not None:
                await asyncio.sleep(0.01)
        assert supervisor.snapshot().state == "unavailable"
        assert not second_artifact.exists()
        assert tuple(runtime_root.iterdir()) == ()

        recovered = await service.start_and_test_audio_cpp()
        assert [model.model_id for model in recovered.models] == ["second-model"]
        third_artifact = launches[2].working_directory
        assert third_artifact.exists()

        await service.shutdown_audio_cpp()
        assert launched[2].returncode is not None
        assert not third_artifact.exists()
        assert tuple(runtime_root.iterdir()) == ()
    finally:
        try:
            await service.close()
            await service.wait_closed()
        finally:
            for process in launched:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    assert len(launched) == 3
    assert all(process.returncode is not None for process in launched)
    for process in launched:
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
    assert supervisor._generation is None
    assert supervisor._startup_task is None
    assert supervisor._stop_task is None
    assert runtime_root.exists()
    assert tuple(runtime_root.iterdir()) == ()
