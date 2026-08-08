from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS._async_lifecycle import current_shutdown_deadline
from tldw_chatbook.TTS.adapter_types import (
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    AudioCppGenerationHooks,
    AudioCppProcessAdmissionSnapshot,
    AudioCppProcessSnapshot,
    AudioCppReadyEndpoint,
    _AudioCppGenerationChanged,
)
from tldw_chatbook.TTS.effective_settings import (
    TTSCharacterProfileSelection,
    TTSSelectionOverrides,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService


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


class _PreparationSupervisor:
    def __init__(self) -> None:
        self.state = "stopped"
        self.lifecycle_epoch = 0
        self.process_generation = 0
        self.observation_version = 0
        self.ensure_calls = 0
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
            tts_capability=("available" if self.state == "running" else "unknown"),
            consecutive_health_failures=0,
            last_failure=None,
            diagnostics=(),
            dropped_diagnostic_lines=0,
        )

    async def ensure_running(
        self,
        launch: Any,
        *,
        generation_hooks_factory: Callable[[int], Awaitable[AudioCppGenerationHooks]],
        require_existing: AudioCppProcessAdmissionSnapshot | None = None,
    ) -> AudioCppReadyEndpoint:
        self.ensure_calls += 1
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
            or require_existing.state != self.state
        ):
            raise _AudioCppGenerationChanged
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
        await self.hooks.contract_probe()
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
    supervisor: _PreparationSupervisor,
    *,
    shutdown_timeout_seconds: float = 10.0,
) -> tuple[TTSService, list[dict[str, Any]]]:
    factory_configs: list[dict[str, Any]] = []

    def factory(config: Mapping[str, Any]) -> AudioCppAdapter:
        factory_configs.append(dict(config))
        return AudioCppAdapter(
            AudioCppConfig.from_mapping(config),
            transport=httpx.MockTransport(_handler),
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
