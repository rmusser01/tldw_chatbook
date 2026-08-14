"""POSIX generation-local launch artifacts for guided audio.cpp setup."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.TTS.audio_cpp_guided_config import AudioCppSettingsConfig
from tldw_chatbook.TTS.audio_cpp_package_scanner import (
    scan_audio_cpp_package_root,
)


def _launch_api():
    from tldw_chatbook.TTS.audio_cpp_guided_launch import (
        AudioCppGuidedLaunchError,
        materialize_audio_cpp_guided_launch,
    )

    return AudioCppGuidedLaunchError, materialize_audio_cpp_guided_launch


def _write_gguf(root: Path, filename: str) -> None:
    root.mkdir(parents=True)
    (root / filename).write_bytes(b"GGUF" + (3).to_bytes(4, "little"))


def _accept(root: Path, package_variant: str, public_model_id: str):
    scan = scan_audio_cpp_package_root(root)
    candidates = tuple(
        candidate
        for discovery in scan.discoveries
        for candidate in discovery.match.candidates
        if candidate.recipe.package_variant == package_variant
    )
    assert len(candidates) == 1
    return candidates[0].accept(public_model_id=public_model_id)


def _managed_identity():
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppManagedArtifactIdentity,
    )

    return AudioCppManagedArtifactIdentity(
        artifact_id="audio-cpp-supertonic-3-orig",
        revision=AUDIO_CPP_ARTIFACT_COMMIT,
        variant="orig",
    )


def _pocket_managed_identity():
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppManagedArtifactIdentity,
    )

    return AudioCppManagedArtifactIdentity(
        artifact_id="audio-cpp-pocket-tts-english-q8-0",
        revision=AUDIO_CPP_ARTIFACT_COMMIT,
        variant="q8_0",
    )


class _ManagedLeaseSpy:
    def __init__(self, reference: object, root: Path) -> None:
        self.handle = SimpleNamespace(
            root=reference,
            closure=(reference,),
            paths=((reference, root),),
        )
        self.close_calls = 0
        self.fail_close = False

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("PRIVATE_LEASE_CLOSE_DETAIL")


class _ManagedServiceSpy:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.activate_calls: list[object] = []
        self.acquire_calls: list[object] = []
        self.leases: list[_ManagedLeaseSpy] = []

    def activate(self, reference: object) -> object:
        self.activate_calls.append(reference)
        return reference

    def acquire(self, reference: object) -> _ManagedLeaseSpy:
        self.acquire_calls.append(reference)
        lease = _ManagedLeaseSpy(reference, self.root)
        self.leases.append(lease)
        return lease


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "audiocpp_server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    return binary


def test_binary_validation_applies_shared_arbitrary_path_security_rules(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_guided_launch import _validate_binary

    binary = tmp_path / "bin" / "audio;cpp_server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)

    assert _validate_binary(str(binary)) is None


def _settings(
    binary: Path,
    packages: list[object],
    **updates: object,
) -> AudioCppSettingsConfig:
    values: dict[str, object] = {
        "mode": "managed",
        "managed_setup_source": "guided",
        "guided_binary_path": str(binary),
        "guided_packages": [
            package.model_dump(mode="json")  # type: ignore[attr-defined]
            for package in packages
        ],
        "guided_default_model_id": packages[0].public_model_id,  # type: ignore[attr-defined]
        "guided_backend_preference": "auto",
        "guided_device": 0,
        "guided_threads": 4,
        "guided_max_request_body_bytes": 64 * 1024 * 1024,
        "guided_busy_timeout_ms": 90_000,
        "managed_startup_timeout_seconds": 45.0,
        "managed_health_check_interval_seconds": 15.0,
        "managed_termination_grace_seconds": 8.0,
    }
    values.update(updates)
    return AudioCppSettingsConfig.from_mapping(values)


@pytest.mark.asyncio
async def test_materializes_exact_private_multi_model_server_json(
    tmp_path: Path,
) -> None:
    _, materialize = _launch_api()
    supertonic_root = tmp_path / "models" / "supertonic"
    pocket_root = tmp_path / "models" / "pocket"
    _write_gguf(supertonic_root, "supertonic-3-orig.gguf")
    _write_gguf(pocket_root, "pocket-tts-english-q8_0.gguf")
    supertonic = _accept(
        supertonic_root,
        "supertonic_3_orig",
        "narrator",
    )
    pocket = _accept(
        pocket_root,
        "pocket_tts_english_q8_0",
        "clone-voice",
    )
    settings = _settings(_binary(tmp_path), [supertonic, pocket])
    runtime_root = tmp_path / "runtime"

    launch = await materialize(
        settings,
        artifact_root=runtime_root,
        port_selector=lambda: 54_321,
        system="darwin",
        architecture="arm64",
    )

    assert launch.base_url == "http://127.0.0.1:54321"
    assert launch.working_directory == launch.server_json_path.parent
    assert launch.generated_artifact is not None
    assert tuple(
        (
            model.model_id,
            model.family,
            model.task,
            model.mode,
            model.speech_capabilities,
        )
        for model in launch.expected_models
    ) == (
        (
            "narrator",
            "supertonic",
            "tts",
            "offline",
            ("tts",),
        ),
        (
            "clone-voice",
            "pocket_tts",
            "tts",
            "offline",
            ("tts", "clone"),
        ),
    )
    document = json.loads(launch.server_json_path.read_text(encoding="utf-8"))
    assert document == {
        "host": "127.0.0.1",
        "port": 54_321,
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "lazy_load": True,
        "log_request_body": False,
        "max_request_body_bytes": 64 * 1024 * 1024,
        "busy_timeout_ms": 90_000,
        "models": [
            {
                "id": "narrator",
                "family": "supertonic",
                "path": str(supertonic_root / "supertonic-3-orig.gguf"),
                "task": "tts",
                "mode": "offline",
            },
            {
                "id": "clone-voice",
                "family": "pocket_tts",
                "path": str(pocket_root / "pocket-tts-english-q8_0.gguf"),
                "task": "tts",
                "mode": "offline",
                "load_options": {"language": "english"},
                "session_options": {"language": "english"},
            },
        ],
    }
    assert "cors_origins" not in document
    assert stat.S_IMODE(launch.working_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(launch.server_json_path.stat().st_mode) == 0o400

    artifact_directory = launch.working_directory
    launch.generated_artifact.cleanup()
    launch.generated_artifact.cleanup()
    assert not artifact_directory.exists()


@pytest.mark.asyncio
async def test_managed_launch_activates_acquires_and_retains_exact_root_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    _, materialize = _launch_api()
    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    scan = scan_audio_cpp_package_root(root)
    candidate = scan.discoveries[0].match.candidates[0]
    accepted = candidate.accept(
        public_model_id="managed-narrator",
        managed_artifact=_managed_identity(),
    )
    service = _ManagedServiceSpy(root)
    monkeypatch.setattr(launch_module, "managed_service", lambda: service)

    launch = await materialize(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime-managed",
        port_selector=lambda: 54_322,
        system="darwin",
        architecture="arm64",
    )

    assert launch.expected_models[0].model_id == "managed-narrator"
    assert launch.expected_models[0].family == "supertonic"
    assert launch.server_json_path.is_file()
    identity = _managed_identity()
    reference = ArtifactRef(identity.artifact_id, identity.revision, identity.variant)
    assert service.activate_calls == [reference]
    assert service.acquire_calls == [reference]
    assert service.leases[0].close_calls == 0
    assert json.loads(launch.server_json_path.read_text())["models"][0]["path"] == str(
        root / "supertonic-3-orig.gguf"
    )
    launch.generated_artifact.cleanup()
    assert service.leases[0].close_calls == 1


@pytest.mark.asyncio
async def test_local_launch_does_not_construct_managed_artifact_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "local-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "local-narrator")

    def unexpected_service() -> object:
        raise AssertionError("local packages must not touch the managed store")

    monkeypatch.setattr(launch_module, "managed_service", unexpected_service)
    launch = await launch_module.materialize_audio_cpp_guided_launch(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime-local",
        port_selector=lambda: 54_327,
        system="darwin",
        architecture="arm64",
    )

    assert launch.generated_artifact is not None
    launch.generated_artifact.cleanup()


@pytest.mark.asyncio
async def test_managed_service_factory_failure_is_off_loop_stable_and_cleans_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-artifact service construction is contained by artifact ownership."""

    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    main_thread = threading.get_ident()
    factory_threads: list[int] = []

    def failed_factory() -> object:
        factory_threads.append(threading.get_ident())
        raise RuntimeError("PRIVATE_SERVICE_FACTORY_CANARY")

    monkeypatch.setattr(launch_module, "managed_service", failed_factory)
    artifact_root = tmp_path / "runtime-service-failure"

    with pytest.raises(launch_module.AudioCppGuidedLaunchError) as caught:
        await launch_module.materialize_audio_cpp_guided_launch(
            _settings(_binary(tmp_path), [accepted]),
            artifact_root=artifact_root,
            port_selector=lambda: 54_333,
            system="darwin",
            architecture="arm64",
        )

    assert caught.value.code == "package_changed"
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert factory_threads and factory_threads[0] != main_thread
    assert tuple(artifact_root.iterdir()) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("control_type", (SystemExit, GeneratorExit, KeyboardInterrupt))
async def test_managed_activate_control_flow_preserves_exact_signal_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_type: type[BaseException],
) -> None:
    """Managed activation cannot translate interpreter control into validation."""

    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    signal = control_type("PRIVATE_CONTROL_CANARY")

    class Service:
        def activate(self, _reference: object) -> None:
            raise signal

        def acquire(self, _reference: object) -> object:
            raise AssertionError("acquire must not run after activation control flow")

    monkeypatch.setattr(launch_module, "managed_service", Service)
    artifact_root = tmp_path / f"runtime-{control_type.__name__}"
    real_rmdir = launch_module.os.rmdir
    rmdir_calls = 0

    def fail_first_cleanup(*args: object, **kwargs: object) -> None:
        nonlocal rmdir_calls
        rmdir_calls += 1
        if rmdir_calls == 1:
            raise OSError("PRIVATE_CONTROL_CLEANUP_CANARY")
        real_rmdir(*args, **kwargs)

    monkeypatch.setattr(launch_module.os, "rmdir", fail_first_cleanup)

    with pytest.raises(control_type) as caught:
        await launch_module.materialize_audio_cpp_guided_launch(
            _settings(_binary(tmp_path), [accepted]),
            artifact_root=artifact_root,
            port_selector=lambda: 54_334,
            system="darwin",
            architecture="arm64",
        )

    assert caught.value is signal
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    cleanup_owner = launch_module.take_audio_cpp_guided_cleanup_owner(caught.value)
    assert cleanup_owner is not None
    cleanup_owner.cleanup()
    assert rmdir_calls == 2
    assert tuple(artifact_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_managed_service_factory_cancellation_is_shielded_and_cleans_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation joins off-loop service construction before final cleanup."""

    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    factory_started = threading.Event()
    allow_factory = threading.Event()

    class Service:
        def activate(self, _reference: object) -> None:
            raise AssertionError("activation must not follow cancelled construction")

    def factory() -> object:
        factory_started.set()
        assert allow_factory.wait(2)
        return Service()

    monkeypatch.setattr(launch_module, "managed_service", factory)
    artifact_root = tmp_path / "runtime-cancel-service"
    task = asyncio.create_task(
        launch_module.materialize_audio_cpp_guided_launch(
            _settings(_binary(tmp_path), [accepted]),
            artifact_root=artifact_root,
            port_selector=lambda: 54_335,
            system="darwin",
            architecture="arm64",
        )
    )
    assert await asyncio.to_thread(factory_started.wait, 2)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    allow_factory.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert tuple(artifact_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_post_acquisition_launch_construction_failure_releases_exact_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final snapshot construction remains inside generated-artifact ownership."""

    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    service = _ManagedServiceSpy(root)
    monkeypatch.setattr(launch_module, "managed_service", lambda: service)

    def fail_snapshot(**_kwargs: object) -> object:
        raise RuntimeError("PRIVATE_FINAL_SNAPSHOT_CANARY")

    monkeypatch.setattr(launch_module, "AudioCppManagedLaunchConfig", fail_snapshot)
    artifact_root = tmp_path / "runtime-final-construction"

    with pytest.raises(launch_module.AudioCppGuidedLaunchError) as caught:
        await launch_module.materialize_audio_cpp_guided_launch(
            _settings(_binary(tmp_path), [accepted]),
            artifact_root=artifact_root,
            port_selector=lambda: 54_337,
            system="darwin",
            architecture="arm64",
        )

    assert caught.value.code == "package_changed"
    assert "PRIVATE" not in str(caught.value)
    assert service.leases[0].close_calls == 1
    assert tuple(artifact_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_generated_cleanup_retries_before_releasing_managed_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    service = _ManagedServiceSpy(root)
    monkeypatch.setattr(launch_module, "managed_service", lambda: service)
    launch = await launch_module.materialize_audio_cpp_guided_launch(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime-retry",
        port_selector=lambda: 54_328,
        system="darwin",
        architecture="arm64",
    )
    artifact = launch.generated_artifact
    assert artifact is not None
    real_rmdir = launch_module.os.rmdir
    attempts = 0

    def fail_once(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("PRIVATE_CONFIG_CLEANUP_DETAIL")
        real_rmdir(*args, **kwargs)

    monkeypatch.setattr(launch_module.os, "rmdir", fail_once)

    with pytest.raises(launch_module.AudioCppGuidedLaunchError) as first:
        artifact.cleanup()
    assert first.value.code == "artifact_cleanup_failed"
    assert service.leases[0].close_calls == 0

    artifact.cleanup()
    assert service.leases[0].close_calls == 1
    assert not launch.working_directory.exists()


@pytest.mark.asyncio
async def test_real_store_removal_stays_blocked_until_generated_cleanup_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live real-store lease survives failed config cleanup exactly."""

    from Tests.Model_Artifacts.test_service import (
        install_descriptor_payload,
        single_file_descriptor,
    )
    from tldw_chatbook.Model_Artifacts import (
        ArtifactInUseError,
        ArtifactRef,
        ArtifactRole,
        ModelArtifactService,
    )
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    content = b"real-managed-payload"
    reference = ArtifactRef("audio-cpp-real-contention", "a" * 40, "f16")
    descriptor = single_file_descriptor(reference, ArtifactRole.ROOT, content)
    service = ModelArtifactService(
        tmp_path / "artifact-store",
        lease_timeout_seconds=0.02,
    )
    install_descriptor_payload(service, tmp_path, descriptor, content)
    service.activate(reference)
    leased = service.acquire(reference)

    local_root = tmp_path / "local-model"
    _write_gguf(local_root, "supertonic-3-orig.gguf")
    launch = await launch_module.materialize_audio_cpp_guided_launch(
        _settings(
            _binary(tmp_path),
            [_accept(local_root, "supertonic_3_orig", "local-narrator")],
        ),
        artifact_root=tmp_path / "runtime-real-contention",
        port_selector=lambda: 54_336,
        system="darwin",
        architecture="arm64",
    )
    artifact = launch.generated_artifact
    assert artifact is not None
    artifact.retain_managed_handle(leased)
    real_rmdir = launch_module.os.rmdir
    fail_cleanup = True

    def controlled_rmdir(*args: object, **kwargs: object) -> None:
        if fail_cleanup:
            raise OSError("PRIVATE_REAL_STORE_CLEANUP")
        real_rmdir(*args, **kwargs)

    monkeypatch.setattr(launch_module.os, "rmdir", controlled_rmdir)

    with pytest.raises(launch_module.AudioCppGuidedLaunchError):
        artifact.cleanup()
    with pytest.raises(ArtifactInUseError):
        service.delete(reference)

    fail_cleanup = False
    artifact.cleanup()
    service.delete(reference)
    assert not service.artifact_path(reference).exists()


@pytest.mark.asyncio
async def test_generated_cleanup_preserves_control_flow_and_retains_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    service = _ManagedServiceSpy(root)
    monkeypatch.setattr(launch_module, "managed_service", lambda: service)
    launch = await launch_module.materialize_audio_cpp_guided_launch(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime-control",
        port_selector=lambda: 54_331,
        system="darwin",
        architecture="arm64",
    )
    lease = service.leases[0]
    real_close = lease.close
    attempts = 0

    def interrupt_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise KeyboardInterrupt
        real_close()

    monkeypatch.setattr(lease, "close", interrupt_once)

    with pytest.raises(KeyboardInterrupt):
        launch.generated_artifact.cleanup()
    launch.generated_artifact.cleanup()

    assert attempts == 2
    assert lease.close_calls == 1


@pytest.mark.asyncio
async def test_descriptor_close_error_is_not_unsafely_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX close ambiguity is terminal after the fd number loses authority."""

    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    launch = await launch_module.materialize_audio_cpp_guided_launch(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime-close",
        port_selector=lambda: 54_332,
        system="darwin",
        architecture="arm64",
    )
    real_close = launch_module.os.close
    closed: list[int] = []

    def ambiguous_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        if len(closed) == 1:
            raise OSError("PRIVATE_AMBIGUOUS_CLOSE_DETAIL")

    monkeypatch.setattr(launch_module.os, "close", ambiguous_close)

    launch.generated_artifact.cleanup()
    launch.generated_artifact.cleanup()

    assert len(closed) == 2
    assert not launch.working_directory.exists()


@pytest.mark.asyncio
async def test_partial_managed_acquisition_failure_retains_failed_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    supertonic_root = tmp_path / "models" / "supertonic"
    pocket_root = tmp_path / "models" / "pocket"
    _write_gguf(supertonic_root, "supertonic-3-orig.gguf")
    _write_gguf(pocket_root, "pocket-tts-english-q8_0.gguf")
    first = _accept(supertonic_root, "supertonic_3_orig", "narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    second = _accept(pocket_root, "pocket_tts_english_q8_0", "clone").model_copy(
        update={"managed_artifact": _pocket_managed_identity()}
    )
    roots = {
        ArtifactRef(
            first.managed_artifact.artifact_id,
            first.managed_artifact.revision,
            first.managed_artifact.variant,
        ): supertonic_root,
        ArtifactRef(
            second.managed_artifact.artifact_id,
            second.managed_artifact.revision,
            second.managed_artifact.variant,
        ): pocket_root,
    }
    first_lease: _ManagedLeaseSpy | None = None

    class PartialService:
        def activate(self, reference: object) -> object:
            return reference

        def acquire(self, reference: object) -> _ManagedLeaseSpy:
            nonlocal first_lease
            if first_lease is None:
                first_lease = _ManagedLeaseSpy(reference, roots[reference])
                first_lease.fail_close = True
                return first_lease
            raise RuntimeError("PRIVATE_SECOND_ACQUIRE_DETAIL")

    monkeypatch.setattr(launch_module, "managed_service", PartialService)

    with pytest.raises(launch_module.AudioCppGuidedLaunchError) as caught:
        await launch_module.materialize_audio_cpp_guided_launch(
            _settings(_binary(tmp_path), [first, second]),
            artifact_root=tmp_path / "runtime-partial",
            port_selector=lambda: 54_329,
            system="darwin",
            architecture="arm64",
        )

    assert caught.value.code == "artifact_cleanup_failed"
    assert "PRIVATE" not in str(caught.value)
    cleanup_owner = caught.value.take_cleanup_owner()
    assert cleanup_owner is not None
    assert first_lease is not None
    assert first_lease.close_calls == 1
    first_lease.fail_close = False
    cleanup_owner.cleanup()
    assert first_lease.close_calls == 2


@pytest.mark.asyncio
async def test_cancellation_waits_for_managed_acquisition_then_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "managed-narrator").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    service = _ManagedServiceSpy(root)
    acquired = threading.Event()
    release = threading.Event()
    real_acquire = service.acquire

    def blocked_acquire(reference: object) -> _ManagedLeaseSpy:
        lease = real_acquire(reference)
        acquired.set()
        release.wait(timeout=2)
        return lease

    service.acquire = blocked_acquire  # type: ignore[method-assign]
    monkeypatch.setattr(launch_module, "managed_service", lambda: service)
    task = asyncio.create_task(
        launch_module.materialize_audio_cpp_guided_launch(
            _settings(_binary(tmp_path), [accepted]),
            artifact_root=tmp_path / "runtime-cancel-acquire",
            port_selector=lambda: 54_330,
            system="darwin",
            architecture="arm64",
        )
    )
    assert await asyncio.to_thread(acquired.wait, 2)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.leases[0].close_calls == 1
    assert tuple((tmp_path / "runtime-cancel-acquire").iterdir()) == ()


@pytest.mark.asyncio
async def test_managed_launch_rejects_a_new_second_exact_candidate(
    tmp_path: Path,
) -> None:
    error_type, materialize = _launch_api()
    root = tmp_path / "models" / "managed-supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    (root / "pocket-tts-english-bf16.gguf").write_bytes(
        b"GGUF" + (3).to_bytes(4, "little")
    )
    candidates = scan_audio_cpp_package_root(root).discoveries[0].match.candidates
    accepted = next(
        candidate
        for candidate in candidates
        if candidate.recipe.package_variant == "supertonic_3_orig"
    ).accept(
        public_model_id="managed-narrator",
        managed_artifact=_managed_identity(),
    )

    with pytest.raises(error_type) as raised:
        await materialize(
            _settings(_binary(tmp_path), [accepted]),
            artifact_root=tmp_path / "runtime-managed",
            port_selector=lambda: 54_322,
            system="darwin",
            architecture="arm64",
        )

    assert raised.value.code == "package_changed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_launch_revalidation_uses_managed_contract_only_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    managed_root = tmp_path / "managed"
    legacy_root = tmp_path / "legacy"
    _write_gguf(managed_root, "supertonic-3-orig.gguf")
    _write_gguf(legacy_root, "supertonic-3-orig.gguf")
    managed = _accept(managed_root, "supertonic_3_orig", "managed").model_copy(
        update={"managed_artifact": _managed_identity()}
    )
    legacy = _accept(legacy_root, "supertonic_3_orig", "legacy")
    real_scan = launch_module.scan_audio_cpp_package_root_async
    calls: list[dict[str, object]] = []

    async def recording_scan(root, **kwargs):
        calls.append(kwargs)
        return await real_scan(root, **kwargs)

    monkeypatch.setattr(
        launch_module,
        "scan_audio_cpp_package_root_async",
        recording_scan,
    )

    recipes = await launch_module.revalidate_audio_cpp_guided_packages(
        (managed, legacy)
    )

    assert len(recipes) == 2
    assert calls == [
        {
            "expected_managed_artifact": managed.managed_artifact,
            "expected_canonical_root": managed.canonical_root,
        },
        {},
    ]


@pytest.mark.asyncio
async def test_materializes_explicitly_accepted_models_from_one_ambiguous_root(
    tmp_path: Path,
) -> None:
    _, materialize = _launch_api()
    package_root = tmp_path / "models"
    package_root.mkdir()
    for filename in (
        "supertonic-3-orig.gguf",
        "pocket-tts-english-bf16.gguf",
    ):
        (package_root / filename).write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
    scan = scan_audio_cpp_package_root(package_root)
    assert scan.discoveries[0].match.state.value == "ambiguous"
    candidates = {
        candidate.recipe.package_variant: candidate
        for candidate in scan.discoveries[0].match.candidates
    }
    packages = [
        candidates["supertonic_3_orig"].accept(public_model_id="narrator"),
        candidates["pocket_tts_english_bf16"].accept(public_model_id="clone-voice"),
    ]

    launch = await materialize(
        _settings(_binary(tmp_path), packages),
        artifact_root=tmp_path / "runtime",
        port_selector=lambda: 54_322,
        system="darwin",
        architecture="arm64",
    )

    assert [model.model_id for model in launch.expected_models] == [
        "narrator",
        "clone-voice",
    ]
    assert launch.generated_artifact is not None
    launch.generated_artifact.cleanup()


@pytest.mark.asyncio
async def test_explicit_backend_requires_every_recipe_platform_tuple(
    tmp_path: Path,
) -> None:
    error_type, materialize = _launch_api()
    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    settings = _settings(
        _binary(tmp_path),
        [accepted],
        guided_backend_preference="metal",
    )
    runtime_root = tmp_path / "runtime"

    with pytest.raises(error_type) as caught:
        await materialize(
            settings,
            artifact_root=runtime_root,
            port_selector=lambda: 54_321,
            system="darwin",
            architecture="arm64",
        )

    assert caught.value.code == "backend_unsupported"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not runtime_root.exists()


@pytest.mark.asyncio
async def test_source_change_fails_with_context_free_path_independent_error(
    tmp_path: Path,
) -> None:
    error_type, materialize = _launch_api()
    root = tmp_path / "private-model-package"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    settings = _settings(_binary(tmp_path), [accepted])
    (root / "supertonic-3-orig.gguf").write_bytes(b"changed")

    with pytest.raises(error_type) as caught:
        await materialize(
            settings,
            artifact_root=tmp_path / "runtime",
            port_selector=lambda: 54_321,
            system="darwin",
            architecture="arm64",
        )

    assert caught.value.code == "package_changed"
    assert str(root) not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert not (tmp_path / "runtime").exists()


@pytest.mark.asyncio
async def test_linux_arm64_alias_is_resolved_against_aarch64_recipe_evidence(
    tmp_path: Path,
) -> None:
    _, materialize = _launch_api()
    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")

    launch = await materialize(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime",
        port_selector=lambda: 54_322,
        system="linux",
        architecture="arm64",
    )

    assert json.loads(launch.server_json_path.read_text())["backend"] == "cpu"
    assert launch.generated_artifact is not None
    launch.generated_artifact.cleanup()


@pytest.mark.asyncio
async def test_cancellation_during_artifact_creation_retires_completed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    settings = _settings(_binary(tmp_path), [accepted])
    runtime_root = tmp_path / "runtime"
    real_create = launch_module._create_artifact
    created = threading.Event()
    release = threading.Event()

    def blocked_create(*args, **kwargs):
        artifact = real_create(*args, **kwargs)
        created.set()
        release.wait(timeout=2)
        return artifact

    monkeypatch.setattr(launch_module, "_create_artifact", blocked_create)
    task = asyncio.create_task(
        launch_module.materialize_audio_cpp_guided_launch(
            settings,
            artifact_root=runtime_root,
            port_selector=lambda: 54_323,
            system="darwin",
            architecture="arm64",
        )
    )
    assert await asyncio.to_thread(created.wait, 2)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime_root.exists()
    assert tuple(runtime_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_cleanup_never_deletes_a_foreign_replacement(tmp_path: Path) -> None:
    error_type, materialize = _launch_api()
    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    launch = await materialize(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime",
        port_selector=lambda: 54_324,
        system="darwin",
        architecture="arm64",
    )
    artifact = launch.generated_artifact
    assert artifact is not None
    launch.server_json_path.unlink()
    launch.server_json_path.write_text("foreign", encoding="utf-8")

    with pytest.raises(error_type) as caught:
        artifact.cleanup()

    assert caught.value.code == "artifact_cleanup_failed"
    assert launch.server_json_path.read_text(encoding="utf-8") == "foreign"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_hardlinked_generated_config_is_no_longer_exactly_owned(
    tmp_path: Path,
) -> None:
    error_type, materialize = _launch_api()
    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    launch = await materialize(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime",
        port_selector=lambda: 54_325,
        system="darwin",
        architecture="arm64",
    )
    artifact = launch.generated_artifact
    assert artifact is not None
    retained_link = tmp_path / "retained-server.json"
    os.link(launch.server_json_path, retained_link)

    with pytest.raises(error_type) as validation:
        artifact.validate()
    with pytest.raises(error_type) as cleanup:
        artifact.cleanup()

    assert validation.value.code == "artifact_changed"
    assert cleanup.value.code == "artifact_cleanup_failed"
    assert launch.server_json_path.exists()
    assert retained_link.exists()
    assert validation.value.__cause__ is None
    assert validation.value.__context__ is None
    assert cleanup.value.__cause__ is None
    assert cleanup.value.__context__ is None


@pytest.mark.asyncio
async def test_cleanup_removes_exact_owned_directory_when_file_is_already_gone(
    tmp_path: Path,
) -> None:
    _, materialize = _launch_api()
    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    launch = await materialize(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime",
        port_selector=lambda: 54_325,
        system="darwin",
        architecture="arm64",
    )
    artifact = launch.generated_artifact
    assert artifact is not None
    artifact_directory = launch.working_directory
    launch.server_json_path.unlink()

    artifact.cleanup()

    assert not artifact_directory.exists()


@pytest.mark.asyncio
async def test_validation_rejects_size_change_without_reading_enlarged_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.TTS import audio_cpp_guided_launch as launch_module

    error_type, materialize = _launch_api()
    root = tmp_path / "models" / "supertonic"
    _write_gguf(root, "supertonic-3-orig.gguf")
    accepted = _accept(root, "supertonic_3_orig", "narrator")
    launch = await materialize(
        _settings(_binary(tmp_path), [accepted]),
        artifact_root=tmp_path / "runtime",
        port_selector=lambda: 54_326,
        system="darwin",
        architecture="arm64",
    )
    artifact = launch.generated_artifact
    assert artifact is not None
    launch.server_json_path.chmod(0o600)
    with launch.server_json_path.open("ab") as stream:
        stream.write(b"x" * (2 * 1024 * 1024))
    launch.server_json_path.chmod(0o400)

    def unexpected_read(_fd: int, _size: int) -> bytes:
        raise AssertionError("an enlarged generated config must not be read")

    monkeypatch.setattr(launch_module.os, "read", unexpected_read)
    with pytest.raises(error_type) as caught:
        artifact.validate()

    assert caught.value.code == "artifact_changed"
    artifact.cleanup()
