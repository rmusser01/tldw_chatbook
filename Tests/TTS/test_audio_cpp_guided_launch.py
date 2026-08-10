"""POSIX generation-local launch artifacts for guided audio.cpp setup."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from pathlib import Path

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


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "audiocpp_server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    return binary


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
