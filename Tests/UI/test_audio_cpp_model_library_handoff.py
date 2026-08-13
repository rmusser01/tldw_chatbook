"""Typed Settings-to-Model-Library handoff contracts for audio.cpp."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
import hashlib
from pathlib import Path
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
    AudioCppModelInstallOwner,
    AudioCppModelLibraryRequest,
    AudioCppModelLibraryResult,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffValueError,
    PendingHandoffStore,
)


def _request() -> AudioCppModelLibraryRequest:
    return AudioCppModelLibraryRequest(token="request-token-1", draft_revision=7)


def _result(root: Path) -> AudioCppModelLibraryResult:
    return AudioCppModelLibraryResult(
        token="request-token-1",
        draft_revision=7,
        artifact_id="audio-cpp-supertonic-3",
        revision="a" * 40,
        variant="f16",
        canonical_root=str(root),
    )


def test_handoff_values_are_frozen_slotted_and_root_redacted(tmp_path: Path) -> None:
    request = _request()
    result = _result(tmp_path.resolve())

    assert [item.name for item in fields(request)] == ["token", "draft_revision"]
    assert [item.name for item in fields(result)] == [
        "token",
        "draft_revision",
        "artifact_id",
        "revision",
        "variant",
        "canonical_root",
    ]
    assert not hasattr(request, "__dict__")
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        request.draft_revision = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.variant = "q8"  # type: ignore[misc]
    assert result.canonical_root not in repr(result)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (AudioCppModelLibraryRequest, {"token": "", "draft_revision": 1}),
        (AudioCppModelLibraryRequest, {"token": " request ", "draft_revision": 1}),
        (AudioCppModelLibraryRequest, {"token": "request", "draft_revision": True}),
        (AudioCppModelLibraryRequest, {"token": "request", "draft_revision": -1}),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "../private",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "/managed/root",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "relative/root",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "/managed/../private",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "/",
            },
        ),
        (
            AudioCppModelLibraryResult,
            {
                "token": "request",
                "draft_revision": 1,
                "artifact_id": "audio-cpp-model",
                "revision": "a" * 40,
                "variant": "f16",
                "canonical_root": "C:/managed/model",
            },
        ),
    ],
)
def test_handoff_values_reject_noncanonical_scalars(factory, kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


def test_audio_cpp_handoff_channels_are_explicit_and_independent(
    tmp_path: Path,
) -> None:
    assert HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST.value == (
        "audio_cpp_model_library_request"
    )
    assert HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT.value == (
        "audio_cpp_model_library_result"
    )
    store = PendingHandoffStore()

    assert store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, _request()) == 1
    assert (
        store.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            _result(tmp_path.resolve()),
        )
        == 1
    )

    request_claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    result_claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
    assert request_claim is not None
    assert result_claim is not None
    assert request_claim.value == _request()
    assert result_claim.value == _result(tmp_path.resolve())


@pytest.mark.parametrize(
    ("channel", "value_factory"),
    [
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, _request),
        (
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            lambda: _result(Path("/managed/root")),
        ),
    ],
)
def test_audio_cpp_claim_is_one_time_and_release_replays_exact_value(
    channel: HandoffChannel,
    value_factory,
) -> None:
    store = PendingHandoffStore()
    original = value_factory()
    revision = store.stage(channel, original)
    claim = store.claim(channel)

    assert claim is not None
    assert claim.revision == revision
    assert claim.value == original
    assert claim.value is not original
    assert store.claim(channel) is None
    assert store.release(claim) is True
    assert store.release(claim) is False

    replay = store.claim(channel)
    assert replay is not None
    assert replay.revision == revision
    assert replay.value == original
    assert replay.value is not claim.value
    assert store.acknowledge(claim) is False
    assert store.acknowledge(replay) is True
    assert store.acknowledge(replay) is False
    assert store.claim(channel) is None


def test_audio_cpp_store_rejects_hostile_subclasses_partial_and_wrong_values(
    tmp_path: Path,
) -> None:
    class HostileRequest(AudioCppModelLibraryRequest):
        pass

    class HostileResult(AudioCppModelLibraryResult):
        pass

    partial = object.__new__(AudioCppModelLibraryRequest)
    values = (
        (
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
            HostileRequest(token="request", draft_revision=1),
        ),
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, partial),
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, {"token": "request"}),
        (
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            HostileResult(
                token="request",
                draft_revision=1,
                artifact_id="audio-cpp-model",
                revision="a" * 40,
                variant="f16",
                canonical_root=str(tmp_path.resolve()),
            ),
        ),
        (HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, _request()),
    )

    for channel, value in values:
        store = PendingHandoffStore()
        with pytest.raises(HandoffValueError):
            store.stage(channel, value)
        assert store.claim(channel) is None


def test_audio_cpp_detached_copy_reconstructs_every_scalar(tmp_path: Path) -> None:
    source = _result(tmp_path.resolve())
    store = PendingHandoffStore()
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, source)
    object.__setattr__(source, "artifact_id", "producer-mutated")
    object.__setattr__(source, "canonical_root", "/private/producer-mutated")

    claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)

    assert claim is not None
    assert claim.value.artifact_id == "audio-cpp-supertonic-3"
    assert claim.value.canonical_root == str(tmp_path.resolve())
    object.__setattr__(claim.value, "variant", "consumer-mutated")
    assert store.release(claim) is True
    replay = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
    assert replay is not None
    assert replay.value.variant == "f16"


@pytest.mark.parametrize(
    "root",
    (
        r"\\?\C:\managed\model",
        r"\\.\C:\managed\model",
        "//?/C:/managed/model",
        "//./C:/managed/model",
    ),
)
def test_audio_cpp_result_rejects_windows_device_namespace_roots(root: str) -> None:
    with pytest.raises(ValueError, match="root"):
        AudioCppModelLibraryResult(
            token="request-token",
            draft_revision=1,
            artifact_id="audio-cpp-model",
            revision="a" * 40,
            variant="f16",
            canonical_root=root,
        )


@pytest.mark.parametrize(
    "root",
    (r"C:\managed\model", r"\\server\share\managed\model"),
)
def test_audio_cpp_result_accepts_canonical_windows_drive_and_unc_roots(
    root: str,
) -> None:
    result = AudioCppModelLibraryResult(
        token="request-token",
        draft_revision=1,
        artifact_id="audio-cpp-model",
        revision="a" * 40,
        variant="f16",
        canonical_root=root,
    )

    assert result.canonical_root == root


@pytest.mark.asyncio
async def test_install_owner_cancel_joins_actual_executor_before_settlement(
    tmp_path: Path,
) -> None:
    owner = AudioCppModelInstallOwner()
    thread_started = threading.Event()
    thread_finished = threading.Event()
    settlements: list[
        tuple[AudioCppModelLibraryResult | None, BaseException | None, bool]
    ] = []

    async def runner(cancel_event: threading.Event):
        def blocking_work() -> AudioCppModelLibraryResult:
            thread_started.set()
            assert cancel_event.wait(2)
            thread_finished.set()
            return _result(tmp_path.resolve())

        return await asyncio.to_thread(blocking_work)

    operation = owner.start(
        runner,
        lambda result, error, cancelled: settlements.append((result, error, cancelled)),
    )
    assert await asyncio.to_thread(thread_started.wait, 2)

    owner.request_cancel(operation)
    await owner.wait(operation)

    assert thread_finished.is_set()
    assert settlements == [(None, None, True)]
    assert owner.active_count == 0


@pytest.mark.asyncio
async def test_install_owner_shutdown_drains_and_seals_all_work(tmp_path: Path) -> None:
    owner = AudioCppModelInstallOwner()
    started = threading.Event()
    finished = threading.Event()

    async def runner(cancel_event: threading.Event):
        def blocking_work() -> AudioCppModelLibraryResult:
            started.set()
            assert cancel_event.wait(2)
            finished.set()
            return _result(tmp_path.resolve())

        return await asyncio.to_thread(blocking_work)

    owner.start(runner, lambda *_args: None)
    assert await asyncio.to_thread(started.wait, 2)

    await owner.shutdown()

    assert finished.is_set()
    assert owner.active_count == 0
    with pytest.raises(RuntimeError, match="shut down"):
        owner.start(runner, lambda *_args: None)


async def _wait_for(condition, pilot, *, attempts: int = 160) -> bool:
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause()
    return condition()


@pytest.mark.asyncio
async def test_mounted_audio_cpp_consent_provision_recompose_and_detached_return(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from textual.widgets import Button

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_model_curated_view import _descriptor, _registry_with
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef, ModelArtifactService
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    reference = ArtifactRef("audio-cpp-mounted", "a" * 40, "f16")
    payload = b"mounted-audio-model"
    descriptor = _descriptor(reference, payload, consumer="audio_cpp")
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(payload)
    report = PreflightReport(
        root=reference,
        closure_fingerprint=hashlib.sha256(b"mounted-plan").hexdigest(),
        entries=(),
        download_bytes=len(payload),
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=tmp_path / "store",
        free_bytes=10_000_000,
        required_bytes=len(payload),
        sufficient_space=True,
        gating_errors=(),
    )
    provision_started = asyncio.Event()
    release_provision = asyncio.Event()
    provision_calls: list[bool] = []

    class _FixtureAcquisition:
        def __init__(self, core) -> None:
            self.core = core

        async def preflight(self, *_args, **_kwargs):
            return report

        async def provision(self, root, _consent, _registry, **kwargs):
            provision_calls.append(kwargs["activate"])
            provision_started.set()
            await release_provision.wait()
            self.core.install(descriptor, source)
            return root

    monkeypatch.setattr(
        "tldw_chatbook.Model_Artifacts.acquisition.ArtifactAcquisitionService",
        _FixtureAcquisition,
    )
    monkeypatch.setattr(CuratedView, "_service_for_worker", lambda _self: service)
    monkeypatch.setattr(CuratedView, "_registry_for_worker", lambda _self: registry)
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if section == "splash_screen" and key == "enabled"
            else real_get_cli_setting(section, key, default)
        ),
    )

    app = _build_test_app()
    request = AudioCppModelLibraryRequest("mounted-request", 9)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    preference_service = MagicMock()
    app._ensure_parakeet_source_service = MagicMock(return_value=preference_service)
    app.start_server = MagicMock()
    app.set_default_model = MagicMock()

    async with app.run_test(size=(120, 44)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        screen.notify = MagicMock()
        assert await _wait_for(lambda: bool(screen.query(CuratedView)), pilot)
        view = screen.query_one(CuratedView)
        assert view._consumer_filter == "audio_cpp"
        assert view._allow_installed_return is True
        assert await _wait_for(lambda: view._loaded, pilot)
        install = screen.query_one(".curated-install", Button)
        install.press()
        assert await _wait_for(
            lambda: bool(app.screen.query("#model-install-confirm")), pilot
        )
        app.screen.query_one("#model-install-confirm", Button).press()
        assert await _wait_for(provision_started.is_set, pilot)

        screen.refresh(recompose=True)
        await pilot.pause()
        await pilot.pause()
        release_provision.set()
        assert await _wait_for(
            lambda: app.audio_cpp_model_install_owner.active_count == 0, pilot
        )
        returned = app.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )
        fresh_view = screen.query_one(CuratedView)
        assert await _wait_for(lambda: fresh_view._loaded, pilot)
        installed = screen.query_one(".curated-install", Button)
        assert str(installed.label) == "Installed"
        assert installed.disabled is True
        installed.press()
        await pilot.pause()
        assert app.audio_cpp_model_install_owner.active_count == 0

    assert returned is not None
    assert returned.value.token == request.token
    assert returned.value.draft_revision == request.draft_revision
    assert returned.value.canonical_root == str(
        service.artifact_path(reference).resolve()
    )
    assert provision_calls == [False]
    screen.notify.assert_called_once_with(
        "Installed — ready for review", severity="information"
    )
    preference_service.prefer_managed.assert_not_called()
    app.start_server.assert_not_called()
    app.set_default_model.assert_not_called()


@pytest.mark.asyncio
async def test_real_worker_cancel_on_screen_unmount_drains_before_request_release(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from textual.screen import Screen

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen

    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if section == "splash_screen" and key == "enabled"
            else real_get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    request = AudioCppModelLibraryRequest("cancel-request", 3)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    reference = ArtifactRef("audio-cpp-cancel", "b" * 40, "f16")
    executor_started = threading.Event()
    executor_finished = threading.Event()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        screen._model_install_kind = "curated"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = MagicMock()
        screen._model_install_registry.descriptor.return_value.consumer = "audio_cpp"
        screen._model_install_sources = {}
        screen._model_install_pending_report = MagicMock(root=reference)

        async def provision(_report, cancel_event=None):
            assert cancel_event is not None

            def executor_work() -> ArtifactRef:
                executor_started.set()
                assert cancel_event.wait(3)
                executor_finished.set()
                return reference

            return await asyncio.to_thread(executor_work)

        screen._provision_curated = provision
        screen._audio_cpp_installed_result = MagicMock(
            return_value=_result(tmp_path.resolve())
        )
        screen._start_audio_cpp_provision()
        worker = screen._model_install_worker
        assert worker is not None
        assert await asyncio.to_thread(executor_started.wait, 2)

        await app.switch_screen(Screen())
        assert await _wait_for(executor_finished.is_set, pilot)
        assert await _wait_for(
            lambda: app.audio_cpp_model_install_owner.active_count == 0, pilot
        )
        assert worker.is_finished
        assert (
            app.pending_handoffs.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
            is None
        )
        replay = app.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )

    assert replay is not None
    assert replay.value == request
    assert executor_finished.is_set()


@pytest.mark.asyncio
async def test_mounted_already_installed_audio_cpp_returns_exact_leased_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from textual.widgets import Button

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_model_curated_view import _descriptor, _registry_with
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef, ModelArtifactService
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    reference = ArtifactRef("audio-cpp-installed", "c" * 40, "f16")
    payload = b"already-installed-audio"
    descriptor = _descriptor(reference, payload, consumer="audio_cpp")
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(payload)
    service.install(descriptor, source)
    monkeypatch.setattr(CuratedView, "_service_for_worker", lambda _self: service)
    monkeypatch.setattr(CuratedView, "_registry_for_worker", lambda _self: registry)
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if section == "splash_screen" and key == "enabled"
            else real_get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    request = AudioCppModelLibraryRequest("installed-request", 11)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        screen.notify = MagicMock()
        assert await _wait_for(lambda: bool(screen.query(CuratedView)), pilot)
        view = screen.query_one(CuratedView)
        assert await _wait_for(lambda: view._loaded, pilot)
        button = screen.query_one(".curated-install", Button)
        assert str(button.label) == "Use installed package"
        button.press()
        assert await _wait_for(
            lambda: app.pending_handoffs.has_pending(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            ),
            pilot,
        )
        returned = app.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )

    assert returned is not None
    assert returned.value.token == request.token
    assert returned.value.canonical_root == str(
        service.artifact_path(reference).resolve()
    )
    screen.notify.assert_called_once_with(
        "Installed — ready for review", severity="information"
    )


@pytest.mark.asyncio
async def test_real_app_shutdown_drains_audio_cpp_owner_executor(
    monkeypatch,
) -> None:
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if section == "splash_screen" and key == "enabled"
            else real_get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    started = threading.Event()
    finished = threading.Event()

    async def runner(cancel_event: threading.Event):
        def executor_work() -> None:
            started.set()
            assert cancel_event.wait(3)
            finished.set()

        await asyncio.to_thread(executor_work)
        return None

    async with app.run_test() as _pilot:
        app.audio_cpp_model_install_owner.start(runner, lambda *_args: None)
        assert await asyncio.to_thread(started.wait, 2)

    assert finished.is_set()
    assert app.audio_cpp_model_install_owner.active_count == 0


@pytest.mark.asyncio
async def test_mounted_unmount_during_blocked_audio_preflight_drains_once(
    monkeypatch,
) -> None:
    from textual.screen import Screen

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_model_curated_view import _descriptor, _registry_with
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if section == "splash_screen" and key == "enabled"
            else real_get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    request = AudioCppModelLibraryRequest("preflight-cancel", 12)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    reference = ArtifactRef("audio-cpp-preflight", "d" * 40, "f16")
    registry = _registry_with(_descriptor(reference, consumer="audio_cpp"))
    started = threading.Event()
    release = threading.Event()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        screen._model_install_kind = "curated"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = registry
        screen._model_install_sources = {}
        screen._provision_curated = AsyncMock()
        screen.notify = MagicMock()

        async def blocked_preflight(_reference):
            def block():
                started.set()
                assert release.wait(3)
                return MagicMock(root=reference)

            return await asyncio.to_thread(block)

        screen._preflight_curated = blocked_preflight
        screen._start_audio_cpp_preflight()
        worker = screen._model_install_worker
        assert worker is not None
        assert await asyncio.to_thread(started.wait, 2)
        assert app.audio_cpp_model_install_owner.active_count == 1

        await app.switch_screen(Screen())
        release.set()
        assert await _wait_for(
            lambda: app.audio_cpp_model_install_owner.active_count == 0, pilot
        )
        replay = app.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )

    assert worker.is_finished
    assert screen._model_install_worker is None
    assert replay is not None and replay.value == request
    screen._provision_curated.assert_not_called()
    screen.notify.assert_not_called()
    assert (
        app.pending_handoffs.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
        is None
    )


@pytest.mark.asyncio
async def test_mounted_unmount_with_audio_consent_pending_invalidates_generation(
    monkeypatch,
) -> None:
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_model_curated_view import _descriptor, _registry_with
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting

    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if section == "splash_screen" and key == "enabled"
            else real_get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    request = AudioCppModelLibraryRequest("consent-cancel", 13)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    reference = ArtifactRef("audio-cpp-consent", "e" * 40, "f16")
    descriptor = _descriptor(reference, consumer="audio_cpp")
    registry = _registry_with(descriptor)
    report = PreflightReport(
        root=reference,
        closure_fingerprint=hashlib.sha256(b"consent-plan").hexdigest(),
        entries=(),
        download_bytes=0,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=Path("/managed/audio-cpp-consent"),
        free_bytes=1,
        required_bytes=0,
        sufficient_space=True,
        gating_errors=(),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        screen._model_install_kind = "curated"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = registry
        screen._model_install_sources = {}
        screen._preflight_curated = AsyncMock(return_value=report)
        screen._provision_curated = AsyncMock()
        screen.notify = MagicMock()
        screen._start_audio_cpp_preflight()
        assert await _wait_for(
            lambda: bool(app.screen.query("#model-install-confirm")), pilot
        )
        operation = screen._audio_cpp_model_install_operation
        assert operation is not None

        await screen.remove()
        assert await _wait_for(
            lambda: app.audio_cpp_model_install_owner.active_count == 0, pilot
        )
        replay = app.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )

    assert operation.task.done()
    assert screen._model_install_worker is None
    assert screen._audio_cpp_consent_modal is None
    assert screen._audio_cpp_consent_future is None
    assert replay is not None and replay.value == request
    screen._provision_curated.assert_not_called()
    screen.notify.assert_not_called()
    assert (
        app.pending_handoffs.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
        is None
    )
