"""Typed Settings-to-Model-Library handoff contracts for audio.cpp."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
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


@pytest.fixture(autouse=True)
async def _close_task6_test_app_resources(monkeypatch: pytest.MonkeyPatch):
    """Close inner TldwCli resources that DestinationHarness does not own."""

    from Tests.UI import app_factory, test_destination_shells

    built_apps: list[object] = []
    real_build = app_factory._build_test_app

    def tracked_build(*args: object, **kwargs: object):
        app = real_build(*args, **kwargs)
        built_apps.append(app)
        return app

    monkeypatch.setattr(app_factory, "_build_test_app", tracked_build)
    monkeypatch.setattr(test_destination_shells, "_build_test_app", tracked_build)
    try:
        yield
    finally:
        for app in reversed(built_apps):
            await app._shutdown_app_owned_lifecycles()
            closeables = [
                getattr(app, name, None)
                for name in (
                    "tts_service",
                    "_tts_profile_repository",
                    "local_library_collections_db",
                    "local_workspace_db",
                    "subscriptions_db",
                    "client_notifications_db",
                    "server_parity_state",
                    "event_state_repository",
                    "sync_state_repository",
                    "local_research_service",
                    "local_writing_service",
                    "scheduled_tasks_db",
                )
            ]
            orchestrator = getattr(app, "evaluation_orchestrator", None)
            closeables.append(getattr(orchestrator, "db", None))
            closed: set[int] = set()
            for resource in closeables:
                if resource is None or id(resource) in closed:
                    continue
                close = getattr(resource, "close", None)
                if not callable(close):
                    continue
                closed.add(id(resource))
                result = close()
                if inspect.isawaitable(result):
                    await result


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
    "hostile",
    (
        b"private-bytes",
        bytearray(b"private-bytes"),
        memoryview(b"private-bytes"),
        ValueError("private-exception"),
        KeyboardInterrupt("private-base-exception"),
        lambda: None,
        object(),
        {1: "private-non-string-key"},
        {"access_token": "private-token-canary"},
        {"nested": {"token": "private-token-canary"}},
        {"nested": {"auth_token": "private-auth-canary"}},
        {"nested": {"client_secret": "private-secret-canary"}},
        {"nested": {"credentials": "private-credential-canary"}},
        {"nested": {"passphrase": "private-passphrase-canary"}},
        {"nested": {"api_key": "private-key-canary"}},
        {
            "nested": {
                "nested": {
                    "nested": {
                        "nested": {
                            "nested": {
                                "nested": {"nested": {"nested": {"nested": "too-deep"}}}
                            }
                        }
                    }
                }
            }
        },
    ),
)
def test_panel_snapshot_rejects_private_non_data_leaves(hostile: object) -> None:
    """Process-local state accepts bounded data, never executable/private graphs."""

    from dataclasses import replace

    from Tests.UI.test_settings_speech_tts_panel import _audio_cpp_state
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSPanelDraftSnapshot,
        _RealtimeSettingsDraft,
    )

    state = _audio_cpp_state(saved_provider=True)
    state.providers["openai"]["base_url"] = hostile
    realtime = _RealtimeSettingsDraft(
        False,
        "openai",
        "gpt-realtime",
        "",
        "30",
        "auto",
        "semantic_vad",
        "0.5",
        "500",
    )

    with pytest.raises((TypeError, ValueError)):
        SpeechTTSPanelDraftSnapshot(
            state=state,
            original_state=_audio_cpp_state(saved_provider=True),
            realtime_draft=realtime,
            realtime_original=replace(realtime),
            configure_provider="audio_cpp",
            draft_revision=1,
        )


def test_panel_snapshot_roundtrips_invalid_url_but_strips_credential_metadata() -> None:
    """Editable invalid text survives while credential provenance does not."""

    from dataclasses import replace

    from Tests.UI.test_settings_speech_tts_panel import _audio_cpp_state
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSPanelDraftSnapshot,
        _RealtimeSettingsDraft,
    )

    state = _audio_cpp_state(saved_provider=True)
    state.providers["audio_cpp"]["base_url"] = "ftp://invalid.example"
    realtime = _RealtimeSettingsDraft(
        False,
        "openai",
        "gpt-realtime",
        "",
        "30",
        "auto",
        "semantic_vad",
        "0.5",
        "500",
    )
    snapshot = SpeechTTSPanelDraftSnapshot(
        state=state,
        original_state=_audio_cpp_state(saved_provider=True),
        realtime_draft=realtime,
        realtime_original=replace(realtime),
        configure_provider="audio_cpp",
        draft_revision=1,
    )

    assert snapshot.state.providers["audio_cpp"]["base_url"] == (
        "ftp://invalid.example"
    )
    assert snapshot.state.credentials == {}
    assert "invalid.example" not in repr(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "widget_type", "value"),
    (
        ("#settings-speech-default-provider", "select", "audio_cpp"),
        ("#settings-speech-default-profile", "select", "profile-2"),
        ("#settings-speech-model-policy", "select", "exact"),
        ("#settings-speech-voice-policy", "select", "exact"),
        ("#settings-speech-output-format", "select", "wav"),
        ("#settings-speech-speed", "input", "1.25"),
        ("#settings-speech-audio_cpp-base-url", "input", "http://127.0.0.1:18091"),
        ("#settings-speech-audio_cpp-mode", "select", "managed"),
        ("#settings-speech-realtime-model", "input", "realtime-one-action"),
    ),
)
async def test_one_mounted_draft_action_advances_revision_exactly_once(
    selector: str,
    widget_type: str,
    value: str,
) -> None:
    """Each independent widget action owns one semantic draft revision."""

    from textual.widgets import Input, Select

    from Tests.UI.test_settings_speech_tts_panel import (
        _StyledPanelHarness,
        _audio_cpp_state,
    )

    state = _audio_cpp_state(saved_provider=False)
    state.defaults.provider_id = "openai"
    state.defaults.response_format = "mp3"
    app = _StyledPanelHarness(
        state=state,
        profiles=(("Profile one", "profile-1"), ("Profile two", "profile-2")),
    )
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        await pilot.pause()
        before = panel.draft_snapshot().draft_revision
        widget = panel.query_one(selector, Select if widget_type == "select" else Input)
        widget.value = value
        await pilot.pause()
        after = panel.draft_snapshot().draft_revision

        assert after - before == 1
        assert panel.draft_snapshot().draft_revision == after


@pytest.mark.asyncio
async def test_value_identical_widget_echo_does_not_advance_revision() -> None:
    """A framework echo of an unchanged value is not a semantic action."""

    from textual.widgets import Input

    from Tests.UI.test_settings_speech_tts_panel import (
        _StyledPanelHarness,
        _audio_cpp_state,
    )

    app = _StyledPanelHarness(state=_audio_cpp_state(saved_provider=False))
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        await pilot.pause()
        before = panel.draft_snapshot().draft_revision
        widget = panel.query_one("#settings-speech-speed", Input)
        widget.value = widget.value
        await pilot.pause()

        assert panel.draft_snapshot().draft_revision == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "value"),
    (
        ("#settings-speech-model-value", "model-b"),
        ("#settings-speech-voice-value", "voice-b"),
    ),
)
async def test_exact_model_or_voice_action_advances_revision_once(
    selector: str,
    value: str,
) -> None:
    """An exact catalog choice is one action even when it recomposes dependents."""

    from textual.widgets import Select

    from Tests.UI.test_settings_speech_tts_panel import (
        _StyledPanelHarness,
        _audio_cpp_observation,
        _audio_cpp_state,
    )

    app = _StyledPanelHarness(
        state=_audio_cpp_state(
            model_mode="exact",
            model_id="model-a",
            voice_mode="exact",
            voice_id="voice-a",
        ),
        observation=_audio_cpp_observation(),
    )
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        await pilot.pause()
        before = panel.draft_snapshot().draft_revision
        panel.query_one(selector, Select).value = value
        await pilot.pause()

        assert panel.draft_snapshot().draft_revision == before + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("revert", "restore_defaults"))
async def test_reset_action_advances_revision_exactly_once(action: str) -> None:
    """Each whole-draft reset is one semantic mutation transaction."""

    from textual.widgets import Button, Input

    from Tests.UI.test_settings_speech_tts_panel import (
        _StyledPanelHarness,
        _audio_cpp_state,
    )

    state = _audio_cpp_state(saved_provider=False)
    state.defaults.speed = 1.5 if action == "restore_defaults" else 1.0
    app = _StyledPanelHarness(state=state)
    async with app.run_test(size=(150, 55)) as pilot:
        panel = app.query_one("#panel")
        await pilot.pause()
        if action == "revert":
            panel.query_one("#settings-speech-speed", Input).value = "1.25"
            await pilot.pause()
        before = panel.draft_snapshot().draft_revision
        panel.query_one(
            "#settings-speech-revert"
            if action == "revert"
            else "#settings-speech-restore-defaults",
            Button,
        ).press()
        await pilot.pause()

        assert panel.draft_snapshot().draft_revision == before + 1


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
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        PreflightReport,
    )
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactFile,
        ArtifactRef,
        ModelArtifactService,
    )
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    reference = ArtifactRef("audio-cpp-mounted", "a" * 40, "f16")
    payload = b"mounted-audio-model"
    companion = b"phoneme-companion"
    descriptor = replace(
        _descriptor(reference, payload, consumer="audio_cpp"),
        files=(
            ArtifactFile(
                "model.bin", len(payload), hashlib.sha256(payload).hexdigest()
            ),
            ArtifactFile(
                "companions/phonemes.json",
                len(companion),
                hashlib.sha256(companion).hexdigest(),
            ),
        ),
        expected_installed_bytes=len(payload) + len(companion),
    )
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(payload)
    (source / "companions").mkdir()
    (source / "companions/phonemes.json").write_bytes(companion)
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
    provision_started = threading.Event()
    release_provision = threading.Event()
    provision_calls: list[bool] = []

    class _FixtureAcquisition:
        def __init__(self, core) -> None:
            self.core = core

        async def preflight(self, *_args, **_kwargs):
            return report

        async def provision(self, root, _consent, _registry, **kwargs):
            provision_calls.append(kwargs["activate"])
            await asyncio.sleep(0.1)
            kwargs["progress"](
                AcquisitionProgress("fetch", root, "model.bin", 1, len(payload))
            )
            provision_started.set()
            await asyncio.to_thread(release_provision.wait)
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
        event_loop_thread = threading.get_ident()
        delivery_threads: list[int] = []
        deliver = screen._deliver_curated

        def record_delivery(message):
            delivery_threads.append(threading.get_ident())
            deliver(message)

        screen._deliver_curated = record_delivery
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
        modal = app.screen
        modal_text = "\n".join(
            str(item.renderable) for item in modal.query(".model-plan-panel")
        )
        for artifact_file in descriptor.files:
            assert f"Path: {artifact_file.path}" in modal_text
            assert f"Bytes: {artifact_file.size_bytes}" in modal_text
            assert f"SHA-256: {artifact_file.sha256}" in modal_text
            assert (
                f"Pinned source URL: https://example.test/{artifact_file.path}"
                in modal_text
            )
        assert "Authorization" not in modal_text
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
    assert delivery_threads and set(delivery_threads) == {event_loop_thread}
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
        screen._start_audio_cpp_operation(installed=False)
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
    request = AudioCppModelLibraryRequest("preflight-cancel", 12)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    reference = ArtifactRef("audio-cpp-preflight", "d" * 40, "f16")
    registry = _registry_with(_descriptor(reference, consumer="audio_cpp"))
    started = threading.Event()
    release = threading.Event()
    report = PreflightReport(
        root=reference,
        closure_fingerprint=hashlib.sha256(b"blocked-preflight").hexdigest(),
        entries=(),
        download_bytes=0,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=Path("/managed/audio-cpp-preflight"),
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
        screen._provision_curated = AsyncMock()
        screen.notify = MagicMock()

        async def blocked_preflight(_reference):
            started.set()
            assert release.wait(3)
            return report

        screen._preflight_curated = blocked_preflight
        heartbeat = threading.Event()
        heartbeat_seen_before_release: list[bool] = []

        async def beat() -> None:
            await asyncio.sleep(0.01)
            heartbeat.set()

        beat_task = asyncio.create_task(beat())
        screen._start_audio_cpp_preflight()
        worker = screen._model_install_worker
        assert worker is not None
        assert await asyncio.to_thread(started.wait, 2)
        heartbeat_seen_before_release.append(await asyncio.to_thread(heartbeat.wait, 1))
        await beat_task
        assert started.is_set()
        assert not release.is_set()
        assert app.audio_cpp_model_install_owner.active_count == 1
        assert heartbeat_seen_before_release == [True]

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
        screen._model_install_sources = {reference: registry.sources(reference)}
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


@pytest.mark.asyncio
async def test_rapid_away_back_reclaims_request_after_old_operation_drains(
    monkeypatch,
) -> None:
    from textual.screen import Screen
    from textual.widgets import Button

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_model_curated_view import _descriptor, _registry_with
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    app = _build_test_app()
    request = AudioCppModelLibraryRequest("rapid-return", 14)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    reference = ArtifactRef("audio-cpp-rapid", "f" * 40, "f16")
    registry = _registry_with(_descriptor(reference, consumer="audio_cpp"))
    report = PreflightReport(
        root=reference,
        closure_fingerprint=hashlib.sha256(b"rapid").hexdigest(),
        entries=(),
        download_bytes=0,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=Path("/managed/audio-cpp-rapid"),
        free_bytes=1,
        required_bytes=0,
        sufficient_space=True,
        gating_errors=(),
    )
    started = threading.Event()
    release = threading.Event()

    async with app.run_test(size=(120, 40)) as pilot:
        old = LLMScreen(app)
        await app.push_screen(old)
        old._model_install_kind = "curated"
        old._model_install_reference = reference
        old._model_install_service = MagicMock()
        old._model_install_registry = registry
        old._model_install_sources = {reference: registry.sources(reference)}
        old._model_install_pending_report = report

        async def blocked_provision(_report, cancel_event=None):
            started.set()
            await asyncio.to_thread(release.wait)
            raise asyncio.CancelledError

        old._provision_curated = blocked_provision
        old._start_audio_cpp_operation(installed=False)
        assert await asyncio.to_thread(started.wait, 2)

        await app.switch_screen(Screen())
        replacement = LLMScreen(app)
        await app.push_screen(replacement)
        assert replacement._audio_cpp_model_request_claim is None
        assert await _wait_for(lambda: replacement.llm_window is not None, pilot)
        replacement.llm_window.active_view = "remote"
        release.set()
        assert await _wait_for(
            lambda: app.audio_cpp_model_install_owner.active_count == 0, pilot
        )
        assert await _wait_for(
            lambda: replacement._audio_cpp_model_request_claim is not None, pilot
        )
        assert await _wait_for(
            lambda: (
                replacement.llm_window is not None
                and replacement.llm_window.active_view == "curated"
            ),
            pilot,
        )
        replacement_view = replacement.query_one(CuratedView)
        assert await _wait_for(lambda: replacement_view._loaded, pilot)
        assert replacement_view._consumer_filter == "audio_cpp"
        assert replacement_view.display
        assert any(
            not button.disabled
            for button in replacement_view.query(".curated-install").results(Button)
        )
        reclaimed = replacement._audio_cpp_model_request_claim

    assert reclaimed is not None and reclaimed.value == request
    replay = app.pending_handoffs.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert replay is not None and replay.value == request
    assert (
        app.pending_handoffs.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
        is None
    )


@pytest.mark.asyncio
async def test_audio_cpp_presentation_reveals_slow_load_once_and_keeps_error_retry(
    monkeypatch,
) -> None:
    from textual.widgets import Button, Static

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _self: asyncio.sleep(0, result=False),
    )
    attempts: list[CuratedView] = []

    def remain_loading(view, *, force=False):
        if view._loading:
            return
        attempts.append(view)
        view._loading = True
        view.refresh(recompose=True)

    monkeypatch.setattr(CuratedView, "ensure_loaded", remain_loading)
    app = _build_test_app()
    request = AudioCppModelLibraryRequest("slow-presentation", 15)
    app.pending_handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)

    async with app.run_test(size=(120, 40)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        assert await _wait_for(
            lambda: (
                screen.llm_window is not None
                and screen.llm_window.active_view == "curated"
            ),
            pilot,
        )
        view = screen.query_one(CuratedView)
        assert attempts == [view]
        assert view._consumer_filter == "audio_cpp"
        assert "Loading curated models…" in "\n".join(
            str(item.renderable) for item in view.query(Static)
        )
        await asyncio.sleep(2.1)
        assert attempts == [view]
        assert screen.llm_window.active_view == "curated"

        view._apply_rows((), "The curated model catalog could not be loaded.")
        await pilot.pause()
        error_text = "\n".join(str(item.renderable) for item in view.query(Static))
        assert "The curated model catalog could not be loaded." in error_text
        assert view.query_one("#curated-models-refresh", Button)
        assert screen._audio_cpp_model_request_claim is not None
        assert screen._audio_cpp_model_request_claim.value == request


@pytest.mark.asyncio
async def test_mounted_settings_snapshot_preserves_complete_speech_tts_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Settings save/restore retains global, provider, and Realtime drafts."""

    from textual.widgets import Button, Input, Select, Switch

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    async def open_panel(host, pilot) -> tuple[object, SpeechTTSSettingsPanel]:
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen,
            pilot,
            "#settings-speech-tts-panel",
            timeout=8.0,
        )
        return screen, screen.query_one(
            "#settings-speech-tts-panel", SpeechTTSSettingsPanel
        )

    app_instance = _build_test_app()
    original_host = DestinationHarness(app_instance, "settings")
    async with original_host.run_test(size=(190, 55)) as pilot:
        screen, panel = await open_panel(original_host, pilot)
        screen.query_one(
            "#settings-speech-configure-provider", Select
        ).value = "audio_cpp"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio_cpp-base-url", Input
        ).value = "http://127.0.0.1:18081"
        screen.query_one("#settings-speech-speed", Input).value = "1.25"
        screen.query_one("#settings-speech-realtime-enabled", Switch).value = True
        screen.query_one(
            "#settings-speech-realtime-model", Input
        ).value = "gpt-realtime-draft"
        screen.query_one("#settings-speech-realtime-voice", Input).value = "cedar"
        await pilot.pause()

        before = panel.draft_snapshot()
        saved = screen.save_state()

    assert saved["speech_tts_panel_draft"] == before
    assert "18081" not in repr(saved["speech_tts_panel_draft"])

    restored_host = DestinationHarness(
        _build_test_app(),
        "settings",
        restored_state=saved,
    )
    async with restored_host.run_test(size=(190, 55)) as pilot:
        restored_screen, restored_panel = await open_panel(restored_host, pilot)
        assert restored_screen.active_category == SettingsCategoryId.SPEECH_TTS.value
        assert restored_panel.draft_snapshot() == before
        assert (
            restored_screen.query_one(
                "#settings-speech-audio_cpp-base-url", Input
            ).value
            == "http://127.0.0.1:18081"
        )
        assert restored_screen.query_one("#settings-speech-speed", Input).value == (
            "1.25"
        )
        assert (
            restored_screen.query_one("#settings-speech-realtime-enabled", Switch).value
            is True
        )
        assert (
            restored_screen.query_one("#settings-speech-realtime-model", Input).value
            == "gpt-realtime-draft"
        )
        assert (
            restored_screen.query_one("#settings-speech-realtime-voice", Input).value
            == "cedar"
        )


@pytest.mark.asyncio
async def test_mounted_settings_stages_exact_request_after_collecting_widgets() -> None:
    """The explicit Library action captures the post-collection draft revision."""

    from textual.widgets import Button, Input, Select

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    seen_routes: list[str] = []
    host = DestinationHarness(app_instance, "settings", seen_routes=seen_routes)
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        screen.query_one(
            "#settings-speech-configure-provider", Select
        ).value = "audio_cpp"
        await pilot.pause()
        screen.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "guided"
        await pilot.pause()
        panel = screen.query_one(SpeechTTSSettingsPanel)
        screen.query_one(
            "#settings-speech-audio_cpp-base-url", Input
        ).value = "http://127.0.0.1:18082"
        screen.query_one(
            "#settings-speech-audio-cpp-open-model-library", Button
        ).press()
        assert await _wait_for(lambda: seen_routes == ["llm"], pilot)

        request_claim = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        assert request_claim is not None
        assert (
            request_claim.value.draft_revision == panel.draft_snapshot().draft_revision
        )
        assert request_claim.value.draft_revision > 0
        assert await screen.flush_pending_work() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ("stage_raise", "stage_interrupt", "post_false", "post_raise", "post_interrupt"),
)
async def test_model_library_route_token_is_cleared_when_dispatch_fails(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed request/route transaction cannot leave bypass authority behind."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = screen.query_one(SpeechTTSSettingsPanel)
        snapshot = panel.draft_snapshot()
        foreign = AudioCppModelLibraryRequest("foreign-request", 91)
        successor = AudioCppModelLibraryRequest("successor-request", 92)
        if failure.startswith("stage_"):
            app_instance.pending_handoffs.stage(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
                foreign,
            )

        with monkeypatch.context() as scoped:
            if failure.startswith("stage_"):

                def fail_stage(*_args: object) -> int:
                    if failure == "stage_interrupt":
                        raise KeyboardInterrupt("private-stage-interrupt")
                    raise RuntimeError("private-stage-canary")

                scoped.setattr(app_instance.pending_handoffs, "stage", fail_stage)
            else:

                def fail_post(_message: object) -> bool:
                    app_instance.pending_handoffs.stage(
                        HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
                        successor,
                    )
                    if failure == "post_interrupt":
                        raise KeyboardInterrupt("private-post-interrupt")
                    if failure == "post_raise":
                        raise RuntimeError("private-post-canary")
                    return False

                scoped.setattr(screen, "post_message", fail_post)

            if failure.endswith("interrupt"):
                with pytest.raises(KeyboardInterrupt):
                    screen.stage_audio_cpp_model_library_request(snapshot)
            else:
                assert screen.stage_audio_cpp_model_library_request(snapshot) is False
        assert screen._speech_tts_model_library_route_token is None
        assert not hasattr(
            app_instance,
            "_audio_cpp_settings_model_library_request",
        )
        retained = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        assert retained is not None
        assert retained.value == (
            foreign if failure.startswith("stage_") else successor
        )


@pytest.mark.asyncio
async def test_failed_route_cleanup_retries_after_foreign_request_claim_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-flight foreign request cannot permanently block exact cleanup."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        snapshot = screen.query_one(SpeechTTSSettingsPanel).draft_snapshot()
        foreign = AudioCppModelLibraryRequest("foreign-in-flight", 41)
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
            foreign,
        )
        foreign_claim = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        assert foreign_claim is not None
        with monkeypatch.context() as scoped:
            scoped.setattr(screen, "post_message", lambda _message: False)
            assert screen.stage_audio_cpp_model_library_request(snapshot) is False
        assert screen._audio_cpp_staged_request_cleanup is not None
        assert app_instance.pending_handoffs.acknowledge(foreign_claim)

        screen._retry_audio_cpp_staged_request_cleanup()

        assert screen._audio_cpp_staged_request_cleanup is None
        assert (
            app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
            )
            is None
        )


@pytest.mark.asyncio
async def test_failed_route_cleanup_preserves_value_identical_newer_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value-identical successor is foreign when its store revision is newer."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        snapshot = screen.query_one(SpeechTTSSettingsPanel).draft_snapshot()

        def fail_after_successor(_message: object) -> bool:
            staged = getattr(
                app_instance,
                "_audio_cpp_settings_model_library_request",
            )
            app_instance.pending_handoffs.stage(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
                staged,
            )
            return False

        with monkeypatch.context() as scoped:
            scoped.setattr(screen, "post_message", fail_after_successor)
            assert screen.stage_audio_cpp_model_library_request(snapshot) is False
        successor = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )

        assert successor is not None
        assert successor.value.draft_revision == snapshot.draft_revision


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated_first", (True, False))
async def test_model_library_leave_bypass_is_fifo_route_exclusive(
    unrelated_first: bool,
) -> None:
    """Only the exact queued curated audio route bypasses dirty confirmation."""

    from textual.widgets import Button, Select

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        screen.query_one(
            "#settings-speech-configure-provider", Select
        ).value = "audio_cpp"
        await pilot.pause()
        screen.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "guided"
        await pilot.pause()
        panel = screen.query_one(SpeechTTSSettingsPanel)
        panel.confirm_leave = AsyncMock(return_value=False)
        if unrelated_first:
            screen.post_message(NavigateToScreen("home", {"source": "competing"}))
            await pilot.pause()
        screen.query_one(
            "#settings-speech-audio-cpp-open-model-library", Button
        ).press()
        await pilot.pause()
        if not unrelated_first:
            screen.post_message(NavigateToScreen("home", {"source": "competing"}))
            await pilot.pause()

        outcomes = (
            await screen.flush_pending_work(),
            await screen.flush_pending_work(),
        )

        assert outcomes == ((False, True) if unrelated_first else (True, False))
        assert panel.confirm_leave.await_count == 1


@pytest.mark.asyncio
async def test_mounted_settings_reviews_and_merges_return_under_exact_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restored real Settings draft changes only by one reviewed package."""

    import copy
    import struct
    from types import SimpleNamespace

    from textual.screen import Screen
    from textual.widgets import Button, Input, Select, Switch

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
    )
    from tldw_chatbook.UI.Screens import settings_screen as settings_module
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Settings_Widgets import (
        speech_tts_settings_panel as panel_module,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    root = (tmp_path / "managed-supertonic").resolve()
    root.mkdir()
    (root / "supertonic-3-orig.gguf").write_bytes(b"GGUF" + struct.pack("<I", 3))
    reference = ArtifactRef(
        "audio-cpp-supertonic-3-orig",
        AUDIO_CPP_ARTIFACT_COMMIT,
        "orig",
    )
    lease_active = False
    lease_released = False

    class Lease:
        handle = SimpleNamespace(
            root=reference,
            closure=(reference,),
            paths=((reference, root),),
        )

        def __enter__(self):
            nonlocal lease_active
            lease_active = True
            return self

        def __exit__(self, *_args):
            nonlocal lease_active, lease_released
            lease_active = False
            lease_released = True

    service = SimpleNamespace(acquire_installed_root=lambda value: Lease())
    monkeypatch.setattr(settings_module, "managed_service", lambda: service)
    real_scan = settings_module.scan_audio_cpp_package_root
    scan_calls: list[dict[str, object]] = []

    def counted_scan(path, **kwargs):
        assert lease_active
        scan_calls.append({"path": path, **kwargs})
        return real_scan(path, **kwargs)

    monkeypatch.setattr(settings_module, "scan_audio_cpp_package_root", counted_scan)
    real_merge = SpeechTTSSettingsPanel.merge_managed_audio_cpp_package

    def leased_merge(self, package, *, expected_revision):
        assert lease_active
        return real_merge(self, package, expected_revision=expected_revision)

    monkeypatch.setattr(
        SpeechTTSSettingsPanel,
        "merge_managed_audio_cpp_package",
        leased_merge,
    )
    save_config = MagicMock()
    monkeypatch.setattr(panel_module, "save_settings_to_cli_config", save_config)

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        screen.query_one(
            "#settings-speech-configure-provider", Select
        ).value = "audio_cpp"
        await pilot.pause()
        screen.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "guided"
        screen.query_one(
            "#settings-speech-audio_cpp-base-url", Input
        ).value = "http://127.0.0.1:18083"
        screen.query_one("#settings-speech-speed", Input).value = "1.33"
        screen.query_one("#settings-speech-realtime-enabled", Switch).value = True
        screen.query_one("#settings-speech-realtime-model", Input).value = "draft-model"
        screen.query_one("#settings-speech-realtime-voice", Input).value = "cedar"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio-cpp-open-model-library", Button
        ).press()
        await pilot.pause()
        request_claim = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        assert request_claim is not None
        request = request_claim.value
        assert app_instance.pending_handoffs.acknowledge(request_claim)
        before = screen.query_one(SpeechTTSSettingsPanel).draft_snapshot()
        saved = screen.save_state()
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            AudioCppModelLibraryResult(
                token=request.token,
                draft_revision=request.draft_revision,
                artifact_id=reference.artifact_id,
                revision=reference.revision,
                variant=reference.variant,
                canonical_root=str(root),
            ),
        )

        await host.switch_screen(Screen())
        replacement = SettingsScreen(app_instance)
        replacement.restore_state(saved)
        await host.switch_screen(replacement)
        await _wait_for_selector(
            replacement, pilot, "#settings-speech-tts-panel", timeout=8.0
        )

        def merged() -> bool:
            panel = replacement.query_one(SpeechTTSSettingsPanel)
            packages = panel._audio_cpp_guided_packages()
            return len(packages) == 1

        assert await _wait_for(merged, pilot)
        after = replacement.query_one(SpeechTTSSettingsPanel).draft_snapshot()

    expected_state = copy.deepcopy(before.state)
    expected_state.providers["audio_cpp"]["guided_packages"] = after.state.providers[
        "audio_cpp"
    ]["guided_packages"]
    expected_state.providers["audio_cpp"]["guided_default_model_id"] = (
        after.state.providers["audio_cpp"]["guided_default_model_id"]
    )
    assert after.state == expected_state
    assert after.original_state == before.original_state
    assert after.realtime_draft == before.realtime_draft
    assert after.realtime_original == before.realtime_original
    assert after.configure_provider == before.configure_provider
    assert after.draft_revision == before.draft_revision + 1
    assert len(scan_calls) == 1
    assert scan_calls[0]["expected_canonical_root"] == str(root)
    assert lease_released and not lease_active
    assert save_config.call_count == 0


@pytest.mark.asyncio
async def test_mounted_return_is_stale_after_edits_in_every_draft_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-Library edits make the exact result terminal without scanning."""

    from textual.screen import Screen
    from textual.widgets import Button, Input, Select, Switch

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.UI.Screens import settings_screen as settings_module
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    scanner = MagicMock(side_effect=AssertionError("stale return must not scan"))
    monkeypatch.setattr(settings_module, "scan_audio_cpp_package_root", scanner)
    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        screen.query_one(
            "#settings-speech-configure-provider", Select
        ).value = "audio_cpp"
        await pilot.pause()
        screen.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "guided"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio-cpp-open-model-library", Button
        ).press()
        await pilot.pause()
        request_claim = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        assert request_claim is not None
        request = request_claim.value
        assert app_instance.pending_handoffs.acknowledge(request_claim)
        saved = screen.save_state()

        await host.switch_screen(Screen())
        replacement = SettingsScreen(app_instance)
        replacement.restore_state(saved)
        await host.switch_screen(replacement)
        await _wait_for_selector(
            replacement, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = replacement.query_one(SpeechTTSSettingsPanel)
        replacement.query_one(
            "#settings-speech-audio_cpp-base-url", Input
        ).value = "http://127.0.0.1:18084"
        replacement.query_one("#settings-speech-speed", Input).value = "1.41"
        replacement.query_one("#settings-speech-realtime-enabled", Switch).value = True
        replacement.query_one(
            "#settings-speech-realtime-model", Input
        ).value = "edited-after-library"
        await pilot.pause()
        changed = panel.draft_snapshot()
        assert changed.draft_revision > request.draft_revision
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            AudioCppModelLibraryResult(
                token=request.token,
                draft_revision=request.draft_revision,
                artifact_id="audio-cpp-supertonic-3-orig",
                revision="a" * 40,
                variant="orig",
                canonical_root=str(tmp_path.resolve()),
            ),
        )
        replacement._consume_audio_cpp_model_library_result()
        await pilot.pause()

        assert panel.draft_snapshot() == changed
        assert panel.result_text == "Installed, not added to this changed draft"
        assert (
            app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )
            is None
        )
        assert scanner.call_count == 0


@pytest.mark.asyncio
async def test_foreign_result_preserves_expected_request_for_later_exact_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign token is terminal only for its result, not our request."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = screen.query_one(SpeechTTSSettingsPanel)
        before = panel.draft_snapshot()
        expected = AudioCppModelLibraryRequest("ours", before.draft_revision)
        setattr(app_instance, "_audio_cpp_settings_model_library_request", expected)
        foreign = AudioCppModelLibraryResult(
            token="foreign",
            draft_revision=expected.draft_revision,
            artifact_id="audio-cpp-supertonic-3-orig",
            revision="a" * 40,
            variant="orig",
            canonical_root=str(tmp_path.resolve()),
        )
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            foreign,
        )

        screen._consume_audio_cpp_model_library_result()

        assert (
            getattr(app_instance, "_audio_cpp_settings_model_library_request")
            is expected
        )
        assert panel.draft_snapshot() == before
        reviewed: list[AudioCppModelLibraryResult] = []

        def review_exact(claim, result, *_args):
            reviewed.append(result)
            assert app_instance.pending_handoffs.acknowledge(claim)
            screen._finish_audio_cpp_result_cleanup(claim)

        monkeypatch.setattr(
            screen,
            "_review_audio_cpp_model_library_result",
            review_exact,
        )
        exact = replace(foreign, token=expected.token)
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            exact,
        )
        screen._consume_audio_cpp_model_library_result()

        assert reviewed == [exact]


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_failure", ("false", "raise"))
@pytest.mark.parametrize("release_failure", ("false", "raise"))
async def test_foreign_result_cleanup_retries_without_touching_current_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ack_failure: str,
    release_failure: str,
) -> None:
    """Foreign ack/release failures retain release-only cleanup authority."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = screen.query_one(SpeechTTSSettingsPanel)
        before = panel.draft_snapshot()
        expected = AudioCppModelLibraryRequest("ours", before.draft_revision)
        setattr(app_instance, "_audio_cpp_settings_model_library_request", expected)
        foreign = AudioCppModelLibraryResult(
            token="foreign",
            draft_revision=expected.draft_revision,
            artifact_id="audio-cpp-supertonic-3-orig",
            revision="a" * 40,
            variant="orig",
            canonical_root=str(tmp_path.resolve()),
        )
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            foreign,
        )
        real_ack = app_instance.pending_handoffs.acknowledge

        def fail_ack(_claim: object) -> bool:
            cleanup = screen._audio_cpp_result_cleanup
            assert cleanup is not None and cleanup.release_only
            if ack_failure == "raise":
                raise RuntimeError("private-foreign-ack")
            return False

        def fail_release(_claim: object) -> bool:
            if release_failure == "raise":
                raise RuntimeError("private-foreign-release")
            return False

        with monkeypatch.context() as scoped:
            scoped.setattr(app_instance.pending_handoffs, "acknowledge", fail_ack)
            scoped.setattr(app_instance.pending_handoffs, "release", fail_release)
            screen._consume_audio_cpp_model_library_result()

        assert screen._audio_cpp_result_cleanup is not None
        assert (
            getattr(
                app_instance,
                "_audio_cpp_settings_model_library_request",
            )
            is expected
        )
        assert panel.draft_snapshot() == before

        screen._retry_audio_cpp_result_cleanup()
        assert screen._audio_cpp_result_cleanup is None
        screen._consume_audio_cpp_model_library_result()
        assert screen._audio_cpp_result_cleanup is None
        assert (
            getattr(
                app_instance,
                "_audio_cpp_settings_model_library_request",
            )
            is expected
        )

        reviewed: list[AudioCppModelLibraryResult] = []

        def review_exact(claim, result, *_args):
            cleanup = screen._audio_cpp_result_cleanup
            assert cleanup is not None and not cleanup.release_only
            reviewed.append(result)
            assert real_ack(claim)
            screen._finish_audio_cpp_result_cleanup(claim)

        monkeypatch.setattr(
            screen,
            "_review_audio_cpp_model_library_result",
            review_exact,
        )
        exact = replace(foreign, token=expected.token)
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            exact,
        )
        screen._consume_audio_cpp_model_library_result()

        assert reviewed == [exact]
        assert screen._audio_cpp_result_cleanup is None


@pytest.mark.asyncio
async def test_same_token_revision_mismatch_terminally_clears_expected_request(
    tmp_path: Path,
) -> None:
    """A stale revision for our token settles that exact request."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = screen.query_one(SpeechTTSSettingsPanel)
        before = panel.draft_snapshot()
        expected = AudioCppModelLibraryRequest("ours", before.draft_revision)
        setattr(app_instance, "_audio_cpp_settings_model_library_request", expected)
        stale = AudioCppModelLibraryResult(
            token=expected.token,
            draft_revision=expected.draft_revision + 1,
            artifact_id="audio-cpp-supertonic-3-orig",
            revision="a" * 40,
            variant="orig",
            canonical_root=str(tmp_path.resolve()),
        )
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            stale,
        )

        screen._consume_audio_cpp_model_library_result()

        assert not hasattr(
            app_instance,
            "_audio_cpp_settings_model_library_request",
        )
        assert panel.draft_snapshot() == before
        assert panel.result_text == "Installed, not added to this changed draft"
        assert (
            app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transaction_failure",
    (
        "ack_false",
        "ack_raise",
        "merge_raise",
        "merge_interrupt",
        "merge_generator_exit",
        "lease_acquire",
        "lease_enter",
        "lease_exit",
        "handle_root",
        "handle_closure",
        "handle_path",
        "scan_raise",
        "scan_generator_exit",
        "scan_partial",
        "scan_cancelled",
        "scan_zero",
        "scan_many_discoveries",
        "scan_many_candidates",
        "scan_malformed",
        "preack_edit",
        "preack_overlap",
        "preack_unmount",
        "duplicate",
    ),
)
async def test_transaction_failure_rolls_back_and_requeues_exact_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transaction_failure: str,
) -> None:
    """A failed final acknowledgement cannot leave a partial package merge."""

    import struct

    from types import SimpleNamespace

    from textual.widgets import Button, Input, Select

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import (
        AudioCppManagedArtifactIdentity,
    )
    from tldw_chatbook.TTS.audio_cpp_package_scanner import (
        scan_audio_cpp_package_root,
    )
    from tldw_chatbook.UI.Screens import settings_screen as settings_module
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSPanelDraftSnapshot,
        SpeechTTSSettingsPanel,
    )

    root = (tmp_path / "ack-false-package").resolve()
    root.mkdir()
    (root / "supertonic-3-orig.gguf").write_bytes(b"GGUF" + struct.pack("<I", 3))
    identity = AudioCppManagedArtifactIdentity(
        artifact_id="audio-cpp-supertonic-3-orig",
        revision=AUDIO_CPP_ARTIFACT_COMMIT,
        variant="orig",
    )
    scan = scan_audio_cpp_package_root(
        root,
        expected_managed_artifact=identity,
        expected_canonical_root=str(root),
    )
    package = scan.discoveries[0].match.candidates[0].accept(managed_artifact=identity)

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        screen.query_one(
            "#settings-speech-configure-provider", Select
        ).value = "audio_cpp"
        await pilot.pause()
        screen.query_one("#settings-speech-audio_cpp-mode", Select).value = "managed"
        await pilot.pause()
        screen.query_one(
            "#settings-speech-audio_cpp-managed-setup-source", Select
        ).value = "guided"
        await pilot.pause()
        panel = screen.query_one(SpeechTTSSettingsPanel)
        if transaction_failure == "duplicate":
            values = panel.state.providers["audio_cpp"]
            values["guided_packages"] = [package.model_dump(mode="json")]
            values["guided_default_model_id"] = package.public_model_id
        before = panel.draft_snapshot()
        request = AudioCppModelLibraryRequest(
            "ack-false-request", before.draft_revision
        )
        result = AudioCppModelLibraryResult(
            token=request.token,
            draft_revision=request.draft_revision,
            artifact_id=identity.artifact_id,
            revision=identity.revision,
            variant=identity.variant,
            canonical_root=str(root),
        )
        setattr(
            app_instance,
            "_audio_cpp_settings_model_library_request",
            request,
        )
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, result
        )
        claim = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )
        assert claim is not None
        screen._begin_audio_cpp_result_cleanup(
            claim,
            request,
            panel=panel,
            before=before,
            before_result_text=panel.result_text,
            release_only=False,
        )

        worker_failures = {
            "lease_acquire",
            "lease_enter",
            "handle_root",
            "handle_closure",
            "handle_path",
            "scan_raise",
            "scan_generator_exit",
            "scan_partial",
            "scan_cancelled",
            "scan_zero",
            "scan_many_discoveries",
            "scan_many_candidates",
            "scan_malformed",
        }
        if transaction_failure in worker_failures:
            from tldw_chatbook.Model_Artifacts.service import ArtifactRef
            from tldw_chatbook.TTS.audio_cpp_package_scanner import (
                AudioCppScanOutcome,
            )

            reference = ArtifactRef(
                identity.artifact_id,
                identity.revision,
                identity.variant,
            )
            handle_root = (
                ArtifactRef("audio-cpp-foreign", identity.revision, identity.variant)
                if transaction_failure == "handle_root"
                else reference
            )
            handle_closure = (
                (reference, reference)
                if transaction_failure == "handle_closure"
                else (reference,)
            )
            handle_path = (
                root / "foreign" if transaction_failure == "handle_path" else root
            )
            lease_state = {"active": False, "released": False}

            class FailureLease:
                handle = SimpleNamespace(
                    root=handle_root,
                    closure=handle_closure,
                    paths=((reference, handle_path),),
                )

                def __enter__(self):
                    if transaction_failure == "lease_enter":
                        raise RuntimeError("private-lease-enter-canary")
                    lease_state["active"] = True
                    return self

                def __exit__(self, *_args: object) -> None:
                    lease_state["active"] = False
                    lease_state["released"] = True

            def acquire(_reference: object) -> FailureLease:
                if transaction_failure == "lease_acquire":
                    raise RuntimeError("private-lease-acquire-canary")
                return FailureLease()

            worker_scan = scan_audio_cpp_package_root(
                root,
                request_revision=result.draft_revision,
                expected_managed_artifact=identity,
                expected_canonical_root=str(root),
            )
            if transaction_failure == "scan_partial":
                worker_scan = replace(
                    worker_scan,
                    outcome=AudioCppScanOutcome.PARTIAL,
                )
            elif transaction_failure == "scan_cancelled":
                worker_scan = replace(
                    worker_scan,
                    outcome=AudioCppScanOutcome.CANCELLED,
                )
            elif transaction_failure == "scan_zero":
                worker_scan = replace(worker_scan, discoveries=())
            elif transaction_failure == "scan_many_discoveries":
                worker_scan = replace(
                    worker_scan,
                    discoveries=worker_scan.discoveries * 2,
                )
            elif transaction_failure == "scan_many_candidates":
                discovery = worker_scan.discoveries[0]
                match = replace(
                    discovery.match,
                    candidates=discovery.match.candidates * 2,
                )
                worker_scan = replace(
                    worker_scan,
                    discoveries=(replace(discovery, match=match),),
                )

            scanner = MagicMock(return_value=worker_scan)
            if transaction_failure == "scan_raise":
                scanner.side_effect = RuntimeError("private-scanner-canary")
            elif transaction_failure == "scan_generator_exit":
                scanner.side_effect = GeneratorExit("private-scanner-control")
            elif transaction_failure == "scan_malformed":
                scanner.return_value = object()
            monkeypatch.setattr(
                settings_module,
                "managed_service",
                lambda: SimpleNamespace(acquire_installed_root=acquire),
            )
            monkeypatch.setattr(
                settings_module,
                "scan_audio_cpp_package_root",
                scanner,
            )
            monkeypatch.setattr(
                host,
                "call_from_thread",
                lambda callback, *args: callback(*args),
            )

            def invoke() -> None:
                SettingsScreen._review_audio_cpp_model_library_result.__wrapped__(
                    screen,
                    claim,
                    result,
                    panel,
                    before,
                    panel.result_text,
                    request,
                )

            if transaction_failure == "scan_generator_exit":
                with pytest.raises(GeneratorExit):
                    invoke()
            else:
                invoke()
            replay = app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )

            assert panel.draft_snapshot() == before
            assert replay is not None and replay.value == result
            assert scanner.call_count <= 1
            assert not lease_state["active"]
            if transaction_failure not in {"lease_acquire", "lease_enter"}:
                assert lease_state["released"]
            assert screen._audio_cpp_result_cleanup is None
            return

        if transaction_failure == "preack_edit":
            merged = screen._merge_and_ack_audio_cpp_model_library_result(
                claim,
                result,
                package,
                False,
            )
            await pilot.pause()
            screen.query_one("#settings-speech-speed", Input).value = "1.77"
            await pilot.pause()
            settled = screen._ack_merged_audio_cpp_model_library_result(
                claim,
                panel,
                before,
                panel.result_text,
                request,
                result,
                merged,
            )
            await pilot.pause()
            replay = app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )
            after = panel.draft_snapshot()

            assert settled is False
            assert after.state.defaults.speed == 1.77
            assert (
                after.state.providers["audio_cpp"]["guided_packages"]
                == (before.state.providers["audio_cpp"]["guided_packages"])
            )
            assert after.state.providers["audio_cpp"].get(
                "guided_default_model_id"
            ) == before.state.providers["audio_cpp"].get("guided_default_model_id")
            assert after.draft_revision > merged.draft_revision
            assert replay is not None and replay.value == result
            return

        if transaction_failure == "preack_overlap":
            merged = screen._merge_and_ack_audio_cpp_model_library_result(
                claim,
                result,
                package,
                False,
            )
            assert type(merged) is SpeechTTSPanelDraftSnapshot
            await pilot.pause()
            panel.state.providers["audio_cpp"]["guided_packages"] = []
            panel._synchronize_draft_revision()
            overlapped = panel.draft_snapshot()

            settled = screen._ack_merged_audio_cpp_model_library_result(
                claim,
                panel,
                before,
                panel.result_text,
                request,
                result,
                merged,
            )

            assert settled is False
            assert panel.draft_snapshot() == overlapped
            assert screen._audio_cpp_result_cleanup is not None
            assert (
                getattr(
                    app_instance,
                    "_audio_cpp_settings_model_library_request",
                )
                is request
            )
            assert not app_instance.pending_handoffs.has_pending(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )
            await pilot.pause()
            assert panel.result_text == "Finishing installed package review…"
            assert screen.query_one("#settings-speech-save", Button).disabled
            assert screen.query_one("#settings-speech-revert", Button).disabled
            assert screen.query_one(
                "#settings-speech-restore-defaults", Button
            ).disabled
            assert screen.query_one(
                "#settings-speech-audio-cpp-guided-add-package", Button
            ).disabled
            assert screen.query_one(
                "#settings-speech-audio_cpp-guided-default-model-id", Select
            ).disabled
            assert not screen.query_one("#settings-speech-speed", Input).disabled
            assert panel.request_save() is None
            assert panel._latest_request_id is None
            assert await panel.confirm_leave() is False
            assert await screen.flush_pending_work() is False

            panel.state.providers["audio_cpp"]["guided_packages"] = list(
                merged.state.providers["audio_cpp"]["guided_packages"]
            )
            panel._synchronize_draft_revision()
            screen._retry_audio_cpp_result_cleanup()
            replay = app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )
            assert replay is not None and replay.value == result
            assert screen._audio_cpp_result_cleanup is None
            await _wait_for_selector(
                screen, pilot, "#settings-speech-save", timeout=8.0
            )
            assert not screen.query_one("#settings-speech-save", Button).disabled
            return

        if transaction_failure == "preack_unmount":
            from textual.css.query import NoMatches
            from textual.screen import Screen

            from tldw_chatbook.Model_Artifacts.service import ArtifactRef

            merged = screen._merge_and_ack_audio_cpp_model_library_result(
                claim,
                result,
                package,
                False,
            )
            panel.state.providers["audio_cpp"]["guided_packages"] = []
            panel._synchronize_draft_revision()
            await host.switch_screen(Screen())
            saved = screen.save_state()
            settled = screen._ack_merged_audio_cpp_model_library_result(
                claim,
                panel,
                before,
                panel.result_text,
                request,
                result,
                merged,
            )
            assert settled is False
            assert app_instance.pending_handoffs.has_pending(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )
            await pilot.pause()

            reference = ArtifactRef(
                identity.artifact_id,
                identity.revision,
                identity.variant,
            )
            worker_scan = scan_audio_cpp_package_root(
                root,
                request_revision=result.draft_revision,
                expected_managed_artifact=identity,
                expected_canonical_root=str(root),
            )

            class RetryLease:
                handle = SimpleNamespace(
                    root=reference,
                    closure=(reference,),
                    paths=((reference, root),),
                )

                def __enter__(self):
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

            monkeypatch.setattr(
                settings_module,
                "managed_service",
                lambda: SimpleNamespace(
                    acquire_installed_root=lambda _reference: RetryLease()
                ),
            )
            monkeypatch.setattr(
                settings_module,
                "scan_audio_cpp_package_root",
                lambda *_args, **_kwargs: worker_scan,
            )
            replacement = SettingsScreen(app_instance)
            replacement.restore_state(saved)
            await host.switch_screen(replacement)
            await _wait_for_selector(
                replacement,
                pilot,
                "#settings-speech-tts-panel",
                timeout=8.0,
            )

            def retried() -> bool:
                try:
                    restored_panel = replacement.query_one(SpeechTTSSettingsPanel)
                except NoMatches:
                    return False
                return len(restored_panel._audio_cpp_guided_packages()) == 1

            assert await _wait_for(retried, pilot)
            assert (
                app_instance.pending_handoffs.claim(
                    HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
                )
                is None
            )
            return

        if transaction_failure == "lease_exit":
            from tldw_chatbook.Model_Artifacts.service import ArtifactRef

            reference = ArtifactRef(
                identity.artifact_id,
                identity.revision,
                identity.variant,
            )
            worker_scan = scan_audio_cpp_package_root(
                root,
                request_revision=result.draft_revision,
                expected_managed_artifact=identity,
                expected_canonical_root=str(root),
            )

            class ExitFailureLease:
                handle = SimpleNamespace(
                    root=reference,
                    closure=(reference,),
                    paths=((reference, root),),
                )

                def __enter__(self):
                    return self

                def __exit__(self, *_args: object) -> None:
                    raise RuntimeError("private-lease-exit-canary")

            monkeypatch.setattr(
                settings_module,
                "managed_service",
                lambda: SimpleNamespace(
                    acquire_installed_root=lambda _reference: ExitFailureLease()
                ),
            )
            monkeypatch.setattr(
                settings_module,
                "scan_audio_cpp_package_root",
                lambda *_args, **_kwargs: worker_scan,
            )
            monkeypatch.setattr(
                host,
                "call_from_thread",
                lambda callback, *args: callback(*args),
            )

            SettingsScreen._review_audio_cpp_model_library_result.__wrapped__(
                screen,
                claim,
                result,
                panel,
                before,
                panel.result_text,
                request,
            )
            replay = app_instance.pending_handoffs.claim(
                HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
            )

            assert panel.draft_snapshot() == before
            assert replay is not None and replay.value == result
            assert screen._audio_cpp_result_cleanup is None
            return

        def fail_acknowledgement(_claim: object) -> bool:
            if transaction_failure == "ack_raise":
                raise RuntimeError("private-ack-canary")
            return False

        if transaction_failure.startswith("ack_"):
            monkeypatch.setattr(
                app_instance.pending_handoffs,
                "acknowledge",
                fail_acknowledgement,
            )
        elif transaction_failure == "merge_raise":

            def partially_merge(*_args: object, **_kwargs: object) -> None:
                panel.state.defaults.speed = 99.0
                raise RuntimeError("private-merge-canary")

            monkeypatch.setattr(
                panel,
                "merge_managed_audio_cpp_package",
                partially_merge,
            )
        elif transaction_failure in {"merge_interrupt", "merge_generator_exit"}:
            control = (
                GeneratorExit("private-generator-canary")
                if transaction_failure == "merge_generator_exit"
                else KeyboardInterrupt("private-interrupt-canary")
            )
            monkeypatch.setattr(
                panel,
                "merge_managed_audio_cpp_package",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(control),
            )

        if transaction_failure in {"merge_interrupt", "merge_generator_exit"}:
            expected_control = (
                GeneratorExit
                if transaction_failure == "merge_generator_exit"
                else KeyboardInterrupt
            )
            with pytest.raises(expected_control):
                screen._merge_and_ack_audio_cpp_model_library_result(
                    claim, result, package
                )
            settled = False
        else:
            settled = screen._merge_and_ack_audio_cpp_model_library_result(
                claim, result, package
            )
        replay = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )

        assert settled is False
        assert panel.draft_snapshot() == before
        assert replay is not None and replay.value == result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("restore", "release"))
async def test_result_cleanup_retries_after_one_owner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """A transient owner-thread cleanup failure keeps one exact retry record."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = screen.query_one(SpeechTTSSettingsPanel)
        before = panel.draft_snapshot()
        request = AudioCppModelLibraryRequest("cleanup-retry", before.draft_revision)
        result = AudioCppModelLibraryResult(
            token=request.token,
            draft_revision=request.draft_revision,
            artifact_id="audio-cpp-supertonic-3-orig",
            revision="a" * 40,
            variant="orig",
            canonical_root=str(tmp_path.resolve()),
        )
        setattr(app_instance, "_audio_cpp_settings_model_library_request", request)
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            result,
        )
        claim = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )
        assert claim is not None
        screen._begin_audio_cpp_result_cleanup(
            claim,
            request,
            panel=panel,
            before=before,
            before_result_text=panel.result_text,
            release_only=False,
        )
        panel.state.defaults.speed = 99.0
        attempts = 0
        if failure == "restore":
            real = panel.restore_draft_snapshot

            def fail_once(*args: object, **kwargs: object) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("private-restore-once")
                real(*args, **kwargs)

            monkeypatch.setattr(panel, "restore_draft_snapshot", fail_once)
        else:
            real = app_instance.pending_handoffs.release

            def fail_once(value: object) -> bool:
                nonlocal attempts
                attempts += 1
                if attempts <= 2:
                    raise RuntimeError("private-release-once")
                return real(value)

            monkeypatch.setattr(app_instance.pending_handoffs, "release", fail_once)

        with pytest.raises(RuntimeError):
            screen._rollback_and_release_audio_cpp_model_library_result(
                claim,
                panel,
                before,
                panel.result_text,
                request,
            )
        assert screen._audio_cpp_result_cleanup is not None

        if failure == "release":
            panel.state.defaults.speed = 77.0
            edited = panel.draft_snapshot()
            assert screen.stage_audio_cpp_model_library_request(edited) is False
            with pytest.raises(RuntimeError):
                screen._retry_audio_cpp_result_cleanup()
            assert panel.draft_snapshot() == edited
            screen._retry_audio_cpp_result_cleanup()
            assert panel.draft_snapshot() == edited
        else:
            screen.on_unmount()
        replay = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )

        if failure == "restore":
            assert panel.draft_snapshot() == before
        assert replay is not None and replay.value == result
        assert screen._audio_cpp_result_cleanup is None


@pytest.mark.asyncio
async def test_worker_call_from_thread_failure_retries_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed cleanup dispatch is retried with the complete exact rollback."""

    from textual.widgets import Button

    from Tests.UI.test_destination_shells import (
        DestinationHarness,
        _active_destination_screen,
        _build_test_app,
        _wait_for_selector,
    )
    from tldw_chatbook.UI.Screens import settings_screen as settings_module
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSSettingsPanel,
    )

    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)
        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_for_selector(
            screen, pilot, "#settings-speech-tts-panel", timeout=8.0
        )
        panel = screen.query_one(SpeechTTSSettingsPanel)
        before = panel.draft_snapshot()
        request = AudioCppModelLibraryRequest("dispatch-retry", before.draft_revision)
        result = AudioCppModelLibraryResult(
            token=request.token,
            draft_revision=request.draft_revision,
            artifact_id="audio-cpp-supertonic-3-orig",
            revision="a" * 40,
            variant="orig",
            canonical_root=str(tmp_path.resolve()),
        )
        setattr(app_instance, "_audio_cpp_settings_model_library_request", request)
        app_instance.pending_handoffs.stage(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT,
            result,
        )
        monkeypatch.setattr(
            settings_module,
            "managed_service",
            MagicMock(side_effect=RuntimeError("private-lease-canary")),
        )
        dispatches = 0

        def fail_once(callback, *args):
            nonlocal dispatches
            dispatches += 1
            raise RuntimeError("private-dispatch-canary")

        def review(*args):
            return SettingsScreen._review_audio_cpp_model_library_result.__wrapped__(
                screen,
                *args,
            )

        with monkeypatch.context() as scoped:
            scoped.setattr(host, "call_from_thread", fail_once)
            scoped.setattr(screen, "_review_audio_cpp_model_library_result", review)
            with pytest.raises(RuntimeError, match="private-dispatch-canary"):
                screen._consume_audio_cpp_model_library_result()
        assert screen._audio_cpp_result_cleanup is not None

        screen.on_unmount()
        replay = app_instance.pending_handoffs.claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
        )

        assert dispatches == 2
        assert replay is not None and replay.value == result
        assert screen._audio_cpp_result_cleanup is None
