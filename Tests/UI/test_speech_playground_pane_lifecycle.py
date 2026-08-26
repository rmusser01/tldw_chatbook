"""TASK-2951 port: the retired legacy playground widget's remaining
behavioral coverage, proven against the live `SpeechPlaygroundPane`.

`tldw_chatbook/UI/STTS_Window.py` used to carry a duplicate, independently
written TTS Playground widget -- dead code (never mounted in production) by
the time this port happened, deleted in the same branch along with its
dedicated test file, `Tests/UI/test_stts_playground_audio_cpp.py` (94
tests, all mounting only that widget). A prior research pass confirmed
several mechanisms named in that file's tests -- `_pending_voice_selections`,
`_remember_current_controls`, `_control_snapshot_for`,
`_catalog_request_generations`, the "stts-playback" worker group,
`_rehydrate_handler_state`, and the `controls_from_profile_preset`/
`profile_availability_from_catalog` pure functions -- exist verbatim in the
still-live replacement, `SpeechPlaygroundPane`
(`tldw_chatbook/UI/Speech/speech_playground_pane.py`, built from
`SpeechSynthesisMixin`, `SpeechCatalogMixin`, `SpeechPlaybackMixin` and
`SpeechProfileMixin`) -- independently written, not inherited, but
functionally parallel. This file ports the tests that exercise those shared
mechanisms so their guarantee is proven against the pane that is actually
reachable from the app, not the deleted widget.

3 of the widget file's tests were already covered by the pane's own suite
and are not ported again here (`test_unavailable_profile_preset_disables_
generation_and_points_to_edit`, `test_profile_service_acquisition_failure_
projects_exact_disabled_recovery`,
`test_save_profile_action_is_hidden_for_legacy_failed_and_in_progress_
states`). 14 more were RED against the widget itself (dead code) and are
excluded outright. Two pure-function tests
(`test_profile_preset_projection_*`) were ported into
`Tests/UI/test_stts_playground_catalog.py` instead, next to that module's
existing `controls_from_profile_preset` coverage.
"""

from __future__ import annotations

import asyncio
import threading
import wave
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    RichLog,
    Select,
    Static,
    TextArea,
)

from Tests.UI.background_signals import wait_for_signal
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS import (
    AudioCppCloneSetupProjection,
    AudioCppRuntimeObservation,
    LoadedTTSProfile,
    ProfileRepositoryError,
    ProfileServiceError,
    TTSGenerationProfile,
    TTSPlaygroundSelectionPreset,
)
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    AudioCppDiagnosticLine,
    AudioCppProcessSnapshot,
    AudioCppSupervisor,
)
from tldw_chatbook.TTS.playground_types import (
    STTSGeneratedAudio,
    STTSPlaygroundResultProjection,
)
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.UI import stts_profile_library as profile_library_module
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Constants import TAB_PERSONAS
from tldw_chatbook.UI.stts_playground_catalog import (
    LOADING_SELECT_VALUE,
    SERVER_DEFAULT_VOICE_ID,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech import speech_playground_pane as playground_pane_module
from tldw_chatbook.UI.Speech.speech_clone_setup import SpeechCloneSetup
from tldw_chatbook.UI.Speech.audio_cpp_runtime_card import (
    AudioCppRuntimeCardObservation,
    AudioCppRuntimeCard,
    project_audio_cpp_runtime_card,
)

from Tests.UI.app_factory import _build_test_app
from Tests.UI.speech_playground_fixtures import (
    FakeTTSService,
    _native_profile_artifact,
    _profile_preset,
    _resolved,
    _wait_until,
)

#: Every provider id the fake service and catalogs below know about. Copied
#: locally (not imported) per the porting plan: it names a local test
#: constant, not a shared fixture.
PROVIDER_IDS = (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)


def _runtime_observation(
    *,
    saved_mode: str = "managed",
    applied_mode: str = "managed",
    saved_generation: int = 1,
    applied_generation: int = 1,
    process_state: str = "stopped",
    process_generation: int = 0,
    capability: str = "unknown",
    endpoint: str | None = None,
    active_endpoint: str | None = None,
    catalog_revision: int | None = None,
    catalog_fresh: bool = False,
    service_closed: bool = False,
    diagnostics: tuple[AudioCppDiagnosticLine, ...] = (),
    dropped_diagnostics: int = 0,
    saved_setup_source: str | None = "user_json",
    applied_setup_source: str | None = "user_json",
    saved_guided_model_ids: tuple[str, ...] = (),
    applied_guided_model_ids: tuple[str, ...] = (),
    saved_guided_default_model_id: str | None = None,
    applied_guided_default_model_id: str | None = None,
    saved_guided_text_ready: bool = False,
    applied_guided_text_ready: bool = False,
    clone_setup: AudioCppCloneSetupProjection | None = None,
) -> AudioCppRuntimeObservation:
    return AudioCppRuntimeObservation(
        saved_mode=saved_mode,  # type: ignore[arg-type]
        saved_configuration_generation=saved_generation,
        applied_mode=applied_mode,  # type: ignore[arg-type]
        applied_configuration_generation=applied_generation,
        provider_configuration_revision=3,
        pending_configuration=saved_generation != applied_generation,
        process=AudioCppProcessSnapshot(
            state=process_state,  # type: ignore[arg-type]
            process_generation=process_generation,
            observation_version=process_generation + 4,
            endpoint=endpoint,
            tts_capability=capability,  # type: ignore[arg-type]
            consecutive_health_failures=(1 if process_state == "unhealthy" else 0),
            last_failure=None,
            diagnostics=diagnostics,
            dropped_diagnostic_lines=dropped_diagnostics,
        ),
        catalog_revision=catalog_revision,
        catalog_fresh=catalog_fresh,
        catalog_observed_at=None,
        tts_capability=capability,  # type: ignore[arg-type]
        service_closed=service_closed,
        saved_managed_binary_path=(
            "/private/saved/audiocpp_server" if saved_mode == "managed" else None
        ),
        saved_managed_server_json_path=(
            "/private/saved/server.json" if saved_mode == "managed" else None
        ),
        applied_managed_binary_path=(
            "/private/applied/audiocpp_server" if applied_mode == "managed" else None
        ),
        applied_managed_server_json_path=(
            "/private/applied/server.json" if applied_mode == "managed" else None
        ),
        active_endpoint=active_endpoint,
        saved_managed_setup_source=(
            saved_setup_source if saved_mode == "managed" else None
        ),
        applied_managed_setup_source=(
            applied_setup_source if applied_mode == "managed" else None
        ),
        saved_guided_model_ids=saved_guided_model_ids,
        applied_guided_model_ids=applied_guided_model_ids,
        saved_guided_default_model_id=saved_guided_default_model_id,
        applied_guided_default_model_id=applied_guided_default_model_id,
        saved_guided_text_ready=saved_guided_text_ready,
        applied_guided_text_ready=applied_guided_text_ready,
        clone_setup=clone_setup,
    )


def _clone_setup(model_id: str = "clone-voice") -> AudioCppCloneSetupProjection:
    return AudioCppCloneSetupProjection(
        model_id=model_id,
        recipe_id="audio-cpp-0.5.1.pocket_tts.pocket_tts_english_bf16",
        recipe_revision=2,
        family_label="Pocket TTS",
        recipe_label="Pocket TTS English BF16",
        reference_requirement="required",
        voice_reference_policy="reference_only",
    )


def test_clone_transcript_boundary_runs_shared_input_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, bool]] = []

    def validate_text_input(
        text: str,
        max_length: int = 10_000,
        allow_html: bool = False,
    ) -> bool:
        calls.append((text, max_length, allow_html))
        return True

    monkeypatch.setattr(
        playground_pane_module,
        "validate_text_input",
        validate_text_input,
    )

    assert playground_pane_module._validate_clone_transcript_input("  spoken words  ") == (
        "spoken words"
    )
    assert calls == [("  spoken words  ", 4_096, True)]


def test_reference_required_guided_model_projects_setup_actions() -> None:
    stopped = _runtime_observation(
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=("clone-voice",),
        applied_guided_model_ids=("clone-voice",),
        saved_guided_default_model_id="clone-voice",
        applied_guided_default_model_id="clone-voice",
        clone_setup=_clone_setup(),
    )
    stopped_projection = project_audio_cpp_runtime_card(
        AudioCppRuntimeCardObservation(runtime=stopped)
    )

    assert stopped_projection.primary_action.label == "Start & Set Up Voice"
    assert stopped_projection.primary_action.operation == "test"

    ready = _runtime_observation(
        process_state="running",
        process_generation=2,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=("clone-voice",),
        applied_guided_model_ids=("clone-voice",),
        saved_guided_default_model_id="clone-voice",
        applied_guided_default_model_id="clone-voice",
        clone_setup=_clone_setup(),
    )
    ready_projection = project_audio_cpp_runtime_card(
        AudioCppRuntimeCardObservation(runtime=ready)
    )

    assert ready_projection.primary_action.label == "Create Voice & Generate"
    assert ready_projection.primary_action.operation == "clone_generate"
    assert ready_projection.primary_action.enabled is False
    assert "reference WAV" in ready_projection.primary_action.disabled_reason


@pytest.mark.asyncio
async def test_ready_clone_model_mounts_and_canonicalizes_path_free_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_id = "<opaque:model>"
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=2,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
            saved_setup_source="guided",
            applied_setup_source="guided",
            saved_guided_model_ids=(model_id,),
            applied_guided_model_ids=(model_id,),
            saved_guided_default_model_id=model_id,
            applied_guided_default_model_id=model_id,
            clone_setup=_clone_setup(model_id),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    source = tmp_path / "private-speaker-name.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 2_400)
    app = _PaneHost(
        provider="audio_cpp",
        studio_preferences=StudioTTSPreferencesSnapshot(),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_until(
            pilot,
            lambda: len(app.query(SpeechCloneSetup)) == 1,
        )
        pane = app.query_one(SpeechPlaygroundPane)
        setup = app.query_one(SpeechCloneSetup)
        assert "plaintext" in str(
            setup.query_one("#speech-clone-privacy", Static).render()
        ).lower()
        app.query_one("#tts-text-input", TextArea).text = "Generate cloned speech."
        await pilot.pause()
        ordinary_generate = app.query_one("#tts-generate-btn", Button)
        assert ordinary_generate.disabled is True
        assert "reference WAV" in str(ordinary_generate.tooltip)

        transcript = app.query_one("#speech-clone-reference-text", TextArea)
        transcript.text = "The exact words spoken in this local reference."
        pane._handle_clone_reference_selection(source)
        await _wait_until(pilot, lambda: pane._clone_setup_canonical is not None)

        status_copy = str(
            app.query_one("#speech-clone-reference-status", Static).render()
        )
        assert "selected" in status_copy.lower()
        assert source.name not in status_copy
        assert str(tmp_path) not in status_copy
        primary = app.query_one("#tts-test-connection-btn", Button)
        assert str(primary.label) == "Create Voice & Generate"
        assert primary.disabled is False
        assert ordinary_generate.disabled is True
        assert "Use Create Voice & Generate" in str(ordinary_generate.tooltip)

        pane.action_generate_tts()
        await pilot.pause()
        assert app.generation_events == []
        assert app.notices[-1] == (
            "Use Create Voice & Generate for this reference-based model.",
            "warning",
        )

        # Reference-only recipes do not consume a catalog voice. A slow or
        # unsupported voices endpoint must not block the exact clone action,
        # even when the general controls projection cannot validate that voice.
        pane._pending_voice_selections["audio_cpp"] = "[stale voice]"
        pane._catalog_generation_allowed = False
        primary.press()
        await _wait_until(pilot, lambda: len(app.generation_events) == 1)
        request = app.generation_events[0].request
        snapshot = request.clone_audition
        assert snapshot is not None
        assert request.voice_id is None
        assert snapshot.canonical_reference.reference_text == transcript.text
        assert str(source) not in repr(snapshot)
        assert pane._generation_operation_id == request.operation_id

        transcript.text = "A newer exact transcript for the same reference."
        await _wait_until(
            pilot,
            lambda: (
                pane._clone_setup_canonical is not None
                and pane._clone_setup_canonical.reference_text == transcript.text
            ),
        )
        pane._accept_clone_generation_result(
            request.operation_id,
            snapshot.draft_revision,
        )
        assert pane._clone_setup_canonical is not None

        current_snapshot = pane._clone_audition_for_request("audio_cpp", model_id)
        assert current_snapshot is not None
        pane._accept_clone_generation_result(
            request.operation_id,
            current_snapshot.draft_revision,
        )
        assert pane._clone_setup_canonical is None
        assert primary.disabled is True


@pytest.mark.asyncio
async def test_applied_clone_generation_change_clears_prior_private_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The same public model/recipe cannot retain a prior applied generation."""

    model_id = "<opaque:model>"
    running = _runtime_observation(
        process_state="running",
        process_generation=2,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=(model_id,),
        applied_guided_model_ids=(model_id,),
        saved_guided_default_model_id=model_id,
        applied_guided_default_model_id=model_id,
        clone_setup=_clone_setup(model_id),
    )
    service = _RuntimeObservationService(running)
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    source = tmp_path / "old-generation.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 2_400)
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_until(pilot, lambda: len(app.query(SpeechCloneSetup)) == 1)
        pane = app.query_one(SpeechPlaygroundPane)
        app.query_one("#speech-clone-reference-text", TextArea).text = (
            "Exact words from the old applied generation."
        )
        pane._handle_clone_reference_selection(source)
        await _wait_until(pilot, lambda: pane._clone_setup_canonical is not None)
        prior_calls = service.runtime_observation_calls

        service.runtime_observation = replace(
            running,
            saved_configuration_generation=2,
            applied_configuration_generation=2,
            provider_configuration_revision=4,
            process=replace(
                running.process,
                process_generation=3,
                observation_version=8,
            ),
        )
        pane._request_audio_cpp_runtime_observation()
        await _wait_until(
            pilot,
            lambda: service.runtime_observation_calls > prior_calls,
        )
        await _wait_until(pilot, lambda: pane._clone_setup_canonical is None)

        assert pane._clone_setup_source_path is None
        assert len(app.query(SpeechCloneSetup)) == 1
        assert pane._clone_setup_context is not None
        assert pane._clone_setup_context[-1] == 2


@pytest.mark.asyncio
async def test_replacing_clone_source_joins_prior_canonicalization_before_next(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rapid replacement cannot fan out private WAV reads or publish the old one."""

    model_id = "<opaque:model>"
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=2,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
            saved_setup_source="guided",
            applied_setup_source="guided",
            saved_guided_model_ids=(model_id,),
            applied_guided_model_ids=(model_id,),
            saved_guided_default_model_id=model_id,
            applied_guided_default_model_id=model_id,
            clone_setup=_clone_setup(model_id),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    sources = (
        tmp_path / "first.wav",
        tmp_path / "second.wav",
        tmp_path / "third.wav",
    )
    for source in sources:
        with wave.open(str(source), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(b"\x00\x00" * 2_400)
    original = playground_pane_module.canonicalize_reference_wav
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[Path] = []

    def gated_canonicalize(source: Path, transcript: str):
        calls.append(source)
        if source == sources[0]:
            first_started.set()
            release_first.wait()
        return original(source, transcript)

    monkeypatch.setattr(
        playground_pane_module,
        "canonicalize_reference_wav",
        gated_canonicalize,
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_until(pilot, lambda: len(app.query(SpeechCloneSetup)) == 1)
        pane = app.query_one(SpeechPlaygroundPane)
        transcript = app.query_one("#speech-clone-reference-text", TextArea)
        transcript.text = "Exact replacement transcript."
        pane._handle_clone_reference_selection(sources[0])
        await asyncio.to_thread(first_started.wait, 5)

        pane._handle_clone_reference_selection(sources[1])
        pane._handle_clone_reference_selection(sources[2])
        await pilot.pause(0.05)
        calls_before_release = list(calls)
        release_first.set()
        await _wait_until(
            pilot,
            lambda: (
                pane._clone_setup_canonical is not None
                and pane._clone_setup_canonical.reference_text == transcript.text
                and len(calls) >= 2
                and calls[-1] == sources[2]
            ),
        )
        assert calls_before_release == [sources[0]]
        assert pane._clone_setup_source_path == sources[2]


@pytest.mark.asyncio
async def test_unmount_joins_active_clone_canonicalization_before_clearing_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unmount retains ownership until the private WAV worker has settled."""

    model_id = "<opaque:model>"
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=2,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
            saved_setup_source="guided",
            applied_setup_source="guided",
            saved_guided_model_ids=(model_id,),
            applied_guided_model_ids=(model_id,),
            saved_guided_default_model_id=model_id,
            applied_guided_default_model_id=model_id,
            clone_setup=_clone_setup(model_id),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    source = tmp_path / "private-reference.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 2_400)
    original = playground_pane_module.canonicalize_reference_wav
    started = threading.Event()
    release = threading.Event()

    def gated_canonicalize(source_path: Path, transcript: str):
        started.set()
        release.wait()
        return original(source_path, transcript)

    monkeypatch.setattr(
        playground_pane_module,
        "canonicalize_reference_wav",
        gated_canonicalize,
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_until(pilot, lambda: len(app.query(SpeechCloneSetup)) == 1)
        pane = app.query_one(SpeechPlaygroundPane)
        transcript = app.query_one("#speech-clone-reference-text", TextArea)
        transcript.text = "Exact words from the reference."
        pane._handle_clone_reference_selection(source)
        await asyncio.to_thread(started.wait, 5)

        async def remove_pane() -> None:
            await pane.remove()

        removal = asyncio.create_task(remove_pane())
        try:
            await pilot.pause(0.05)
            assert not removal.done()
        finally:
            release.set()
        await removal

        assert pane._clone_setup_validation_task is None
        assert pane._clone_setup_retained_tasks == set()
        assert pane._clone_setup_source_path is None
        assert pane._clone_setup_canonical is None


@pytest.mark.asyncio
async def test_clone_picker_cancel_and_field_errors_preserve_the_other_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation is inert and each invalid field keeps the other draft field."""

    model_id = "<opaque:model>"
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=2,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
            saved_setup_source="guided",
            applied_setup_source="guided",
            saved_guided_model_ids=(model_id,),
            applied_guided_model_ids=(model_id,),
            saved_guided_default_model_id=model_id,
            applied_guided_default_model_id=model_id,
            clone_setup=_clone_setup(model_id),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    source = tmp_path / "valid.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 2_400)
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_until(pilot, lambda: len(app.query(SpeechCloneSetup)) == 1)
        pane = app.query_one(SpeechPlaygroundPane)
        transcript = app.query_one("#speech-clone-reference-text", TextArea)
        transcript.text = "Exact words from the valid WAV."
        pane._handle_clone_reference_selection(source)
        await _wait_until(pilot, lambda: pane._clone_setup_canonical is not None)
        accepted = pane._clone_setup_canonical

        pane._handle_clone_reference_selection(None)
        assert pane._clone_setup_canonical is accepted
        assert pane._clone_setup_source_path == source

        pane._handle_clone_reference_selection(tmp_path)
        await _wait_until(
            pilot,
            lambda: pane._clone_setup_error
            == "Choose a valid, bounded PCM WAV reference.",
        )
        await _wait_until(
            pilot,
            lambda: getattr(app.focused, "id", None)
            == "speech-clone-reference-choose",
        )
        assert transcript.text == "Exact words from the valid WAV."
        assert pane._clone_setup_source_path == tmp_path

        transcript.focus()
        transcript.text = "x" * 4_097
        await _wait_until(
            pilot,
            lambda: pane._clone_setup_error == "Enter the exact spoken transcript.",
        )
        assert pane._clone_setup_source_path == tmp_path
        assert transcript.has_focus


@pytest.mark.asyncio
async def test_stopped_clone_model_tests_before_mounting_voice_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "<opaque:model>"
    stopped = _runtime_observation(
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=(model_id,),
        applied_guided_model_ids=(model_id,),
        saved_guided_default_model_id=model_id,
        applied_guided_default_model_id=model_id,
        clone_setup=_clone_setup(model_id),
    )
    running = _runtime_observation(
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=(model_id,),
        applied_guided_model_ids=(model_id,),
        saved_guided_default_model_id=model_id,
        applied_guided_default_model_id=model_id,
        clone_setup=_clone_setup(model_id),
    )
    service = _RuntimeObservationService(stopped)
    service.lifecycle_result_observation = running
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(100, 30)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: str(primary.label) == "Start & Set Up Voice")
        assert len(app.query(SpeechCloneSetup)) == 0

        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["test"])
        await _wait_until(pilot, lambda: len(app.query(SpeechCloneSetup)) == 1)

        assert str(primary.label) == "Create Voice & Generate"
        assert primary.disabled is True
        assert "reference WAV" in str(primary.tooltip)


@pytest.mark.parametrize(
    ("observation", "primary_label", "primary_operation", "restart", "shutdown"),
    (
        (
            _runtime_observation(),
            "Start & Test Connection",
            "test",
            False,
            False,
        ),
        (
            _runtime_observation(
                process_state="running",
                process_generation=2,
                capability="available",
                endpoint="http://127.0.0.1:19001",
                catalog_revision=11,
                catalog_fresh=True,
            ),
            "Test Connection",
            "test",
            True,
            True,
        ),
        (
            _runtime_observation(
                saved_generation=2,
                applied_generation=1,
                process_state="running",
                process_generation=2,
                capability="available",
                endpoint="http://127.0.0.1:19001",
            ),
            "Restart & Apply Settings",
            "restart",
            False,
            True,
        ),
        (
            _runtime_observation(
                saved_mode="external",
                applied_mode="managed",
                saved_generation=2,
                applied_generation=1,
                process_state="running",
                process_generation=2,
                capability="available",
                endpoint="http://127.0.0.1:19001",
            ),
            "Apply Settings & Stop Managed Server",
            "restart",
            False,
            True,
        ),
        (
            _runtime_observation(
                saved_generation=2,
                applied_generation=1,
                process_state="unhealthy",
                process_generation=2,
            ),
            "Restart & Apply Settings",
            "restart",
            False,
            True,
        ),
        (
            _runtime_observation(
                saved_mode="external",
                applied_mode="managed",
                saved_generation=2,
                applied_generation=1,
                process_state="unhealthy",
                process_generation=2,
            ),
            "Apply Settings & Stop Managed Server",
            "restart",
            False,
            True,
        ),
        (
            _runtime_observation(
                process_state="unhealthy",
                process_generation=2,
            ),
            "Restart",
            "restart",
            False,
            True,
        ),
        (
            _runtime_observation(process_state="unavailable"),
            "Start & Test Connection",
            "test",
            False,
            False,
        ),
        (
            _runtime_observation(
                saved_mode="external",
                applied_mode="external",
            ),
            "Test Connection",
            "test",
            False,
            False,
        ),
    ),
)
def test_audio_cpp_runtime_projection_uses_state_specific_actions(
    observation: AudioCppRuntimeObservation,
    primary_label: str,
    primary_operation: str,
    restart: bool,
    shutdown: bool,
) -> None:
    """Each managed runtime state exposes only its valid lifecycle actions."""

    projection = project_audio_cpp_runtime_card(observation)

    assert projection.primary_action.label == primary_label
    assert projection.primary_action.operation == primary_operation
    assert projection.primary_action.enabled is True
    assert projection.restart_action.enabled is restart
    assert projection.shutdown_action.enabled is shutdown
    if not restart:
        assert projection.restart_action.disabled_reason
    if not shutdown:
        assert projection.shutdown_action.disabled_reason


def test_guided_text_ready_projection_is_one_complete_sample_action() -> None:
    """A text-ready Guided default projects one combined first-sample action."""

    observation = _runtime_observation(
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=("model-a", "model-b"),
        applied_guided_model_ids=("model-a", "model-b"),
        saved_guided_default_model_id="model-a",
        applied_guided_default_model_id="model-a",
        saved_guided_text_ready=True,
        applied_guided_text_ready=True,
    )

    projection = project_audio_cpp_runtime_card(
        AudioCppRuntimeCardObservation(runtime=observation)
    )

    assert projection.primary_action.operation == "sample"
    assert projection.primary_action.label == "Start & Generate Sample"
    assert projection.primary_action.enabled is True
    assert projection.primary_action.disabled_reason == ""
    assert projection.primary_action.tooltip == (
        "Start audio.cpp, verify model-a, and generate one complete WAV"
    )
    assert projection.primary_action.progress_label == "Starting & Generating…"
    assert projection.primary_action.post_operation_focus == "#audio-play-btn"
    assert projection.restart_action.enabled is False
    assert projection.shutdown_action.enabled is False
    assert "Guided" in projection.saved_copy
    assert "model-a" in projection.saved_copy
    assert "2 models" in projection.saved_copy
    assert "/private/" not in repr(projection)


def test_failed_guided_sample_projects_retry_without_duplicate_restart() -> None:
    observation = _runtime_observation(
        process_state="running",
        process_generation=4,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=("model-a",),
        applied_guided_model_ids=("model-a",),
        saved_guided_default_model_id="model-a",
        applied_guided_default_model_id="model-a",
        saved_guided_text_ready=True,
        applied_guided_text_ready=True,
    )

    projection = project_audio_cpp_runtime_card(
        AudioCppRuntimeCardObservation(runtime=observation, sample_state="failed")
    )

    assert projection.primary_action.operation == "sample"
    assert projection.primary_action.label == "Retry Sample"
    assert projection.primary_action.progress_label == "Generating Sample…"
    assert projection.primary_action.post_operation_focus == "#audio-play-btn"
    assert projection.restart_action.enabled is True
    assert projection.restart_action.operation == "restart"


@pytest.mark.parametrize("state", ("starting", "draining", "stopping"))
def test_audio_cpp_runtime_projection_keeps_busy_actions_visible_but_disabled(
    state: str,
) -> None:
    projection = project_audio_cpp_runtime_card(
        _runtime_observation(
            saved_generation=2,
            applied_generation=1,
            process_state=state,
            process_generation=2,
        )
    )

    assert projection.primary_action.enabled is False
    assert projection.restart_action.enabled is False
    assert projection.shutdown_action.enabled is False
    assert "in progress" in projection.primary_action.disabled_reason.lower()


def test_audio_cpp_runtime_projection_describes_generation_and_catalog_truth() -> None:
    projection = project_audio_cpp_runtime_card(
        _runtime_observation(
            saved_mode="external",
            applied_mode="managed",
            saved_generation=7,
            applied_generation=4,
            process_state="running",
            process_generation=3,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=False,
        )
    )

    assert projection.primary_status.startswith("[RUNNING]")
    assert "external mode is saved" in projection.pending_copy.lower()
    assert projection.saved_copy == "Saved: External · generation 7"
    assert projection.applied_copy == (
        "Active: Managed · Manual server.json · generation 4"
    )
    assert projection.process_copy == "Process: Running · generation 3"
    assert projection.endpoint_copy.endswith("http://127.0.0.1:19001")
    assert projection.catalog_copy == "Catalog: Stale · revision 11"


def test_staged_managed_projection_keeps_applied_external_truth() -> None:
    projection = project_audio_cpp_runtime_card(
        _runtime_observation(
            saved_mode="managed",
            applied_mode="external",
            saved_generation=7,
            applied_generation=4,
            active_endpoint="http://127.0.0.1:19001",
        )
    )

    assert projection.primary_status == (
        "[EXTERNAL] Chatbook will connect to the active external server."
    )
    assert projection.capability_copy == "TTS capability: Unknown"
    assert "managed mode is saved" in projection.pending_copy.lower()
    assert projection.endpoint_copy == "Active endpoint: http://127.0.0.1:19001"


def test_audio_cpp_runtime_projection_labels_bounded_managed_diagnostics() -> None:
    projection = project_audio_cpp_runtime_card(
        _runtime_observation(
            process_state="unavailable",
            process_generation=5,
            diagnostics=(
                AudioCppDiagnosticLine("stdout", "loaded model"),
                AudioCppDiagnosticLine("stderr", "request failed"),
            ),
            dropped_diagnostics=7,
        )
    )

    assert projection.diagnostics_generation_copy == "Process generation: 5"
    assert projection.diagnostic_lines == (
        "STDOUT · loaded model",
        "STDERR · request failed",
    )
    assert projection.dropped_diagnostics_copy == "7 older lines were dropped."


@pytest.mark.asyncio
async def test_runtime_diagnostics_are_collapsed_and_interactions_are_passive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="unavailable",
            process_generation=5,
            diagnostics=tuple(
                AudioCppDiagnosticLine("stderr", f"safe output {index:02}")
                for index in range(30)
            ),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await _wait_until(pilot, lambda: service.runtime_observation_calls > 0)
        diagnostics = app.query_one("#audio-cpp-runtime-diagnostics", Collapsible)
        await _wait_until(
            pilot,
            lambda: "generation: 5"
            in str(
                app.query_one(
                    "#audio-cpp-diagnostics-generation",
                    Static,
                ).render()
            ).lower(),
        )

        assert diagnostics.collapsed is True
        assert (
            "potentially sensitive"
            in str(
                app.query_one("#audio-cpp-diagnostics-warning", Static).render()
            ).lower()
        )
        assert (
            "generation: 5"
            in str(
                app.query_one("#audio-cpp-diagnostics-generation", Static).render()
            ).lower()
        )
        diagnostics.collapsed = False
        await pilot.pause()
        diagnostics.query_one("CollapsibleTitle").focus()
        await pilot.press("tab")

        diagnostic_log = app.query_one("#audio-cpp-diagnostics-lines", RichLog)
        assert app.focused is diagnostic_log
        assert any(
            "STDERR · safe output 00" in line.text for line in diagnostic_log.lines
        )

        await pilot.press("home")
        await pilot.pause()
        assert diagnostic_log.scroll_y == 0
        await pilot.press("pagedown")
        await pilot.pause()
        assert diagnostic_log.scroll_y > 0
        assert service.lifecycle_calls == []
        assert service.synthesize_calls == 0


class _RuntimeObservationService(FakeTTSService):
    def __init__(self, observation: AudioCppRuntimeObservation) -> None:
        super().__init__()
        self.runtime_observation = observation
        self.runtime_observation_calls = 0
        self.runtime_selected_models: list[str | None] = []
        self.lifecycle_calls: list[str] = []
        self.lifecycle_started = asyncio.Event()
        self.lifecycle_gate: asyncio.Event | None = None
        self.lifecycle_error: BaseException | None = None
        self.lifecycle_result_observation: AudioCppRuntimeObservation | None = None

    async def audio_cpp_runtime_observation(
        self,
        *,
        selected_model_id: str | None = None,
    ) -> AudioCppRuntimeObservation:
        self.runtime_observation_calls += 1
        self.runtime_selected_models.append(selected_model_id)
        return self.runtime_observation

    async def _run_lifecycle(self, operation: str) -> None:
        self.lifecycle_calls.append(operation)
        self.lifecycle_started.set()
        if self.lifecycle_gate is not None:
            await self.lifecycle_gate.wait()
        if self.lifecycle_error is not None:
            raise self.lifecycle_error
        if self.lifecycle_result_observation is not None:
            self.runtime_observation = self.lifecycle_result_observation

    async def start_and_test_audio_cpp(self) -> TTSProviderCatalog:
        await self._run_lifecycle("test")
        return self.catalogs["audio_cpp"]

    async def restart_audio_cpp(self) -> TTSProviderCatalog | None:
        await self._run_lifecycle("restart")
        return (
            self.catalogs["audio_cpp"]
            if self.runtime_observation.saved_mode == "managed"
            else None
        )

    async def shutdown_audio_cpp(self) -> None:
        await self._run_lifecycle("shutdown")


@pytest.mark.asyncio
async def test_audio_cpp_runtime_card_mounts_passively_and_updates_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: service.runtime_observation_calls > 0,
        )
        card = app.query_one("#audio-cpp-runtime-card", AudioCppRuntimeCard)
        assert card.display is True
        primary = app.query_one("#tts-test-connection-btn", Button)
        primary.focus()
        service.runtime_observation = _runtime_observation(
            saved_generation=2,
            applied_generation=1,
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane._request_audio_cpp_runtime_observation()
        await _wait_until(
            pilot,
            lambda: "Restart & Apply" in str(primary.label),
        )

        assert app.focused is primary
        assert service.lifecycle_calls == []
        assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_unhealthy_runtime_has_one_restart_and_a_labeled_test_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="unhealthy",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        secondary_restart = app.query_one("#audio-cpp-runtime-restart", Button)
        probe = app.query_one("#tts-refresh-catalog-btn", Button)
        await _wait_until(pilot, lambda: str(primary.label) == "Restart")

        assert secondary_restart.disabled is True
        assert "primary action" in str(secondary_restart.tooltip).lower()
        assert str(probe.label) == "Test Connection"

        service.runtime_observation = _runtime_observation(
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
        )
        app.query_one(SpeechPlaygroundPane)._request_audio_cpp_runtime_observation()
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")

        assert secondary_restart.disabled is False
        assert str(probe.label) == "Refresh"


@pytest.mark.asyncio
async def test_runtime_primary_action_reflows_when_passive_state_changes_its_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_runtime_observation())
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Start & Test Connection",
        )
        await pilot.pause()

        assert primary.region.width >= len(str(primary.label)) + 2


@pytest.mark.asyncio
async def test_superseded_runtime_observation_cannot_overwrite_newer_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _runtime_observation(
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
    )
    service = _RuntimeObservationService(initial)
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await _wait_until(pilot, lambda: service.runtime_observation_calls > 0)
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0
        stale = _runtime_observation(
            process_state="stopped",
            process_generation=1,
        )
        newer = _runtime_observation(
            saved_generation=2,
            applied_generation=1,
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
        )

        async def observe(
            *, selected_model_id: str | None = None
        ) -> AudioCppRuntimeObservation:
            del selected_model_id
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                return stale
            return newer

        service.audio_cpp_runtime_observation = observe  # type: ignore[method-assign]
        pane._request_audio_cpp_runtime_observation()
        await asyncio.wait_for(first_started.wait(), timeout=2.0)
        pane._request_audio_cpp_runtime_observation()
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: "Restart & Apply" in str(primary.label))

        release_first.set()
        await pilot.pause(0.1)

        assert "Restart & Apply" in str(primary.label)
        assert pane._audio_cpp_runtime_observation == newer


@pytest.mark.asyncio
async def test_audio_cpp_runtime_card_is_hidden_for_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_runtime_observation())
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    _patch_cli_setting(
        monkeypatch,
        {("app_tts", "default_provider"): "openai"},
    )
    app = _PaneHost(provider="openai")

    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        card = app.query_one("#audio-cpp-runtime-card", AudioCppRuntimeCard)

        assert card.display is False
        assert service.runtime_observation_calls == 0


@pytest.mark.asyncio
async def test_audio_cpp_runtime_poll_uses_a_bounded_five_second_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals: list[float] = []
    original_set_interval = SpeechPlaygroundPane.set_interval

    def capture_interval(
        self: SpeechPlaygroundPane,
        interval: float,
        callback: object = None,
        **kwargs: object,
    ) -> object:
        if getattr(callback, "__name__", "") == "_poll_audio_cpp_runtime_observation":
            intervals.append(interval)
        return original_set_interval(self, interval, callback, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(SpeechPlaygroundPane, "set_interval", capture_interval)
    app = _PaneHost(provider="openai")

    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()

    assert intervals == [5.0]


@pytest.mark.asyncio
async def test_switch_back_before_runtime_read_cannot_reuse_stale_primary_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(
        _runtime_observation(
            saved_generation=2,
            applied_generation=1,
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: "Restart & Apply" in str(primary.label))
        provider = app.query_one("#tts-provider-select", Select)
        provider.value = "openai"
        await _wait_until(
            pilot,
            lambda: provider.value == "openai" and str(primary.label) == "Test",
        )

        observation_started = asyncio.Event()
        release_observation = asyncio.Event()

        async def gated_observation(
            *, selected_model_id: str | None = None
        ) -> AudioCppRuntimeObservation:
            del selected_model_id
            observation_started.set()
            await release_observation.wait()
            return service.runtime_observation

        service.audio_cpp_runtime_observation = gated_observation  # type: ignore[method-assign]
        provider.value = "audio_cpp"
        await asyncio.wait_for(observation_started.wait(), timeout=2.0)
        assert str(primary.label) == "Test Connection"

        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["test"])

        assert service.lifecycle_calls == ["test"]
        release_observation.set()


@pytest.mark.asyncio
async def test_audio_cpp_runtime_card_links_to_canonical_global_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_runtime_observation())
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await _wait_until(pilot, lambda: service.runtime_observation_calls > 0)
        app.query_one("#audio-cpp-runtime-open-settings", Button).press()
        await pilot.pause()

        assert len(app.navigation) == 1
        assert app.navigation[0].screen_name == "settings"
        assert app.navigation[0].screen_context == {
            "category": "speech-tts",
            "provider": "audio_cpp",
            "intent": "configure",
        }


@pytest.mark.asyncio
async def test_managed_start_and_test_runs_async_and_preserves_current_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _RuntimeObservationService(_runtime_observation())
    service.lifecycle_gate = asyncio.Event()
    service.lifecycle_result_observation = _runtime_observation(
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                str(app.query_one("#tts-test-connection-btn", Button).label)
                == "Start & Test Connection"
            ),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        existing = tmp_path / "current.wav"
        existing.write_bytes(b"RIFF")
        pane.current_audio_file = existing

        primary = app.query_one("#tts-test-connection-btn", Button)
        original_run_worker = app.run_worker
        lifecycle_worker_options: dict[str, object] = {}

        def capture_run_worker(
            awaitable: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            if kwargs.get("group") == "speech-audio-cpp-lifecycle":
                lifecycle_worker_options.update(kwargs)
            return original_run_worker(awaitable, *args, **kwargs)  # type: ignore[arg-type,return-value]

        monkeypatch.setattr(app, "run_worker", capture_run_worker)
        primary.press()
        await asyncio.wait_for(service.lifecycle_started.wait(), timeout=2.0)
        await pilot.pause()

        assert service.lifecycle_calls == ["test"]
        assert lifecycle_worker_options["exclusive"] is True
        assert primary.disabled is True
        assert "Starting" in str(primary.label)
        assert app.query_one("#tts-refresh-catalog-btn", Button).disabled is True
        assert pane.current_audio_file == existing

        service.lifecycle_gate.set()
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Test Connection" and not primary.disabled,
        )

        assert pane.current_audio_file == existing
        assert app.query_one("#tts-refresh-catalog-btn", Button).disabled is False
        assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_running_managed_test_exposes_busy_reasons_on_disabled_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _runtime_observation(
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
    )
    service = _RuntimeObservationService(running)
    service.lifecycle_gate = asyncio.Event()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")
        primary.press()
        await asyncio.wait_for(service.lifecycle_started.wait(), timeout=2.0)
        await pilot.pause()

        reason = str(app.query_one("#audio-cpp-runtime-action-reason", Static).render())
        assert "operation is in progress" in reason.lower()
        for selector in (
            "#tts-test-connection-btn",
            "#tts-refresh-catalog-btn",
            "#tts-generate-btn",
            "#audio-cpp-runtime-restart",
            "#audio-cpp-runtime-shutdown",
        ):
            action = app.query_one(selector, Button)
            assert action.disabled is True
            assert "operation is in progress" in str(action.tooltip).lower()

        service.lifecycle_gate.set()
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")


@pytest.mark.asyncio
async def test_pending_managed_primary_routes_to_restart_and_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _runtime_observation(
        saved_generation=2,
        applied_generation=1,
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
    )
    service = _RuntimeObservationService(pending)
    service.lifecycle_result_observation = _runtime_observation(
        saved_generation=2,
        applied_generation=2,
        process_state="running",
        process_generation=2,
        capability="available",
        endpoint="http://127.0.0.1:19002",
        catalog_revision=12,
        catalog_fresh=True,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: "Restart & Apply" in str(primary.label))
        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["restart"])
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")

        assert service.lifecycle_calls == ["restart"]
        assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_switching_back_during_lifecycle_restores_visible_busy_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _runtime_observation(
        saved_generation=2,
        applied_generation=1,
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
    )
    service = _RuntimeObservationService(pending)
    service.lifecycle_gate = asyncio.Event()
    service.lifecycle_result_observation = _runtime_observation(
        saved_generation=2,
        applied_generation=2,
        process_state="running",
        process_generation=2,
        capability="available",
        endpoint="http://127.0.0.1:19002",
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: "Restart & Apply" in str(primary.label))
        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["restart"])

        provider = app.query_one("#tts-provider-select", Select)
        provider.value = "openai"
        await pilot.pause()
        provider.value = "audio_cpp"
        await pilot.pause()

        assert primary.disabled is True
        assert "Restarting & Applying" in str(primary.label)
        assert app.query_one("#tts-refresh-catalog-btn", Button).disabled is True
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        app.query_one("#tts-text-input", TextArea).text = "Do not generate yet."
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert app.generation_events == []

        service.lifecycle_gate.set()
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")


@pytest.mark.asyncio
async def test_audio_cpp_lifecycle_completion_does_not_disable_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _runtime_observation(
        saved_generation=2,
        applied_generation=1,
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
    )
    service = _RuntimeObservationService(pending)
    service.lifecycle_gate = asyncio.Event()
    service.lifecycle_result_observation = _runtime_observation(
        saved_generation=2,
        applied_generation=2,
        process_state="running",
        process_generation=2,
        capability="available",
        endpoint="http://127.0.0.1:19002",
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: "Restart & Apply" in str(primary.label))
        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["restart"])

        provider = app.query_one("#tts-provider-select", Select)
        provider.value = "openai"
        generate = app.query_one("#tts-generate-btn", Button)
        await _wait_until(
            pilot,
            lambda: provider.value == "openai" and not generate.disabled,
        )

        service.lifecycle_gate.set()
        pane = app.query_one(SpeechPlaygroundPane)
        await _wait_until(
            pilot,
            lambda: pane._audio_cpp_lifecycle_busy is None,
        )
        await pilot.pause()

        assert provider.value == "openai"
        assert generate.disabled is False


@pytest.mark.asyncio
async def test_saved_external_primary_applies_settings_and_stops_managed_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending_external = _runtime_observation(
        saved_mode="external",
        applied_mode="managed",
        saved_generation=2,
        applied_generation=1,
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
    )
    service = _RuntimeObservationService(pending_external)
    service.lifecycle_gate = asyncio.Event()
    service.lifecycle_result_observation = _runtime_observation(
        saved_mode="external",
        applied_mode="external",
        saved_generation=2,
        applied_generation=2,
        process_state="stopped",
        process_generation=1,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: "Apply Settings & Stop" in str(primary.label),
        )
        calls_before = list(service.catalog_calls)
        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["restart"])
        assert "Applying & Stopping" in str(primary.label)
        service.lifecycle_gate.set()
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")

        assert service.catalog_calls == calls_before
        assert service.runtime_observation.applied_mode == "external"


@pytest.mark.asyncio
async def test_server_shutdown_is_distinct_from_playback_stop_and_does_not_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _runtime_observation(
        process_state="running",
        process_generation=1,
        capability="available",
        endpoint="http://127.0.0.1:19001",
        catalog_revision=11,
        catalog_fresh=True,
    )
    service = _RuntimeObservationService(running)
    service.lifecycle_result_observation = _runtime_observation(
        process_state="stopped",
        process_generation=1,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: not app.query_one("#audio-cpp-runtime-shutdown", Button).disabled,
        )
        server_shutdown = app.query_one("#audio-cpp-runtime-shutdown", Button)
        playback_stop = app.query_one("#stop-audio-btn", Button)
        assert str(server_shutdown.label) == "Shut down server"
        assert str(playback_stop.label) == "Stop"

        server_shutdown.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["shutdown"])
        await _wait_until(
            pilot,
            lambda: (
                str(app.query_one("#tts-test-connection-btn", Button).label)
                == "Start & Test Connection"
            ),
        )

        assert service.lifecycle_calls == ["shutdown"]
        assert service.catalog_calls.count(("audio_cpp", False)) == 1


@pytest.mark.asyncio
async def test_lifecycle_failure_uses_safe_copy_and_keeps_existing_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _RuntimeObservationService(_runtime_observation())
    service.lifecycle_error = RuntimeError("PRIVATE /user/server.json prompt text")
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Start & Test Connection",
        )
        pane = app.query_one(SpeechPlaygroundPane)
        existing = tmp_path / "current.wav"
        existing.write_bytes(b"RIFF")
        pane.current_audio_file = existing

        primary.press()
        await _wait_until(pilot, lambda: bool(app.notices))

        rendered_notices = repr(app.notices)
        assert "PRIVATE" not in rendered_notices
        assert "server.json" not in rendered_notices
        assert "prompt text" not in rendered_notices
        assert pane.current_audio_file == existing


@pytest.mark.asyncio
async def test_lifecycle_and_passive_status_failure_restore_safe_recovery_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
        )
    )
    service.lifecycle_error = RuntimeError("PRIVATE lifecycle failure")
    original_observe = service.audio_cpp_runtime_observation

    async def observe_until_lifecycle_fails(
        *, selected_model_id: str | None = None
    ) -> AudioCppRuntimeObservation:
        if service.lifecycle_calls:
            raise RuntimeError("PRIVATE passive status failure")
        return await original_observe(selected_model_id=selected_model_id)

    service.audio_cpp_runtime_observation = observe_until_lifecycle_fails  # type: ignore[method-assign]
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")
        primary.press()
        await _wait_until(
            pilot,
            lambda: (
                "status could not be read"
                in str(
                    app.query_one("#audio-cpp-runtime-action-reason", Static).render()
                ).lower()
            ),
        )

        assert str(primary.label) == "Test Connection"
        assert primary.disabled is False
        assert "retry" in str(primary.tooltip).lower()
        refresh = app.query_one("#tts-refresh-catalog-btn", Button)
        assert refresh.disabled is False
        assert "retry" in str(refresh.tooltip).lower()
        generate = app.query_one("#tts-generate-btn", Button)
        assert generate.disabled is True
        assert "checked" in str(generate.tooltip).lower()
        for selector in (
            "#audio-cpp-runtime-restart",
            "#audio-cpp-runtime-shutdown",
        ):
            action = app.query_one(selector, Button)
            assert action.disabled is True
            assert "checked" in str(action.tooltip).lower()


@pytest.mark.asyncio
async def test_unknown_runtime_test_does_not_reuse_stale_pending_restart_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(
        _runtime_observation(
            saved_generation=2,
            applied_generation=1,
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(pilot, lambda: "Restart & Apply" in str(primary.label))

        async def unavailable_observation(
            *, selected_model_id: str | None = None
        ) -> AudioCppRuntimeObservation:
            del selected_model_id
            raise RuntimeError("PRIVATE passive status failure")

        service.audio_cpp_runtime_observation = unavailable_observation  # type: ignore[method-assign]
        pane = app.query_one(SpeechPlaygroundPane)
        pane._request_audio_cpp_runtime_observation()
        await _wait_until(pilot, lambda: str(primary.label) == "Test Connection")

        primary.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["test"])

        assert service.lifecycle_calls == ["test"]


def _audio_catalog(
    *,
    health: ProviderHealth | None = None,
    revision: int = 11,
) -> TTSProviderCatalog:
    """Copied verbatim from the retired widget's test file (and identical to
    the private helper `FakeTTSService` builds internally in
    `speech_playground_fixtures.py`) so tests here can swap in a differently
    shaped audio_cpp catalog (stale health, a new revision, ...).
    """
    return TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=revision,
        health=health or ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id="<opaque:model>",
                display_name="[bold red]Opaque model[/]",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
            TTSModelInfo(
                model_id="second-model",
                display_name="Second model",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )


def _guided_text_catalog(model_id: str = "model-a") -> TTSProviderCatalog:
    return TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=21,
        health=ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id=model_id,
                display_name="Guided model",
                family="supertonic",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )


def _guided_runtime_observation(
    *,
    process_state: str = "stopped",
    process_generation: int = 0,
    catalog_fresh: bool = False,
) -> AudioCppRuntimeObservation:
    running = process_state == "running"
    return _runtime_observation(
        process_state=process_state,
        process_generation=process_generation,
        capability="available" if running else "unknown",
        endpoint="http://127.0.0.1:19001" if running else None,
        catalog_revision=21 if catalog_fresh else None,
        catalog_fresh=catalog_fresh,
        saved_setup_source="guided",
        applied_setup_source="guided",
        saved_guided_model_ids=("model-a",),
        applied_guided_model_ids=("model-a",),
        saved_guided_default_model_id="model-a",
        applied_guided_default_model_id="model-a",
        saved_guided_text_ready=True,
        applied_guided_text_ready=True,
    )


def _option_values(select: Select[Any]) -> tuple[Any, ...]:
    return tuple(value for _label, value in select._options)


def _option_labels(select: Select[Any]) -> tuple[str, ...]:
    labels = []
    for label, _value in select._options:
        labels.append(label.plain if isinstance(label, Text) else str(label))
    return tuple(labels)


def _label_for_value(select: Select[Any], value: str) -> str:
    for label, option_value in select._options:
        if option_value == value:
            return label.plain if isinstance(label, Text) else str(label)
    raise AssertionError(f"Missing Select value: {value}")


class _SaveProfileService:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, STTSGeneratedAudio]] = []
        self.error: BaseException | None = None

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> object:
        self.create_calls.append((display_name, artifact))
        if self.error is not None:
            raise self.error
        return object()


class _PaneHost(App[None]):
    """Hosts `SpeechPlaygroundPane` alone -- the pane-side equivalent of the
    retired widget test file's `_PlaygroundHost`. Captures generation events
    and notifications the same way, and answers `_ensure_tts_profile_
    service` for the profile-save path.
    """

    def __init__(
        self,
        *,
        provider: str = "audio_cpp",
        preset: TTSPlaygroundSelectionPreset | None = None,
        profile_service: object | None = None,
        studio_preferences: StudioTTSPreferencesSnapshot | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.preset = preset
        self.profile_service = profile_service
        self.studio_preferences = studio_preferences
        self.profile_service_requests = 0
        self.notices: list[tuple[str, str]] = []
        self.generation_events: list[STTSPlaygroundGenerateEvent] = []
        self.navigation: list[NavigateToScreen] = []

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(
            id="speech-playground-pane",
            provider=self.provider,
            profile_preset=self.preset,
            studio_preferences=self.studio_preferences,
        )

    async def _ensure_tts_profile_service(self) -> object | None:
        self.profile_service_requests += 1
        return self.profile_service

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        del title, timeout
        self.notices.append((message, severity))

    def post_message(self, message: Any) -> bool:
        if isinstance(message, STTSPlaygroundGenerateEvent):
            self.generation_events.append(message)
            return True
        if isinstance(message, NavigateToScreen):
            self.navigation.append(message)
            return True
        return super().post_message(message)


def _saved_clone_profile(*, profile_id: int, name: str) -> TTSGenerationProfile:
    timestamp = datetime(2026, 8, 11, tzinfo=UTC)
    return TTSGenerationProfile(
        profile_id=UUID(int=profile_id),
        display_name=name,
        normalized_name=name.casefold(),
        provider_id="audio_cpp",
        model_id="artifact/model",
        voice_id="artifact/voice",
        response_format="wav",
        speed=1.0,
        options={},
        revision=2,
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest.mark.asyncio
async def test_clone_profile_save_handoff_carries_exact_saved_identity_without_assignment(
    tmp_path: Path,
) -> None:
    """The pane routes only the committed profile identity to Roleplay."""

    service = _SaveProfileService()
    app = _PaneHost(profile_service=service)
    saved_profile = _saved_clone_profile(profile_id=123, name="Story voice")
    loaded = LoadedTTSProfile(repository_generation=17, profile=saved_profile)

    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        source_path = tmp_path / "clone.wav"
        source_path.write_bytes(b"RIFF")
        source = _native_profile_artifact(source_path)
        projection = replace(
            STTSPlaygroundResultProjection.from_artifact(source),
            clone_profile_save_eligible=True,
        )
        pane.current_audio_artifact = projection
        pane.current_audio_file = projection.path
        app.push_screen_wait = AsyncMock(
            return_value=profile_library_module.TTSCloneProfileSaveReview(
                display_name="Story voice",
                choose_character=True,
            )
        )
        save = AsyncMock(return_value=loaded)
        app._stts_handler = SimpleNamespace(save_current_playground_profile=save)

        await pane._save_current_result_as_profile()
        await pilot.pause()

    save.assert_awaited_once_with(projection.operation_id, "Story voice", service)
    assert len(app.navigation) == 1
    navigation = app.navigation[0]
    assert navigation.screen_name == TAB_PERSONAS
    suggestion = navigation.screen_context["voice_profile_suggestion"]
    assert suggestion.profile_id == saved_profile.profile_id
    assert suggestion.repository_generation == 17
    assert suggestion.profile_revision == 2
    assert set(navigation.screen_context) == {"view", "voice_profile_suggestion"}
    assert navigation.screen_context["view"] == "characters"


@pytest.mark.asyncio
async def test_clone_profile_save_unassigned_does_not_navigate(
    tmp_path: Path,
) -> None:
    service = _SaveProfileService()
    app = _PaneHost(profile_service=service)
    saved_profile = _saved_clone_profile(profile_id=124, name="Unassigned voice")

    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        source_path = tmp_path / "clone-unassigned.wav"
        source_path.write_bytes(b"RIFF")
        source = _native_profile_artifact(source_path)
        pane.current_audio_artifact = replace(
            STTSPlaygroundResultProjection.from_artifact(source),
            clone_profile_save_eligible=True,
        )
        pane.current_audio_file = source.path
        app.push_screen_wait = AsyncMock(
            return_value=profile_library_module.TTSCloneProfileSaveReview(
                display_name="Unassigned voice",
                choose_character=False,
            )
        )
        app._stts_handler = SimpleNamespace(
            save_current_playground_profile=AsyncMock(
                return_value=LoadedTTSProfile(17, saved_profile)
            )
        )

        await pane._save_current_result_as_profile()

    assert app.navigation == []


@pytest.mark.asyncio
async def test_clone_profile_review_completion_cannot_save_replaced_result(
    tmp_path: Path,
) -> None:
    profile_service = _SaveProfileService()
    app = _PaneHost(profile_service=profile_service)
    old_path = tmp_path / "old-clone.wav"
    new_path = tmp_path / "new-clone.wav"
    old_path.write_bytes(b"RIFF-old")
    new_path.write_bytes(b"RIFF-new")
    old = _native_profile_artifact(old_path, operation_id="old-clone")
    new = _native_profile_artifact(new_path, operation_id="new-clone")
    handler = STTSEventHandler(app)
    handler._accept_playground_artifact(old)
    app._stts_handler = handler

    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = replace(
            STTSPlaygroundResultProjection.from_artifact(old),
            clone_profile_save_eligible=True,
        )
        pane.current_audio_file = old.path

        async def replace_before_review_returns(_modal: object) -> object:
            handler._accept_playground_artifact(new)
            pane._store_delivered_artifact(new, announce=False)
            return profile_library_module.TTSCloneProfileSaveReview(
                display_name="Stale voice",
                choose_character=True,
            )

        app.push_screen_wait = replace_before_review_returns
        await pane._save_current_result_as_profile()

        assert profile_service.create_calls == []
        assert app.navigation == []
        assert pane.current_audio_artifact is not None
        assert pane.current_audio_artifact.operation_id == "new-clone"
        assert str(app.query_one("#audio-player-status", Static).render()) == (
            profile_library_module.PROFILE_ACTION_FAILED_COPY
        )


@pytest.mark.asyncio
async def test_guided_primary_starts_verifies_and_uses_existing_generation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_guided_runtime_observation())
    service.catalogs["audio_cpp"] = _guided_text_catalog()
    service.lifecycle_result_observation = _guided_runtime_observation(
        process_state="running",
        process_generation=1,
        catalog_fresh=True,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Start & Generate Sample",
        )
        app.query_one("#tts-text-input", TextArea).text = "Hello from audio.cpp."

        primary.press()
        await _wait_until(pilot, lambda: len(app.generation_events) == 1)

        request = app.generation_events[0].request
        assert service.lifecycle_calls == ["test"]
        assert request.provider_id == "audio_cpp"
        assert request.model_id == "model-a"
        assert request.text == "Hello from audio.cpp."
        assert request.response_format == "wav"
        assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_guided_sample_completion_is_fenced_after_provider_switch_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_guided_runtime_observation())
    service.catalogs["audio_cpp"] = _guided_text_catalog()
    service.lifecycle_gate = asyncio.Event()
    service.lifecycle_result_observation = _guided_runtime_observation(
        process_state="running",
        process_generation=1,
        catalog_fresh=True,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Start & Generate Sample",
        )
        app.query_one("#tts-text-input", TextArea).text = "Do not generate late."
        primary.press()
        await asyncio.wait_for(service.lifecycle_started.wait(), timeout=2.0)

        provider = app.query_one("#tts-provider-select", Select)
        provider.value = "openai"
        await _wait_until(
            pilot,
            lambda: provider.value == "openai" and str(primary.label) == "Test",
        )
        provider.value = "audio_cpp"
        await _wait_until(
            pilot,
            lambda: (
                provider.value == "audio_cpp"
                and str(primary.label) == "Starting & Generating…"
            ),
        )
        service.lifecycle_gate.set()
        await pilot.pause(0.2)

        assert service.lifecycle_calls == ["test"]
        assert app.generation_events == []


@pytest.mark.asyncio
async def test_late_guided_sample_result_cannot_overwrite_shutdown_busy_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _guided_runtime_observation(
        process_state="running",
        process_generation=1,
        catalog_fresh=True,
    )
    service = _RuntimeObservationService(running)
    service.lifecycle_gate = asyncio.Event()
    service.lifecycle_result_observation = _guided_runtime_observation()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        shutdown = app.query_one("#audio-cpp-runtime-shutdown", Button)
        await _wait_until(pilot, lambda: not shutdown.disabled)
        pane = app.query_one(SpeechPlaygroundPane)
        pane._audio_cpp_sample_state = "generating"
        pane._audio_cpp_sample_focus_target = "#audio-play-btn"
        pane._render_current_audio_cpp_observation()
        primary = app.query_one("#tts-test-connection-btn", Button)
        assert str(primary.label) == "Generating Sample…"

        shutdown.press()
        await _wait_until(pilot, lambda: service.lifecycle_calls == ["shutdown"])
        assert str(primary.label) == "Shutting down…"
        assert primary.disabled is True

        path = tmp_path / "sample.wav"
        path.write_bytes(b"RIFF")
        artifact = STTSGeneratedAudio(
            path=path,
            provider_id="audio_cpp",
            model_id="model-a",
            voice_id=None,
            source_text="sample",
            operation_id="sample-operation",
            audio_format="wav",
            content_type="audio/wav",
        )
        try:
            pane._on_generation_result(artifact)
            await pilot.pause()

            assert pane._audio_cpp_lifecycle_busy == "shutdown"
            assert str(primary.label) == "Shutting down…"
            for selector in (
                "#tts-test-connection-btn",
                "#tts-refresh-catalog-btn",
                "#tts-generate-btn",
                "#audio-cpp-runtime-restart",
                "#audio-cpp-runtime-shutdown",
            ):
                action = app.query_one(selector, Button)
                assert action.disabled is True
                assert "operation is in progress" in str(action.tooltip).lower()
            assert app.focused is not app.query_one("#audio-play-btn", Button)
        finally:
            service.lifecycle_gate.set()
        await app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_guided_primary_click_executes_visible_immutable_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_guided_runtime_observation())
    service.catalogs["audio_cpp"] = _guided_text_catalog()
    service.lifecycle_result_observation = _guided_runtime_observation(
        process_state="running",
        process_generation=1,
        catalog_fresh=True,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Start & Generate Sample",
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane._audio_cpp_runtime_observation = _runtime_observation(
            saved_generation=2,
            applied_generation=1,
            process_state="running",
            process_generation=1,
            capability="available",
            endpoint="http://127.0.0.1:19001",
        )
        app.query_one("#tts-text-input", TextArea).text = "Use visible action."

        primary.press()
        await _wait_until(pilot, lambda: len(app.generation_events) == 1)

        assert service.lifecycle_calls == ["test"]
        assert app.generation_events[0].request.model_id == "model-a"


@pytest.mark.asyncio
async def test_failed_guided_generation_projects_and_executes_retry_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RuntimeObservationService(_guided_runtime_observation())
    service.catalogs["audio_cpp"] = _guided_text_catalog()
    service.lifecycle_result_observation = _guided_runtime_observation(
        process_state="running",
        process_generation=1,
        catalog_fresh=True,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        primary = app.query_one("#tts-test-connection-btn", Button)
        await _wait_until(
            pilot,
            lambda: str(primary.label) == "Start & Generate Sample",
        )
        app.query_one("#tts-text-input", TextArea).text = "Retry this sample."
        primary.press()
        await _wait_until(pilot, lambda: len(app.generation_events) == 1)

        pane = app.query_one(SpeechPlaygroundPane)
        pane._generation_complete(None)
        await _wait_until(pilot, lambda: str(primary.label) == "Retry Sample")

        assert primary.disabled is False
        assert "another complete WAV" in str(primary.tooltip)
        primary.press()
        await _wait_until(pilot, lambda: len(app.generation_events) == 2)

        assert service.lifecycle_calls == ["test", "test"]
        assert app.generation_events[1].request.model_id == "model-a"


@pytest.mark.asyncio
async def test_mounting_external_audio_cpp_speech_lab_never_starts_managed_child(
    monkeypatch: pytest.MonkeyPatch,
    audio_cpp_playground: FakeTTSService,
) -> None:
    del audio_cpp_playground
    launches: list[str] = []

    async def forbidden_start(self, *_args, **_kwargs):
        del self
        launches.append("ensure_running")
        raise AssertionError("Speech Lab launched managed audio.cpp while mounting")

    monkeypatch.setattr(AudioCppSupervisor, "ensure_running", forbidden_start)
    app = _PaneHost(provider="audio_cpp")

    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        assert app.query_one("#speech-playground-pane", SpeechPlaygroundPane)

    assert launches == []


class _PaneScreen(Screen):
    """Hosts the pane under the real app-wide CSS bundle.

    Used by exactly one test in this file (the narrow-width save-button
    geometry check). `_PaneHost` above -- a bare `App[None]`, the same
    shape the retired widget's own `_PlaygroundHost` used -- loads no CSS
    at all. That was harmless for the widget (its layout came from an
    inline `DEFAULT_CSS` on the widget class itself), but verified
    empirically NOT harmless for `SpeechPlaygroundPane`: every axis
    `Select` reports a degenerate `Region(x=<viewport width>, width=1,
    ...)` under a bare `_PaneHost`, reproducible on a plain no-preset mount
    with nothing test-specific involved, because the pane's own CSS classes
    (`.speech-axis-control`, `.speech-pane`, ...) live in the app-wide TCSS
    bundle a real `TldwCli` loads, not on the widget. Mirrors
    `test_speech_playground_pane.py`'s own `_PaneScreen`, the pattern that
    file's geometry-sensitive tests already rely on.
    """

    def compose(self) -> ComposeResult:
        body = Vertical(
            SpeechPlaygroundPane(id="speech-playground-pane"),
            id="lab-body",
        )
        body.styles.width = "100%"
        body.styles.height = "100%"
        yield body


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", ((120, 40), (80, 24)))
async def test_runtime_card_actions_remain_scroll_reachable_at_supported_widths(
    monkeypatch: pytest.MonkeyPatch,
    terminal_size: tuple[int, int],
) -> None:
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_request_audio_cpp_runtime_observation",
        lambda self: None,
    )
    app = _build_test_app()

    async with app.run_test(size=terminal_size) as pilot:
        screen = _PaneScreen()
        await app.push_screen(screen)
        await pilot.pause()
        pane = screen.query_one(SpeechPlaygroundPane)
        card = screen.query_one("#audio-cpp-runtime-card", AudioCppRuntimeCard)
        strip = screen.query_one("#audio-cpp-runtime-actions")

        assert card.display is True
        assert (
            screen.query_one("#audio-player-container").virtual_region.y
            < card.virtual_region.y
        )
        card.scroll_visible(animate=False)
        await pilot.pause()
        escaped = [
            (str(button.label), strip.region, button.region)
            for button in strip.query(Button)
            if button.region.width and not strip.region.contains_region(button.region)
        ]

        assert not escaped, f"runtime actions clipped at {terminal_size}: {escaped}"
        assert pane.max_scroll_y >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_size", ((134, 34), (100, 30), (94, 28), (94, 22), (80, 24))
)
async def test_clone_setup_controls_and_status_remain_scroll_reachable(
    monkeypatch: pytest.MonkeyPatch,
    terminal_size: tuple[int, int],
) -> None:
    """Narrow terminals retain the setup inputs, status, and one primary action."""

    model_id = "<opaque:model>"
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=2,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
            saved_setup_source="guided",
            applied_setup_source="guided",
            saved_guided_model_ids=(model_id,),
            applied_guided_model_ids=(model_id,),
            saved_guided_default_model_id=model_id,
            applied_guided_default_model_id=model_id,
            clone_setup=_clone_setup(model_id),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _build_test_app()

    async with app.run_test(size=terminal_size) as pilot:
        screen = _PaneScreen()
        await app.push_screen(screen)
        await _wait_until(
            pilot,
            lambda: len(screen.query("#speech-clone-reference-text")) == 1,
        )
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: len(screen.query("#speech-clone-reference-text")) == 1,
        )
        pane = screen.query_one(SpeechPlaygroundPane)
        primary = screen.query_one("#tts-test-connection-btn", Button)
        assert str(primary.label) == "Create Voice & Generate"

        for selector in (
            "#tts-test-connection-btn",
            "#tts-text-input",
            "#audio-player-container",
            "#speech-clone-reference-choose",
            "#speech-clone-reference-text",
            "#speech-clone-error",
            "#speech-clone-use-profile",
        ):
            control = screen.query_one(selector)
            pane.scroll_to_widget(control, animate=False, force=True)
            await _wait_until(
                pilot,
                lambda: screen.region.overlaps(control.region),
            )
            assert control.region.width > 0, (terminal_size, selector)
            assert control.region.height > 0, (terminal_size, selector)
            assert screen.region.overlaps(control.region), (terminal_size, selector)
        assert pane.max_scroll_y > 0


@pytest.mark.asyncio
async def test_ready_clone_setup_preserves_editable_text_and_current_result_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reference setup must scroll the pane, not compress its core split away."""

    model_id = "<opaque:model>"
    service = _RuntimeObservationService(
        _runtime_observation(
            process_state="running",
            process_generation=2,
            capability="available",
            endpoint="http://127.0.0.1:19001",
            catalog_revision=11,
            catalog_fresh=True,
            saved_setup_source="guided",
            applied_setup_source="guided",
            saved_guided_model_ids=(model_id,),
            applied_guided_model_ids=(model_id,),
            saved_guided_default_model_id=model_id,
            applied_guided_default_model_id=model_id,
            clone_setup=_clone_setup(model_id),
        )
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    source = tmp_path / "reference.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 2_400)
    app = _build_test_app()

    # The real Speech Lab gives the pane about 100 cells after its catalog
    # rail. At that width the long managed provenance wraps onto four rows;
    # a full-screen 134-cell pane misses the clipping defect found in UAT.
    async with app.run_test(size=(100, 28)) as pilot:
        screen = _PaneScreen()
        await app.push_screen(screen)
        await _wait_until(pilot, lambda: len(screen.query(SpeechCloneSetup)) == 1)
        await _wait_until(
            pilot,
            lambda: len(screen.query("#speech-clone-reference-text")) == 1,
        )
        pane = screen.query_one(SpeechPlaygroundPane)
        screen.query_one("#tts-text-input", TextArea).text = "Generate speech."
        screen.query_one("#speech-clone-reference-text", TextArea).text = (
            "Exact words in the reference."
        )
        pane._handle_clone_reference_selection(source)
        await _wait_until(pilot, lambda: pane._clone_setup_canonical is not None)
        generated = _native_profile_artifact(tmp_path / "generated.wav")
        assert generated.requested_selection is not None
        generated = replace(
            generated,
            model_id="pocket-tts-english-bf16",
            metadata={"process_generation": 1},
            requested_selection=replace(
                generated.requested_selection,
                model_id="pocket-tts-english-bf16",
            ),
        )
        pane._store_delivered_artifact(generated, announce=False)
        await pilot.pause()

        text_input = screen.query_one("#tts-text-input", TextArea)
        current_result = screen.query_one("#audio-player-container")
        export_result = screen.query_one("#audio-export-btn", Button)
        save_profile = screen.query_one("#audio-save-profile-btn", Button)
        await _wait_until(
            pilot,
            lambda: save_profile.display and save_profile.region.height > 0,
        )
        split = screen.query_one(".speech-split")
        assert split.region.height >= 6
        assert text_input.region.height >= 4
        assert current_result.region.height > 0
        assert screen.region.overlaps(text_input.region)
        assert screen.region.overlaps(current_result.region)
        assert save_profile.display is True
        assert save_profile.region.height > 0
        assert not save_profile.region.overlaps(export_result.region)
        assert current_result.content_region.contains_region(save_profile.region)
        assert split.content_region.contains_region(save_profile.region)
        assert screen.region.overlaps(save_profile.region)
        assert pane.max_scroll_y > 0
        pane.post_message(
            events.MouseScrollDown(pane, 10, 20, 0, 1, 0, False, False, False)
        )
        await pilot.pause()
        assert pane.scroll_y > 0


@pytest.fixture
def audio_cpp_playground(monkeypatch: pytest.MonkeyPatch) -> FakeTTSService:
    """Point the pane's service hook at a fresh `FakeTTSService`.

    Recipe step 3/4: replaces the retired widget test's `monkeypatch.
    setattr(STTS_Window, "get_tts_service", ...)` with the pane's own
    `_tts_service_factory` hook, and its `_check_higgs_installation` patch
    the same way. `provider=` on `_PaneHost` (recipe step 1) replaces the
    widget fixture's `get_cli_setting` override that forced
    ``default_provider == "audio_cpp"``.
    """
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_check_higgs_installation",
        lambda self: None,
    )
    return service


def _patch_cli_setting(
    monkeypatch: pytest.MonkeyPatch,
    configured: dict[tuple[str, str], Any],
) -> None:
    """Patch `_cli_setting` for a test whose original `get_cli_setting`
    fixture patch configured more than `default_provider` (e.g. a
    `default_voice`) -- recipe step 2's "keep an equivalent" case.
    """

    def _cli_setting(
        self: SpeechPlaygroundPane,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:
        return configured.get((section, key), default)

    monkeypatch.setattr(SpeechPlaygroundPane, "_cli_setting", _cli_setting)


# =====================================================================
# Section A -- mount, catalog loading, voice discovery, provider switching
# =====================================================================


@pytest.mark.asyncio
async def test_mount_uses_descriptors_and_resolves_only_selected_provider(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        provider_select = app.query_one("#tts-provider-select", Select)
        assert _option_values(provider_select) == PROVIDER_IDS
        assert provider_select.value == "audio_cpp"
        assert _option_labels(provider_select)[0] == "[b]audio.cpp[/]"

        model_select = app.query_one("#tts-model-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)
        assert model_select.value == "<opaque:model>"
        assert _option_labels(model_select)[0] == "[bold red]Opaque model[/]"
        assert voice_select.value == SERVER_DEFAULT_VOICE_ID
        assert _option_values(voice_select) == (
            SERVER_DEFAULT_VOICE_ID,
            "[voice]",
            "<script>alert(1)</script>",
        )
        assert _option_labels(voice_select)[2] == "<script>alert(1)</script>"

        assert app.query_one("#tts-format-select", Select).value == "wav"
        assert app.query_one("#tts-format-select", Select).disabled is True
        assert app.query_one("#tts-speed-input", Input).value == "1.0"
        assert app.query_one("#tts-speed-input", Input).disabled is True
        restriction = app.query_one("#tts-audio-cpp-restrictions", Static)
        assert "complete wav" in str(restriction.render()).lower()

    assert service.descriptor_calls == 1
    assert service.catalog_calls == [("audio_cpp", False)]
    assert service.voice_calls == [
        ("audio_cpp", "<opaque:model>", False),
    ]
    assert service.voice_observation_calls == [
        ("audio_cpp", "<opaque:model>", False),
    ]
    assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_configuration_change_marks_catalog_stale_without_connecting(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        model_values = _option_values(app.query_one("#tts-model-select", Select))
        calls_before = list(service.catalog_calls)

        pane.current_audio_file = tmp_path / "existing.wav"
        app.query_one("#audio-play-btn", Button).disabled = False
        app.query_one("#audio-export-btn", Button).disabled = False
        service.revisions["audio_cpp"] = 2
        pane.mark_provider_configuration_changed("audio_cpp", 2)
        await pilot.pause()

        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            model_values
        )
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False
        assert (
            "refresh"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )
        assert service.catalog_calls == calls_before


@pytest.mark.asyncio
async def test_catalog_result_is_discarded_when_configuration_revision_changes(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_started = asyncio.Event()
    service.allow_catalog = asyncio.Event()
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await wait_for_signal(service.catalog_started, what="the catalog worker start")
        service.revisions["audio_cpp"] = 2
        service.allow_catalog.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            LOADING_SELECT_VALUE,
        )
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert "settings changed" in status


@pytest.mark.asyncio
async def test_exact_profile_revision_invalidated_catalog_projects_but_stays_blocked(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_started = asyncio.Event()
    service.allow_catalog = asyncio.Event()
    app = _PaneHost(preset=_profile_preset())

    async with app.run_test(size=(180, 70)) as pilot:
        await wait_for_signal(service.catalog_started, what="the catalog worker start")
        service.revisions["audio_cpp"] = 2
        service.allow_catalog.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        assert app.query_one("#tts-format-select", Select).value == "wav"
        assert app.query_one("#tts-speed-input", Input).value == "1.0"
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert (
            "settings changed"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )
        banner = app.query_one("#tts-profile-preview-status", Static)
        banner_copy = str(banner.render()).lower()
        assert "blocked" in banner_copy
        assert "refresh" in banner_copy
        assert "exact saved selection" not in banner_copy
        assert banner.has_class("profile-preview-unverified")


@pytest.mark.asyncio
async def test_superseded_catalog_failure_cannot_overwrite_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0
        newer_catalog = _audio_catalog(revision=12)

        async def get_catalog(
            provider_id: str,
            refresh: bool = False,
        ) -> TTSProviderCatalog:
            nonlocal call_count
            del refresh
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                raise RuntimeError("obsolete refresh failed")
            assert provider_id == "audio_cpp"
            service.catalogs[provider_id] = newer_catalog
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await wait_for_signal(first_started, what="the first catalog worker start")
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and pane._catalogs.get("audio_cpp") is newer_catalog
                and "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._catalogs["audio_cpp"] is newer_catalog
        assert "audio_cpp" not in pane._stale_providers
        assert (
            "ready"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )


@pytest.mark.asyncio
async def test_superseded_catalog_success_cannot_invalidate_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0
        older_catalog = _audio_catalog(revision=11)
        newer_catalog = _audio_catalog(revision=12)

        async def get_catalog(
            provider_id: str,
            refresh: bool = False,
        ) -> TTSProviderCatalog:
            nonlocal call_count
            del refresh
            assert provider_id == "audio_cpp"
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                return older_catalog
            service.catalogs[provider_id] = newer_catalog
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await wait_for_signal(first_started, what="the first catalog worker start")
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and pane._catalogs.get("audio_cpp") is newer_catalog
                and "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._catalogs["audio_cpp"] is newer_catalog
        assert "audio_cpp" not in pane._stale_providers
        assert pane._catalog_generation_allowed is True
        assert (
            "ready"
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("obsolete_fails", (False, True), ids=("success", "failure"))
async def test_superseded_same_model_voice_result_cannot_overwrite_newer_success(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    obsolete_fails: bool,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0
        model_id = "<opaque:model>"
        catalog_revision = service.catalogs["audio_cpp"].revision

        async def get_voices(
            provider_id: str,
            requested_model_id: str,
            refresh: bool = False,
        ) -> tuple[str, ...]:
            nonlocal call_count
            del refresh
            assert (provider_id, requested_model_id) == ("audio_cpp", model_id)
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
                if obsolete_fails:
                    raise RuntimeError("obsolete voice request failed")
                return ("obsolete-voice",)
            return ("new-voice",)

        monkeypatch.setattr(service, "get_voices", get_voices)

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await wait_for_signal(first_started, what="the first voice worker start")
        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await _wait_until(
            pilot,
            lambda: (
                call_count == 2
                and pane._discovered_voices.get(("audio_cpp", model_id))
                == ("new-voice",)
            ),
        )

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._discovered_voices[("audio_cpp", model_id)] == ("new-voice",)
        assert _option_values(app.query_one("#tts-voice-select", Select)) == (
            SERVER_DEFAULT_VOICE_ID,
            "new-voice",
        )
        assert (
            "voices are unavailable"
            not in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )


@pytest.mark.asyncio
async def test_catalog_generation_is_reserved_before_exclusive_worker_cancellation(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        baseline_catalog = pane._catalogs["audio_cpp"]
        first_started = asyncio.Event()
        first_returned_on_cancel = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        call_count = 0
        obsolete_catalog = _audio_catalog(revision=10)
        newer_catalog = _audio_catalog(revision=12)

        async def get_catalog(
            provider_id: str,
            refresh: bool = False,
        ) -> TTSProviderCatalog:
            nonlocal call_count
            del refresh
            assert provider_id == "audio_cpp"
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_returned_on_cancel.set()
                    return obsolete_catalog
            second_started.set()
            await release_second.wait()
            service.catalogs[provider_id] = newer_catalog
            return newer_catalog

        monkeypatch.setattr(service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await wait_for_signal(first_started, what="the first catalog worker start")
        first_generation = pane._catalog_request_generations["audio_cpp"]

        pane._load_provider_catalog("audio_cpp", refresh=True)

        assert pane._catalog_request_generations["audio_cpp"] == (first_generation + 1)
        await wait_for_signal(
            first_returned_on_cancel,
            what="the cancelled first catalog worker returning",
        )
        await wait_for_signal(second_started, what="the second catalog worker start")
        await pilot.pause()
        assert pane._catalogs["audio_cpp"] is baseline_catalog

        release_second.set()
        await app.workers.wait_for_complete()
        assert pane._catalogs["audio_cpp"] is newer_catalog


@pytest.mark.asyncio
async def test_voice_generation_is_reserved_before_exclusive_worker_cancellation(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        model_id = "<opaque:model>"
        request_key = ("audio_cpp", model_id)
        baseline_voices = pane._discovered_voices[request_key]
        catalog_revision = service.catalogs["audio_cpp"].revision
        first_started = asyncio.Event()
        first_returned_on_cancel = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        call_count = 0

        async def get_voices(
            provider_id: str,
            requested_model_id: str,
            refresh: bool = False,
        ) -> tuple[str, ...]:
            nonlocal call_count
            del refresh
            assert (provider_id, requested_model_id) == request_key
            call_count += 1
            if call_count == 1:
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_returned_on_cancel.set()
                    return ("obsolete-on-cancel",)
            second_started.set()
            await release_second.wait()
            return ("new-voice",)

        monkeypatch.setattr(service, "get_voices", get_voices)

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )
        await wait_for_signal(first_started, what="the first voice worker start")
        first_generation = pane._voice_request_generations[request_key]

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog_revision,
            refresh=True,
        )

        assert pane._voice_request_generations[request_key] == first_generation + 1
        await wait_for_signal(
            first_returned_on_cancel,
            what="the cancelled first voice worker returning",
        )
        await wait_for_signal(second_started, what="the second voice worker start")
        await pilot.pause()
        assert pane._discovered_voices[request_key] == baseline_voices

        release_second.set()
        await app.workers.wait_for_complete()
        assert pane._discovered_voices[request_key] == ("new-voice",)


@pytest.mark.asyncio
async def test_voice_discovery_does_not_cancel_inflight_catalog_refresh(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        service.catalog_started = asyncio.Event()
        service.allow_catalog = asyncio.Event()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await wait_for_signal(service.catalog_started, what="the catalog worker start")
        app.query_one("#tts-model-select", Select).value = "second-model"
        await _wait_until(
            pilot,
            lambda: any(
                provider_id == "audio_cpp" and model_id == "second-model"
                for provider_id, model_id, _refresh in service.voice_calls
            ),
        )

        assert service.catalog_cancelled is False
        service.allow_catalog.set()
        await _wait_until(
            pilot,
            lambda: (
                "ready"
                in str(app.query_one("#tts-provider-status", Static).render()).lower()
            ),
        )


@pytest.mark.asyncio
async def test_catalog_revision_invalidates_old_voices_before_rediscovery(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "pending voice"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voice_started = asyncio.Event()
        service.allow_voices = asyncio.Event()
        pane = app.query_one(SpeechPlaygroundPane)
        notices_before = list(app.notices)
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await wait_for_signal(service.voice_started, what="the voice worker start")
        await pilot.pause()

        assert _option_values(voice_select) == (
            SERVER_DEFAULT_VOICE_ID,
            "[voice]",
        )
        assert voice_select.value == "[voice]"
        assert app.notices == notices_before
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        pending_notices = [
            *notices_before,
            ("Voices are still loading; wait before generating", "warning"),
        ]
        assert app.notices == pending_notices

        service.allow_voices.set()
        await app.workers.wait_for_complete()

        assert voice_select.value == "[voice]"
        assert app.notices == pending_notices
        assert app.query_one("#tts-generate-btn", Button).disabled is False


@pytest.mark.asyncio
async def test_catalog_revision_preserves_exact_voice_removed_by_refresh(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voices[("audio_cpp", "<opaque:model>")] = ("replacement",)
        service.voice_started = asyncio.Event()
        service.allow_voices = asyncio.Event()
        notices_before = list(app.notices)
        pane = app.query_one(SpeechPlaygroundPane)
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await wait_for_signal(service.voice_started, what="the voice worker start")
        await pilot.pause()

        assert app.notices == notices_before

        service.allow_voices.set()
        await app.workers.wait_for_complete()

        assert voice_select.value == "[voice]"
        assert "[voice]" in _option_values(voice_select)
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.notices == notices_before


@pytest.mark.asyncio
async def test_voice_discovery_failure_preserves_pending_explicit_voice(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "fallback voice"
        await pilot.pause()

        service.catalogs["audio_cpp"] = _audio_catalog(revision=12)
        service.voice_error = RuntimeError("untrusted upstream detail")
        app.query_one(SpeechPlaygroundPane)._load_provider_catalog(
            "audio_cpp",
            refresh=True,
        )
        await app.workers.wait_for_complete()

        assert voice_select.value == "[voice]"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            str(app.query_one("#tts-provider-status", Static).render())
            == "Voices are unavailable; the exact selection remains unverified"
        )

        app.query_one(SpeechPlaygroundPane).action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert "untrusted upstream detail" not in str(app.notices)


@pytest.mark.asyncio
async def test_voice_discovery_failure_preserves_configured_explicit_default(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = audio_cpp_playground
    _patch_cli_setting(
        monkeypatch,
        {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        },
    )
    service.voice_error = RuntimeError("untrusted upstream detail")
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-text-input", TextArea).text = "configured fallback"
        await pilot.pause()

        pane = app.query_one(SpeechPlaygroundPane)
        assert app.query_one("#tts-voice-select", Select).value == "[voice]"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            str(app.query_one("#tts-provider-status", Static).render())
            == "Voices are unavailable; the exact selection remains unverified"
        )

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert "untrusted upstream detail" not in str(app.notices)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lifecycle_error", "expected_status"),
    (
        (
            TTSProviderReconfiguringError("PRIVATE_RECONFIGURING"),
            "settings are being applied",
        ),
        (
            TTSRegistryClosedError("PRIVATE_REGISTRY_CLOSED"),
            "tts service is unavailable",
        ),
    ),
)
async def test_voice_discovery_lifecycle_failure_preserves_pending_selection(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_error: Exception,
    expected_status: str,
) -> None:
    service = audio_cpp_playground
    _patch_cli_setting(
        monkeypatch,
        {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        },
    )
    service.voice_error = lifecycle_error
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-text-input", TextArea).text = "lifecycle pending"
        await pilot.pause()

        pane = app.query_one(SpeechPlaygroundPane)
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert pane._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert expected_status in status

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert "PRIVATE_" not in str(app.notices)
        assert "PRIVATE_" not in status


@pytest.mark.asyncio
async def test_server_default_override_survives_provider_switch(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del audio_cpp_playground
    _patch_cli_setting(
        monkeypatch,
        {
            ("app_tts", "default_provider"): "audio_cpp",
            ("app_tts", "default_voice"): "[voice]",
        },
    )
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        provider_select = app.query_one("#tts-provider-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)
        assert voice_select.value == "[voice]"

        voice_select.value = SERVER_DEFAULT_VOICE_ID
        provider_select.value = "openai"
        await _wait_until(
            pilot,
            lambda: app.query_one("#tts-model-select", Select).value == "tts-1",
        )
        provider_select.value = "audio_cpp"
        await _wait_until(
            pilot,
            lambda: (
                app.query_one("#tts-model-select", Select).value == "<opaque:model>"
            ),
        )

        assert voice_select.value == SERVER_DEFAULT_VOICE_ID


@pytest.mark.asyncio
async def test_stale_model_voice_failure_cannot_release_current_pending_voice(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        voice_select = app.query_one("#tts-voice-select", Select)
        model_select = app.query_one("#tts-model-select", Select)
        voice_select.value = "[voice]"
        app.query_one("#tts-text-input", TextArea).text = "current pending voice"
        await pilot.pause()

        old_key = ("audio_cpp", "<opaque:model>")
        current_key = ("audio_cpp", "second-model")
        service.voice_started_by_request[old_key] = asyncio.Event()
        service.voice_finished_by_request[old_key] = asyncio.Event()
        service.voice_gates[old_key] = asyncio.Event()
        service.voice_errors[old_key] = RuntimeError("late old-model failure")
        service.voice_ignore_cancellation.add(old_key)
        service.voice_started_by_request[current_key] = asyncio.Event()
        service.voice_finished_by_request[current_key] = asyncio.Event()
        service.voice_gates[current_key] = asyncio.Event()

        pane._load_provider_voices(
            "audio_cpp",
            "<opaque:model>",
            service.catalogs["audio_cpp"].revision,
            refresh=True,
        )
        await service.voice_started_by_request[old_key].wait()

        model_select.value = "second-model"
        await service.voice_started_by_request[current_key].wait()
        await pilot.pause()

        assert pane._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        service.voice_gates[old_key].set()
        await service.voice_finished_by_request[old_key].wait()
        await pilot.pause()

        assert pane._pending_voice_selections == {"audio_cpp": "[voice]"}
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        pane.action_generate_tts()
        await pilot.pause()
        assert app.generation_events == []
        assert app.notices[-1] == (
            "Voices are still loading; wait before generating",
            "warning",
        )

        service.voice_gates[current_key].set()
        await service.voice_finished_by_request[current_key].wait()
        await _wait_until(pilot, lambda: pane._pending_voice_selections == {})


@pytest.mark.asyncio
async def test_legacy_control_state_is_restored_after_audio_cpp_switch(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        provider = app.query_one("#tts-provider-select", Select)
        provider.value = "openai"
        await _wait_until(
            pilot,
            lambda: app.query_one("#tts-model-select", Select).value == "tts-1",
        )

        model = app.query_one("#tts-model-select", Select)
        voice = app.query_one("#tts-voice-select", Select)
        response_format = app.query_one("#tts-format-select", Select)
        speed = app.query_one("#tts-speed-input", Input)
        model.value = "tts-1-hd"
        voice.value = "nova"
        response_format.value = "flac"
        speed.value = "1.35"
        await pilot.pause()

        provider.value = "audio_cpp"
        await _wait_until(pilot, lambda: response_format.disabled)
        assert response_format.value == "wav"
        assert response_format.disabled is True
        assert speed.value == "1.0"
        assert speed.disabled is True

        provider.value = "openai"
        await _wait_until(
            pilot,
            lambda: model.value == "tts-1-hd" and not response_format.disabled,
        )
        assert model.value == "tts-1-hd"
        assert voice.value == "nova"
        assert response_format.value == "flac"
        assert response_format.disabled is False
        assert speed.value == "1.35"
        assert speed.disabled is False

    assert [provider_id for provider_id, _refresh in service.catalog_calls] == [
        "audio_cpp",
        "openai",
        "audio_cpp",
        "openai",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "provider_id",
        "model_id",
        "model_label",
        "voice_id",
        "voice_label",
    ),
    (
        ("openai", "tts-1", "TTS-1 (Standard)", "alloy", "Alloy"),
        (
            "elevenlabs",
            "eleven_multilingual_v2",
            "Eleven Multilingual v2 (Default)",
            "21m00Tcm4TlvDq8ikWAM",
            "Rachel",
        ),
        ("kokoro", "kokoro", "Kokoro 82M", "af_alloy", "Alloy (US Female)"),
        (
            "chatterbox",
            "chatterbox",
            "Chatterbox 0.5B",
            "default",
            "Default Voice",
        ),
        (
            "higgs",
            "higgs-audio-v2",
            "Higgs Audio V2 3B",
            "professional_female",
            "Professional Female",
        ),
        ("alltalk", "alltalk", "AllTalk TTS", "female_01.wav", "Female 01"),
    ),
)
async def test_legacy_provider_defaults_and_labels_are_preserved(
    audio_cpp_playground: FakeTTSService,
    provider_id: str,
    model_id: str,
    model_label: str,
    voice_id: str,
    voice_label: str,
) -> None:
    del audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-provider-select", Select).value = provider_id
        await _wait_until(
            pilot,
            lambda: app.query_one("#tts-model-select", Select).value == model_id,
        )

        model_select = app.query_one("#tts-model-select", Select)
        voice_select = app.query_one("#tts-voice-select", Select)
        assert model_select.value == model_id
        assert _label_for_value(model_select, model_id) == model_label
        assert voice_select.value == voice_id
        assert _label_for_value(voice_select, voice_id) == voice_label


@pytest.mark.asyncio
async def test_higgs_saved_profile_is_prefixed_exactly_once_in_request(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del audio_cpp_playground
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_higgs_profile_choices",
        staticmethod(lambda: [("Saved voice", "profile:saved-voice")]),
    )
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.query_one("#tts-provider-select", Select).value = "higgs"
        await _wait_until(
            pilot,
            lambda: (
                app.query_one("#tts-model-select", Select).value == "higgs-audio-v2"
            ),
        )
        app.query_one("#tts-voice-select", Select).value = "profile:saved-voice"
        app.query_one("#tts-text-input", TextArea).text = "use saved profile"
        await pilot.pause()

        app.query_one(SpeechPlaygroundPane)._generate_tts()
        await pilot.pause()

        assert len(app.generation_events) == 1
        assert app.generation_events[0].request.voice_id == "profile:saved-voice"


#: TASK-2970 (fixed): `_load_provider_catalog_worker` used to add a
#: provider to `_stale_providers` whenever the *freshly-fetched* catalog
#: itself reported `health.fresh is False`, unconditionally --
#:
#:     if catalog.health.fresh:
#:         self._stale_providers.discard(provider_id)
#:     else:
#:         self._stale_providers.add(provider_id)
#:
#: -- and `_catalog_health_copy` checks `_stale_providers` FIRST, ahead of
#: `health.state`, so any fresh (first-ever) load whose health happened to
#: report `fresh=False` (a "reconfiguring" provider, or a naturally stale
#: "available" catalog) was shown the generic "settings changed; refresh
#: models" instead of the accurate, state-specific recovery copy -- even
#: though nothing had changed; this was simply how the very first read
#: came back. Confirmed NOT present in the retired widget: its equivalent
#: success path was an unconditional `self._stale_providers.discard(
#: provider_id)` with no `health.fresh` branch at all (verified by diffing
#: `git show HEAD:tldw_chatbook/UI/STTS_Window.py` against the pane before
#: TASK-2951 deleted the widget). Fixed by only adding to `_stale_
#: providers` when this load's own configuration revision genuinely
#: supersedes a previously recorded one for the same provider -- a real
#: configuration change this particular fetch has not caught up to yet --
#: not merely because this load's health happens to read non-fresh with no
#: revision history behind it. The two health2/health3 parametrizations
#: below were `xfail(strict=True)` against the pre-fix code; both now pass.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("health", "expected_copy"),
    (
        (
            ProviderHealth(state="unavailable", fresh=True),
            "unavailable",
        ),
        (
            ProviderHealth(state="not_configured", fresh=True),
            "not configured",
        ),
        (
            ProviderHealth(state="reconfiguring", fresh=False),
            "settings are being applied",
        ),
        (
            ProviderHealth(state="available", fresh=False),
            "catalog is stale",
        ),
    ),
)
async def test_audio_cpp_health_states_use_fixed_safe_recovery_copy(
    audio_cpp_playground: FakeTTSService,
    health: ProviderHealth,
    expected_copy: str,
) -> None:
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(health=health)
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert expected_copy in status
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#tts-refresh-catalog-btn", Button).disabled is False
        assert app.query_one("#tts-provider-select", Select).value == "audio_cpp"
        assert _option_values(app.query_one("#tts-model-select", Select)) == (
            "<opaque:model>",
            "second-model",
        )
        assert service.catalog_calls == [("audio_cpp", False)]


# TASK-2970 positive branch: health2/health3 above pin the negative case (a
# first-ever non-fresh load must NOT be marked stale). This test pins the
# other half of AC#3 -- a SECOND load whose own configuration revision has
# genuinely moved past the one recorded from the first is a real
# supersession, and must still be marked stale with the settings-changed
# copy. Mutation-checked by disabling the `elif` branch in
# `_load_provider_catalog_worker` (`speech_catalog_mixin.py`, the genuine-
# supersession check) so it always falls through to `discard` -- this test
# alone goes red; health2/health3 and the TASK-3000 tests stay green.
@pytest.mark.asyncio
async def test_second_load_after_genuine_config_change_marks_provider_stale(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert "audio_cpp" not in pane._stale_providers
        assert pane._catalog_configuration_revisions["audio_cpp"] == 1

        # A genuine configuration-revision bump, then a second, successful
        # (token-current) reload whose catalog still reports non-fresh
        # health -- unlike the first-ever load above, this one really did
        # follow a real config change.
        service.revisions["audio_cpp"] = 2
        service.catalogs["audio_cpp"] = _audio_catalog(
            health=ProviderHealth(state="available", fresh=False)
        )
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._catalog_configuration_revisions["audio_cpp"] == 2
        assert "audio_cpp" in pane._stale_providers
        status = str(app.query_one("#tts-provider-status", Static).render()).lower()
        assert "settings changed" in status


# =====================================================================
# Section B -- playback, export, unmount cleanup, mount rehydration
# =====================================================================


@pytest.mark.asyncio
async def test_playback_uses_dedicated_widget_worker_group(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    path = tmp_path / "artifact.wav"
    path.write_bytes(b"RIFF")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        source_text="source",
        operation_id="operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    captured: dict[str, object] = {}
    worker_calls = 0

    def run_worker(
        _self: SpeechPlaygroundPane,
        awaitable: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        nonlocal worker_calls
        worker_calls += 1
        captured.update(kwargs)
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return SimpleNamespace(is_finished=False, cancel=lambda: None)

    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(SpeechPlaygroundPane, "run_worker", run_worker)
        monkeypatch.setattr(
            SpeechPlaygroundPane,
            "_ensure_audio_player",
            lambda _self: True,
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = artifact
        pane.current_audio_file = artifact.path

        pane._play_audio()
        pane._play_audio()

        assert captured["group"] == "stts-playback"
        assert captured["exclusive"] is True
        assert worker_calls == 1


@pytest.mark.asyncio
async def test_playback_uses_artifact_captured_before_worker_runs(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id=None,
        source_text="old",
        operation_id="old-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="new-model",
        voice_id=None,
        source_text="new",
        operation_id="new-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    jobs: list[object] = []

    def run_worker(
        _self: SpeechPlaygroundPane,
        job: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        jobs.append(job)
        return SimpleNamespace(is_finished=False, cancel=lambda: None)

    class Player:
        def __init__(self) -> None:
            self.played: list[Path] = []

        async def get_state(self) -> str:
            return "stopped"

        async def stop(self) -> bool:
            return True

        async def play(self, path: Path) -> bool:
            self.played.append(path)
            return False

    lease_handler = SimpleNamespace(
        lease_playground_result=Mock(return_value=True),
        release_playground_result=Mock(),
    )
    app = _PaneHost()
    app._stts_handler = lease_handler
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(SpeechPlaygroundPane, "run_worker", run_worker)
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = STTSPlaygroundResultProjection.from_artifact(
            old_artifact
        )
        pane.current_audio_file = old_path

        pane._play_audio()
        pane.current_audio_artifact = STTSPlaygroundResultProjection.from_artifact(
            new_artifact
        )
        pane.current_audio_file = new_path

        job = jobs[0]
        if callable(job):
            await job()
        else:
            await job  # type: ignore[misc]

        assert player.played == [old_path]
        lease_handler.lease_playground_result.assert_called_once_with(
            old_artifact.operation_id,
            old_artifact.path,
        )
        lease_handler.release_playground_result.assert_called_once_with(
            old_artifact.operation_id,
            old_artifact.path,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pause_before_replacement", "terminal_action"),
    (
        (False, "stop"),
        (True, "stop"),
        (False, "finish"),
        (False, "unmount"),
    ),
    ids=("playing-stop", "paused-stop", "natural-finish", "unmount"),
)
async def test_playback_lease_survives_replacement_until_playback_ends(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
    pause_before_replacement: bool,
    terminal_action: str,
) -> None:
    del audio_cpp_playground
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id=None,
        source_text="old",
        operation_id="old-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="new-model",
        voice_id=None,
        source_text="new",
        operation_id="new-operation",
        audio_format="wav",
        content_type="audio/wav",
    )

    class Player:
        def __init__(self) -> None:
            self.state = PlaybackState.IDLE

        async def get_state(self) -> PlaybackState:
            return self.state

        async def stop(self) -> bool:
            self.state = PlaybackState.IDLE
            return True

        async def play(self, _path: Path) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def is_playing(self) -> bool:
            return self.state == PlaybackState.PLAYING

        async def pause(self) -> bool:
            self.state = PlaybackState.PAUSED
            return True

        async def resume(self) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def get_position(self) -> float:
            return 0.0

        async def get_duration(self) -> float:
            return 1.0

    app = _PaneHost()
    handler = STTSEventHandler(app)
    handler._accept_playground_artifact(old_artifact)
    app._stts_handler = handler
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._play_audio()
        await _wait_until(
            pilot,
            lambda: (
                player.state == PlaybackState.PLAYING and pane._play_worker_task is None
            ),
        )
        if pause_before_replacement:
            pane._pause_audio()
            await _wait_until(
                pilot,
                lambda: player.state == PlaybackState.PAUSED,
            )

        handler._accept_playground_artifact(new_artifact)
        pane._store_delivered_artifact(new_artifact, announce=False)

        assert old_path.exists()
        assert handler._playground_file_leases[old_path] == 1

        if terminal_action == "stop":
            await pane._stop_audio_async()
        elif terminal_action == "finish":
            player.state = PlaybackState.FINISHED
            await _wait_until(pilot, lambda: not old_path.exists())

    assert not old_path.exists()
    assert old_path not in handler._playground_file_leases


@pytest.mark.asyncio
async def test_export_uses_artifact_captured_before_dialog_completes(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    destination = tmp_path / "export.wav"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id=None,
        source_text="old",
        operation_id="old-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="new-model",
        voice_id=None,
        source_text="new",
        operation_id="new-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    callbacks: list[Callable[[str | None], None]] = []
    lease_handler = SimpleNamespace(
        lease_playground_result=Mock(return_value=True),
        release_playground_result=Mock(),
    )
    app = _PaneHost()
    app._stts_handler = lease_handler

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = STTSPlaygroundResultProjection.from_artifact(
            old_artifact
        )
        pane.current_audio_file = old_path

        pane._export_audio()
        pane.current_audio_artifact = STTSPlaygroundResultProjection.from_artifact(
            new_artifact
        )
        pane.current_audio_file = new_path
        callbacks[0](str(destination))

        assert destination.read_bytes() == b"old"
        lease_handler.lease_playground_result.assert_called_once_with(
            old_artifact.operation_id,
            old_artifact.path,
        )
        lease_handler.release_playground_result.assert_called_once_with(
            old_artifact.operation_id,
            old_artifact.path,
        )


@pytest.mark.asyncio
async def test_export_cancel_releases_captured_artifact(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    path = tmp_path / "artifact.wav"
    path.write_bytes(b"audio")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="model",
        voice_id=None,
        source_text="source",
        operation_id="operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    callbacks: list[Callable[[str | None], None]] = []
    lease_handler = SimpleNamespace(
        lease_playground_result=Mock(return_value=True),
        release_playground_result=Mock(),
    )
    app = _PaneHost()
    app._stts_handler = lease_handler

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda _screen, callback: callbacks.append(callback),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        pane.current_audio_artifact = STTSPlaygroundResultProjection.from_artifact(
            artifact
        )
        pane.current_audio_file = path

        pane._export_audio()
        callbacks[0](None)

        lease_handler.release_playground_result.assert_called_once_with(
            artifact.operation_id,
            artifact.path,
        )


@pytest.mark.asyncio
async def test_audio_export_rejects_unsafe_dialog_destination(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    source = tmp_path / "source.wav"
    unsafe_destination = tmp_path / "bad;name.wav"
    source.write_bytes(b"audio")
    app = _PaneHost()

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._handle_audio_export(
            str(unsafe_destination),
            source_path=source,
        )

    assert not unsafe_destination.exists()
    assert app.notices[-1][1] == "error"
    assert "dangerous pattern" in app.notices[-1][0]


@pytest.mark.asyncio
async def test_unmount_cancels_only_widget_owned_worker_groups(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del audio_cpp_playground
    app = _PaneHost()
    cleanup = Mock()
    app._stts_handler = SimpleNamespace(cleanup_tts_resources=cleanup)
    cancelled_groups: list[str] = []

    async with app.run_test(size=(180, 70)):
        original_cancel_group = app.workers.cancel_group

        def cancel_group(node: object, group: str) -> None:
            cancelled_groups.append(group)
            original_cancel_group(node, group)

        monkeypatch.setattr(app.workers, "cancel_group", cancel_group)
        await app.workers.wait_for_complete()

    assert {
        "stts-catalog-discovery",
        "stts-voice-discovery",
        "stts-playback",
    }.issubset(cancelled_groups)
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_new_mount_rehydrates_handler_owned_artifact(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    path = tmp_path / "retained.wav"
    path.write_bytes(b"RIFF")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="response-model",
        voice_id=None,
        source_text="source",
        operation_id="completed-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    state = SimpleNamespace(
        active_operation_id=None,
        artifact=STTSPlaygroundResultProjection.from_artifact(artifact),
        generation_active=False,
    )
    cleanup = Mock()
    app = _PaneHost()
    app._stts_handler = SimpleNamespace(
        playground_state=lambda: state,
        cleanup_tts_resources=cleanup,
    )

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert type(pane.current_audio_artifact) is STTSPlaygroundResultProjection
        assert pane.current_audio_artifact.operation_id == artifact.operation_id
        assert pane.current_audio_file == path
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False

    assert path.exists()
    cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_new_mount_rehydrates_active_generation_without_starting_another(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    state = SimpleNamespace(
        active_operation_id="active-operation",
        artifact=None,
        generation_active=True,
    )
    app = _PaneHost()
    app._stts_handler = SimpleNamespace(playground_state=lambda: state)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert pane._generation_operation_id == "active-operation"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert (
            "in progress"
            in str(app.query_one("#generation-status-text", Static).render()).lower()
        )
        assert service.synthesize_calls == 0


# =====================================================================
# Section C -- exact profile preview and adoption
# =====================================================================


@pytest.mark.asyncio
async def test_fresh_catalog_missing_exact_profile_stays_visible_but_unavailable(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    preset = _profile_preset()
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-provider-select", Select).value == "audio_cpp"
        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        assert app.query_one("#tts-profile-preview-status", Static).has_class(
            "profile-preview-unavailable"
        )
        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert pane._profile_preset is preset
        assert app.generation_events == []

    assert service.voice_calls == []
    assert service.synthesize_calls == 0


@pytest.mark.asyncio
async def test_complete_voice_observation_missing_exact_voice_blocks_profile(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    preset = _profile_preset(
        model_id="<opaque:model>",
        voice_id="missing-profile-voice",
    )
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#tts-profile-preview-status", Static).has_class(
            "profile-preview-unavailable"
        )

    assert service.voice_calls == [("audio_cpp", "<opaque:model>", False)]


# --------------------------------------------------------------------------
# TASK-2951 re-review: the readiness-gate unification.
#
# Both tests below were originally classified as "RED against the dead
# widget, proves nothing about live code" and dropped without porting. That
# classification was wrong for these two specifically: they describe real
# guarantees `SpeechPlaygroundPane` did not (yet) have, not dead-widget-only
# behavior. `action_generate_tts()` -- unreachable from the keyboard before
# this same branch's `_playground()` fix (TASK-2951 AC#2), since the screen-
# level mirror always found nothing to call it on -- is now reachable, which
# is what makes the gap below a live bypass rather than a theoretical one.
#
# Root cause: `_sync_generate_enabled` (the button's visual state) was
# profile-preset-aware; `_generation_readiness_error` (what `_generate_tts`
# actually consults before firing) was not -- it never had the profile
# branch the retired widget's own version carried (verified via `git show
# HEAD:tldw_chatbook/UI/STTS_Window.py` against the version that predates
# this branch's widget deletion). Fixed by making `_sync_generate_enabled`
# derive the button's disabled state from `_generation_readiness_error`
# instead of re-deriving its own answer, so the two can no longer disagree.
@pytest.mark.asyncio
async def test_exact_profile_cannot_generate_while_voice_validation_is_pending(
    audio_cpp_playground: FakeTTSService,
) -> None:
    """A pending voice-validation token must block generation -- not just
    the button's visual state, but a direct `action_generate_tts()` call
    too, which is exactly how the keyboard `g` mirror (STTSScreen ->
    `_playground()` -> `SpeechPlaygroundPane`) reaches it."""
    service = audio_cpp_playground
    service.voice_started = asyncio.Event()
    service.allow_voices = asyncio.Event()
    preset = _profile_preset(
        model_id="<opaque:model>",
        voice_id="missing-profile-voice",
    )
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await wait_for_signal(service.voice_started, what="the voice worker start")
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-generate-btn", Button).disabled is True
        assert app.query_one("#tts-profile-preview-status", Static).has_class(
            "profile-preview-loading"
        )

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == [], (
            "action_generate_tts() bypassed the disabled Generate button "
            "while the exact profile's voice validation was still pending"
        )
        assert app.notices[-1] == (
            "The exact profile voice is still being checked; wait before generating",
            "warning",
        )

        service.allow_voices.set()
        await app.workers.wait_for_complete()

        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-generate-btn", Button).disabled is True


# TASK-3000 (fixed): a config change arriving while an exact profile's
# voice validation is in flight, on a request that ignores cancellation
# and keeps running, never cleared `_profile_voice_validation_token` --
# not in `mark_provider_configuration_changed`, not in the failed-reload
# exception path, not in `_catalog_failure`. `_generation_readiness_
# error`'s preset branch blocks unconditionally on that token being
# non-None, so Generate stayed disabled forever with no way to recover
# short of leaving and re-entering the Playground. Confirmed pre-existing
# by task-2951's own re-review, identical against pre- and post-fix
# task-2951 code -- a fail-*closed* gap, the opposite direction from the
# two fail-open CRITICALs that task fixed. The retired widget's own
# `mark_provider_configuration_changed` (`git show
# f560217fb~1:tldw_chatbook/UI/STTS_Window.py`) detached the token
# immediately, as soon as the change targeted the token's own provider,
# before any reload even ran -- ported verbatim rather than inventing a
# parallel settling path; the existing `_catalog_failure` preset branch
# (task-2951 CRITICAL-2) already settles availability to "unverified" and
# re-enables Generate for one warned exact attempt once the token is out
# of the way.
@pytest.mark.asyncio
async def test_configuration_change_detaches_cancellation_resistant_profile_voice_gate(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    request_key = ("audio_cpp", "<opaque:model>")
    service.voice_started_by_request[request_key] = asyncio.Event()
    service.voice_finished_by_request[request_key] = asyncio.Event()
    service.voice_gates[request_key] = asyncio.Event()
    service.voice_ignore_cancellation.add(request_key)
    preset = _profile_preset(model_id=request_key[1], voice_id="[voice]")
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await service.voice_started_by_request[request_key].wait()
        pane = app.query_one(SpeechPlaygroundPane)

        service.revisions["audio_cpp"] = 2
        pane.mark_provider_configuration_changed("audio_cpp", 2)
        service.catalog_error = RuntimeError("private catalog failure")
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(
            pilot,
            lambda: (
                pane._profile_effective_availability == "unverified"
                and pane._catalog_generation_allowed
            ),
        )

        assert pane._profile_voice_validation_token is None
        assert app.query_one("#tts-generate-btn", Button).disabled is False

        pane.action_generate_tts()
        await pilot.pause()

        assert len(app.generation_events) == 1
        assert any(
            "unverified" in message.lower() and severity == "warning"
            for message, severity in app.notices
        )

        service.voice_gates[request_key].set()
        await service.voice_finished_by_request[request_key].wait()


# TASK-3000 regression guard (a): the token must detach ONLY on a genuine,
# provider-matched `mark_provider_configuration_changed` call -- not on any
# catalog-load failure whatsoever. A plain, unrelated catalog reload failure
# while a voice validation is in flight (no config change anywhere in this
# sequence) must leave the token, and Generate's disabled state, untouched.
# Mutation-checked: widening `_load_provider_catalog_worker`'s `except
# Exception` handler (speech_catalog_mixin.py ~:424-426) to clear
# `self._profile_voice_validation_token` unconditionally, instead of only
# its own locally-reserved `profile_voice_token`, turns this test red.
@pytest.mark.asyncio
async def test_unrelated_catalog_failure_does_not_detach_in_flight_profile_voice_token(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    request_key = ("audio_cpp", "<opaque:model>")
    service.voice_started_by_request[request_key] = asyncio.Event()
    service.voice_finished_by_request[request_key] = asyncio.Event()
    service.voice_gates[request_key] = asyncio.Event()
    preset = _profile_preset(model_id=request_key[1], voice_id="[voice]")
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await service.voice_started_by_request[request_key].wait()
        pane = app.query_one(SpeechPlaygroundPane)
        pending_token = pane._profile_voice_validation_token
        assert pending_token is not None
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        # No configuration change anywhere in this sequence -- just an
        # unrelated catalog reload that fails for its own reasons.
        service.catalog_error = RuntimeError("unrelated catalog failure")
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await _wait_until(pilot, lambda: "audio_cpp" in pane._stale_providers)

        assert pane._profile_voice_validation_token is pending_token
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        service.voice_gates[request_key].set()
        await service.voice_finished_by_request[request_key].wait()


# TASK-3000 regression guard (b): `mark_provider_configuration_changed`'s
# token-detach only fires when the changed provider matches the token's OWN
# provider. A config change for a completely different provider must leave
# an audio_cpp exact profile's in-flight voice-validation token (and
# Generate's disabled state) untouched. Mutation-checked: removing the
# `pending_voice_token.provider_id == provider_id` guard
# (speech_catalog_mixin.py ~:1527-1532) turns this test red.
@pytest.mark.asyncio
async def test_configuration_change_for_unrelated_provider_does_not_detach_profile_voice_token(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    request_key = ("audio_cpp", "<opaque:model>")
    service.voice_started_by_request[request_key] = asyncio.Event()
    service.voice_finished_by_request[request_key] = asyncio.Event()
    service.voice_gates[request_key] = asyncio.Event()
    preset = _profile_preset(model_id=request_key[1], voice_id="[voice]")
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await service.voice_started_by_request[request_key].wait()
        pane = app.query_one(SpeechPlaygroundPane)
        pending_token = pane._profile_voice_validation_token
        assert pending_token is not None

        pane.mark_provider_configuration_changed("openai", 2)
        await pilot.pause()

        assert pane._profile_voice_validation_token is pending_token
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        service.voice_gates[request_key].set()
        await service.voice_finished_by_request[request_key].wait()


@pytest.mark.asyncio
async def test_stale_catalog_downgrades_available_profile_to_warned_unverified_attempt(
    audio_cpp_playground: FakeTTSService,
) -> None:
    """A naturally stale (not reconfigured) catalog downgrades an
    "available" preset to "unverified" -- Generate must stay ENABLED and
    actually produce one exact attempt with a warning, not sit enabled
    while silently doing nothing when pressed."""
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(
        health=ProviderHealth(state="available", fresh=False)
    )
    preset = _profile_preset(model_id="<opaque:model>", voice_id="[voice]")
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert preset.availability == "available"
        assert pane._profile_effective_availability == "unverified"
        assert app.query_one("#tts-model-select", Select).value == preset.model_id
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert app.query_one("#tts-generate-btn", Button).disabled is False

        pane.action_generate_tts()
        await pilot.pause()

        assert len(app.generation_events) == 1, (
            "the enabled Generate button did nothing when pressed for an "
            "unverified profile preset"
        )
        assert any(
            "unverified" in message.lower() and severity == "warning"
            for message, severity in app.notices
        )


@pytest.mark.asyncio
async def test_profile_service_acquisition_failure_projects_exact_disabled_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_profile_preview_blocked_presentation`'s `service is None` branch
    had zero behavioral coverage anywhere -- `test_speech_profile_port.py`
    only checks the method is inherited, never that acquiring the TTS
    service failing actually projects the disabled, private-detail-free
    recovery copy."""

    async def _service_unavailable() -> object:
        raise RuntimeError("private service detail")

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _service_unavailable(),
    )
    preset = _profile_preset()
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.query_one("#tts-provider-select", Select).value == preset.provider_id
        assert app.query_one("#tts-model-select", Select).value == preset.model_id
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render())
        assert "unavailable" in status.lower()
        assert "private service detail" not in status
        banner = app.query_one("#tts-profile-preview-status", Static)
        banner_copy = str(banner.render()).lower()
        assert "blocked" in banner_copy
        assert "unavailable" in banner_copy
        assert "exact saved selection" not in banner_copy
        assert banner.has_class("profile-preview-unavailable")


@pytest.mark.asyncio
async def test_long_exact_profile_ids_stay_reachable_via_tooltip(
    audio_cpp_playground: FakeTTSService,
) -> None:
    """MECHANICAL adaptation, reduced scope: the retired widget rendered
    each axis select in its own fixed-height "row" (a `#label` Static plus
    the Select, stacked one row per axis) and asserted that row's exact
    geometry (`select.region.height == 3`, single visible truncated line
    ending in "…", and the voice row sitting directly above the model row).
    `SpeechAxisRow` (the live replacement) lays every axis out as a `Grid`
    of reflowing chips -- there is no equivalent fixed single-row container
    or vertical row adjacency to assert.

    Verified empirically (not assumed) that this specific scenario cannot
    be ported with real layout assertions at all: `SpeechPlaygroundPane`'s
    own CSS classes (`.speech-axis-control`, `.speech-pane`, ...) are
    defined in the app-wide TCSS bundle, not on the widget itself. A bare
    `App[None]` test host -- the shape both the retired widget's
    `_PlaygroundHost` and this file's `_PaneHost` use, and the only
    lightweight host available outside a full `TldwCli` -- loads no CSS at
    all, so every `Select` on the axis row (even in tests that never
    reference geometry) reports a degenerate `Region(x=<viewport-width>,
    width=1, ...)`, reproducible on a plain no-preset mount with no long
    id involved. This is an artifact of the harness, not of the id length or
    the preset path, and does not reproduce inside the pane's own Screen-
    hosted test suite (`test_speech_playground_pane.py`, built on
    `Tests.UI.app_factory._build_test_app`'s real `TldwCli` app and its
    real CSS).

    What survives without any layout dependency, because it is pure
    application state (`Select.tooltip` is set to the exact preset id in
    `SpeechCatalogMixin._apply_controls`, shared code, unaffected by
    whether CSS ever painted a pixel): the tooltip carries the untruncated
    id. That is the one guarantee this reduced test still proves.
    """
    del audio_cpp_playground
    model_id = "[opaque-model]" + "/segment" * 24
    voice_id = "<opaque-voice>" + ":segment" * 24
    app = _PaneHost(preset=_profile_preset(model_id=model_id, voice_id=voice_id))

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        voice_select = app.query_one("#tts-voice-select", Select)
        model_select = app.query_one("#tts-model-select", Select)

        for select, exact_id in (
            (voice_select, voice_id),
            (model_select, model_id),
        ):
            assert isinstance(select.tooltip, Text)
            assert select.tooltip.plain == exact_id


@pytest.mark.asyncio
async def test_unavailable_exact_profile_catalog_failure_projects_but_stays_blocked(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_error = RuntimeError("private catalog detail")
    app = _PaneHost(preset=_profile_preset(availability="unavailable"))

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render())
        assert "unavailable" in status.lower()
        assert "Edit" in status

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice_error",
    (
        None,
        RuntimeError("private voice detail"),
        TTSProviderReconfiguringError("private reconfiguring detail"),
        TTSRegistryClosedError("private closed detail"),
    ),
    ids=("success", "generic", "reconfiguring", "closed"),
)
async def test_fresh_not_configured_catalog_makes_exact_profile_unavailable(
    audio_cpp_playground: FakeTTSService,
    voice_error: Exception | None,
) -> None:
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(
        health=ProviderHealth(state="not_configured", fresh=True)
    )
    service.voice_error = voice_error
    preset = _profile_preset()
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == preset.model_id
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert pane._profile_effective_availability == "unavailable"
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        status = str(app.query_one("#tts-provider-status", Static).render())
        assert "unavailable" in status.lower()
        assert "Edit" in status
        banner = app.query_one("#tts-profile-preview-status", Static)
        assert "unavailable" in str(banner.render()).lower()
        assert "Edit" in str(banner.render())
        assert banner.has_class("profile-preview-unavailable")

        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("availability", "expected_status"),
    (
        ("available", "service is unavailable"),
        ("unavailable", "choose Edit"),
    ),
)
async def test_closed_catalog_health_projects_exact_profile_but_stays_blocked(
    audio_cpp_playground: FakeTTSService,
    availability: str,
    expected_status: str,
) -> None:
    service = audio_cpp_playground
    service.catalogs["audio_cpp"] = _audio_catalog(
        health=ProviderHealth(state="closed", fresh=False)
    )
    preset = _profile_preset(availability=availability)
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        assert app.query_one("#tts-model-select", Select).value == preset.model_id
        assert app.query_one("#tts-voice-select", Select).value == preset.voice_id
        assert app.query_one("#tts-generate-btn", Button).disabled is True
        pane.action_generate_tts()
        await pilot.pause()

        assert app.generation_events == []
        assert (
            expected_status.lower()
            in str(app.query_one("#tts-provider-status", Static).render()).lower()
        )
        banner_copy = str(
            app.query_one("#tts-profile-preview-status", Static).render()
        ).lower()
        assert "generate makes one exact attempt" not in banner_copy
        assert "blocked" in banner_copy or "unavailable" in banner_copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original_availability", "failure_availability"),
    (
        ("available", "unverified"),
        ("unverified", "unverified"),
        ("unavailable", "unavailable"),
    ),
)
async def test_current_voice_discovery_success_restores_only_original_preset_state(
    audio_cpp_playground: FakeTTSService,
    original_availability: str,
    failure_availability: str,
) -> None:
    service = audio_cpp_playground
    service.voice_error = RuntimeError("private upstream detail")
    preset = _profile_preset(
        model_id="<opaque:model>",
        voice_id="[voice]",
        availability=original_availability,
    )
    app = _PaneHost(preset=preset)

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        assert pane._profile_effective_availability == failure_availability
        assert preset.availability == original_availability
        if original_availability == "unavailable":
            status = str(app.query_one("#tts-provider-status", Static).render())
            assert "unavailable" in status.lower()
            assert "Edit" in status

        service.voice_error = None
        pane._load_provider_voices(
            "audio_cpp",
            preset.model_id,
            service.catalogs["audio_cpp"].revision,
            refresh=True,
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert pane._profile_effective_availability == original_availability
        assert preset.availability == original_availability


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edited_control", ("provider", "model", "voice", "format", "speed", "options")
)
async def test_user_generation_selection_edits_end_profile_preset_association(
    audio_cpp_playground: FakeTTSService,
    edited_control: str,
) -> None:
    """MECHANICAL adaptation for the "options" case: the retired widget
    mounted every provider's parameter container up front (hidden but
    focusable), so it could edit `#tts-stability-input` (an ElevenLabs-only
    control) while previewing an *audio.cpp* preset. `SpeechParamGroup`
    mounts only the SELECTED provider's own knobs, and
    `PROVIDER_PARAMS["audio_cpp"] == ()` -- audio.cpp has no options-class
    control at all in the redesigned split, so there is nothing to edit
    there anymore. This parametrization previews an ElevenLabs preset
    instead (whose `SpeechParamGroup`, including `#tts-stability-input`, is
    mounted straight from `compose()` because the pane is constructed with
    `provider="elevenlabs"` up front). The guarantee under test -- editing
    ANY control category ends the profile preset association -- is
    unchanged; only the fixture needed to reach an "options" control at all
    had to change.
    """
    del audio_cpp_playground
    if edited_control == "options":
        app = _PaneHost(
            provider="elevenlabs",
            preset=_profile_preset(
                provider_id="elevenlabs",
                model_id="eleven_multilingual_v2",
                voice_id="21m00Tcm4TlvDq8ikWAM",
            ),
        )
    else:
        app = _PaneHost(preset=_profile_preset())

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        assert pane._profile_preset is not None

        if edited_control == "provider":
            app.query_one("#tts-provider-select", Select).value = "openai"
        elif edited_control == "model":
            app.query_one("#tts-model-select", Select).value = "second-model"
        elif edited_control == "voice":
            app.query_one("#tts-voice-select", Select).value = SERVER_DEFAULT_VOICE_ID
        elif edited_control == "format":
            select = app.query_one("#tts-format-select", Select)
            select.disabled = False
            select.set_options([("WAV", "wav"), ("MP3", "mp3")])
            select.value = "mp3"
        elif edited_control == "speed":
            speed = app.query_one("#tts-speed-input", Input)
            speed.disabled = False
            speed.value = "1.1"
        else:
            options = app.query_one("#tts-stability-input", Input)
            options.focus()
            options.value = "0.6"
        await pilot.pause()

        assert pane._profile_preset is None


@pytest.mark.asyncio
async def test_provider_edit_during_initial_profile_catalog_load_ends_preset(
    audio_cpp_playground: FakeTTSService,
) -> None:
    service = audio_cpp_playground
    service.catalog_started = asyncio.Event()
    service.allow_catalog = asyncio.Event()
    app = _PaneHost(preset=_profile_preset())

    async with app.run_test(size=(180, 70)) as pilot:
        await wait_for_signal(service.catalog_started, what="the catalog worker start")
        pane = app.query_one(SpeechPlaygroundPane)
        provider = app.query_one("#tts-provider-select", Select)
        assert pane._profile_controls_applied is True
        assert app.query_one("#tts-model-select", Select).value == "profile/model"
        assert app.query_one("#tts-voice-select", Select).value == "profile/voice"
        banner = app.query_one("#tts-profile-preview-status", Static)
        banner_copy = str(banner.render()).lower()
        assert "loading" in banner_copy
        assert "blocked" not in banner_copy
        assert "unavailable" not in banner_copy
        assert banner.has_class("profile-preview-loading")

        provider.value = "openai"
        await _wait_until(pilot, lambda: pane._profile_preset is None)
        assert pane._profile_controls_applied is True

        service.allow_catalog.set()
        await app.workers.wait_for_complete()


# =====================================================================
# Section D -- save current result as a voice profile
# =====================================================================


@pytest.mark.asyncio
async def test_save_profile_action_is_visible_and_focusable_at_narrow_width(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
) -> None:
    """Needs the real app CSS bundle to be a meaningful geometry check --
    see `_PaneScreen`'s docstring for why this test alone uses
    `_build_test_app()`/`_PaneScreen` instead of this file's usual bare
    `_PaneHost`.
    """
    del audio_cpp_playground
    artifact = _native_profile_artifact(tmp_path / "narrow.wav")
    app = _build_test_app()

    async with app.run_test(size=(50, 24)) as pilot:
        screen = _PaneScreen()
        await app.push_screen(screen)
        await pilot.pause()
        # No `app.workers.wait_for_complete()` here: unlike `_PaneHost` (a
        # bare `App[None]` with no other workers), a real `TldwCli` from
        # `_build_test_app()` carries its own persistent background
        # workers, so waiting for the WHOLE app's worker set to go idle
        # never returns. `test_speech_playground_pane.py`'s own
        # `_build_test_app()`-based tests use plain `pilot.pause()` for the
        # same reason -- this test doesn't need the catalog to finish
        # loading, only the pane mounted.
        await pilot.pause()
        pane = screen.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)
        button = screen.query_one("#audio-save-profile-btn", Button)
        player = screen.query_one("#audio-player-container")
        button.focus()
        button.scroll_visible(animate=False)
        await pilot.pause()

        assert button.has_focus
        assert player.content_region.contains_region(button.region)


@pytest.mark.asyncio
async def test_successful_native_artifact_save_uses_only_immutable_provenance(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = audio_cpp_playground
    profile_service = _SaveProfileService()
    app = _PaneHost(profile_service=profile_service)
    artifact_path = tmp_path / "result.wav"
    artifact_path.write_bytes(b"RIFF")
    artifact = _native_profile_artifact(artifact_path)
    handler = STTSEventHandler(app)
    handler._accept_playground_artifact(artifact)
    app._stts_handler = handler

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)
        save_button = app.query_one("#audio-save-profile-btn", Button)
        assert save_button.display is True
        assert save_button.disabled is False

        model_select = app.query_one("#tts-model-select", Select)
        model_select.set_options([("Changed model", "changed/model")])
        model_select.value = "changed/model"
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.set_options([("Changed voice", "changed/voice")])
        voice_select.value = "changed/voice"
        await pilot.pause()
        discovery_before = (list(service.catalog_calls), list(service.voice_calls))

        async def _name(screen: object) -> str:
            assert isinstance(screen, profile_library_module.TTSProfileNameModal)
            return "Saved exact voice"

        monkeypatch.setattr(app, "push_screen_wait", _name)
        await pane._save_current_result_as_profile()

        assert profile_service.create_calls == [("Saved exact voice", artifact)]
        saved = profile_service.create_calls[0][1].requested_selection
        assert saved is artifact.requested_selection
        assert saved is not None
        assert (saved.model_id, saved.voice_id) == (
            "artifact/model",
            "artifact/voice",
        )
        assert app.profile_service_requests == 1
        assert (service.catalog_calls, service.voice_calls) == discovery_before
        assert type(pane.current_audio_artifact) is STTSPlaygroundResultProjection
        assert pane.current_audio_artifact.operation_id == artifact.operation_id
        assert artifact_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ("worker", "unmount"))
async def test_cancelled_profile_name_modal_is_dismissed_without_saving(
    audio_cpp_playground: FakeTTSService,
    tmp_path: Path,
    cancel_mode: str,
) -> None:
    del audio_cpp_playground
    profile_service = _SaveProfileService()
    app = _PaneHost(profile_service=profile_service)
    artifact = _native_profile_artifact(tmp_path / "cancelled.wav")

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)
        save_button = app.query_one("#audio-save-profile-btn", Button)
        save_button.focus()
        await pilot.pause()

        assert save_button.has_focus
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: isinstance(
                app.screen,
                profile_library_module.TTSProfileNameModal,
            ),
        )
        modal = app.screen
        save_workers = tuple(
            worker
            for worker in app.workers
            if getattr(worker, "group", None) == "save_tts_result_as_profile"
        )
        assert len(save_workers) == 1
        assert not save_workers[0].is_finished

        if cancel_mode == "worker":
            save_workers[0].cancel()
        else:
            await pane.remove()

        await _wait_until(pilot, lambda: app.screen is not modal)
        await _wait_until(
            pilot,
            lambda: all(worker.is_finished for worker in save_workers),
        )

        assert not isinstance(
            app.screen,
            profile_library_module.TTSProfileNameModal,
        )
        assert profile_service.create_calls == []
        assert app.profile_service_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_copy"),
    (
        (
            ProfileRepositoryError("conflict"),
            profile_library_module.PROFILE_CONFLICT_COPY,
        ),
        (
            ProfileRepositoryError("stale"),
            profile_library_module.PROFILE_CONFLICT_COPY,
        ),
        (
            ProfileRepositoryError("unavailable"),
            profile_library_module.PROFILE_STORE_UNAVAILABLE_COPY,
        ),
        (
            ProfileServiceError("stale_configuration"),
            (
                "TTS settings changed after this audio was generated. Generate a new "
                "result before saving it as a profile."
            ),
        ),
    ),
)
async def test_save_profile_failures_use_value_independent_recovery_copy(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
    expected_copy: str,
) -> None:
    del audio_cpp_playground
    profile_service = _SaveProfileService()
    profile_service.error = error
    app = _PaneHost(profile_service=profile_service)
    artifact = _native_profile_artifact(tmp_path / "result.wav")
    handler = STTSEventHandler(app)
    handler._accept_playground_artifact(artifact)
    app._stts_handler = handler

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)

        async def _name(_screen: object) -> str:
            return "Safe name"

        monkeypatch.setattr(app, "push_screen_wait", _name)
        await pane._save_current_result_as_profile()

        status = str(app.query_one("#audio-player-status", Static).render())
        assert status == expected_copy
        assert app.profile_service_requests == 1


@pytest.mark.asyncio
async def test_unavailable_lazy_profile_service_keeps_audio_artifact_owned_by_handler(
    audio_cpp_playground: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del audio_cpp_playground
    app = _PaneHost(profile_service=None)
    artifact = _native_profile_artifact(tmp_path / "retained.wav")

    async with app.run_test(size=(180, 70)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(artifact, announce=False)

        async def _name(_screen: object) -> str:
            return "Unavailable store"

        monkeypatch.setattr(app, "push_screen_wait", _name)
        await pane._save_current_result_as_profile()

        assert (
            str(app.query_one("#audio-player-status", Static).render())
            == profile_library_module.PROFILE_STORE_UNAVAILABLE_COPY
        )
        assert app.profile_service_requests == 1
