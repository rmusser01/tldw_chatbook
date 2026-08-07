"""The assembled Playground: fold position, truncation, and containment."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, Select, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_stts_playground_audio_cpp import (
    FakeTTSService,
    _native_profile_artifact,
    _profile_preset,
    _resolved,
    _wait_until,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
)
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.playground_types import STTSGeneratedAudio
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.UI.Speech.speech_axis_row import axis_chip_id
from tldw_chatbook.UI.Speech.speech_playground_model import AXIS_CONTROLS
from tldw_chatbook.UI.Speech.speech_playground_pane import (
    PLAYGROUND_ACTIONS,
    SpeechPlaygroundPane,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSNavigationIntent,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    SERVER_DEFAULT_VOICE_ID,
    PlaygroundControls,
)
from tldw_chatbook.UI.stts_profile_library import TTSProfileNameModal
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


class _PaneScreen(Screen):
    """Hosts the pane on its own, under the real app CSS.

    The pane is not the mounted `playground` view yet: dev's
    `TTSPlaygroundWidget` keeps that while its profile presets and this
    pane's axis row are reconciled. So these mount the pane directly rather
    than navigating to Speech and asserting on whatever is there -- which
    would be testing the routing, not the pane.
    """

    def compose(self):
        body = Vertical(
            SpeechPlaygroundPane(id="speech-playground-pane"), id="lab-body"
        )
        # Inline, because the app-tier bundle outranks a test Screen's
        # DEFAULT_CSS. The Lab frame constrains its body to the viewport;
        # without that the pane is sized by its own content -- measured 236
        # cells wide in a 60-column terminal -- so nothing ever looks narrow
        # and the stacking rule never fires.
        body.styles.width = "100%"
        body.styles.height = "100%"
        yield body

    def on_mount(self) -> None:
        """Pin the pane to the viewport width once it exists."""
        pane = self.query_one("#speech-playground-pane")
        pane.styles.width = "100%"


async def _speech_screen(app):
    screen = _PaneScreen()
    await app.push_screen(screen)
    return screen


def _generated_artifact(tmp_path, *, metadata=None) -> STTSGeneratedAudio:
    path = tmp_path / "current-result.wav"
    path.write_bytes(b"RIFF")
    return STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="model-a",
        voice_id=None,
        source_text="Synthetic UAT text",
        operation_id="current-result-operation",
        audio_format="wav",
        content_type="audio/wav",
        metadata=metadata or {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_the_primary_action_is_above_the_fold(size):
    """The defect this phase exists to fix.

    `Generate Speech` rendered at y=60 in a 34-row viewport -- 21 rows below
    the fold, reachable only by scrolling ~2.5 screens.
    """
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        body = screen.query_one("#lab-body")
        # The bare id, not `workbench-action-tts-generate-btn`. This test
        # originally asserted the prefixed one, which is what `CommandStrip`
        # mounts -- so it passed against a button whose id the handler
        # (`event.button.id == "tts-generate-btn"`) could never match. A
        # visible, above-the-fold, permanently dead button.
        generate = screen.query_one("#tts-generate-btn", Button)
        assert body.region.contains_region(generate.region), (
            f"Generate below the fold at {size}: y={generate.region.y}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_no_control_is_clipped_by_its_container(size):
    """Containment, not just self-rendering.

    `render_line(0).text` reads a widget in its OWN coordinate space, so it
    reports a full label for a control the parent is clipping. That is how
    "Export" shipped rendering as "Exp": it sat at x=101..111 inside a pane
    ending at 107 and every self-oriented check called it clean. Assert the
    region is inside its parent.
    """
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()

        escaped = []
        for strip_id in ("#speech-playground-actions", "#speech-result-actions"):
            strip = screen.query_one(strip_id)
            for button in strip.query(Button):
                if not button.region.width:
                    continue
                if not strip.region.contains_region(button.region):
                    escaped.append(
                        (
                            strip_id,
                            str(button.label),
                            strip.region,
                            button.region,
                            tuple(
                                sorted(screen.query_one(SpeechPlaygroundPane).classes)
                            ),
                        )
                    )
        assert not escaped, f"clipped by container at {size}: {escaped}"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_delivered_result_actions_are_visible_inside_result_and_viewport(
    size,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for UAT completing a WAV with no visible Play button."""

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        pane = screen.query_one(SpeechPlaygroundPane)

        pane._generation_complete(
            _generated_artifact(tmp_path, metadata={"audio_duration_ms": 4_250.0})
        )
        await pilot.pause()
        await pilot.pause()

        body = screen.query_one("#lab-body")
        result = screen.query_one("#speech-result-pane")
        actions = screen.query_one("#speech-result-actions")
        assert result.region.contains_region(actions.region)
        for control_id in ("audio-play-btn", "audio-export-btn"):
            control = screen.query_one(f"#{control_id}", Button)
            assert control.region.width > 0
            assert result.region.contains_region(control.region), control_id
            assert body.region.contains_region(control.region), control_id


@pytest.mark.asyncio
async def test_native_profile_save_action_tracks_generation_lifecycle(
    faked_service: FakeTTSService,
    tmp_path,
) -> None:
    """A native result is saveable only while it is the idle current take."""

    del faked_service
    app = _AxisHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                "Ready"
                in str(
                    app.query_one(
                        "#speech-status-catalog-freshness",
                        Static,
                    ).renderable
                )
            ),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        button = app.query_one("#audio-save-profile-btn", Button)
        first_path = tmp_path / "first.wav"
        first_path.write_bytes(b"RIFF")

        pane._store_delivered_artifact(
            _native_profile_artifact(first_path),
            announce=False,
        )
        await pilot.pause()

        assert not button.has_class("hidden")
        assert button.disabled is False

        app.query_one("#tts-text-input").text = "A second synthetic take."
        pane._generate_tts()
        await pilot.pause()

        assert pane._profile_save_suppressed is True
        assert button.has_class("hidden")
        assert button.disabled is True
        operation_id = pane._generation_operation_id
        assert isinstance(operation_id, str)

        second_path = tmp_path / "second.wav"
        second_path.write_bytes(b"RIFF")
        pane._generation_complete(
            _native_profile_artifact(
                second_path,
                operation_id=operation_id,
            )
        )
        await pilot.pause()

        assert pane._profile_save_suppressed is False
        assert not button.has_class("hidden")
        assert button.disabled is False

        app.query_one("#tts-text-input").text = "A failing synthetic take."
        pane._generate_tts()
        pane._generation_complete(None)
        await pilot.pause()

        assert pane._profile_save_suppressed is True
        assert button.has_class("hidden")
        assert button.disabled is True


@pytest.mark.asyncio
async def test_options_bearing_result_states_why_it_cannot_be_saved(
    faked_service: FakeTTSService,
    tmp_path,
) -> None:
    """A missing Save with no explanation is the chrome-honesty defect.

    Slice 1 profiles hold empty options, so a generation that used
    provider-specific options cannot be reproduced exactly and is refused
    provenance. The result region must say that in user language rather than
    silently dropping the affordance.
    """

    del faked_service
    app = _AxisHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        blocked_path = tmp_path / "blocked.mp3"
        blocked_path.write_bytes(b"ID3")

        pane._store_delivered_artifact(
            STTSGeneratedAudio(
                path=blocked_path,
                provider_id="higgs",
                model_id="higgs-v2",
                voice_id="narrator",
                source_text="private text",
                operation_id="blocked-operation",
                audio_format="mp3",
                content_type="audio/mpeg",
                profile_save_block_code="provider_options",
            ),
            announce=False,
        )
        await pilot.pause()

        button = app.query_one("#audio-save-profile-btn", Button)
        lifecycle = str(app.query_one("#audio-result-lifecycle", Static).renderable)
        assert button.has_class("hidden")
        assert "provider-specific options" in lifecycle
        assert "voice profile" in lifecycle.casefold()

        clean_path = tmp_path / "clean.wav"
        clean_path.write_bytes(b"RIFF")
        pane._store_delivered_artifact(
            _native_profile_artifact(clean_path),
            announce=False,
        )
        await pilot.pause()

        clean_lifecycle = str(
            app.query_one("#audio-result-lifecycle", Static).renderable
        )
        assert "provider-specific options" not in clean_lifecycle
        assert "temporary" in clean_lifecycle.casefold()


@pytest.mark.asyncio
async def test_save_profile_button_opens_the_name_dialog(
    faked_service: FakeTTSService,
    tmp_path,
) -> None:
    """The redesigned pane must dispatch the ported profile workflow."""

    del faked_service
    app = _AxisHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                "Ready"
                in str(
                    app.query_one(
                        "#speech-status-catalog-freshness",
                        Static,
                    ).renderable
                )
            ),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        artifact_path = tmp_path / "saveable.wav"
        artifact_path.write_bytes(b"RIFF")
        pane._store_delivered_artifact(
            _native_profile_artifact(artifact_path),
            announce=False,
        )
        await pilot.pause()

        await pilot.click("#audio-save-profile-btn")
        await _wait_until(
            pilot,
            lambda: isinstance(app.screen, TTSProfileNameModal),
        )
        modal = app.screen

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app.screen is not modal)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_no_chip_text_is_truncated(size):
    """The axes are what the user is comparing; they may not be cut off."""
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        for chip in screen.query(".speech-chip").results(Static):
            # A child keeps its own ``display`` flag when an ancestor is
            # hidden, but Textual gives it a zero-sized region. Only inspect
            # chips that actually participate in this layout.
            if not chip.display or not chip.region.width or not chip.region.height:
                continue
            text = str(chip.renderable)
            assert text in chip.render_line(0).text, f"truncated at {size}: {text!r}"


@pytest.mark.asyncio
async def test_the_pane_scrolls_rather_than_clipping_when_stacked():
    """`1fr` children compress instead of overflowing, which clips content
    that should scroll. The pane must be genuinely taller than its viewport."""
    app = _build_test_app()
    # 60 cells, not 80. Hosted directly the pane gets the whole terminal
    # width, where inside the Lab frame it got the body minus rail and
    # inspector -- so 80 columns used to leave it under its own 64-cell
    # threshold and now does not. The test is about what the pane does when
    # it IS too narrow, so it is measured below the threshold.
    async with app.run_test(size=(60, 24)) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        pane = screen.query_one("#speech-playground-pane")
        assert pane.has_class("speech-split-stacked")
        assert pane.virtual_size.height > pane.container_size.height


@pytest.mark.asyncio
async def test_the_axes_and_the_text_input_are_both_present():
    """The comparison loop needs both: what you are varying, and what you
    are synthesizing."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#tts-text-input")
        assert screen.query_one("#speech-axis-row")
        assert screen.query_one("#audio-player-container")
        assert not screen.query("#speech-result-history")
        assert screen.query_one("#speech-param-group")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_status"),
    [
        ({"audio_duration_ms": 4_250.0}, "Ready · WAV · 0:04"),
        ({}, "Ready · WAV"),
    ],
)
async def test_current_result_reports_only_known_artifact_facts(
    metadata,
    expected_status,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _AxisHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)

        pane._generation_complete(_generated_artifact(tmp_path, metadata=metadata))
        await pilot.pause()

        status = str(app.query_one("#audio-player-status", Static).renderable)
        lifecycle = str(app.query_one("#audio-result-lifecycle", Static).renderable)
        time_copy = str(app.query_one("#audio-time-display", Static).renderable)
        assert status == expected_status
        assert "temporary" in lifecycle.casefold()
        assert "export" in lifecycle.casefold()
        assert "0:00 / 0:00" not in time_copy


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_play", [False, True])
async def test_generation_honors_studio_auto_play_and_keyboard_recovery(
    auto_play,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _AxisHarness(
        studio_preferences=StudioTTSPreferencesSnapshot(auto_play=auto_play)
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        play_calls: list[bool] = []
        monkeypatch.setattr(pane, "_play_audio", lambda: play_calls.append(True))

        pane._generation_complete(_generated_artifact(tmp_path))
        await pilot.pause()
        await pilot.pause()

        if auto_play:
            assert play_calls == [True]
        else:
            assert play_calls == []
            assert app.query_one("#audio-play-btn", Button).has_focus


@pytest.mark.asyncio
async def test_new_result_stops_active_playback_before_replacing_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivered result must not strand older audio without Stop."""

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )

    class Player:
        def __init__(self) -> None:
            self.state = PlaybackState.IDLE
            self.stop_calls = 0

        async def get_state(self) -> PlaybackState:
            return self.state

        async def stop(self) -> bool:
            self.stop_calls += 1
            self.state = PlaybackState.IDLE
            return True

        async def play(self, _path: Path) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def is_playing(self) -> bool:
            return self.state is PlaybackState.PLAYING

        async def get_position(self) -> float:
            return 0.0

        async def get_duration(self) -> float:
            return 1.0

    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    old_path.write_bytes(b"RIFF")
    new_path.write_bytes(b"RIFF")
    old_artifact = _native_profile_artifact(
        old_path,
        operation_id="old-operation",
    )
    new_artifact = _native_profile_artifact(
        new_path,
        operation_id="new-operation",
    )
    app = _AxisHarness()
    player = Player()
    app.audio_player = player

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(old_artifact, announce=False)
        pane._play_audio()
        await _wait_until(
            pilot,
            lambda: (
                player.state is PlaybackState.PLAYING and pane._play_worker_task is None
            ),
        )

        pane._generation_complete(new_artifact)
        await _wait_until(
            pilot,
            lambda: (
                pane.current_audio_artifact is new_artifact
                and player.state is PlaybackState.IDLE
            ),
        )

        assert player.stop_calls >= 2
        assert app.query_one("#audio-play-btn", Button).disabled is False
        save_profile = app.query_one("#audio-save-profile-btn", Button)
        assert save_profile.disabled is False
        assert not save_profile.has_class("hidden")


@pytest.mark.asyncio
async def test_auto_play_new_result_cancels_prior_start_worker_before_takeover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-play must take over even while the previous start is pending."""

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )

    class GatedPlayer:
        def __init__(self) -> None:
            self.state = PlaybackState.IDLE
            self.first_get_state_started = asyncio.Event()
            self.get_state_calls = 0
            self.played: list[Path] = []

        async def get_state(self) -> PlaybackState:
            self.get_state_calls += 1
            if self.get_state_calls == 1:
                self.first_get_state_started.set()
                await asyncio.Event().wait()
            return self.state

        async def stop(self) -> bool:
            self.state = PlaybackState.IDLE
            return True

        async def play(self, path: Path) -> bool:
            self.played.append(path)
            self.state = PlaybackState.PLAYING
            return True

        async def is_playing(self) -> bool:
            return self.state is PlaybackState.PLAYING

        async def get_position(self) -> float:
            return 0.0

        async def get_duration(self) -> float:
            return 1.0

    old_path = tmp_path / "old-starting.wav"
    new_path = tmp_path / "new-auto-play.wav"
    old_path.write_bytes(b"RIFF")
    new_path.write_bytes(b"RIFF")
    old_artifact = STTSGeneratedAudio(
        path=old_path,
        provider_id="audio_cpp",
        model_id="model-a",
        voice_id=None,
        source_text="Synthetic old result",
        operation_id="old-start-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="model-a",
        voice_id=None,
        source_text="Synthetic new result",
        operation_id="new-auto-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    app = _AxisHarness(studio_preferences=StudioTTSPreferencesSnapshot(auto_play=True))
    player = GatedPlayer()
    app.audio_player = player

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(old_artifact, announce=False)
        pane._play_audio()
        await player.first_get_state_started.wait()

        pane._generation_complete(new_artifact)
        await _wait_until(pilot, lambda: player.played == [new_path])

        assert pane.current_audio_artifact is new_artifact


@pytest.mark.asyncio
async def test_profile_navigation_fences_result_waiting_for_playback_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retired result must not publish after an exact preset is applied."""

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )

    class GatedStopPlayer:
        def __init__(self) -> None:
            self.state = PlaybackState.IDLE
            self.stop_calls = 0
            self.replacement_stop_started = asyncio.Event()
            self.release_replacement_stop = asyncio.Event()

        async def get_state(self) -> PlaybackState:
            return self.state

        async def stop(self) -> bool:
            self.stop_calls += 1
            if self.stop_calls == 2:
                self.replacement_stop_started.set()
                await self.release_replacement_stop.wait()
            self.state = PlaybackState.IDLE
            return True

        async def play(self, _path: Path) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def is_playing(self) -> bool:
            return self.state is PlaybackState.PLAYING

        async def get_position(self) -> float:
            return 0.0

        async def get_duration(self) -> float:
            return 1.0

    old_artifact = _generated_artifact(tmp_path)
    new_path = tmp_path / "retired-result.wav"
    new_path.write_bytes(b"RIFF")
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="model-a",
        voice_id=None,
        source_text="Synthetic retired result",
        operation_id="retired-result-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    app = _AxisHarness()
    player = GatedStopPlayer()
    app.audio_player = player

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(old_artifact, announce=False)
        pane._play_audio()
        await _wait_until(
            pilot,
            lambda: (
                player.state is PlaybackState.PLAYING and pane._play_worker_task is None
            ),
        )

        pane._generation_complete(new_artifact)
        await player.replacement_stop_started.wait()
        pane.apply_profile_preset(_profile_preset(model_id="new-profile-model"))
        player.release_replacement_stop.set()
        await app.workers.wait_for_complete()

        assert pane.current_audio_artifact is old_artifact
        assert pane.current_audio_file == old_artifact.path


@pytest.mark.asyncio
async def test_play_is_blocked_while_new_result_waits_for_playback_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old result cannot be replayed during replacement ownership."""

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )

    class GatedStopPlayer:
        def __init__(self) -> None:
            self.state = PlaybackState.IDLE
            self.stop_calls = 0
            self.get_state_calls = 0
            self.replacement_stop_started = asyncio.Event()
            self.release_replacement_stop = asyncio.Event()

        async def get_state(self) -> PlaybackState:
            self.get_state_calls += 1
            return self.state

        async def stop(self) -> bool:
            self.stop_calls += 1
            if self.stop_calls == 2:
                self.replacement_stop_started.set()
                await self.release_replacement_stop.wait()
            self.state = PlaybackState.IDLE
            return True

        async def play(self, _path: Path) -> bool:
            self.state = PlaybackState.PLAYING
            return True

        async def is_playing(self) -> bool:
            return self.state is PlaybackState.PLAYING

        async def get_position(self) -> float:
            return 0.0

        async def get_duration(self) -> float:
            return 1.0

    old_artifact = _generated_artifact(tmp_path)
    new_path = tmp_path / "replacement.wav"
    new_path.write_bytes(b"RIFF")
    new_artifact = STTSGeneratedAudio(
        path=new_path,
        provider_id="audio_cpp",
        model_id="model-a",
        voice_id=None,
        source_text="Synthetic replacement result",
        operation_id="replacement-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    app = _AxisHarness()
    player = GatedStopPlayer()
    app.audio_player = player

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        pane._store_delivered_artifact(old_artifact, announce=False)
        pane._play_audio()
        await _wait_until(
            pilot,
            lambda: (
                player.state is PlaybackState.PLAYING and pane._play_worker_task is None
            ),
        )

        pane._generation_complete(new_artifact)
        await player.replacement_stop_started.wait()
        try:
            play = app.query_one("#audio-play-btn", Button)
            stop = app.query_one("#stop-audio-btn", Button)
            assert play.disabled is True
            assert stop.disabled is False
            get_state_calls = player.get_state_calls
            pane._play_audio()
            await pilot.pause(0.05)
            assert player.get_state_calls == get_state_calls
        finally:
            player.release_replacement_stop.set()
        await app.workers.wait_for_complete()

        assert pane.current_audio_artifact is new_artifact


@pytest.mark.asyncio
async def test_secondary_diagnostics_start_collapsed_and_use_neutral_detail_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _AxisHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        details = app.query_one("#speech-connection-details", Collapsible)
        assert details.collapsed is True
        assert not details.query_one(
            "#speech-status-provider-runtime", Static
        ).has_class("speech-status-line")
        assert not details.query("#tts-provider-status")


@pytest.mark.asyncio
async def test_audio_cpp_hides_its_non_applicable_language_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _AxisHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        language_cell = app.query_one("#speech-axis-cell-tts-language-select")
        language = app.query_one("#tts-language-select", Select)
        assert language_cell.display is False
        assert language.disabled is True


@pytest.mark.asyncio
async def test_speech_disclosures_are_compact_when_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()

        for control_id in (
            "speech-param-group",
            "speech-connection-details",
            "speech-log-group",
        ):
            disclosure = screen.query_one(f"#{control_id}", Collapsible)
            assert disclosure.collapsed is True
            assert disclosure.region.height == 1, (control_id, disclosure.region)


def test_playground_action_copy_names_the_resulting_action() -> None:
    labels = {action.id: action.label for action in PLAYGROUND_ACTIONS}
    assert labels["tts-random-text-btn"] == "Sample text"
    assert labels["tts-clear-text-btn"] == "Clear text"


# --- Axis-model ownership (Docs/superpowers/specs/2026-07-30-speech-preset-axis-ownership.md) ---
#
# `SpeechPlaygroundPane.axis_values`/`axis_defaults` are the model of record
# for axis presentation; every writer of axis state also refreshes the row's
# painted markers. Internal-dict-only assertions have lied before in this
# codebase, so these assert the painted label text and CSS class -- the same
# discipline `test_speech_axis_row.py` uses for the row in isolation.


class _AxisHarness(App[None]):
    """Mounts the pane alone, so a real catalog load can drive `_apply_controls`
    without the Lab frame or screen-navigation plumbing."""

    def __init__(self, **pane_kwargs) -> None:
        super().__init__()
        self._pane_kwargs = pane_kwargs

    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(**self._pane_kwargs)


def _force_default_provider(monkeypatch: pytest.MonkeyPatch, provider_id: str) -> None:
    """Pin which provider `_load_provider_catalog`'s initialize branch picks.

    Unpatched, `_cli_setting("app_tts", "default_provider", options[0][1])`
    is deterministic, not flaky: the sandboxed test config has no
    `[app_tts]` section, so it falls through to `options[0][1]`, which is
    always `audio_cpp` (`FakeTTSService`'s descriptor order matches
    production's `adapter_bootstrap.build_default_tts_service`). These
    tests pin the provider anyway because several exercise it explicitly
    as `"openai"` (to get an unlocked speed axis) or want the assertion to
    read as "this provider" rather than "whichever one happens to sort
    first" -- not to route around nondeterminism that was never there.
    """
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_cli_setting",
        lambda self, section, key, default=None: (
            provider_id
            if (section, key) == ("app_tts", "default_provider")
            else default
        ),
    )


@pytest.fixture
def faked_service(monkeypatch: pytest.MonkeyPatch) -> FakeTTSService:
    """Point the pane's service hook at the legacy tests' fake.

    Same fake and same patch shape `test_speech_axes_populate.py` uses --
    the point of sharing `SpeechCatalogMixin` is that both panes are driven
    by identical inputs.
    """
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane, "_check_higgs_installation", lambda self: None
    )
    return service


@pytest.mark.asyncio
async def test_playground_status_keeps_external_provider_and_local_deps_independent(
    faked_service: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for dependency in (
        "stt_processing",
        "kokoro_onnx",
        "chatterbox",
        "higgs_tts",
    ):
        monkeypatch.setitem(DEPENDENCIES_AVAILABLE, dependency, False)
    app = _AxisHarness()

    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                "Ready"
                in str(
                    app.query_one("#speech-status-provider-runtime", Static).renderable
                )
                and "Ready"
                in str(
                    app.query_one(
                        "#speech-status-catalog-freshness",
                        Static,
                    ).renderable
                )
            ),
        )

        assert "Ready" in str(
            app.query_one("#speech-status-provider-runtime", Static).renderable
        )
        assert "Ready" in str(
            app.query_one("#speech-status-catalog-freshness", Static).renderable
        )
        for row_id in (
            "stt-dependency",
            "kokoro-dependency",
            "chatterbox-dependency",
            "higgs-dependency",
        ):
            assert "Unavailable" in str(
                app.query_one(f"#speech-status-{row_id}", Static).renderable
            )


@pytest.mark.asyncio
async def test_cancelled_old_catalog_worker_cannot_clear_new_checking_state(
    faked_service: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _AxisHarness()

    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: "audio_cpp" in app.query_one(SpeechPlaygroundPane)._catalogs,
        )
        pane = app.query_one(SpeechPlaygroundPane)
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        call_count = 0

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
                    first_cancelled.set()
                    raise
            second_started.set()
            await release_second.wait()
            return faked_service.catalogs[provider_id]

        monkeypatch.setattr(faked_service, "get_catalog", get_catalog)

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await first_started.wait()
        pane._load_provider_catalog("audio_cpp", refresh=True)
        await first_cancelled.wait()
        await second_started.wait()

        assert "audio_cpp" in pane._catalog_checking_providers

        release_second.set()
        await _wait_until(
            pilot,
            lambda: "audio_cpp" not in pane._catalog_checking_providers,
        )


@pytest.mark.asyncio
async def test_failed_refresh_retains_catalog_as_stale_with_model_recovery(
    faked_service: FakeTTSService,
) -> None:
    app = _AxisHarness()

    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                "Ready"
                in str(
                    app.query_one(
                        "#speech-status-catalog-freshness",
                        Static,
                    ).renderable
                )
            ),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        faked_service.catalog_error = RuntimeError("private catalog detail")

        pane._load_provider_catalog("audio_cpp", refresh=True)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert "Unavailable" in str(
            app.query_one("#speech-status-provider-runtime", Static).renderable
        )
        assert "Stale" in str(
            app.query_one("#speech-status-catalog-freshness", Static).renderable
        )
        model_id = pane._current_select_value("#tts-model-select")
        assert isinstance(model_id, str)
        status = pane._runtime_status_store.catalog_status("audio_cpp", model_id)
        assert status is not None
        assert status.recovery_action is SpeechTTSNavigationIntent.REFRESH_MODELS


@pytest.mark.asyncio
async def test_initial_service_failure_is_unavailable_not_loading_or_not_checked(
    faked_service: FakeTTSService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del faked_service

    async def fail_service() -> object:
        raise RuntimeError("private initialization detail")

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: fail_service(),
    )
    app = _AxisHarness()

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert "Unavailable" in str(
            app.query_one("#speech-status-provider-runtime", Static).renderable
        )
        assert "Unavailable" in str(
            app.query_one("#speech-status-catalog-freshness", Static).renderable
        )
        provider_copy = str(app.query_one("#tts-provider-status", Static).renderable)
        assert "Loading" not in provider_copy
        assert "private initialization detail" not in provider_copy


@pytest.mark.asyncio
async def test_voice_failure_is_stale_with_voice_recovery(
    faked_service: FakeTTSService,
) -> None:
    app = _AxisHarness()

    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                "Ready"
                in str(
                    app.query_one(
                        "#speech-status-catalog-freshness",
                        Static,
                    ).renderable
                )
            ),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        catalog = pane._catalogs["audio_cpp"]
        model_id = pane._current_select_value("#tts-model-select")
        assert isinstance(model_id, str)
        faked_service.voice_error = RuntimeError("private voice detail")

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog.revision,
            refresh=True,
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert "Stale" in str(
            app.query_one("#speech-status-catalog-freshness", Static).renderable
        )
        status = pane._runtime_status_store.catalog_status("audio_cpp", model_id)
        assert status is not None
        assert status.recovery_action is SpeechTTSNavigationIntent.REFRESH_VOICES


@pytest.mark.asyncio
async def test_unverified_exact_voice_blocks_until_server_default_is_selected(
    faked_service: FakeTTSService,
) -> None:
    """Keep an exact voice, then let the user recover with Server default."""
    from textual.widgets._select import SelectCurrent

    app = _AxisHarness()

    async with app.run_test(size=(160, 60)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                "Ready"
                in str(
                    app.query_one(
                        "#speech-status-catalog-freshness",
                        Static,
                    ).renderable
                )
            ),
        )
        pane = app.query_one(SpeechPlaygroundPane)
        catalog = pane._catalogs["audio_cpp"]
        model_id = pane._current_select_value("#tts-model-select")
        assert isinstance(model_id, str)
        voice_select = app.query_one("#tts-voice-select", Select)
        voice_select.value = "[voice]"
        await pilot.pause()
        assert voice_select.value == "[voice]"
        faked_service.voice_states[("audio_cpp", model_id)] = "unverified"

        pane._load_provider_voices(
            "audio_cpp",
            model_id,
            catalog.revision,
            refresh=True,
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert voice_select.value == "[voice]"
        assert app.query_one("#tts-generate-btn", Button).disabled is True

        voice_select.focus()
        await pilot.press("enter")
        await pilot.press("home")
        await pilot.press("enter")
        await pilot.pause()

        assert voice_select.value is SERVER_DEFAULT_VOICE_ID
        assert app.query_one("#tts-generate-btn", Button).disabled is False
        assert str(voice_select.query_one(SelectCurrent).label) == "Server default"


@pytest.mark.asyncio
async def test_a_fresh_install_seeds_no_axis_defaults_and_paints_no_markers(
    faked_service: FakeTTSService,
) -> None:
    """Contract 5: a missing preference leaves the axis absent from
    `defaults`, never substituted with a hardcoded fallback.

    Deliberately does NOT patch `get_cli_setting`/`_cli_setting` and does
    NOT use `_force_default_provider`: `Tests/conftest.py`'s sandboxed
    bootstrap config has no `[app_tts]` section at all, which is exactly
    the fresh-install condition contract 5 describes -- and the resulting
    provider selection is deterministic (`options[0][1]` = `audio_cpp`,
    since `FakeTTSService`'s descriptor order matches production's
    `adapter_bootstrap.build_default_tts_service`), not flaky, so this is a
    real assertion rather than one that happens to pass.

    An earlier version of `_seed_axis_defaults()` fabricated five defaults
    here (`openai`/`tts-1`/`alloy`/`mp3`/`1.0`) by copying
    `SpeechSettingsMixin._set_initial_values`'s form-populating fallbacks,
    which marked four axes overridden the first time the pane was ever
    opened -- against "saved defaults" that were never saved.
    """
    from tldw_chatbook.UI.STTS_Window import _seed_axis_defaults

    seeded = _seed_axis_defaults()
    assert seeded == {}, (
        f"a fresh install (no [app_tts] section) must seed no axis "
        f"defaults at all, got {seeded!r}"
    )

    app = _AxisHarness(axis_defaults=seeded)
    async with app.run_test(size=(160, 60)) as pilot:
        provider_select = app.query_one("#tts-provider-select", Select)
        await _wait_until(pilot, lambda: isinstance(provider_select.value, str))
        await pilot.pause()

        marked = [
            axis
            for axis in AXIS_CONTROLS
            if app.query_one(f"#{axis_chip_id(axis)}", Static).has_class(
                "speech-chip-override"
            )
        ]
        assert not marked, (
            f"first run painted override markers with no saved defaults: {marked}"
        )


def test_studio_provider_override_does_not_relabel_other_provider_defaults() -> None:
    """Global model/voice values belong to their provider, not every provider."""

    from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
    from tldw_chatbook.TTS.studio_preferences import (
        StudioTTSPreferencesSnapshot,
        StudioTTSSelectionOverrides,
    )
    from tldw_chatbook.UI.STTS_Window import _seed_axis_defaults

    global_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="tts-1-hd",
        voice_mode="exact",
        voice_id="shimmer",
        response_format="mp3",
        speed=1.0,
    )
    studio_preferences = StudioTTSPreferencesSnapshot(
        selection=StudioTTSSelectionOverrides(provider_id="chatterbox")
    )

    assert _seed_axis_defaults(studio_preferences, global_preferences) == {
        "tts-provider-select": "chatterbox"
    }


def _controls(
    *,
    provider_id: str = "audio_cpp",
    model_label: str = "Model A",
    selected_model_id: str | None = "model-a",
    selected_voice_id: str | None = "voice-a",
    selected_format: str | None = "wav",
    speed: float = 1.0,
) -> PlaygroundControls:
    """Build a minimal projection for a direct `_apply_controls` call."""
    return PlaygroundControls(
        provider_id=provider_id,
        model_options=((model_label, "model-a"),),
        selected_model_id=selected_model_id,
        voice_options=(("Voice A", "voice-a"),),
        selected_voice_id=selected_voice_id,
        format_options=("wav",),
        selected_format=selected_format,
        format_locked=False,
        speed=speed,
        speed_locked=False,
        generation_allowed=True,
        selection_changed=False,
    )


@pytest.mark.asyncio
async def test_reapplying_same_model_value_refreshes_its_visible_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery must remove an old trust qualifier from the closed Select.

    A catalog refresh can legitimately keep the same model ID while changing
    only its display label from ``(Unverified)`` to verified.  Textual resets
    a non-blank Select to its first value when options are replaced; assigning
    that same value again does not run the value watcher, so this specifically
    asserts the user-visible prompt rather than merely ``Select.value``.
    """
    from textual.widgets._select import SelectCurrent

    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    app = _AxisHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        model_select = app.query_one("#tts-model-select", Select)
        current = model_select.query_one(SelectCurrent)

        pane._apply_controls(
            _controls(
                model_label="supertonic-3 (Unverified)",
                selected_model_id="model-a",
            )
        )
        assert "Unverified" in str(current.label)

        pane._apply_controls(
            _controls(model_label="supertonic-3", selected_model_id="model-a")
        )

        assert str(current.label) == "supertonic-3"


@pytest.mark.asyncio
async def test_applying_controls_paints_the_override_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_apply_controls` writing a value that differs from the persisted
    default must repaint the chip -- not merely update `axis_values`.

    Calls `_apply_controls` directly with a synthetic projection (this
    acceptance criterion's own "simulate a catalog application" phrasing)
    rather than driving a full fake-service catalog load through
    `on_mount`. `_apply_controls`'s Select/Input writes ALSO post deferred
    `Select.Changed`/`Input.Changed` messages, which the pane's own edit
    handlers -- this ruling's OTHER axis-model writer -- pick up once
    `_applying_catalog_controls` has already flipped back to False; a test
    that lets those messages drain (e.g. via `pilot.pause()`) would have
    both writers cover for each other and could not isolate this one.
    Asserting synchronously, with no intervening `pilot.pause()`, means
    those deferred messages are still unprocessed in the queue -- so only
    `_apply_controls`'s own write-then-refresh can be responsible for what
    the assertions see.

    `on_mount`'s own catalog load is disabled outright (rather than merely
    left unfaked) so it cannot race this direct call either.
    """
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda self, *args, **kwargs: None,
    )
    seed = {"tts-provider-select": "openai", "tts-format-select": "mp3"}
    app = _AxisHarness(axis_values=dict(seed), axis_defaults=dict(seed))
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        provider_chip = app.query_one(f"#{axis_chip_id('tts-provider-select')}", Static)
        format_chip = app.query_one(f"#{axis_chip_id('tts-format-select')}", Static)

        # Baseline: seeded equal to the defaults, so nothing is marked yet.
        assert "*" not in provider_chip.render_line(0).text
        assert "*" not in format_chip.render_line(0).text

        pane._apply_controls(_controls(provider_id="audio_cpp", selected_format="wav"))

        assert "*" in provider_chip.render_line(0).text, (
            "provider diverges from its seeded openai default but the chip "
            "still reads 'matches the default'"
        )
        assert provider_chip.has_class("speech-chip-override")
        assert "*" in format_chip.render_line(0).text, (
            "format diverges from its seeded mp3 default but the chip "
            "still reads 'matches the default'"
        )
        assert format_chip.has_class("speech-chip-override")


@pytest.mark.asyncio
async def test_a_profile_preset_marks_the_provider_axis_as_overridden(
    monkeypatch: pytest.MonkeyPatch,
    faked_service: FakeTTSService,
) -> None:
    """Priming a preset must repaint the marker from the SCREEN's value, not
    leave it describing whatever `axis_values` held at construction.

    Seeds the provider axis matching its default -- the unmarked state an
    unprimed pane would show -- then opens on a preset for a DIFFERENT
    provider. Before this ruling `_prime_profile_preset_controls` wrote only
    the Select widget, so the chip kept reporting "matches the saved
    default" even though priming had just switched the provider under it --
    which is the axis row's entire reason to exist for a preset-opened pane.
    """
    _force_default_provider(monkeypatch, "audio_cpp")
    preset = _profile_preset()  # provider_id="audio_cpp"
    app = _AxisHarness(
        profile_preset=preset,
        axis_values={"tts-provider-select": "openai"},
        axis_defaults={"tts-provider-select": "openai"},
    )
    async with app.run_test(size=(160, 60)) as pilot:
        provider_select = app.query_one("#tts-provider-select", Select)
        await _wait_until(pilot, lambda: provider_select.value == "audio_cpp")
        await pilot.pause()

        chip = app.query_one(f"#{axis_chip_id('tts-provider-select')}", Static)
        assert "*" in chip.render_line(0).text, (
            "priming switched the provider to the preset's audio_cpp but "
            "the chip still reads 'matches the saved default'"
        )
        assert chip.has_class("speech-chip-override")


@pytest.mark.asyncio
async def test_a_user_edit_updates_the_marker(
    monkeypatch: pytest.MonkeyPatch,
    faked_service: FakeTTSService,
) -> None:
    """A user's own edit, not just a catalog application, must update the
    marker -- `_apply_controls` is not the only axis-value writer.

    Seeds the speed axis matching its default (1.0, which is also what a
    fresh `openai` catalog application resolves to, so the marker is
    legitimately off going into the edit -- not merely stale-and-coincidentally
    right). Uses `openai`, unlike `audio_cpp`, which locks speed to 1.0 and
    would make the edit a no-op.
    """
    _force_default_provider(monkeypatch, "openai")
    app = _AxisHarness(
        axis_values={"tts-speed-input": "1.0"},
        axis_defaults={"tts-speed-input": "1.0"},
    )
    async with app.run_test(size=(160, 60)) as pilot:
        pane = app.query_one(SpeechPlaygroundPane)
        model_select = app.query_one("#tts-model-select", Select)
        await _wait_until(pilot, lambda: isinstance(model_select.value, str))
        await pilot.pause()

        chip = app.query_one(f"#{axis_chip_id('tts-speed-input')}", Static)
        assert "*" not in chip.render_line(0).text, (
            "sanity check: the resolved speed still matches its default here"
        )

        speed_input = app.query_one("#tts-speed-input", Input)
        speed_input.value = "1.7"
        await pilot.pause()

        assert "*" in chip.render_line(0).text, (
            "user edited the speed axis but the chip still reads 'matches "
            "the saved default'"
        )
        assert chip.has_class("speech-chip-override")
        assert pane.axis_values.get("tts-speed-input") == "1.7"


#: (provider_id, model_id, voice_id) for the two provider classes this
#: pane must distinguish: audio.cpp has real catalog authority, so its
#: "unverified" is transient; the six legacy-bridge providers (openai
#: stands in for all of them here) have none, so theirs is permanent.
#: Matches each service's own catalog exactly, so `profile_availability_
#: from_catalog` resolves both to "unverified" rather than "unavailable".
_ADOPTED_PRESET_PROVIDER_CASES = (
    ("audio_cpp", "<opaque:model>", "[voice]"),
    ("openai", "tts-1", "alloy"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "model_id", "voice_id"),
    _ADOPTED_PRESET_PROVIDER_CASES,
    ids=("audio_cpp", "openai"),
)
async def test_adopted_preset_unverified_copy_distinguishes_no_catalog_legacy(
    faked_service: FakeTTSService,
    provider_id: str,
    model_id: str,
    voice_id: str,
) -> None:
    """Task 3's two highest-traffic adoption sites: the provider-status line
    (`_apply_controls`) and the preview banner (`_sync_profile_preview_status`).

    Before this, both read identically for a legacy preset and an
    audio.cpp preset once either settled "unverified" -- "Profile
    availability is unverified. Generate makes one exact attempt without
    fallback and shows a warning." implies a transient state Refresh can
    resolve, which is false for a provider with no catalog to refresh
    against. A legacy preset must instead read the permanent "no catalog
    check" story; audio.cpp's transient story is unchanged.
    """
    preset = _profile_preset(
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        availability="unverified",
    )
    app = _AxisHarness(profile_preset=preset)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        pane = app.query_one(SpeechPlaygroundPane)
        assert pane._profile_effective_availability == "unverified"

        status = str(app.query_one("#tts-provider-status", Static).renderable).lower()
        banner = str(
            app.query_one("#tts-profile-preview-status", Static).renderable
        ).lower()

        if provider_id == "audio_cpp":
            assert "unverified" in status
            assert "no catalog check" not in status
            assert "unverified" in banner
            assert "no catalog check" not in banner
        else:
            assert "no catalog check" in status
            assert "unverified" not in status
            assert "no catalog check" in banner
            assert "unverified" not in banner
        # The behavioral tail -- one exact attempt, no fallback -- is true
        # for both provider classes and must survive the copy split.
        assert "generate makes one exact attempt" in status
        assert "without fallback" in status
        assert "generate makes one exact attempt" in banner
        assert "without fallback" in banner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "model_id", "voice_id"),
    _ADOPTED_PRESET_PROVIDER_CASES,
    ids=("audio_cpp", "openai"),
)
async def test_reconfiguring_voice_failure_status_distinguishes_no_catalog_legacy(
    faked_service: FakeTTSService,
    provider_id: str,
    model_id: str,
    voice_id: str,
) -> None:
    """`_load_provider_voices_worker`'s reconfiguring branch (task 3 site 2)
    must not promise a legacy provider's permanent no-catalog state will
    resolve on retry, the way it honestly can for audio.cpp.
    """
    preset = _profile_preset(
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        availability="available",
    )
    app = _AxisHarness(profile_preset=preset)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        catalog = pane._catalogs[provider_id]

        faked_service.voice_error = TTSProviderReconfiguringError(
            "private reconfiguring detail"
        )
        pane._load_provider_voices(
            provider_id,
            model_id,
            catalog.revision,
            refresh=True,
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#tts-provider-status", Static).renderable).lower()
        if provider_id == "audio_cpp":
            assert "unverified" in status
            assert "no catalog check" not in status
        else:
            assert "no catalog check" in status
            assert "unverified" not in status
        assert "private reconfiguring detail" not in status
        assert "generate makes one exact attempt" in status
        assert "shows a warning" in status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "model_id", "voice_id"),
    _ADOPTED_PRESET_PROVIDER_CASES,
    ids=("audio_cpp", "openai"),
)
async def test_generic_voice_failure_status_distinguishes_no_catalog_legacy(
    faked_service: FakeTTSService,
    provider_id: str,
    model_id: str,
    voice_id: str,
) -> None:
    """`_load_provider_voices_worker`'s generic-exception branch (task 3
    site 3): the "the exact selection remains selected without fallback"
    line must carry the same no-catalog distinction, not just the two
    "Generate makes one exact attempt" sites.
    """
    preset = _profile_preset(
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        availability="available",
    )
    app = _AxisHarness(profile_preset=preset)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        pane = app.query_one(SpeechPlaygroundPane)
        catalog = pane._catalogs[provider_id]

        faked_service.voice_error = RuntimeError("private upstream detail")
        pane._load_provider_voices(
            provider_id,
            model_id,
            catalog.revision,
            refresh=True,
        )
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#tts-provider-status", Static).renderable).lower()
        if provider_id == "audio_cpp":
            assert "unverified" in status
            assert "no catalog check" not in status
        else:
            assert "no catalog check" in status
            assert "unverified" not in status
        assert "private upstream detail" not in status
        assert "without fallback" in status
