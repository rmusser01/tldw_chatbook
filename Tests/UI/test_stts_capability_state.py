import pytest
from textual.app import App
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Lab_Modules.lab_speech_status import speech_capability_detail
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
)
from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# The capability line moved out of STTSWindow's sidebar and into the Lab
# frame's rail when Speech adopted the frame, so these mount the SCREEN now.
# Its id and the recovery-taxonomy copy are unchanged -- that is the point of
# asserting it here rather than deleting the coverage with the sidebar.
# Fresh module-presence probes are patched on lab_speech_status, which owns
# the visible capability snapshot.


class _SpeechHarness(App):
    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self, app_instance):
        super().__init__()
        self._app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(STTSScreen(self._app_instance))


async def _wait_until(pilot, predicate, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_speech_rail_exposes_and_opens_voice_profiles():
    app = _SpeechHarness(_build_test_app())

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, STTSScreen)
        await _wait_until(
            pilot,
            lambda: screen.stts_window is not None,
        )

        profile_row = screen.query_one("#lab-speech-row-profiles", Button)
        assert "Voice Profiles" in str(profile_row.label)

        await pilot.click("#lab-speech-row-profiles")
        await _wait_until(
            pilot,
            lambda: len(screen.query(STTSProfileLibrary)) == 1,
        )

        assert screen.stts_window is not None
        assert screen.stts_window.current_view == "profiles"
        assert profile_row.has_class("is-active")


@pytest.mark.asyncio
async def test_stts_window_explains_missing_local_speech_dependencies(monkeypatch):
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "tts_processing", False)
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "stt_processing", False)
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "kokoro_onnx", False)
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "chatterbox", False)
    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "higgs_tts", False)
    monkeypatch.setattr(
        "tldw_chatbook.UI.Lab_Modules.lab_speech_status.find_spec",
        lambda _module_name: None,
    )

    app = _SpeechHarness(_build_test_app())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        status = app.screen.query_one("#speech-capability-status", Static)
        rendered_status = str(status.render())

        assert "OpenAI-compatible speech: available when configured" in rendered_status
        assert (
            'Local transcription: missing - pip install '
            '"tldw_chatbook[transcription_faster_whisper]"'
        ) in rendered_status
        assert (
            'Local Kokoro: missing - pip install "tldw_chatbook[local_tts]"'
        ) in rendered_status
        assert (
            'Local Chatterbox: missing - pip install "tldw_chatbook[chatterbox]"'
        ) in rendered_status
        assert (
            'Local Higgs: missing - pip install "tldw_chatbook[higgs_tts]"'
        ) in rendered_status
        assert app.screen.lab_header_state().status == "ready"


def test_speech_capability_detail_names_each_ready_local_capability():
    detail = speech_capability_detail(
        SpeechLocalDependencyAvailability.all_available()
    )

    assert "Local transcription: ready" in detail
    assert "Local Kokoro: ready" in detail
    assert "Local Chatterbox: ready" in detail
    assert "Local Higgs: ready" in detail


def test_visible_capability_refresh_ignores_stale_stt_cache(monkeypatch):
    for dependency in (
        "stt_processing",
        "faster_whisper",
        "kokoro_onnx",
        "chatterbox",
        "higgs_tts",
    ):
        monkeypatch.setitem(DEPENDENCIES_AVAILABLE, dependency, False)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Lab_Modules.lab_speech_status.find_spec",
        lambda module_name: object() if module_name == "faster_whisper" else None,
    )

    assert "Local transcription: ready" in speech_capability_detail()


@pytest.mark.asyncio
async def test_speech_surfaces_share_fresh_local_dependency_snapshot(monkeypatch):
    for dependency in (
        "tts_processing",
        "stt_processing",
        "faster_whisper",
        "kokoro_onnx",
        "chatterbox",
        "higgs_tts",
    ):
        monkeypatch.setitem(DEPENDENCIES_AVAILABLE, dependency, False)

    installed_modules = {
        "faster_whisper",
        "kokoro_onnx",
        "chatterbox",
        "boson_multimodal",
    }
    monkeypatch.setattr(
        "tldw_chatbook.UI.Lab_Modules.lab_speech_status.find_spec",
        lambda module_name: object() if module_name in installed_modules else None,
    )
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_load_provider_catalog",
        lambda _self, *_args, **_kwargs: None,
    )

    app = _SpeechHarness(_build_test_app())

    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_until(
            pilot,
            lambda: bool(app.screen.query("#speech-status-stt-dependency")),
        )

        # Rail, inspector, and deferred Playground must project one coherent
        # fresh snapshot instead of mixing fresh probes with stale globals.
        summary = app.screen.query_one("#speech-capability-summary", Static)
        assert str(summary.render()) == "Remote TTS | Local 4/4"

        detail = app.screen.query_one("#speech-capability-status", Static)
        rendered_detail = str(detail.render())
        assert "Local transcription: ready" in rendered_detail
        assert "Local Kokoro: ready" in rendered_detail
        assert "Local Chatterbox: ready" in rendered_detail
        assert "Local Higgs: ready" in rendered_detail

        for row_id in (
            "stt-dependency",
            "kokoro-dependency",
            "chatterbox-dependency",
            "higgs-dependency",
        ):
            row = app.screen.query_one(f"#speech-status-{row_id}", Static)
            assert "Ready" in str(row.renderable)


@pytest.mark.asyncio
async def test_speech_capability_summary_stays_visible_at_minimum_terminal(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.UI.Lab_Modules.lab_speech_status.find_spec",
        lambda _module_name: None,
    )
    app = _SpeechHarness(_build_test_app())

    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_until(
            pilot,
            lambda: bool(app.screen.query("#speech-capability-summary")),
        )
        await pilot.pause()

        rail = app.screen.query_one("#lab-rail")
        summary = app.screen.query_one("#speech-capability-summary", Static)
        assert summary.region.height == 1
        assert summary.region.bottom <= rail.region.bottom


@pytest.mark.asyncio
async def test_speech_screen_mounts_rail_rows_exactly_once():
    """TASK-2610 regression: one Mount event must yield exactly one rail.

    Textual dispatches every ``on_mount`` along the MRO for a single Mount
    event. STTSScreen used to call ``super().on_mount()`` on top of that,
    running the Lab frame's handler twice — double-mounting the rail rows and
    crashing the app with ``DuplicateIds`` on every visit to Lab > Speech.
    """
    app = _SpeechHarness(_build_test_app())

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, STTSScreen)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#lab-speech-row-playground")),
        )

        assert len(screen.query("#lab-speech-row-playground")) == 1

        # The Speech screen's combined footer hints must win over the Lab
        # frame's plain set — the ordering the removed super() call was
        # (wrongly) trying to guarantee, now owned by _lab_footer_registration.
        source, shortcuts = screen._footer_shortcut_registration
        assert source == "stts"
        assert shortcuts == screen.STTS_SHORTCUTS + screen.LAB_FOOTER_SHORTCUTS
