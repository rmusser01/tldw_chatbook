"""The assembled Playground: fold position, truncation, and containment."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_stts_playground_audio_cpp import (
    FakeTTSService,
    _profile_preset,
    _resolved,
    _wait_until,
)
from tldw_chatbook.UI.Speech.speech_axis_row import axis_chip_id
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.stts_playground_catalog import PlaygroundControls


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
                    escaped.append((strip_id, str(button.label)))
        assert not escaped, f"clipped by container at {size}: {escaped}"


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
        assert screen.query_one("#speech-result-history")
        assert screen.query_one("#speech-param-group")


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
    """Make the pane's default-provider read deterministic.

    Without this the initial provider selection falls through to whatever
    the sandboxed test config happens to contain (usually the first
    descriptor, but not guaranteed), which would make the override
    assertions below flaky rather than a property of the code.
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


def _controls(
    *,
    provider_id: str = "audio_cpp",
    selected_model_id: str | None = "model-a",
    selected_voice_id: str | None = "voice-a",
    selected_format: str | None = "wav",
    speed: float = 1.0,
) -> PlaygroundControls:
    """Build a minimal projection for a direct `_apply_controls` call."""
    return PlaygroundControls(
        provider_id=provider_id,
        model_options=(("Model A", "model-a"),),
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
        provider_chip = app.query_one(
            f"#{axis_chip_id('tts-provider-select')}", Static
        )
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
