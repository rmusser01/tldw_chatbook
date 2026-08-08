"""task-3304 (MI-07/08/17): ingest structural fixes on the Library canvas.

Three finding families from the 2026-08-07 Media Ingestion review, each
pinned here against the real defect:

- MI-07 disabled legibility: every ``enabled_when_values``-gated field
  (Parakeet model folder / transcription model under provider ``default``,
  web page/depth limits under ``individual``) and the Parakeet install
  button rendered identically to enabled controls -- no stated reason, and
  the only visual difference was Textual's app-default ``opacity: 0.7``
  fade (the documented all-themes-below-3:1 disabled-contrast trap family).
  The fix is two-part: a reason annotation at the control derived from the
  schema's gate metadata, and app-tier Legible Disabled styling (the
  DESIGN.md TASK-1801 rule -- app stylesheet, never widget DEFAULT_CSS).
- MI-08 receipt into view: after Start the queue's outcome area sat below
  the fold on every submit, and the ``VerticalScroll`` canvas had no fold
  indicator (task-1623 convention).
- MI-17 clipped install command: the missing-dependency warning's pip
  command was only copyable from inside the guardrail modal.

Geometry/legibility tests run under a CSS-true host (the established
``_CssTrueConsoleHarness`` pattern): a bare ``App`` loads none of the app
stylesheet, so contrast/paint assertions made there are void.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, Select, Static

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
    LibraryIngestCanvas,
)


_INSTALLED_PATCH = "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed"


@pytest.fixture(autouse=True)
def _stub_cli_settings(monkeypatch):
    """Keep screen-level tests off the real on-disk CLI config.

    Same isolation as ``test_library_shell``'s autouse stub (autouse
    fixtures do not travel with imports, so this module needs its own).
    """
    monkeypatch.setattr(
        library_screen_module, "get_cli_setting", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        library_screen_module,
        "save_setting_to_cli_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        library_screen_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: None,
    )


class _CanvasHost(App):
    """Bare host: message/behaviour assertions only (no app CSS)."""

    def __init__(self, state) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")


class _CssTrueCanvasHost(_CanvasHost):
    """Canvas host that loads the real app CSS bundle.

    Required for any paint/legibility assertion: the app-tier disabled
    rules under test here live in the bundle, which a bare ``App`` never
    loads (the ``_CssTrueConsoleHarness`` lesson).
    """

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )


def _preflight(type_groups: dict[str, list[str]], warnings=None) -> PreflightResult:
    return PreflightResult(
        type_groups=type_groups,
        warnings=list(warnings or []),
        errors=[],
        total_size=0,
        truncated=False,
        total_files=sum(len(files) for files in type_groups.values()),
    )


def _audio_state(
    *,
    provider: str = "default",
    model_dir: str = "/models/parakeet-v2",
    expanded: bool = True,
):
    form = LibraryIngestFormState(path="/tmp/talk.mp3")
    form.type_options = {
        "audio_video": {
            "transcription_provider": provider,
            "transcription_model_dir": model_dir,
        }
    }
    if expanded:
        form.expanded_type_groups = {"audio_video"}
    return build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight({"audio_video": ["/tmp/talk.mp3"]}),
    )


def _panel_texts(app: App, group: str) -> list[str]:
    """All rendered text carriers inside one options panel."""
    panel = app.query_one(f"#type-group-{group}")
    texts = [
        str(getattr(widget, "renderable", "")) for widget in panel.query(Static)
    ]
    texts.extend(str(widget.label) for widget in panel.query(Checkbox))
    texts.extend(str(widget.label) for widget in panel.query(Button))
    return texts


# --- MI-07: reason annotation at the control --------------------------------


@pytest.mark.asyncio
async def test_value_gated_audio_fields_state_their_reason_when_disabled():
    """Under provider ``default`` the Parakeet folder and the
    faster-whisper model select are schema-disabled -- their labels must
    say why, not render identically to enabled fields (MI-07)."""
    app = _CanvasHost(_audio_state(provider="default"))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test() as pilot:
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            model = pilot.app.query_one(
                "#opt-audio_video-transcription_model", Select
            )
            assert model_dir.disabled is True, "precondition: gate closed"
            assert model.disabled is True, "precondition: gate closed"
            texts = _panel_texts(pilot.app, "audio_video")
            assert any(
                "Local Parakeet model folder — needs the parakeet-onnx provider"
                in text
                for text in texts
            ), f"no reason at the model-folder control; panel texts: {texts!r}"
            assert any(
                "Transcription model — needs the faster-whisper provider" in text
                for text in texts
            ), f"no reason at the model control; panel texts: {texts!r}"


@pytest.mark.asyncio
async def test_web_limits_state_single_page_reason_when_disabled():
    """``max_pages``/``max_depth`` under the single-page fetch method are
    schema-disabled; their labels must carry the reason (MI-07)."""
    form = LibraryIngestFormState(path="https://example.com")
    form.expanded_type_groups = {"web"}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight({"web": ["https://example.com"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert (
            pilot.app.query_one("#opt-web-max_pages", Input).disabled is True
        ), "precondition: single-page method gates the limits"
        texts = _panel_texts(pilot.app, "web")
        assert any(
            "Maximum pages — single-page fetch selected" in text for text in texts
        ), f"no reason at the max-pages control; panel texts: {texts!r}"
        assert any(
            "Maximum depth — single-page fetch selected" in text for text in texts
        ), f"no reason at the max-depth control; panel texts: {texts!r}"


@pytest.mark.asyncio
async def test_enabled_fields_carry_no_disabled_reason():
    """The annotation is a DISABLED-state marker: once the gate opens the
    label returns to its plain form (and the sibling that is still gated
    keeps its reason)."""
    app = _CanvasHost(_audio_state(provider="faster-whisper"))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test() as pilot:
            model = pilot.app.query_one(
                "#opt-audio_video-transcription_model", Select
            )
            assert model.disabled is False
            texts = _panel_texts(pilot.app, "audio_video")
            assert any(text == "Transcription model" for text in texts), (
                f"enabled field label must be plain; panel texts: {texts!r}"
            )
            assert any(
                "Local Parakeet model folder — needs the parakeet-onnx provider"
                in text
                for text in texts
            )


@pytest.mark.asyncio
async def test_gate_hint_checkboxes_are_not_double_annotated():
    """task-3303 already bakes the gate into some labels ("Enable OCR
    (docling or docext engines only)") -- the disabled state must not
    append a second, redundant reason to those."""
    form = LibraryIngestFormState(path="/tmp/doc.pdf")
    form.expanded_type_groups = {"pdf"}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight({"pdf": ["/tmp/doc.pdf"]}),
    )
    app = _CanvasHost(state)
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test() as pilot:
            ocr = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            assert ocr.disabled is True, (
                "precondition: default engine (pymupdf4llm) gates OCR"
            )
            assert str(ocr.label) == "Enable OCR (docling or docext engines only)"


@pytest.mark.asyncio
async def test_install_button_states_reason_only_while_gated():
    """The Parakeet install button is an inert action under any other
    provider -- DESIGN.md's Inert-actions rule requires the reason in the
    label, never dimming alone (MI-07)."""
    app = _CanvasHost(_audio_state(provider="default"))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test() as pilot:
            button = pilot.app.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            )
            assert button.disabled is True
            assert str(button.label).endswith("— needs the parakeet-onnx provider")

    app = _CanvasHost(_audio_state(provider="parakeet-onnx"))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test() as pilot:
            button = pilot.app.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            )
            assert button.disabled is False
            assert "needs the parakeet-onnx provider" not in str(button.label)


# --- MI-07: legible disabled paint (CSS-true) --------------------------------


def _relative_luminance(color) -> float:
    """WCAG relative luminance of a Rich ``Color``."""
    triplet = color.get_truecolor()

    def _channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _channel(triplet.red)
        + 0.7152 * _channel(triplet.green)
        + 0.0722 * _channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    """WCAG contrast ratio between two Rich colours."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App, region, needle: str):
    """The compositor style painting the first occurrence of ``needle``.

    Reads the compositor's own strips (what actually reached the screen),
    not ``styles.color`` -- a declared colour halved by a dimmer downstream
    is exactly the defect family under test.
    """
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.y + region.height):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


@pytest.mark.asyncio
async def test_schema_disabled_fields_paint_legibly_inert():
    """Legible Disabled rule (TASK-1801): a disabled control's value must
    paint at >= 3:1 against its own background while still reading dimmer
    than an enabled sibling. The app-default ``*:disabled:can-focus``
    fade must be neutralised by the app-tier rule (its removal is the
    mutation this test guards)."""
    app = _CssTrueCanvasHost(_audio_state(provider="default"))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test(size=(120, 46)) as pilot:
            await pilot.pause()
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            language = pilot.app.query_one("#opt-audio_video-language", Input)
            assert model_dir.disabled and not language.disabled

            # The app-tier rule must neutralise the whole-widget fade --
            # Textual's ``*:disabled:can-focus { opacity: 0.7 }`` -- and
            # state the colours instead (a faded stack is how every other
            # disabled surface measured below 3:1).
            assert model_dir.styles.opacity == 1.0, (
                f"disabled field still faded: opacity={model_dir.styles.opacity}"
            )

            disabled_style = _painted_style_of_text(
                pilot.app, model_dir.region, "/models/parakeet-v2"
            )
            enabled_style = _painted_style_of_text(
                pilot.app, language.region, "en"
            )
            assert disabled_style is not None and disabled_style.color is not None
            assert enabled_style is not None and enabled_style.color is not None

            # Measured 2026-08-07 on this harness: disabled 7.25:1,
            # enabled sibling 12.63:1 (pre-fix fade: 6.77:1 on an
            # identical background with no stated reason).
            ratio = _contrast(disabled_style.color, disabled_style.bgcolor)
            assert ratio >= 3.0, (
                f"disabled value paints at {ratio:.2f}:1 -- below the "
                "Legible Disabled floor"
            )
            # Still visibly the dimmer state: not the enabled ink.
            assert (
                disabled_style.color.get_truecolor()
                != enabled_style.color.get_truecolor()
            ), "disabled and enabled values paint in the same ink"

            # The select under the same gate gets the same treatment.
            model = pilot.app.query_one(
                "#opt-audio_video-transcription_model", Select
            )
            select_style = _painted_style_of_text(pilot.app, model.region, "base")
            assert select_style is not None and select_style.color is not None
            select_ratio = _contrast(select_style.color, select_style.bgcolor)
            assert select_ratio >= 3.0, (
                f"disabled select paints at {select_ratio:.2f}:1"
            )


# --- MI-08: fold indicator ----------------------------------------------------


@pytest.mark.asyncio
async def test_fold_hint_shows_only_while_canvas_overflows():
    """task-1623 convention: a reserved bottom row says more content
    exists, shown only while the canvas actually overflows."""
    # Small terminal: the default form alone overflows the viewport.
    app = _CssTrueCanvasHost(_audio_state(expanded=True))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test(size=(100, 16)) as pilot:
            await pilot.pause()
            canvas = pilot.app.query_one(LibraryIngestCanvas)
            hint = pilot.app.query_one("#library-ingest-fold-hint", Static)
            assert canvas.virtual_size.height > canvas.container_size.height, (
                "precondition: content overflows at this size"
            )
            assert hint.display is True
            assert "▼ more" in str(hint.renderable)

    # Tall terminal with a minimal form: no overflow, no hint row.
    minimal = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="")
    )
    app = _CssTrueCanvasHost(minimal)
    async with app.run_test(size=(120, 46)) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryIngestCanvas)
        hint = pilot.app.query_one("#library-ingest-fold-hint", Static)
        assert canvas.virtual_size.height <= canvas.container_size.height, (
            "precondition: content fits at this size"
        )
        assert hint.display is False


@pytest.mark.asyncio
async def test_fold_hint_is_pinned_not_scrolled():
    """The hint is chrome, not content: it must hold the canvas's bottom
    row across scrolling instead of scrolling away with the form."""
    app = _CssTrueCanvasHost(_audio_state(expanded=True))
    with patch(_INSTALLED_PATCH, return_value=True):
        async with app.run_test(size=(100, 16)) as pilot:
            await pilot.pause()
            canvas = pilot.app.query_one(LibraryIngestCanvas)
            hint = pilot.app.query_one("#library-ingest-fold-hint", Static)
            assert hint.display is True
            region_before = hint.region
            canvas.scroll_end(animate=False)
            await pilot.pause()
            assert hint.region == region_before, (
                "fold hint scrolled with the content instead of staying pinned"
            )


# --- MI-08: receipt into view on submit (screen-level) ------------------------


def _screen_harness():
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    return LibraryHarness(app)


@pytest.mark.asyncio
async def test_submit_brings_the_queue_heading_into_view(monkeypatch):
    """MI-08: after Start, the outcome area must not be left below the
    fold -- the canvas scrolls the queue heading into view."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
    from Tests.UI.test_library_shell import (
        _wait_for_library_shell,
        _wait_for_selector,
    )

    host = _screen_harness()
    submitted: list[str] = []

    async with host.run_test(size=(170, 30)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        monkeypatch.setattr(
            host.app_instance,
            "submit_library_ingest_job",
            lambda **kwargs: submitted.append(kwargs.get("source_path", "")),
            raising=False,
        )
        await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await pilot.pause()

        canvas = screen.query_one(LibraryIngestCanvas)
        canvas.scroll_to(y=0, animate=False, force=True)
        await pilot.pause()
        heading = screen.query_one("#library-ingest-queue-heading", Static)
        assert (
            heading.virtual_region.y
            >= canvas.scroll_offset.y + canvas.container_size.height
        ), "precondition: at this height the queue heading starts below the fold"

        screen._do_submit_ingest("/tmp/anything.txt")
        await pilot.pause()
        await pilot.pause()

        assert submitted == ["/tmp/anything.txt"]
        canvas = screen.query_one(LibraryIngestCanvas)
        heading = screen.query_one("#library-ingest-queue-heading", Static)
        top = canvas.scroll_offset.y
        assert (
            top
            <= heading.virtual_region.y
            < top + canvas.container_size.height
        ), (
            f"queue heading (virtual y={heading.virtual_region.y}) still out of "
            f"view after submit (scroll y={top}, "
            f"viewport={canvas.container_size.height})"
        )

        # The fold hint is display-managed by the updater and must keep
        # object identity across the in-place hot path (task-2042
        # discipline): the dynamic-regions update must not remount it.
        hint_before = screen.query_one("#library-ingest-fold-hint", Static)
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        assert screen.query_one("#library-ingest-fold-hint", Static) is hint_before


# --- MI-17: install command recoverable at the warning -------------------------


_LONG_COMMAND = 'pip install -e ".[transcription_lightning_whisper]"'


def _warning_state():
    form = LibraryIngestFormState(path="/tmp/talk.mp3")
    return build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight(
            {"audio_video": ["/tmp/talk.mp3"]},
            warnings=[
                {
                    "feature": "lightning_whisper_mlx",
                    "label": "Lightning Whisper MLX",
                    "hint": "audio transcription",
                    "command": _LONG_COMMAND,
                }
            ],
        ),
    )


@pytest.mark.asyncio
async def test_warning_command_paints_unclipped_in_the_summary():
    """MI-17: the pip command must be readable at the warning itself --
    wrapped onto further rows rather than clipped at the canvas edge."""
    app = _CssTrueCanvasHost(_warning_state())
    async with app.run_test(size=(80, 46)) as pilot:
        await pilot.pause()
        warning = pilot.app.query_one("#ingest-preflight-warning-0", Static)
        strips = list(pilot.app.screen._compositor.render_strips())
        painted = "".join(
            "".join(segment.text for segment in strips[y]._segments).strip() + " "
            for y in range(
                warning.region.y, warning.region.y + warning.region.height
            )
            if y < len(strips)
        )
        # The command's tail must survive paint; pre-fix it clipped at the
        # canvas edge mid-token ("...[transcription_lig").
        assert "transcription_lightning_whisper" in painted.replace(" ", ""), (
            f"command tail not painted; warning rows: {painted!r}"
        )


@pytest.mark.asyncio
async def test_summary_offers_copy_command_and_copies_it():
    """MI-17: a compact copy affordance sits AT the warning (consistent
    with the guardrail modal's) -- the modal must no longer be the only
    place the command can be copied from."""
    copied: list[str] = []

    class _ClipboardHost(_CanvasHost):
        def copy_to_clipboard(self, text: str) -> None:  # noqa: D102
            copied.append(text)

    app = _ClipboardHost(_warning_state())
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#ingest-preflight-copy-command-0", Button)
        assert "cop" in str(button.label).lower()
        button.press()
        await pilot.pause()

    assert copied == [_LONG_COMMAND]
