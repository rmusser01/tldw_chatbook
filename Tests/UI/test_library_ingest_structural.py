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
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
)
from textual.widgets._collapsible import CollapsibleTitle

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
    INGEST_PATH_LABEL_COPY,
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

            # task-15790: the 15513 Import-behavior controls grew the panel,
            # pushing Language below the 46-row viewport -- an off-screen
            # widget paints nothing, so the style probe returned None. Bring
            # each field on-screen before reading its painted style; the
            # contrast contract itself is unchanged.
            model_dir.scroll_visible(animate=False)
            await pilot.pause()
            disabled_style = _painted_style_of_text(
                pilot.app, model_dir.region, "/models/parakeet-v2"
            )
            language.scroll_visible(animate=False)
            await pilot.pause()
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
            # (task-3305) The select renders its display label now, not
            # the raw "base" token.
            select_style = _painted_style_of_text(
                pilot.app, model.region, "Base (fast)"
            )
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
        # (task-14822) The per-warning detail now lives behind a fold --
        # open it, because "unclipped once you can see it" is the claim.
        pilot.app.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        ).collapsed = False
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
    place the command can be copied from.

    (xhigh review round, G5) This selection has exactly ONE missing extra,
    which is now served by the single always-visible control rather than
    by that control plus an identical per-extra button inside the fold.
    """
    copied: list[str] = []

    class _ClipboardHost(_CanvasHost):
        def copy_to_clipboard(self, text: str) -> None:  # noqa: D102
            copied.append(text)

    app = _ClipboardHost(_warning_state())
    async with app.run_test() as pilot:
        buttons = list(pilot.app.query(".ingest-preflight-copy-command"))
        assert len(buttons) == 1, [button.id for button in buttons]
        button = buttons[0]
        assert "cop" in str(button.label).lower()
        button.press()
        await pilot.pause()

    assert copied == [_LONG_COMMAND]


# --- task-14822: the folded warning block no longer owns the viewport -------


def _many_warnings(count: int) -> list[dict[str, str]]:
    return [
        {
            "feature": f"feature_{index}",
            "label": f"Backend {index}",
            "hint": f"capability {index}",
            "command": f'pip install -e ".[extra{index}]"',
        }
        for index in range(count)
    ]


def _mixed_folder_state(warning_count: int):
    """The archetypal mixed folder: supported + unsupported + empty files."""
    form = LibraryIngestFormState(path="/tmp/mixed")
    return build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={
                "generic": ["/tmp/mixed/a.txt"],
                "unsupported": ["/tmp/mixed/b.xyz"],
            },
            warnings=_many_warnings(warning_count),
            errors=[],
            total_size=10,
            truncated=False,
            total_files=2,
            empty_files=["/tmp/mixed/c.txt"],
        ),
    )


@pytest.mark.asyncio
async def test_summary_block_height_does_not_grow_with_the_warning_count():
    """AC#2: eleven warnings used to cost ~22 rows plus nine stacked copy
    buttons -- the whole 52-row viewport. Folded, the block is the same
    height whether there are two warnings or eleven."""
    heights = {}
    for count in (2, 11):
        app = _CssTrueCanvasHost(_mixed_folder_state(count))
        async with app.run_test(size=(80, 52)) as pilot:
            await pilot.pause()
            heights[count] = pilot.app.query_one(
                "#library-ingest-preflight-summary"
            ).region.height
    assert heights[2] == heights[11], (
        f"the warning block still scales with the warning count: {heights}"
    )
    assert heights[11] <= 16, (
        f"folded summary block is still a wall: {heights[11]} rows"
    )


@pytest.mark.asyncio
async def test_breakdown_and_start_are_in_view_behind_eleven_warnings():
    """AC#2: with warnings present, the type breakdown AND the Start
    affordance are on screen at a supported terminal size -- the pre-fix
    wall pushed both below the fold."""
    app = _CssTrueCanvasHost(_mixed_folder_state(11))
    async with app.run_test(size=(80, 52)) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryIngestCanvas)
        viewport = canvas.region.height
        breakdown = pilot.app.query_one("#ingest-type-breakdown")
        start = pilot.app.query_one("#library-ingest-start")
        assert 0 < breakdown.region.y < viewport, (
            f"type breakdown below the fold: y={breakdown.region.y} "
            f"viewport={viewport}"
        )
        assert 0 < start.region.y < viewport, (
            f"Start below the fold: y={start.region.y} viewport={viewport}"
        )


@pytest.mark.asyncio
async def test_outcome_lines_paint_heavier_than_the_tooling_summary():
    """AC#3: "1 empty file will fail" is an OUTCOME of this selection; the
    tooling summary is a fact about the install. They must not paint at
    the same weight (they shared one class and one colour)."""
    app = _CssTrueCanvasHost(_mixed_folder_state(11))
    async with app.run_test(size=(80, 52)) as pilot:
        await pilot.pause()
        tooling = pilot.app.query_one("#ingest-preflight-tooling-summary", Static)
        empty = pilot.app.query_one("#ingest-empty-summary", Static)
        tooling_style = _painted_style_of_text(
            pilot.app, tooling.region, "optional"
        )
        empty_style = _painted_style_of_text(pilot.app, empty.region, "empty")
        assert tooling_style is not None and empty_style is not None
        assert bool(empty_style.bold) and not bool(tooling_style.bold), (
            "outcome and environment lines still share one weight: "
            f"outcome bold={empty_style.bold} tooling bold={tooling_style.bold}"
        )
        assert empty_style.color != tooling_style.color, (
            "outcome and environment lines still share one colour"
        )


# --- task-14824: accessibility residue --------------------------------------
#
# (a) `#opt-generic-encoding` focus was COLOUR-ONLY: a per-focusable Tab walk
# found the focused and unfocused plain-text captures byte-identical, at
# 1.12:1 between the two backgrounds. `LibraryIngestCanvas Select:focus {
# outline: heavy $accent }` had been declared since task-2014 and never
# reached the screen: `SelectCurrent` is an opaque child that covers its
# parent's ENTIRE region, and the compositor paints it over the parent's
# outline. Any assertion made from `Select.render_lines()` passes anyway --
# that call renders the widget in isolation, so the covering child is absent.
# These captures come from the compositor.

#: The heavy box-drawing family. None of these appear in the unfocused
#: `tall`-border rendering ("▊", "▔", "▎", "▁").
HEAVY_GLYPHS = ("┏", "┓", "┗", "┛", "━", "┃")


def _composited_rows(app: App, widget) -> list[str]:
    """The widget's region as the COMPOSITOR painted it.

    Not ``widget.render_lines``: that renders the widget alone, so an
    opaque child painting over its parent -- the exact defect here -- is
    invisible to it.
    """
    strips = list(app.screen._compositor.render_strips())
    region = widget.region
    rows = []
    for y in range(region.y, region.y + region.height):
        if y >= len(strips):
            break
        rows.append(strips[y].text[region.x : region.x + region.width])
    return rows


def _generic_panel_state():
    form = LibraryIngestFormState(path="/tmp/a.txt")
    form.expanded_type_groups = {"generic"}
    return build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight({"generic": ["/tmp/a.txt"]}),
    )


@pytest.mark.asyncio
async def test_option_select_focus_is_glyph_level_and_dimensionally_stable():
    """AC#1: every focusable on this canvas, Selects included, must change
    at the glyph level on focus."""
    app = _CssTrueCanvasHost(_generic_panel_state())
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        select = pilot.app.query_one("#opt-generic-encoding", Select)
        assert not select.has_focus
        region_before = select.region
        unfocused = _composited_rows(pilot.app, select)
        assert not any(
            glyph in row for row in unfocused for glyph in HEAVY_GLYPHS
        ), f"unfocused select already paints heavy glyphs: {unfocused!r}"

        select.focus()
        await pilot.pause()
        focused = _composited_rows(pilot.app, select)

        assert focused != unfocused, (
            "focus produced a byte-identical composited capture -- the "
            "colour-only regression this task pinned"
        )
        assert any(
            glyph in row for row in focused for glyph in HEAVY_GLYPHS
        ), f"focused select shows no structural cue: {focused!r}"
        # The cue must not eat the value (the task-3302 one-row trap) and
        # must not move the control.
        assert any("Auto-detect" in row for row in focused), (
            f"focus treatment ate the select's value: {focused!r}"
        )
        assert select.region == region_before
        assert len(focused) == len(unfocused)


@pytest.mark.asyncio
async def test_path_field_carries_a_persistent_visible_label():
    """AC#3: the primary control's identity was placeholder-only, and a
    placeholder vanishes the moment the field is populated -- exactly the
    defect task-2012 fixed for the OPTION fields."""
    form = LibraryIngestFormState(path="/tmp/some/very/long/path.pdf")
    state = build_library_ingest_state((), form=form)
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        path_input = pilot.app.query_one("#library-ingest-path", Input)
        assert path_input.value, "precondition: the field is populated"
        label = pilot.app.query_one("#library-ingest-path-label", Static)
        text = str(label.renderable)
        assert text == INGEST_PATH_LABEL_COPY
        # It must name what the field accepts, which is what the
        # disappearing placeholder used to be the only carrier of.
        assert "file" in text.lower() and "url" in text.lower(), text
        # The label must precede the control it names.
        canvas_children = list(pilot.app.query_one(LibraryIngestCanvas).children)
        assert canvas_children.index(label) < canvas_children.index(path_input)


@pytest.mark.asyncio
async def test_input_placeholders_clear_the_contrast_floor_in_both_states():
    """AC#4: placeholders measured 3.52:1 enabled / 3.49:1 disabled -- below
    AA for normal text, and a 0.03 delta means a placeholder carries no
    state cue of its own."""
    form = LibraryIngestFormState(path="")
    form.expanded_type_groups = {"audio_video"}
    form.type_options = {"audio_video": {"transcription_provider": "default"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight({"audio_video": ["/tmp/a.mp3"]}),
    )
    app = _CssTrueCanvasHost(state)
    async with app.run_test(size=(100, 70)) as pilot:
        await pilot.pause()
        enabled = pilot.app.query_one("#library-ingest-path", Input)
        disabled = pilot.app.query_one(
            "#opt-audio_video-transcription_model_dir", Input
        )
        assert disabled.disabled is True, "precondition: gate closed"
        measured = {}
        for name, widget, needle in (
            ("enabled", enabled, "Path to a local"),
            ("disabled", disabled, "parakeet"),
        ):
            style = _painted_style_of_text(pilot.app, widget.region, needle)
            assert style is not None, f"{name} placeholder not painted"
            measured[name] = _contrast(style.color, style.bgcolor)
        for name, ratio in measured.items():
            assert ratio >= 4.5, (
                f"{name} placeholder still below AA: {ratio:.2f}:1 "
                f"(measured 3.52/3.49 pre-fix)"
            )


@pytest.mark.asyncio
async def test_blocked_group_states_its_reason_on_a_keyboard_reachable_title():
    """AC#2: Textual removes a ``disabled`` widget from the tab order
    outright (``Widget.focusable`` excludes it), so the ``— needs X
    installed`` annotations task-3304 added AT the controls are
    mouse-and-eyes-only for a fully-gated group. The group's collapsible
    TITLE is a tab stop, so the reason is surfaced there too."""
    app = _CanvasHost(_audio_state(provider="default"))
    with patch(_INSTALLED_PATCH, return_value=False):
        async with app.run_test() as pilot:
            panel = pilot.app.query_one("#type-group-audio_video", Collapsible)
            title = panel.query_one(CollapsibleTitle)
            assert title.focusable, (
                "the group title must be a tab stop for this to be the fix"
            )
            assert "unavailable" in panel.title, panel.title
            assert "needs" in panel.title, panel.title
            # And the controls themselves really are unreachable, which is
            # what makes the title the only honest place for the reason.
            fields = [
                widget
                for widget in panel.query(Input)
                if widget.id and widget.id.startswith("opt-audio_video-")
            ]
            assert fields and not any(widget.focusable for widget in fields)


# --- task-14822 AC#2, re-measured in the SHIPPED screen ----------------------
#
# The AC was first ticked against the canvas mounted ALONE at 80x52
# (``test_breakdown_and_start_are_in_view_behind_eleven_warnings`` above).
# A live pass then found Start ~16 rows below the fold in the real Library
# screen at 235x52 -- the canvas there sits inside the shell (rail + header
# chrome), the queue block renders below the form, and a real folder stages
# FOUR option panels rather than one. This harness measures the shipped
# geometry, so the number the AC stands on comes from the surface the user
# actually sees.


def _four_group_selection(warning_count: int = 11) -> PreflightResult:
    """The live shape: a mixed folder spanning four option panels."""
    type_groups = {
        "pdf": [f"/tmp/mixed/doc{i}.pdf" for i in range(3)],
        "audio_video": [f"/tmp/mixed/clip{i}.mp3" for i in range(2)],
        "ebook": ["/tmp/mixed/book.epub"],
        "generic": [f"/tmp/mixed/note{i}.txt" for i in range(4)],
        "unsupported": ["/tmp/mixed/thing.xyz"],
    }
    return PreflightResult(
        type_groups=type_groups,
        warnings=_many_warnings(warning_count),
        errors=[],
        total_size=4096,
        truncated=False,
        total_files=sum(len(files) for files in type_groups.values()),
        empty_files=["/tmp/mixed/empty.txt"],
    )


async def _shipped_ingest_screen(host, pilot, *, warning_count: int = 11):
    """Drive the real Library screen to a warned, four-group selection."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
    from Tests.UI.test_library_shell import (
        _wait_for_library_shell,
        _wait_for_selector,
    )

    screen = host.screen_stack[-1]
    await _wait_for_library_shell(screen, pilot)
    await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
    await _wait_for_selector(screen, pilot, "#library-ingest-path")
    await pilot.pause()
    screen.app_instance.media_db = object()
    screen._ingest_state.form.path = "/tmp/mixed"
    screen._ingest_state.form.preflight = _four_group_selection(warning_count)
    screen._update_library_ingest_dynamic_regions()
    await pilot.pause()
    await pilot.pause()
    return screen


def _rows_below_the_fold(canvas, widget) -> int:
    """How far ``widget`` starts past the canvas viewport's bottom edge."""
    fold = canvas.scroll_offset.y + canvas.container_size.height
    return widget.virtual_region.y - fold + 1


@pytest.mark.asyncio
async def test_the_fold_pays_for_itself_in_the_shipped_screen():
    """AC#2's first half, re-measured where the canvas actually ships.

    Measured 2026-08-10 at 235x52, four staged groups, 11 warnings: the
    canvas viewport is 43 rows (the shell's rail/header chrome takes 9 of
    the 52), the type breakdown lands at virtual y=6 -- in view -- and
    folding the wall moves Start from virtual y=92 to y=59, a 33-row
    saving. That saving is the fold's real win and is asserted here; what
    it does NOT buy is Start clearing the fold at 52 rows (see
    ``test_start_still_needs_scrolling_at_52_rows`` -- AC#2's second half
    is un-ticked on that evidence).
    """
    host = _screen_harness()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _shipped_ingest_screen(host, pilot)
        canvas = screen.query_one(LibraryIngestCanvas)
        breakdown = screen.query_one("#ingest-type-breakdown", Static)
        # task-15790: `virtual_region.y` stopped being an absolute row in
        # the canvas content when the 15513 Import-behavior work nested the
        # form's actions in their own Vertical -- Start's y became relative
        # to that small parent (measured: y=2 folded AND unfolded, "saving"
        # 0). The fold's cost is the COLLAPSIBLE's own rendered height
        # delta, which no re-nesting can distort; the breakdown check
        # switches to screen-space visibility for the same reason.
        detail = screen.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        folded_height = detail.region.height
        breakdown_visible = (
            breakdown.region.height > 0
            and breakdown.region.y < canvas.region.bottom
        )

        detail.collapsed = False
        await pilot.pause()
        await pilot.pause()
        unfolded_height = screen.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        ).region.height

    assert breakdown_visible, (
        "type breakdown below the fold in the shipped screen"
    )
    saving = unfolded_height - folded_height
    assert saving >= 25, (
        "the fold no longer pays for itself in the shipped screen: "
        f"folded detail height={folded_height}, unfolded "
        f"height={unfolded_height} (saving {saving} rows)"
    )


@pytest.mark.asyncio
async def test_the_open_fold_survives_a_registry_tick_in_the_shipped_screen():
    """G3 on the path that actually broke it.

    ``_update_library_ingest_dynamic_regions`` runs on EVERY registry tick
    (each queued/parsing/writing/done transition of every job) and rebuilds
    the pre-flight summary with ``refresh(recompose=True)``. The fold was
    composed ``collapsed=True`` unconditionally, so it snapped shut under a
    user reading it during an active import.
    """
    host = _screen_harness()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _shipped_ingest_screen(host, pilot)
        fold = screen.query_one("#ingest-preflight-tooling-detail", Collapsible)
        fold.collapsed = False
        await pilot.pause()

        # The tick itself, not a stand-in for it.
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        await pilot.pause()

        reborn = screen.query_one(
            "#ingest-preflight-tooling-detail", Collapsible
        )
        assert reborn.collapsed is False, (
            "the fold snapped shut on a registry tick in the shipped screen"
        )


@pytest.mark.asyncio
async def test_start_and_forecast_are_visible_without_scrolling_at_52_rows():
    """TASK-15702: the shipped shell pins the commit decision above the fold."""
    host = _screen_harness()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _shipped_ingest_screen(host, pilot, warning_count=0)
        canvas = screen.query_one(LibraryIngestCanvas)
        start = screen.query_one("#library-ingest-start", Button)
        summary = screen.query_one("#library-ingest-commit-summary", Static)
        assert canvas.region.y <= summary.region.y < canvas.region.bottom, (
            f"canvas={canvas.region}, summary={summary.region}, start={start.region}, "
            f"copy={screen._build_library_ingest_state().commit_summary_line!r}, "
            f"display={summary.display}"
        )
        assert canvas.region.y <= start.region.y < canvas.region.bottom, (
            f"canvas={canvas.region}, summary={summary.region}, start={start.region}"
        )
        assert start.region.bottom <= canvas.region.bottom


@pytest.mark.asyncio
async def test_every_canvas_focusable_changes_at_the_glyph_level_on_focus():
    """AC#1 as a sweep, not a spot check.

    The Select defect was invisible to per-widget ``render_lines`` (which
    renders a widget WITHOUT the opaque child that covers it), so it
    survived two focus-contract rounds. This walks every focusable the
    canvas actually offers and diffs the COMPOSITED capture, which is the
    only capture a user can see.
    """
    form = LibraryIngestFormState(path="/tmp/a.txt")
    form.expanded_type_groups = {"generic"}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight({"generic": ["/tmp/a.txt"]}),
    )
    app = _CssTrueCanvasHost(state)
    async with app.run_test(size=(100, 80)) as pilot:
        await pilot.pause()
        canvas = pilot.app.query_one(LibraryIngestCanvas)
        focusables = [
            widget
            for widget in canvas.query("*")
            if widget.focusable and widget.region.area
        ]
        assert len(focusables) >= 6, f"too few focusables swept: {focusables}"
        colour_only = []
        for widget in focusables:
            pilot.app.screen.set_focus(None)
            await pilot.pause()
            before = _composited_rows(pilot.app, widget)
            widget.focus()
            await pilot.pause()
            after = _composited_rows(pilot.app, widget)
            if before == after:
                colour_only.append(f"{type(widget).__name__}#{widget.id}")
        assert not colour_only, (
            f"focus is colour-only on: {colour_only}"
        )
