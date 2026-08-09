"""Tests for the Library ingest guardrail confirmation modal."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.UI.Screens.library_screen import (
    IngestGuardrailModal,
    LibraryScreen,
    _affected_counts,
)


class GuardrailApp(App):
    """Minimal app for exercising the modal in isolation."""

    def __init__(self):
        super().__init__()
        self.copied: list[str] = []

    def compose(self):
        return []

    def copy_to_clipboard(self, text: str) -> None:
        self.copied.append(text)


@pytest.fixture
def sample_warnings() -> list[dict]:
    return [
        {
            "feature": "pdf_processing",
            "label": "PDF processing",
            "hint": "Install pdfplumber to ingest PDFs.",
            "command": "pip install pdfplumber",
        },
        {
            "feature": "ocr_docext",
            "label": "OCR document extraction",
            "hint": "Install pytesseract for OCR.",
        },
    ]


@pytest.fixture
def sample_counts() -> dict[str, int]:
    return {"pdf_processing": 3, "ocr_docext": 3}


@pytest.mark.asyncio
async def test_guardrail_modal_confirm(sample_warnings, sample_counts):
    app = GuardrailApp()
    async with app.run_test() as pilot:
        captured: list[bool] = []
        modal = IngestGuardrailModal(sample_warnings, sample_counts)
        await app.push_screen(modal, lambda result: captured.append(result))
        await pilot.pause()

        confirm = app.screen.query_one("#ingest-guardrail-confirm", Button)
        await pilot.click(confirm)
        await pilot.pause()

        assert captured == [True]


@pytest.mark.asyncio
async def test_guardrail_modal_cancel(sample_warnings, sample_counts):
    app = GuardrailApp()
    async with app.run_test() as pilot:
        captured: list[bool] = []
        modal = IngestGuardrailModal(sample_warnings, sample_counts)
        await app.push_screen(modal, lambda result: captured.append(result))
        await pilot.pause()

        cancel = app.screen.query_one("#ingest-guardrail-cancel", Button)
        await pilot.click(cancel)
        await pilot.pause()

        assert captured == [False]


@pytest.mark.asyncio
async def test_guardrail_modal_escape_dismisses(sample_warnings, sample_counts):
    """TASK-596 Task 7: BINDINGS used ``dismiss(false)`` (lowercase) --

    ``textual.actions.parse`` runs ``ast.literal_eval`` on the action's
    argument text, and lowercase ``false`` is not a Python literal, so
    pressing Escape raised ``ActionError: unable to parse 'false' in
    action 'dismiss(false)'`` instead of ever dismissing the modal.
    Confirmed directly against ``textual.actions.parse`` before fixing it
    to ``dismiss(False)``. This test exercises the real key binding
    through ``pilot.press`` -- not the mapped Python callable directly --
    so it would have caught the bug: before the fix, this raises instead
    of reaching the ``assert``.

    Args:
        sample_warnings: Fixture; two representative guardrail warnings
            (PDF processing, OCR extraction) the modal renders.
        sample_counts: Fixture; per-feature affected-file counts shown
            alongside each warning.
    """
    app = GuardrailApp()
    async with app.run_test() as pilot:
        captured: list[bool] = []
        modal = IngestGuardrailModal(sample_warnings, sample_counts)
        await app.push_screen(modal, lambda result: captured.append(result))
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert captured == [False]


@pytest.mark.asyncio
async def test_guardrail_modal_copy_command(sample_warnings, sample_counts):
    app = GuardrailApp()
    async with app.run_test() as pilot:
        modal = IngestGuardrailModal(sample_warnings, sample_counts)
        await app.push_screen(modal)
        await pilot.pause()

        copy_button = app.screen.query_one("#ingest-guardrail-copy-command-0", Button)
        await pilot.click(copy_button)
        await pilot.pause()

        assert app.copied == ["pip install pdfplumber"]


@pytest.mark.asyncio
async def test_guardrail_modal_renders_warning_details(sample_warnings, sample_counts):
    app = GuardrailApp()
    async with app.run_test() as pilot:
        modal = IngestGuardrailModal(sample_warnings, sample_counts)
        await app.push_screen(modal)
        await pilot.pause()

        statics = list(app.screen.query(Static))
        labels = {str(s.renderable) for s in statics}
        assert "Some files may fail to import:" in labels
        assert any("PDF processing (3 files):" in label for label in labels)
        assert any("OCR document extraction (3 files):" in label for label in labels)


# ---------------------------------------------------------------------------
# task-3300: rendered-geometry regression tests. The modal's per-warning rows
# sat in bare ``Vertical()`` wrappers whose default ``height: 1fr`` inside the
# ``height: auto`` modal starved every Static to zero rendered height (live
# capture: an empty full-height column). These tests assert on *rendered
# regions*, not the DOM, so they are RED on the pre-fix CSS.
# ---------------------------------------------------------------------------


def _warning(feature: str, label: str, command: str | None = None) -> dict:
    w = {"feature": feature, "label": label, "hint": f"Install {feature}."}
    if command:
        w["command"] = command
    return w


async def _mount_modal(app: GuardrailApp, pilot, warnings, counts):
    modal = IngestGuardrailModal(warnings, counts)
    await app.push_screen(modal)
    await pilot.pause()
    return modal


@pytest.mark.asyncio
@pytest.mark.parametrize("warning_count", [1, 3])
async def test_guardrail_modal_warnings_render_in_compact_modal(warning_count):
    """Every warning line and its Copy-install-command button must occupy
    real rendered rows inside a compact (not full-screen-height) modal."""
    warnings = [
        _warning(f"feat_{i}", f"Feature {i}", command=f"pip install feat-{i}")
        for i in range(warning_count)
    ]
    counts = {f"feat_{i}": i + 1 for i in range(warning_count)}

    app = GuardrailApp()
    async with app.run_test(size=(90, 32)) as pilot:
        modal = await _mount_modal(app, pilot, warnings, counts)

        container = modal.query_one("#ingest-guardrail-modal")
        statics = list(container.query(Static))
        # Header + one Static per warning.
        assert len(statics) == 1 + warning_count
        for static in statics:
            assert static.region.height >= 1, (
                f"warning Static {str(static.renderable)!r} rendered with "
                f"zero height (region={static.region}) -- the bare Vertical "
                "wrapper is starving its children"
            )

        for i in range(warning_count):
            button = modal.query_one(f"#ingest-guardrail-copy-command-{i}", Button)
            assert button.region.height >= 1, (
                f"copy button {i} rendered with zero height "
                f"(region={button.region})"
            )
            assert (
                button.region.y + button.region.height
                <= container.region.y + container.region.height
            ), f"copy button {i} clipped below the modal container"

        screen_height = app.size.height
        assert container.region.height < screen_height, (
            f"modal container fills the full screen height "
            f"({container.region.height} of {screen_height} rows) -- the "
            "empty black full-height column defect"
        )
        # Compact: header + warnings + buttons + chrome, with slack for
        # wrapped lines; nowhere near the 32-row harness screen.
        assert container.region.height <= 8 + 6 * warning_count, (
            f"modal is {container.region.height} rows tall for "
            f"{warning_count} warning(s); expected a compact dialog"
        )


@pytest.mark.asyncio
async def test_guardrail_modal_action_buttons_single_line_aligned(
    sample_warnings, sample_counts
):
    """Both action buttons render their full labels on one line with
    aligned baselines ("Start import anyway" wrapped inside width: 14)."""
    app = GuardrailApp()
    async with app.run_test(size=(90, 32)) as pilot:
        modal = await _mount_modal(app, pilot, sample_warnings, sample_counts)

        cancel = modal.query_one("#ingest-guardrail-cancel", Button)
        confirm = modal.query_one("#ingest-guardrail-confirm", Button)

        for button in (cancel, confirm):
            label = str(button.label)
            assert button.content_region.width >= len(label), (
                f"button label {label!r} needs {len(label)} columns but got "
                f"{button.content_region.width} -- it wraps to multiple lines"
            )
        assert cancel.region.y == confirm.region.y, "button baselines misaligned"
        assert cancel.region.height == confirm.region.height


@pytest.mark.asyncio
async def test_guardrail_modal_actions_reachable_with_many_warnings_on_short_screen():
    """(task-3300 xhigh review round 2, F3) With 7 warnings on a 24-row
    screen, the Cancel / "Start import anyway" row must remain fully
    on-screen and clickable: the warning list scrolls; the title and the
    action row stay pinned. Before the fix the plain Vertical grew past
    max-height 90% and the action row was clipped off the bottom --
    unreachable by mouse at any warning count >= 5."""
    warnings = [
        _warning(f"feat_{i}", f"Feature {i}", command=f"pip install feat-{i}")
        for i in range(7)
    ]
    counts = {f"feat_{i}": 1 for i in range(7)}

    app = GuardrailApp()
    async with app.run_test(size=(80, 24)) as pilot:
        modal = await _mount_modal(app, pilot, warnings, counts)

        container = modal.query_one("#ingest-guardrail-modal")
        assert (
            container.region.y + container.region.height <= app.size.height
        ), "modal container itself overflows the screen"

        for button_id in ("#ingest-guardrail-cancel", "#ingest-guardrail-confirm"):
            button = modal.query_one(button_id, Button)
            assert button.region.height >= 1, (
                f"{button_id} rendered with zero height (region="
                f"{button.region}) -- clipped out of the modal"
            )
            assert button.region.y >= 0
            assert (
                button.region.y + button.region.height <= app.size.height
            ), (
                f"{button_id} extends past the bottom of the screen "
                f"(region={button.region}, screen height {app.size.height})"
            )


@pytest.mark.asyncio
async def test_guardrail_modal_warning_overflow_scrolls_not_clips():
    """(F3) The warnings region is a scroll container: with 7 warnings its
    content overflows internally (scrollable), while the title stays pinned
    above it and every action button below it."""
    from textual.containers import VerticalScroll

    warnings = [
        _warning(f"feat_{i}", f"Feature {i}", command=f"pip install feat-{i}")
        for i in range(7)
    ]
    counts = {f"feat_{i}": 1 for i in range(7)}

    app = GuardrailApp()
    async with app.run_test(size=(80, 24)) as pilot:
        modal = await _mount_modal(app, pilot, warnings, counts)

        scroll = modal.query_one("#ingest-guardrail-warnings", VerticalScroll)
        assert scroll.max_scroll_y > 0, (
            "7 warnings fit without scrolling on a 24-row screen -- the "
            "overflow either clips or the screen is not exercising it"
        )

        title = modal.query_one("#ingest-guardrail-modal > Static", Static)
        cancel = modal.query_one("#ingest-guardrail-cancel", Button)
        assert title.region.y < scroll.region.y, "title not pinned above the list"
        assert (
            cancel.region.y >= scroll.region.y + scroll.region.height
        ), "actions not pinned below the scrolling list"


@pytest.mark.asyncio
async def test_guardrail_modal_pluralizes_file_counts():
    """"1 file" for a single affected file, "2 files" for many (not
    "(1 files)")."""
    warnings = [
        _warning("solo_feat", "Solo feature"),
        _warning("multi_feat", "Multi feature"),
    ]
    counts = {"solo_feat": 1, "multi_feat": 2}

    app = GuardrailApp()
    async with app.run_test(size=(90, 32)) as pilot:
        modal = await _mount_modal(app, pilot, warnings, counts)

        labels = {str(s.renderable) for s in modal.query(Static)}
        assert any("Solo feature (1 file):" in label for label in labels), (
            f"expected singular '1 file'; rendered labels: {labels}"
        )
        assert any("Multi feature (2 files):" in label for label in labels)
        assert not any("(1 files)" in label for label in labels)


@pytest.mark.asyncio
async def test_guardrail_modal_cancel_not_destructive_confirm_emphasized(
    sample_warnings, sample_counts
):
    """Cancel (the safe action) must not carry the red destructive variant;
    the confirm carries the action emphasis -- the repo-wide convention
    (Cancel variant="default", confirm variant="primary")."""
    app = GuardrailApp()
    async with app.run_test(size=(90, 32)) as pilot:
        modal = await _mount_modal(app, pilot, sample_warnings, sample_counts)

        cancel = modal.query_one("#ingest-guardrail-cancel", Button)
        confirm = modal.query_one("#ingest-guardrail-confirm", Button)
        assert cancel.variant != "error", "Cancel is styled as the destructive action"
        assert confirm.variant == "primary"


def test_guardrail_modal_css_uses_theme_tokens():
    """No off-token color literals: ``background: black`` / ``border: tall
    gray`` must be replaced with theme tokens ($surface / $primary etc.)."""
    css = IngestGuardrailModal.DEFAULT_CSS
    assert "black" not in css, "off-token 'black' literal in modal CSS"
    assert "gray" not in css, "off-token 'gray' literal in modal CSS"


def _empty_preflight(**kwargs) -> PreflightResult:
    defaults = {
        "type_groups": {},
        "warnings": [],
        "errors": [],
        "total_size": 0,
        "truncated": False,
        "total_files": 0,
    }
    defaults.update(kwargs)
    return PreflightResult(**defaults)


def test_affected_counts_aggregates_by_feature():
    preflight = _empty_preflight(
        type_groups={
            "pdf": ["/a.pdf", "/b.pdf"],
            "audio_video": ["/a.mp3"],
        }
    )
    counts = _affected_counts(preflight)
    # Features match the current ingest_capabilities definitions.
    assert counts["pdf_processing"] == 2
    assert counts["pymupdf4llm"] == 2
    assert counts["docling"] == 2
    assert counts["audio_processing"] == 1
    assert counts["video_processing"] == 1
    assert counts["faster_whisper"] == 1


def _minimal_library_screen() -> LibraryScreen:
    """Return a LibraryScreen instance without mounting the full UI."""
    screen = object.__new__(LibraryScreen)
    screen._library_ingest_form = LibraryIngestFormState()
    # Set by ``__init__``, which this shortcut bypasses; submit cancels any
    # in-flight pre-flight so a late result cannot repopulate the summary it
    # just cleared.
    screen._library_ingest_preflight_worker = None
    # Also seeded by ``__init__`` (library_screen.py:1841); submit bumps it
    # to invalidate in-flight pre-flight results, so the bypass must seed it
    # too (stale-helper repair, task-3300).
    screen._library_ingest_preflight_generation = 0
    screen._notify_library_ingest_warning = MagicMock()
    screen.refresh = MagicMock()
    # Submit schedules the scroll-receipt-into-view callback (task-3304);
    # the real method posts a message, which this unmounted shortcut
    # cannot do (stale-helper repair, same family as the generation seed).
    screen.call_after_refresh = MagicMock()
    screen.app_instance = MagicMock()
    return screen


def test_submit_with_blank_path_warns_to_import_not_ingest():
    """(task-2857 review) ``_resolve_ingest_source`` is reachable with a
    blank path directly through ``_submit_library_ingest_form`` -- shared
    by the Start button and Enter-in-path-field -- even though the UI gate
    normally keeps Start disabled and Enter a no-op for a blank path (a
    stale ``start_enabled`` read, or a direct call, still reaches it). The
    warning must say "import", matching every sibling warning on this
    form; no job is submitted."""
    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = ""

    mock_app = MagicMock()
    with patch.object(LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)):
        screen._submit_library_ingest_form()

    screen._notify_library_ingest_warning.assert_called_once_with(
        "Please choose a file to import."
    )
    screen.app_instance.submit_library_ingest_job.assert_not_called()
    mock_app.push_screen.assert_not_called()


def test_submit_with_warnings_shows_guardrail_modal(tmp_path: Path):
    pdf = tmp_path / "file.pdf"
    pdf.write_text("dummy")

    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = str(pdf)
    form.preflight = _empty_preflight(
        type_groups={"pdf": [str(pdf)]},
        warnings=[
            {
                "feature": "pdf_processing",
                "label": "PDF processing",
                "hint": "Install pdfplumber.",
            }
        ],
        total_files=1,
    )

    mock_app = MagicMock()
    with patch.object(LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)):
        screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert mock_app.push_screen.called
    modal = mock_app.push_screen.call_args.args[0]
    assert isinstance(modal, IngestGuardrailModal)
    assert modal.warnings == form.preflight.warnings


def test_submit_confirm_guardrail_calls_submit(tmp_path: Path):
    pdf = tmp_path / "file.pdf"
    pdf.write_text("dummy")

    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = str(pdf)
    form.preflight = _empty_preflight(
        type_groups={"pdf": [str(pdf)]},
        warnings=[
            {
                "feature": "pdf_processing",
                "label": "PDF processing",
                "hint": "Install pdfplumber.",
            }
        ],
        total_files=1,
    )

    mock_app = MagicMock()
    with patch.object(LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)):
        screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert mock_app.push_screen.called
    modal = mock_app.push_screen.call_args.args[0]
    callback = mock_app.push_screen.call_args.args[1]
    assert isinstance(modal, IngestGuardrailModal)

    callback(True)

    screen.app_instance.submit_library_ingest_job.assert_called_once()
    call_kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert call_kwargs["source_path"] == str(pdf)


def test_submit_without_warnings_calls_submit(tmp_path: Path):
    txt = tmp_path / "file.txt"
    txt.write_text("hello")

    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = str(txt)
    form.preflight = _empty_preflight(
        type_groups={"generic": [str(txt)]}, total_files=1
    )

    mock_app = MagicMock()
    with patch.object(LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)):
        screen._submit_library_ingest_form()

    mock_app.push_screen.assert_not_called()
    screen.app_instance.submit_library_ingest_job.assert_called_once()
    call_kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert call_kwargs["source_path"] == str(txt)


def test_submit_clears_the_stale_preflight_summary(tmp_path: Path):
    """Submitting must not leave the previous file's summary on screen.

    The path and title clear on submit, but the pre-flight result did not, so
    the form simultaneously said "Enter a file path to start." and "1 plain
    text file - 333 B" for the file already submitted, which reads as though
    a file is still staged (task-665).
    """
    txt = tmp_path / "file.txt"
    txt.write_text("hello")

    screen = _minimal_library_screen()
    form = screen._library_ingest_form
    form.path = str(txt)
    form.title = "Some title"
    form.preflight = _empty_preflight(
        type_groups={"generic": [str(txt)]}, total_files=1
    )

    mock_app = MagicMock()
    with patch.object(LibraryScreen, "app", new_callable=lambda: property(lambda self: mock_app)):
        screen._submit_library_ingest_form()

    assert form.path == ""
    assert form.title == ""
    assert form.preflight is None, "stale pre-flight summary survived the submit"
    assert form.preflight_checking is False


@pytest.mark.asyncio
async def test_guardrail_warning_line_never_echoes_label_as_its_own_hint():
    """task-3312 (#3): when the capability hint equals the feature label
    the line read "- Audio processing (1 file): Audio processing" (live
    2026-08-08). The echo is suppressed -- same rule the inline pre-flight
    builder (``build_warning_lines``) already applies. A distinct hint
    still renders after the colon."""
    warnings = [
        {
            "feature": "audio_processing",
            "label": "Audio processing",
            "hint": "Audio processing",
        },
        {
            "feature": "pdf_processing",
            "label": "PDF processing",
            "hint": "PDF ingestion",
        },
    ]
    counts = {"audio_processing": 1, "pdf_processing": 2}
    app = GuardrailApp()
    async with app.run_test() as pilot:
        modal = IngestGuardrailModal(warnings, counts)
        await app.push_screen(modal, lambda result: None)
        await pilot.pause()

        texts = [
            str(static.renderable)
            for static in app.screen.query(".ingest-guardrail-warning Static")
        ]
        echo_line = next(t for t in texts if "Audio processing" in t)
        assert echo_line == "- Audio processing (1 file)", echo_line
        distinct_line = next(t for t in texts if "PDF processing" in t)
        assert distinct_line == "- PDF processing (2 files): PDF ingestion", (
            distinct_line
        )
