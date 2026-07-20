"""Tests for ``LibraryIngestCanvas`` rendering and message contracts.

Widget-only tests mount the canvas directly in a bare ``App`` subclass and
assert on widget existence, rendered text, and posted messages. The canvas is
render-only: all state is supplied by ``build_library_ingest_state``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestCanvasState,
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
    LibraryIngestCanvas,
)


class _CanvasHost(App):
    def __init__(self, state: LibraryIngestCanvasState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")


class _MessageRecordingHost(App):
    """Host that records ``OptionValueChanged`` and ``OptionPanelToggled``."""

    def __init__(self, state: LibraryIngestCanvasState) -> None:
        super().__init__()
        self._state = state
        self.option_changes: list[LibraryIngestCanvas.OptionValueChanged] = []
        self.panel_toggles: list[LibraryIngestCanvas.OptionPanelToggled] = []

    def compose(self) -> ComposeResult:
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")

    @on(LibraryIngestCanvas.OptionValueChanged)
    def _record_option_change(self, event: LibraryIngestCanvas.OptionValueChanged) -> None:
        self.option_changes.append(event)

    @on(LibraryIngestCanvas.OptionPanelToggled)
    def _record_panel_toggle(self, event: LibraryIngestCanvas.OptionPanelToggled) -> None:
        self.panel_toggles.append(event)


def _default_form() -> LibraryIngestFormState:
    return LibraryIngestFormState(path="/tmp/test")


@pytest.mark.asyncio
async def test_preflight_checking_renders_status_static():
    """When ``preflight_checking`` is true, a "Checking…" status appears."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight_checking=True,
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        status = pilot.app.query_one("#ingest-preflight-status", Static)
        assert "Checking…" in str(status.renderable)


@pytest.mark.asyncio
async def test_preflight_errors_render_with_retry_button():
    """Pre-flight errors are listed and a Retry button is provided."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found"],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        error_static = pilot.app.query_one("#ingest-preflight-error-0", Static)
        assert "Path not found" in str(error_static.renderable)
        retry_button = pilot.app.query_one("#ingest-preflight-retry", Button)
        assert "Retry" in str(retry_button.label)


@pytest.mark.asyncio
async def test_preflight_warnings_render_with_prefix():
    """Pre-flight warnings are rendered with a warning prefix."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[{"label": "PDF processing", "hint": "PyMuPDF is not installed."}],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        warning_static = pilot.app.query_one("#ingest-preflight-warning-0", Static)
        text = str(warning_static.renderable)
        assert text.startswith("⚠")
        assert "PDF processing" in text
        assert "PyMuPDF is not installed." in text


@pytest.mark.asyncio
async def test_type_breakdown_and_estimate_render():
    """The type-breakdown and estimate Statics render the expected copy."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf", "/tmp/b.pdf"],
                "generic": ["/tmp/c.txt"],
            },
            warnings=[],
            errors=[],
            total_size=2048,
            truncated=False,
            total_files=3,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        breakdown = pilot.app.query_one("#ingest-type-breakdown", Static)
        breakdown_text = str(breakdown.renderable)
        assert "2 PDF documents" in breakdown_text
        assert "1 plain text file" in breakdown_text

        estimate = pilot.app.query_one("#ingest-estimate", Static)
        assert "3 files · 2.0 KB" == str(estimate.renderable)


@pytest.mark.asyncio
async def test_unsupported_files_summary_renders():
    """Unsupported files are summarized with a failure note."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/weird.xyz"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        summary = pilot.app.query_one("#ingest-unsupported-summary", Static)
        assert (
            "1 unsupported file will be recorded as a failure."
            == str(summary.renderable)
        )


@pytest.mark.asyncio
async def test_existing_controls_are_still_present():
    """The path input, Browse, Start ingest, and queue heading remain."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-ingest-path", Input)
        assert pilot.app.query_one("#library-ingest-browse", Button)
        assert pilot.app.query_one("#library-ingest-start", Button)
        assert pilot.app.query_one("#library-ingest-queue-heading", Static)
        assert pilot.app.query_one("#library-ingest-queue-empty", Static)


@pytest.mark.asyncio
async def test_no_preflight_renders_no_summary_widgets():
    """Without a pre-flight result, only the generic panel is mounted."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        for widget_id in (
            "#ingest-preflight-status",
            "#ingest-preflight-error-0",
            "#ingest-preflight-retry",
            "#ingest-preflight-warning-0",
            "#ingest-type-breakdown",
            "#ingest-estimate",
            "#ingest-unsupported-summary",
            "#type-group-pdf",
        ):
            assert len(pilot.app.query(widget_id)) == 0
        # Generic panel is always rendered so global options stay accessible.
        assert len(pilot.app.query("#type-group-generic")) == 1


@pytest.mark.asyncio
async def test_preflight_checking_suppresses_summary():
    """``preflight_checking=True`` hides the full summary, even if a result is
    already available."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=1024,
            truncated=False,
            total_files=1,
        ),
        preflight_checking=True,
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#ingest-preflight-status")) == 1
        for widget_id in (
            "#ingest-type-breakdown",
            "#ingest-estimate",
            "#ingest-unsupported-summary",
            "#type-group-pdf",
        ):
            assert len(pilot.app.query(widget_id)) == 0


@pytest.mark.asyncio
async def test_multiple_errors_and_warnings_are_enumerated():
    """Several errors/warnings each get their own indexed Static."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[
                {"label": "PDF processing", "hint": "PyMuPDF is not installed."},
                {"label": "Audio", "hint": "ffmpeg not found."},
            ],
            errors=["Path not found", "URL unreachable"],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert "Path not found" in str(
            pilot.app.query_one("#ingest-preflight-error-0", Static).renderable
        )
        assert "URL unreachable" in str(
            pilot.app.query_one("#ingest-preflight-error-1", Static).renderable
        )
        assert pilot.app.query_one("#ingest-preflight-retry", Button)

        assert "PyMuPDF is not installed." in str(
            pilot.app.query_one("#ingest-preflight-warning-0", Static).renderable
        )
        assert "ffmpeg not found." in str(
            pilot.app.query_one("#ingest-preflight-warning-1", Static).renderable
        )


@pytest.mark.asyncio
async def test_error_and_warning_markup_is_escaped():
    """Rich markup metacharacters in errors/warnings are rendered verbatim."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[{"label": "Hint", "hint": "[/bracket]"}],
            errors=["[bold]not bold[/bold]"],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        error_static = pilot.app.query_one("#ingest-preflight-error-0", Static)
        assert error_static.visual.plain == "[bold]not bold[/bold]"

        warning_static = pilot.app.query_one("#ingest-preflight-warning-0", Static)
        assert warning_static.visual.plain == "⚠ Hint: [/bracket]"


# --- Per-type options panels ------------------------------------------------


@pytest.mark.asyncio
async def test_type_group_panels_render_for_detected_groups():
    """One collapsible panel is rendered per detected supported type group."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf"],
                "generic": ["/tmp/b.txt"],
            },
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=2,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        pdf_panel = pilot.app.query_one("#type-group-pdf", Collapsible)
        generic_panel = pilot.app.query_one("#type-group-generic", Collapsible)
        assert "PDF documents" in str(pdf_panel.title)
        assert "pdf_engine=" in str(pdf_panel.title)
        assert "Plain text / documents / HTML" in str(generic_panel.title)
        assert "chunk_size=" in str(generic_panel.title)

        scope = pilot.app.query_one("#type-group-pdf .type-group-scope", Static)
        assert "These options apply to all PDF documents files" in str(scope.renderable)

        assert pilot.app.query_one("#opt-pdf-reset", Button)
        assert pilot.app.query_one("#opt-generic-reset", Button)


@pytest.mark.asyncio
async def test_expand_collapse_all_buttons_render_when_type_groups_present():
    """Bulk expand/collapse buttons render only when there are type groups."""
    with_groups = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(with_groups)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#ingest-expand-all", Button)
        assert pilot.app.query_one("#ingest-collapse-all", Button)


@pytest.mark.asyncio
async def test_dependent_controls_disabled_when_dependency_missing():
    """Fields whose ``depends_on`` feature is unavailable render disabled."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=False,
    ):
        async with app.run_test() as pilot:
            engine_select = pilot.app.query_one("#opt-pdf-pdf_engine", Select)
            extract_checkbox = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            assert engine_select.disabled is True
            assert extract_checkbox.disabled is True


@pytest.mark.asyncio
async def test_non_dependent_controls_stay_enabled():
    """Fields with no ``depends_on`` dependency render enabled."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        analyze_checkbox = pilot.app.query_one("#opt-generic-analyze", Checkbox)
        chunk_checkbox = pilot.app.query_one("#opt-generic-chunk", Checkbox)
        encoding_input = pilot.app.query_one("#opt-generic-encoding", Input)
        assert analyze_checkbox.disabled is False
        assert chunk_checkbox.disabled is False
        assert encoding_input.disabled is False


@pytest.mark.asyncio
async def test_chunk_size_disabled_when_chunk_unchecked():
    """Chunk size and overlap inputs are disabled until Chunk is checked."""
    form = _default_form()
    form.chunk = False
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        chunk_size_input = pilot.app.query_one("#opt-generic-chunk_size", Input)
        chunk_overlap_input = pilot.app.query_one("#opt-generic-chunk_overlap", Input)
        assert chunk_size_input.disabled is True
        assert chunk_overlap_input.disabled is True


@pytest.mark.asyncio
async def test_option_value_changed_posted_on_checkbox_change():
    """Toggling a checkbox posts ``OptionValueChanged`` with the right group/name."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            checkbox = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            checkbox.value = True
            await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "pdf"
        and event.name == "ocr"
        and event.value is True
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_value_changed_posted_on_select_change():
    """Changing a select posts ``OptionValueChanged`` with the new value."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            select = pilot.app.query_one("#opt-pdf-pdf_engine", Select)
            select.value = "pymupdf"
            await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "pdf" and event.name == "pdf_engine" and event.value == "pymupdf"
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_value_changed_posted_on_input_change():
    """Typing in a number/text option input posts ``OptionValueChanged``."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    async with app.run_test() as pilot:
        option_input = pilot.app.query_one("#opt-generic-chunk_size", Input)
        option_input.value = "1234"
        await pilot.pause()

    matching = [
        event
        for event in app.option_changes
        if event.group == "generic" and event.name == "chunk_size" and event.value == "1234"
    ]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_option_panel_toggled_posted_on_expand_collapse():
    """Expanding/collapsing a type-group panel posts ``OptionPanelToggled``."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _MessageRecordingHost(state)
    async with app.run_test() as pilot:
        panel = pilot.app.query_one("#type-group-generic", Collapsible)
        panel.collapsed = False
        await pilot.pause()
        panel.collapsed = True
        await pilot.pause()

    assert len(app.panel_toggles) == 2
    assert app.panel_toggles[0].group == "generic"
    assert app.panel_toggles[0].expanded is True
    assert app.panel_toggles[1].group == "generic"
    assert app.panel_toggles[1].expanded is False


@pytest.mark.asyncio
async def test_type_group_number_input_renders_with_value_and_placeholder():
    """Generic number options render as Inputs with their default value/placeholder."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(type_options={"generic": {"chunk_size": 750}}),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        chunk_input = pilot.app.query_one("#opt-generic-chunk_size", Input)
        assert chunk_input.value == "750"
        assert chunk_input.placeholder == "Chunk size"


# --- Progress, structured errors, retry, and recent ingests -----------------


@pytest.mark.asyncio
async def test_progress_line_renders_when_present():
    """A parsing job with structured progress shows a progress line."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.PARSING,
        progress={"message": "Extracting text…"},
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        text = str(progress.renderable)
        assert "parsing" in text
        assert "Extracting text…" in text


@pytest.mark.asyncio
async def test_progress_line_absent_when_not_present():
    """A queued job without progress does not render a progress line."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.QUEUED,
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-progress-ingest-job-1")) == 0


@pytest.mark.asyncio
async def test_show_details_button_renders_for_error_detail():
    """A failed job with structured error detail gets a Show details button."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.FAILED,
        error="Bad codec",
        error_detail={"category": "codec_error", "message": "Codec missing"},
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        btn = pilot.app.query_one("#library-ingest-details-ingest-job-1", Button)
        assert "Show details" in str(btn.label)


@pytest.mark.asyncio
async def test_show_details_button_absent_without_error_detail():
    """A failed job without error detail does not render Show details."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.FAILED,
        error="Bad codec",
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-details-ingest-job-1")) == 0


@pytest.mark.asyncio
async def test_retry_button_renders_for_retryable_failure():
    """A retryable failed job still renders the Retry action."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.FAILED,
        error="Network error",
        permanent=False,
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        btn = pilot.app.query_one("#library-ingest-retry-ingest-job-1", Button)
        assert "Retry" in str(btn.label)


@pytest.mark.asyncio
async def test_retry_button_hidden_for_unsupported_file_type():
    """Retry is withheld when the error category is unsupported_file_type."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.xyz",
        state=IngestJobState.FAILED,
        error="Unsupported file type",
        permanent=False,
        error_detail={
            "category": "unsupported_file_type",
            "message": "Unsupported extension",
        },
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-retry-ingest-job-1")) == 0
        # The structured-error surface is still available.
        assert pilot.app.query_one("#library-ingest-details-ingest-job-1", Button)


@pytest.mark.asyncio
async def test_recent_ingests_section_renders_terminal_jobs():
    """The Recent ingests collapsible lists done/failed jobs but not queued."""
    done = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/done.txt",
        state=IngestJobState.DONE,
        media_id=1,
    )
    failed = LibraryIngestJob(
        job_id="ingest-job-2",
        source_path="/tmp/failed.txt",
        state=IngestJobState.FAILED,
        error="boom",
    )
    queued = LibraryIngestJob(
        job_id="ingest-job-3",
        source_path="/tmp/queued.txt",
        state=IngestJobState.QUEUED,
    )
    state = build_library_ingest_state(
        (done, failed, queued), form=_default_form()
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        recent = pilot.app.query_one("#library-ingest-recent", Collapsible)
        assert str(recent.title) == "Recent ingests"
        assert recent.collapsed is True
        items = pilot.app.query(".library-ingest-recent-item")
        texts = [str(item.renderable) for item in items]
        assert any("done.txt" in t and "done" in t for t in texts)
        assert any("failed.txt" in t and "failed" in t for t in texts)
        assert not any("queued.txt" in t for t in texts)


@pytest.mark.asyncio
async def test_recent_ingests_section_renders_when_queue_empty():
    """Recent ingests is visible even when there are no jobs at all."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-ingest-recent", Collapsible)
        assert len(pilot.app.query("#library-ingest-queue-empty")) == 1
