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
        self.parakeet_install_requests = 0
        self.transcribe_cpp_gguf_requests = 0

    def compose(self) -> ComposeResult:
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")

    @on(LibraryIngestCanvas.OptionValueChanged)
    def _record_option_change(self, event: LibraryIngestCanvas.OptionValueChanged) -> None:
        self.option_changes.append(event)

    @on(LibraryIngestCanvas.OptionPanelToggled)
    def _record_panel_toggle(self, event: LibraryIngestCanvas.OptionPanelToggled) -> None:
        self.panel_toggles.append(event)

    @on(LibraryIngestCanvas.ParakeetInstallRequested)
    def _record_parakeet_install_request(
        self, _event: LibraryIngestCanvas.ParakeetInstallRequested
    ) -> None:
        self.parakeet_install_requests += 1

    @on(LibraryIngestCanvas.TranscribeCppGGUFRequested)
    def _record_transcribe_cpp_gguf_request(
        self, _event: LibraryIngestCanvas.TranscribeCppGGUFRequested
    ) -> None:
        self.transcribe_cpp_gguf_requests += 1


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
        # (task-2100) Unsupported-only selection is gate-blocked, so the
        # line names the file instead of promising a failure row that a
        # blocked submit never records.
        assert str(summary.renderable) == (
            "Unsupported: weird.xyz."
            " Supported: PDF documents, Word/Office documents, audio/video"
            " files, e-books, plain text files, web pages (by URL)."
        )


@pytest.mark.asyncio
async def test_existing_controls_are_still_present():
    """The path input, Browse, Start import, and queue heading remain."""
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
        ):
            assert len(pilot.app.query(widget_id)) == 0
        # (task-2042) Options panels stay mounted during a re-analysis:
        # hiding them made ``preflight_checking`` a STRUCTURAL flag, and the
        # resulting full recompose swallowed clicks in flight. A re-check
        # lasts well under a second; last-known panels are less flicker.
        assert len(pilot.app.query("#type-group-pdf")) == 1


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
        assert warning_static.visual.plain == (
            "⚠ Hint isn't installed — needed for [/bracket]."
        )


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
        # (task-3305) The receipt shows the display label, never the token.
        assert "PDF engine: PyMuPDF4LLM (Markdown)" in str(pdf_panel.title)
        assert "pymupdf4llm" not in str(pdf_panel.title)
        assert "Plain text & HTML" in str(generic_panel.title)
        assert "Chunk size: 1000" in str(generic_panel.title)

        scope = pilot.app.query_one("#type-group-pdf .type-group-scope", Static)
        assert "Applies to every PDF document in this import." in str(
            scope.renderable
        )

        assert pilot.app.query_one("#opt-pdf-reset", Button)
        assert pilot.app.query_one("#opt-generic-reset", Button)


@pytest.mark.asyncio
async def test_expand_collapse_all_buttons_render_when_type_groups_present():
    """Bulk expand/collapse buttons render only when MULTIPLE panels exist
    (task-2016: a single panel has nothing to expand "all" of)."""
    with_groups = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"], "generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=2,
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
        encoding_input = pilot.app.query_one("#opt-generic-encoding", Select)
        assert analyze_checkbox.disabled is False
        assert chunk_checkbox.disabled is False
        assert encoding_input.disabled is False


@pytest.mark.asyncio
async def test_chunk_size_disabled_when_chunk_unchecked():
    """Chunk size and overlap inputs are disabled until Chunk is checked."""
    form = _default_form()
    # The panel renders from ``type_options``; the screen writes the scalar
    # ``form.chunk`` mirror and this dict together on every toggle. Chunking
    # is on by default now, so the off case has to be stated explicitly.
    form.chunk = False
    form.type_options = {"generic": {"chunk": False}}
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
async def test_local_stt_cancel_then_force_stop_actions_render_exclusively():
    running = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/speech.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing"},
    )
    state = build_library_ingest_state((running,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#library-ingest-cancel-ingest-job-1", Button)
        assert not list(pilot.app.query("#library-ingest-force-stop-ingest-job-1"))

    cancelling = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/speech.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing", "cancel_requested": True},
    )
    state = build_library_ingest_state((cancelling,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        force = pilot.app.query_one(
            "#library-ingest-force-stop-ingest-job-1",
            Button,
        )
        assert str(force.label) == "Force stop"
        assert not list(pilot.app.query("#library-ingest-cancel-ingest-job-1"))


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
async def test_transcribe_cpp_failure_renders_only_eligible_recovery_actions():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/private/voice.wav",
        state=IngestJobState.FAILED,
        error="The selected GGUF cannot be used by transcribe.cpp.",
        permanent=False,
        error_detail={
            "category": "stt_failure",
            "code": "artifact_incompatible",
            "message": "The selected GGUF cannot be used by transcribe.cpp.",
            "actions": ["choose_another_gguf", "retry_faster_whisper"],
        },
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)

    async with app.run_test() as pilot:
        choose = pilot.app.query_one(
            "#library-ingest-choose-gguf-ingest-job-1", Button
        )
        retry = pilot.app.query_one(
            "#library-ingest-retry-faster-whisper-ingest-job-1", Button
        )
        assert "Choose another GGUF" in str(choose.label)
        assert str(retry.label) == "Retry with faster-whisper"
        assert not list(pilot.app.query("#library-ingest-retry-ingest-job-1"))


@pytest.mark.asyncio
async def test_transcribe_cpp_provider_shows_path_free_configured_picker():
    form = _default_form()
    form.type_options = {
        "audio_video": {"transcription_provider": "transcribe-cpp"}
    }
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/voice.wav"]},
            warnings=[],
            errors=[],
            total_size=1,
            truncated=False,
            total_files=1,
        ),
        transcribe_cpp_configured=True,
    )
    app = _MessageRecordingHost(state)

    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            button = pilot.app.query_one(
                "#opt-audio_video-choose-transcribe-cpp-gguf", Button
            )
            status = pilot.app.query_one(
                "#opt-audio_video-transcribe-cpp-status", Static
            )
            assert "Choose another GGUF" in str(button.label)
            assert "configured" in str(status.renderable).lower()
            assert "/" not in str(status.renderable)

            button.press()
            await pilot.pause()

    assert app.transcribe_cpp_gguf_requests == 1


@pytest.mark.asyncio
async def test_recent_ingests_section_renders_terminal_jobs():
    """The Recent imports collapsible lists done/failed jobs but not queued."""
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
        assert str(recent.title) == "Recent imports"
        assert recent.collapsed is True
        items = pilot.app.query(".library-ingest-recent-item")
        texts = [str(item.renderable) for item in items]
        assert any("done.txt" in t and "done" in t for t in texts)
        assert any("failed.txt" in t and "failed" in t for t in texts)
        assert not any("queued.txt" in t for t in texts)


@pytest.mark.asyncio
async def test_recent_ingests_section_renders_when_queue_empty():
    """(task-2100) Recent imports is HIDDEN when there is nothing recent --
    it used to render always, and after a clear it expanded to an empty,
    unlabeled shell (round-3 critique evidence)."""
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert not list(pilot.app.query("#library-ingest-recent"))
        assert len(pilot.app.query("#library-ingest-queue-empty")) == 1


@pytest.mark.asyncio
async def test_mounting_option_panels_posts_no_option_changes():
    """Mounting a type-group panel must not look like a user edit.

    Textual's ``Select`` posts ``Changed`` as soon as it mounts, and an
    ``Input`` does the same for a non-empty ``value=``. Bubbling those as
    ``OptionValueChanged`` made the screen recompose, which remounted the
    select, which posted again -- an unbounded recompose cycle that pinned
    the UI at 100% CPU for every pdf/audio/ebook pre-flight (task-673).

    Every type group is mounted at once: ``pdf``, ``audio_video`` and
    ``ebook`` each carry a select and so each reproduced the freeze on its
    own, while ``generic`` (the one group with no select) is the reason
    plain text was the only content type that ever worked.
    """
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf"],
                "audio_video": ["/tmp/a.mp3"],
                "ebook": ["/tmp/a.epub"],
                "generic": ["/tmp/a.txt"],
            },
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=4,
        ),
    )
    app = _MessageRecordingHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()

    assert app.option_changes == [], (
        "mounting option panels emitted "
        f"{[(e.group, e.name, e.value) for e in app.option_changes]}"
    )


@pytest.mark.asyncio
async def test_audio_video_defaults_to_semantic_provider_with_exact_controls_disabled():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
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
            provider = pilot.app.query_one(
                "#opt-audio_video-transcription_provider", Select
            )
            model = pilot.app.query_one("#opt-audio_video-transcription_model", Select)
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            assert provider.value == "default"
            assert model.disabled is True
            assert model_dir.disabled is True
            assert model_dir.value == ""
            install = pilot.app.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            )
            assert "Install verified Parakeet v2 INT8" in str(install.label)
            assert install.disabled is True
            install.press()
            await pilot.pause()
            assert app.parakeet_install_requests == 0


@pytest.mark.asyncio
async def test_explicit_parakeet_enables_model_dir_and_verified_installer():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
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
            provider = pilot.app.query_one(
                "#opt-audio_video-transcription_provider", Select
            )
            model = pilot.app.query_one("#opt-audio_video-transcription_model", Select)
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            install = pilot.app.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            )

            assert provider.value == "parakeet-onnx"
            assert model.disabled is True
            assert model_dir.disabled is False
            assert install.disabled is False
            install.press()
            await pilot.pause()
            assert app.parakeet_install_requests == 1


@pytest.mark.asyncio
async def test_explicit_faster_whisper_enables_only_whisper_model_control():
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "faster-whisper"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/a.mp3"]},
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
        return_value=True,
    ):
        async with app.run_test() as pilot:
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-transcription_provider", Select
                ).value
                == "faster-whisper"
            )
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-transcription_model", Select
                ).disabled
                is False
            )
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-transcription_model_dir", Input
                ).disabled
                is True
            )
            assert (
                pilot.app.query_one(
                    "#opt-audio_video-install-parakeet-v2", Button
                ).disabled
                is True
            )


@pytest.mark.asyncio
async def test_chunk_size_enabled_when_chunk_checked():
    """Chunk size and overlap become editable once Chunk is on.

    They were gated through the installed-feature lookup on the name
    "chunk", which is a sibling field rather than a package, so they were
    disabled no matter what the user did (task-676).
    """
    form = _default_form()
    form.chunk = True
    form.type_options = {"generic": {"chunk": True}}
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
        assert pilot.app.query_one("#opt-generic-chunk_size", Input).disabled is False
        assert (
            pilot.app.query_one("#opt-generic-chunk_overlap", Input).disabled is False
        )


@pytest.mark.asyncio
async def test_path_error_offers_a_way_to_pick_a_file_not_retry():
    """A path that cannot be found offers correction, not Retry.

    Re-running the same analysis against the same bad path fails identically,
    so Retry was the wrong verb for the most common pre-flight error a
    first-time user hits (task-666).
    """
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found: /tmp/nope.txt"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=True,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#ingest-preflight-retry")) == 0
        assert pilot.app.query_one("#ingest-preflight-choose", Button)


@pytest.mark.asyncio
async def test_retryable_error_still_offers_retry():
    """A URL that failed to respond is worth another attempt."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["URL unreachable: timed out"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=False,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#ingest-preflight-retry", Button)
        assert len(pilot.app.query("#ingest-preflight-choose")) == 0


@pytest.mark.asyncio
async def test_option_panel_title_reads_as_plain_language():
    """The collapsed panel title describes settings, not internal field names."""
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
        return_value=True,
    ):
        async with app.run_test() as pilot:
            title = str(pilot.app.query_one("#type-group-pdf", Collapsible).title)

    assert "pdf_engine=" not in title
    assert "ocr=False" not in title
    assert "PDF engine: PyMuPDF4LLM (Markdown)" in title
    assert "pymupdf4llm" not in title
    assert "Enable OCR: off" in title


def test_warning_line_does_not_repeat_itself():
    """Warning copy names the gap once, then how to close it."""
    from tldw_chatbook.Library.library_ingest_state import build_warning_lines

    lines = build_warning_lines(
        [
            {
                "feature": "pdf_processing",
                "label": "PDF processing",
                "hint": "PDF ingestion",
                "command": 'pip install -e ".[pdf]"',
            },
            {
                "feature": "audio_processing",
                "label": "Audio processing",
                "hint": "Audio processing",
                "command": 'pip install -e ".[audio]"',
            },
        ]
    )

    assert lines[0] == (
        "PDF processing isn't installed — needed for PDF ingestion. "
        'Install it with: pip install -e ".[pdf]"'
    )
    # The label and the capability are the same words here, so say it once.
    assert lines[1] == (
        'Audio processing isn\'t installed. Install it with: pip install -e ".[audio]"'
    )


@pytest.mark.asyncio
async def test_first_visit_explains_what_can_be_imported():
    """An untouched form orients the user instead of showing blank space."""
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        first = str(pilot.app.query_one("#library-ingest-intro-0", Static).renderable)
        second = str(pilot.app.query_one("#library-ingest-intro-1", Static).renderable)

    assert "folder" in first and "URL" in first
    assert "PDF documents" in first and "e-books" in first
    assert "searchable" in second


@pytest.mark.asyncio
async def test_intro_gives_way_to_the_real_summary():
    """Orientation never competes with an actual pre-flight result."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        # (task-2042) Intro Statics stay mounted (display-managed) so their
        # appearance/disappearance never changes the canvas structure; the
        # user-visible contract is that they are not DISPLAYED.
        intro = pilot.app.query_one("#library-ingest-intro-0")
        assert intro.display is False


@pytest.mark.asyncio
async def test_clear_button_appears_only_with_a_path():
    """The path field can be emptied in one press once it has content."""
    empty = build_library_ingest_state((), form=LibraryIngestFormState())
    app = _CanvasHost(empty)
    async with app.run_test() as pilot:
        # (task-2042) Always mounted, display-managed -- the visible
        # contract is unchanged (hidden without a path, shown with one),
        # but the widget STRUCTURE no longer flips while typing.
        clear = pilot.app.query_one("#library-ingest-clear-path", Button)
        assert clear.display is False

    filled = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(filled)
    async with app.run_test() as pilot:
        clear = pilot.app.query_one("#library-ingest-clear-path", Button)
        assert clear.display is True


@pytest.mark.asyncio
async def test_backend_switch_appears_only_with_a_server_configured():
    """A dead toggle is worse than none, so it is offered only when real."""
    local_only = build_library_ingest_state((), form=LibraryIngestFormState())
    app = _CanvasHost(local_only)
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-ingest-backend-switch")) == 0
        assert len(pilot.app.query("#library-ingest-server-line")) == 0

    with_server = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server",
        server_ingest_available=True
    )
    app = _CanvasHost(with_server)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-ingest-backend-switch", Button)
        assert "server" in str(button.label).lower()
        line = pilot.app.query_one("#library-ingest-server-line", Static)
        assert "this machine" in str(line.renderable).lower()


@pytest.mark.asyncio
async def test_backend_switch_offers_the_way_back_when_targeting_the_server():
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        ingest_backend="server",
        runtime_source="server",
        server_ingest_available=True,
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-ingest-backend-switch", Button)
        assert "this machine" in str(button.label).lower()
        line = pilot.app.query_one("#library-ingest-server-line", Static)
        assert "server" in str(line.renderable).lower()


@pytest.mark.asyncio
async def test_backend_state_is_rendered_before_the_switch():
    """"Imports run on X" must precede the button that changes it.

    With the button first, the pane read as a contradiction top-to-bottom:
    "Import on the server" directly above "Imports run on this machine."
    """
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server",
        server_ingest_available=True
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        children = list(pilot.app.query_one(LibraryIngestCanvas).children)
        ids = [c.id for c in children if c.id]
        assert "library-ingest-server-line" in ids
        assert "library-ingest-backend-switch" in ids
        assert ids.index("library-ingest-server-line") < ids.index(
            "library-ingest-backend-switch"
        ), f"switch rendered before the state line: {ids[:6]}"


@pytest.mark.asyncio
async def test_unsupported_files_summary_pluralizes_correctly():
    """(task-2015) Plural counts must not read "recorded as a failures"."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/a.xyz", "/tmp/b.xyz"]},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=2,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        summary = pilot.app.query_one("#ingest-unsupported-summary", Static)
        # (task-2100) Gate-blocked (nothing importable): names only.
        assert str(summary.renderable) == (
            "Unsupported: a.xyz, b.xyz."
            " Supported: PDF documents, Word/Office documents, audio/video"
            " files, e-books, plain text files, web pages (by URL)."
        )


# --- task-2016: P3 polish ----------------------------------------------------


@pytest.mark.asyncio
async def test_done_row_progress_line_has_no_state_prefix():
    """(task-2016) The row line already says "✓ done"; prefixing the progress
    line with "done" again read as stuttering. Terminal states render the
    message alone; active states keep the prefix (previous test)."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/report.txt",
        state=IngestJobState.DONE,
        media_id=1,
        progress={"message": "Imported report.txt"},
    )
    state = build_library_ingest_state((job,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        progress = pilot.app.query_one(
            "#library-ingest-progress-ingest-job-1", Static
        )
        assert str(progress.renderable) == "Imported report.txt"


@pytest.mark.asyncio
async def test_expand_collapse_all_hidden_for_single_panel():
    """(task-2016) Bulk expand/collapse over exactly one panel is noise."""
    single = build_library_ingest_state(
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
    app = _CanvasHost(single)
    async with app.run_test() as pilot:
        assert not list(pilot.app.query("#ingest-expand-all"))
        assert not list(pilot.app.query("#ingest-collapse-all"))


@pytest.mark.asyncio
async def test_generic_scope_line_reworded_when_no_generic_files_staged():
    """(task-2016) The always-present generic panel claimed "Applies to all
    Plain text & HTML in this import." even when the import
    contained zero such files."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    # Expand the generic panel so its scope line mounts.
    state.form.expanded_type_groups.add("generic")
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        scopes = [
            str(w.renderable)
            for w in pilot.app.query(".type-group-scope").results(Static)
        ]
        generic_scope = [s for s in scopes if "plain text" in s.lower()]
        assert generic_scope, f"generic scope line missing: {scopes}"
        assert "if this import contains any" in generic_scope[0]
        assert "in this import." not in generic_scope[0]


# --- task-2043: P2 batch ----------------------------------------------------


@pytest.mark.asyncio
async def test_expanded_details_render_inline_and_flip_button_label():
    """(task-2043) Expanded rows render their detail lines inline (the old
    surface was a ~4s uncopyable toast) and the button reads Hide details."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="Failed to process pdf file: PDF Extraction Error.",
        error_detail={
            "category": "parse_error",
            "message": "Failed to process pdf file: PDF Extraction Error.",
        },
    )
    state = build_library_ingest_state(
        (job,),
        form=_default_form(),
        expanded_details={"ingest-job-1"},
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        detail = pilot.app.query_one("#library-ingest-detail-ingest-job-1-0", Static)
        assert "Category: parse error" in str(detail.renderable)
        button = pilot.app.query_one("#library-ingest-details-ingest-job-1", Button)
        assert str(button.label) == "Hide details"


@pytest.mark.asyncio
async def test_select_fields_carry_visible_labels():
    """(task-2043) Selects missed task-2012's labeling: 'pymupdf4llm' bare
    carries no meaning. Every select gets a label Static."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"pdf": ["/tmp/a.pdf"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    state.form.expanded_type_groups.add("pdf")
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        labels = [
            str(w.renderable)
            for w in pilot.app.query(".type-group-field-label").results(Static)
        ]
        from tldw_chatbook.Library.ingest_capabilities import get_capabilities

        select_labels = [
            f.label for f in get_capabilities("pdf").fields if f.type == "select"
        ]
        assert select_labels, "pdf group unexpectedly has no selects"
        for label in select_labels:
            # (task-3304) A schema-disabled select's label carries a
            # " — <reason>" annotation, and whether one applies here
            # depends on which optional packages THIS environment has --
            # so accept the label with or without the suffix, pinned to
            # the exact separator so a missing label still fails.
            assert any(
                text == label or text.startswith(f"{label} — ")
                for text in labels
            ), f"select {label!r} has no visible label; rendered: {labels!r}"


@pytest.mark.asyncio
async def test_checkbox_glyph_tracks_state_without_color():
    """(task-2043) Stock ToggleButton renders 'X' for both states; the
    subclass carries on/off in the glyph itself."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        StateGlyphCheckbox,
    )

    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    state.form.expanded_type_groups.add("generic")
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        analyze = pilot.app.query_one("#opt-generic-analyze", StateGlyphCheckbox)
        chunk = pilot.app.query_one("#opt-generic-chunk", StateGlyphCheckbox)
        assert chunk.value is True and chunk.BUTTON_INNER == "✓"
        assert analyze.value is False and analyze.BUTTON_INNER == " "
        analyze.value = True
        await pilot.pause()
        assert analyze.BUTTON_INNER == "✓"


@pytest.mark.asyncio
async def test_duplicate_forecast_line_renders_in_summary():
    """(task-2043) The pre-flight duplicate forecast renders as a quiet
    line in the summary block."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
            already_in_library=1,
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        line = pilot.app.query_one("#ingest-duplicate-summary", Static)
        assert "already be in your Library" in str(line.renderable)


@pytest.mark.asyncio
async def test_severity_colour_supplements_glyphs_and_invalid_field_marked() -> None:
    """Severity colour supplements glyphs; invalid fields stay marked.

    (task-2230 a11y) Failed/skipped rows carry a severity class ON TOP of
    the glyph+word they already have (never colour-only), and an invalid
    option field stays marked without focus -- the gate line's
    "highlighted" pointed at a border that only existed while focused.
    """
    failed = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="Failed to process pdf file.",
    )
    skipped = LibraryIngestJob(
        job_id="ingest-job-2",
        source_path="/tmp/photo.jpg",
        state=IngestJobState.SKIPPED,
        error="Unsupported file type: .jpg.",
    )
    done = LibraryIngestJob(
        job_id="ingest-job-3",
        source_path="/tmp/report.txt",
        state=IngestJobState.DONE,
    )
    form = LibraryIngestFormState(path="/tmp/report.txt")
    form.type_options["generic"] = {"chunk_size": "abc"}
    state = build_library_ingest_state((failed, skipped, done), form=form)

    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        rows = list(pilot.app.query(".library-ingest-row"))
        classes = [set(row.classes) for row in rows]
        assert any("library-ingest-row-failed" in c for c in classes)
        assert any("library-ingest-row-skipped" in c for c in classes)
        # The done row carries neither severity class.
        assert sum(
            1
            for c in classes
            if "library-ingest-row-failed" in c
            or "library-ingest-row-skipped" in c
        ) == 2

        # Glyph + word survive alongside the colour (monochrome contract).
        by_id = {row.job_id: row for row in state.queue_rows}
        assert by_id["ingest-job-1"].line.startswith("✗ failed")
        assert by_id["ingest-job-2"].line.startswith("○ skipped")

        invalid = pilot.app.query_one("#opt-generic-chunk_size", Input)
        assert invalid.has_class("-ingest-option-invalid"), (
            "an invalid field must stay marked without focus"
        )
        assert pilot.app.focused is not invalid


@pytest.mark.asyncio
async def test_analysis_hint_renders_when_state_carries_one():
    """task-3301: the Analyze-readiness hint renders beside the Start gate."""
    form = LibraryIngestFormState(path="/tmp/test", analyze=True)
    state = build_library_ingest_state(
        (),
        form=form,
        analysis_unready_hint=(
            "Analyze after import is on, but OpenAI is not ready: Missing "
            "API key. Imports will run without analysis."
        ),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        hint = pilot.app.query_one("#library-ingest-analysis-hint", Static)
        assert hint.display is True
        assert "OpenAI" in str(hint.renderable)
        assert "without analysis" in str(hint.renderable)


@pytest.mark.asyncio
async def test_analysis_hint_hidden_when_state_has_none():
    state = build_library_ingest_state((), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        hint = pilot.app.query_one("#library-ingest-analysis-hint", Static)
        assert hint.display is False


# ---------------------------------------------------------------------------
# task-3303: document panel, PDF OCR gating, AV translate gating, web honesty
# ---------------------------------------------------------------------------


def _preflight_for(groups: dict[str, list[str]]) -> PreflightResult:
    return PreflightResult(
        type_groups=groups,
        warnings=[],
        errors=[],
        total_size=0,
        truncated=False,
        total_files=sum(len(v) for v in groups.values()),
    )


@pytest.mark.asyncio
async def test_document_panel_renders_with_processing_and_ocr_controls():
    """(task-3303 AC1) A .docx selection gets its own options panel."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=_preflight_for({"document": ["/tmp/report.docx"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            panel = pilot.app.query_one("#type-group-document", Collapsible)
            assert "Word/Office documents" in str(panel.title)
            method = pilot.app.query_one(
                "#opt-document-processing_method", Select
            )
            assert method.value == "auto"
            ocr = pilot.app.query_one("#opt-document-ocr", Checkbox)
            # Under the default "auto" method OCR is offerable (docling is
            # "installed" in this test), and the label carries its scope.
            assert ocr.disabled is False
            assert "docling" in str(ocr.label)
            language = pilot.app.query_one("#opt-document-ocr_language", Input)
            # OCR is off, so the language field rides the gate.
            assert language.disabled is True


@pytest.mark.asyncio
async def test_pdf_ocr_checkbox_inert_under_pymupdf_engine_with_reason():
    """(task-3303 AC2) OCR under a non-OCR engine is inert, with the reason."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=_preflight_for({"pdf": ["/tmp/a.pdf"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            ocr = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            # Default engine is pymupdf4llm, which cannot OCR.
            assert ocr.disabled is True
            assert "docling or docext engines only" in str(ocr.label)


@pytest.mark.asyncio
async def test_pdf_ocr_checkbox_enabled_under_docling_engine():
    form = _default_form()
    form.type_options = {"pdf": {"pdf_engine": "docling"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"pdf": ["/tmp/a.pdf"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            ocr = pilot.app.query_one("#opt-pdf-ocr", Checkbox)
            assert ocr.disabled is False
            backend = pilot.app.query_one("#opt-pdf-ocr_backend", Select)
            # Backend selection applies to the docext engine only.
            assert backend.disabled is True


@pytest.mark.asyncio
async def test_translate_checkbox_inert_under_parakeet_provider():
    """(task-3303 AC4) Only faster-whisper translates; parakeet renders it inert."""
    form = _default_form()
    form.type_options = {
        "audio_video": {"transcription_provider": "parakeet-onnx"}
    }
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"audio_video": ["/tmp/talk.mp3"]}),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            translate = pilot.app.query_one(
                "#opt-audio_video-translate_to_english", Checkbox
            )
            assert translate.disabled is True
            assert "faster-whisper" in str(translate.label)
            vad = pilot.app.query_one("#opt-audio_video-vad_filter", Checkbox)
            assert vad.disabled is False


@pytest.mark.asyncio
async def test_web_local_multi_page_note_visible_when_sitemap_selected():
    """(task-3303 AC5) A local sitemap selection says it fetches one page."""
    form = _default_form()
    form.type_options = {"web": {"scrape_method": "sitemap"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"web": ["https://example.com/post"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#web-local-scope-note", Static)
        assert note.display is True
        assert "one page" in str(note.renderable)
        assert "server" in str(note.renderable)


@pytest.mark.asyncio
async def test_web_note_hidden_for_single_page_selection():
    form = _default_form()
    form.type_options = {"web": {"scrape_method": "individual"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_for({"web": ["https://example.com/post"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#web-local-scope-note", Static)
        assert note.display is False


@pytest.mark.asyncio
async def test_web_note_hidden_when_targeting_the_server():
    """Server behavior is unchanged: the clip path honors multi-page methods."""
    form = _default_form()
    form.type_options = {"web": {"scrape_method": "sitemap"}}
    state = build_library_ingest_state(
        (),
        form=form,
        runtime_source="server",
        server_ingest_available=True,
        ingest_backend="server",
        preflight=_preflight_for({"web": ["https://example.com/post"]}),
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        note = pilot.app.query_one("#web-local-scope-note", Static)
        assert note.display is False


# --- task-3305: copy & labels batch -----------------------------------------


@pytest.mark.asyncio
async def test_every_select_renders_human_labels_never_raw_tokens():
    """(task-3305, MI-09) Every option select shows human display copy while
    persisting the internal value: no rendered option prompt may be the raw
    schema token (``pymupdf4llm``, ``url_level``, ``recursive_scraping``…)."""
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities

    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={
                "pdf": ["/tmp/a.pdf"],
                "document": ["/tmp/b.docx"],
                "audio_video": ["/tmp/c.mp3"],
                "ebook": ["/tmp/d.epub"],
                "web": ["https://example.com/article"],
            },
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=5,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            selects = list(pilot.app.query(Select))
            assert selects, "no selects rendered -- panel sweep is broken"
            seen_groups: set[str] = set()
            for select in selects:
                widget_id = select.id or ""
                assert widget_id.startswith("opt-")
                _, group, name = widget_id.split("-", 2)
                seen_groups.add(group)
                cap = get_capabilities(group)
                field = next(f for f in cap.fields if f.name == name)
                rendered = [
                    (str(prompt), value)
                    for prompt, value in select._options
                    if value is not Select.BLANK
                ]
                assert {value for _prompt, value in rendered} == set(
                    field.options
                ), f"{group}.{name}: persisted values must stay the tokens"
                for prompt, value in rendered:
                    assert prompt != value, (
                        f"{group}.{name}: option {value!r} renders its raw "
                        "internal token"
                    )
                # The persisted selection is still an internal token.
                assert select.value in field.options
            # Every group with a select was actually swept.
            assert {"pdf", "document", "audio_video", "ebook", "web",
                    "generic"} <= seen_groups


@pytest.mark.asyncio
async def test_recent_import_bracket_filename_renders_clean():
    """(task-3305, MI-14) escape_markup on a ``markup=False`` Static is
    double defense: the escape backslashes render literally. A bracketed
    filename must come out byte-identical."""
    done = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/[draft] notes.txt",
        state=IngestJobState.DONE,
        media_id=1,
    )
    state = build_library_ingest_state((done,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        items = [
            str(item.renderable)
            for item in pilot.app.query(".library-ingest-recent-item")
        ]
        assert any("[draft] notes.txt" in text for text in items), items
        assert not any("\\[" in text for text in items), items
        paths = [
            str(item.renderable)
            for item in pilot.app.query(".library-ingest-recent-path")
        ]
        assert any("/tmp/[draft] notes.txt" == text for text in paths), paths
        assert not any("\\[" in text for text in paths), paths


@pytest.mark.asyncio
async def test_collapsed_title_caps_pairs_and_skips_empty_values():
    """(task-3305, MI-16) The audio panel title was a ~140-char run-on with
    a dangling empty value ("Local Parakeet model folder: ,"). The title
    caps at the most salient pairs and never renders an empty value."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/c.mp3"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            title = str(
                pilot.app.query_one("#type-group-audio_video", Collapsible).title
            )
    assert "Local Parakeet model folder" not in title, (
        "empty value must be skipped, not rendered dangling"
    )
    assert ": ," not in title
    label, _, pairs_text = title.partition(" — ")
    assert label == "Audio & video"
    parts = pairs_text.split(", ")
    assert parts[-1] == "…", f"omitted pairs need an ellipsis: {title!r}"
    assert len(parts) <= 4, f"more than 3 pairs rendered: {title!r}"


@pytest.mark.asyncio
async def test_collapsed_title_promotes_changed_values_first():
    """A changed option is what the user cares to see in the receipt; it
    must outrank untouched defaults when the cap bites."""
    form = _default_form()
    form.type_options["audio_video"] = {"diarization": True}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/c.mp3"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            title = str(
                pilot.app.query_one("#type-group-audio_video", Collapsible).title
            )
    _, _, pairs_text = title.partition(" — ")
    assert pairs_text.startswith("Speaker diarization: on"), title


@pytest.mark.asyncio
async def test_parakeet_folder_placeholder_is_an_example_not_the_label():
    """(task-3305, MI-16) The empty model-folder Input showed its own label
    as placeholder -- pure stutter. The placeholder is example content."""
    state = build_library_ingest_state(
        (),
        form=_default_form(),
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/c.mp3"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    app = _CanvasHost(state)
    with patch(
        "tldw_chatbook.Widgets.Library.library_ingest_canvas._is_installed",
        return_value=True,
    ):
        async with app.run_test() as pilot:
            model_dir = pilot.app.query_one(
                "#opt-audio_video-transcription_model_dir", Input
            )
            assert model_dir.placeholder == "/path/to/parakeet-model"
            assert model_dir.placeholder != "Local Parakeet model folder"


@pytest.mark.asyncio
async def test_failed_queue_row_bracket_error_renders_clean():
    """(task-3312 #2) The queue-row Static rendered ``escape_markup``ed
    text WITH markup enabled. ``rich.markup.escape`` skips a bracket run
    that never closes as a tag (``[remedy: … [``) while escaping the inner
    closed ones (``[keys]``), and Textual's content markup then leaves the
    FIRST escape's backslash literal -- the live 2026-08-08 receipt showed
    ``\\[web_security]`` verbatim (REPL-pinned: escape+from_markup on this
    exact shape leaks). Verbatim ``markup=False`` rendering: byte-identical
    text, no escape artifacts, nothing swallowed as a tag."""
    error = (
        "note [remedy: check [keys] in config.toml, or set [keys] off]"
    )
    failed = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/[draft] notes.txt",
        state=IngestJobState.FAILED,
        error=error,
    )
    state = build_library_ingest_state((failed,), form=_default_form())
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        row = pilot.app.query_one("#library-ingest-row-0", Static)
        text = row.visual.plain
        assert "[draft] notes.txt" in text, text
        assert "\\[" not in text, text
        # Nothing swallowed as a markup tag: the error text is intact.
        assert error in text, text
