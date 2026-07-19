"""Tests for ``LibraryIngestCanvas`` pre-flight summary rendering.

Widget-only tests mount the canvas directly in a bare ``App`` subclass and
assert on widget existence and rendered text. The canvas is render-only: all
pre-flight state is supplied by ``build_library_ingest_state``.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.ingest_types import PreflightResult
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
    """Without a pre-flight result, no summary widgets are mounted."""
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
        ):
            assert len(pilot.app.query(widget_id)) == 0


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
