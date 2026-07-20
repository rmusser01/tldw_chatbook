"""Tests for the Library ingest guardrail confirmation modal."""

from pathlib import Path
from unittest.mock import MagicMock

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
        assert "Some files may fail to ingest:" in labels
        assert any("PDF processing (3 files):" in label for label in labels)
        assert any("OCR document extraction (3 files):" in label for label in labels)


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
    screen._notify_library_ingest_warning = MagicMock()
    screen.push_screen = MagicMock()
    screen.refresh = MagicMock()
    screen.app_instance = MagicMock()
    return screen


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

    screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen.push_screen.called
    modal = screen.push_screen.call_args.args[0]
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

    screen._submit_library_ingest_form()

    screen.app_instance.submit_library_ingest_job.assert_not_called()
    assert screen.push_screen.called
    modal = screen.push_screen.call_args.args[0]
    callback = screen.push_screen.call_args.args[1]
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

    screen._submit_library_ingest_form()

    screen.push_screen.assert_not_called()
    screen.app_instance.submit_library_ingest_job.assert_called_once()
    call_kwargs = screen.app_instance.submit_library_ingest_job.call_args.kwargs
    assert call_kwargs["source_path"] == str(txt)
