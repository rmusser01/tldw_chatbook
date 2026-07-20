"""Integration tests for the Library ingest flow."""

import pytest
import pytest_asyncio

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


def _preflight_result(**overrides):
    """Build a PreflightResult with sensible defaults."""
    defaults = {
        "type_groups": {},
        "warnings": [],
        "errors": [],
        "total_size": 0,
        "truncated": False,
        "total_files": 0,
    }
    defaults.update(overrides)
    return PreflightResult(**defaults)


@pytest_asyncio.fixture
async def library_screen(tmp_path):
    """Provide a mounted LibraryScreen with ingest seams isolated."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())

    # Use a real in-memory registry and a temp DB so persistence assertions work.
    registry = LibraryIngestJobRegistry()
    db = LibraryIngestJobsDB(tmp_path / "ingest_jobs.db")
    registry.attach_store(db)
    app.library_ingest_jobs = registry

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _wait_for_library_shell(host.screen, pilot)
        yield host.screen, pilot
        db.close()


@pytest.mark.asyncio
async def test_ingest_button_opens_canvas(library_screen):
    """Click top ingest button, verify canvas switches to ingest view."""
    screen, pilot = library_screen
    button = screen.query_one("#library-ingest-top-button")
    assert button is not None

    # Directly invoke the async row-selection handler; button.press() is unreliable
    # for async handlers in the test harness.
    await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
    await pilot.pause()
    await pilot.pause()

    # The ingest canvas should be visible and the path input mounted.
    assert screen.query_one("#library-ingest-path") is not None


@pytest.mark.asyncio
async def test_preflight_detects_pdf(library_screen, tmp_path, monkeypatch):
    """Select a PDF file and verify the pre-flight summary shows it."""
    screen, pilot = library_screen
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("%PDF-1.4 dummy")

    form = screen._library_ingest_form
    form.path = str(pdf)

    # Run pre-flight synchronously to avoid worker timing issues.
    screen._trigger_preflight(str(pdf))
    await screen.app.workers.wait_for_complete()

    assert form.preflight is not None
    assert "pdf" in form.preflight.type_groups
    assert form.preflight.type_groups["pdf"] == [str(pdf)]


@pytest.mark.asyncio
async def test_guardrail_modal_shows_when_pdf_deps_missing(library_screen, tmp_path, monkeypatch):
    """Select PDF with deps mocked missing, verify modal text."""
    screen, pilot = library_screen
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("%PDF-1.4 dummy")

    warning = {
        "feature": "pdf_processing",
        "label": "PDF processing",
        "hint": "Install pdf support",
        "command": "pip install pdfplumber",
    }

    form = screen._library_ingest_form
    form.path = str(pdf)
    form.preflight = _preflight_result(
        type_groups={"pdf": [str(pdf)]},
        warnings=[warning],
    )

    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    # Modal should be pushed and contain the warning text.
    modal = screen.app.screen_stack[-1]
    assert modal.__class__.__name__ == "IngestGuardrailModal"
    texts = [str(w.renderable) for w in modal.query("Static")]
    assert any("PDF processing" in t for t in texts)
    assert any("Install pdf support" in t for t in texts)


@pytest.mark.asyncio
async def test_options_persist_to_config(library_screen, tmp_path, monkeypatch):
    """Change option, start ingest, verify config updated."""
    screen, pilot = library_screen
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("%PDF-1.4 dummy")

    saved_calls = []

    def fake_save(section, key, value):
        saved_calls.append((section, key, value))
        return True

    monkeypatch.setattr(library_screen_module, "save_setting_to_cli_config", fake_save)

    form = screen._library_ingest_form
    form.path = str(pdf)
    form.type_options = {"pdf": {"pdf_engine": "pymupdf"}}
    form.chunk = True
    form.chunk_size = "1024"
    form.preflight = _preflight_result(type_groups={"pdf": [str(pdf)]})

    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    assert ("library.ingest_options.pdf", "pdf_engine", "pymupdf") in saved_calls
    assert ("library.ingest_options.generic", "chunk", True) in saved_calls
    assert ("library.ingest_options.generic", "chunk_size", 1024) in saved_calls


@pytest.mark.asyncio
async def test_job_persists_to_db(library_screen, tmp_path):
    """Start ingest, verify row in Library_Ingest_Jobs_DB has ingest_options."""
    screen, pilot = library_screen
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("%PDF-1.4 dummy")

    form = screen._library_ingest_form
    form.path = str(pdf)
    form.type_options = {"pdf": {"pdf_engine": "pymupdf"}}
    form.preflight = _preflight_result(type_groups={"pdf": [str(pdf)]})

    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    db = LibraryIngestJobsDB(tmp_path / "ingest_jobs.db")
    rows = db.all_jobs()
    assert len(rows) == 1
    assert rows[0]["source_path"] == str(pdf)
    assert '"pdf": {"pdf_engine": "pymupdf"}' in rows[0]["ingest_options"]
    db.close()


@pytest.mark.asyncio
async def test_unsupported_file_not_retryable(library_screen, tmp_path):
    """Ingest unsupported file, verify no Retry button."""
    screen, pilot = library_screen
    unsupported = tmp_path / "file.xyz"
    unsupported.write_text("dummy")

    form = screen._library_ingest_form
    form.path = str(unsupported)
    form.preflight = _preflight_result(type_groups={"generic": [str(unsupported)]})

    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    # Manually mark the job failed as unsupported (as the parse worker would).
    registry = screen.app_instance.library_ingest_jobs
    job = registry.jobs()[0]
    registry.mark_failed(
        job.job_id,
        error="Unsupported file type",
        permanent=False,
        error_detail={"category": "unsupported_file_type", "message": "Unsupported file type"},
    )
    job = registry.get_job(job.job_id)

    # Verify the state layer suppresses retry for unsupported-file-type failures.
    state = screen._build_library_ingest_state()
    row = next((r for r in state.queue_rows if r.job_id == job.job_id), None)
    assert row is not None
    assert row.can_retry is False

    # The state layer already suppresses retry for unsupported-file-type failures.
    # The UI-level rendering is covered by Tests/UI/test_library_ingest_canvas.py.
    # Here we just verify the registry-level flag that drives the UI.
    assert job.error_detail is not None
    assert job.error_detail.get("category") == "unsupported_file_type"
