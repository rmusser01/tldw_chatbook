"""LibraryScreen rail-level UI tests."""

import pytest
import pytest_asyncio
from textual.widgets import Button

from tldw_chatbook.Library.library_ingest_jobs import DEFAULT_CHUNK_SIZE
from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


@pytest_asyncio.fixture
async def library_screen():
    """Provide a mounted LibraryScreen with its rail fully loaded."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _wait_for_library_shell(host.screen, pilot)
        yield host.screen


@pytest.mark.asyncio
async def test_ingest_button_present(library_screen):
    """The rail-top Ingest button is rendered with the expected label."""
    button = library_screen.query_one("#library-ingest-top-button", Button)
    assert str(button.label) == "Ingest content…"


# ----- Ingest options snapshot (Task 13) ------------------------------------


def _minimal_ingest_screen() -> LibraryScreen:
    """Return a LibraryScreen instance without mounting the full UI."""
    screen = object.__new__(LibraryScreen)
    screen._library_ingest_form = LibraryIngestFormState()
    return screen


def test_build_ingest_options_snapshot_returns_shallow_copy() -> None:
    screen = _minimal_ingest_screen()
    form = screen._library_ingest_form
    form.type_options = {
        "pdf": {"pdf_engine": "docling"},
        "audio_video": {"transcription_model": "small"},
    }

    snapshot = screen._build_ingest_options_snapshot()

    assert snapshot is not form.type_options
    assert snapshot["pdf"] is not form.type_options["pdf"]
    assert snapshot["pdf"] == {"pdf_engine": "docling"}
    assert snapshot["audio_video"] == {"transcription_model": "small"}
    assert "generic" in snapshot


def test_build_ingest_options_snapshot_includes_generic_toggles() -> None:
    screen = _minimal_ingest_screen()
    form = screen._library_ingest_form
    form.analyze = True
    form.chunk = True
    form.chunk_size = "2048"

    snapshot = screen._build_ingest_options_snapshot()

    assert snapshot["generic"] == {
        "analyze": True,
        "chunk": True,
        "chunk_size": 2048,
    }


def test_build_ingest_options_snapshot_merges_generic_without_clobbering() -> None:
    screen = _minimal_ingest_screen()
    form = screen._library_ingest_form
    form.type_options = {
        "generic": {"encoding": "utf-8"},
        "pdf": {"pdf_engine": "pymupdf"},
    }
    form.analyze = False
    form.chunk = False
    form.chunk_size = "1024"

    snapshot = screen._build_ingest_options_snapshot()

    assert snapshot["generic"] == {
        "encoding": "utf-8",
        "analyze": False,
        "chunk": False,
        "chunk_size": 1024,
    }
    assert snapshot["pdf"] == {"pdf_engine": "pymupdf"}


def test_build_ingest_options_snapshot_clamps_invalid_chunk_size() -> None:
    screen = _minimal_ingest_screen()
    form = screen._library_ingest_form
    form.chunk_size = "not-a-number"

    snapshot = screen._build_ingest_options_snapshot()

    # clamp_chunk_size returns DEFAULT_CHUNK_SIZE for non-integer input.
    assert snapshot["generic"]["chunk_size"] == DEFAULT_CHUNK_SIZE
