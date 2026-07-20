"""LibraryScreen rail-level UI tests."""

import pytest
import pytest_asyncio
from textual.widgets import Button
from unittest.mock import MagicMock

from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    IngestJobState,
    LibraryIngestJob,
)
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


# ----- Ingest options persistence/load (Task 17) ----------------------------


def test_do_submit_ingest_persists_options(monkeypatch) -> None:
    """Starting an ingest writes the current option snapshot to config."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.submit_library_ingest_job = MagicMock()
    screen.refresh = MagicMock()

    form = screen._library_ingest_form
    form.path = "/tmp/test.pdf"
    form.title = "A title"
    form.author = "An author"
    form.keywords = "foo, bar"
    form.analyze = True
    form.chunk = False
    form.chunk_size = "1500"
    form.type_options = {
        "pdf": {"pdf_engine": "docling", "ocr": True},
        "audio_video": {"transcription_model": "small"},
    }

    saved: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )

    screen._do_submit_ingest("/tmp/test.pdf")

    assert screen.app_instance.submit_library_ingest_job.called
    assert ("library.ingest_options.pdf", "pdf_engine", "docling") in saved
    assert ("library.ingest_options.pdf", "ocr", True) in saved
    assert ("library.ingest_options.audio_video", "transcription_model", "small") in saved
    assert ("library.ingest_options.generic", "analyze", True) in saved
    assert ("library.ingest_options.generic", "chunk", False) in saved
    assert ("library.ingest_options.generic", "chunk_size", 1500) in saved


def test_load_ingest_options_from_config(monkeypatch) -> None:
    """Mounting the screen restores previously persisted per-type options."""
    screen = _minimal_ingest_screen()

    stored = {
        ("library.ingest_options.pdf", "pdf_engine"): "docling",
        ("library.ingest_options.pdf", "ocr"): True,
        ("library.ingest_options.audio_video", "transcription_model"): "small",
    }

    def fake_get_cli_setting(section: str, key: str = None, default: object = None):
        return stored.get((section, key), default)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        fake_get_cli_setting,
    )

    screen._load_library_ingest_options_from_config()

    assert screen._library_ingest_form.type_options["pdf"] == {
        "pdf_engine": "docling",
        "ocr": True,
    }
    assert screen._library_ingest_form.type_options["audio_video"] == {
        "transcription_model": "small"
    }


# ----- Pre-flight retry (Task 18) -------------------------------------------


def test_trigger_preflight_delegates_to_library_preflight() -> None:
    """``_trigger_preflight`` is a thin seam around the real worker trigger."""
    screen = _minimal_ingest_screen()
    screen._library_ingest_preflight_worker = None
    screen._trigger_library_ingest_preflight = MagicMock()

    screen._trigger_preflight("/tmp/some-file.pdf")

    screen._trigger_library_ingest_preflight.assert_called_once_with(
        "/tmp/some-file.pdf"
    )


def test_on_preflight_retry_triggers_preflight() -> None:
    """Pressing the retry button re-runs pre-flight for the current path."""
    screen = _minimal_ingest_screen()
    screen._library_ingest_preflight_worker = None
    screen._trigger_preflight = MagicMock()
    screen._library_ingest_form.path = "/tmp/retry-target.pdf"

    screen._on_preflight_retry()

    screen._trigger_preflight.assert_called_once_with("/tmp/retry-target.pdf")


# ----- Open in Library fallback (Task 19) -----------------------------------


def _minimal_ingest_job(**kwargs: object) -> LibraryIngestJob:
    """Build a minimal ``LibraryIngestJob`` with safe defaults."""
    defaults: dict[str, object] = {
        "job_id": "ingest-job-1",
        "source_path": "/tmp/test.txt",
        "state": IngestJobState.DONE,
    }
    defaults.update(kwargs)
    return LibraryIngestJob(**defaults)


def test_open_job_in_library_uses_stamped_media_id() -> None:
    """When the job already has a media_id, navigation is immediate."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=42)
    screen._open_job_in_library(job)

    screen._navigate_to_media.assert_called_once_with(42)
    screen.app_instance.media_db.execute_query.assert_not_called()
    screen.notify.assert_not_called()


def test_open_job_in_library_falls_back_to_source_url() -> None:
    """A deduplicated job with a matching source URL resolves to that media row."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.media_db.execute_query.return_value = [{"id": 7}]
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, source_path="/tmp/foo.txt")
    screen._open_job_in_library(job)

    screen.app_instance.media_db.execute_query.assert_called_once_with(
        "SELECT id FROM Media WHERE url = ? ORDER BY created_at DESC LIMIT 1",
        ("/tmp/foo.txt",),
    )
    screen._navigate_to_media.assert_called_once_with(7)
    screen.notify.assert_not_called()


def test_open_job_in_library_falls_back_to_content_hash() -> None:
    """When the URL lookup misses, a recorded content hash is used."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.media_db.execute_query.side_effect = [[], [{"id": 9}]]
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, content_hash="abc123")
    screen._open_job_in_library(job)

    assert screen.app_instance.media_db.execute_query.call_count == 2
    screen._navigate_to_media.assert_called_once_with(9)
    screen.notify.assert_not_called()


def test_open_job_in_library_notifies_when_no_match() -> None:
    """A deduplicated job with no resolvable match shows a transient status."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.media_db.execute_query.return_value = []
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, content_hash="abc")
    screen._open_job_in_library(job)

    screen._navigate_to_media.assert_not_called()
    screen.notify.assert_called_once_with(
        "Already in Library — no single match found"
    )


def test_open_job_in_library_handles_missing_media_db() -> None:
    """The fallback is skipped entirely when the media database is unavailable."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock(media_db=None)
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, content_hash="abc")
    screen._open_job_in_library(job)

    screen._navigate_to_media.assert_not_called()
    screen.notify.assert_called_once_with(
        "Already in Library — no single match found"
    )


@pytest.mark.asyncio
async def test_handle_library_ingest_open_wires_to_open_job_in_library() -> None:
    """The ingest canvas Open button delegates through ``_open_job_in_library``."""
    screen = _minimal_ingest_screen()
    job = _minimal_ingest_job(media_id=123)
    screen._library_ingest_job_by_id = MagicMock(return_value=job)
    screen._open_job_in_library = MagicMock()

    event = MagicMock()
    event.button.id = "library-ingest-open-ingest-job-1"
    await screen.handle_library_ingest_open(event)

    screen._library_ingest_job_by_id.assert_called_once_with("ingest-job-1")
    screen._open_job_in_library.assert_called_once_with(job)
