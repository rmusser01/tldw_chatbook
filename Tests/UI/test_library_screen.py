"""LibraryScreen rail-level UI tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
import toml
from textual.widgets import Button
from unittest.mock import MagicMock

from tldw_chatbook.config import _get_effective_config_path, get_cli_setting
from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    IngestJobState,
    LibraryIngestJob,
)
from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.app_factory import _build_test_app


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
    """The rail-top primary button names the action in plain language
    (F-013) and explains where it takes the user."""
    button = library_screen.query_one("#library-ingest-top-button", Button)
    assert str(button.label) == "Import…"
    assert str(button.tooltip) == "Add files, links, and transcripts to your Library."


# ----- Ingest options snapshot (Task 13) ------------------------------------


def _minimal_ingest_screen() -> LibraryScreen:
    """Return a fully-initialized LibraryScreen without mounting the full UI.

    (task-3022) This used to bypass ``__init__`` entirely via
    ``object.__new__(LibraryScreen)``, leaving every ingest-canvas instance
    attribute ``__init__`` sets -- the pre-flight-cancellation generation
    stamp (task-2011's ``_library_ingest_preflight_generation``),
    ``_library_ingest_clear_finished_armed``, ``_library_selected_row_id``,
    and others -- missing, so any exercised code path that touched one
    raised ``AttributeError`` (e.g. ``_do_submit_ingest`` /
    ``handle_library_ingest_backend_switch`` via
    ``_cancel_library_ingest_preflight``/``_invalidate_library_ingest_
    preflight``, or ``handle_library_ingest_retry_faster_whisper`` /
    ``_build_library_ingest_state`` directly). ``LibraryScreen.__init__`` is
    pure attribute setup -- no I/O, no ``compose()``, no worker starts --
    so constructing for real with a throwaway app stand-in is both cheap
    and immune to future attributes added there. Every test below
    immediately replaces ``app_instance`` with its own ``MagicMock`` for
    its own assertions, same as before.
    """
    screen = LibraryScreen(MagicMock())
    screen._library_ingest_form = LibraryIngestFormState()
    screen._transcribe_cpp_configured = False
    # Set by ``__init__``, which this shortcut bypasses. (task-3303 branch
    # repair: siblings 3301/3302 taught the submit/state paths to read these
    # instance attributes, and this helper had drifted -- three tests in
    # this file were failing at HEAD with bare AttributeErrors.)
    screen._library_ingest_preflight_worker = None
    screen._library_ingest_preflight_generation = 0
    screen._library_selected_row_id = ""
    screen._library_ingest_clear_finished_armed = False
    screen._library_ingest_clear_finished_armed_at = 0.0
    screen._library_ingest_expanded_details = set()
    screen._library_ingest_recent_ledger = []
    # Submit schedules the scroll-receipt-into-view callback (task-3304);
    # the real method posts a message this unmounted shortcut cannot.
    screen.call_after_refresh = lambda *_args, **_kwargs: None
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
        "overwrite_existing": False,
        "custom_prompt": "",
        "system_prompt": "",
        "generate_embeddings": True,
        "keep_original_file": False,
        "chunk": True,
        "chunk_size": 2048,
        "chunk_overlap": 100,
        "encoding": "auto",
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
        "overwrite_existing": False,
        "custom_prompt": "",
        "system_prompt": "",
        "generate_embeddings": True,
        "keep_original_file": False,
        "chunk": False,
        "chunk_size": 1024,
        "chunk_overlap": 100,
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

    # task-15470: the actual write moved into a `@work(thread=True)`
    # instance method (`_save_library_ingest_options`), which needs a
    # running app to dispatch through `run_worker` -- this screen was never
    # mounted. Patching that instance method (rather than the module-level
    # `save_settings_to_cli_config` it wraps) keeps this test's own
    # subject -- the batched shape of what the submit path decides to
    # persist -- intact.
    batches: list[dict] = []
    screen._save_library_ingest_options = lambda section_values: (
        batches.append({s: dict(v) for s, v in section_values.items()}) or True
    )

    screen._do_submit_ingest("/tmp/test.pdf")

    # One batched write, not one full config read/parse/reload per option key.
    assert len(batches) == 1, f"expected a single batched save, got {len(batches)}"
    saved = [
        (section, key, value)
        for section, values in batches[0].items()
        for key, value in values.items()
    ]

    assert screen.app_instance.submit_library_ingest_job.called
    assert ("library.ingest_options.pdf", "pdf_engine", "docling") in saved
    assert ("library.ingest_options.pdf", "ocr", True) in saved
    assert (
        "library.ingest_options.audio_video",
        "transcription_model",
        "small",
    ) in saved
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
        ("transcription.transcribe_cpp", "model_path"): "/private/model.gguf",
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
    assert screen._transcribe_cpp_configured is True
    assert "/private/model.gguf" not in repr(screen._library_ingest_form)


def test_task_3303_options_round_trip_persisted_config(monkeypatch) -> None:
    """(task-3303 AC6) Every new per-type option persists and loads back.

    Drives the two REAL seams -- ``_do_submit_ingest``'s batched save and
    ``_load_library_ingest_options_from_config`` -- with only the config I/O
    stubbed as a dict, so a field missing from the capability schema (the
    load path iterates ``cap.field_names``) fails here.
    """
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.submit_library_ingest_job = MagicMock()
    screen.refresh = MagicMock()

    form = screen._library_ingest_form
    form.path = "/tmp/report.docx"
    submitted_options = {
        "document": {
            "processing_method": "docling",
            "ocr": True,
            "ocr_language": "de",
        },
        "pdf": {
            "pdf_engine": "docext",
            "ocr": True,
            "ocr_language": "fr",
            "ocr_backend": "tesseract",
        },
        "ebook": {"chunk_method": "chapters"},
        "audio_video": {"translate_to_english": True, "vad_filter": True},
        "web": {"scrape_method": "sitemap", "max_pages": 5},
    }
    form.type_options = {g: dict(v) for g, v in submitted_options.items()}

    # task-15470: see `test_do_submit_ingest_persists_options` for why this
    # patches the `@work(thread=True)` instance method rather than the
    # module-level config function it wraps.
    saved_sections: dict[str, dict] = {}
    screen._save_library_ingest_options = lambda section_values: (
        saved_sections.update({s: dict(v) for s, v in section_values.items()})
        or True
    )

    screen._do_submit_ingest("/tmp/report.docx")

    for group, values in submitted_options.items():
        section = f"library.ingest_options.{group}"
        for name, value in values.items():
            assert saved_sections.get(section, {}).get(name) == value, (
                f"{group}.{name} did not persist"
            )

    # Fresh screen, fed only what the save wrote: everything loads back.
    loader = _minimal_ingest_screen()

    def fake_get_cli_setting(section: str, key: str = None, default: object = None):
        return saved_sections.get(section, {}).get(key, default)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        fake_get_cli_setting,
    )

    loader._load_library_ingest_options_from_config()

    restored = loader._library_ingest_form.type_options
    for group, values in submitted_options.items():
        for name, value in values.items():
            assert restored.get(group, {}).get(name) == value, (
                f"{group}.{name} did not load back"
            )


def test_task_3306_av_options_round_trip_persisted_config(monkeypatch) -> None:
    """(task-3306 AC2) Trim, cookies file, recursive summary, and an
    extended-catalog whisper model persist and load back through the same
    two real seams as the task-3303 round-trip above."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.submit_library_ingest_job = MagicMock()
    screen.refresh = MagicMock()

    form = screen._library_ingest_form
    form.path = "/tmp/talk.mp3"
    submitted = {
        "audio_video": {
            "start_time": "0:30",
            "end_time": "10:00",
            "cookies_file": "/home/u/cookies.txt",
            "summarize_recursively": True,
            "transcription_model": "distil-large-v3",
        },
    }
    form.type_options = {g: dict(v) for g, v in submitted.items()}

    # task-15470: see `test_do_submit_ingest_persists_options` for why this
    # patches the `@work(thread=True)` instance method rather than the
    # module-level config function it wraps.
    saved_sections: dict[str, dict] = {}
    screen._save_library_ingest_options = lambda section_values: (
        saved_sections.update({s: dict(v) for s, v in section_values.items()})
        or True
    )

    screen._do_submit_ingest("/tmp/talk.mp3")

    section = saved_sections.get("library.ingest_options.audio_video", {})
    for name, value in submitted["audio_video"].items():
        assert section.get(name) == value, f"audio_video.{name} did not persist"

    loader = _minimal_ingest_screen()

    def fake_get_cli_setting(section: str, key: str = None, default: object = None):
        return saved_sections.get(section, {}).get(key, default)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        fake_get_cli_setting,
    )

    loader._load_library_ingest_options_from_config()

    restored = loader._library_ingest_form.type_options.get("audio_video", {})
    for name, value in submitted["audio_video"].items():
        assert restored.get(name) == value, f"audio_video.{name} did not load back"


def test_transcribe_cpp_picker_filters_to_gguf() -> None:
    filters = library_screen_module._transcribe_cpp_gguf_filters()

    assert filters.selections == [("GGUF models", 0)]
    assert filters[0](Path("model.gguf"))
    assert not filters[0](Path("model.bin"))


def test_transcribe_cpp_config_worker_reports_path_free_success(
    tmp_path, monkeypatch
) -> None:
    screen = _minimal_ingest_screen()
    selected = tmp_path / "private-model.gguf"
    configured: list[Path] = []
    fake_app = MagicMock()
    monkeypatch.setattr(
        LibraryScreen, "app", property(lambda _self: fake_app)
    )
    monkeypatch.setattr(
        library_screen_module,
        "configure_transcribe_cpp_model_path",
        lambda path: configured.append(path),
    )
    screen._apply_transcribe_cpp_gguf_result = MagicMock()

    LibraryScreen._configure_transcribe_cpp_gguf.__wrapped__(
        screen, selected, retry_job_id="ingest-job-1"
    )

    assert configured == [selected]
    fake_app.call_from_thread.assert_called_once_with(
        screen._apply_transcribe_cpp_gguf_result,
        True,
        "ingest-job-1",
    )
    assert str(selected) not in repr(fake_app.call_from_thread.call_args)


def test_transcribe_cpp_config_success_requeues_failed_job_without_path() -> None:
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.refresh = MagicMock()

    screen._apply_transcribe_cpp_gguf_result(True, "ingest-job-1")

    screen.app_instance.retry_library_ingest_job.assert_called_once_with(
        "ingest-job-1"
    )
    assert screen._transcribe_cpp_configured is True
    assert "GGUF configured" in screen.app_instance.notify.call_args.args[0]


def test_faster_whisper_recovery_handler_uses_explicit_provider() -> None:
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.refresh = MagicMock()
    event = MagicMock()
    event.button.id = "library-ingest-retry-faster-whisper-ingest-job-1"

    screen.handle_library_ingest_retry_faster_whisper(event)

    event.stop.assert_called_once_with()
    screen.app_instance.retry_library_ingest_job_with_provider.assert_called_once_with(
        "ingest-job-1", "faster-whisper"
    )


def test_local_cancel_handler_targets_the_job_attempt() -> None:
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.refresh = MagicMock()
    screen.app_instance.library_ingest_jobs.get_job.return_value = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/speech.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing"},
    )
    event = MagicMock()
    event.button.id = "library-ingest-cancel-ingest-job-1"

    screen.handle_library_ingest_cancel(event)

    event.stop.assert_called_once_with()
    screen.app_instance.cancel_local_ingest_job.assert_called_once_with("ingest-job-1")
    screen.app_instance.cancel_remote_ingest_batch.assert_not_called()


def test_local_force_stop_handler_targets_the_job_attempt() -> None:
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.refresh = MagicMock()
    event = MagicMock()
    event.button.id = "library-ingest-force-stop-ingest-job-1"

    screen.handle_library_ingest_force_stop(event)

    event.stop.assert_called_once_with()
    screen.app_instance.force_stop_local_ingest_job.assert_called_once_with(
        "ingest-job-1"
    )


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
    screen.app_instance.media_db.get_media_by_url.return_value = {"id": 7}
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, source_path="/tmp/foo.txt")
    screen._open_job_in_library(job)

    screen.app_instance.media_db.get_media_by_url.assert_called_once_with(
        "/tmp/foo.txt"
    )
    screen._navigate_to_media.assert_called_once_with(7)
    screen.notify.assert_not_called()


def test_open_job_in_library_falls_back_to_content_hash() -> None:
    """When the URL lookup misses, a recorded content hash is used."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.media_db.get_media_by_url.return_value = None
    screen.app_instance.media_db.get_media_by_hash.return_value = {"id": 9}
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, content_hash="abc123")
    screen._open_job_in_library(job)

    screen.app_instance.media_db.get_media_by_url.assert_called_once()
    screen.app_instance.media_db.get_media_by_hash.assert_called_once_with("abc123")
    screen._navigate_to_media.assert_called_once_with(9)
    screen.notify.assert_not_called()


def test_open_job_in_library_notifies_when_no_match() -> None:
    """A deduplicated job with no resolvable match shows a transient status."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.media_db.get_media_by_url.return_value = None
    screen.app_instance.media_db.get_media_by_hash.return_value = None
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, content_hash="abc")
    screen._open_job_in_library(job)

    screen._navigate_to_media.assert_not_called()
    screen.notify.assert_called_once_with("Already in Library — no single match found")


def test_open_job_in_library_handles_missing_media_db() -> None:
    """The fallback is skipped entirely when the media database is unavailable."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock(media_db=None)
    screen._navigate_to_media = MagicMock()
    screen.notify = MagicMock()

    job = _minimal_ingest_job(media_id=None, content_hash="abc")
    screen._open_job_in_library(job)

    screen._navigate_to_media.assert_not_called()
    screen.notify.assert_called_once_with("Already in Library — no single match found")


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


def test_ingest_browse_location_prefers_last_used_then_home(tmp_path, monkeypatch) -> None:
    """The file browser opens somewhere the user actually keeps files.

    It defaulted to ``"."`` -- whichever directory the process was started
    from, which for anyone launching from a shell is arbitrary (task-668).
    """
    screen = _minimal_ingest_screen()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        lambda *args, **kwargs: str(tmp_path),
    )
    assert screen._library_ingest_browse_location() == str(tmp_path)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        lambda *args, **kwargs: None,
    )
    assert screen._library_ingest_browse_location() == str(Path.home())

    # A remembered directory that no longer exists must not be handed back.
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        lambda *args, **kwargs: str(tmp_path / "deleted"),
    )
    assert screen._library_ingest_browse_location() == str(Path.home())


def test_ingest_browse_remembers_the_directory_of_the_picked_file(
    tmp_path, monkeypatch
) -> None:
    """Picking a file stores its folder, so the next Browse starts there."""
    screen = _minimal_ingest_screen()
    picked = tmp_path / "doc.txt"
    picked.write_text("hi")

    saved: list[tuple] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )

    screen._remember_library_ingest_location(picked)

    assert saved == [("library.ingest", "last_directory", str(tmp_path))]


def test_ingestible_file_filters_separate_importable_from_the_rest() -> None:
    """The picker distinguishes files ingest can handle from ones it cannot."""
    from tldw_chatbook.UI.Screens.library_screen import _ingestible_file_filters

    filters = _ingestible_file_filters()
    importable = filters[0]

    assert importable(Path("/tmp/notes.txt")) is True
    assert importable(Path("/tmp/paper.pdf")) is True
    # (task-3307) Images are a supported group now, so the picker offers
    # them; a truly handler-less extension stays filtered out.
    assert importable(Path("/tmp/cover.jpg")) is True
    assert importable(Path("/tmp/subtitles.srt")) is False


def test_backend_switch_flips_and_persists_the_target(monkeypatch) -> None:
    """Switching backends writes the preference, so it survives a restart.

    A user who deliberately points imports at their server should not silently
    be back on local next launch (task-684.1).
    """
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance._resolve_ingest_backend = MagicMock(return_value="local")
    screen.refresh = MagicMock()

    # task-15470: the actual write moved into a `@work(thread=True)`
    # instance method (`_save_library_ingest_backend`), which needs a
    # running app to dispatch through `run_worker` -- this screen was never
    # mounted. Patching that instance method (rather than the module-level
    # `save_setting_to_cli_config` it wraps) keeps this test's own subject
    # -- which target the handler decides to persist -- intact.
    saved: list[tuple] = []
    screen._save_library_ingest_backend = lambda target, _generation: (
        saved.append(("library.ingest", "backend", target)) or True
    )

    screen.handle_library_ingest_backend_switch(MagicMock())

    assert saved == [("library.ingest", "backend", "server")]
    assert screen.refresh.called


def test_backend_switch_returns_to_local(monkeypatch) -> None:
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance._resolve_ingest_backend = MagicMock(return_value="server")
    screen.refresh = MagicMock()

    # task-15470: see the sibling test above for why this patches the
    # `@work(thread=True)` instance method rather than the module-level
    # config function it wraps.
    saved: list[tuple] = []
    screen._save_library_ingest_backend = lambda target, _generation: (
        saved.append(("library.ingest", "backend", target)) or True
    )

    screen.handle_library_ingest_backend_switch(MagicMock())

    assert saved == [("library.ingest", "backend", "local")]


def test_save_library_ingest_backend_writes_the_real_dotted_section() -> None:
    """task-15470 review round: the sibling tests above patch
    `_save_library_ingest_backend` itself (unavoidable -- it is
    `@work(thread=True)`, which needs a running app to dispatch through
    `run_worker`, and none of these screens are mounted), so none of them
    exercise the REAL method body -- specifically its literal
    ``"library.ingest"``/``"backend"`` section/key strings, which a typo
    there (e.g. ``"library.Ingest"``) would let every mocked test above
    stay green through.

    `@work` wraps with `functools.wraps`, so `.__wrapped__` is the
    undecorated body -- calling it directly bypasses `run_worker` (and the
    app it needs) entirely while still running the exact production code,
    against the real (test-sandboxed, per the root conftest's autouse
    profile isolation) `save_setting_to_cli_config`. This is the one
    representative site the review round asked for; the same
    `.__wrapped__` pattern already used for `_prepare_library_external_
    submission` elsewhere in this test suite.
    """
    screen = _minimal_ingest_screen()
    screen._library_ingest_backend_generation = 1

    LibraryScreen._save_library_ingest_backend.__wrapped__(screen, "server", 1)

    assert get_cli_setting("library.ingest", "backend", None) == "server"
    on_disk = toml.load(_get_effective_config_path())
    assert on_disk["library"]["ingest"]["backend"] == "server"


def test_switch_is_not_offered_when_the_server_seam_cannot_submit() -> None:
    """A service object that cannot submit must not advertise a switch.

    Otherwise the canvas offers a toggle whose only outcome is a failed job.
    """
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.media_db = object()
    screen.app_instance._resolve_ingest_backend = MagicMock(return_value="local")
    # A stand-in with no submit methods at all.
    screen.app_instance.server_media_reading_service = object()
    screen._library_ingest_registry = MagicMock(return_value=MagicMock(jobs=lambda: ()))

    state = screen._build_library_ingest_state()

    assert state.show_backend_switch is False


def test_pending_backend_choice_does_not_impersonate_persisted_owner() -> None:
    """The canvas stays on the durable owner until its worker confirms the save."""

    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance._resolve_ingest_backend = MagicMock(return_value="server")
    screen.app_instance.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_source="server", server_configured=True)
    )
    screen.app_instance.media_db = object()
    screen.app_instance.server_media_reading_service = SimpleNamespace(
        submit_ingest_jobs=lambda **_kwargs: None
    )
    screen._server_binding_is_shipped_placeholder = lambda: False
    screen._library_ingest_backend_target = "local"
    screen._library_ingest_registry = MagicMock(return_value=MagicMock(jobs=lambda: ()))

    state = screen._build_library_ingest_state()

    assert state.ingest_backend == "server"


def test_task_3307_image_options_round_trip_persisted_config(monkeypatch) -> None:
    """(task-3307) The image group's options persist on submit and load
    back on a fresh screen -- same two real seams the task-3303 round-trip
    drives, with only config I/O stubbed."""
    screen = _minimal_ingest_screen()
    screen.app_instance = MagicMock()
    screen.app_instance.submit_library_ingest_job = MagicMock()
    screen.refresh = MagicMock()

    form = screen._library_ingest_form
    form.path = "/tmp/scan.png"
    submitted = {
        "image": {"ocr": True, "ocr_language": "de", "ocr_backend": "tesseract"},
    }
    form.type_options = {g: dict(v) for g, v in submitted.items()}

    # task-15470: see `test_do_submit_ingest_persists_options` for why this
    # patches the `@work(thread=True)` instance method rather than the
    # module-level config function it wraps.
    saved_sections: dict[str, dict] = {}
    screen._save_library_ingest_options = lambda section_values: (
        saved_sections.update({s: dict(v) for s, v in section_values.items()})
        or True
    )

    screen._do_submit_ingest("/tmp/scan.png")

    section = saved_sections.get("library.ingest_options.image", {})
    for name, value in submitted["image"].items():
        assert section.get(name) == value, f"image.{name} did not persist"

    loader = _minimal_ingest_screen()

    def fake_get_cli_setting(section: str, key: str = None, default: object = None):
        return saved_sections.get(section, {}).get(key, default)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen.get_cli_setting",
        fake_get_cli_setting,
    )

    loader._load_library_ingest_options_from_config()

    restored = loader._library_ingest_form.type_options
    for name, value in submitted["image"].items():
        assert restored.get("image", {}).get(name) == value, (
            f"image.{name} did not load back"
        )
