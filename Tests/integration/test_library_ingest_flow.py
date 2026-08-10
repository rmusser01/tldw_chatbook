"""Integration tests for the Library ingest flow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import (
    LibraryIngestJobRegistry,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.app_factory import _build_test_app


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
async def test_inline_consent_gates_start_when_pdf_deps_missing(
    library_screen, tmp_path, monkeypatch
):
    """(task-3314) Select PDF with deps mocked missing: the FIRST Start
    press arms the inline confirm (no modal on any Start path), the SECOND
    press submits and the job lands in the real registry."""
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

    # No modal, no job yet: the gate line carries the consent instead.
    assert screen.app.screen_stack[-1] is screen
    assert screen.app_instance.library_ingest_jobs.jobs() == ()
    assert screen._library_ingest_start_confirm_armed is True

    # Second press (a decision, not a double-click) submits for real.
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    jobs = screen.app_instance.library_ingest_jobs.jobs()
    assert [job.source_path for job in jobs] == [str(pdf)]


def test_options_persist_to_config(monkeypatch):
    """Submitting ingest options persists one atomic configuration batch."""
    saved_batches = []

    def fake_save(section_values):
        saved_batches.append(section_values)
        return True

    monkeypatch.setattr(
        library_screen_module,
        "save_settings_to_cli_config",
        fake_save,
    )

    submitted_jobs = []
    screen = library_screen_module.LibraryScreen.__new__(
        library_screen_module.LibraryScreen
    )
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=lambda **kwargs: submitted_jobs.append(kwargs)
    )
    screen._library_ingest_form = SimpleNamespace(
        type_options={"pdf": {"pdf_engine": "pymupdf"}},
        analyze=False,
        chunk=True,
        chunk_size="1024",
        title="",
        author="",
        keywords="",
        path="doc.pdf",
        preflight=_preflight_result(type_groups={"pdf": ["doc.pdf"]}),
        preflight_checking=False,
    )
    screen._cancel_library_ingest_preflight = lambda: None
    # Seeded by ``__init__`` (bypassed by ``__new__``); ``_do_submit_ingest``
    # bumps it to invalidate in-flight pre-flights (stale-helper repair,
    # task-3300).
    screen._library_ingest_preflight_generation = 0
    screen.refresh = lambda **_kwargs: None
    # Submit schedules the scroll-receipt-into-view callback (task-3304);
    # the real method posts a message this unmounted shortcut cannot.
    screen.call_after_refresh = lambda *_args, **_kwargs: None

    library_screen_module.LibraryScreen._do_submit_ingest(screen, "doc.pdf")

    assert len(submitted_jobs) == 1
    assert saved_batches == [
        {
            "library.ingest_options.pdf": {"pdf_engine": "pymupdf"},
            "library.ingest_options.generic": {
                "analyze": False,
                "chunk": True,
                "chunk_size": 1024,
            },
            # (task-3303 xhigh review round 2, F11) Every NEW snapshot
            # carries the ebook chunk-method explicitly (scheme identity):
            # the job-option builder reads an ABSENT value as "legacy
            # snapshot, keep the pre-branch sentences scheme", so the seed
            # persists too.
            "library.ingest_options.ebook": {"chunk_method": "chapters"},
        }
    ]


def test_snapshot_coerces_display_string_chunk_numbers(monkeypatch):
    """task-3301: the generic panel's Inputs hand back display text
    (``"1000"``); the submitted snapshot must carry ints so processors and
    the persisted config never see a string chunk size/overlap."""
    monkeypatch.setattr(
        library_screen_module,
        "save_settings_to_cli_config",
        lambda section_values: True,
    )

    submitted_jobs = []
    screen = library_screen_module.LibraryScreen.__new__(
        library_screen_module.LibraryScreen
    )
    screen.app_instance = SimpleNamespace(
        submit_library_ingest_job=lambda **kwargs: submitted_jobs.append(kwargs)
    )
    screen._library_ingest_form = SimpleNamespace(
        type_options={
            "generic": {"chunk_size": "1000", "chunk_overlap": "150"}
        },
        analyze=False,
        chunk=True,
        chunk_size="1000",
        title="",
        author="",
        keywords="",
        path="notes.txt",
        preflight=_preflight_result(type_groups={"generic": ["notes.txt"]}),
        preflight_checking=False,
    )
    screen._cancel_library_ingest_preflight = lambda: None
    screen._library_ingest_preflight_generation = 0
    screen.refresh = lambda **_kwargs: None
    # Submit schedules the scroll-receipt-into-view callback (task-3304);
    # the real method posts a message this unmounted shortcut cannot.
    screen.call_after_refresh = lambda *_args, **_kwargs: None

    library_screen_module.LibraryScreen._do_submit_ingest(screen, "notes.txt")

    assert len(submitted_jobs) == 1
    generic = submitted_jobs[0]["ingest_options"]["generic"]
    assert generic["chunk_size"] == 1000
    assert generic["chunk_overlap"] == 150
    assert isinstance(generic["chunk_size"], int)
    assert isinstance(generic["chunk_overlap"], int)


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


# --- task-14820: the forecast must match the receipt ------------------------


@pytest.mark.asyncio
async def test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder(
    tmp_path,
):
    """(task-14820 AC#3) GOVERNANCE: the forecast is measured against what
    the pipeline actually does, not against the other line on screen.

    A mixed folder is staged on an install with no pdf/ebook/OCR backends
    (this venv), the forecast is computed from the REAL pre-flight, and
    the same folder is then run through the REAL submit path -- real
    ``submit_library_ingest_job``, real ``run_parse_job``, real
    ``persist_parsed_media``, real ``MediaDatabase`` (only the process
    pool is an in-process stand-in, per the runner suite's own contract).
    Every forecast bucket is asserted against the terminal job states.

    The old forecast promised ``5 will import`` for this folder and
    delivered 2 -- ``will_import = supported_total - will_match`` counted
    the pdf/epub/png files as imports even though the pre-flight had
    already warned that their required tooling was absent.
    """
    from tldw_chatbook.Library.ingest_capabilities import (
        get_capabilities,
        _is_installed,
    )
    from tldw_chatbook.Library.ingest_preflight import analyze_path
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )
    from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
    from Tests.Library.test_library_ingest_runner import (
        _IngestRunnerHarness,
        _make_db,
        _wait_for_runner_idle,
    )

    # The bug lives in the absence of optional backends; if a future venv
    # installs them the fixture no longer exercises it, so say so loudly
    # rather than passing vacuously.
    for group in ("pdf", "ebook", "image"):
        missing = [
            feature
            for feature in get_capabilities(group).required_features
            if not _is_installed(feature)
        ]
        assert missing, (
            f"{group} tooling is installed in this environment; this "
            "governance fixture cannot exercise the tooling forecast"
        )

    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "notes.txt").write_text("Tides are driven by the moon.")
    (folder / "memo.txt").write_text("A second perfectly ingestible note.")
    (folder / "empty.txt").write_text("")
    (folder / "weird.xyz").write_bytes(b"no handler for this")
    (folder / "doc.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (folder / "book.epub").write_bytes(b"PK not really an epub")
    (folder / "diagram.png").write_bytes(b"\x89PNG not really a png")

    preflight = analyze_path(str(folder))
    forecast = build_ingest_forecast(preflight)
    assert forecast is not None
    # The forecast the user would read at the commit point.
    assert forecast_summary_line(forecast) == (
        "2 will import · 1 will skip · 4 will fail (3 need tooling, 1 empty)"
    )

    db = _make_db(tmp_path, name="forecast-governance.db")
    app = _IngestRunnerHarness(db, worker_count=2)
    try:
        async with app.run_test() as pilot:
            app.submit_library_ingest_job(source_path=str(folder))
            jobs = app.library_ingest_jobs.jobs()
            assert len(jobs) == 7

            terminal = {
                IngestJobState.DONE,
                IngestJobState.FAILED,
                IngestJobState.SKIPPED,
                IngestJobState.CANCELLED,
            }
            for _ in range(600):
                snapshot = app.library_ingest_jobs.jobs()
                if all(job.state in terminal for job in snapshot):
                    break
                await pilot.pause(0.02)
            else:  # pragma: no cover - diagnostic only
                raise AssertionError(
                    "jobs never settled: "
                    f"{[(j.source_path, j.state) for j in snapshot]}"
                )
            await _wait_for_runner_idle(app, pilot)

            outcomes = {
                Path(job.source_path).name: job
                for job in app.library_ingest_jobs.jobs()
            }
            actual_done = sum(
                1
                for job in outcomes.values()
                if job.state is IngestJobState.DONE
            )
            actual_skipped = sum(
                1
                for job in outcomes.values()
                if job.state is IngestJobState.SKIPPED
            )
            actual_failed = sum(
                1
                for job in outcomes.values()
                if job.state is IngestJobState.FAILED
            )

            assert (
                forecast.will_import,
                forecast.will_skip,
                forecast.will_fail,
            ) == (actual_done, actual_skipped, actual_failed), (
                "forecast disagreed with the receipt: "
                f"{[(name, job.state.value) for name, job in outcomes.items()]}"
            )
            # ...and each bucket holds the files the forecast said it would.
            assert forecast.will_fail_tooling == 3
            assert forecast.will_fail_empty == 1
            assert {
                name
                for name, job in outcomes.items()
                if job.state is IngestJobState.FAILED
            } == {"doc.pdf", "book.epub", "diagram.png", "empty.txt"}

            # (task-14821 AC#3) The no-content refusals happen BEFORE any
            # write; the receipt must not call them write errors.
            png_detail = outcomes["diagram.png"].error_detail or {}
            assert png_detail.get("category") == "no_content"
            empty_detail = outcomes["empty.txt"].error_detail or {}
            assert empty_detail.get("category") == "empty_source"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_empty_folder_creates_no_job_at_all(library_screen, tmp_path):
    """(task-14823 AC#3) Pressing Start on an empty folder used to leave a
    permanent "✗ failed · emptydir" receipt in the queue and in Recent
    imports -- for a selection the pre-flight had already diagnosed."""
    screen, pilot = library_screen
    folder = tmp_path / "emptydir"
    folder.mkdir()

    form = screen._library_ingest_form
    form.path = str(folder)
    screen._trigger_preflight(str(folder))
    await screen.app.workers.wait_for_complete()
    await pilot.pause()

    state = screen._build_library_ingest_state()
    assert state.start_enabled is False
    assert state.selection_has_nothing_importable is True

    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    assert screen.app_instance.library_ingest_jobs.jobs() == ()
