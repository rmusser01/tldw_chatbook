"""Integration tests for the Library ingest flow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.ingest_capabilities import get_capabilities
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


def _force_missing_required_features(monkeypatch, *groups: str) -> None:
    """Make capability-sensitive integration cases independent of installed extras."""
    from tldw_chatbook.Library import ingest_capabilities

    missing = {
        feature
        for group in groups
        for feature in ingest_capabilities.get_capabilities(group).required_features
    }
    monkeypatch.setattr(
        ingest_capabilities,
        "_is_installed",
        lambda feature: feature not in missing,
    )


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
    consent = screen._library_ingest_start_consent
    assert consent is not None
    assert consent.owed is True
    assert consent.active_job_ids == ()
    assert consent.tooling_affected_count > 0

    # Second press (a decision, not a double-click) submits for real.
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    jobs = screen.app_instance.library_ingest_jobs.jobs()
    assert [job.source_path for job in jobs] == [str(pdf)]


@pytest.mark.asyncio
async def test_real_screen_to_app_folder_member_change_rearms_without_queueing(
    library_screen, tmp_path
):
    """The real second press cannot authorize a changed folder expansion."""
    screen, pilot = library_screen
    app = screen.app_instance
    app.media_db = SimpleNamespace()
    app._top_up_ingest_parse_pool = lambda: None
    folder = tmp_path / "batch"
    folder.mkdir()
    first = folder / "a.txt"
    matching = folder / "b.txt"
    first.write_text("first")
    matching.write_text("matching")
    active = app.library_ingest_jobs.submit(source_path=str(matching))
    form = screen._library_ingest_form
    form.path = str(folder)
    form.preflight = _preflight_result(
        type_groups={"generic": [str(first), str(matching)]},
        total_files=2,
    )

    screen._submit_library_ingest_form()
    armed = screen._library_ingest_start_consent
    assert armed is not None
    (folder / "added.txt").write_text("added")
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    await pilot.pause()

    jobs = app.library_ingest_jobs.jobs()
    assert [job.job_id for job in jobs] == [active.job_id]
    rearmed = screen._library_ingest_start_consent
    assert rearmed is not None
    assert rearmed.candidate_changed is True
    assert rearmed.admission_scope.candidate_count == 3
    assert rearmed.admission_scope.candidate_digest != (
        armed.admission_scope.candidate_digest
    )


def test_options_persist_to_config(monkeypatch):
    """Submitting ingest options persists one atomic configuration batch.

    (task-15782) The "generic" group's expected persisted dict is derived
    from ``ingest_capabilities.get_capabilities("generic").fields`` -- the
    same schema `_build_ingest_options_snapshot` reads to fill every unset
    field (`library_screen.py`'s `generic.setdefault(field.name,
    field.default)`) -- instead of a hand-copied literal dict. A hardcoded
    copy of that dict already drifted out from under this test once
    (task-15470's notes: the assertion started failing on a "content
    mismatch" once an earlier `run_worker` crash stopped masking it; fixed
    reactively in 0acc6eeeb by hand-adding the seven fields the schema had
    grown since the dict was last hand-copied -- `overwrite_existing`,
    `custom_prompt`, `system_prompt`, `generate_embeddings`,
    `keep_original_file`, `chunk_overlap`, `encoding`). That fix was itself
    just another hardcoded dict, due to rot again the next time a
    ``generic`` field is added or its default changes. Deriving the
    expectation from the schema dataclass (never from
    `_build_ingest_options_snapshot` itself, which would make the
    assertion tautological) keeps the test self-updating for that class of
    change while still catching the two regressions that actually matter:
    a submitted override (`form.analyze`/`chunk`/`chunk_size`) failing to
    win over the schema default, and the snapshot silently DROPPING a
    schema field before it reaches the persisted batch (the task-3309
    silent-drop class).
    """
    saved_batches = []

    def fake_save(section_values):
        saved_batches.append(section_values)
        return True

    submitted_jobs = []
    screen = library_screen_module.LibraryScreen.__new__(
        library_screen_module.LibraryScreen
    )
    # task-15470: the actual write moved into a `@work(thread=True)`
    # instance method (`_save_library_ingest_options`), which needs a
    # running app to dispatch through `run_worker` -- this screen was never
    # mounted. Patching that instance method (rather than the module-level
    # `save_settings_to_cli_config` it wraps) keeps this test's own
    # subject -- the one atomic batch shape -- intact.
    screen._save_library_ingest_options = fake_save
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
    screen._library_ingest_start_consent = None
    screen.refresh = lambda **_kwargs: None
    # Submit schedules the scroll-receipt-into-view callback (task-3304);
    # the real method posts a message this unmounted shortcut cannot.
    screen.call_after_refresh = lambda *_args, **_kwargs: None

    library_screen_module.LibraryScreen._do_submit_ingest(screen, "doc.pdf")

    assert len(submitted_jobs) == 1

    # Source of truth for every "generic" field the submission did not
    # explicitly override -- the same schema production reads from.
    expected_generic = {
        field.name: field.default for field in get_capabilities("generic").fields
    }
    expected_generic.update(
        {
            # Explicit form overrides (set on ``form`` above); asserted as
            # literals rather than schema defaults because that is exactly
            # what this test exercises -- a submitted value winning over
            # the schema default, not just the default surviving untouched.
            "analyze": False,  # form.analyze
            "chunk": True,  # form.chunk
            "chunk_size": 1024,  # form.chunk_size == "1024", coerced to int
        }
    )

    # (task-3303 xhigh review round 2, F11) Every NEW snapshot carries the
    # ebook chunk-method explicitly (scheme identity): the job-option
    # builder reads an ABSENT value as "legacy snapshot, keep the
    # pre-branch sentences scheme", so the seed persists too. Mirror
    # production's own schema lookup + "chapters" fallback (`library_
    # screen.py::_build_ingest_options_snapshot`) instead of hardcoding
    # the resolved literal.
    expected_ebook_chunk_method = next(
        (
            field.default
            for field in get_capabilities("ebook").fields
            if field.name == "chunk_method"
        ),
        "chapters",
    )

    assert saved_batches == [
        {
            "library.ingest_options.pdf": {"pdf_engine": "pymupdf"},
            "library.ingest_options.generic": expected_generic,
            "library.ingest_options.ebook": {
                "chunk_method": expected_ebook_chunk_method
            },
        }
    ]


def test_snapshot_coerces_display_string_chunk_numbers(monkeypatch):
    """task-3301: the generic panel's Inputs hand back display text
    (``"1000"``); the submitted snapshot must carry ints so processors and
    the persisted config never see a string chunk size/overlap."""
    submitted_jobs = []
    screen = library_screen_module.LibraryScreen.__new__(
        library_screen_module.LibraryScreen
    )
    # task-15470: see `test_options_persist_to_config` above for why this
    # patches the `@work(thread=True)` instance method rather than the
    # module-level config function it wraps.
    screen._save_library_ingest_options = lambda section_values: True
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
    screen._library_ingest_start_consent = None
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
    tmp_path, monkeypatch
):
    """(task-14820 AC#3) GOVERNANCE: the forecast is measured against what
    the pipeline actually does, not against the other line on screen.

    A mixed folder is staged with pdf/ebook/OCR capabilities forced missing,
    the forecast is computed from the REAL pre-flight, and
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

    _force_missing_required_features(monkeypatch, "pdf", "ebook", "image")

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


# --- task-14827: the SAME governance, for a SERVER submission ---------------


class _RecordingIngestTransport:
    """The HTTP transport, and ONLY the HTTP transport.

    (task-14827 AC#2) The narrowest boundary the server path can be cut
    at from a test: ``ServerMediaReadingService`` is the real one, so the
    real ``MediaIngestJobSubmitRequest`` is built and the real response
    models are parsed and reconciled -- this stands in for
    ``TLDWAPIClient`` alone, i.e. for the network.

    Two rules keep it from becoming the "fake written to match the call
    site" the repo has been burned by:

    * every call is bound against ``inspect.signature`` of the REAL
      ``TLDWAPIClient`` method, so a drifting call site fails here rather
      than being absorbed by ``**kwargs``;
    * a ``media_type`` outside :data:`SERVER_ACCEPTED_MEDIA_TYPES` is
      REJECTED, mirroring the live server's runtime validator that
      constant was established against.

    It accepts everything else it is handed. That is the fabrication
    line: this proves what the app SENDS and what it refuses to send, not
    what a real server then does with a file it received.
    """

    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self._batches: dict[str, list] = {}
        self._next_id = 0

    @staticmethod
    def _bind(name: str, args: tuple, kwargs: dict):
        """Bind a call against the real client's signature, or raise."""
        import inspect

        from tldw_chatbook.tldw_api.client import TLDWAPIClient

        bound = inspect.signature(getattr(TLDWAPIClient, name)).bind(
            None, *args, **kwargs
        )
        bound.apply_defaults()
        return bound.arguments

    async def submit_media_ingest_jobs(self, *args, **kwargs):
        from tldw_chatbook.Library.server_ingest_request import (
            SERVER_ACCEPTED_MEDIA_TYPES,
        )
        from tldw_chatbook.tldw_api.media_reading_schemas import (
            MediaIngestJobItem,
            MediaIngestJobStatus,
            SubmitMediaIngestJobsResponse,
        )

        arguments = self._bind("submit_media_ingest_jobs", args, kwargs)
        request_data = arguments["request_data"]
        file_paths = arguments["file_paths"] or []
        media_type = str(request_data.media_type)
        if media_type not in SERVER_ACCEPTED_MEDIA_TYPES:
            # What the live server's validator does: "Input should be
            # 'video', 'audio', 'document', 'pdf' or 'ebook'".
            raise ValueError(f"Input should be one of {SERVER_ACCEPTED_MEDIA_TYPES}")

        sources = list(file_paths) + list(request_data.urls or [])
        self._next_id += 1
        batch_id = f"batch-{self._next_id}"
        items = []
        statuses = []
        for source in sources:
            self._next_id += 1
            remote_id = self._next_id
            self.submitted.append(
                {
                    "source": source,
                    "media_type": media_type,
                    "remote_id": remote_id,
                }
            )
            items.append(
                MediaIngestJobItem(
                    id=remote_id,
                    source=str(source),
                    source_kind="url" if source in (request_data.urls or []) else "file",
                    status="queued",
                )
            )
            statuses.append(
                MediaIngestJobStatus(
                    id=remote_id, status="completed", job_type="media_ingest"
                )
            )
        self._batches[batch_id] = statuses
        return SubmitMediaIngestJobsResponse(batch_id=batch_id, jobs=items)

    async def list_media_ingest_jobs(self, *args, **kwargs):
        from tldw_chatbook.tldw_api.media_reading_schemas import (
            MediaIngestJobListResponse,
        )

        arguments = self._bind("list_media_ingest_jobs", args, kwargs)
        batch_id = str(arguments["batch_id"])
        return MediaIngestJobListResponse(
            batch_id=batch_id,
            jobs=list(self._batches.get(batch_id, ())),
            has_more=False,
        )


@pytest.mark.asyncio
async def test_forecast_counts_equal_the_real_receipt_for_a_server_submission(
    tmp_path, monkeypatch
):
    """(task-14827 AC#2) GOVERNANCE, server edition.

    ``test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder``
    drives the LOCAL submit path only, and that blind spot shipped two
    server-path divergences in one review round: local tooling gaps
    subtracted from a server-bound forecast (fixed in the 14820-14826
    arc), and an unsupported file forecast as "will skip" while
    ``build_server_ingest_kwargs`` raised and the job FAILED (this task).

    Real ``analyze_path``, real ``build_ingest_forecast``, real
    ``submit_library_ingest_job`` routing, real
    ``build_server_ingest_kwargs``, real ``ServerMediaReadingService``,
    real request/response schemas, real registry and real reconciler.
    Only the HTTP transport is a stand-in
    (:class:`_RecordingIngestTransport`) -- see its docstring for exactly
    what that means the assertions do and do not prove.

    (task-14910) It now DOES hold a 0-byte file. It deliberately held
    none while the app sent one: the forecast counted it as a failure, but
    the outcome belonged to a server this process cannot inspect, so any
    assertion here would have been the stub deciding. The client refuses a
    0-byte file itself now -- ``empty.txt`` below never reaches
    ``transport.submitted``, and its fate is decided entirely by code this
    test runs for real.
    """
    from tldw_chatbook.Library.ingest_preflight import analyze_path
    from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )
    from tldw_chatbook.Media.server_media_reading_service import (
        ServerMediaReadingService,
    )
    import tldw_chatbook.app as app_module
    from Tests.Library.test_library_ingest_runner import (
        _IngestRunnerHarness,
        _make_db,
    )

    # The .mp3 below re-pins the first divergence under a deterministic local
    # capability gap, even when the developer venv has the audio extra.
    _force_missing_required_features(monkeypatch, "audio_video")

    folder = tmp_path / "mixed-server"
    folder.mkdir()
    (folder / "notes.txt").write_text("Tides are driven by the moon.")
    (folder / "memo.txt").write_text("A second perfectly ingestible note.")
    (folder / "talk.mp3").write_bytes(b"ID3 not really an mp3")
    (folder / "diagram.png").write_bytes(b"\x89PNG not really a png")
    (folder / "weird.xyz").write_bytes(b"no handler for this")
    # (task-14910) A perfectly well-mapped type (.txt -> document) with
    # nothing in it: the one file whose server-side fate used to be
    # unknowable from here.
    (folder / "empty.txt").write_text("")

    preflight = analyze_path(str(folder))
    forecast = build_ingest_forecast(preflight, targets_server=True)
    assert forecast is not None
    # The line the user reads at the commit point, captured BEFORE the run.
    # Asserted after the receipt below, so that a broken forecast fails on
    # the governance claim itself rather than on its wording.
    commit_line = forecast_summary_line(forecast)

    transport = _RecordingIngestTransport()
    real_get_cli_setting = app_module.get_cli_setting

    def _server_backend(*args, **kwargs):
        if args[:2] == ("library.ingest", "backend"):
            return "server"
        return real_get_cli_setting(*args, **kwargs)

    monkeypatch.setattr(app_module, "get_cli_setting", _server_backend)

    db = _make_db(tmp_path, name="server-forecast-governance.db")
    app = _IngestRunnerHarness(db)
    # The two preconditions ``_resolve_ingest_backend`` reads: the opt-in
    # (patched above) and a server-mode runtime.
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_source="server")
    )
    app.server_media_reading_service = ServerMediaReadingService(transport)
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    try:
        async with app.run_test() as pilot:
            assert app._resolve_ingest_backend() == "server"
            app.submit_library_ingest_job(source_path=str(folder))
            assert len(app.library_ingest_jobs.jobs()) == 6

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
                    "server jobs never settled: "
                    f"{[(j.source_path, j.state) for j in snapshot]}"
                )

            outcomes = {
                Path(job.source_path).name: job
                for job in app.library_ingest_jobs.jobs()
            }
            actual_done = sum(
                1 for job in outcomes.values() if job.state is IngestJobState.DONE
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
                "server forecast disagreed with the receipt: "
                f"{[(name, job.state.value) for name, job in outcomes.items()]}"
            )
            assert {
                name
                for name, job in outcomes.items()
                if job.state is IngestJobState.FAILED
            } == {"diagram.png", "weird.xyz", "empty.txt"}
            # (task-14910) ...and the 0-byte one failed HERE, without the
            # transport being consulted: the stub decides nothing about it.
            assert "empty" in (outcomes["empty.txt"].error or "")
            assert outcomes["empty.txt"].permanent is True
            # ...and the ones it promised to SEND really reached the wire,
            # under the media types the server accepts.
            assert {
                Path(record["source"]).name: record["media_type"]
                for record in transport.submitted
            } == {
                "notes.txt": "document",
                "memo.txt": "document",
                "talk.mp3": "audio",
            }
            # ...and the sentence that carried those numbers to the user.
            assert commit_line == (
                "3 will be sent to the server · 3 will fail (2 unsupported "
                "by the server, 1 empty) · server tooling isn't checked "
                "from here"
            ), commit_line
    finally:
        db.close_connection()


# --- task-14911: the gate must ask the backend the run is aimed at ----------


@pytest.mark.asyncio
async def test_server_mode_start_creates_no_job_for_a_selection_the_server_refuses(
    library_screen, tmp_path
):
    """(task-14911 AC#1) The server counterpart of
    ``test_empty_folder_creates_no_job_at_all``.

    A folder of nothing but images imports fine on this machine and has no
    server media type at all (task-3307). Aimed at the server it forecast
    "0 will be sent to the server · 2 will fail (unsupported by the
    server)" while Start stayed ENABLED, and pressing it queued two rows
    that could only ever land as permanent failures -- the guaranteed
    failure submit task-14823 gated on the local path.

    Driven through the REAL pre-flight and the REAL submit seam, and the
    assertion is the registry's own contents: gating the button alone
    would leave Enter in the path field (and every other caller) free to
    manufacture the receipt.
    """
    screen, pilot = library_screen
    folder = tmp_path / "shots"
    folder.mkdir()
    (folder / "one.png").write_bytes(b"\x89PNG not really a png")
    (folder / "two.png").write_bytes(b"\x89PNG not really a png either")

    app = screen.app_instance
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_source="server", server_configured=True)
    )
    app.server_media_reading_service = SimpleNamespace(
        submit_ingest_jobs=lambda **kwargs: None
    )
    app._resolve_ingest_backend = lambda: "server"
    screen._server_binding_is_shipped_placeholder = lambda: False
    # Every OTHER gate open, so a closed Start can only mean this one: the
    # harness leaves ``media_db`` unset, which closes it for an unrelated
    # reason and would make this test pass vacuously.
    app.media_db = SimpleNamespace()
    app._top_up_ingest_parse_pool = lambda: None

    form = screen._library_ingest_form
    form.path = str(folder)
    screen._trigger_preflight(str(folder))
    await screen.app.workers.wait_for_complete()
    await pilot.pause()

    state = screen._build_library_ingest_state()
    assert state.ingest_backend == "server"
    assert state.forecast is not None
    assert state.forecast.will_import == 0
    assert state.start_enabled is False
    assert state.selection_has_nothing_importable is True
    assert "unsupported by the server" in state.start_quiet_line

    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    assert screen.app_instance.library_ingest_jobs.jobs() == (), (
        "a submit the server was certain to refuse reached the queue"
    )


@pytest.mark.asyncio
async def test_the_same_folder_still_imports_on_this_machine(
    library_screen, tmp_path, monkeypatch
):
    """(task-14911 AC#2) Guard, through the same real pre-flight: the gate
    is a fact about the TARGET, not about the files. Locally these images
    are ordinary import candidates, so Start stays live and the submit
    reaches the queue."""
    screen, pilot = library_screen
    _force_missing_required_features(monkeypatch, "image")
    folder = tmp_path / "shots-local"
    folder.mkdir()
    (folder / "one.png").write_bytes(b"\x89PNG not really a png")

    app = screen.app_instance
    app.media_db = SimpleNamespace()
    app._top_up_ingest_parse_pool = lambda: None

    form = screen._library_ingest_form
    form.path = str(folder)
    screen._trigger_preflight(str(folder))
    await screen.app.workers.wait_for_complete()
    await pilot.pause()

    state = screen._build_library_ingest_state()
    assert state.ingest_backend == "local"
    assert state.start_enabled is True
    assert state.selection_has_nothing_importable is False

    # The local press takes the CONSENT route (the fixture forces OCR
    # unavailable, so the image import is at risk) -- not the refusal route.
    screen._submit_library_ingest_form()
    await pilot.pause()
    consent = screen._library_ingest_start_consent
    assert consent is not None
    assert consent.owed is True
    assert consent.active_job_ids == ()
    assert consent.tooling_affected_count > 0
    screen._library_ingest_start_confirm_armed_at -= 1.0
    screen._submit_library_ingest_form()
    await pilot.pause()
    await pilot.pause()

    assert len(screen.app_instance.library_ingest_jobs.jobs()) == 1
