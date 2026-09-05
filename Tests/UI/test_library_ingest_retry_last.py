"""task-3313: "Retry this batch" — re-stage the last ingest submission.

After Start the form auto-clears, but the likeliest next action after a
failure (or after installing the dependency a warning just named) is the
SAME source again. One visible action re-stages the last submission's
source with its options and metadata restored to the form, and runs a
FRESH pre-flight (the old forecast is never reused — tooling installed
since the last run must change the forecast).

The affordance is canvas-level, always-mounted, display-managed chrome —
NEVER conditionally composed, and deliberately OUTSIDE the recomposing
queue panel so it keeps object identity across job ticks (the in-place
update discipline; AC#4's identity test lives here).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
    wire_bypass_ingest_controller,
)


def _preflight(**overrides) -> PreflightResult:
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


def _job(**overrides) -> LibraryIngestJob:
    defaults = dict(
        job_id="ingest-job-1",
        source_path="/tmp/example.txt",
        state=IngestJobState.QUEUED,
        submitted_at=100.0,
    )
    defaults.update(overrides)
    return LibraryIngestJob(**defaults)


# --- state builder: visibility matrix ----------------------------------------


def test_retry_last_hidden_without_a_last_submission():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), last_submission_available=False
    )
    assert state.show_retry_last is False


def test_retry_last_hidden_while_a_job_is_still_active():
    """AC#1's framing: the affordance appears once the batch has settled —
    an active job means the submission has NOT reached a terminal state."""
    for active in (
        IngestJobState.QUEUED,
        IngestJobState.PARSING,
        IngestJobState.WRITING,
    ):
        state = build_library_ingest_state(
            (_job(state=active),),
            form=LibraryIngestFormState(),
            last_submission_available=True,
        )
        assert state.show_retry_last is False, f"visible with a {active} job"


def test_retry_last_shows_once_the_queue_has_settled():
    state = build_library_ingest_state(
        (
            _job(
                state=IngestJobState.FAILED,
                error="boom",
                finished_at=101.0,
            ),
        ),
        form=LibraryIngestFormState(),
        last_submission_available=True,
    )
    assert state.show_retry_last is True

    empty_queue = build_library_ingest_state(
        (), form=LibraryIngestFormState(), last_submission_available=True
    )
    assert empty_queue.show_retry_last is True


# --- snapshot capture at submit ----------------------------------------------


def test_do_submit_ingest_captures_the_last_submission_snapshot(tmp_path):
    """The snapshot is taken BEFORE the form clears: source, metadata,
    generic toggles, and a per-group copy of the options."""
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens.library_screen import (
        LibraryIngestState,
        LibraryScreen,
    )

    screen = object.__new__(LibraryScreen)
    screen._ingest_state = LibraryIngestState()
    wire_bypass_ingest_controller(screen)
    form = LibraryIngestFormState(path="/tmp/talk.mp3")
    form.title = "My title"
    form.author = "An author"
    form.keywords = "one, two"
    form.analyze = True
    form.chunk = True
    form.chunk_size = "900"
    form.type_options = {
        "audio_video": {"transcription_provider": "faster-whisper"}
    }
    screen._ingest_state.form = form
    screen._ingest_state.preflight_worker = None
    screen._ingest_state.preflight_generation = 0
    screen._library_selected_row_id = ""
    screen._library_ingest_start_confirm_armed = False
    screen._library_ingest_start_confirm_warnings = []
    screen._ingest_state.last_submission = None
    screen._notify_library_ingest_warning = MagicMock()
    screen.refresh = MagicMock()
    screen.call_after_refresh = MagicMock()
    screen.app_instance = MagicMock()

    # task-15470: the actual write moved into a `@work(thread=True)`
    # instance method (`_save_library_ingest_options`), which needs a
    # running app to dispatch through `run_worker` -- this screen was never
    # mounted. Patching that instance method (rather than the module-level
    # `save_settings_to_cli_config` it wraps) keeps this test's own
    # subject -- the pre-clear submission snapshot -- intact.
    screen._save_library_ingest_options = lambda *_a, **_k: True
    screen._do_submit_ingest("/tmp/talk.mp3")

    snapshot = screen._ingest_state.last_submission
    assert snapshot is not None
    assert snapshot.source == "/tmp/talk.mp3"
    assert snapshot.title == "My title"
    assert snapshot.author == "An author"
    assert snapshot.keywords == "one, two"
    assert snapshot.analyze is True
    assert snapshot.chunk is True
    assert snapshot.chunk_size == "900"
    assert (
        snapshot.type_options["audio_video"]["transcription_provider"]
        == "faster-whisper"
    )
    # The snapshot is a COPY: later form edits must not mutate it.
    form.type_options.setdefault("audio_video", {})["transcription_provider"] = (
        "parakeet-onnx"
    )
    assert (
        snapshot.type_options["audio_video"]["transcription_provider"]
        == "faster-whisper"
    )


# --- keyboard: binding gate + advertisement ------------------------------------


def test_check_action_gates_retry_last_to_ingest_with_a_snapshot():
    """The `r` binding is inert outside Ingest and without a snapshot (the
    bindings-audit contract: gated or universal, never leaking into F1)."""
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestLastSubmission,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    # Landing, no snapshot — inert.
    assert screen.check_action("library_ingest_retry_last", ()) is False

    # Ingest canvas but no snapshot — still inert.
    screen._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA
    assert screen.check_action("library_ingest_retry_last", ()) is False

    # Ingest canvas with a snapshot — active.
    screen._ingest_state.last_submission = LibraryIngestLastSubmission(
        source="/tmp/talk.mp3"
    )
    assert screen.check_action("library_ingest_retry_last", ()) is True

    # A different canvas with a snapshot — inert.
    screen._library_selected_row_id = "browse-conversations"
    assert screen.check_action("library_ingest_retry_last", ()) is False


def test_ingest_shortcuts_advertise_retry_only_when_the_queue_is_settled():
    """Footer/F1 expose Retry only while the shared availability gate is open."""
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestLastSubmission,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    registry = LibraryIngestJobRegistry()
    app.library_ingest_jobs = registry
    screen = LibraryScreen(app)

    assert ("r", "retry") not in screen._library_ingest_shortcuts_for_current_state()

    screen._ingest_state.last_submission = LibraryIngestLastSubmission(
        source="/tmp/talk.mp3"
    )
    registry.restore([_job(state=IngestJobState.PARSING)], next_id=2)
    assert ("r", "retry") not in screen._library_ingest_shortcuts_for_current_state()

    registry.restore(
        [_job(state=IngestJobState.FAILED, error="boom", finished_at=101.0)],
        next_id=2,
    )
    shortcuts = screen._library_ingest_shortcuts_for_current_state()
    assert ("r", "retry") in shortcuts
    assert shortcuts.index(("r", "retry")) < shortcuts.index(("/", "search"))


@pytest.mark.asyncio
async def test_registry_ticks_only_reflow_footer_when_retry_availability_changes(
    monkeypatch,
):
    """Identical registry ticks are footer no-ops; Retry transitions reflow once."""
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestLastSubmission,
    )
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

    app = _pilot_app()
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        footer = screen.query_one(AppFooterStatus)
        real_set_shortcuts = footer.set_workbench_shortcuts
        registrations: list[tuple[str, tuple]] = []

        def record_registration(*, source: str, shortcuts: tuple) -> None:
            registrations.append((source, shortcuts))
            real_set_shortcuts(source=source, shortcuts=shortcuts)

        monkeypatch.setattr(
            footer, "set_workbench_shortcuts", record_registration
        )

        screen._handle_library_ingest_registry_changed()
        screen._handle_library_ingest_registry_changed()
        assert registrations == []

        screen._ingest_state.last_submission = LibraryIngestLastSubmission(
            source="/tmp/talk.mp3"
        )
        screen._handle_library_ingest_registry_changed()
        assert len(registrations) == 1
        assert ("r", "retry") in registrations[0][1]

        screen._handle_library_ingest_registry_changed()
        assert len(registrations) == 1


# --- pilot: restore + fresh preflight + identity --------------------------------


def _pilot_app():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = object()
    return app


async def _ingest_screen(host, pilot):
    screen = host.screen_stack[-1]
    await _wait_for_library_shell(screen, pilot)
    await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
    await _wait_for_selector(screen, pilot, "#library-ingest-path")
    await pilot.pause()
    return screen


async def _submit_batch(screen, pilot, monkeypatch, tmp_path, submitted):
    """Stage a source with metadata + options and submit it (no warnings)."""
    source = tmp_path / "talk.mp3"
    source.write_bytes(b"RIFFxxxx")
    clean = _preflight(
        type_groups={"audio_video": [str(source)]}, total_files=1
    )
    results = {"current": clean}
    monkeypatch.setattr(
        library_screen_module,
        "analyze_path",
        lambda path, scan_limit=1000, **_kwargs: results["current"],
    )

    form = screen._ingest_state.form
    form.title = "My talk"
    form.author = "A speaker"
    form.keywords = "alpha, beta"
    form.type_options["audio_video"] = {
        "transcription_provider": "faster-whisper"
    }
    path_input = screen.query_one("#library-ingest-path", Input)
    path_input.value = str(source)
    screen._trigger_library_ingest_preflight(str(source))
    await _wait_for_condition(
        pilot,
        lambda: screen._ingest_state.form.preflight is not None,
        message="pre-flight never landed",
    )
    await pilot.pause()

    # The programmatic ``value =`` above armed the 0.8s typing debounce;
    # in production it fires (as a no-op on the cleared path) long before
    # a human reaches the retry affordance, but at test speed it can land
    # AFTER the re-stage and mask a dropped fresh-preflight trigger
    # (caught by the mutation check: the debounce re-ran the analysis the
    # mutant no longer requested). Stop it so the only trigger left is
    # the one under test.
    debounce = screen._ingest_state.path_debounce_timer
    if debounce is not None:
        debounce.stop()
        screen._ingest_state.path_debounce_timer = None

    screen._submit_library_ingest_form()
    await pilot.pause()
    assert [k.get("source_path") for k in submitted] == [str(source)]
    assert screen._ingest_state.form.path == ""
    assert screen._ingest_state.form.title == ""
    return str(source), results


@pytest.mark.asyncio
async def test_retry_last_restores_the_form_and_runs_a_fresh_preflight(
    monkeypatch, tmp_path
):
    """AC#1 + AC#3: the press restores source/options/metadata AND the
    forecast is re-derived — tooling that went missing (or got installed)
    since the last run changes it, so the old forecast was not reused."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        source, results = await _submit_batch(
            screen, pilot, monkeypatch, tmp_path, submitted
        )

        retry = await _wait_for_selector(
            screen, pilot, "#library-ingest-retry-last"
        )
        await _wait_for_condition(
            pilot,
            lambda: retry.display,
            message="retry-last affordance never became visible",
        )

        # The user meddled with the form after the submit — options and
        # metadata persist across submits by design, so WITHOUT this
        # mutation the restore assertions below would be vacuously true
        # (caught by the drop-the-options-restore mutation check: the
        # never-cleared form satisfied them).
        form = screen._ingest_state.form
        form.type_options["audio_video"]["transcription_provider"] = (
            "parakeet-onnx"
        )
        form.author = "Somebody Else"
        form.keywords = "gamma"

        # The environment changed between runs: the availability probe now
        # reports a missing dependency (AC#3's simulate-by-toggling rule).
        warned = _preflight(
            type_groups={"audio_video": [source]},
            warnings=[
                {
                    "feature": "audio_processing",
                    "label": "Audio processing",
                    "hint": "audio transcription",
                }
            ],
            total_files=1,
        )
        results["current"] = warned

        # Meddling with the form above is exactly the "would discard work"
        # leg of the destructive-re-stage consent added in the xhigh
        # review + live-verify round, so this restore now takes the
        # incumbent two presses. The subject of THIS test is the restore
        # and the fresh forecast, not the press count -- the consent's own
        # two legs are pinned by
        # ``test_retry_over_an_edited_form_takes_two_presses`` and
        # ``test_retry_over_a_pristine_form_re_stages_on_one_press``.
        screen.query_one("#library-ingest-retry-last", Button).press()
        await pilot.pause()
        assert screen._ingest_state.retry_confirm_armed is True
        assert screen._ingest_state.form.author == "Somebody Else", (
            "the arming press must change nothing"
        )
        screen._ingest_state.retry_confirm_armed_at -= 1.0
        screen.query_one("#library-ingest-retry-last", Button).press()
        await pilot.pause()

        # Options + metadata restored to the form...
        form = screen._ingest_state.form
        assert (
            form.type_options["audio_video"]["transcription_provider"]
            == "faster-whisper"
        )
        assert form.title == "My talk"
        assert form.author == "A speaker"
        assert form.keywords == "alpha, beta"
        # ...including the visible widgets after the re-render.
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-ingest-title"))
            and screen.query_one("#library-ingest-path", Input).value == source,
            message="restaged ingest form never finished mounting",
        )
        assert (
            screen.query_one("#library-ingest-title", Input).value == "My talk"
        )

        # AC#3: the FRESH forecast landed — the stale no-warning forecast
        # was not reused.
        await _wait_for_condition(
            pilot,
            lambda: screen._ingest_state.form.preflight is warned,
            message="fresh pre-flight never ran after re-stage",
        )
        tooling = await _wait_for_selector(
            screen, pilot, "#ingest-preflight-tooling-detail"
        )
        if getattr(tooling, "collapsed", False):
            tooling.collapsed = False
        warning = await _wait_for_selector(
            screen, pilot, "#ingest-preflight-warning-0"
        )
        assert "Audio processing" in str(warning.renderable)


@pytest.mark.asyncio
async def test_retry_last_survives_queue_ticks_with_object_identity(
    monkeypatch, tmp_path
):
    """AC#4: the affordance widget must survive the in-place update path —
    same object across dynamic-region ticks, visibility display-managed."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)

        retry_before = screen.query_one("#library-ingest-retry-last", Button)
        assert retry_before.display is False, (
            "affordance visible without a last submission (AC#4)"
        )

        await _submit_batch(screen, pilot, monkeypatch, tmp_path, submitted)

        # Queue ticks take the in-place path: identity must hold while the
        # visibility flips with the state.
        retry_after_submit = screen.query_one(
            "#library-ingest-retry-last", Button
        )
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        assert (
            screen.query_one("#library-ingest-retry-last", Button)
            is retry_after_submit
        )
        assert retry_after_submit.display is True

        # An active job hides it in place — same object, display off.
        app.library_ingest_jobs.submit(source_path="/tmp/other.txt")
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        assert (
            screen.query_one("#library-ingest-retry-last", Button)
            is retry_after_submit
        )
        assert retry_after_submit.display is False


@pytest.mark.asyncio
async def test_r_key_re_stages_when_focus_is_not_in_a_text_field(
    monkeypatch, tmp_path
):
    """AC#2: keyboard-reachable — `r` re-stages from a non-text-entry
    focus, while `r` inside the path field stays a literal keystroke."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        source, _results = await _submit_batch(
            screen, pilot, monkeypatch, tmp_path, submitted
        )

        # Focus a non-text widget (the Browse button) and press `r`.
        screen.set_focus(screen.query_one("#library-ingest-browse", Button))
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()

        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-ingest-path", Input).value
            == source,
            message="`r` never re-staged the last submission",
        )

        # Guard: inside the path field, `r` types.
        path_input = screen.query_one("#library-ingest-path", Input)
        path_input.value = ""
        screen.set_focus(path_input)
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        assert screen.query_one("#library-ingest-path", Input).value == "r"


# --- xhigh review + live-verify round: gate parity + destructive consent -----


def test_check_action_and_state_builder_share_one_retry_predicate():
    """The `r` route and the button must appear and disappear TOGETHER.

    ``check_action`` gated on (Ingest canvas AND snapshot) only, omitting
    the settled-queue condition the state builder uses for
    ``show_retry_last`` — so mid-run, exactly when the button is
    deliberately hidden to prevent a duplicate batch, the key stayed live.
    The two must read ONE predicate; duplicating the condition is the
    defect, not the wording.
    """
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestLastSubmission,
        library_ingest_retry_available,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA
    screen._ingest_state.last_submission = LibraryIngestLastSubmission(
        source="/tmp/talk.mp3"
    )

    # Settled queue: button visible, key live.
    settled = build_library_ingest_state(
        (), form=LibraryIngestFormState(), last_submission_available=True
    )
    assert settled.show_retry_last is True
    assert screen.check_action("library_ingest_retry_last", ()) is True

    # A job is running: the builder hides the button...
    app.library_ingest_jobs.submit(source_path="/tmp/talk.mp3")
    jobs = app.library_ingest_jobs.jobs()
    mid_run = build_library_ingest_state(
        jobs, form=LibraryIngestFormState(), last_submission_available=True
    )
    assert mid_run.show_retry_last is False
    # ...so the key must be inert too.
    assert screen.check_action("library_ingest_retry_last", ()) is False

    # And the one shared predicate agrees with both.
    assert (
        library_ingest_retry_available(jobs, last_submission_available=True)
        is False
    )
    assert (
        library_ingest_retry_available((), last_submission_available=True)
        is True
    )


@pytest.mark.asyncio
async def test_retry_over_an_edited_form_takes_two_presses(
    monkeypatch, tmp_path
):
    """A re-stage overwrites path + title + author + keywords + options
    from the snapshot with no undo. When that would DISCARD work the user
    has entered since the submit, it takes the repo's incumbent two-press
    consent (Clear-finished / Start): the first press arms and changes
    nothing, the second replaces the form."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        source, _results = await _submit_batch(
            screen, pilot, monkeypatch, tmp_path, submitted
        )

        # The user is mid-way through staging something ELSE.
        form = screen._ingest_state.form
        path_input = screen.query_one("#library-ingest-path", Input)
        path_input.value = "/tmp/half-typed-other"
        await pilot.pause()
        form.title = "Half-typed title"
        assert form.path == "/tmp/half-typed-other"

        screen.set_focus(screen.query_one("#library-ingest-browse", Button))
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()

        assert screen._ingest_state.form.path == "/tmp/half-typed-other", (
            "the first `r` destroyed in-progress form content with no "
            "confirmation"
        )
        assert screen._ingest_state.form.title == "Half-typed title"
        retry = screen.query_one("#library-ingest-retry-last", Button)
        assert "again" in str(retry.label).casefold(), (
            f"the pending consent is not visible on the affordance "
            f"({retry.label!r})"
        )

        # Second press (past the repeat-gesture dead zone) replaces it.
        screen._ingest_state.retry_confirm_armed_at -= 1.0
        await pilot.press("r")
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: screen._ingest_state.form.path == source,
            message="the confirmed retry never re-staged",
        )
        assert screen._ingest_state.form.title == "My talk"
        assert (
            "again"
            not in str(
                screen.query_one("#library-ingest-retry-last", Button).label
            ).casefold()
        )


@pytest.mark.asyncio
async def test_retry_over_a_pristine_form_re_stages_on_one_press(
    monkeypatch, tmp_path
):
    """The other leg: right after a submit the form holds nothing the
    re-stage would discard, so consent would be pure friction — one press
    re-stages."""
    app = _pilot_app()
    submitted: list[dict] = []
    monkeypatch.setattr(
        app,
        "submit_library_ingest_job",
        lambda **kwargs: submitted.append(kwargs),
        raising=False,
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _ingest_screen(host, pilot)
        source, _results = await _submit_batch(
            screen, pilot, monkeypatch, tmp_path, submitted
        )

        screen.set_focus(screen.query_one("#library-ingest-browse", Button))
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()

        await _wait_for_condition(
            pilot,
            lambda: screen._ingest_state.form.path == source,
            message="a pristine form should re-stage on a single press",
        )
        assert screen._ingest_state.retry_confirm_armed is False
