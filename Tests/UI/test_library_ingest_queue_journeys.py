"""Import queue keyboard journeys through the production Library shell."""

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Library.library_ingest_state import LibraryIngestLastSubmission


def _queue_host(tmp_path, theme):
    source = tmp_path / "notes.txt"
    source.write_text("Local preflight fixture; never imported.")
    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    app.media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="queue-ui")
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    return source, app, host


async def _stage_draft(screen, pilot, source):
    await _wait_for_library_shell(screen, pilot)
    await pilot.press("i")
    path = await _wait_for_selector(screen, pilot, "#library-ingest-path")
    path.value = str(source)
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._ingest_state.form.preflight is not None
            and screen._ingest_state.form.path == str(source)
            and screen._ingest_state.form.preflight.total_files == 1
            and not screen._ingest_state.form.preflight_checking
        ),
        message="Local preflight did not settle",
    )
    group = await _wait_for_selector(screen, pilot, "#type-group-generic")
    group.collapsed = False
    title = screen.query_one("#library-ingest-title", Input)
    title.value = "Unsaved next import"
    await pilot.pause()
    return title


async def _tab_to(screen, host, pilot, selector, label):
    # Expanded options and compact metadata can add more than twelve stops.
    for _ in range(48):
        target = await _wait_for_selector(screen, pilot, selector)
        if screen.focused is target:
            break
        await pilot.press("tab")
    await _focus(screen, host, pilot, selector, label)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_tab_to_batch_retry_paints_action_above_fold_hint(
    tmp_path, monkeypatch, size, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        screen._ingest_state.last_submission = LibraryIngestLastSubmission(
            source=str(source), title="Previous import"
        )
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        screen.query_one("#library-ingest-start", Button).focus()
        await _focus(screen, host, pilot, "#library-ingest-start", "Start import")
        await _tab_to(
            screen, host, pilot, "#library-ingest-retry-last", "Retry this batch"
        )
        await pilot.press("enter")
        await _focus(
            screen,
            host,
            pilot,
            "#library-ingest-retry-last",
            "Press again to replace form",
        )
        assert screen._ingest_state.retry_confirm_armed
        assert title.value == "Unsaved next import"
        assert screen._ingest_state.form.path == str(source)
        await pilot.press("shift+tab")
        assert screen.focused is screen.query_one("#library-ingest-keywords")
        await pilot.press("tab")
        await _focus(
            screen,
            host,
            pilot,
            "#library-ingest-retry-last",
            "Press again to replace form",
        )
        assert app.library_ingest_jobs.jobs() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_queue_details_and_activity_keep_keyboard_context(
    tmp_path, monkeypatch, size, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    registry = app.library_ingest_jobs
    failed = registry.submit(source_path=str(tmp_path / "failed.txt"))
    registry.mark_failed(failed.job_id, error="Fixture failure: dependency unavailable")
    active = registry.submit(source_path=str(tmp_path / "active.txt"))
    registry.mark_parsing(active.job_id)
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        screen.query_one("#library-ingest-start", Button).focus()
        await _focus(screen, host, pilot, "#library-ingest-start", "Start import")
        details = f"#library-ingest-details-{failed.job_id}"
        await _tab_to(screen, host, pilot, details, "Show details")
        await pilot.press("enter")
        await _focus(screen, host, pilot, details, "Hide details")
        registry.update_progress(active.job_id, progress={"message": "Parsing fixture"})
        await pilot.pause()
        await _focus(screen, host, pilot, details, "Hide details")
        old_details = screen.query_one(details)
        registry.mark_writing(active.job_id)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(details) is not old_details,
            message="Queue transition did not replace its projection",
        )
        await _focus(screen, host, pilot, details, "Hide details")
        await pilot.press("tab")
        await _focus(
            screen, host, pilot, f"#library-ingest-retry-{failed.job_id}", "Retry"
        )
        await pilot.press("tab")
        await _focus(
            screen, host, pilot, f"#library-ingest-dismiss-{failed.job_id}", "Dismiss"
        )
        await pilot.press("shift+tab")
        await _focus(
            screen, host, pilot, f"#library-ingest-retry-{failed.job_id}", "Retry"
        )
        assert screen.query_one("#library-ingest-title", Input) is title
        assert title.value == "Unsaved next import"
        assert screen._ingest_state.form.path == str(source)
        assert len(registry.jobs()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("move_to_draft", [False, True])
async def test_queue_transition_handles_removed_control_and_newer_focus(
    tmp_path, monkeypatch, move_to_draft
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, "textual-dark")
    registry = app.library_ingest_jobs
    active = registry.submit(source_path=str(tmp_path / "active.wav"))
    registry.mark_parsing(active.job_id)
    registry.update_progress(active.job_id, progress={"phase": "transcribing"})
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        cancel = f"#library-ingest-cancel-{active.job_id}"
        (await _wait_for_selector(screen, pilot, cancel)).focus()
        await _focus(screen, host, pilot, cancel, "Cancel")
        registry.mark_failed(active.job_id, error="Fixture failure")
        if move_to_draft:
            # User intent arrives in the gap before the queued recompose.
            screen.set_focus(title)
        await _wait_for_selector(
            screen, pilot, f"#library-ingest-details-{active.job_id}"
        )
        await _focus(
            screen,
            host,
            pilot,
            "#library-ingest-title" if move_to_draft else "#library-ingest-path",
            "Unsaved next import" if move_to_draft else "notes.txt",
        )
        assert not screen.query(cancel)
        assert title.value == "Unsaved next import"
        assert screen._ingest_state.form.path == str(source)


@pytest.mark.asyncio
async def test_queue_focus_entered_after_tick_is_scheduled_survives(tmp_path):
    source, app, host = _queue_host(tmp_path, "textual-dark")
    registry = app.library_ingest_jobs
    failed = registry.submit(source_path=str(tmp_path / "failed.txt"))
    registry.mark_failed(failed.job_id, error="Fixture failure")
    active = registry.submit(source_path=str(tmp_path / "active.txt"))
    registry.mark_parsing(active.job_id)
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _stage_draft(screen, pilot, source)
        details = f"#library-ingest-details-{failed.job_id}"
        old_details = await _wait_for_selector(screen, pilot, details)
        screen.set_focus(screen.query_one("#library-ingest-keywords"))
        registry.mark_writing(active.job_id)
        # Keyboard traversal sets focus synchronously. It may arrive after
        # the listener schedules the update but before the old row is pruned.
        screen.set_focus(old_details)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(details) is not old_details,
            message="Queue transition did not replace its projection",
        )
        await _focus(screen, host, pilot, details, "Show details")


@pytest.mark.asyncio
async def test_pending_details_action_respects_newer_queue_focus(tmp_path):
    source, app, host = _queue_host(tmp_path, "textual-dark")
    registry = app.library_ingest_jobs
    failed = registry.submit(source_path=str(tmp_path / "failed.txt"))
    registry.mark_failed(failed.job_id, error="Fixture failure")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _stage_draft(screen, pilot, source)
        details_id = f"#library-ingest-details-{failed.job_id}"
        await _tab_to(screen, host, pilot, details_id, "Show details")
        details = screen.query_one(details_id, Button)
        retry_id = f"#library-ingest-retry-{failed.job_id}"
        old_retry = screen.query_one(retry_id, Button)
        screen._on_ingest_job_details(Button.Pressed(details))
        screen.set_focus(old_retry)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(retry_id) is not old_retry,
            message="Details action did not replace its projection",
        )
        await _focus(screen, host, pilot, retry_id, "Retry")
        assert str(screen.query_one(details_id, Button).label) == "Hide details"


@pytest.mark.asyncio
async def test_queue_action_that_becomes_disabled_uses_path_fallback(
    tmp_path, monkeypatch
):
    from Tests.Library.test_library_ingest_state import _skipped_job
    from Tests.UI.test_library_ingest_analyze_skipped import _ready_provider

    _ready_provider(monkeypatch)
    source, app, host = _queue_host(tmp_path, "textual-dark")
    app.library_ingest_jobs.restore([_skipped_job()], next_id=2)
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _stage_draft(screen, pilot, source)
        selector = "#library-ingest-analyze-skipped"
        await _tab_to(screen, host, pilot, selector, "Analyze 1 skipped")
        old_action = screen.query_one(selector)
        screen._media_state.analyze_running = True
        screen._update_library_ingest_dynamic_regions()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(selector) is not old_action,
            message="Analysis availability did not reach the queue",
        )
        assert screen.query_one(selector).disabled
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-ingest-path"),
            message=lambda: f"Disabled action left focus at {screen.focused!r}",
        )
        await _focus(screen, host, pilot, "#library-ingest-path", "notes.txt")


@pytest.mark.asyncio
async def test_retry_stays_readable_after_compact_reentry(tmp_path):
    source, _app, host = _queue_host(tmp_path, "textual-dark")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = host.screen
        for size in ((170, 48), (80, 24)):
            await pilot.resize_terminal(*size)
            host.theme = "textual-light" if size == (80, 24) else "textual-dark"
            await _stage_draft(screen, pilot, source)
            screen._ingest_state.last_submission = LibraryIngestLastSubmission(
                source=str(source), title="Previous import"
            )
            screen._update_library_ingest_dynamic_regions()
            await pilot.pause()
            screen.query_one("#library-ingest-start", Button).focus()
            await _tab_to(
                screen, host, pilot, "#library-ingest-retry-last", "Retry this batch"
            )
            await pilot.press("escape")
            await _wait_for_selector(screen, pilot, "#library-hub-action-import")


@pytest.mark.asyncio
async def test_late_layout_after_fast_focus_keeps_retry_readable(tmp_path):
    source, _app, host = _queue_host(tmp_path, "textual-light")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        screen._ingest_state.last_submission = LibraryIngestLastSubmission(
            source=str(source), title="Previous import"
        )
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        retry = screen.query_one("#library-ingest-retry-last", Button)
        screen.set_focus(title, scroll_visible=False)
        # Textual's focus path queues this scroll after refresh. Newer
        # keyboard focus can arrive before that earlier scroll is applied.
        screen.scroll_to_center(title)
        screen.set_focus(retry)
        await pilot.pause()
        await _focus(
            screen, host, pilot, "#library-ingest-retry-last", "Retry this batch"
        )
        assert screen.focused is retry
        assert title.value == "Unsaved next import"


@pytest.mark.asyncio
async def test_earlier_scroll_animation_cannot_hide_visible_retry_focus(tmp_path):
    source, _app, host = _queue_host(tmp_path, "textual-light")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _stage_draft(screen, pilot, source)
        screen._ingest_state.last_submission = LibraryIngestLastSubmission(
            source=str(source), title="Previous import"
        )
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        await _tab_to(
            screen, host, pilot, "#library-ingest-retry-last", "Retry this batch"
        )
        canvas = screen.query_one("#library-ingest-canvas")
        retry = screen.focused
        screen.set_focus(
            screen.query_one("#library-ingest-keywords"), scroll_visible=False
        )
        # An earlier keyboard focus started an animation which has not
        # moved yet. Retry is currently visible, so a reveal can be a no-op.
        canvas.scroll_to(
            y=canvas.scroll_y - 2, animate=True, duration=0.3, immediate=True
        )
        screen.set_focus(retry, scroll_visible=False)
        await pilot.wait_for_scheduled_animations()
        await _focus(
            screen, host, pilot, "#library-ingest-retry-last", "Retry this batch"
        )
