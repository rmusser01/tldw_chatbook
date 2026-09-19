"""Recent imports and grouped outcomes through the production Library shell."""

import pytest
from textual.widgets import Collapsible, Input

from Tests.UI.test_library_ingest_queue_journeys import (
    _queue_host,
    _stage_draft,
    _tab_to,
)
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import _wait_for_condition

RECENT_TITLE = "#library-ingest-recent > CollapsibleTitle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,theme", [((170, 48), "textual-dark"), ((80, 24), "textual-light")]
)
@pytest.mark.parametrize(
    "opened,newer_focus", [(True, False), (False, False), (True, True)]
)
async def test_recent_disclosure_survives_queue_transition(
    tmp_path, monkeypatch, size, theme, opened, newer_focus
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    registry = app.library_ingest_jobs
    failed = registry.submit(source_path=str(tmp_path / "failed.txt"))
    registry.mark_failed(failed.job_id, error="Fixture failure")
    active = registry.submit(source_path=str(tmp_path / "active.txt"))
    registry.mark_parsing(active.job_id)
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        screen.query_one("#library-ingest-start").focus()
        await _tab_to(screen, host, pilot, RECENT_TITLE, "Recent imports")
        # Exercise both expansion and explicit collapse, not just the default.
        await pilot.press("enter")
        if not opened:
            await pilot.press("enter")
        recent = screen.query_one("#library-ingest-recent", Collapsible)
        assert recent.collapsed is not opened
        registry.mark_writing(active.job_id)
        if newer_focus:
            screen.set_focus(title)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-ingest-recent") is not recent,
            message="Queue transition did not rebuild Recent imports",
        )
        assert (
            screen.query_one("#library-ingest-recent", Collapsible).collapsed
            is not opened
        )
        await _focus(
            screen,
            host,
            pilot,
            "#library-ingest-title" if newer_focus else RECENT_TITLE,
            "Unsaved next import" if newer_focus else "Recent imports",
        )
        assert screen.query_one("#library-ingest-title", Input) is title
        assert title.value == "Unsaved next import"
        assert screen._ingest_state.form.path == str(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,theme", [((170, 48), "textual-dark"), ((80, 24), "textual-light")]
)
async def test_group_recovery_and_clear_preserve_draft_and_recent_ledger(
    tmp_path, monkeypatch, size, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    registry = app.library_ingest_jobs
    jobs = [
        registry.submit(source_path=str(tmp_path / f"file-{n}.txt"), batch_id="fixture")
        for n in range(4)
    ]
    for job in jobs:
        registry.mark_failed(job.job_id, error="Fixture failure")
    retried = []

    def retry_projection(job_id):
        retried.append(job_id)
        return registry.requeue(job_id)

    # Exercise the real controller and registry, without any import executor.
    app.retry_library_ingest_job = retry_projection
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        screen.query_one("#library-ingest-start").focus()
        expand = screen.query_one(".library-ingest-group-expand")
        expand_selector = f"#{expand.id}"
        await _tab_to(screen, host, pilot, expand_selector, "Show the 4 files")
        await pilot.press("enter")
        await _focus(screen, host, pilot, expand_selector, "Hide the 4 files")
        for job in jobs:
            assert screen.query(f"#library-ingest-details-{job.job_id}")
        await pilot.press("enter")
        await _focus(screen, host, pilot, expand_selector, "Show the 4 files")
        await pilot.press("tab")
        retry_selector = f"#{screen.query_one('.library-ingest-group-retry').id}"
        await _focus(screen, host, pilot, retry_selector, "Retry all")
        await pilot.press("enter")
        await _focus(screen, host, pilot, "#library-ingest-path", "notes.txt")
        assert sorted(retried) == sorted(job.job_id for job in jobs)
        replacements = registry.jobs()
        assert len(replacements) == 4
        assert {job.job_id for job in replacements}.isdisjoint(retried)
        for job in replacements:
            registry.mark_failed(job.job_id, error="Fixture failure")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query(".library-ingest-group-dismiss")),
            message="Failed group did not return",
        )
        dismiss_selector = f"#{screen.query_one('.library-ingest-group-dismiss').id}"
        await _tab_to(screen, host, pilot, dismiss_selector, "Dismiss all")
        await pilot.press("enter")
        await _focus(screen, host, pilot, "#library-ingest-path", "notes.txt")
        assert registry.jobs() == ()
        ledger = screen._ingest_state.recent_ledger
        assert {job.job_id for job in ledger} == {job.job_id for job in replacements}
        assert all(job.dismissed for job in ledger)

        finished = registry.submit(source_path=str(tmp_path / "clear-me.txt"))
        registry.mark_failed(finished.job_id, error="Another fixture failure")
        await _tab_to(screen, host, pilot, RECENT_TITLE, "Recent imports")
        await pilot.press("enter")
        assert not screen.query_one("#library-ingest-recent", Collapsible).collapsed
        await pilot.press("shift+tab")
        await _focus(
            screen, host, pilot, "#library-ingest-clear-finished", "Clear finished"
        )
        clear = screen.query_one("#library-ingest-clear-finished")
        await pilot.press("enter")
        await _focus(
            screen,
            host,
            pilot,
            "#library-ingest-clear-finished",
            "Press again to clear 1 finished",
        )
        assert screen.query_one("#library-ingest-clear-finished") is clear
        assert len(registry.jobs()) == 1
        # A deliberate second key press must also outlast Textual's press flash.
        await _wait_for_condition(
            pilot,
            lambda: not clear.has_class("-active"),
            message="Clear press flash did not end",
        )
        screen._ingest_state.clear_finished_armed_at -= 1.0
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-ingest-path"),
            message=lambda: (
                f"After clear: focus={screen.focused!r}; jobs={registry.jobs()!r}"
            ),
        )
        await _focus(screen, host, pilot, "#library-ingest-path", "notes.txt")
        assert registry.jobs() == ()
        assert not screen.query_one("#library-ingest-recent", Collapsible).collapsed
        assert {
            job.job_id for job in screen._build_library_ingest_state().recent_jobs
        } == {finished.job_id, *(job.job_id for job in replacements)}
        assert screen.query_one("#library-ingest-title", Input) is title
        assert title.value == "Unsaved next import"
        assert screen._ingest_state.form.path == str(source)
