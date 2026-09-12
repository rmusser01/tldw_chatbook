"""Mounted bounded run-log paging contracts."""

import gc
import threading
import weakref
from typing import ClassVar

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Agents.run_log_format import RunLogRecord
from tldw_chatbook.Agents.run_log_paging import (
    RunLogPage,
    RunLogPageCursor,
    RunLogRecordSlice,
)
from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal


def page(index, text, *, more=True):
    record = RunLogRecord(index, "run", "tool", "result", "", text)
    return RunLogPage(
        (RunLogRecordSlice(record, 0, len(text), True),),
        RunLogPageCursor(0, index),
        RunLogPageCursor(0, index + 1) if more else None,
        len(text),
    )


class Host(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list] = list(APP_STYLESHEETS)


async def settled(pilot, predicate):
    for _ in range(100):
        await pilot.pause(0.01)
        if predicate():
            return
    assert predicate()


@pytest.mark.asyncio
async def test_pages_replace_off_thread_reload_previous_and_keep_only_cursors():
    ui_thread = threading.get_ident()
    calls = []

    def loader(cursor):
        calls.append((cursor, threading.get_ident()))
        return page(
            cursor.record_offset if cursor else 0,
            "SECOND CANARY" if cursor and cursor.record_offset else "FIRST CANARY",
        )

    first = page(0, "FIRST CANARY")
    reference = weakref.ref(first)
    modal = ConsoleRunLogModal(run_id="run", first_page=first, page_loader=loader)
    del first
    host = Host()
    async with host.run_test(size=(100, 32)) as pilot:
        await host.push_screen(modal)
        await pilot.click("#console-run-log-next")
        await settled(pilot, lambda: "SECOND CANARY" in modal.query_one(TextArea).text)
        gc.collect()
        assert reference() is None
        assert "FIRST CANARY" not in modal.query_one(TextArea).text
        await pilot.click("#console-run-log-previous")
        await settled(pilot, lambda: "FIRST CANARY" in modal.query_one(TextArea).text)
        assert [c.record_offset for c, _ in calls] == [1, 0]
        assert all(t != ui_thread for _, t in calls)


@pytest.mark.asyncio
async def test_gated_navigation_single_admission_and_close_ignores_late_result():
    gate, entered = threading.Event(), threading.Event()
    calls = []

    def loader(cursor):
        calls.append(cursor)
        entered.set()
        gate.wait(5)
        return page(1, "LATE CANARY")

    modal = ConsoleRunLogModal(
        run_id="run", first_page=page(0, "FIRST"), page_loader=loader
    )
    host = Host()
    async with host.run_test(size=(80, 24)) as pilot:
        await host.push_screen(modal)
        await pilot.click("#console-run-log-next")
        await settled(pilot, entered.is_set)
        modal.query_one("#console-run-log-next", Button).press()
        await pilot.pause()
        assert len(calls) == 1
        assert not modal.query_one("#console-run-log-close", Button).disabled
        await pilot.click("#console-run-log-close")
        await pilot.pause()
        gate.set()
        await host.workers.wait_for_complete()
        assert modal not in host.screen_stack
        assert modal._page.slices[0].record.content == "FIRST"


@pytest.mark.asyncio
async def test_failure_keeps_last_page_and_empty_continuation_is_navigable():
    empty = RunLogPage((), RunLogPageCursor(0, 0), RunLogPageCursor(0, 1), 100)
    modal = ConsoleRunLogModal(
        run_id="run", first_page=empty, page_loader=lambda cursor: None
    )
    host = Host()
    async with host.run_test(size=(80, 24)) as pilot:
        await host.push_screen(modal)
        assert not modal.query_one("#console-run-log-next", Button).disabled
        assert "Continue" in str(
            modal.query_one("#console-run-log-status", Static).content
        )
        await pilot.click("#console-run-log-next")
        await settled(pilot, lambda: not modal._loading)
        assert modal._page is empty
        assert "no longer available" in str(
            modal.query_one("#console-run-log-status", Static).content
        )


@pytest.mark.asyncio
async def test_target_change_ignores_gated_page_publication():
    gate, entered = threading.Event(), threading.Event()
    current = [True]

    def loader(cursor):
        entered.set()
        gate.wait(3)
        return page(1, "WRONG TARGET")

    modal = ConsoleRunLogModal(
        run_id="run",
        first_page=page(0, "FIRST"),
        page_loader=loader,
        target_is_current=lambda: current[0],
    )
    host = Host()
    async with host.run_test(size=(80, 24)) as pilot:
        await host.push_screen(modal)
        await pilot.click("#console-run-log-next")
        await settled(pilot, entered.is_set)
        current[0] = False
        gate.set()
        await host.workers.wait_for_complete()
        assert "WRONG TARGET" not in modal.query_one(TextArea).text
        assert modal._page.slices[0].record.content == "FIRST"


@pytest.mark.asyncio
async def test_history_is_bounded_and_first_recovers_dropped_history():
    modal = ConsoleRunLogModal(
        run_id="run",
        first_page=page(0, "page-0"),
        page_loader=lambda cursor: page(
            cursor.record_offset if cursor else 0, "current-only"
        ),
    )
    host = Host()
    async with host.run_test(size=(80, 24)) as pilot:
        await host.push_screen(modal)
        for _ in range(260):
            modal.query_one("#console-run-log-next", Button).press()
            await settled(pilot, lambda: not modal._loading)
        assert len(modal._history) == 256
        assert all(isinstance(cursor, RunLogPageCursor) for cursor in modal._history)
        assert modal._history[0].record_offset == 4
        await pilot.click("#console-run-log-first")
        await settled(pilot, lambda: modal._page_number == 1)
        assert not modal._history
        assert modal._page.start_cursor.record_offset == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 48), (80, 24)])
async def test_painted_pager_controls_fit_supported_sizes(size, tmp_path):
    from pathlib import Path

    modal = ConsoleRunLogModal(
        run_id="run-2026-09-12",
        first_page=page(0, "Stored tool output\n" * 24),
        page_loader=lambda cursor: page(1, "Second stored page", more=False),
    )
    host = Host()
    async with host.run_test(size=size) as pilot:
        await host.push_screen(modal)
        await pilot.pause()
        content = modal.query_one("#console-run-log-modal").content_region
        for button in modal.query(Button):
            assert content.contains_region(button.region), (
                button.id,
                button.region,
                content,
            )
        assert modal.query_one(TextArea).region.height >= 5
        modal_region = modal.query_one("#console-run-log-modal").region
        textarea_region = modal.query_one(TextArea).region
        assert modal_region.contains_region(textarea_region)
        strips = modal._compositor.render_strips()
        for strip in strips[modal_region.y : modal_region.bottom]:
            assert not strip.crop(modal_region.right, size[0]).text.strip()
            assert not strip.crop(0, modal_region.x).text.strip()
        import os

        output = Path(os.environ.get("RUN_LOG_QA_OUTPUT", str(tmp_path)))
        output.mkdir(parents=True, exist_ok=True)
        (output / f"pager-{size[0]}-cells.txt").write_text(
            f"modal={modal_region} textarea={textarea_region}\n"
            + "\n".join(strip.text for strip in strips)
        )
        svg = host.export_screenshot()
        (output / f"pager-{size[0]}.svg").write_text(svg)
        if os.environ.get("RUN_LOG_PAINT_QA"):
            import cairosvg

            cairosvg.svg2png(
                bytestring=svg.encode(), write_to=str(output / f"pager-{size[0]}.png")
            )
