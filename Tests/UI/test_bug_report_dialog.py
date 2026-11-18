"""The reviewed text is the only clipboard candidate (ADR-358)."""

import asyncio
from collections import deque
import threading

import pytest
from textual.widgets import Button, Checkbox, TextArea
from textual.binding import Binding

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity
from tldw_chatbook.Utils.bug_report import BugReportEvidence, BugReportItem
from tldw_chatbook.UI.bug_report_capture import (
    BugReportCapture,
    capture_bug_report_context,
)
import tldw_chatbook.Widgets.bug_report_dialog as report_ui
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.confirmation_dialog import (
    confirm_quit_screens,
    quit_confirmation_screens,
)
from tldw_chatbook.app_bug_report import BugReportActions

pytestmark = pytest.mark.bootstrap_profile


class ReportHarness(BugReportActions, ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    BINDINGS = [Binding("ctrl+q", "quit_walk", "Quit", priority=True)]

    def __init__(self):
        super().__init__()
        self.console_runtime = None
        self.current_tab = "logs"
        self.payloads = []
        self.identity = RuntimeIdentity("local")
        self._log_records = deque()
        self.quit_answers = []

    def action_quit_walk(self):
        async def walk():
            self.quit_answers.append(
                await confirm_quit_screens(quit_confirmation_screens(self))
            )

        self.run_worker(walk(), group="quit-walk")

    def _current_runtime_identity(self):
        return self.identity

    def compose(self):
        yield Button("Opener", id="opener")

    def copy_to_clipboard(self, text):
        self.payloads.append(text)


@pytest.fixture
def capture(monkeypatch):
    async def collect(collector):
        return BugReportCapture(
            collector.context,
            BugReportEvidence(
                logs=(BugReportItem("logs", "captured exception"),),
                inventory=("Application-wide session logs; frozen test excerpt.",),
            ),
        )

    monkeypatch.setattr(report_ui.BugReportCollector, "collect", collect)


async def open_dialog(app):
    dialog = report_ui.BugReportDialog(app, capture_bug_report_context(app))
    await asyncio.wait_for(app.push_screen(dialog), 10)
    await asyncio.wait_for(app.workers.wait_for_complete(), 10)
    # Textual suppresses a second click during its 200ms active animation.
    # Exercise report serialization without making test pacing depend on that timer.
    for button in dialog.query(Button):
        button.active_effect_duration = 0
    return dialog


async def generate(app, dialog, pilot):
    dialog.query_one("#bug-report-description", TextArea).load_text("Observed failure")
    await pilot.pause()
    await pilot.click("#bug-report-generate")
    await app.workers.wait_for_complete()
    await pilot.pause()
    return dialog.query_one("#bug-report-review", TextArea)


async def answer(pilot, selector):
    """Wait for the pushed prompt's layout and the resulting screen pop."""
    await pilot.pause()
    await pilot.click(selector)
    await pilot.pause()


@pytest.mark.parametrize("size", [(60, 24), (120, 40)])
async def test_initial_focus_defaults_and_keyboard_reachable_actions(capture, size):
    app = ReportHarness()
    async with app.run_test(size=size) as pilot:
        dialog = await open_dialog(app)
        assert dialog.focused.id == "bug-report-description"
        assert dialog.query_one("#bug-report-tools", Checkbox).value is False
        assert dialog.query_one("#bug-report-logs", Checkbox).value is True
        assert dialog.query_one("#bug-report-trace", Checkbox).value is True
        await pilot.click("#bug-report-generate")
        await app.workers.wait_for_complete()
        assert dialog.focused.id == "bug-report-description"
        assert (
            "description"
            in str(dialog.query_one("#bug-report-status").render()).lower()
        )
        await pilot.press("tab", "enter")
        reached_fields = set()
        for _ in range(16):
            await pilot.press("tab")
            if app.focused:
                reached_fields.add(app.focused.id)
        assert {
            "bug-report-steps",
            "bug-report-expected",
            "bug-report-actual",
            "bug-report-logs",
            "bug-report-trace",
        } <= reached_fields
        review = await generate(app, dialog, pilot)
        assert "Observed failure" in review.text and not review.read_only
        reached = set()
        for _ in range(8):
            await pilot.press("tab")
            if app.focused:
                reached.add(app.focused.id)
        assert {"bug-report-back", "bug-report-copy", "bug-report-close"} <= reached
        for name in ("copy", "close"):
            button = dialog.query_one(f"#bug-report-{name}", Button)
            assert 0 <= button.region.y < size[1]
            assert button.region.right <= size[0]


async def test_copy_redacts_edits_requires_second_press_and_requests_client_copy(
    capture,
):
    app = ReportHarness()
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        review = await generate(app, dialog, pilot)
        review.load_text("diagnosis api_key=synthetic-private-key")
        await pilot.pause()
        await pilot.click("#bug-report-copy")
        await app.workers.wait_for_complete()
        assert app.payloads == [] and "synthetic-private-key" not in review.text
        await pilot.click("#bug-report-copy")
        await app.workers.wait_for_complete()
        assert app.payloads == [review.text], str(
            dialog.query_one("#bug-report-status").render()
        )
        assert "Copy requested" in str(dialog.query_one("#bug-report-status").render())


async def test_oversize_edits_and_clipboard_failure_preserve_selectable_review(capture):
    app = ReportHarness()
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        review = await generate(app, dialog, pilot)
        oversized = "界" * 20001
        review.load_text(oversized)
        await pilot.pause()
        await pilot.click("#bug-report-copy")
        await app.workers.wait_for_complete()
        assert review.text == oversized and app.payloads == []
        assert "60,003" in str(dialog.query_one("#bug-report-status").render())
        review.load_text("selectable diagnosis")

        def unavailable(_text):
            raise OSError("PRIVATE_DRIVER_ERROR")

        app.copy_to_clipboard = unavailable
        await pilot.pause()
        await pilot.click("#bug-report-copy")
        await app.workers.wait_for_complete()
        assert review.text == "selectable diagnosis"
        assert "select" in str(dialog.query_one("#bug-report-status").render()).lower()
        assert "PRIVATE_DRIVER_ERROR" not in str(
            dialog.query_one("#bug-report-status").render()
        )


async def test_regeneration_confirms_before_replacing_manual_review_edits(capture):
    app = ReportHarness()
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        review = await generate(app, dialog, pilot)
        review.load_text("KEEP_MY_EDIT")
        await pilot.pause()
        await pilot.click("#bug-report-back")
        await pilot.click("#bug-report-generate")
        assert isinstance(app.screen, ConfirmationDialog)
        await answer(pilot, "#cancel-button")
        assert review.text == "KEEP_MY_EDIT"
        await pilot.click("#bug-report-generate")
        await answer(pilot, "#confirm-button")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert "Observed failure" in review.text and "KEEP_MY_EDIT" not in review.text


@pytest.mark.parametrize("close", ["escape", "button", "backdrop"])
async def test_dirty_close_prompts_and_restores_opener_focus(capture, close):
    app = ReportHarness()
    async with app.run_test() as pilot:
        app.query_one("#opener").focus()
        dialog = await open_dialog(app)
        dialog.query_one("#bug-report-description", TextArea).load_text(
            "unsaved diagnosis"
        )
        await pilot.pause()
        if close == "escape":
            await pilot.press("escape")
        elif close == "button":
            await pilot.click("#bug-report-close")
        else:
            await pilot.click(offset=(0, 0))
        assert isinstance(app.screen, ConfirmationDialog)
        await answer(pilot, "#cancel-button")
        assert app.screen is dialog
        await pilot.press("escape")
        await answer(pilot, "#confirm-button")
        await pilot.pause()
        assert dialog not in app.screen_stack
        assert app.focused.id == "opener"


async def test_edit_during_copy_check_and_repeated_copy_never_deliver_stale_text(
    capture, monkeypatch
):
    app = ReportHarness()
    entered, release = threading.Event(), threading.Event()
    original = report_ui.prepare_bug_report_copy
    calls = []

    def held(text):
        calls.append(text)
        entered.set()
        release.wait(3)
        return original(text)

    monkeypatch.setattr(report_ui, "prepare_bug_report_copy", held)
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        review = await generate(app, dialog, pilot)
        await pilot.click("#bug-report-copy")
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await pilot.click("#bug-report-copy")
            review.load_text("newer edit")
            await pilot.pause()
        finally:
            release.set()
        await app.workers.wait_for_complete()
        assert len(calls) == 1 and app.payloads == []
        assert review.text == "newer edit"


async def test_ctrl_q_uses_real_quit_walk_and_keeps_dirty_report(capture):
    app = ReportHarness()
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        dialog.query_one("#bug-report-description", TextArea).load_text("keep report")
        await pilot.pause()
        await pilot.press("ctrl+q")
        assert isinstance(app.screen, ConfirmationDialog)
        await answer(pilot, "#cancel-button")
        await app.workers.wait_for_complete()
        assert app.quit_answers == [False] and app.screen is dialog
        await pilot.press("ctrl+q")
        await answer(pilot, "#confirm-button")
        await app.workers.wait_for_complete()
        assert app.quit_answers == [False, True]


async def test_authored_field_budget_is_validated_at_description(capture):
    app = ReportHarness()
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        description = dialog.query_one("#bug-report-description", TextArea)
        description.load_text("界" * 2667)
        await pilot.pause()
        await pilot.click("#bug-report-generate")
        await app.workers.wait_for_complete()
        assert dialog.focused is description
        assert dialog.query_one("#bug-report-review", TextArea).text == ""


@pytest.mark.parametrize("change", ["description", "runtime"])
async def test_generation_preserves_form_when_inputs_or_runtime_change(
    capture, monkeypatch, change
):
    app = ReportHarness()
    entered, release = threading.Event(), threading.Event()
    original = report_ui.render_bug_report

    def held(*args):
        entered.set()
        release.wait(4)
        return original(*args)

    monkeypatch.setattr(report_ui, "render_bug_report", held)
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        description = dialog.query_one("#bug-report-description", TextArea)
        description.load_text("original description")
        await pilot.pause()
        await pilot.click("#bug-report-generate")
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            if change == "description":
                description.load_text("newer description")
            else:
                app.identity = RuntimeIdentity("server", "different-server")
            await pilot.pause()
        finally:
            release.set()
        await app.workers.wait_for_complete()
        assert dialog.query_one("#bug-report-review", TextArea).text == ""
        assert description.text == (
            "newer description" if change == "description" else "original description"
        )


@pytest.mark.parametrize("interruption", ["close", "runtime"])
async def test_pending_copy_cannot_deliver_after_close_or_runtime_change(
    capture, monkeypatch, interruption
):
    app = ReportHarness()
    entered, release = threading.Event(), threading.Event()
    original = report_ui.prepare_bug_report_copy

    def held(text):
        entered.set()
        release.wait(4)
        return original(text)

    monkeypatch.setattr(report_ui, "prepare_bug_report_copy", held)
    async with app.run_test() as pilot:
        dialog = await open_dialog(app)
        await generate(app, dialog, pilot)
        await pilot.click("#bug-report-copy")
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            if interruption == "close":
                await pilot.click("#bug-report-close")
                await answer(pilot, "#confirm-button")
                await pilot.pause()
                assert dialog not in app.screen_stack
            else:
                app.identity = RuntimeIdentity("server", "different-server")
        finally:
            release.set()
        await app.workers.wait_for_complete()
        assert app.payloads == []


async def test_close_during_collection_rejects_late_delivery_and_reopen_is_independent(
    monkeypatch,
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(collector):
        entered.set()
        await release.wait()
        return BugReportCapture(collector.context, BugReportEvidence())

    monkeypatch.setattr(report_ui.BugReportCollector, "collect", held)
    app = ReportHarness()
    async with app.run_test() as pilot:
        first = report_ui.BugReportDialog(app, capture_bug_report_context(app))
        await app.push_screen(first)
        await asyncio.wait_for(entered.wait(), 2)
        await pilot.press("escape")
        await pilot.pause()
        assert first not in app.screen_stack
        release.set()
        second = await open_dialog(app)
        assert app.screen is second and first._collector is not second._collector
