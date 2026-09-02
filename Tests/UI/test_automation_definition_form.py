"""Tests for the recurring-question automation-definition create modal.

Modeled on ``Tests/UI/test_reminder_form.py`` for the harness pattern
(``App.run_test()`` -- ``AppTest`` is unavailable here). Most tests drive
a REAL ``SchedulingService`` over a real in-memory-file ``ScheduledTasksDB``
(local owner needs no server client at all, since Task 1's preview/Task 4's
save are pure/local for that owner) rather than a hand-rolled fake, per
the "unmocked-integration-test" lesson (accessor-mock fakes have hidden
protocol-signature mismatches in this codebase before). AsyncMock is used
only where a real server round trip needs to be simulated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from textual.app import App
from textual.widgets import Checkbox, Input, Select, Static, TextArea

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def local_service(db):
    return SchedulingService(db=db, runtime_source="local")


class _FormHost(App):
    """Hosts the modal under test and captures its dismiss result."""

    def __init__(self, service, **kwargs) -> None:
        super().__init__()
        self._service = service
        self._form_kwargs = kwargs
        self.result = "not-yet-dismissed"

    def on_mount(self) -> None:
        self.run_worker(self._drive)

    async def _drive(self) -> None:
        self.result = await self.push_screen_wait(
            AutomationDefinitionForm(self._service, **self._form_kwargs)
        )


async def _wait_for_form_worker(pilot) -> None:
    """Wait for the form's own preview/save worker to settle.

    Plain ``pilot.app.workers.wait_for_complete()`` (no filter) waits for
    EVERY worker, including ``_FormHost._drive`` -- that one only finishes
    when the modal dismisses, which is exactly what several of these tests
    are checking has NOT happened yet, so waiting on it unfiltered
    deadlocks. Filtering by group does not fix it either: `WorkerManager.
    wait_for_complete([])` falls back to "all workers" the moment the
    local preview/save worker (no real I/O -- it resolves within a
    handful of event-loop turns) has already finished and been pruned
    before this coroutine gets to filter for it, reproducing the same
    deadlock. A few plain pauses give the worker's task enough event-loop
    turns to run to completion and update the UI, without depending on a
    worker handle that may already be gone.
    """
    for _ in range(5):
        await pilot.pause()


def _future_run_at() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")


async def _fill_minimal_valid_form(screen) -> None:
    """Name + question + a valid future one-time run-at -- nothing else."""
    screen.query_one("#automation-name", Input).value = "Daily standup digest"
    screen.query_one("#automation-question", TextArea).text = "What shipped today?"
    screen.query_one("#automation-run-at", Input).value = _future_run_at()


# --- rendering ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_form_renders_expected_fields(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        for selector in (
            "#automation-runs-on",
            "#automation-name",
            "#automation-question",
            "#automation-scope-mode",
            "#automation-schedule-kind",
            "#automation-generation-mode",
            "#automation-finding-policy",
            "#automation-notify",
            "#automation-provider",
            "#automation-model",
            "#automation-preview-btn",
            "#automation-save",
            "#automation-cancel",
        ):
            assert screen.query_one(selector) is not None, selector

        # Default schedule kind is one-time -- the run-at group is visible,
        # the recurring cron group is not.
        assert screen.query_one("#automation-run-at-group").display
        assert not screen.query_one("#automation-cron-group").display


@pytest.mark.asyncio
async def test_scope_sources_checkboxes_hidden_until_sources_mode(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert not screen.query_one("#automation-scope-sources-group").display

        screen.query_one("#automation-scope-mode", Select).value = "sources"
        await pilot.pause()

        assert screen.query_one("#automation-scope-sources-group").display
        for widget_id in (
            "#automation-scope-media",
            "#automation-scope-notes",
            "#automation-scope-chats",
        ):
            assert screen.query_one(widget_id, Checkbox).value is True


@pytest.mark.asyncio
async def test_runs_on_select_offers_constructor_owners_and_defaults(local_service):
    app = _FormHost(
        local_service,
        available_owners=[("This device", "local"), ("Server (example.com)", "server:example.com")],
        default_owner="server:example.com",
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        select = app.screen.query_one("#automation-runs-on", Select)
        option_values = [value for _label, value in select._options]
        assert option_values == ["local", "server:example.com"]
        assert select.value == "server:example.com"


# --- payload building (unit-level, no worker/network involved) --------------


@pytest.mark.asyncio
async def test_build_payload_is_create_only_recurring_question(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        await pilot.pause()

        payload = screen._build_payload()
        assert payload["family"] == "recurring_question"
        assert payload["mode"] == "create"
        assert "definition_id" not in payload
        assert "definition_version" not in payload
        assert payload["name"] == "Daily standup digest"
        assert payload["input"]["question"] == "What shipped today?"
        assert payload["schedule"]["kind"] == "one_time"
        assert payload["config"]["scope"] == {"mode": "all_searchable_library"}
        assert payload["config"]["generation_mode"] == "optional"
        assert payload["config"]["finding_policy"] == {"preset": "balanced_findings"}
        assert payload["notification_policy"] == {"on_success": True, "on_failure": True}


@pytest.mark.asyncio
async def test_build_payload_sources_scope_only_includes_checked_boxes(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-scope-mode", Select).value = "sources"
        await pilot.pause()
        screen.query_one("#automation-scope-notes", Checkbox).value = False
        await pilot.pause()

        payload = screen._build_payload()
        assert payload["config"]["scope"] == {
            "mode": "sources",
            "sources": ["media_db", "chats"],
        }


@pytest.mark.asyncio
async def test_build_payload_optional_provider_model_pin(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-provider", Input).value = "openai"
        screen.query_one("#automation-model", Input).value = "gpt-5"
        await pilot.pause()

        payload = screen._build_payload()
        assert payload["input"]["provider"] == "openai"
        assert payload["input"]["model"] == "gpt-5"


# --- client-side schedule guard -----------------------------------------------


@pytest.mark.asyncio
async def test_blank_run_at_blocks_preview_without_calling_the_service(local_service):
    local_service.preview_definition = AsyncMock()
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#automation-name", Input).value = "No run at"
        screen.query_one("#automation-question", TextArea).text = "Question?"
        await pilot.pause()

        await pilot.click("#automation-preview-btn")
        await pilot.pause()

        local_service.preview_definition.assert_not_awaited()
        errors = screen.query_one("#automation-schedule-error", Static)
        assert errors.display
        assert "run at is required" in errors.visual.plain.lower()


# --- preview -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_maps_validation_errors_onto_their_fields(local_service):
    """A real local preview: blank name + blank question -> both field
    errors render under their own widgets, not merged into one blob."""
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#automation-run-at", Input).value = _future_run_at()
        await pilot.pause()

        await pilot.click("#automation-preview-btn")
        await pilot.pause()
        await _wait_for_form_worker(pilot)
        await pilot.pause()

        name_error = screen.query_one("#automation-name-error", Static)
        question_error = screen.query_one("#automation-question-error", Static)
        assert name_error.display and "name is required" in name_error.visual.plain.lower()
        assert (
            question_error.display
            and "question is required" in question_error.visual.plain.lower()
        )
        # No unmatched-field spillover for a fully-recognized error set.
        assert not screen.query_one("#automation-form-errors", Static).display


@pytest.mark.asyncio
async def test_preview_shows_next_occurrences_when_valid(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        await pilot.pause()

        await pilot.click("#automation-preview-btn")
        await pilot.pause()
        await _wait_for_form_worker(pilot)
        await pilot.pause()

        preview_text = screen.query_one("#automation-preview-text", Static)
        assert "Next runs:" in preview_text.visual.plain
        assert not screen.query_one("#automation-name-error", Static).display


@pytest.mark.asyncio
async def test_set_validation_errors_puts_unrecognized_fields_in_form_level_area(
    local_service,
):
    """Unit-level pin: an error whose `field` this form does not map
    renders in the form-level area, not silently dropped -- covers a
    server preview's own error vocabulary too (Task 4's report: not
    guaranteed byte-identical to the local port's `code`/`message`, but
    every error still carries a `field` to match on)."""
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._set_validation_errors(
            [
                {"field": "name", "message": "Name is required."},
                {
                    "field": "config.retention_policy.mode",
                    "message": "Unsupported retention policy mode: bogus",
                },
            ]
        )
        await pilot.pause()
        name_error = screen.query_one("#automation-name-error", Static)
        form_errors = screen.query_one("#automation-form-errors", Static)
        assert name_error.display
        assert "name is required" in name_error.visual.plain.lower()
        assert form_errors.display
        assert "retention_policy" in form_errors.visual.plain


# --- save ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_local_writes_the_definition_and_dismisses(local_service, db):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        await pilot.pause()

        await pilot.click("#automation-save")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert app.result is not None
    assert app.result.status == "saved"
    row = db.get_automation_definition(app.result.definition_id)
    assert row is not None
    assert row["name"] == "Daily standup digest"
    assert row["owner_id"] == "local"
    assert row["family"] == "recurring_question"


@pytest.mark.asyncio
async def test_save_invalid_shows_errors_and_stays_open(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#automation-run-at", Input).value = _future_run_at()
        await pilot.pause()

        await pilot.click("#automation-save")
        await pilot.pause()
        await _wait_for_form_worker(pilot)
        await pilot.pause()

        assert app.result == "not-yet-dismissed"
        assert isinstance(pilot.app.screen, AutomationDefinitionForm)
        name_error = screen.query_one("#automation-name-error", Static)
        assert name_error.display


@pytest.mark.asyncio
async def test_save_server_owner_offline_queues_and_reports_queued(db):
    """A server owner whose seam is unreachable still writes the local row
    and reports "queued" honestly (Task 4's save_definition contract) --
    same offline-fallback shape `create_reminder` uses."""
    server_client = AsyncMock()
    server_client.preview_automation_definition.side_effect = ServerUnavailableError(
        "offline"
    )
    service = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    app = _FormHost(
        service,
        available_owners=[("Server (1)", "server:1")],
        default_owner="server:1",
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        await pilot.pause()

        await pilot.click("#automation-save")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert app.result is not None
    assert app.result.status == "queued"
    row = db.get_automation_definition(app.result.definition_id)
    assert row is not None
    assert row["owner_id"] == "server:1"
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"


# --- discard guard -------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_without_edits_dismisses_immediately(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.click("#automation-cancel")
        await pilot.pause()

    assert app.result is None


@pytest.mark.asyncio
async def test_cancel_with_edits_asks_for_confirmation(local_service):
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#automation-name", Input).value = "Something typed"
        await pilot.pause()

        await pilot.click("#automation-cancel")
        await pilot.pause()

        assert isinstance(pilot.app.screen, ConfirmationDialog)
