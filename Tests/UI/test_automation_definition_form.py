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
    """Name + question + a valid future one-time run-at -- nothing else.

    31712 AC#2 flipped create mode's default Schedule Kind to Recurring,
    so a helper that means "one-time schedule" now sets that explicitly
    rather than relying on the form's own default.
    """
    screen.query_one("#automation-schedule-kind", Select).value = "one_time"
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

        # 31712 AC#2: create mode defaults Schedule Kind to Recurring (the
        # form's whole purpose) -- the cron group is visible, the one-time
        # run-at group is not.
        assert screen.query_one("#automation-schedule-kind", Select).value == "recurring"
        assert not screen.query_one("#automation-run-at-group").display
        assert screen.query_one("#automation-cron-group").display


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


@pytest.mark.asyncio
async def test_build_payload_emits_blank_provider_model_as_explicit_null(local_service):
    """Final review I4: `save_definition` merges an edit payload onto the
    stored row, where an OMITTED key keeps its stored value -- so the two
    fields this form DOES expose must always be emitted, or clearing them
    would silently resurrect the old provider/model."""
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        await pilot.pause()

        payload = screen._build_payload()
        assert payload["input"]["provider"] is None
        assert payload["input"]["model"] is None


# --- client-side schedule guard -----------------------------------------------


@pytest.mark.asyncio
async def test_blank_run_at_blocks_preview_without_calling_the_service(local_service):
    local_service.preview_definition = AsyncMock()
    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        # 31712 AC#2 flipped the create-mode default to Recurring (whose
        # cron fields are pre-populated and therefore always valid) --
        # switch to One Time explicitly so a blank run-at is the thing
        # under test.
        screen.query_one("#automation-schedule-kind", Select).value = "one_time"
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


# --- task-5 fix round: edit mode ------------------------------------------------


@pytest.mark.asyncio
async def test_edit_mode_prefills_all_major_fields_and_disables_runs_on(
    local_service, db
):
    """Create a real local definition via the facade, then reopen it in
    edit mode and confirm every field reverse-maps correctly -- including
    the cron schedule via `cron_to_preset` (the module-docstring fix)."""
    create_payload = {
        "family": "recurring_question",
        "mode": "create",
        "name": "Weekly digest",
        "input": {"question": "What shipped this week?", "provider": "openai", "model": "gpt-5"},
        "schedule": {"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
        "config": {
            "scope": {"mode": "sources", "sources": ["notes", "chats"]},
            "generation_mode": "required",
            "finding_policy": {"preset": "high_confidence_only"},
        },
        "notification_policy": {"on_success": True, "on_failure": False},
    }
    outcome = await local_service.save_definition(create_payload, "local")
    assert outcome.status == "saved"
    row = db.get_automation_definition(outcome.definition_id)
    assert row is not None

    app = _FormHost(
        local_service,
        definition_row=row,
        definition_id=row["id"],
        available_owners=[("This device", "local")],
        default_owner="local",
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen

        assert screen.query_one("#automation-name", Input).value == "Weekly digest"
        assert (
            screen.query_one("#automation-question", TextArea).text
            == "What shipped this week?"
        )
        assert screen.query_one("#automation-provider", Input).value == "openai"
        assert screen.query_one("#automation-model", Input).value == "gpt-5"

        assert screen.query_one("#automation-scope-mode", Select).value == "sources"
        assert screen.query_one("#automation-scope-notes", Checkbox).value is True
        assert screen.query_one("#automation-scope-chats", Checkbox).value is True
        assert screen.query_one("#automation-scope-media", Checkbox).value is False

        assert (
            screen.query_one("#automation-generation-mode", Select).value == "required"
        )
        assert (
            screen.query_one("#automation-finding-policy", Select).value
            == "high_confidence_only"
        )
        assert screen.query_one("#automation-notify", Checkbox).value is True

        # Schedule: cron_to_preset maps "0 9 * * 1" -> ("monday", "09:00").
        assert (
            screen.query_one("#automation-schedule-kind", Select).value == "recurring"
        )
        assert screen.query_one("#automation-cron-preset", Select).value == "monday"
        assert screen.query_one("#automation-preset-time", Input).value == "09:00"
        assert screen.query_one("#automation-cron", Input).value == "0 9 * * 1"
        assert screen.query_one("#automation-timezone", Select).value == "UTC"

        runs_on = screen.query_one("#automation-runs-on", Select)
        assert runs_on.disabled
        assert runs_on.value == "local"

        assert "Edit Recurring Question" in str(screen.query_one(".form-title").render())


@pytest.mark.asyncio
async def test_edit_mode_unrecognized_schedule_falls_back_to_create_defaults(
    local_service, db
):
    """A schedule shape this form cannot itself produce (e.g. `interval`)
    must never crash prefill -- it is left at the one-time/blank default."""
    outcome = await local_service.save_definition(
        {
            "family": "recurring_question",
            "name": "Interval one",
            "input": {"question": "Q?"},
            "schedule": {"kind": "interval", "every_seconds": 3600},
        },
        "local",
    )
    assert outcome.status == "saved"
    row = db.get_automation_definition(outcome.definition_id)

    app = _FormHost(local_service, definition_row=row, definition_id=row["id"])
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert (
            screen.query_one("#automation-schedule-kind", Select).value == "one_time"
        )
        assert screen.query_one("#automation-run-at", Input).value == ""


@pytest.mark.asyncio
async def test_edit_mode_save_updates_the_existing_row_via_definition_id(
    local_service, db
):
    outcome = await local_service.save_definition(
        {
            "family": "recurring_question",
            "name": "Original name",
            "input": {"question": "Original question?"},
            "schedule": {"kind": "one_time", "run_at": _future_run_at()},
        },
        "local",
    )
    assert outcome.status == "saved"
    definition_id = outcome.definition_id
    row = db.get_automation_definition(definition_id)
    assert row["version"] == 1

    app = _FormHost(local_service, definition_row=row, definition_id=definition_id)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.query_one("#automation-name", Input).value = "Renamed"
        await pilot.pause()

        await pilot.click("#automation-save")
        await pilot.pause()
        await _wait_for_form_worker(pilot)

    assert app.result is not None
    assert app.result.status == "saved"
    assert app.result.definition_id == definition_id
    updated_row = db.get_automation_definition(definition_id)
    assert updated_row["name"] == "Renamed"
    assert updated_row["version"] == 2
    # No second row was created.
    assert len(db.list_automation_definitions(owner_id="local")) == 1


@pytest.mark.asyncio
async def test_edit_mode_payload_targets_server_definition_id_when_mirrored(
    local_service, db
):
    """A server-mirrored row's preview payload must reference the SERVER's
    definition id -- the local id means nothing to the server preview seam.
    Local rows (no server_id) keep the local id."""
    create_payload = {
        "family": "recurring_question",
        "mode": "create",
        "name": "Mirrored digest",
        "input": {"question": "Anything new?"},
        "schedule": {"kind": "interval", "interval_minutes": 60},
        "config": {"scope": {"mode": "sources", "sources": ["notes"]}},
        "notification_policy": {"on_success": True, "on_failure": True},
    }
    outcome = await local_service.save_definition(create_payload, "local")
    assert outcome.status == "saved"
    row = dict(db.get_automation_definition(outcome.definition_id))
    row["server_id"] = "srv-def-42"

    app = _FormHost(
        local_service,
        definition_row=row,
        definition_id=row["id"],
        available_owners=[("This device", "local")],
        default_owner="local",
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        payload = app.screen._build_payload()
        assert payload["definition_id"] == "srv-def-42"
        assert payload["definition_version"] == row["version"]


@pytest.mark.parametrize("stored_zone", ["Pacific/Apia", "Mars/Olympus_Mons"])
@pytest.mark.asyncio
async def test_edit_mode_prefills_a_non_curated_stored_timezone(
    local_service, db, stored_zone
):
    """Qodo HIGH: a definition saved with a valid-but-non-curated zone
    (`Pacific/Apia`) assigned a Select value outside its own options and
    raised `InvalidSelectValueError`, taking the whole edit modal down.

    Mirrors `ReminderForm._timezone_options` (review F4): the row's own zone
    is always offered, and a zone that does not even resolve locally is
    offered with an honest label so an unrelated edit round-trips it rather
    than silently rewriting it to the system zone."""
    outcome = await local_service.save_definition(
        {
            "family": "recurring_question",
            "mode": "create",
            "name": "Apia digest",
            "input": {"question": "What shipped?"},
            "schedule": {"kind": "cron", "cron": "0 9 * * 1", "timezone": "UTC"},
            "config": {},
        },
        "local",
    )
    assert outcome.status == "saved"
    # Write the awkward zone straight onto the row: the normalizer may or
    # may not accept it as authoring input, but the row can already hold it
    # (a server mirror or an older client wrote it), and prefill must cope.
    stored = db.get_automation_definition(outcome.definition_id)
    stored["schedule"] = {**stored["schedule"], "timezone": stored_zone}

    app = _FormHost(
        local_service, definition_row=stored, definition_id=stored["id"]
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        tz_select = app.screen.query_one("#automation-timezone", Select)
        assert tz_select.value == stored_zone
        assert stored_zone in [value for _label, value in tz_select._options]


@pytest.mark.asyncio
async def test_preview_renders_junk_occurrences_instead_of_crashing(local_service):
    """Qodo MEDIUM: `next_occurrences` crosses the network boundary from the
    server preview, so entries are not guaranteed to be ISO-8601 strings --
    or strings at all. `datetime.fromisoformat` raises `TypeError` (not
    `ValueError`) on a non-string, which took the whole preview render down.
    """
    from tldw_chatbook.Scheduling.models import AutomationFamily, AutomationPreview
    from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
        _format_occurrence,
    )

    assert _format_occurrence("not-a-date") == "not-a-date"
    assert _format_occurrence(12345) == "12345"
    assert _format_occurrence(None) == "None"

    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._render_preview(
            AutomationPreview(
                mode="create",
                family=AutomationFamily.RECURRING_QUESTION,
                status="valid",
                schedule_preview={
                    "next_occurrences": [
                        "2099-01-01T09:00:00+00:00",
                        {"nope": 1},
                        None,
                    ]
                },
            )
        )
        await pilot.pause()
        rendered = str(screen.query_one("#automation-preview-text", Static).render())
        assert "Next runs:" in rendered
        assert "{'nope': 1}" in rendered

        # A non-list `next_occurrences` must not be sliced/iterated either.
        screen._render_preview(
            AutomationPreview(
                mode="create",
                family=AutomationFamily.RECURRING_QUESTION,
                status="valid",
                schedule_preview={"next_occurrences": "junk"},
            )
        )
        await pilot.pause()
        assert "Valid." in str(
            screen.query_one("#automation-preview-text", Static).render()
        )


@pytest.mark.asyncio
async def test_edit_mode_prefills_a_server_shaped_cron_schedule(local_service):
    """Final review F1 carry-forward: the wire's `schedule.expression`.

    `_prefill_from_row` read `schedule["cron"]` only -- the key THIS
    client's own writer emits. A definition authored on the server sends
    `expression` instead, so editing a mirrored server-only definition
    fell through to the form's one-time/blank default, and saving would
    then write that default OVER the server's real schedule. That is a
    silent overwrite, not a cosmetic gap, which is why it rides this PR.

    Fed the repo's own RECORDED server payload rather than a hand-written
    dict -- client-shaped fixtures are exactly what hid this.
    """
    import json
    from pathlib import Path

    row = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "Scheduling/fixtures/server_responses/automation_definition_list.json"
        ).read_text()
    )["items"][0]
    # Assert the PREMISE: if the recorded payload ever grows a `cron` key,
    # this test silently stops testing what it claims to.
    assert row["schedule"] == {
        "kind": "cron",
        "expression": "0 9 * * 1-5",
        "timezone": "UTC",
    }
    assert "cron" not in row["schedule"]

    app = _FormHost(
        local_service,
        definition_row=row,
        definition_id=row["id"],
        available_owners=[("This device", "local")],
        default_owner="local",
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen

        assert (
            screen.query_one("#automation-schedule-kind", Select).value == "recurring"
        ), "a server-shaped cron schedule must not land on the one-time default"
        assert screen.query_one("#automation-cron", Input).value == "0 9 * * 1-5"
        assert screen.query_one("#automation-timezone", Select).value == "UTC"
