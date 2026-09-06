"""Tests for the reminder create/edit form."""

from datetime import datetime, timezone

import pytest
from textual.app import App
from textual.widgets import Input, Select

from tldw_chatbook.Scheduling.events import ReminderFormSubmitted
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import (
    ReminderForm,
    cron_to_preset,
    parse_forgiving_datetime,
    preset_to_cron,
    system_timezone_name,
)


# --- task-23102: forgiving datetime parsing (pure unit tests) --------------


class TestParseForgivingDatetime:
    def test_local_form_without_offset_is_interpreted_in_system_zone(self):
        parsed, assumed_local = parse_forgiving_datetime("2026-08-28 09:00")
        assert parsed is not None
        assert assumed_local is True
        assert parsed.tzinfo is not None
        assert (parsed.hour, parsed.minute) == (9, 0)
        # Interpreted as the machine's local wall clock.
        local = datetime(2026, 8, 28, 9, 0).astimezone()
        assert parsed.utcoffset() == local.utcoffset()

    def test_t_separator_accepted(self):
        parsed, assumed_local = parse_forgiving_datetime("2026-08-28T09:00")
        assert parsed is not None
        assert assumed_local is True

    def test_single_digit_hour_accepted(self):
        parsed, _ = parse_forgiving_datetime("2026-08-28 9:05")
        assert parsed is not None
        assert (parsed.hour, parsed.minute) == (9, 5)

    def test_full_iso_with_offset_keeps_offset(self):
        parsed, assumed_local = parse_forgiving_datetime("2030-07-20T14:00:00+00:00")
        assert parsed == datetime(2030, 7, 20, 14, 0, tzinfo=timezone.utc)
        assert assumed_local is False

    def test_garbage_returns_none(self):
        parsed, _ = parse_forgiving_datetime("not-a-datetime")
        assert parsed is None

    def test_blank_returns_none(self):
        parsed, _ = parse_forgiving_datetime("   ")
        assert parsed is None


# --- task-23102: preset -> cron generation (pure unit tests) ---------------


class TestPresetCron:
    def test_daily(self):
        assert preset_to_cron("daily", "09:00") == "0 9 * * *"

    def test_weekday(self):
        assert preset_to_cron("weekday", "07:30") == "30 7 * * 1-5"

    def test_monday(self):
        assert preset_to_cron("monday", "18:05") == "5 18 * * 1"

    def test_hourly_ignores_time(self):
        assert preset_to_cron("hourly", "") == "0 * * * *"

    def test_single_digit_hour_accepted(self):
        assert preset_to_cron("daily", "9:00") == "0 9 * * *"

    def test_invalid_time_returns_none(self):
        assert preset_to_cron("daily", "25:00") is None
        assert preset_to_cron("daily", "nine") is None
        assert preset_to_cron("daily", "") is None

    def test_custom_returns_none(self):
        assert preset_to_cron("custom", "09:00") is None

    def test_cron_to_preset_roundtrip(self):
        assert cron_to_preset("0 9 * * *") == ("daily", "09:00")
        assert cron_to_preset("30 7 * * 1-5") == ("weekday", "07:30")
        assert cron_to_preset("5 18 * * 1") == ("monday", "18:05")
        assert cron_to_preset("0 * * * *") == ("hourly", "")

    def test_unrecognized_cron_maps_to_custom(self):
        assert cron_to_preset("*/5 * * * *") == ("custom", "")
        assert cron_to_preset("0 9 1 * *") == ("custom", "")


def test_system_timezone_name_is_a_valid_zone():
    from zoneinfo import ZoneInfo

    name = system_timezone_name()
    ZoneInfo(name)  # must not raise


def test_humanize_cron_covers_the_weekday_preset():
    """task-23102: the weekday preset's cron reads as prose, not raw cron."""
    from tldw_chatbook.UI.Screens.scheduling.task_detail import _humanize_cron

    assert _humanize_cron("30 7 * * 1-5", "UTC") == "Weekdays at 07:30 UTC"


def test_unicode_digit_cron_maps_to_custom_instead_of_crashing():
    """Review F14: '²'.isdigit() is True but int('²') raises -- a synced or
    DB-sourced cron with unicode digits must not crash the edit form."""
    assert cron_to_preset("² 9 * * *") == ("custom", "")
    assert cron_to_preset("0 ² * * *") == ("custom", "")


def test_unicode_digit_cron_does_not_crash_humanize():
    """Same digit-class bug in _humanize_cron (hit on every detail render)."""
    from tldw_chatbook.UI.Screens.scheduling.task_detail import _humanize_cron

    rendered = _humanize_cron("² 9 * * *", "UTC")  # must not raise
    assert "²" in rendered  # falls back to showing the raw expression


@pytest.mark.asyncio
async def test_editing_one_time_task_initializes_preset_fields():
    """Review F2 (live-verified): editing a one-time task (cron None) never
    initialized the preset sub-groups; Kind->Recurring showed BOTH the
    time-of-day and raw cron fields, and _save took the default preset,
    silently discarding a typed cron."""
    task = ReminderTask(
        id="task-ot",
        title="One-timer",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc),
    )
    app = FormTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(ReminderForm(task))
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-kind", Select).value = ScheduleKind.RECURRING.value
        await pilot.pause()

        # Preset fields initialized like create mode: daily preset, time
        # visible, raw cron hidden.
        assert str(form.query_one("#reminder-cron-preset", Select).value) == "daily"
        assert form.query_one("#reminder-preset-time-group").display
        assert not form.query_one("#reminder-cron-custom-group").display
        assert form.query_one("#reminder-preset-time", Input).value == "09:00"


@pytest.mark.asyncio
async def test_editing_one_time_task_to_custom_cron_saves_the_typed_cron():
    """Review F2: the typed custom cron must be saved, not the preset's."""
    task = ReminderTask(
        id="task-ot2",
        title="One-timer",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc),
    )
    app = FormTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(ReminderForm(task))
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-kind", Select).value = ScheduleKind.RECURRING.value
        await pilot.pause()
        form.query_one("#reminder-cron-preset", Select).value = "custom"
        await pilot.pause()
        form.query_one("#reminder-cron", Input).value = "*/5 * * * *"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        assert app.submitted["cron"] == "*/5 * * * *"


class FormTestApp(App):
    """Minimal app used to host the modal form under test."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: dict | None = None

    def on_reminder_form_submitted(self, event: ReminderFormSubmitted) -> None:
        self.submitted = event.form_data


@pytest.mark.asyncio
async def test_reminder_form_requires_title():
    """Clicking save with an empty title surfaces a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.click("#reminder-save")
        error_widget = pilot.app.screen.query_one("#reminder-errors")
        # Textual 8.2.8 does not expose Static.renderable; use visual.plain instead
        assert "title is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_submits_when_valid_one_time():
    """A valid one-time reminder dismisses the modal after posting the event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Water plants"
        run_at_input = pilot.app.screen.query_one("#reminder-run-at", Input)
        run_at_input.value = "2030-07-20T14:00:00+00:00"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is not None
        assert app.submitted["title"] == "Water plants"
        assert app.submitted["schedule_kind"] == "one_time"
        assert app.submitted["run_at"] == datetime(2030, 7, 20, 14, 0, tzinfo=timezone.utc)
        assert app.submitted["cron"] is None
        # task-31711 AC#2: a one-time reminder captures the machine's
        # detected zone (matching the recurring form's own Select
        # default) instead of storing `None`, which later round-tripped
        # through the DB and displayed back as a bare "UTC".
        assert app.submitted["timezone"] == system_timezone_name()
        assert app.submitted["timezone"] is not None


@pytest.mark.asyncio
async def test_reminder_form_submits_when_valid_recurring():
    """A valid recurring reminder dismisses the modal after posting the event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Weekly sync"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        preset = pilot.app.screen.query_one("#reminder-cron-preset", Select)
        preset.value = "custom"
        await pilot.pause()
        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = "0 9 * * 1"
        tz_select = pilot.app.screen.query_one("#reminder-timezone", Select)
        tz_select.value = "UTC"

        await pilot.click("#reminder-save")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is not None
        assert app.submitted["title"] == "Weekly sync"
        assert app.submitted["schedule_kind"] == "recurring"
        assert app.submitted["cron"] == "0 9 * * 1"
        assert app.submitted["timezone"] == "UTC"
        assert app.submitted["run_at"] is None


@pytest.mark.asyncio
async def test_reminder_form_requires_run_at_for_one_time():
    """A one-time reminder without a run_at value shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "No run_at"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "run at is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_rejects_invalid_run_at():
    """A one-time reminder with an unparseable run_at shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Bad run_at"
        run_at_input = pilot.app.screen.query_one("#reminder-run-at", Input)
        run_at_input.value = "not-a-datetime"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert (
            "run at must be a date and time"
            in error_widget.visual.plain.lower()
        )


@pytest.mark.asyncio
async def test_reminder_form_accepts_forgiving_local_datetime():
    """task-23102: '2099-08-28 09:00' (no offset) creates a one-time task."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-title", Input).value = "Local time task"
        form.query_one("#reminder-run-at", Input).value = "2099-08-28 09:00"
        await pilot.pause()

        # The live preview confirms the local interpretation before save.
        from textual.widgets import Static

        preview = str(
            form.query_one("#reminder-run-at-preview", Static).render()
        )
        assert "Runs:" in preview
        assert "local" in preview.lower()

        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        run_at = app.submitted["run_at"]
        assert run_at is not None and run_at.tzinfo is not None
        expected = datetime(2099, 8, 28, 9, 0).astimezone()
        assert run_at.utcoffset() == expected.utcoffset()
        assert (run_at.hour, run_at.minute) == (9, 0)


@pytest.mark.asyncio
async def test_weekday_preset_generates_cron_without_typing_cron():
    """task-23102: 'Every weekday at' + a time of day saves without raw cron."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-title", Input).value = "Standup"
        form.query_one("#reminder-kind", Select).value = ScheduleKind.RECURRING.value
        await pilot.pause()

        form.query_one("#reminder-cron-preset", Select).value = "weekday"
        await pilot.pause()
        form.query_one("#reminder-preset-time", Input).value = "07:30"
        await pilot.pause()

        # Raw cron stays behind "Custom cron..." -- hidden for presets.
        assert not form.query_one("#reminder-cron-custom-group").display

        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        assert app.submitted["cron"] == "30 7 * * 1-5"


@pytest.mark.asyncio
async def test_custom_preset_reveals_cron_with_live_preview():
    """task-23102: raw cron stays available behind Custom cron with preview."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-kind", Select).value = ScheduleKind.RECURRING.value
        await pilot.pause()
        form.query_one("#reminder-cron-preset", Select).value = "custom"
        await pilot.pause()

        assert form.query_one("#reminder-cron-custom-group").display
        form.query_one("#reminder-cron", Input).value = "0 9 * * 1"
        await pilot.pause()

        from textual.widgets import Static

        preview = str(form.query_one("#reminder-cron-preview", Static).render())
        assert "Weekly on Monday" in preview


@pytest.mark.asyncio
async def test_reminder_form_requires_cron_for_recurring():
    """A recurring reminder with a blank custom cron shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "No cron"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        # The create form pre-fills cron from the default preset; choosing a
        # custom frequency with a blank expression must still be caught.
        preset = pilot.app.screen.query_one("#reminder-cron-preset", Select)
        preset.value = "custom"
        await pilot.pause()
        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = ""

        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "cron expression is required" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_reminder_form_rejects_invalid_cron():
    """A recurring reminder with an invalid cron expression shows a validation error."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Bad cron"
        kind_select = pilot.app.screen.query_one("#reminder-kind", Select)
        kind_select.value = ScheduleKind.RECURRING.value
        await pilot.pause()

        preset = pilot.app.screen.query_one("#reminder-cron-preset", Select)
        preset.value = "custom"
        await pilot.pause()
        cron_input = pilot.app.screen.query_one("#reminder-cron", Input)
        cron_input.value = "not-a-cron"
        await pilot.click("#reminder-save")
        await pilot.pause()

        error_widget = pilot.app.screen.query_one("#reminder-errors")
        assert "cron expression is invalid" in error_widget.visual.plain.lower()


@pytest.mark.asyncio
async def test_timezone_is_a_select_defaulting_to_system_zone():
    """task-23102: the timezone field is a Select, not free text, and
    defaults to the system zone with known task zones included."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(
            ReminderForm(known_timezones=["Pacific/Auckland"])
        )
        await pilot.pause()
        form = pilot.app.screen
        tz_select = form.query_one("#reminder-timezone", Select)
        assert isinstance(tz_select, Select)
        assert tz_select.value == system_timezone_name()
        option_values = [value for _prompt, value in tz_select._options]
        assert "Pacific/Auckland" in option_values
        assert "UTC" in option_values


@pytest.mark.asyncio
async def test_timezone_select_refuses_unknown_zone():
    """task-23102: an arbitrary zone string cannot be selected at all."""
    from textual.widgets._select import InvalidSelectValueError

    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        tz_select = pilot.app.screen.query_one("#reminder-timezone", Select)
        with pytest.raises(InvalidSelectValueError):
            tz_select.value = "Mars/Phobos"


@pytest.mark.asyncio
async def test_editing_task_with_uncommon_zone_keeps_it_selectable():
    """task-23102: an existing task's zone joins the list when editing."""
    task = ReminderTask(
        id="task-tz",
        title="Uncommon zone",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 9 * * *",
        timezone="Pacific/Chatham",
    )
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm(task))
        await pilot.pause()
        tz_select = pilot.app.screen.query_one("#reminder-timezone", Select)
        assert tz_select.value == "Pacific/Chatham"


@pytest.mark.asyncio
async def test_unrecognized_stored_zone_round_trips_on_unrelated_edit():
    """Review F4 (live-verified): a stored zone that doesn't resolve in
    local tzdata was silently replaced by the system zone when the edit
    form opened, so ANY unrelated save rewrote the task's timezone and
    shifted its recurrence. The stored zone must stay selected (labeled
    as unrecognized) and survive an unrelated edit-save."""
    task = ReminderTask(
        id="task-tz2",
        title="Server zone",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 9 * * *",
        timezone="Mars/Phobos",  # not in local tzdata
    )
    app = FormTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(ReminderForm(task))
        await pilot.pause()
        form = pilot.app.screen
        tz_select = form.query_one("#reminder-timezone", Select)
        # Stored zone selected, offered as an explicitly labeled option.
        assert tz_select.value == "Mars/Phobos"
        labels = {str(prompt) for prompt, _value in tz_select._options}
        assert any(
            "Mars/Phobos" in label and "not recognized" in label
            for label in labels
        ), labels

        # An unrelated edit (title) round-trips the zone untouched.
        form.query_one("#reminder-title", Input).value = "Renamed"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        assert app.submitted["timezone"] == "Mars/Phobos"


@pytest.mark.asyncio
async def test_undetected_machine_zone_is_labeled_honestly(monkeypatch):
    """Review F7: when the machine zone cannot be detected the default is
    UTC -- the UI must say so instead of claiming it is the machine's."""
    import tldw_chatbook.UI.Screens.scheduling.forms.reminder_form as form_mod

    monkeypatch.setattr(form_mod, "detect_system_timezone", lambda: None)
    app = FormTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen
        tz_select = form.query_one("#reminder-timezone", Select)
        assert tz_select.value == "UTC"
        labels = {str(prompt) for prompt, _value in tz_select._options}
        assert any("not detected" in label for label in labels), labels

        from textual.widgets import Static

        helper = form.query_one("#reminder-timezone-helper", Static)
        assert "not detected" in str(helper.render()).lower()


@pytest.mark.asyncio
async def test_reminder_form_preserves_enabled_state_when_editing():
    """Editing a disabled reminder preserves its enabled state in the payload."""
    app = FormTestApp()
    disabled_task = ReminderTask(
        id="task-1",
        title="Existing",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc),
        enabled=False,
    )
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm(disabled_task))
        title_input = pilot.app.screen.query_one("#reminder-title", Input)
        title_input.value = "Updated"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        assert app.submitted["enabled"] is False


@pytest.mark.asyncio
async def test_reminder_form_cancel_dismisses_without_submitting():
    """Clicking Cancel dismisses the modal without emitting a submission event."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.click("#reminder-cancel")
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ReminderForm)
        assert app.submitted is None


# --- task-23100: the form must not clip fields at common terminal sizes ----
#
# The oracle for "is this widget actually visible" is the compositor
# (get_widget_at), not the widget's own cached region: a clipped widget
# still reports a plausible region (see lessons-live-verification.md).
# Shared with the other Schedules UI test files (task-23106 review F15).
from Tests.UI.schedules_test_helpers import (
    painted_at_own_center as _painted_at_own_center,
)


@pytest.mark.asyncio
async def test_recurring_fields_reachable_and_preview_pinned_at_80x24():
    """At 80x24 a focused cron field is painted and the live preview stays visible."""
    app = FormTestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-kind", Select).value = ScheduleKind.RECURRING.value
        await pilot.pause()
        form.query_one("#reminder-cron-preset", Select).value = "custom"
        await pilot.pause()

        cron_input = form.query_one("#reminder-cron", Input)
        cron_input.focus()
        await pilot.pause()

        # A focused widget must never be rendered invisible (AC).
        assert _painted_at_own_center(app, cron_input), (
            "the focused cron input is not painted at its own center -- it is "
            "clipped out of the modal"
        )

        # The live "Runs:" preview stays visible while the cron is edited (AC):
        # frame-level assertion, not a style probe (lessons-testing-evidence.md).
        frame = app.export_screenshot()
        assert "Runs:" in frame, "the live schedule preview is not on the frame"

        # Save/Cancel must be reachable too.
        assert _painted_at_own_center(app, form.query_one("#reminder-save"))


@pytest.mark.asyncio
async def test_all_recurring_fields_painted_at_235x52():
    """At 235x52 every recurring field, helper, and the preview is painted."""
    app = FormTestApp()
    async with app.run_test(size=(235, 52)) as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-kind", Select).value = ScheduleKind.RECURRING.value
        await pilot.pause()

        for selector in (
            "#reminder-title",
            "#reminder-body",
            "#reminder-kind",
            "#reminder-cron-preset",
            "#reminder-preset-time",
            "#reminder-timezone",
            "#reminder-save",
            "#reminder-cancel",
        ):
            widget = form.query_one(selector)
            assert _painted_at_own_center(app, widget), (
                f"{selector} is not painted at 235x52"
            )
        frame = app.export_screenshot()
        assert "Runs:" in frame

        # The raw cron field behind "Custom cron..." paints too.
        form.query_one("#reminder-cron-preset", Select).value = "custom"
        await pilot.pause()
        assert _painted_at_own_center(app, form.query_one("#reminder-cron"))


@pytest.mark.asyncio
async def test_one_time_fields_painted_at_80x24():
    """At 80x24 the one-time Run At field and the buttons are all painted."""
    app = FormTestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen

        run_at = form.query_one("#reminder-run-at", Input)
        run_at.focus()
        await pilot.pause()
        assert _painted_at_own_center(app, run_at)
        assert _painted_at_own_center(app, form.query_one("#reminder-save"))


@pytest.mark.asyncio
async def test_footer_survives_wrapped_error_lines_at_45x24():
    """Review F8: the old footer height budget counted one row per error
    line; at ~45 columns errors wrap and used to push Save off screen.
    The docked footer must keep Save painted with multi-line wrapped
    errors present."""
    app = FormTestApp()
    async with app.run_test(size=(45, 24)) as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = pilot.app.screen

        # Empty form -> two validation errors, each wrapping at 45 cols.
        await pilot.click("#reminder-save")
        await pilot.pause()
        errors = form.query_one("#reminder-errors")
        assert errors.display and errors.region.height >= 2

        save = form.query_one("#reminder-save")
        assert _painted_at_own_center(app, save), (
            "wrapped validation lines pushed Save out of the modal"
        )

        # And a focused field is still painted above the docked footer.
        run_at = form.query_one("#reminder-run-at", Input)
        run_at.focus()
        await pilot.pause()
        assert _painted_at_own_center(app, run_at)


# --- task-5: "Runs on" owner selector --------------------------------------


@pytest.mark.asyncio
async def test_runs_on_defaults_to_the_given_default_owner_when_creating():
    """Create mode: the selector is enabled and starts on `default_owner`."""
    app = FormTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(
            ReminderForm(
                available_owners=[
                    ("This device", "local"),
                    ("Server (example.com)", "server:example.com"),
                ],
                default_owner="server:example.com",
            )
        )
        await pilot.pause()
        runs_on = pilot.app.screen.query_one("#reminder-runs-on", Select)
        assert not runs_on.disabled
        assert runs_on.value == "server:example.com"
        option_values = [value for _label, value in runs_on._options]
        assert option_values == ["local", "server:example.com"]


@pytest.mark.asyncio
async def test_runs_on_is_disabled_when_editing_an_existing_task():
    """Edit mode: the owner is shown but fixed -- transfer is a separate feature."""
    task = ReminderTask(
        id="task-runs-on",
        title="Existing",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc),
        owner_id="local",
    )
    app = FormTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(
            ReminderForm(
                task,
                available_owners=[("This device", "local")],
                default_owner="local",
            )
        )
        await pilot.pause()
        runs_on = pilot.app.screen.query_one("#reminder-runs-on", Select)
        assert runs_on.disabled
        assert runs_on.value == "local"


@pytest.mark.asyncio
async def test_runs_on_appends_a_drifted_task_owner_not_in_the_offered_list():
    """Review F4 precedent (timezone selector): an edited task's real owner
    always round-trips, even when the offered choices no longer include it
    (e.g. the connected server changed since the task was created)."""
    task = ReminderTask(
        id="task-drifted-owner",
        title="Server task",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc),
        owner_id="server:drifted.example",
    )
    app = FormTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(
            ReminderForm(
                task,
                available_owners=[("This device", "local")],
                default_owner="local",
            )
        )
        await pilot.pause()
        runs_on = pilot.app.screen.query_one("#reminder-runs-on", Select)
        assert runs_on.disabled
        assert runs_on.value == "server:drifted.example"
        option_values = [value for _label, value in runs_on._options]
        assert option_values == ["local", "server:drifted.example"]


def test_initial_owner_prefers_the_tasks_own_owner_over_the_default():
    task = ReminderTask(
        id="task-owner-pref",
        title="T",
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2099, 7, 20, 14, 0, tzinfo=timezone.utc),
        owner_id="server:1",
    )
    form = ReminderForm(task, default_owner="local")
    assert form._initial_owner() == "server:1"


def test_initial_owner_falls_back_to_default_when_creating():
    form = ReminderForm(default_owner="server:2")
    assert form._initial_owner() == "server:2"


@pytest.mark.asyncio
async def test_runs_on_selection_is_included_in_the_submitted_form_data():
    app = FormTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(
            ReminderForm(
                available_owners=[
                    ("This device", "local"),
                    ("Server (example.com)", "server:example.com"),
                ],
                default_owner="local",
            )
        )
        await pilot.pause()
        form = pilot.app.screen
        form.query_one("#reminder-title", Input).value = "Owner-routed"
        form.query_one("#reminder-run-at", Input).value = "2099-08-28 09:00"
        form.query_one("#reminder-runs-on", Select).value = "server:example.com"
        await pilot.click("#reminder-save")
        await pilot.pause()

        assert app.submitted is not None
        assert app.submitted["owner_id"] == "server:example.com"
