"""Reminder create/edit modal form."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from tldw_chatbook.Scheduling.events import ReminderFormSubmitted
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind


_DEFAULT_TIMEZONE = "UTC"

#: Curated common zones offered alongside the system zone (task-23102).
#: The list is deliberately short: the system zone leads, zones already
#: used by the user's existing tasks are appended, and UTC is always
#: present -- a user needing an exotic zone gets it the moment one of
#: their tasks (or their machine) uses it.
_CURATED_TIMEZONES: tuple[str, ...] = (
    "UTC",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Berlin",
    "Europe/Paris",
    "Europe/Madrid",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Asia/Kolkata",
    "Australia/Sydney",
)

#: Forgiving local datetime formats (naive -> system local zone).
_FORGIVING_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)

_TIME_OF_DAY_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

#: Day-of-week field per time-of-day preset (task-23102).
_PRESET_DOW: dict[str, str] = {
    "daily": "*",
    "weekday": "1-5",
    "monday": "1",
}

_TIME_OF_DAY_PRESETS = frozenset(_PRESET_DOW)


# Why these parsers live here and not in ``Utils/input_validation.py``:
# that module is the *security* boundary ("Input validation utilities for
# secure user input handling") -- boolean gatekeepers for traversal,
# injection, SSRF and size-class risks. The helpers below are domain
# format parsers: they normalize text into schedule values (an aware
# datetime, an (hour, minute) pair, a cron expression) and hand back
# presentation signals the form renders as live hints, such as
# ``parse_forgiving_datetime``'s ``assumed_local`` flag. Nothing here
# guards a trust boundary -- the parsed values reach SQLite only through
# parameterized queries, and cron validity is enforced by ``croniter`` --
# so hoisting them would import croniter, zoneinfo and this screen's
# preset vocabulary into a module the security-critical paths depend on,
# for no safety gain. Bounds checks that DO protect a boundary belong in
# the shared module; these belong with the form whose messages they feed.
def _is_valid_zone(name: str) -> bool:
    """Return True when ``name`` resolves to an IANA timezone.

    Args:
        name: Candidate IANA zone name, e.g. ``"Europe/Berlin"``.

    Returns:
        True when the local tzdata can resolve ``name``, False otherwise.
    """
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return False
    return True


def detect_system_timezone() -> str | None:
    """Best-effort IANA name for the machine's local timezone, or None.

    Checks ``TZ`` first, then the ``/etc/localtime`` symlink (macOS and
    Linux both point it into a ``zoneinfo`` tree). Returns None where
    neither yields a valid zone (copied-file distros, containers,
    Windows) so callers can label the UTC fallback honestly instead of
    claiming it is the machine's zone (review F7).

    Returns:
        The detected IANA zone name, or None when detection fails.
    """
    tz_env = os.environ.get("TZ", "").strip()
    if tz_env and _is_valid_zone(tz_env):
        return tz_env
    try:
        localtime = os.path.realpath("/etc/localtime")
    except OSError:
        localtime = ""
    if "/zoneinfo/" in localtime:
        name = localtime.split("/zoneinfo/", 1)[1]
        if _is_valid_zone(name):
            return name
    return None


def system_timezone_name() -> str:
    """The detected machine zone, or UTC when detection fails.

    Returns:
        The IANA zone name from :func:`detect_system_timezone`, falling
        back to ``"UTC"``.
    """
    return detect_system_timezone() or _DEFAULT_TIMEZONE


def parse_forgiving_datetime(raw: str) -> tuple[datetime | None, bool]:
    """Parse a run-at datetime, accepting forgiving local forms.

    Returns ``(parsed, assumed_local)``: full ISO-8601 keeps its offset
    (``assumed_local`` False); a naive form such as ``2026-08-28 09:00``
    is interpreted in the system's local timezone (``assumed_local``
    True). ``(None, False)`` when nothing parses.

    Args:
        raw: The user-entered run-at text; surrounding whitespace is
            ignored and an empty string is treated as "not a date".

    Returns:
        A ``(datetime | None, bool)`` pair. The datetime is always
        timezone-aware when parsing succeeds. The bool is True only when
        a naive input was assumed to be local time, so the caller can say
        so in the UI.
    """
    text = raw.strip()
    if not text:
        return None, False
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in _FORGIVING_DATETIME_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            break
    if parsed is None:
        return None, False
    if parsed.tzinfo is None:
        return parsed.astimezone(), True
    return parsed, False


def parse_time_of_day(raw: str) -> tuple[int, int] | None:
    """Parse ``HH:MM`` (24-hour, single-digit hour allowed) to (hour, minute).

    Args:
        raw: The user-entered time of day; surrounding whitespace is
            ignored.

    Returns:
        An ``(hour, minute)`` pair with ``0 <= hour <= 23`` and
        ``0 <= minute <= 59``, or None when ``raw`` is not a 24-hour time.
    """
    match = _TIME_OF_DAY_RE.match(raw.strip())
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def preset_to_cron(preset: str, time_of_day: str) -> str | None:
    """Generate the cron expression for a frequency preset (task-23102).

    Returns None for the custom preset (the raw cron field owns it) and
    for an unparseable time of day.

    Args:
        preset: A frequency preset key -- ``"hourly"`` or one of the
            time-of-day presets in ``_PRESET_DOW`` (``"daily"``,
            ``"weekday"``, ``"monday"``). Any other value, including
            ``"custom"``, yields None.
        time_of_day: ``HH:MM`` text for the time-of-day presets. Ignored
            for ``"hourly"``.

    Returns:
        A five-field cron expression, or None when the preset owns no
        generated cron or ``time_of_day`` does not parse.
    """
    if preset == "hourly":
        return "0 * * * *"
    dow = _PRESET_DOW.get(preset)
    if dow is None:
        return None
    parsed = parse_time_of_day(time_of_day)
    if parsed is None:
        return None
    hour, minute = parsed
    return f"{minute} {hour} * * {dow}"


def _is_ascii_digit(value: str) -> bool:
    """True for plain ASCII digits only.

    ``str.isdigit()`` alone accepts unicode digits like '²' that
    ``int()`` then refuses -- a synced or DB-sourced cron with such a
    field must map to custom, not crash (review F14).

    Args:
        value: A single cron field.

    Returns:
        True only for a non-empty run of ASCII ``0-9``.
    """
    return bool(value) and value.isascii() and value.isdigit()


def cron_to_preset(cron: str) -> tuple[str, str]:
    """Map a cron expression back to ``(preset, "HH:MM")``.

    Unrecognized expressions map to ``("custom", "")`` so editing an
    advanced task lands on the raw cron field with the expression intact.

    Args:
        cron: The stored cron expression to reverse-map.

    Returns:
        A ``(preset, time_text)`` pair. ``time_text`` is zero-padded
        ``HH:MM`` for the time-of-day presets and ``""`` for ``"hourly"``
        and ``"custom"``.
    """
    if cron.strip() == "0 * * * *":
        return "hourly", ""
    parts = cron.split()
    if len(parts) == 5:
        minute, hour, dom, month, dow = parts
        if (
            _is_ascii_digit(minute)
            and _is_ascii_digit(hour)
            and dom == "*"
            and month == "*"
        ):
            time_text = f"{int(hour):02d}:{int(minute):02d}"
            for preset, preset_dow in _PRESET_DOW.items():
                if dow == preset_dow:
                    return preset, time_text
    return "custom", ""


class ReminderForm(ModalScreen):
    """Modal form for creating or editing a reminder."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ReminderForm {
        align: center middle;
    }

    /* task-23100 (shape revised in the review round, F8): the modal box
       IS the scroll container (height:auto clamped by max-height means
       overflow becomes a scrollbar, never clipping), and the
       preview/errors/buttons footer is DOCKED to its bottom -- docked
       children are positioned against the visible bottom edge, so the
       footer can never be clipped and no height arithmetic is needed
       (wrapped error/preview lines simply grow the dock reserve). Same
       pattern as voice_blend_dialog / feedback_dialog. */
    ReminderForm > VerticalScroll {
        width: 80;
        max-width: 100%;
        height: auto;
        max-height: 100%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #reminder-form-footer {
        dock: bottom;
        height: auto;
        background: $surface;
    }

    /* Plain Vertical defaults to height:1fr, which measures as ~1 row in
       the scroll container's virtual size while its children paint taller
       -- the invisible-but-focusable trap (task-23100). */
    #reminder-run-at-group,
    #reminder-cron-group,
    #reminder-timezone-group,
    #reminder-preset-time-group,
    #reminder-cron-custom-group {
        height: auto;
    }

    #reminder-errors {
        display: none;
    }

    #reminder-body {
        height: 3;
        max-height: 5;
    }

    .form-title {
        text-style: bold;
        text-align: center;
        padding: 0;
    }

    .form-label {
        color: $text-muted;
        padding: 1 0 0 0;
    }

    .form-helper {
        color: $text-muted;
        height: auto;
        padding: 0;
    }

    .form-preview {
        color: $text-muted;
        height: auto;
        min-height: 1;
        padding: 0;
    }

    .error-text {
        color: $error;
        text-style: bold;
        height: auto;
        padding: 0;
    }

    .button-container {
        align: center middle;
        height: auto;
        padding: 0;
        margin-top: 1;
    }

    .button-container Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        task: ReminderTask | None = None,
        *,
        known_timezones: Sequence[str] = (),
    ) -> None:
        """Initialize the form.

        Args:
            task: Existing reminder to edit, or ``None`` to create a new one.
            known_timezones: Zones already used by existing tasks; offered
                in the timezone selector alongside the system zone and the
                curated common zones (task-23102).
        """
        super().__init__()
        self._reminder_task = task
        self._known_timezones = tuple(known_timezones)
        self._dirty = False
        self._ready = False

    def _timezone_options(self) -> list[tuple[str, str]]:
        """(label, zone) options: system zone first, then curated zones,
        then task-used zones, then this task's stored zone.

        The edited task's OWN stored zone is always offered -- even when
        it does not resolve in local tzdata -- labeled as unrecognized,
        so an unrelated edit round-trips it instead of silently rewriting
        the task's timezone to the system zone (review F4). An undetected
        machine zone is labeled honestly (review F7).
        """
        detected = detect_system_timezone()
        zones = [detected or _DEFAULT_TIMEZONE]
        candidates = list(_CURATED_TIMEZONES) + [
            zone for zone in self._known_timezones if zone
        ]
        task_zone = getattr(self._reminder_task, "timezone", None)
        if task_zone:
            candidates.append(task_zone)
        for zone in candidates:
            if zone not in zones and _is_valid_zone(zone):
                zones.append(zone)

        options: list[tuple[str, str]] = []
        for zone in zones:
            if detected is None and zone == _DEFAULT_TIMEZONE:
                options.append((f"{zone} — machine zone not detected", zone))
            else:
                options.append((zone, zone))
        if task_zone and task_zone not in zones:
            options.append(
                (f"{task_zone} — stored on this task, not recognized here", task_zone)
            )
        return options

    def _initial_timezone(self) -> str:
        """The zone preselected on open: the task's own, else the system's.

        The stored zone wins even when it does not resolve locally
        (review F4): replacing it on open is what caused unrelated saves
        to shift the recurrence.
        """
        task_zone = getattr(self._reminder_task, "timezone", None)
        if task_zone:
            return task_zone
        return system_timezone_name()

    def _timezone_helper_copy(self) -> str:
        """Helper copy that does not over-claim detection (review F7)."""
        if detect_system_timezone() is None:
            return "Machine timezone not detected — defaulting to UTC."
        return "Defaults to this machine's timezone."

    def action_dismiss(self) -> None:
        """Dismiss the modal when the Escape key is pressed."""
        self._maybe_discard()

    def _maybe_discard(self) -> None:
        """Close the form, confirming first when edits would be lost."""
        if not self._dirty:
            self.dismiss(None)
            return
        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        async def _discard() -> None:
            self.dismiss(None)

        self.app.push_screen(
            ConfirmationDialog(
                title="Discard changes?",
                message="You have unsaved changes in this form.",
                confirm_label="Discard",
                cancel_label="Keep editing",
                confirm_callback=_discard,
            )
        )

    def compose(self) -> ComposeResult:
        """Build the form layout.

        The modal box is the scroll container itself; the footer is
        docked to its bottom so the live preview, validation line, and
        actions are visible at every terminal size (task-23100, review
        round F8).
        """
        with VerticalScroll(id="reminder-form-box"):
            yield Label(
                "Edit Scheduled Task" if self._reminder_task else "New Scheduled Task",
                classes="form-title",
            )

            yield Label("Title:", classes="form-label")
            yield Input(placeholder="Name this scheduled task…", id="reminder-title")

            yield Label("Body:", classes="form-label")
            yield TextArea(id="reminder-body")

            yield Label("Schedule Kind:", classes="form-label")
            yield Select(
                self._schedule_options(),
                allow_blank=False,
                value=ScheduleKind.ONE_TIME.value,
                id="reminder-kind",
            )
            yield Static(
                "One-time runs once; recurring repeats on a cron schedule.",
                classes="form-helper",
            )

            with Vertical(id="reminder-run-at-group"):
                yield Label("Run at:", classes="form-label")
                yield Input(
                    placeholder="2026-08-28 09:00",
                    id="reminder-run-at",
                )
                yield Static(
                    "A local time like 2026-08-28 09:00, or full ISO-8601 with offset.",
                    classes="form-helper",
                )

            with Vertical(id="reminder-cron-group"):
                yield Label("Frequency:", classes="form-label")
                yield Select(
                    self._preset_options(),
                    allow_blank=False,
                    value="daily",
                    id="reminder-cron-preset",
                )
                with Vertical(id="reminder-preset-time-group"):
                    yield Label("Time of day (24-hour):", classes="form-label")
                    yield Input(
                        placeholder="09:00",
                        id="reminder-preset-time",
                    )
                with Vertical(id="reminder-cron-custom-group"):
                    yield Label("Cron Expression:", classes="form-label")
                    yield Input(placeholder="0 9 * * 1", id="reminder-cron")
                    yield Static(
                        "5-field cron (minute hour day month weekday): 0 9 * * * = daily at 9:00.",
                        classes="form-helper",
                    )

            with Vertical(id="reminder-timezone-group"):
                yield Label("Timezone:", classes="form-label")
                yield Select(
                    self._timezone_options(),
                    allow_blank=False,
                    value=self._initial_timezone(),
                    id="reminder-timezone",
                )
                yield Static(
                    self._timezone_helper_copy(),
                    id="reminder-timezone-helper",
                    classes="form-helper",
                )

            # Docked footer (task-23100/F8): the live schedule preview,
            # the validation line, and the actions stay pinned to the
            # visible bottom edge while the fields scroll behind them.
            with Vertical(id="reminder-form-footer"):
                yield Static("", id="reminder-run-at-preview", classes="form-preview")
                yield Static("", id="reminder-cron-preview", classes="form-preview")
                yield Static("", id="reminder-errors", classes="error-text")
                with Horizontal(classes="button-container"):
                    yield Button("Save", variant="success", id="reminder-save")
                    yield Button("Cancel", id="reminder-cancel")

    def on_mount(self) -> None:
        """Prefill the form when editing an existing reminder."""
        if self._reminder_task is None:
            # Create mode: start from the default preset (daily at 09:00).
            self.query_one("#reminder-preset-time", Input).value = "09:00"
            self.query_one("#reminder-cron", Input).value = "0 9 * * *"
            self._update_preset_field_visibility("daily")
            self._update_cron_preview()
            self._update_schedule_field_visibility(ScheduleKind.ONE_TIME.value)
            self.call_after_refresh(self._mark_ready)
            return

        self.query_one("#reminder-title", Input).value = self._reminder_task.title
        body = self._reminder_task.body or ""
        self.query_one("#reminder-body", TextArea).text = body
        self.query_one("#reminder-kind", Select).value = self._reminder_task.schedule_kind.value
        if self._reminder_task.run_at is not None:
            self.query_one("#reminder-run-at", Input).value = self._reminder_task.run_at.isoformat()
            self._update_run_at_preview()
        if self._reminder_task.cron is not None:
            self.query_one("#reminder-cron", Input).value = self._reminder_task.cron
            self._apply_cron_to_preset_fields(self._reminder_task.cron)
            self._update_cron_preview()
        else:
            # Editing a one-time task (review F2): initialize the preset
            # fields exactly like create mode, so switching Kind to
            # Recurring reveals a coherent preset + time (not both
            # sub-groups at once with a preset that silently overrides a
            # typed cron on save).
            self.query_one("#reminder-preset-time", Input).value = "09:00"
            self.query_one("#reminder-cron", Input).value = "0 9 * * *"
            self._update_preset_field_visibility("daily")
            self._update_cron_preview()
        self._update_schedule_field_visibility(self._reminder_task.schedule_kind.value)
        self.call_after_refresh(self._mark_ready)

    def _mark_ready(self) -> None:
        """Enable dirty tracking after mount-time prefill settles."""
        self._dirty = False
        self._ready = True

    def _set_errors(self, errors: list[str]) -> None:
        """Render (or clear) the validation lines in the docked footer."""
        error_widget = self.query_one("#reminder-errors", Static)
        error_widget.update("\n".join(errors))
        error_widget.display = bool(errors)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Show/hide schedule field groups based on the selected schedule kind."""
        if not self._ready:
            return
        self._dirty = True
        if event.select.id == "reminder-kind":
            self._update_schedule_field_visibility(str(event.value))
        elif event.select.id == "reminder-cron-preset":
            preset = str(event.value)
            self._update_preset_field_visibility(preset)
            self._regenerate_preset_cron()
            self._update_cron_preview()
        elif event.select.id == "reminder-timezone":
            self._update_cron_preview()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Keep the live previews in sync with typing."""
        if not self._ready:
            return
        self._dirty = True
        if event.input.id == "reminder-cron":
            # Custom cron edits never flip the preset out from under the
            # user (task-23102); the preview is the feedback channel.
            self._update_cron_preview()
        elif event.input.id == "reminder-preset-time":
            self._regenerate_preset_cron()
            self._update_cron_preview()
        elif event.input.id == "reminder-run-at":
            self._update_run_at_preview()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Mark the form dirty when the body text changes."""
        if self._ready:
            self._dirty = True

    def _update_schedule_field_visibility(self, kind: str) -> None:
        """Toggle which schedule input groups (and pinned preview) are visible."""
        run_at_group = self.query_one("#reminder-run-at-group", Vertical)
        cron_group = self.query_one("#reminder-cron-group", Vertical)
        tz_group = self.query_one("#reminder-timezone-group", Vertical)
        run_at_preview = self.query_one("#reminder-run-at-preview", Static)
        cron_preview = self.query_one("#reminder-cron-preview", Static)
        one_time = kind == ScheduleKind.ONE_TIME.value
        run_at_group.display = one_time
        cron_group.display = not one_time
        tz_group.display = not one_time
        run_at_preview.display = one_time
        cron_preview.display = not one_time
        # Reveal the newly shown group: at short terminal heights it may sit
        # below the scroll window (task-23100).
        shown_group = run_at_group if one_time else cron_group
        self.call_after_refresh(shown_group.scroll_visible)

    @staticmethod
    def _schedule_options() -> list[tuple[str, str]]:
        """Return labelled options for the schedule kind selector."""
        return [
            (kind.value.replace("_", " ").title(), kind.value) for kind in ScheduleKind
        ]

    #: Recurring presets (task-23102): time-of-day presets pair with the
    #: "Time of day" field and generate the cron; "custom" reveals the raw
    #: cron input for power users (UX-069).
    _CRON_PRESETS: tuple[tuple[str, str], ...] = (
        ("Every day at…", "daily"),
        ("Every weekday at…", "weekday"),
        ("Every Monday at…", "monday"),
        ("Every hour", "hourly"),
        ("Custom cron…", "custom"),
    )

    @classmethod
    def _preset_options(cls) -> list[tuple[str, str]]:
        """Return labelled options for the frequency preset selector."""
        return [(label, value) for label, value in cls._CRON_PRESETS]

    def _selected_preset(self) -> str:
        """The frequency preset currently selected."""
        return str(self.query_one("#reminder-cron-preset", Select).value)

    def _selected_timezone(self) -> str:
        """The timezone currently selected."""
        return str(self.query_one("#reminder-timezone", Select).value)

    def _update_preset_field_visibility(self, preset: str) -> None:
        """Show the time-of-day field for presets, the raw cron for custom."""
        time_group = self.query_one("#reminder-preset-time-group", Vertical)
        custom_group = self.query_one("#reminder-cron-custom-group", Vertical)
        time_group.display = preset in _TIME_OF_DAY_PRESETS
        custom_group.display = preset == "custom"

    def _regenerate_preset_cron(self) -> None:
        """Regenerate the cron expression from the preset + time of day."""
        preset = self._selected_preset()
        if preset == "custom":
            return
        time_text = self.query_one("#reminder-preset-time", Input).value
        cron = preset_to_cron(preset, time_text)
        if cron is not None:
            self.query_one("#reminder-cron", Input).value = cron

    def _apply_cron_to_preset_fields(self, cron: str) -> None:
        """Reflect an existing task's cron in the preset + time fields."""
        preset, time_text = cron_to_preset(cron)
        self.query_one("#reminder-cron-preset", Select).value = preset
        if time_text:
            self.query_one("#reminder-preset-time", Input).value = time_text
        self._update_preset_field_visibility(preset)

    def _update_cron_preview(self) -> None:
        """Live-translate the recurrence into plain English."""
        from ..task_detail import _humanize_cron

        preview = self.query_one("#reminder-cron-preview", Static)
        preset = self._selected_preset()
        timezone = self._selected_timezone()
        if preset in _TIME_OF_DAY_PRESETS:
            time_text = self.query_one("#reminder-preset-time", Input).value
            if parse_time_of_day(time_text) is None:
                preview.update("Enter a time of day like 09:00.")
                return
        cron = self.query_one("#reminder-cron", Input).value.strip()
        if not cron:
            preview.update("")
        elif croniter.is_valid(cron):
            preview.update(f"Runs: {_humanize_cron(cron, timezone or _DEFAULT_TIMEZONE)}")
        else:
            preview.update("Not a valid cron expression yet.")

    def _update_run_at_preview(self) -> None:
        """Live-parse the one-time run date into a friendly confirmation."""
        raw = self.query_one("#reminder-run-at", Input).value.strip()
        preview = self.query_one("#reminder-run-at-preview", Static)
        if not raw:
            preview.update("")
            return
        parsed, assumed_local = parse_forgiving_datetime(raw)
        if parsed is None:
            preview.update("Not a date-time yet — try 2026-08-28 09:00.")
            return
        rendered = f"{parsed.strftime('%Y-%m-%d %H:%M')} {parsed.tzname() or 'UTC'}"
        if assumed_local:
            preview.update(f"Runs: {rendered} (your local time)")
        else:
            preview.update(f"Runs: {rendered}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle save/cancel button presses."""
        if event.button.id == "reminder-cancel":
            self._maybe_discard()
        elif event.button.id == "reminder-save":
            self._save()

    def _save(self) -> None:
        """Validate the form and emit the submitted event on success."""
        title = self.query_one("#reminder-title", Input).value.strip()
        body = self.query_one("#reminder-body", TextArea).text.strip()
        schedule_kind = str(self.query_one("#reminder-kind", Select).value)
        run_at = self.query_one("#reminder-run-at", Input).value.strip()
        timezone = self._selected_timezone()
        preset = self._selected_preset()
        if preset == "custom":
            cron = self.query_one("#reminder-cron", Input).value.strip()
        else:
            time_text = self.query_one("#reminder-preset-time", Input).value
            cron = preset_to_cron(preset, time_text) or ""

        errors: list[str] = []
        if not title:
            errors.append("Title is required")

        parsed_run_at: datetime | None = None
        if schedule_kind == ScheduleKind.ONE_TIME.value:
            if not run_at:
                errors.append("Run At is required for one-time tasks")
            else:
                parsed_run_at, _assumed_local = parse_forgiving_datetime(run_at)
                if parsed_run_at is None:
                    errors.append(
                        "Run At must be a date and time like 2026-08-28 09:00"
                    )
                else:
                    # Creating a one-time task in the past can never run;
                    # editing an existing (possibly missed) task stays allowed.
                    if self._reminder_task is None:
                        now = datetime.now(parsed_run_at.tzinfo)
                        if parsed_run_at < now:
                            errors.append(
                                "That time is in the past — pick a future time."
                            )
        elif schedule_kind == ScheduleKind.RECURRING.value:
            if preset in _TIME_OF_DAY_PRESETS and not cron:
                errors.append("Time of day must be a 24-hour time like 09:00")
            elif not cron:
                errors.append("Cron expression is required for recurring tasks")
            elif not croniter.is_valid(cron):
                errors.append("Cron expression is invalid")

            if not timezone:
                errors.append("Timezone is required for recurring tasks")
            else:
                # The Select only offers vetted zones plus the edited
                # task's own stored zone; that stored zone must round-trip
                # even when local tzdata cannot resolve it (review F4).
                # The unknown-zone error stays as a defensive backstop for
                # programmatic value assignment only.
                stored_zone = getattr(self._reminder_task, "timezone", None)
                if timezone != stored_zone and not _is_valid_zone(timezone):
                    errors.append(f"Unknown timezone: {timezone}")

        if errors:
            self._set_errors(errors)
            return

        self._set_errors([])

        form_data: dict[str, Any] = {
            "title": title,
            "body": body,
            "schedule_kind": schedule_kind,
        }
        if schedule_kind == ScheduleKind.ONE_TIME.value:
            form_data["run_at"] = parsed_run_at
            form_data["cron"] = None
            form_data["timezone"] = None
        else:
            form_data["run_at"] = None
            form_data["cron"] = cron
            form_data["timezone"] = timezone

        if self._reminder_task is not None:
            # Preserve the current enabled state when editing.
            form_data["enabled"] = self._reminder_task.enabled

        self.post_message(ReminderFormSubmitted(form_data, task_id=self._reminder_task.id if self._reminder_task else None))
        self.dismiss(form_data)
