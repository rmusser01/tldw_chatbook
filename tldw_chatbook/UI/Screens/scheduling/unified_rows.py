"""Pure row adapter for the unified Schedules Queue list (redesign PR-2, Task 1).

Turns reminders (`ReminderTask` model instances) and automation-definition
rows (raw dicts, local and server-mirrored alike) into one `UnifiedRow`
shape the Queue tab renders, filters by chip, and searches -- spanning
both primitives and both owners (spec
`backlog/docs/spec-2026-09-02-schedules-screen-redesign.md` SS3/SS4, plan
`backlog/docs/plan-2026-09-03-schedules-redesign-pr2.md` ruling 2).

Deliberately Textual-free (no `import textual` anywhere in this file, nor
in anything it imports): this module's OWN dependency chain never needs
Textual, so its logic can be exercised with plain function calls, no
Textual harness required. That claim is scoped to this file's own
imports, not to importing its dotted package path in general -- `import
tldw_chatbook.UI.Screens.scheduling.unified_rows` still runs
`.../scheduling/__init__.py` first, which already eagerly imports
`SchedulesWorkbench` (Textual) regardless of what this module needs;
Textual is a hard dependency present in every real run/test environment
anyway, so that pre-existing package-init cost is harmless in practice.
`task_detail.py`, `definition_detail.py`, and `results_tab.py` each used
to define a handful of small, genuinely pure formatters this module also
needs (schedule-summary prose, the owner label, the dual local/server
id-space result->definition resolver). Importing them FROM those modules
would execute those modules' own top-level `textual` imports as a side
effect -- so they are HOISTED here instead, and those three modules now
import the names back (`from .unified_rows import ...`), keeping every
existing call site, test import, and docstring cross-reference unchanged.
"""

from __future__ import annotations

import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import DORMANT_TRANSFER_STATES
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.Scheduling.schedule_compute import _parse_time_of_day

RowKind = Literal["reminder", "definition"]
RowBucket = Literal["active", "paused", "completed"]
Chip = Literal["all", "active", "paused", "completed"]

_CHIP_BUCKETS: dict[str, tuple[RowBucket, ...]] = {
    "all": ("active", "paused"),
    "active": ("active",),
    "paused": ("paused",),
    "completed": ("completed",),
}


@dataclass
class UnifiedRow:
    """One reminder or automation-definition row, shaped for the unified list.

    Attributes:
        kind: ``"reminder"`` or ``"definition"``.
        row_id: A stable, cross-kind-collision-safe list key
            (``f"{kind}:{local_id}"``).
        title: The row's display title (``ReminderTask.title`` /
            ``definition["name"]``).
        schedule_summary: Humanized cadence prose ("Daily at 09:00 UTC",
            "One-time at 2026-09-03 09:00 UTC"), via the existing
            formatters -- never re-derived.
        next_run_at: The row's next scheduled fire time, normalized to a
            real ``datetime`` for both primitives (a definition's
            ``next_run_at`` is a raw ISO string at the DB layer).
        owner_id: The row's raw owner id (``"local"`` / ``"server:<id>"``).
        owner_label: The prose owner label (`owner_display_label`).
        transfer_state: The row's raw transfer-state column value, passed
            through for the caller's own badge rendering -- this module
            never renders badges.
        bucket: Which of the three chip buckets the row currently lives
            in (`filter_rows`/`sort_rows` consume this).
        glyph: The one-character status glyph (spec SS4).
        unread_count: Unread `automation_results` rows resolved to this
            definition (always ``0`` for a reminder row -- reminders
            never produce results).
        search_blob: ``title`` + body/question text, for `filter_rows`'s
            case-insensitive substring search.
        source_row: The original `ReminderTask` or definition ``dict``,
            for a caller that needs a field this adapter does not expose.
    """

    kind: RowKind
    row_id: str
    title: str
    schedule_summary: str
    next_run_at: datetime | None
    owner_id: str
    owner_label: str
    transfer_state: str | None
    bucket: RowBucket
    glyph: str
    unread_count: int
    search_blob: str
    source_row: ReminderTask | dict[str, Any]


# ---------------------------------------------------------------------------
# Hoisted formatters (moved from task_detail.py / definition_detail.py /
# results_tab.py -- see module docstring for why). Each keeps its ORIGINAL
# name and behavior; the donor modules now import these back.
# ---------------------------------------------------------------------------

#: Weekday names for `_humanize_cron`'s single-weekday preset (index 0-6,
#: Sunday-first -- matches `croniter`'s day-of-week numbering).
_WEEKDAYS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]


def _format_timezone(dt) -> str:
    """Return a timezone label for a datetime, defaulting to UTC.

    Args:
        dt: A ``datetime`` (naive or aware).

    Returns:
        The datetime's ``tzname()``, or ``"UTC"`` for a naive value or one
        whose ``tzname()`` is falsy.
    """
    if dt.tzinfo is None:
        return "UTC"
    return dt.tzname() or "UTC"


def _humanize_cron(cron: str | None, timezone: str | None = None) -> str:
    """Summarize a cron expression in plain English.

    Args:
        cron: A 5-field cron string, or ``None``.
        timezone: The cron's timezone label; defaults to ``"UTC"``.

    Returns:
        A human-readable cadence summary, or ``"-"``/the raw string when
        the expression is empty/not a recognized 5-field preset.
    """
    if not cron:
        return "-"
    parts = cron.split()
    if len(parts) != 5:
        return cron
    minute, hour, dom, month, dow = parts
    tz = f" {timezone}" if timezone else " UTC"

    def _is_wildcard(value: str) -> bool:
        return value == "*"

    def _is_digit(value: str) -> bool:
        # ASCII only: '²'.isdigit() is True but int('²') raises, and this
        # runs on every detail render of a synced cron (review F14).
        return bool(value) and value.isascii() and value.isdigit()

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_wildcard(dom)
        and _is_wildcard(month)
        and _is_wildcard(dow)
    ):
        return f"Daily at {int(hour):02d}:{int(minute):02d}{tz}"

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_wildcard(dom)
        and _is_wildcard(month)
        and dow == "1-5"
    ):
        # The "Every weekday at..." preset (task-23102).
        return f"Weekdays at {int(hour):02d}:{int(minute):02d}{tz}"

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_wildcard(dom)
        and _is_wildcard(month)
        and _is_digit(dow)
    ):
        day_index = int(dow)
        if 0 <= day_index <= 6:
            return f"Weekly on {_WEEKDAYS[day_index]} at {int(hour):02d}:{int(minute):02d}{tz}"

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_digit(dom)
        and _is_wildcard(month)
        and _is_wildcard(dow)
    ):
        return f"Monthly on the {int(dom)} at {int(hour):02d}:{int(minute):02d}{tz}"

    return f"cron: {cron}{tz}"


def _humanize_schedule(task: ReminderTask) -> str:
    """Return a human-readable schedule summary for a reminder.

    Args:
        task: The reminder to summarize.

    Returns:
        "One-time" prose for a one-time reminder, or the humanized cron
        for a recurring one.
    """
    if task.schedule_kind == ScheduleKind.ONE_TIME:
        if task.run_at is None:
            return "One-time"
        return f"One-time at {task.run_at.strftime('%Y-%m-%d %H:%M')} {_format_timezone(task.run_at)}"
    return _humanize_cron(task.cron, task.timezone)


def definition_cron_expression(schedule: dict[str, Any]) -> Any:
    """An automation definition's cron string, under EITHER key.

    The two writers disagree: this client writes ``schedule["cron"]``
    (`AutomationDefinitionForm`'s save payload), the real server sends
    ``schedule["expression"]`` (recorded fixture
    ``Tests/Scheduling/fixtures/server_responses/
    automation_definition_list.json``), and ``_load_server_automations``
    passes the payload through raw, stamping only ``owner_id``.

    Both readers of that field go through here (final review F1 + its
    carry-forward, originally in `task_detail.py` before this hoist):
    `definition_detail`'s "At" row -- where a cron-only read rendered
    ``At: -`` for EVERY server-owned definition -- and
    ``AutomationDefinitionForm._prefill_from_row``, where the same read
    was worse than cosmetic: editing a mirrored server-only definition
    fell through to the form's default preset, so a save wrote that
    default OVER the server's real schedule.

    Args:
        schedule: A definition's ``schedule`` dict, from either source.

    Returns:
        The cron string, or ``None``/empty when neither key carries one.
    """
    return schedule.get("cron") or schedule.get("expression")


def owner_display_label(owner_id: Any) -> str:
    """Prose owner label shared by every schedules surface (final review F6/F7).

    ``"This device"`` for a locally-owned row, the server's own id for a
    server-scoped one (``"server:srv-1"`` -> ``"srv-1"``) -- the
    vocabulary the spec, the User Guide, and the Automations table's own
    Name-cell prefix already use. The reminder pane's `Runs on` row used
    to render the raw metadata string instead (``local``, ``server:1``/
    ``server <id>``), so the two panes spoke two dialects for the
    flagship row of the redesign; one helper, one vocabulary.

    `TaskInspector`'s Owner row deliberately keeps `_task_owner_label`'s
    raw value instead: that pane is the metadata inspector, where the
    unprettied owner/server ids are the point.

    Args:
        owner_id: A row's ``owner_id`` (any type tolerated -- anything
            that is not a server-scoped string reads as local).

    Returns:
        ``"This device"``, or the server's own id.
    """
    # ADR-097: scheduler.queue stays off the boot census -- function-local
    # import, matching the original call site this was hoisted from.
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    if not is_server_scoped_owner(owner_id):
        return "This device"
    owner_id = str(owner_id)
    return owner_id.split(":", 1)[1] if ":" in owner_id else owner_id


def _parse_iso(value: Any) -> datetime | None:
    """Best-effort ISO-8601 parse; ``None`` for anything else, never raises.

    Args:
        value: A candidate ISO-8601 timestamp string.

    Returns:
        The parsed ``datetime``, or ``None`` when ``value`` is not a
        parseable string.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _definition_at_label(schedule: dict[str, Any]) -> str:
    """The full schedule summary for a definition's ``schedule`` dict.

    The definition-side counterpart of `_humanize_schedule`, reusing
    `_humanize_cron` for a cron-kind schedule rather than re-deriving
    cron-cadence prose.

    ``daily``/``weekly`` render their own ``time_of_day`` (Qodo follow-up
    to finding 8): those two used to fall through to ``"-"`` here, which
    only became a contradiction once the `At` row was made genuinely
    editable for them -- editing the time and repainting a dash is the
    dishonest-repaint class ruling 2 exists to forbid. Deliberately NOT
    routed through `_humanize_cron` by synthesizing a cron: that
    function's ``_WEEKDAYS`` is Sunday-first (croniter's day-of-week),
    while a definition ``schedule["weekday"]`` is Monday-first
    (`schedule_compute._compute_weekly`, Python's own ``weekday()``), so
    the round trip would confidently name the wrong day. The wording
    still MATCHES `_humanize_cron`'s ("Daily at HH:MM TZ" / "Weekly on
    <Day> at HH:MM TZ") so the same cadence reads identically whichever
    kind expressed it.

    ``interval`` keeps its ``"-"``: it has no single time to show, and
    its `At` row is read-only, so no edit can repaint into it.

    Args:
        schedule: A definition's ``schedule`` dict -- ``kind`` is one of
            `schedule_compute.py`'s five (``one_time``/``interval``/
            ``daily``/``weekly``/``cron``).

    Returns:
        A human-readable schedule summary, or ``"-"`` for an
        unrecognized/absent schedule shape.
    """
    if not isinstance(schedule, dict):
        return "-"
    kind = schedule.get("kind")
    if kind == "cron":
        return _humanize_cron(
            definition_cron_expression(schedule), schedule.get("timezone")
        )
    if kind == "one_time":
        run_at = schedule.get("run_at")
        dt = _parse_iso(run_at) if run_at else None
        if dt is None:
            return f"One-time at {run_at}" if run_at else "One-time"
        return f"One-time at {dt.strftime('%Y-%m-%d %H:%M')} {_format_timezone(dt)}"
    if kind in ("daily", "weekly"):
        # The SAME parser the scheduler itself applies to this field, so
        # a value this label renders is exactly a value that will fire --
        # and unparseable text says "-" rather than being echoed as if it
        # were a schedule.
        parsed = _parse_time_of_day(schedule.get("time_of_day"))
        if parsed is None:
            return "-"
        at = f"{parsed.strftime('%H:%M')} {schedule.get('timezone') or 'UTC'}"
        weekday = schedule.get("weekday")
        if kind == "weekly":
            if isinstance(weekday, bool) or not isinstance(weekday, int):
                return f"Weekly at {at}"
            if not 0 <= weekday <= 6:
                return f"Weekly at {at}"
            return f"Weekly on {calendar.day_name[weekday]} at {at}"
        return f"Daily at {at}"
    return "-"


def _definition_question_text(definition: dict[str, Any]) -> str:
    """The recurring question/body text for search + question-card display.

    Args:
        definition: An automation-definition row (dict).

    Returns:
        The definition's ``input.question``, falling back to its
        ``description``, then its ``name``, then ``"Untitled automation"``.
    """
    input_fields = (
        definition.get("input") if isinstance(definition.get("input"), dict) else {}
    )
    question = str(input_fields.get("question") or "").strip()
    if question:
        return question
    description = str(definition.get("description") or "").strip()
    if description:
        return description
    return str(definition.get("name") or "Untitled automation")


def index_definitions_by_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index definition rows under BOTH id spaces they are referred to by.

    A result's ``definition_id`` is whatever the side that produced it
    calls the definition: a locally-created result carries the LOCAL row
    id, while a result mirrored down from the server carries the
    SERVER's id (`upsert_automation_results_from_server` copies
    ``definition_id`` verbatim -- it has no local id to translate it to).
    Indexing by ``row["id"]`` alone therefore misses every synced result.
    Local ids are UUID4 and server ids are the server's own opaque ids,
    so the two key spaces do not collide in practice; the local id is
    written first regardless, so a pathological clash still resolves to
    a row this device owns.

    Args:
        rows: Definition rows from
            `ScheduledTasksDB.list_automation_definitions` (or the
            server-mirrored equivalent).

    Returns:
        Each row keyed by its local ``id`` and, when present, by its
        ``server_id`` too.
    """
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        index[str(row["id"])] = row
    for row in rows:
        server_id = row.get("server_id")
        if server_id:
            index.setdefault(str(server_id), row)
    return index


def definition_for_result(
    result: dict[str, Any], definitions_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """The definition row a result belongs to, across both id spaces.

    Pairs with `index_definitions_by_id`.

    Args:
        result: An ``automation_results`` row.
        definitions_by_id: The index built by `index_definitions_by_id`.

    Returns:
        The owning definition row, or ``None`` when the result's
        ``definition_id`` matches neither id space.
    """
    return definitions_by_id.get(str(result.get("definition_id") or ""))


# ---------------------------------------------------------------------------
# Bucket predicates (plan ruling 2)
# ---------------------------------------------------------------------------


def reminder_has_fired(task: ReminderTask) -> bool:
    """True for a fired one-time reminder (survey SS6, task_detail.py:163-167).

    Dispatching a one-time reminder disables it AND clears
    ``next_run_at``; "has run at least once, has no next run" is what
    distinguishes "fired" from a user-initiated pause (which keeps its
    ``next_run_at``) and from a reminder that never ran (no
    ``last_run_at``).

    ``schedule_kind == ScheduleKind.ONE_TIME`` is REQUIRED (Qodo MEDIUM):
    only a one-time reminder's dispatch clears ``next_run_at`` for good.
    A recurring reminder can land in that same shape too -- ``cron``
    exhausted (end-date-passed) or an anomalous row -- and
    `mark_reminder_dispatched` never touches ``enabled`` on the recurring
    branch, so without this guard such a row read as "fired" and bucketed
    Completed despite still being armed to re-enable. `_reminder_bucket`
    routes that shape to Paused instead (disabled-not-finished, not
    finished-for-good).

    ``enabled`` is deliberately NOT part of the predicate (final review
    F9, ruled): re-enabling a fired one-time reminder does not give it a
    future run -- `_set_reminder_enabled` sends only ``{"enabled": ...}``
    and `SchedulingService.update_reminder` recomputes ``next_run_at``
    only when a SCHEDULE key is in the payload, while the due query
    filters ``next_run_at IS NOT NULL``. Bucketing that row Active would
    advertise armed status the scheduler will never honour.

    Args:
        task: The reminder to check.

    Returns:
        ``True`` when the reminder has fired its one-time schedule.
    """
    return (
        task.schedule_kind == ScheduleKind.ONE_TIME
        and task.next_run_at is None
        and task.last_run_at is not None
    )


def definition_is_armed(definition: dict[str, Any]) -> bool:
    """True when a ``configured`` definition is not sitting out a dormant transfer.

    Mirrors `ScheduledTasksDB.list_armable_automation_definitions`'s own
    lifecycle + `DORMANT_TRANSFER_STATES` gate (imported, never
    restated): a definition whose ``transfer_state`` is
    ``to_server_sent``/``from_server_pending`` has left this device's
    control until the transfer settles, so it does not count as the
    Active chip's "still runs here" claim. ``to_server_pending``/
    ``to_server_failed`` are NOT dormant and keep arming (spec SS3:
    Active "includ[es] to_server_pending/to_server_failed... they still
    execute locally").

    Args:
        definition: An automation-definition row (dict).

    Returns:
        ``True`` when the definition is configured and not dormant.
    """
    return (
        definition.get("lifecycle") == "configured"
        and definition.get("transfer_state") not in DORMANT_TRANSFER_STATES
    )


def _reminder_bucket(task: ReminderTask) -> RowBucket:
    if reminder_has_fired(task):
        return "completed"
    if not task.enabled:
        return "paused"
    if task.schedule_kind == ScheduleKind.RECURRING and task.next_run_at is None:
        # Ruled (Qodo MEDIUM, paired with `reminder_has_fired`'s ONE_TIME
        # guard): a recurring reminder with no next occurrence (exhausted
        # cron / anomalous row) is disabled-not-finished, not Completed --
        # recurring never "completes" the way a one-time reminder does.
        # It also is not caught by the `enabled` check above (`enabled`
        # is untouched on the recurring dispatch branch), so it needs its
        # own route to Paused rather than falling through to Active with
        # nothing left armed to run.
        return "paused"
    if task.transfer_state in DORMANT_TRANSFER_STATES:
        # Mirrors `_definition_bucket`'s dormant fallback (review round
        # 1, finding 1): `PriorityQueue.load()` excludes a dormant
        # transfer_state for BOTH primitives (`scheduler/queue.py:96-108`),
        # and `list_reminder_tasks(armable_only=True)`/
        # `reminders_due_before` already apply the same exclusion on the
        # reminder side. "Armed" (spec SS3) is one shared concept across
        # reminders and definitions, not two independently-scoped rules.
        return "paused"
    return "active"


def _definition_bucket(definition: dict[str, Any]) -> RowBucket:
    lifecycle = definition.get("lifecycle")
    if lifecycle == "archived":
        return "completed"
    if lifecycle in ("paused", "disabled"):
        return "paused"
    if definition_is_armed(definition):
        return "active"
    # A `configured` definition sitting out a dormant transfer (or an
    # unrecognized lifecycle value) is still a real, visible row -- park
    # it under Paused rather than silently dropping it from every chip.
    # `UnifiedRow.bucket` has no 4th "in transfer" state; its
    # `transfer_state` field still carries the raw value for the
    # caller's own badge.
    return "paused"


# ---------------------------------------------------------------------------
# Glyphs (spec SS4)
# ---------------------------------------------------------------------------


def _reminder_glyph(task: ReminderTask, bucket: RowBucket) -> str:
    if bucket == "paused":
        return "⏸"
    if bucket == "completed":
        return "✓"
    return "○" if task.schedule_kind == ScheduleKind.RECURRING else "▶"


def _definition_glyph(definition: dict[str, Any], bucket: RowBucket) -> str:
    if bucket == "paused":
        return "⏸"
    if bucket == "completed":
        return "✓"
    schedule = definition.get("schedule")
    kind = schedule.get("kind") if isinstance(schedule, dict) else None
    return "○" if kind == "cron" else "▶"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _unread_row_key(definition: dict[str, Any]) -> str:
    """The key a definition's unread count is stored and looked up under.

    ``server_id or id`` (final review F2): the DISPLAY list carries a
    transferred definition as the raw server payload (whose ``id`` IS the
    server id), while the local table carries the same definition as a
    row with a local uuid ``id`` and that server id in ``server_id``.
    Keying on ``server_id or id`` is the one expression both shapes agree
    on; a purely local row has no ``server_id`` and keeps its own id.
    """
    return str(definition.get("server_id") or definition.get("id") or "")


def _unread_counts_by_row_key(
    results: list[dict[str, Any]], definitions_by_id: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Group unread results by their resolved definition's row key.

    Routes every result through `definition_for_result` (the same dual
    local/server id-space resolution `results_tab.py` uses) BEFORE
    grouping, so a server-mirrored definition's results are counted for
    that definition even though the result row itself may carry the
    server's id verbatim (plan ruling 3).

    Args:
        results: All-owners `automation_results` rows (one
            `list_automation_results(owner_id=None)` call per refresh).
        definitions_by_id: The index built by `index_definitions_by_id`.

    Returns:
        ``{_unread_row_key(definition): unread_count}``, omitting
        definitions with zero unread results.
    """
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        if result.get("review_state") != "unread":
            continue
        definition = definition_for_result(result, definitions_by_id)
        if definition is None:
            continue
        counts[_unread_row_key(definition)] += 1
    return dict(counts)


def _reminder_row(task: ReminderTask) -> UnifiedRow:
    bucket = _reminder_bucket(task)
    return UnifiedRow(
        kind="reminder",
        row_id=f"reminder:{task.id}",
        title=task.title,
        schedule_summary=_humanize_schedule(task),
        next_run_at=task.next_run_at,
        owner_id=task.owner_id,
        owner_label=owner_display_label(task.owner_id),
        transfer_state=task.transfer_state,
        bucket=bucket,
        glyph=_reminder_glyph(task, bucket),
        unread_count=0,
        search_blob=f"{task.title}\n{task.body or ''}",
        source_row=task,
    )


def _definition_row(
    definition: dict[str, Any], unread_by_row_key: dict[str, int]
) -> UnifiedRow:
    bucket = _definition_bucket(definition)
    local_id = str(definition.get("id") or "")
    owner_id = str(definition.get("owner_id") or "local")
    name = str(definition.get("name") or local_id)
    return UnifiedRow(
        kind="definition",
        row_id=f"definition:{local_id}",
        title=name,
        schedule_summary=_definition_at_label(definition.get("schedule") or {}),
        next_run_at=_parse_iso(definition.get("next_run_at")),
        owner_id=owner_id,
        owner_label=owner_display_label(owner_id),
        transfer_state=definition.get("transfer_state"),
        bucket=bucket,
        glyph=_definition_glyph(definition, bucket),
        unread_count=unread_by_row_key.get(_unread_row_key(definition), 0),
        search_blob=f"{name}\n{_definition_question_text(definition)}",
        source_row=definition,
    )


def build_unified_rows(
    reminders: list[ReminderTask],
    definitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    local_definitions: list[dict[str, Any]] | None = None,
) -> list[UnifiedRow]:
    """Adapt reminders + automation definitions into one unified row list.

    Args:
        reminders: Every owner's reminders (e.g. from
            ``SchedulingService.list_tasks(owner_id=None)``, filtered to
            real `ReminderTask` rows -- briefing/watchlist projections
            stay out of the unified list per plan ruling 1).
        definitions: Local + server-mirrored ``recurring_question``
            definition rows (the existing Automations-tab merge
            precedent: `_load_local_automations` + `_load_server_automations`).
        results: One all-owners `list_automation_results(owner_id=None)`
            listing, used only to derive `UnifiedRow.unread_count`.
        local_definitions: The FULL local definitions table
            (`list_automation_definitions(owner_id=None)`, the same input
            the Results tab's own index uses), for result->definition
            resolution ONLY -- never a source of rows. Final review F2:
            the display merge above deliberately EXCLUDES every local row
            that carries a ``server_id``, so a definition transferred to
            the server is a key in NEITHER of the merge's two id spaces;
            its pre-transfer, locally-produced results resolved to
            nothing and dropped out of the unread count, hiding the
            rail's `Mark all read` button while the Results tab's badge
            still counted them. Resolution indexes these rows AND the
            display rows, since a server definition with no local mirror
            row is absent from the local table. Defaults to
            ``definitions`` -- every caller with no separate local
            listing behaves exactly as before.

    Returns:
        One `UnifiedRow` per reminder and per definition, unsorted and
        unfiltered (`filter_rows`/`sort_rows` do that).
    """
    definitions_by_id = index_definitions_by_id(
        [*local_definitions, *definitions]
        if local_definitions is not None
        else definitions
    )
    unread_by_row_key = _unread_counts_by_row_key(results, definitions_by_id)
    rows = [_reminder_row(task) for task in reminders]
    # PR-4 ruling 1 (was: `family == "recurring_question"` only, Qodo
    # MEDIUM): every definition family renders now, not just
    # `recurring_question`. That original filter existed because the
    # Automations tab was the all-families home and this list's own
    # actions/editors only understood `recurring_question` -- but PR-4
    # retires that tab, so filtering here would make every `agent_task`
    # (or other unrecognized-family) definition permanently invisible,
    # with no surface left at all. `_definition_bucket`/`_definition_
    # glyph`/`_definition_at_label`/`_definition_question_text` are
    # already family-agnostic (read `lifecycle`/`schedule`/`input` generi-
    # cally, degrading to honest defaults for a shape they don't
    # recognize -- verified against the real server fixture, which gives
    # an `agent_task` row the same `schedule`/`lifecycle` shape as a
    # `recurring_question` one). `DefinitionDetail` already has a
    # read-only `_UNSUPPORTED_FAMILY_NOTE` fallback for a non-`recurring_
    # question` row (built for the Automations tab, reused verbatim
    # here) -- viewing is universal, editing stays `recurring_question`-
    # only.
    rows.extend(_definition_row(d, unread_by_row_key) for d in definitions)
    return rows


# ---------------------------------------------------------------------------
# Filter + sort (plan ruling 5)
# ---------------------------------------------------------------------------


def filter_rows(rows: list[UnifiedRow], *, chip: Chip, query: str) -> list[UnifiedRow]:
    """Narrow ``rows`` to one chip's bucket set, then an in-memory search.

    Args:
        rows: The full unified row list (any order).
        chip: ``"all"`` (Active + Paused; Completed excluded by design --
            spec SS3), ``"active"``, ``"paused"``, or ``"completed"``.
        query: Case-insensitive substring matched against each row's
            ``search_blob`` (title + question/body). Blank/whitespace-only
            matches everything.

    Returns:
        The matching rows, in their input order.
    """
    buckets = _CHIP_BUCKETS.get(chip, _CHIP_BUCKETS["all"])
    needle = query.strip().lower()
    return [
        row
        for row in rows
        if row.bucket in buckets and (not needle or needle in row.search_blob.lower())
    ]


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)
_FAR_PAST = datetime.min.replace(tzinfo=timezone.utc)


def _next_run_sort_key(row: UnifiedRow) -> datetime:
    return _as_utc(row.next_run_at) if row.next_run_at is not None else _FAR_FUTURE


def _recency_of(row: UnifiedRow) -> datetime | None:
    """The "last activity" timestamp `sort_rows` uses for Paused/Completed.

    Ruling 5 names ``last_run_at``/``updated_at`` -- the reminder and
    definition field that carries the same meaning on each primitive.
    """
    if row.kind == "reminder":
        task = row.source_row
        assert isinstance(task, ReminderTask)
        return task.last_run_at
    definition = row.source_row
    assert isinstance(definition, dict)
    return _parse_iso(definition.get("updated_at"))


def _recency_sort_key(row: UnifiedRow) -> datetime:
    recency = _recency_of(row)
    return _as_utc(recency) if recency is not None else _FAR_PAST


def sort_rows(rows: list[UnifiedRow], chip: Chip) -> list[UnifiedRow]:
    """Order ``rows`` per plan ruling 5.

    Args:
        rows: Rows already narrowed to one chip (`filter_rows`'s output),
            though any row list is accepted.
        chip: Which chip's order to apply. ``"active"`` sorts by
            ``next_run_at`` ascending (``None`` last). ``"paused"``/
            ``"completed"`` sort by recency descending (most-recent-first,
            ``None`` last). ``"all"`` keeps Active rows first in Active's
            own order, then appends Paused rows in Paused's own order.

    Returns:
        A new, ordered list (input list is not mutated).
    """
    if chip == "active":
        return sorted(rows, key=_next_run_sort_key)
    if chip in ("paused", "completed"):
        return sorted(rows, key=_recency_sort_key, reverse=True)
    active_rows = sorted(
        (row for row in rows if row.bucket == "active"), key=_next_run_sort_key
    )
    paused_rows = sorted(
        (row for row in rows if row.bucket == "paused"),
        key=_recency_sort_key,
        reverse=True,
    )
    return active_rows + paused_rows
