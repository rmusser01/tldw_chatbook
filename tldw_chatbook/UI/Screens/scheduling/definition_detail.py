"""Per-definition detail pane (schedules-redesign PR-1, Task 4).

Introduced as the Automations tab's FIRST per-row detail widget -- that
tab had no field-level rendering of a selected `automation_definition` at
all (only the definitions `DataTable`'s five columns and the audit-trail
history pane; see `redesign-pr1-survey.md` section 1's "no per-row detail
widget" finding). redesign PR-4 task 5 retired that tab; this pane lives
on as the queue's definition-row detail, the sibling of `TaskDetail` in
the same detail pane. It uses the same `DetailValueRow`/`DetailGroup` row
grammar Task 3 used to regrammar `TaskDetail` -- Details/Frequency/
History groups. PR-1 shipped
every row read-only (`affordance` at its `False` default); schedules-
redesign PR-3, task 4 wires in-pane editing onto the Details/Frequency
rows (Model/Generation/Finding policy/Sources/Notifications editable,
Repeat/At/Timezone gated by schedule kind, and all of them gated on the
`recurring_question` family this pane can actually author) plus a header
Pause/Resume affordance -- see the `DefinitionDetail` class docstring
below for the row-editing/lifecycle-toggle design. `Runs on`/History rows
stay read-only.

Like `task_detail.py` and `results_tab.py`, this is a leaf module:
`schedules_workbench.py` imports FROM here (`DefinitionDetail`, plus
`automation_execution_target_label`/`automation_name_cell`, moved here
from `schedules_workbench.py` so this module's own formatters can reuse
their owner-label logic without a circular import -- `schedules_workbench`
re-exports both names so its own DataTable render call site and the
pre-existing `test_execution_target_label_matrix` test, which imports
`automation_execution_target_label` from `schedules_workbench`, both keep
working unchanged).

`DefinitionDetail.set_definition` is a pure "paint what I'm given" method,
mirroring `TaskDetail.set_task`'s data-only feed shape: it performs no
I/O. `schedules_workbench.py` does the DB reads (off the event loop, via
`asyncio.to_thread`, same discipline `_load_local_automations` already
established for a `service.db.*` call made from inside a worker
coroutine) and passes the results in.

Values are drawn from BOTH the local `automation_definitions` table
(client-authored, `schedule.cron`/`config.scope`/etc. per
`automation_definition_form.py`'s own payload shape) and the server's raw
list-response dicts (`_load_server_automations` never runs these through
a JSON round-trip, stamping only `owner_id`). The two shapes genuinely
differ, and the final review proved the cost of assuming otherwise: the
server sends `schedule.expression` where this client writes
`schedule.cron`, so every server definition rendered `At: -`, and the
config keys the server simply does not carry (`generation_mode`,
`scope`, `finding_policy`) were being filled in from the CREATE FORM's
defaults and presented as readings. Both are fixed: the cron read accepts
either key, and an absent key renders "Not set". Every formatter here
reads with `.get()`/`isinstance` guards and degrades to an honest
placeholder instead of raising, the same defensive-read idiom
`automation_execution_target_label` already used -- honest meaning "we
have no value", never a plausible guess.

Escape discipline: every `DetailValueRow` value is safe by construction
(Task 1: the row's `Static` is built with `markup=False` and the value is
wrapped in a literal `rich.text.Text`, proven never to interpret
markup). The one field NOT going through `DetailValueRow` -- the question
card, a plain `Static` -- is built with `markup=False` here too, for the
same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Select, Static

from ....Scheduling.schedule_input_parsing import (
    example_run_at_text,
    parse_forgiving_datetime,
)
from ....Scheduling.events import (
    DefinitionFieldEditRequested,
    DefinitionLifecycleToggleRequested,
    DefinitionOwnerActionRequested,
    DefinitionRunNowRequested,
    DuplicateDefinitionRequested,
    ViewDefinitionAuditRequested,
    ViewDefinitionResultsRequested,
)
# PR-3 task 5: same lock/failed-state gating `task_detail.py`'s own
# Runs-on row uses (survey §3) -- see that module's import comment for
# why this is a plain, no-new-boot-cost constant import.
from ....Scheduling.db.scheduled_tasks_db import IN_FLIGHT_TRANSFER_STATES
from ....Widgets.detail_value_row import DetailGroup, DetailValueRow
from .forms.automation_definition_form import (
    _FINDING_POLICY_OPTIONS,
    _GENERATION_MODE_OPTIONS,
    _SCOPE_SOURCE_CHECKBOXES,
)
# PR-3 task 4: the Frequency row editors reuse the create/edit modal's own
# preset<->cron mapping and timezone-option builder verbatim (never
# re-derived) -- same precedent `task_detail.py`'s Task 3 reminder rows
# already established; `reminder_form.py` is already part of this same
# lazy-loaded Scheduling-screen import chain (ADR-097).
from .forms.reminder_form import (
    DEFAULT_TIME_OF_DAY,
    ReminderForm,
    cron_to_preset,
    parse_time_of_day,
    preset_to_cron,
    timezone_options,
)
from .task_detail import (
    _TRANSFER_STATE_ROW_LABELS,
    _format_timezone,
    definition_cron_expression,
    owner_display_label,
)
# `_definition_at_label`/`_definition_question_text`/`_parse_iso` moved to
# `unified_rows.py` (redesign PR-2 Task 1 -- that pure module needs them
# too and cannot import a Textual-heavy module without dragging Textual in
# as a side effect); imported back here unchanged.
from .unified_rows import _definition_at_label, _definition_question_text, _parse_iso

#: v1 scope guard (`save_definition._reject_unsupported_family`): this pane
#: only ever edits `recurring_question` definitions. A bare string literal
#: here (matching `AutomationDefinitionForm._FAMILY`'s own value) rather
#: than importing that name -- it is a private constant of a sibling UI
#: module for a different surface (the full create/edit modal), and one
#: literal is simpler than a cross-module import for it.
_FAMILY = "recurring_question"

#: Stable ids for the row editors (PR-3 task 4), routed on by
#: `on_select_changed`/`on_input_submitted`/`on_button_pressed` -- same
#: "filter on `.id`" idiom `task_detail.py`'s Task 3 Frequency rows use.
_MODEL_EDITOR_ID = "scheduling-automation-detail-model-editor"
_GENERATION_EDITOR_ID = "scheduling-automation-detail-generation-editor"
_FINDING_POLICY_EDITOR_ID = "scheduling-automation-detail-finding-policy-editor"
_NOTIFICATIONS_EDITOR_ID = "scheduling-automation-detail-notifications-editor"
_REPEAT_EDITOR_ID = "scheduling-automation-detail-repeat-editor"
_AT_EDITOR_ID = "scheduling-automation-detail-at-editor"
_TIMEZONE_EDITOR_ID = "scheduling-automation-detail-timezone-editor"
_SOURCES_EDITOR_ID = "scheduling-automation-detail-sources-editor"
_SOURCES_APPLY_ID = "scheduling-automation-detail-sources-apply"
#: PR-3 task 5: the owner-row transfer dropdown and its proactively-shown
#: Cancel/Retry mini-bar -- same stable-id routing idiom as the rest of
#: this file's row editors.
_RUNS_ON_EDITOR_ID = "scheduling-automation-detail-runs-on-editor"
_RUNS_ON_CANCEL_ID = "scheduling-automation-detail-runs-on-cancel"
_RUNS_ON_RETRY_ID = "scheduling-automation-detail-runs-on-retry"
_SOURCES_CHECKBOX_IDS: dict[str, str] = {
    value: f"scheduling-automation-detail-sources-{value}"
    for _label, value in _SCOPE_SOURCE_CHECKBOXES
}

#: Repeat's "custom" preset has no single-value edit target here (same
#: rule task_detail.py's Task 3 Repeat row already documents) -- shown so
#: the row's CURRENT value round-trips (`Select` requires the initial
#: value be among its options), but selecting it as a NEW target is
#: refused with this copy rather than silently doing nothing (ruling 2).
_REPEAT_CUSTOM_REFUSAL = "Use the full Edit form to set a custom cron expression."

#: Schedule kinds whose `At` row has a single-value edit target, and which
#: payload field that target is (Qodo finding 8). A definition's schedule
#: `kind` is one of `schedule_compute.py`'s five (`one_time`, `interval`,
#: `daily`, `weekly`, `cron`) -- not the two the pane originally assumed.
#: `At` used to be editable for every non-`cron` kind and always committed
#: `{"kind": "one_time", "run_at": ...}`, so editing the time on a `daily`
#: automation silently CONVERTED it to a one-shot and dropped its
#: `time_of_day`/`weekday`. Kind is never converted by a row edit now:
#: `one_time` edits its `run_at`, `daily`/`weekly` edit their shared
#: `time_of_day`, and `interval`/`cron` (and any unknown kind) leave the
#: row read-only, exactly as `task_detail._configure_frequency_
#: editability` gates the reminder pane's own Repeat/At/Timezone trio.
_AT_EDIT_FIELD_BY_KIND: dict[str, str] = {
    "one_time": "run_at",
    "daily": "time_of_day",
    "weekly": "time_of_day",
}

#: Refusal copy for the Automations pane's family gate (Qodo finding 11).
#: `save_definition` only authors `recurring_question` (`_reject_
#: unsupported_family`), and every row-edit payload this module builds
#: hard-codes that family -- so offering editors on, say, an `agent_task`
#: row invited an edit that either bounced or, worse, re-normalized the
#: row through the wrong family's rules. The create form already refuses
#: the same way; this is that refusal, stated in the pane.
_UNSUPPORTED_FAMILY_NOTE = (
    "This automation isn't a recurring question — its fields are "
    "read-only here."
)

#: The header Pause/Resume button's two reachable actions (this pane never
#: offers Archive -- not named anywhere in the task-4 brief, and a
#: destructive third state deserves its own considered affordance, not a
#: bolt-on to a "toggle" button). Deliberately duplicates just these two
#: entries of `SchedulingService._LIFECYCLE_ACTIONS` (a private constant
#: of the service module) rather than importing it -- a two-entry mapping
#: this stable is a low-risk, display-only duplication: if it ever drifted
#: the worst case is a stale button label until the next background
#: refresh repaints from a real DB read, never a wrong write (the actual
#: persisted value always comes from the service's own mapping).
_LIFECYCLE_TOGGLE_RESULTS: dict[str, str] = {"pause": "paused", "resume": "configured"}


def _definition_edit_payload(
    definition: dict[str, Any], **field_changes: Any
) -> dict[str, Any]:
    """Build a `SchedulingService.save_definition` payload for ONE row's edit.

    `SchedulingService._merge_definition_payload` (verified empirically
    against the real service -- no existing test exercises a single-field
    definition edit, since the create/edit MODAL always sends a full
    payload) does NOT behave the way its own docstring implies for a
    payload this small:

    - `description`/`visibility_policy`/`approval_policy` ARE seeded from
      the stored row unconditionally.
    - `config`/`input`/`notification_policy` are merged ONE LEVEL DEEP,
      but ONLY when the payload's own top-level key is present as a dict
      -- an ABSENT key is dropped from the merge entirely, not defaulted
      from storage. A payload missing `notification_policy` altogether
      silently WIPES a previously configured policy to `{}` on save (an
      empty dict `{}` is safe -- `{**stored, **{}}` preserves everything
      -- it is a MISSING key that is dangerous).
    - `name`, `family`, and `schedule` get NO merge treatment at all. A
      payload missing `family` is rejected outright by `_reject_
      unsupported_family` (run on the RAW, pre-merge payload); missing
      `name` fails the preview's own required-field check; a missing
      `schedule` silently drops the definition's schedule.

    So every row-edit payload here resends `family` + `name` verbatim,
    the definition's CURRENT `schedule` dict (unless the edit IS a
    Frequency row, which passes its own already-modified whole dict via
    `field_changes["schedule"]` -- the task-4 brief's "whole-dict resend"
    rule), and an empty-or-partial dict for each of `config`/`input`/
    `notification_policy` so the one-level merge preserves whatever
    subkey this particular edit does not touch.
    """
    schedule = definition.get("schedule")
    payload: dict[str, Any] = {
        "family": _FAMILY,
        "name": definition.get("name"),
        "schedule": dict(schedule) if isinstance(schedule, dict) else {},
        "config": {},
        "input": {},
        "notification_policy": {},
    }
    payload.update(field_changes)
    return payload


#: Absent key -> honest placeholder (final review F2). A definition this
#: client did not author carries none of the create-form's config keys,
#: and substituting that form's DEFAULTS here presented a guess as a
#: reading: a server definition configured `high_confidence_only`
#: rendered "Finding policy: Balanced findings" in a read-only pane the
#: user has no reason to distrust. Defaults belong to the create path.
_NOT_SET = "Not set"

#: History rows for a server-owned definition (final review F3).
#: `automation_runs` is a local-only table with exactly one writer (local
#: dispatch, `scheduler/handlers/automation_handler.py`), so its counts
#: are structurally zero for a server row -- and "Never run"/"0" sat
#: beside a run-history pane listing the server's real audit trail.
#: Mirrors `_load_automation_history`'s own honest local-gap copy.
_SERVER_LAST_RUN = "Kept on the server — see Run history"
_SERVER_RUN_COUNT = "Kept on the server"
_PENDING_SYNC_HISTORY = "Not synced to the server yet"

#: History rows after a failed count read (final review F14): a DB error
#: must not be indistinguishable from a genuinely empty history. Same
#: wording shape as `_load_automation_history`'s read-failure notice.
_HISTORY_READ_FAILED = "Couldn't load — see the log"

#: value -> label reverse lookups, reusing the SAME vocabulary the
#: create/edit form already presents (`automation_definition_form.py`)
#: rather than inventing a second wording for the same underlying value.
_GENERATION_MODE_LABELS: dict[str, str] = {
    value: label for label, value in _GENERATION_MODE_OPTIONS
}
_FINDING_POLICY_LABELS: dict[str, str] = {
    value: label for label, value in _FINDING_POLICY_OPTIONS
}
_SCOPE_SOURCE_LABELS: dict[str, str] = {
    value: label for label, value in _SCOPE_SOURCE_CHECKBOXES
}


def automation_execution_target_label(definition: dict[str, Any]) -> str:
    """Render one definition's per-task execution target (ADR-077 AC#7).

    Moved here from `schedules_workbench.py` (Task 4) so this module's
    own "Model" row can call it without a circular import;
    `schedules_workbench` re-exports the name for its DataTable render
    call site and the pre-existing test that imports it from there.

    ``input.provider``/``input.model`` ride the definition payload and the
    server executor honors them. The column shows what was PINNED here:
    when neither key is set the label is ``auto`` -- the definition pins
    nothing, and the server resolves the run target from its own
    automation-config executor defaults (``[Scheduled_Tasks_Automation]
    executor_provider``/``executor_model``) falling back to the server
    default. Those layers live in server config, not the payload, so
    ``auto`` is the honest client-side rendering, not a claim about which
    server layer actually won.

    Args:
        definition: One row from the server's definition list, as the raw
            dict the scheduling server client returns.

    Returns:
        A short cell label: ``provider/model``, either part alone, or
        ``auto`` when neither is set.
    """
    source = definition.get("input") if isinstance(definition.get("input"), dict) else {}
    provider = str(source.get("provider") or "").strip()
    model = str(source.get("model") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    if provider:
        return provider
    if model:
        return model
    return "auto"


def _definition_owner_label(definition: dict[str, Any]) -> str:
    """Owner label for a definition row.

    ``"This device"`` for a local row, the server id for a server-scoped
    one, and ``"<server id> · pending sync"`` for one authored offline
    that has not reached the server yet -- the same vocabulary
    `automation_name_cell` wraps into its Name-cell prefix, factored out
    here so the detail pane's "Runs on" row and the table cell can never
    drift apart.

    The owner half is `task_detail.owner_display_label`, shared with the
    reminder pane's own "Runs on" row (final review F6/F7); only the
    definition-dict read and the pending-sync suffix live here.
    """
    label = owner_display_label(definition.get("owner_id") or "local")
    if definition.get("pending_sync") and label != "This device":
        label = f"{label} · pending sync"
    return label


def automation_name_cell(definition: dict[str, Any]) -> str:
    """Name cell for the merged local+server Automations list (task-5 fix round).

    Moved here from `schedules_workbench.py` (Task 4) alongside
    `_definition_owner_label`, which this now delegates to.

    31713 AC#1: this used to be a bracket PREFIX shown even for a local
    row (`"[This device] <name>"`), while a reminder row's own queue
    title (`task_detail._queue_owner_suffix`) shows nothing at all for a
    local row and only a parenthetical SUFFIX when server-scoped -- two
    formats, and a local row's own labeling behavior differed between the
    primitives. Now matches that reminder convention exactly: nothing for
    a local definition, ``" (server: <id>)"`` for a server-scoped one
    (the same wording `_queue_owner_suffix` uses, not just the same
    shape), with `_definition_owner_label`'s own ``"· pending sync"``
    qualifier folded into the same parenthetical when it applies.

    Args:
        definition: One merged row (local DB dict or server API dict --
            both carry `owner_id` and `name`, confirmed against the real
            server fixture `automation_definition_list.json`).

    Returns:
        The bare name for a local row, ``"<name> (server: <server id>)"``
        for a server-scoped one, and ``"<name> (server: <server id> ·
        pending sync)"`` for one authored offline that has not reached
        the server yet.
    """
    name = str(definition.get("name") or definition.get("id") or "")
    label = _definition_owner_label(definition)
    if label == "This device":
        return name
    return f"{name} (server: {label})"


def _definition_transfer_suffix(definition: dict[str, Any]) -> str:
    """Definition-dict counterpart of `task_detail._transfer_row_suffix`.

    Reuses the SAME state->label vocabulary (`_TRANSFER_STATE_ROW_LABELS`)
    since a definition's `transfer_state` column shares the schedules-
    handoff spec §6 transfer state machine with a reminder's -- only the
    read (attribute vs. dict key) differs, so the label table itself is
    imported rather than duplicated.
    """
    label = _TRANSFER_STATE_ROW_LABELS.get(definition.get("transfer_state") or "")
    return f" ({label})" if label else ""


def _definition_generation_label(config: dict[str, Any]) -> str:
    """'Generation' row value (Details group): `config.generation_mode`,
    labeled with the same wording the create/edit form's Select uses;
    `"Not set"` when the payload carries no `generation_mode` at all
    (final review F2 -- this used to substitute the form's own default)."""
    mode = config.get("generation_mode") if isinstance(config, dict) else None
    if not mode:
        return _NOT_SET
    mode = str(mode)
    return _GENERATION_MODE_LABELS.get(mode, mode)


def _definition_finding_policy_label(definition: dict[str, Any]) -> str:
    """'Finding policy' row value (Details group): the top-level
    `finding_policy.preset` column (survey §4's authoritative source --
    distinct from the create/edit payload's `config.finding_policy`,
    which the service copies into this column at save time); `"Not set"`
    when absent (final review F2)."""
    finding_policy = definition.get("finding_policy")
    preset = finding_policy.get("preset") if isinstance(finding_policy, dict) else None
    if not preset:
        return _NOT_SET
    return _FINDING_POLICY_LABELS.get(preset, str(preset))


def _definition_sources_label(config: dict[str, Any]) -> str:
    """'Sources' row value (Details group): `config.scope`, joined plurals
    for an explicit source list, or the all-library sentence for the
    default scope mode; `"Not set"` when the payload carries no `scope`
    at all (final review F2 -- a server definition's config has no
    `scope`, and claiming "All searchable library" for it was a guess)."""
    scope = config.get("scope") if isinstance(config, dict) else None
    if not isinstance(scope, dict) or not scope:
        return _NOT_SET
    if scope.get("mode") != "sources":
        return "All searchable library"
    sources = scope.get("sources") or []
    labels = [_SCOPE_SOURCE_LABELS.get(str(item), str(item)) for item in sources]
    return ", ".join(labels) if labels else "None selected"


def _definition_notifications_label(definition: dict[str, Any]) -> str:
    """'Notifications' row value (Frequency group), spec §5's fourth
    Frequency row -- missing until final review F8.

    Reads `notification_policy`, which arrives in two shapes: this
    client's writer emits booleans from the form's one "Notify me about
    results" checkbox (`automation_definition_form.py`), while the real
    server fixture returns per-outcome channel strings
    (`{"on_success": "silent", "on_failure": "toast"}`). Both render;
    an absent policy says so instead of guessing.
    """
    policy = definition.get("notification_policy")
    if not isinstance(policy, dict) or not policy:
        return _NOT_SET
    on_success = policy.get("on_success")
    on_failure = policy.get("on_failure")
    if isinstance(on_success, bool) or isinstance(on_failure, bool):
        return "On" if (on_success or on_failure) else "Off"
    parts = [
        f"{value} on {outcome}"
        for outcome, value in (("success", on_success), ("failure", on_failure))
        if value
    ]
    return " · ".join(parts) if parts else _NOT_SET


def _definition_repeat_label(schedule: dict[str, Any]) -> str:
    """'Repeat' row value (Frequency group): "Recurring" or "One-time" --
    the definition-schedule counterpart of `task_detail._humanize_schedule_
    kind`, which reads a `ReminderTask`'s `ScheduleKind` enum a definition's
    plain `schedule["kind"]` string (`"cron"`/`"one_time"`) does not share."""
    kind = schedule.get("kind") if isinstance(schedule, dict) else None
    if kind == "cron":
        return "Recurring"
    if kind == "one_time":
        return "One-time"
    return "-"


def _definition_timezone_label(schedule: dict[str, Any]) -> str:
    """'Timezone' row value (Frequency group), reusing `_format_timezone`
    for the same per-kind timezone source `_definition_at_label` reads."""
    if not isinstance(schedule, dict):
        return "UTC"
    kind = schedule.get("kind")
    if kind == "cron":
        return schedule.get("timezone") or "UTC"
    if kind == "one_time":
        run_at = schedule.get("run_at")
        dt = _parse_iso(run_at) if run_at else None
        return _format_timezone(dt) if dt is not None else "UTC"
    return "UTC"


def _definition_history_labels(
    definition: dict[str, Any],
    *,
    run_count: int,
    last_run: dict[str, Any] | None,
    history_error: bool,
) -> tuple[str, str]:
    """(Last run, Run count) values -- honest for every owner (F3/F14).

    `automation_runs` is local-only: it has exactly one writer (local
    dispatch) and no server mirror, so a server-owned definition's local
    counts are structurally zero. Rendering them as `Never run` / `0`
    contradicted the run-history pane beside it, which was listing that
    same definition's server audit trail at the time -- and contradicted
    the User Guide, which already claimed the honest behaviour.
    """
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    if is_server_scoped_owner(definition.get("owner_id")):
        if definition.get("pending_sync"):
            # Never reached the server, and never ran locally either.
            return _PENDING_SYNC_HISTORY, _PENDING_SYNC_HISTORY
        return _SERVER_LAST_RUN, _SERVER_RUN_COUNT
    if history_error:
        return _HISTORY_READ_FAILED, _HISTORY_READ_FAILED
    return _definition_last_run_label(last_run), str(run_count)


def _definition_last_run_label(last_run: dict[str, Any] | None) -> str:
    """'Last run' row value (History group): the most recent
    `automation_runs` row's status plus its timestamp -- the same compact
    slice-and-replace timestamp idiom `task_detail.format_run_history`
    uses for a reminder's run history."""
    if not last_run:
        return "Never run"
    status = str(last_run.get("status") or last_run.get("outcome") or "?")
    when = str(
        last_run.get("ended_at")
        or last_run.get("started_at")
        or last_run.get("created_at")
        or ""
    )
    when = when[:16].replace("T", " ") if when else "?"
    return f"{status} — {when}"


class DefinitionDetail(Vertical):
    """Per-definition detail pane (the queue's definition-row detail).

    schedules-redesign PR-3, task 4 wires in-pane editing onto the
    Details/Frequency rows PR-1 left read-only, plus the header Pause/
    Resume affordance (`SchedulingService.set_definition_lifecycle`'s
    first UI caller). `set_definition` is still the single PAINT feed
    point -- called by `SchedulesWorkbench` after a definitions-table row
    highlight, with the DB-derived counts/last-run already fetched off
    the event loop -- and remains pure (no I/O). A row's edit and the
    lifecycle toggle instead POST a `Scheduling.events` message
    (`DefinitionFieldEditRequested`/`DefinitionLifecycleToggleRequested`)
    the workbench handles, mirroring `TaskDetail`'s own Task 3 Frequency
    rows exactly -- this widget still performs no I/O of its own.

    `Run count` stays permanently read-only. `Runs on` gets its own
    owner-transfer affordance (survey §7, PR-3 task 5). `Unread results`
    is activatable (redesign PR-4, task 2): it is the live replacement
    for the old "See Results tab" pointer, pushing a `ResultsTab` scoped
    to this definition. `Last run` is activatable too (redesign PR-4,
    task 3): it is the live replacement for the retired Automations-tab
    third pane, pushing a `definition_audit_view.DefinitionAuditView`
    scoped to this definition -- see `ViewDefinitionResultsRequested`/
    `ViewDefinitionAuditRequested` and `on_detail_value_row_activated`
    below. A header `Run now` button sits beside Pause/Resume (redesign
    PR-4, task 3, ruling 2): the retired Automations-tab `r` key's live
    replacement, posting `DefinitionRunNowRequested`.

    task-31823: a second row (`Duplicate` / `View runs` / `View
    results`) delivers the rest of spec §5's deferred kebab list.
    `View runs`/`View results` are plain shortcuts onto the SAME
    `ViewDefinitionAuditRequested`/`ViewDefinitionResultsRequested`
    messages the `Last run`/`Unread results` rows already post --
    unconditional, like those rows. `Duplicate` posts
    `DuplicateDefinitionRequested`, gated the same way Pause/Resume is
    (`_refresh_duplicate_button`).
    """

    def __init__(self, *args, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._question_static: Static | None = None
        self._runs_on_row: DetailValueRow | None = None
        self._model_row: DetailValueRow | None = None
        self._generation_row: DetailValueRow | None = None
        self._finding_policy_row: DetailValueRow | None = None
        self._sources_row: DetailValueRow | None = None
        self._repeat_row: DetailValueRow | None = None
        self._at_row: DetailValueRow | None = None
        self._timezone_row: DetailValueRow | None = None
        self._notifications_row: DetailValueRow | None = None
        self._last_run_row: DetailValueRow | None = None
        self._run_count_row: DetailValueRow | None = None
        self._unread_row: DetailValueRow | None = None
        # PR-3 task 4:
        self._definition: dict[str, Any] | None = None
        self._pause_resume_button: Button | None = None
        #: redesign PR-4, task 3: the retired Automations-tab `r` key's
        #: live replacement (ruling 2 -- a button, not a new global key).
        self._run_now_button: Button | None = None
        #: task-31823: spec §5 kebab's `Duplicate` action -- gated the
        #: same way Pause/Resume already is (`_refresh_duplicate_button`).
        self._duplicate_button: Button | None = None
        self._why_static: Static | None = None
        #: Cached from `set_lifecycle_lock` (never re-derived here) --
        #: same one-source-of-truth rule survey §8 documents for
        #: `TaskDetail`. `None` means unlocked.
        self._lifecycle_lock_reason: str | None = None
        #: Threaded in by the workbench's `set_definition` call site, same
        #: as `TaskDetail.set_task`'s own `known_timezones` param -- empty
        #: by default so every pre-task-4 caller/test keeps working.
        self._known_timezones: Sequence[str] = ()
        #: PR-3 task 5: same threaded-in-by-the-workbench shape as
        #: `_known_timezones` above, for the Runs-on row's owner-picker.
        self._runs_on_options: Sequence[tuple[str, str]] = ()
        #: PR-3 task 5: the Runs-on row's proactive Cancel/Retry
        #: affordances -- see `task_detail.py`'s own
        #: `_runs_on_cancel_button`/`_runs_on_retry_button` for the full
        #: rationale (shared design, duplicated per-pane like every
        #: other small helper in this file's own Frequency-row editors).
        self._runs_on_cancel_button: Button | None = None
        self._runs_on_retry_button: Button | None = None
        #: PR-3 task 5 fix round 1 (finding 2): see `task_detail.py`'s
        #: own `_runs_on_transfer_errors`.
        self._runs_on_transfer_errors: list[str] = []
        #: Final review I2: see `TaskDetail._editing_task_id` -- the
        #: definition id an open row editor was opened AGAINST, captured
        #: at `begin_edit` time and validated at commit time by
        #: `_editing_definition`. Never reset by a repaint.
        self._editing_definition_id: str | None = None
        #: Qodo finding 11: why every row is read-only right now, or
        #: `None`. Set from the painted definition's `family` and shown
        #: in the same `#scheduling-automation-detail-why` Static the
        #: transfer lock uses (`_refresh_why_note`).
        self._family_note: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the empty pane skeleton (populated by `set_definition`).

        Returns:
            The static children; value rows are mounted per definition.
        """
        yield Static(
            # task-31710 AC#1: matches this primitive's own form title
            # ("New/Edit Recurring Question", automation_definition_form.py)
            # and the Create chooser's "Recurring question…" button --
            # "Definition"/"Automation" were two more names for the same
            # thing.
            "Recurring Question Detail",
            id="scheduling-automation-detail-header",
            classes="scheduling-column-title",
        )
        yield Static(
            "Select a recurring question to see its details.",
            id="scheduling-automation-detail-empty-state",
        )
        with Vertical(id="scheduling-automation-detail-body"):
            # PR-3 task 4: the header Pause/Resume affordance --
            # `set_definition_lifecycle`'s first UI caller. Archive is
            # deliberately not offered here (see `_LIFECYCLE_TOGGLE_
            # RESULTS`'s comment); the label/tooltip are set by
            # `_refresh_lifecycle_button`, called from `set_definition`.
            with Horizontal(id="scheduling-automation-detail-lifecycle"):
                self._pause_resume_button = Button(
                    "Pause",
                    id="scheduling-automation-pause-resume",
                    variant="warning",
                    classes="detail-lifecycle-button",
                )
                yield self._pause_resume_button
                # redesign PR-4, task 3: the retired Automations-tab `r`
                # key's live replacement (ruling 2 -- a button here, no
                # new global keybinding). Unconditionally enabled/never
                # family-gated -- `_run_automation_now` itself is
                # family-agnostic (it only routes by owner), the same as
                # the tab's own `r` key was.
                self._run_now_button = Button(
                    "Run now",
                    id="scheduling-automation-run-now",
                    variant="primary",
                    classes="detail-lifecycle-button",
                )
                yield self._run_now_button
            # task-31823: spec §5's kebab list (Run now · Duplicate ·
            # View runs · View results · Edit in full… · Delete/Archive)
            # minus the four already covered elsewhere (Run now above;
            # Edit in full/Delete/Archive are the `e`/`d` queue-row
            # keybindings, task 3/5's own routing) -- a second compact
            # button row, same "no popup menu" shape `TaskDetail`'s own
            # secondary-actions row uses. `View runs`/`View results`
            # reuse the EXISTING `Last run`/`Unread results` History-row
            # navigation verbatim (same messages, same handlers) --
            # unconditionally enabled, "viewing history is not an edit"
            # (task 2/3's own ruling for those rows). `Duplicate` is
            # gated like Pause/Resume (`_refresh_duplicate_button`).
            with Horizontal(id="scheduling-automation-detail-secondary-actions"):
                self._duplicate_button = Button(
                    "Duplicate",
                    id="scheduling-automation-duplicate",
                    classes="detail-secondary-action-button",
                )
                yield self._duplicate_button
                yield Button(
                    "View runs",
                    id="scheduling-automation-view-runs",
                    tooltip="View this automation's run history.",
                    classes="detail-secondary-action-button",
                )
                yield Button(
                    "View results",
                    id="scheduling-automation-view-results",
                    tooltip="View this automation's results.",
                    classes="detail-secondary-action-button",
                )
            # Visible when the lifecycle button (or an editable row) is
            # locked by an in-flight transfer: keyboard users can't see
            # hover tooltips, so the reason must live in text too
            # (UX-073) -- same idiom `TaskDetail`'s own `#scheduling-
            # transfer-why` uses.
            self._why_static = Static(
                "", id="scheduling-automation-detail-why", classes="follow-why"
            )
            yield self._why_static

            self._question_static = Static(
                "", id="scheduling-automation-detail-question", markup=False
            )
            yield self._question_static

            # `row_key` (final review M12): the per-row identity the
            # workbench's edit worker groups its commit under, so two
            # quick edits on DIFFERENT rows stop cancelling each other.
            # Set on every editable row, nowhere else -- same as
            # `task_detail.py`'s own Frequency/Runs-on rows.
            self._runs_on_row = DetailValueRow(
                "Runs on",
                "-",
                value_id="scheduling-automation-detail-runs-on",
                row_key="runs_on",
            )
            self._model_row = DetailValueRow(
                "Model",
                "-",
                value_id="scheduling-automation-detail-model",
                row_key="model",
            )
            self._generation_row = DetailValueRow(
                "Generation",
                "-",
                value_id="scheduling-automation-detail-generation",
                row_key="generation_mode",
            )
            self._finding_policy_row = DetailValueRow(
                "Finding policy",
                "-",
                value_id="scheduling-automation-detail-finding-policy",
                row_key="finding_policy",
            )
            self._sources_row = DetailValueRow(
                "Sources",
                "-",
                value_id="scheduling-automation-detail-sources",
                row_key="sources",
            )
            # PR-3 task 5: the Runs-on row's own proactive Cancel/Retry
            # affordances -- see `task_detail.py`'s own compose()
            # comment for why these are plain, always-mounted siblings
            # (`.display`-toggled) rather than `begin_edit`-mounted into
            # the row itself.
            self._runs_on_cancel_button = Button(
                "Cancel transfer", id=_RUNS_ON_CANCEL_ID, variant="warning", classes="detail-owner-action-button"
            )
            self._runs_on_retry_button = Button(
                "Retry transfer", id=_RUNS_ON_RETRY_ID, variant="warning", classes="detail-owner-action-button"
            )
            runs_on_actions = Horizontal(
                self._runs_on_cancel_button,
                self._runs_on_retry_button,
                classes="detail-value-row-owner-actions",
            )
            yield DetailGroup(
                self._runs_on_row,
                runs_on_actions,
                self._model_row,
                self._generation_row,
                self._finding_policy_row,
                self._sources_row,
                title="Details",
                id="scheduling-automation-detail-group-details",
            )

            self._repeat_row = DetailValueRow(
                "Repeat",
                "-",
                value_id="scheduling-automation-detail-repeat",
                row_key="cron",
            )
            self._at_row = DetailValueRow(
                "At",
                "-",
                value_id="scheduling-automation-detail-at",
                row_key="run_at",
            )
            self._timezone_row = DetailValueRow(
                "Timezone",
                "-",
                value_id="scheduling-automation-detail-timezone",
                row_key="timezone",
            )
            # Spec §5 gives BOTH columns a Frequency `Notifications` row;
            # the plan's Task 4 text quietly dropped it (final review F8).
            self._notifications_row = DetailValueRow(
                "Notifications",
                "-",
                value_id="scheduling-automation-detail-notifications",
                row_key="notification_policy",
            )
            yield DetailGroup(
                self._repeat_row,
                self._at_row,
                self._timezone_row,
                self._notifications_row,
                title="Frequency",
                id="scheduling-automation-detail-group-frequency",
            )

            # redesign PR-4, task 3: the retired Automations-tab third
            # (run-history/audit) pane's live replacement -- this row's
            # own copy already says "...see Run history" for a
            # server-owned definition (`_SERVER_LAST_RUN`); activating it
            # pushes a `DefinitionAuditView` scoped to this definition.
            # Unconditionally clickable/focusable, same "viewing history
            # is not an edit" rule task 2 established for `_unread_row`
            # below -- a local definition's activation still pushes the
            # view, which renders its OWN honest "not available yet"
            # copy rather than the row being gated/disabled here.
            self._last_run_row = DetailValueRow(
                "Last run",
                "-",
                value_id="scheduling-automation-detail-last-run",
                affordance=True,
                can_focus=True,
                row_key="view_runs",
            )
            self._run_count_row = DetailValueRow(
                "Run count", "-", value_id="scheduling-automation-detail-run-count"
            )
            # redesign PR-4, task 2: the retired "See Results tab" pointer
            # (a dangling string once the tab bar goes away, survey
            # :734-736) becomes THIS row's live activation instead of a
            # separate row -- the count and the destination are the same
            # fact, so a second row saying "Results -- See Results tab"
            # next to it was redundant, not a second affordance.
            # Unconditionally clickable/focusable (never family-gated
            # like the editable rows above): viewing history is sensible
            # for any definition, including one with zero unread results.
            self._unread_row = DetailValueRow(
                "Unread results",
                "-",
                value_id="scheduling-automation-detail-unread-results",
                affordance=True,
                can_focus=True,
                row_key="view_results",
            )
            yield DetailGroup(
                self._last_run_row,
                self._run_count_row,
                self._unread_row,
                title="History",
                collapsed=True,
                id="scheduling-automation-detail-group-history",
            )

    def set_definition(
        self,
        definition: dict[str, Any] | None,
        *,
        run_count: int = 0,
        last_run: dict[str, Any] | None = None,
        unread_count: int = 0,
        history_error: bool = False,
        known_timezones: Sequence[str] = (),
        runs_on_options: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Paint the pane for `definition` (or the empty state for `None`).

        Pure render -- `run_count`/`last_run`/`unread_count` are already
        fetched (off the event loop) by the caller; this method performs
        no I/O of its own. `history_error` says that fetch FAILED, so the
        History rows say so rather than painting a zero the read never
        proved (final review F14).

        Args:
            definition: The definition row dict, or ``None`` for the empty
                state.
            run_count: Pre-fetched local run count for the History group.
            last_run: Pre-fetched most-recent local run row, if any.
            unread_count: Pre-fetched unread-results count.
            history_error: True when the history fetch failed -- the rows
                render "couldn't load" copy instead of unproven zeros.
            known_timezones: PR-3 task 4 -- zones already used by other
                tasks, passed through to the Timezone row editor's option
                source (`timezone_options`), same param `TaskDetail.
                set_task` already threads for its own Timezone row. Every
                pre-task-4 caller/test omits this and is unaffected.
            runs_on_options: PR-3 task 5 -- the workbench's own
                `_runs_on_options()[0]`, threaded in the same shape
                `known_timezones` already is. Every pre-task-5
                caller/test omits this and is unaffected.
        """
        # Final review I2/I3: a repaint that swaps in a DIFFERENT
        # definition takes the open editor and the inline error with it --
        # `TaskDetail.set_task`'s twin; see `_reset_row_editing`.
        previous = self._definition
        if (definition or {}).get("id") != (previous or {}).get("id"):
            self._reset_row_editing()
        self._definition = definition
        self._known_timezones = known_timezones
        self._runs_on_options = runs_on_options
        empty_state = self.query_one(
            "#scheduling-automation-detail-empty-state", Static
        )
        body = self.query_one("#scheduling-automation-detail-body", Vertical)
        if definition is None:
            empty_state.display = True
            body.display = False
            # PR-3 task 4: stale lock state from a PREVIOUS selection must
            # not survive into a cleared pane -- same discipline `TaskDetail.
            # set_task`'s `None` branch already documents.
            self._lifecycle_lock_reason = None
            self._family_note = None
            self._refresh_why_note()
            # PR-3 task 5: the Runs-on row's Cancel/Retry buttons from a
            # PREVIOUS selection must not stay visible in a cleared pane.
            if self._runs_on_cancel_button is not None:
                self._runs_on_cancel_button.display = False
            if self._runs_on_retry_button is not None:
                self._runs_on_retry_button.display = False
            self._runs_on_transfer_errors = []
            return
        empty_state.display = False
        body.display = True

        assert self._question_static is not None, "set_definition called before mount"
        self._question_static.update(_definition_question_text(definition))

        schedule = (
            definition.get("schedule")
            if isinstance(definition.get("schedule"), dict)
            else {}
        )
        config = (
            definition.get("config") if isinstance(definition.get("config"), dict) else {}
        )

        assert self._runs_on_row is not None, "set_definition called before mount"
        self._runs_on_row.update_value(
            _definition_owner_label(definition) + _definition_transfer_suffix(definition)
        )
        self._model_row.update_value(automation_execution_target_label(definition))
        self._generation_row.update_value(_definition_generation_label(config))
        self._finding_policy_row.update_value(
            _definition_finding_policy_label(definition)
        )
        self._sources_row.update_value(_definition_sources_label(config))

        self._repeat_row.update_value(_definition_repeat_label(schedule))
        self._at_row.update_value(_definition_at_label(schedule))
        self._timezone_row.update_value(_definition_timezone_label(schedule))
        self._notifications_row.update_value(
            _definition_notifications_label(definition)
        )

        last_run_label, run_count_label = _definition_history_labels(
            definition,
            run_count=run_count,
            last_run=last_run,
            history_error=history_error,
        )
        self._last_run_row.update_value(last_run_label)
        self._run_count_row.update_value(run_count_label)
        self._unread_row.update_value(
            _HISTORY_READ_FAILED if history_error else str(unread_count)
        )

        self._family_note = (
            None if definition.get("family") == _FAMILY else _UNSUPPORTED_FAMILY_NOTE
        )
        self._configure_row_editability(schedule)
        self._configure_runs_on_row(definition)
        self._refresh_lifecycle_button()
        self._refresh_duplicate_button()
        self._refresh_why_note()

    # -- In-pane row editing (PR-3 task 4) -----------------------------------

    def _refresh_why_note(self) -> None:
        """Repaint `#scheduling-automation-detail-why` from the two
        reasons that can make this pane read-only.

        The transfer lock wins when both apply -- it is the transient one
        and the one with a user action attached (cancel the move); the
        family note is a standing property of the row. Both are plain
        text in an always-visible Static, never a hover tooltip (UX-073).
        """
        if self._why_static is not None:
            self._why_static.update(self._lifecycle_lock_reason or self._family_note or "")

    def _configure_row_editability(self, schedule: dict[str, Any]) -> None:
        """Wire each editable row's affordance for the CURRENT definition.

        Model/Generation/Finding policy/Sources/Notifications do not
        depend on schedule kind and stay editable. Repeat/Timezone only apply
        to a `"cron"` schedule; `At` applies to the kinds that have a
        single-value time target (`_AT_EDIT_FIELD_BY_KIND` -- `one_time`'s
        `run_at`, `daily`/`weekly`'s `time_of_day`) and stays read-only
        for `interval` and `cron`. Same reasoning `task_detail._configure_
        frequency_editability` documents for the reminder pane's own
        Repeat/At/Timezone trio: editing the "wrong" one for the current
        kind has no sensible target, and here it was worse than useless
        -- the commit hard-coded `kind: "one_time"`, so an `At` edit on a
        `daily` automation converted it to a one-shot (Qodo finding 8).

        A definition of any family but `recurring_question` turns EVERY
        row read-only, Runs-on included (Qodo finding 11): this pane's
        payloads all declare `family=recurring_question`, which the
        service's own scope guard is the only thing standing between an
        `agent_task` row and a re-normalization through the wrong
        family's rules. `_family_note` says so under the header.

        Locked rows (a transfer in flight) keep their affordance ON, in
        contrast: ruling 2 requires activation to still respond with the
        lock reason via `show_error`, and `on_detail_value_row_activated`
        checks `self._lifecycle_lock_reason` before ever opening an
        editor -- so the affordance glyph staying lit is what makes that
        reachable at all (`set_lifecycle_lock` never touches row
        affordance). A family refusal needs no such affordance: it is
        permanent for this row and stated in the always-visible note.
        """
        supported_family = self._family_note is None
        recurring = schedule.get("kind") == "cron"
        for row, editable in (
            (self._model_row, True),
            (self._generation_row, True),
            (self._finding_policy_row, True),
            (self._sources_row, True),
            (self._notifications_row, True),
            (self._repeat_row, recurring),
            (self._at_row, schedule.get("kind") in _AT_EDIT_FIELD_BY_KIND),
            (self._timezone_row, recurring),
        ):
            assert row is not None
            row.affordance = editable and supported_family
            row.can_focus = editable and supported_family

    def _configure_runs_on_row(self, definition: dict[str, Any]) -> None:
        """Definition-pane counterpart of `task_detail.TaskDetail._
        configure_runs_on_row` (PR-3 task 5, fix round 1 finding 2) --
        see that method's docstring for the full rationale (affordance
        stays ON unconditionally; `on_detail_value_row_activated` decides
        dropdown-vs-show-why; Cancel/Retry are plain `.display`-toggled
        siblings, never `begin_edit`-mounted into the row); only the
        state-read (dict key vs. attribute) differs here.

        One definition-only departure: the owner dropdown is family-gated
        like every other editor in this pane (Qodo finding 11).
        `transfer_refusal` already refuses a non-`recurring_question`
        transfer on family grounds, so the dropdown could only ever offer
        a move that bounces. Cancel/Retry stay visible either way -- a
        transfer already in flight still needs its exit."""
        row = self._runs_on_row
        assert row is not None
        row.affordance = self._family_note is None
        row.can_focus = row.affordance
        state = definition.get("transfer_state")
        failed = state == "to_server_failed"
        locked = failed or state in IN_FLIGHT_TRANSFER_STATES
        assert self._runs_on_cancel_button is not None
        assert self._runs_on_retry_button is not None
        self._runs_on_cancel_button.display = locked
        self._runs_on_retry_button.display = failed

    def _runs_on_failure_reason(self, definition: dict[str, Any]) -> str:
        """Definition-pane counterpart of `task_detail.TaskDetail._
        runs_on_failure_reason` (fix round 1 finding 2)."""
        errors = self._runs_on_transfer_errors
        if errors:
            return "Last transfer error: " + "; ".join(errors)
        return _TRANSFER_STATE_ROW_LABELS.get(
            definition.get("transfer_state") or "", "This transfer failed."
        )

    def _editable_rows(self) -> tuple[DetailValueRow, ...]:
        """Every row this pane can open an editor on (mounted ones only)
        -- `task_detail.TaskDetail._editable_rows`' twin; see it for why
        the `Activated` router and `_reset_row_editing` share one list."""
        return tuple(
            row
            for row in (
                self._runs_on_row,
                self._model_row,
                self._generation_row,
                self._finding_policy_row,
                self._sources_row,
                self._notifications_row,
                self._repeat_row,
                self._at_row,
                self._timezone_row,
            )
            if row is not None
        )

    def _reset_row_editing(self) -> None:
        """Close every open row editor and clear every row error --
        `task_detail.TaskDetail._reset_row_editing`'s twin (final review
        I2/I3); see it for the full rationale. Called by `set_definition`
        on a row-identity change only."""
        for row in self._editable_rows():
            row.end_edit(restore_focus=False)
            row.clear_error()

    @property
    def runs_on_row(self) -> DetailValueRow | None:
        """The Runs-on row, for the workbench's `m` keybinding (redesign
        PR-4 task 4) to activate programmatically -- posting `DetailValueRow.
        Activated(row)` on it drives the exact same `on_detail_value_row_
        activated` path (honest lock/family-note refusal, then the
        dropdown) a real Enter/click already does, so `m` needs no new
        activation logic, only this reference."""
        return self._runs_on_row

    @property
    def shown_definition_id(self) -> str:
        """The id of the definition this pane is painting, `""` for none.

        The read half of `_editing_definition`'s captured-identity belt,
        exposed publicly for `SchedulesWorkbench._edit_definition_field`
        (Qodo finding 9): its save worker holds a `DetailValueRow`
        reference across an `await`, and a failure landing after the
        selection moved on would otherwise paint the old row's error
        under a different automation's field.
        """
        return str((self._definition or {}).get("id") or "")

    def _editing_definition(self) -> dict[str, Any] | None:
        """The definition an open editor was opened against, or ``None``
        -- `task_detail.TaskDetail._editing_task`'s twin (final review
        I2's belt); see it for the full rationale."""
        definition = self._definition
        if definition is None:
            return None
        current_id = str(definition.get("id") or "")
        if (
            self._editing_definition_id is not None
            and self._editing_definition_id != current_id
        ):
            logger.debug(
                "Discarding an automation row edit opened for {} while the "
                "definition pane now shows {}",
                self._editing_definition_id,
                current_id,
            )
            return None
        return definition

    def on_detail_value_row_activated(self, event: DetailValueRow.Activated) -> None:
        """Open the activated row's editor, or -- locked -- show why
        editing is refused instead of doing nothing (ruling 2).

        The `Unread results` row is not an editable row at all (it never
        opens an in-place editor) -- activating it instead posts
        `ViewDefinitionResultsRequested` upward for the workbench to push
        the definition-filtered Results view. `Last run` is the same
        shape (redesign PR-4, task 3): activating it posts `ViewDefinition
        AuditRequested` to push this definition's audit-trail view.
        Neither is gated on the lifecycle lock or the family note: viewing
        history is not an edit, so a locked/unsupported-family row can
        still be browsed.
        """
        row = event.row
        if row is self._unread_row or row is self._last_run_row:
            event.stop()
            if self._definition is None:
                return
            if row is self._unread_row:
                self.post_message(ViewDefinitionResultsRequested(self._definition))
            else:
                self.post_message(ViewDefinitionAuditRequested(self._definition))
            return
        if row not in self._editable_rows():
            return
        event.stop()
        if self._lifecycle_lock_reason is not None:
            row.show_error(self._lifecycle_lock_reason)
            return
        if self._family_note is not None:
            # Belt for the family gate (Qodo finding 11): a gated row's
            # `affordance` is already off, so it does not normally post
            # `Activated` at all -- but a row activated on the frame
            # before a repaint swapped in an unsupported family would
            # otherwise open an editor whose commit declares the wrong
            # family. Same "say why, never go silent" shape as the lock.
            row.show_error(self._family_note)
            return
        definition = self._definition
        if definition is None:
            return
        if (
            row is self._runs_on_row
            and definition.get("transfer_state") == "to_server_failed"
        ):
            # fix round 1 finding 2: same as the reminder pane -- not
            # `_lifecycle_lock_reason`-covered, but the dropdown has
            # nothing sensible to offer a failed row either.
            row.show_error(self._runs_on_failure_reason(definition))
            return
        row.clear_error()
        # Final review I2: capture the identity this editor is being
        # opened against, for `_editing_definition` to validate at commit.
        self._editing_definition_id = str(definition.get("id") or "")

        if row is self._runs_on_row:
            # A normal, unlocked owner pick (spec §7 flow) -- an
            # in-flight row never reaches here: the top-of-function
            # `_lifecycle_lock_reason` check above already intercepted it.
            current_owner = str(definition.get("owner_id") or "local")
            options = list(self._runs_on_options)
            if current_owner not in {value for _, value in options}:
                options = [*options, (owner_display_label(current_owner), current_owner)]
            row.begin_edit(
                Select(
                    options,
                    allow_blank=False,
                    value=current_owner,
                    id=_RUNS_ON_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._model_row:
            input_fields = (
                definition.get("input")
                if isinstance(definition.get("input"), dict)
                else {}
            )
            provider = str(input_fields.get("provider") or "").strip()
            model = str(input_fields.get("model") or "").strip()
            # The exact inverse of `_commit_model_edit`'s parse (Qodo
            # finding 10), NOT the row's display label: bare text has to
            # mean "provider" there (the common case), so a stored
            # model-with-no-provider must seed the editor as `/model` or
            # submitting it back unchanged silently moved the value into
            # the provider slot. `"".partition("/")` reads `/model` back
            # as exactly `(None, model)`.
            initial = f"{provider}/{model}" if model else provider
            row.begin_edit(
                Input(
                    value=initial,
                    placeholder="provider/model — blank for auto",
                    id=_MODEL_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._generation_row:
            config = (
                definition.get("config")
                if isinstance(definition.get("config"), dict)
                else {}
            )
            current = str(config.get("generation_mode") or "optional")
            row.begin_edit(
                Select(
                    list(_GENERATION_MODE_OPTIONS),
                    allow_blank=False,
                    value=current,
                    id=_GENERATION_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._finding_policy_row:
            finding_policy = (
                definition.get("finding_policy")
                if isinstance(definition.get("finding_policy"), dict)
                else {}
            )
            current = str(finding_policy.get("preset") or "balanced_findings")
            row.begin_edit(
                Select(
                    list(_FINDING_POLICY_OPTIONS),
                    allow_blank=False,
                    value=current,
                    id=_FINDING_POLICY_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._sources_row:
            self._begin_sources_edit(definition, row)
        elif row is self._notifications_row:
            current = "on" if self._notifications_on(definition) else "off"
            row.begin_edit(
                Select(
                    [("On", "on"), ("Off", "off")],
                    allow_blank=False,
                    value=current,
                    id=_NOTIFICATIONS_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._repeat_row:
            schedule = (
                definition.get("schedule")
                if isinstance(definition.get("schedule"), dict)
                else {}
            )
            cron_value = definition_cron_expression(schedule) or ""
            current_preset, _time_text = cron_to_preset(cron_value)
            row.begin_edit(
                Select(
                    ReminderForm._preset_options(),
                    allow_blank=False,
                    value=current_preset,
                    id=_REPEAT_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._at_row:
            schedule = (
                definition.get("schedule")
                if isinstance(definition.get("schedule"), dict)
                else {}
            )
            # Whichever field THIS kind's `At` actually owns (Qodo
            # finding 8) -- `run_at` for `one_time`, `time_of_day` for
            # `daily`/`weekly`. Unreachable for any other kind: the row
            # is not editable there.
            field = _AT_EDIT_FIELD_BY_KIND.get(str(schedule.get("kind") or ""))
            if field is None:
                return
            initial = str(schedule.get(field) or "")
            placeholder = (
                example_run_at_text() if field == "run_at" else DEFAULT_TIME_OF_DAY
            )
            row.begin_edit(
                Input(
                    value=initial,
                    placeholder=placeholder,
                    id=_AT_EDITOR_ID,
                    compact=True,
                )
            )
        elif row is self._timezone_row:
            schedule = (
                definition.get("schedule")
                if isinstance(definition.get("schedule"), dict)
                else {}
            )
            current_tz = str(schedule.get("timezone") or "UTC")
            row.begin_edit(
                Select(
                    timezone_options(
                        current_tz, self._known_timezones, noun="automation"
                    ),
                    allow_blank=False,
                    value=current_tz,
                    id=_TIMEZONE_EDITOR_ID,
                    compact=True,
                )
            )

    @staticmethod
    def _notifications_on(definition: dict[str, Any]) -> bool:
        """The Notifications editor's current On/Off, from either policy
        shape `_definition_notifications_label` already tolerates: this
        client's bool shape, or the server's channel-string shape (e.g.
        `{"on_success": "toast"}`, where any non-empty string is also
        truthy) -- one `bool(...)` reads both correctly."""
        policy = definition.get("notification_policy")
        if not isinstance(policy, dict) or not policy:
            return False
        return bool(policy.get("on_success")) or bool(policy.get("on_failure"))

    def _begin_sources_edit(
        self, definition: dict[str, Any], row: DetailValueRow
    ) -> None:
        """Mount the Sources mini-editor: three checkboxes (Media/Notes/
        Chats) + an Apply button, per the task-4 brief's own suggested
        shape -- a `Select` of the 7 mode/subset combinations would be
        uglier, and a checkbox has no single "commit" event of its own
        (unlike `Select`/`Input`), so an explicit Apply button is the
        smallest honest way to let several checkboxes settle before
        posting one edit.

        Documented simplification: this editor only ever WRITES the
        explicit `{"mode": "sources", "sources": [...]}` shape, even when
        every box ends up checked -- it never writes back `"mode":
        "all_searchable_library"` (which re-resolves the readable-source
        set live at every dispatch, rather than freezing today's three
        names). Reaching the frozen-vs-live distinction needs the full
        Edit form. All three boxes start CHECKED when the stored scope is
        `all_searchable_library` (or unset) -- visually "everything",
        matching what the row's own value currently reads -- and reflect
        the stored subset when the mode is already `"sources"`.
        """
        config = (
            definition.get("config") if isinstance(definition.get("config"), dict) else {}
        )
        scope = config.get("scope") if isinstance(config.get("scope"), dict) else {}
        if scope.get("mode") == "sources":
            selected = set(scope.get("sources") or [])
        else:
            selected = {value for _label, value in _SCOPE_SOURCE_CHECKBOXES}
        checkboxes = [
            Checkbox(
                label,
                value=(value in selected),
                id=_SOURCES_CHECKBOX_IDS[value],
            )
            for label, value in _SCOPE_SOURCE_CHECKBOXES
        ]
        apply_button = Button("Apply", id=_SOURCES_APPLY_ID, variant="primary")
        editor = Horizontal(*checkboxes, apply_button, id=_SOURCES_EDITOR_ID)
        row.begin_edit(editor)
        # `begin_edit`'s own `editor.focus()` is a no-op (a bare
        # `Horizontal` isn't focusable) -- focus the first checkbox
        # explicitly so Enter-to-open lands somewhere Tab/Space can drive,
        # matching a `Select`/`Input` editor's own auto-focus-on-open feel.
        self.call_after_refresh(checkboxes[0].focus)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Route a row Select editor's commit by its stable id."""
        editor_id = event.select.id
        if editor_id == _REPEAT_EDITOR_ID:
            self._commit_repeat_edit(event)
        elif editor_id == _TIMEZONE_EDITOR_ID:
            self._commit_timezone_edit(event)
        elif editor_id == _GENERATION_EDITOR_ID:
            self._commit_generation_edit(event)
        elif editor_id == _FINDING_POLICY_EDITOR_ID:
            self._commit_finding_policy_edit(event)
        elif editor_id == _NOTIFICATIONS_EDITOR_ID:
            self._commit_notifications_edit(event)
        elif editor_id == _RUNS_ON_EDITOR_ID:
            self._commit_runs_on_edit(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Route a row Input editor's commit (Enter submits) by its id."""
        if event.input.id == _AT_EDITOR_ID:
            self._commit_at_edit(event)
        elif event.input.id == _MODEL_EDITOR_ID:
            self._commit_model_edit(event)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == _SOURCES_APPLY_ID:
            self._commit_sources_edit(event)
        elif button_id == "scheduling-automation-pause-resume":
            event.stop()
            self._toggle_lifecycle()
        elif button_id == "scheduling-automation-run-now":
            event.stop()
            self._request_run_now()
        elif button_id == "scheduling-automation-duplicate":
            event.stop()
            self._request_duplicate()
        elif button_id == "scheduling-automation-view-runs":
            event.stop()
            if self._definition is not None:
                self.post_message(ViewDefinitionAuditRequested(self._definition))
        elif button_id == "scheduling-automation-view-results":
            event.stop()
            if self._definition is not None:
                self.post_message(ViewDefinitionResultsRequested(self._definition))
        elif button_id == _RUNS_ON_CANCEL_ID:
            event.stop()
            self._request_runs_on_cancel()
        elif button_id == _RUNS_ON_RETRY_ID:
            event.stop()
            self._request_runs_on_retry()

    def _commit_generation_edit(self, event: Select.Changed) -> None:
        definition = self._editing_definition()
        row = self._generation_row
        if definition is None or row is None:
            return
        event.stop()
        config = (
            definition.get("config") if isinstance(definition.get("config"), dict) else {}
        )
        current = str(config.get("generation_mode") or "optional")
        new_value = str(event.value)
        if new_value == current:
            # `Select` posts a synthetic `Changed` the moment `begin_edit`
            # mounts it with its CURRENT value preselected (same trap
            # task_detail.py's Task 3 Repeat/Timezone commits already
            # guard against) -- correctly no-ops both that mount echo and
            # a genuine reselect of the unchanged value.
            return
        row.end_edit()
        payload = _definition_edit_payload(
            definition, config={"generation_mode": new_value}
        )
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_finding_policy_edit(self, event: Select.Changed) -> None:
        definition = self._editing_definition()
        row = self._finding_policy_row
        if definition is None or row is None:
            return
        event.stop()
        finding_policy = (
            definition.get("finding_policy")
            if isinstance(definition.get("finding_policy"), dict)
            else {}
        )
        current = str(finding_policy.get("preset") or "balanced_findings")
        new_value = str(event.value)
        if new_value == current:
            return
        row.end_edit()
        payload = _definition_edit_payload(
            definition, config={"finding_policy": {"preset": new_value}}
        )
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_notifications_edit(self, event: Select.Changed) -> None:
        definition = self._editing_definition()
        row = self._notifications_row
        if definition is None or row is None:
            return
        event.stop()
        current = "on" if self._notifications_on(definition) else "off"
        new_value = str(event.value)
        if new_value == current:
            return
        row.end_edit()
        notify = new_value == "on"
        payload = _definition_edit_payload(
            definition,
            notification_policy={"on_success": notify, "on_failure": notify},
        )
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_model_edit(self, event: Input.Submitted) -> None:
        """Parse `provider/model` text back into the two `input` fields.

        The inverse of the editor's own seed value (see the `_model_row`
        branch of `on_detail_value_row_activated`, Qodo finding 10): a
        leading `/` means "model only, no provider", bare text means
        "provider only", and empty means neither (`auto`). Every stored
        shape therefore round-trips through an unchanged submit.
        """
        definition = self._editing_definition()
        row = self._model_row
        if definition is None or row is None:
            return
        event.stop()
        row.end_edit()
        raw = event.value.strip()
        if "/" in raw:
            provider_text, _sep, model_text = raw.partition("/")
            provider = provider_text.strip() or None
            model = model_text.strip() or None
        elif raw:
            provider, model = raw, None
        else:
            provider, model = None, None
        payload = _definition_edit_payload(
            definition, input={"provider": provider, "model": model}
        )
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_sources_edit(self, event: Button.Pressed) -> None:
        definition = self._editing_definition()
        row = self._sources_row
        if definition is None or row is None:
            return
        event.stop()
        checked = {cb.id: cb.value for cb in row.query(Checkbox)}
        selected = [
            value
            for _label, value in _SCOPE_SOURCE_CHECKBOXES
            if checked.get(_SOURCES_CHECKBOX_IDS[value])
        ]
        row.end_edit()
        if not selected:
            # Client-side, same rule as `AutomationDefinitionForm`'s own
            # preflight checks -- the server-side equivalent
            # (`config.scope`/`scope_empty`) would refuse the round trip
            # identically, this just skips the trip (ruling 2: never
            # silent, this DOES show why).
            row.show_error("Choose at least one source.")
            return
        payload = _definition_edit_payload(
            definition, config={"scope": {"mode": "sources", "sources": selected}}
        )
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_repeat_edit(self, event: Select.Changed) -> None:
        definition = self._editing_definition()
        row = self._repeat_row
        if definition is None or row is None:
            return
        event.stop()
        schedule = (
            definition.get("schedule")
            if isinstance(definition.get("schedule"), dict)
            else {}
        )
        cron_value = definition_cron_expression(schedule) or ""
        current_preset, current_time_text = cron_to_preset(cron_value)
        new_preset = str(event.value)
        if new_preset == current_preset:
            return
        row.end_edit()
        if new_preset == "custom":
            row.show_error(_REPEAT_CUSTOM_REFUSAL)
            return
        new_cron = preset_to_cron(
            new_preset, current_time_text or DEFAULT_TIME_OF_DAY
        )
        assert new_cron is not None, (
            "every _preset_options() value besides 'custom' always yields a cron"
        )
        # Whole-dict resend (task-4 brief, pinned): built from the row's
        # CURRENT schedule + only the edited field, so an edited Repeat
        # never drops `timezone` -- `_merge_definition_payload` does not
        # deep-merge `schedule` at all (`_definition_edit_payload`'s own
        # docstring). `expression` (the server's own cron key,
        # `definition_cron_expression`'s OTHER read) is dropped rather
        # than left stale beside the new `cron` -- this client always
        # writes `cron`, and leaving both would be confusing, if inert
        # (reads prefer `cron`).
        new_schedule = {**schedule, "kind": "cron", "cron": new_cron}
        new_schedule.pop("expression", None)
        payload = _definition_edit_payload(definition, schedule=new_schedule)
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_timezone_edit(self, event: Select.Changed) -> None:
        definition = self._editing_definition()
        row = self._timezone_row
        if definition is None or row is None:
            return
        event.stop()
        schedule = (
            definition.get("schedule")
            if isinstance(definition.get("schedule"), dict)
            else {}
        )
        current_tz = str(schedule.get("timezone") or "UTC")
        new_zone = str(event.value)
        if new_zone == current_tz:
            return
        row.end_edit()
        new_schedule = {**schedule, "kind": "cron", "timezone": new_zone}
        payload = _definition_edit_payload(definition, schedule=new_schedule)
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_at_edit(self, event: Input.Submitted) -> None:
        """Commit the `At` row, writing THIS schedule kind's own field.

        Never converts the kind (Qodo finding 8): the outgoing schedule
        is the stored dict with one field replaced, so a `daily` row
        keeps `kind`/`weekday`/`timezone` and only its `time_of_day`
        moves. Each kind is validated by the parser that owns its format
        -- `parse_forgiving_datetime` for a `one_time` `run_at` (the same
        one `AutomationDefinitionForm._client_side_schedule_error` runs,
        since the ported server validator only checks `kind`), and
        `parse_time_of_day` for a `daily`/`weekly` `time_of_day` (the
        same one `preset_to_cron` runs).
        """
        definition = self._editing_definition()
        row = self._at_row
        if definition is None or row is None:
            return
        event.stop()
        schedule = (
            definition.get("schedule")
            if isinstance(definition.get("schedule"), dict)
            else {}
        )
        field = _AT_EDIT_FIELD_BY_KIND.get(str(schedule.get("kind") or ""))
        if field is None:
            return
        row.end_edit()
        raw = event.value.strip()
        if field == "run_at":
            if not raw:
                row.show_error("Run at is required for one-time automations.")
                return
            parsed, _assumed_local = parse_forgiving_datetime(raw)
            if parsed is None:
                row.show_error(
                    f"Run at must be a date and time like {example_run_at_text()}."
                )
                return
            new_value: Any = parsed.isoformat()
        else:
            parsed_time = parse_time_of_day(raw)
            if parsed_time is None:
                row.show_error("Time of day must be a 24-hour time like 09:00.")
                return
            hour, minute = parsed_time
            new_value = f"{hour:02d}:{minute:02d}"
        new_schedule = {**schedule, field: new_value}
        payload = _definition_edit_payload(definition, schedule=new_schedule)
        self.post_message(DefinitionFieldEditRequested(definition, payload, row))

    def _commit_runs_on_edit(self, event: Select.Changed) -> None:
        """Commit the owner-picker Select (PR-3 task 5, spec §7 flow) --
        definition-pane counterpart of `task_detail.TaskDetail._commit_
        runs_on_edit`; see that method's docstring for the full
        same-owner-no-op / direction rationale."""
        definition = self._editing_definition()
        row = self._runs_on_row
        if definition is None or row is None:
            return
        event.stop()
        current_owner = str(definition.get("owner_id") or "local")
        new_owner = str(event.value)
        if new_owner == current_owner:
            return
        row.end_edit()
        direction = "to_server" if new_owner.startswith("server:") else "to_local"
        self.post_message(DefinitionOwnerActionRequested(definition, direction, row))

    def _request_runs_on_cancel(self) -> None:
        """Post a cancel request from the Runs-on row's own mini-bar
        (PR-3 task 5). It began as a SEPARATE surface coexisting with the
        Automations tab's `k`-key `_cancel_automation_transfer` (task-5
        brief); redesign PR-4 task 5 retired that tab and its key, so this
        mini-bar is now the only cancel affordance -- and it still renders
        a refusal via `row.show_error`, never a shared notice Static."""
        definition = self._definition
        row = self._runs_on_row
        if definition is not None and row is not None:
            self.post_message(DefinitionOwnerActionRequested(definition, "cancel", row))

    def _request_runs_on_retry(self) -> None:
        """Post a retry request from the Runs-on row's own mini-bar
        (PR-3 task 5) -- the PR-5 retry leg (re-begin), same SEPARATE
        row-scoped surface as `_request_runs_on_cancel` above."""
        definition = self._definition
        row = self._runs_on_row
        if definition is not None and row is not None:
            self.post_message(DefinitionOwnerActionRequested(definition, "retry", row))

    # -- Header lifecycle toggle (PR-3 task 4) -------------------------------

    def _refresh_lifecycle_button(self) -> None:
        """Repaint the Pause/Resume button's label/tooltip from `self.
        _definition`'s CURRENT `lifecycle` -- called after every paint
        (`set_definition`), after a lock-state change (`set_lifecycle_
        lock`), and after a successful toggle's optimistic patch
        (`apply_lifecycle`), so all three share one repaint path.
        """
        button = self._pause_resume_button
        if button is None or self._definition is None:
            return
        lifecycle = str(self._definition.get("lifecycle") or "configured")
        if lifecycle == "configured":
            button.label = "Pause"
            default_tooltip = "Pause this automation."
        else:
            button.label = "Resume"
            default_tooltip = "Resume this automation."
        button.disabled = self._lifecycle_lock_reason is not None
        button.tooltip = self._lifecycle_lock_reason or default_tooltip

    def _refresh_duplicate_button(self) -> None:
        """Disable `Duplicate` with a reason (UX-073) while a transfer is
        in flight OR the row's family is unsupported (task-31823) --
        reuses the SAME combined reason `_refresh_why_note` already
        renders into `#scheduling-automation-detail-why` for Pause/
        Resume, since both share one cause: a locked/unsupported row has
        nothing settled to fork from either way. Called alongside every
        `_refresh_lifecycle_button` call.
        """
        button = self._duplicate_button
        if button is None or self._definition is None:
            return
        reason = self._lifecycle_lock_reason or self._family_note
        button.disabled = reason is not None
        button.tooltip = reason or "Duplicate this automation as a new local definition."

    def _request_duplicate(self) -> None:
        """`Duplicate` button pressed (task-31823)."""
        definition = self._definition
        if definition is None:
            return
        self.post_message(DuplicateDefinitionRequested(definition))

    def _toggle_lifecycle(self) -> None:
        definition = self._definition
        if definition is None or self._lifecycle_lock_reason is not None:
            return
        lifecycle = str(definition.get("lifecycle") or "configured")
        action = "resume" if lifecycle != "configured" else "pause"
        self.post_message(DefinitionLifecycleToggleRequested(definition, action))

    def _request_run_now(self) -> None:
        """`Run now` button pressed (redesign PR-4, task 3).

        Never gated on `_lifecycle_lock_reason`/`_family_note`: the
        Automations tab's own `r` key never gated run-now on either
        (`_run_automation_now` refuses on its own terms -- pending sync,
        paused/archived lifecycle, unready health -- and reports why via
        a notification, the same honest-refusal shape this button keeps).
        """
        definition = self._definition
        if definition is None:
            return
        self.post_message(DefinitionRunNowRequested(definition))

    def apply_lifecycle(self, definition_id: str, lifecycle: str) -> None:
        """Patch + repaint the lifecycle in place after a successful
        toggle (the task-4 brief's "optimistic repaint") -- called by the
        workbench on EVERY mounted `DefinitionDetail` instance right
        after `set_definition_lifecycle` succeeds, ahead of the slower
        background refresh (`_request_tasks_refresh` -- redesign PR-4 task
        5 retired the Automations tab's `_request_automations_refresh`
        this used to name), so this pane
        never shows a stale Pause/Resume label while that worker is still
        running. Task 2's own DB-level pull-guard is what then keeps a
        RACING sync pull from reverting the value that background refresh
        eventually reads back from the DB.

        A no-op when this instance isn't currently showing
        ``definition_id`` (a different row, or nothing selected) --
        sibling `DefinitionDetail` instances never repaint each other for
        the wrong row.
        """
        if self._definition is None or str(self._definition.get("id")) != definition_id:
            return
        self._definition["lifecycle"] = lifecycle
        self._refresh_lifecycle_button()

    def set_runs_on_transfer_errors(self, errors: list[str]) -> None:
        """Definition-pane counterpart of `task_detail.TaskDetail.set_
        runs_on_transfer_errors` (fix round 1, finding 2)."""
        self._runs_on_transfer_errors = list(errors)

    def set_lifecycle_lock(self, reason: str | None) -> None:
        """Freeze the Pause/Resume and Duplicate buttons while a transfer
        is in flight (spec §6.3's "dormant and in-flight rows are
        read-only except cancel"), mirroring `TaskDetail.
        set_lifecycle_lock` exactly: ``reason`` comes from
        `SchedulingService.transfer_lock_reason` (never re-derived here)
        and is both the buttons' tooltip and a line in the
        always-visible `#scheduling-automation-detail-why` Static
        (UX-073). Row affordance is untouched here -- editable rows stay
        lit and surface the SAME cached reason at activation time
        instead (`on_detail_value_row_activated`), the same split
        `TaskDetail`'s own Frequency rows already use.
        """
        self._lifecycle_lock_reason = reason
        self._refresh_lifecycle_button()
        self._refresh_duplicate_button()
        self._refresh_why_note()
