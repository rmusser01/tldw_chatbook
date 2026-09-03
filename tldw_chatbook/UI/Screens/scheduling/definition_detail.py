"""Automations tab per-definition detail pane (schedules-redesign PR-1, Task 4).

The Automations tab's FIRST per-row detail widget -- until now the tab
had no field-level rendering of a selected `automation_definition` at all
(only the definitions `DataTable`'s five columns and the audit-trail
history pane; see `redesign-pr1-survey.md` section 1's "no per-row detail
widget" finding). `DefinitionDetail` fills that gap using the same
`DetailValueRow`/`DetailGroup` row grammar Task 3 used to regrammar the
Queue tab's `TaskDetail` -- Details/Frequency/History groups, read-only
(plan ruling 1: every row's `affordance` stays at its `False` default, no
new bindings/actions).

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
a JSON round-trip, so any camelCase/renamed field a live server used
would need translating separately -- out of this task's scope, matching
`AutomationDefinitionForm._prefill_from_row`'s own documented precedent
of leaving an un-recognized schedule shape at its default rather than
guessing). Every formatter here reads with `.get()`/`isinstance` guards
and degrades to an honest placeholder instead of raising, the same
defensive-read idiom `automation_execution_target_label` already used.

Escape discipline: every `DetailValueRow` value is safe by construction
(Task 1: the row's `Static` is built with `markup=False` and the value is
wrapped in a literal `rich.text.Text`, proven never to interpret
markup). The one field NOT going through `DetailValueRow` -- the question
card, a plain `Static` -- is built with `markup=False` here too, for the
same reason.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ....Widgets.detail_value_row import DetailGroup, DetailValueRow
from .forms.automation_definition_form import (
    _FINDING_POLICY_OPTIONS,
    _GENERATION_MODE_OPTIONS,
    _SCOPE_SOURCE_CHECKBOXES,
)
from .task_detail import _TRANSFER_STATE_ROW_LABELS, _format_timezone, _humanize_cron

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
    """
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    owner_id = str(definition.get("owner_id") or "local")
    if not is_server_scoped_owner(owner_id):
        return "This device"
    label = owner_id.split(":", 1)[1] if ":" in owner_id else owner_id
    if definition.get("pending_sync"):
        label = f"{label} · pending sync"
    return label


def automation_name_cell(definition: dict[str, Any]) -> str:
    """Name cell for the merged local+server Automations list (task-5 fix round).

    Moved here from `schedules_workbench.py` (Task 4) alongside
    `_definition_owner_label`, which this now delegates to; behavior is
    unchanged. `schedules_workbench` re-exports the name for its own
    DataTable render call site.

    Args:
        definition: One merged row (local DB dict or server API dict --
            both carry `owner_id` and `name`, confirmed against the real
            server fixture `automation_definition_list.json`).

    Returns:
        `"[This device] <name>"` for a local row, `"[<server id>] <name>"`
        for a server-scoped one, and `"[<server id> · pending sync]
        <name>"` for one authored offline that has not reached the server
        yet.
    """
    name = str(definition.get("name") or definition.get("id") or "")
    return f"[{_definition_owner_label(definition)}] {name}"


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


def _definition_question_text(definition: dict[str, Any]) -> str:
    """Question-card text: the recurring question, or a fallback for a
    definition that has none (a non-`recurring_question` family, or a
    row from an older/foreign server payload shape)."""
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


def _definition_generation_label(config: dict[str, Any]) -> str:
    """'Generation' row value (Details group): `config.generation_mode`,
    labeled with the same wording the create/edit form's Select uses.
    Defaults to "optional" (the form's own default) when absent."""
    mode = config.get("generation_mode") if isinstance(config, dict) else None
    mode = str(mode) if mode else "optional"
    return _GENERATION_MODE_LABELS.get(mode, mode)


def _definition_finding_policy_label(definition: dict[str, Any]) -> str:
    """'Finding policy' row value (Details group): the top-level
    `finding_policy.preset` column (survey §4's authoritative source --
    distinct from the create/edit payload's `config.finding_policy`,
    which the service copies into this column at save time). Defaults to
    "balanced_findings" (the `AutomationDefinition` model's own default)
    when absent."""
    finding_policy = definition.get("finding_policy")
    preset = (
        finding_policy.get("preset") if isinstance(finding_policy, dict) else None
    ) or "balanced_findings"
    return _FINDING_POLICY_LABELS.get(preset, str(preset))


def _definition_sources_label(config: dict[str, Any]) -> str:
    """'Sources' row value (Details group): `config.scope`, joined plurals
    for an explicit source list, or the all-library sentence for the
    default scope mode."""
    scope = config.get("scope") if isinstance(config, dict) else None
    scope = scope if isinstance(scope, dict) else {}
    if scope.get("mode") != "sources":
        return "All searchable library"
    sources = scope.get("sources") or []
    labels = [_SCOPE_SOURCE_LABELS.get(str(item), str(item)) for item in sources]
    return ", ".join(labels) if labels else "None selected"


def _parse_iso(value: Any) -> datetime | None:
    """Best-effort ISO-8601 parse; ``None`` for anything else, never raises."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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


def _definition_at_label(schedule: dict[str, Any]) -> str:
    """'At' row value (Frequency group): the full schedule summary, reusing
    `_humanize_cron` for a cron-kind schedule -- the same formatter
    `task_detail._humanize_schedule`/`reminder_form.py`'s live cron preview
    already reuse -- rather than re-deriving cron-cadence prose."""
    if not isinstance(schedule, dict):
        return "-"
    kind = schedule.get("kind")
    if kind == "cron":
        return _humanize_cron(schedule.get("cron"), schedule.get("timezone"))
    if kind == "one_time":
        run_at = schedule.get("run_at")
        dt = _parse_iso(run_at) if run_at else None
        if dt is None:
            return f"One-time at {run_at}" if run_at else "One-time"
        return f"One-time at {dt.strftime('%Y-%m-%d %H:%M')} {_format_timezone(dt)}"
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
    """Automations tab per-definition detail pane.

    Read-only (plan ruling 1): every `DetailValueRow` here uses the
    `affordance` default of `False`; no new bindings or actions.
    `set_definition` is the single feed point -- called by
    `SchedulesWorkbench` after a definitions-table row highlight, with
    the DB-derived counts/last-run already fetched off the event loop.
    """

    def __init__(self, *args, **kwargs) -> None:
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
        self._last_run_row: DetailValueRow | None = None
        self._run_count_row: DetailValueRow | None = None
        self._unread_row: DetailValueRow | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Definition Detail",
            id="scheduling-automation-detail-header",
            classes="scheduling-column-title",
        )
        yield Static(
            "Select an automation to see its details.",
            id="scheduling-automation-detail-empty-state",
        )
        with Vertical(id="scheduling-automation-detail-body"):
            self._question_static = Static(
                "", id="scheduling-automation-detail-question", markup=False
            )
            yield self._question_static

            self._runs_on_row = DetailValueRow(
                "Runs on", "-", value_id="scheduling-automation-detail-runs-on"
            )
            self._model_row = DetailValueRow(
                "Model", "-", value_id="scheduling-automation-detail-model"
            )
            self._generation_row = DetailValueRow(
                "Generation", "-", value_id="scheduling-automation-detail-generation"
            )
            self._finding_policy_row = DetailValueRow(
                "Finding policy",
                "-",
                value_id="scheduling-automation-detail-finding-policy",
            )
            self._sources_row = DetailValueRow(
                "Sources", "-", value_id="scheduling-automation-detail-sources"
            )
            yield DetailGroup(
                self._runs_on_row,
                self._model_row,
                self._generation_row,
                self._finding_policy_row,
                self._sources_row,
                title="Details",
                id="scheduling-automation-detail-group-details",
            )

            self._repeat_row = DetailValueRow(
                "Repeat", "-", value_id="scheduling-automation-detail-repeat"
            )
            self._at_row = DetailValueRow(
                "At", "-", value_id="scheduling-automation-detail-at"
            )
            self._timezone_row = DetailValueRow(
                "Timezone", "-", value_id="scheduling-automation-detail-timezone"
            )
            yield DetailGroup(
                self._repeat_row,
                self._at_row,
                self._timezone_row,
                title="Frequency",
                id="scheduling-automation-detail-group-frequency",
            )

            self._last_run_row = DetailValueRow(
                "Last run", "-", value_id="scheduling-automation-detail-last-run"
            )
            self._run_count_row = DetailValueRow(
                "Run count", "-", value_id="scheduling-automation-detail-run-count"
            )
            self._unread_row = DetailValueRow(
                "Unread results",
                "-",
                value_id="scheduling-automation-detail-unread-results",
            )
            view_results_row = DetailValueRow(
                "Results",
                "See Results tab",
                value_id="scheduling-automation-detail-view-results",
            )
            yield DetailGroup(
                self._last_run_row,
                self._run_count_row,
                self._unread_row,
                view_results_row,
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
    ) -> None:
        """Paint the pane for `definition` (or the empty state for `None`).

        Pure render -- `run_count`/`last_run`/`unread_count` are already
        fetched (off the event loop) by the caller; this method performs
        no I/O of its own.
        """
        empty_state = self.query_one(
            "#scheduling-automation-detail-empty-state", Static
        )
        body = self.query_one("#scheduling-automation-detail-body", Vertical)
        if definition is None:
            empty_state.display = True
            body.display = False
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

        self._last_run_row.update_value(_definition_last_run_label(last_run))
        self._run_count_row.update_value(str(run_count))
        self._unread_row.update_value(str(unread_count))
