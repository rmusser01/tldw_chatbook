"""Read-only, pushed audit-trail view for one automation definition.

redesign PR-4, task 3 (audit-view relocation): the Automations tab's
third pane (`SchedulesWorkbench._load_automation_history`, the survey's
":3222-3271") is the only place a definition's SERVER-side execution
audit trail (``list_automation_definition_audit`` -- task-18940 slice 4,
distinct from both the local `automation_runs` history and the Results
inbox) has ever been readable. That tab retires in Task 5; this module
is its replacement surface, pushed via `WorkbenchHostScreen` from a
`DefinitionDetail` pane's `Last run` row (the row whose own copy already
says "...see Run history" for a server-owned definition -- this
activation is what makes that pointer live, `ViewDefinitionAuditRequested`
in `Scheduling/events.py`).

`fetch_definition_audit` factors OUT the fetch-and-branch logic
`_load_automation_history` already established (pending-sync / local-only
/ no-server-client / fetch-failed / success) so both call sites share one
implementation -- `schedules_workbench.py`'s own loader is refactored to
call it too, rather than this module duplicating that branching. Each
caller keeps its OWN paint code: the tab's own DataTable/notice/title ids
and staleness re-check are untouched (it still exists until Task 5), and
this widget owns a fresh, independent set.

Escape discipline: `summary`/`event_type` are free-form SERVER text (an
automation's own audit log, task-18940 slice 4's `summary` field is
explicitly documented there as arbitrary), routed into the `DataTable`
as `rich.text.Text` (never a bare `str`) -- the same "`Text`, not `str`"
rule `schedules_workbench.py`'s own table renderers already use, since a
`DataTable` cell built from a plain string is re-parsed as Rich markup
and can silently eat a bracketed token. The notice line is always an
internally-composed count/placeholder string, never server text, but its
`Static` is still built `markup=False` for the same "safe by
construction" reason `definition_detail.py`'s own question-card Static
is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static


@dataclass
class DefinitionAuditFetch:
    """The result of one `fetch_definition_audit` call.

    ``notice_override`` is set for every non-success branch (pending
    sync / local-only / no server connection / fetch failure) -- the
    caller shows it verbatim instead of painting `items`/`total` (which
    are empty/zero in that case, not a genuine "no events" reading).
    ``None`` means the fetch succeeded (`items` may still be empty --
    a real "no recorded events yet" outcome, distinct from a refusal).
    """

    items: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    notice_override: str | None = None


async def fetch_definition_audit(
    service: Any, definition: dict[str, Any]
) -> DefinitionAuditFetch:
    """Fetch one definition's server audit trail, or an honest refusal.

    Mirrors `SchedulesWorkbench._load_automation_history`'s own
    branching exactly (this IS that logic, factored out so both callers
    share it): a definition that never synced, a local-only definition
    (no durable audit trail exists for local dispatch yet), no connected
    server, or a fetch that raised all return a `notice_override`
    instead of raising -- this function never propagates an exception.

    Args:
        service: The `SchedulingService` (or a test double exposing the
            same `server_client` attribute), or `None`.
        definition: The definition row dict (local DB row or raw server
            list-response dict -- either shape, same as `DefinitionDetail
            .set_definition`).

    Returns:
        A `DefinitionAuditFetch`.
    """
    # ADR-097: scheduler.queue stays off the boot census -- function-local
    # import, matching every other reader of this predicate in this
    # package (`unified_rows.owner_display_label`, `definition_detail
    # ._definition_history_labels`, `schedules_workbench._load_automation
    # _history`).
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    owner_id = (definition or {}).get("owner_id")
    if (definition or {}).get("pending_sync"):
        return DefinitionAuditFetch(
            notice_override=(
                "This automation hasn't synced to the server yet, so it "
                "has no run history."
            )
        )
    if not is_server_scoped_owner(owner_id):
        return DefinitionAuditFetch(
            notice_override="Local automation history isn't available yet."
        )
    server_client = getattr(service, "server_client", None) if service else None
    if server_client is None:
        return DefinitionAuditFetch(
            notice_override="Run history needs a connected server."
        )
    definition_id = str(definition.get("id") or "")
    try:
        response = await server_client.list_automation_definition_audit(
            definition_id
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to load automation audit trail (definition_id={})",
            definition_id,
        )
        return DefinitionAuditFetch(
            notice_override="Could not load the run history — see the log."
        )
    items = list(response.get("items", []))
    total = int(response.get("total", len(items)) or 0)
    return DefinitionAuditFetch(items=items, total=total)


def audit_notice_text(items: list[dict[str, Any]], total: int) -> str:
    """The success-path notice line -- same wording `_load_automation_
    history` already renders, factored out so both callers stay in sync."""
    if not items:
        return "No recorded events for this automation yet."
    suffix = f" of {total}" if total > len(items) else ""
    return f"{len(items)} event{'' if len(items) == 1 else 's'}{suffix}."


class DefinitionAuditView(Vertical):
    """Pushed, read-only widget rendering one definition's audit trail.

    Self-populating (`on_mount` kicks off its own fetch through `fetch_
    definition_audit`) -- unlike `ResultsTab`/`ConflictsTab`'s `initial_*`
    seam, the fetch here is unavoidably async (a `server_client` call),
    so there is no synchronous value to hand over at construction time
    the way those two pre-read before pushing. No mutation surface (pure
    render), so the caller pushes it through the plain `WorkbenchHostScreen`
    -- unlike `ResultsHostScreen`, this widget needs no dedicated Screen
    subclass.
    """

    # Same shape `ConflictsTab`/`ResultsTab`'s own `BUNDLED_CSS` uses
    # (`.workbench-host-body` on the pushed root gives THIS widget
    # height:1fr; the table needs its own rule to actually expand into
    # that space rather than sizing to its rows).
    BUNDLED_CSS = """
    DefinitionAuditView {
        height: 1fr;
    }
    #scheduling-audit-view-table {
        height: 1fr;
    }
    """

    def __init__(
        self, service: Any, definition: dict[str, Any], **kwargs: Any
    ) -> None:
        """Store the fetch inputs; the fetch itself waits for `on_mount`.

        Args:
            service: The `SchedulingService` (or a test double exposing
                the same `server_client` attribute), or `None` --
                forwarded verbatim to `fetch_definition_audit`.
            definition: The definition row dict this audit trail is for
                (local DB row or raw server list-response dict).
            **kwargs: Forwarded to `Vertical.__init__` (e.g. `id`).
        """
        super().__init__(**kwargs)
        self._service = service
        self._definition = definition

    def compose(self) -> ComposeResult:
        """Yield the loading notice and the (initially empty) audit table.

        `_populate` (kicked off from `on_mount`) fills the table and
        replaces the notice text once the fetch resolves.
        """
        yield Static(
            "Loading run history…",
            id="scheduling-audit-view-notice",
            markup=False,
        )
        table = DataTable(id="scheduling-audit-view-table", cursor_type="row")
        table.add_columns("When", "Event", "Summary")
        yield table

    def on_mount(self) -> None:
        """Kick off the self-populating fetch (`_populate`).

        `exclusive=True` + a dedicated worker group: this widget is
        pushed fresh on every activation (Task 1's factory contract), so
        there is never a prior `_populate` run of THIS instance to
        collide with -- the group only needs to be distinct from other
        panes' own worker groups, not de-duplicate repeats.
        """
        self.run_worker(
            self._populate, exclusive=True, group="schedules-audit-view"
        )

    async def _populate(self) -> None:
        notice = self.query_one("#scheduling-audit-view-notice", Static)
        table = self.query_one("#scheduling-audit-view-table", DataTable)
        result = await fetch_definition_audit(self._service, self._definition)
        table.clear()
        if result.notice_override is not None:
            notice.update(result.notice_override)
            return
        for event in result.items:
            created = str(event.get("created_at") or "")
            # Keep the timestamp compact: date and minute-level time, no
            # microseconds/timezone noise in a table cell -- same slice
            # the tab's own history table uses.
            stamp = created[:16].replace("T", " ") if created else "?"
            summary = str(event.get("summary") or "")
            table.add_row(
                Text(stamp),
                Text(str(event.get("event_type") or "?")),
                Text(summary),
            )
        notice.update(audit_notice_text(result.items, result.total))
