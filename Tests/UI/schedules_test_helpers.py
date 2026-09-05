"""Shared stubs and paint oracles for the Schedules UI test files.

task-23106 review round (F15): the scheduling-service stub had been
copy-pasted into five test files while ``test_schedules_workbench.py``
already shipped reusable mixins, and the compositor center-probe
re-implemented the shared painted-region oracle. Everything lives here
once now.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService


class MockServerClient:
    """Stub server client for test scheduling services."""

    def __init__(self, notifications_service: Any = None) -> None:
        self.notifications_service = notifications_service


class MockSchedulingDB:
    """Stub scheduled-tasks DB for test scheduling services."""

    def __init__(
        self,
        sync_state: dict | None = None,
        conflicts: list | None = None,
        automation_definitions: list[dict] | None = None,
        automation_results: list[dict] | None = None,
        automation_runs: list[dict] | None = None,
    ) -> None:
        self._sync_state = sync_state or {}
        self._conflicts = conflicts or []
        #: task-5 fix round: local automation-definition rows this DB
        #: "contains" -- mirrors the real ScheduledTasksDB surface the
        #: Automations tab now reads (list_automation_definitions et al).
        self._automation_definitions = list(automation_definitions or [])
        #: schedules-handoff PR-6 task 3: `automation_results` rows this DB
        #: "contains" -- the Results tab's `on_mount` calls `list_
        #: automation_results`/`count_unread_results` unconditionally, so
        #: EVERY existing SchedulesWorkbench test built on this fake needs
        #: these even when a test never touches the Results tab.
        self._automation_results = list(automation_results or [])
        #: schedules-redesign PR-1, Task 4: `automation_runs` rows this DB
        #: "contains" -- the definitions detail pane's "Last run"/"Run
        #: count" rows read these (`count_automation_runs`/
        #: `list_automation_runs`).
        self._automation_runs = list(automation_runs or [])

    def get_sync_state(self, owner_id: str):
        return dict(self._sync_state)

    def update_sync_state(self, owner_id: str, **kwargs) -> None:
        self._sync_state.update(kwargs)

    def get_conflicts(self, owner_id: str, primitive=None):
        return self._conflicts

    def list_automation_definitions(self, owner_id=None, lifecycle=None, family=None):
        return [
            dict(row)
            for row in self._automation_definitions
            if (owner_id is None or row.get("owner_id") == owner_id)
            and (lifecycle is None or row.get("lifecycle") == lifecycle)
            and (family is None or row.get("family") == family)
        ]

    def get_automation_definition(self, definition_id: str):
        for row in self._automation_definitions:
            if row.get("id") == definition_id:
                return dict(row)
        return None

    def get_automation_definition_by_server_id(self, owner_id: str, server_id: str):
        for row in self._automation_definitions:
            if row.get("owner_id") == owner_id and row.get("server_id") == server_id:
                return dict(row)
        return None

    def get_pending_mutations(self, owner_id: str, primitive: str | None = None):
        """Stub: no pending mutations (schedules-handoff PR-5 task 7 --
        the detail pane's retry-errors lookup calls this for a
        `to_server_failed` row; unused otherwise)."""
        return []

    def get_pending_mutation_for_local_id(self, local_id: str, primitive: str):
        """Stub: no pending mutation (Task 7 fix round finding 3 --
        owner-agnostic retry-error lookup for a `to_server_failed` row;
        unused otherwise)."""
        return None

    def list_automation_results(
        self,
        owner_id: str | None,
        review_state: str | None = None,
        definition_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        """Stub: mirrors the real `list_automation_results` filter set
        (schedules-handoff PR-6 task 3) -- `owner_id=None` spans every
        owner, matching Task 1's all-owners extension."""
        rows = [
            dict(row)
            for row in self._automation_results
            if (owner_id is None or row.get("owner_id") == owner_id)
            and (review_state is None or row.get("review_state") == review_state)
            and (definition_id is None or row.get("definition_id") == definition_id)
        ]
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return rows[offset : offset + limit]

    def count_automation_results(
        self,
        owner_id: str | None,
        review_state: str | None = None,
        definition_id: str | None = None,
    ) -> int:
        """Stub: mirrors the real `count_automation_results` -- the "of N"
        denominator for the capped inbox listing. `_refresh_results_tab`
        calls it unconditionally, so every workbench test built on this
        fake needs it even when it never opens the Results tab.

        `definition_id` (Task 2 seam) added for the definitions detail
        pane's per-definition unread-count row (schedules-redesign PR-1,
        Task 4) -- matched exactly as given, same as the real DB."""
        return len(
            [
                row
                for row in self._automation_results
                if (review_state is None or row.get("review_state") == review_state)
                and (owner_id is None or row.get("owner_id") == owner_id)
                and (definition_id is None or row.get("definition_id") == definition_id)
            ]
        )

    def count_unread_results(
        self, owner_id: str | None, definition_id: str | None = None
    ) -> int:
        """Stub: mirrors the real `count_unread_results` (Results tab
        badge, schedules-handoff PR-6 task 3; `definition_id` filter added
        Task 2/used by Task 4's detail pane)."""
        return self.count_automation_results(
            owner_id, review_state="unread", definition_id=definition_id
        )

    def list_automation_runs(
        self,
        owner_id: str,
        definition_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Stub: mirrors the real `list_automation_runs` (schedules-
        redesign PR-1, Task 4's "Last run" row), newest first by
        `created_at`."""
        rows = [
            dict(row)
            for row in self._automation_runs
            if row.get("owner_id") == owner_id
            and (definition_id is None or row.get("definition_id") == definition_id)
        ]
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return rows[offset : offset + limit]

    def count_automation_runs(
        self, definition_id: str, owner_id: str | None = None
    ) -> int:
        """Stub: mirrors the real `count_automation_runs` (schedules-
        redesign PR-1, Task 4's "Run count" row), including the optional
        owner scope the final review's F11 added."""
        return len(
            [
                row
                for row in self._automation_runs
                if row.get("definition_id") == definition_id
                and (owner_id is None or row.get("owner_id") == owner_id)
            ]
        )

    def upsert_automation_definitions_from_server(self, owner_id: str, items: list[dict]):
        inserted = 0
        updated = 0
        for item in items:
            server_id = item.get("id")
            if not server_id:
                continue
            existing = next(
                (
                    row
                    for row in self._automation_definitions
                    if row.get("owner_id") == owner_id and row.get("server_id") == server_id
                ),
                None,
            )
            if existing is not None:
                existing.update(item)
                updated += 1
                continue
            local_row = dict(item)
            local_row["id"] = f"local-mirror-{server_id}"
            local_row["server_id"] = server_id
            local_row["owner_id"] = owner_id
            self._automation_definitions.append(local_row)
            inserted += 1
        return {"inserted": inserted, "updated": updated}


class MockSchedulingServiceMixin:
    """Common attributes expected by the SchedulesWorkbench UI.

    Subclass and implement ``list_tasks`` (and whichever mutation methods
    the test drives). ``server_client``/``db`` are class-level defaults;
    assign instance attributes to specialize.
    """

    owner_id = "local"
    server_client = MockServerClient()
    db = MockSchedulingDB()
    sync_engine = None

    def set_owner(self, owner_id: str) -> None:
        self.owner_id = owner_id

    async def sync_now(self, owner_id: str | None = None):
        return None

    # -- transfer machine facade (schedules-handoff PR-5 task 7) -----------
    #
    # `SchedulesWorkbench._update_transfer_actions` calls `transfer_refusal`
    # on EVERY row selection, so every existing scheduling-service stub
    # needs these even when a test never touches a transfer button.
    # Mirrors the real facade's first gate honestly: no `notifications_
    # service` wired reads as "no server connection is configured" --
    # every button renders disabled-with-reason but nothing crashes. A
    # test that wants transfer buttons enabled overrides this (or wires a
    # `server_client` whose `notifications_service` is not None and
    # overrides `transfer_refusal` to return ``None``).

    def transfer_refusal(self, row: dict, direction: str) -> str | None:
        if getattr(self.server_client, "notifications_service", None) is None:
            return "No server connection is configured."
        return None

    #: Delegates to the REAL implementation rather than restating the
    #: state set (final review I7): `transfer_lock_reason` is a
    #: `@staticmethod` reading only the row dict, so there is nothing to
    #: fake -- and a second copy of the in-flight state list here is
    #: exactly the drift the shared constant was introduced to kill.
    transfer_lock_reason = staticmethod(SchedulingService.transfer_lock_reason)

    def cancel_refusal(self, row: dict) -> str | None:
        """Mirrors the real facade's `cancel_refusal` (Task 7 fix round
        finding 1): honest state-branching, no server-connection gate --
        cancel never needed one."""
        state = row.get("transfer_state")
        if state in ("to_server_pending", "to_server_failed", "from_server_pending"):
            return None
        if state == "to_server_sent":
            return "Too late to cancel -- start a reverse transfer instead."
        return (
            "No transfer in progress on this row -- if it already moved, "
            "start a reverse transfer instead."
        )

    def transfer_warnings(self, row: dict, direction: str) -> list[str]:
        return []

    async def begin_transfer_to_server(self, table_kind: str, row_id: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            status="refused",
            reason="Stub scheduling service: transfer not implemented.",
            row_id=None,
        )

    async def begin_transfer_to_local(self, table_kind: str, row_id: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            status="refused",
            reason="Stub scheduling service: transfer not implemented.",
            row_id=None,
        )

    async def cancel_transfer(self, table_kind: str, row_id: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            status="refused",
            reason="Stub scheduling service: transfer not implemented.",
            row_id=None,
        )


# --- compositor paint oracles ---------------------------------------------
#
# ``Widget.region`` is reported in an UNCLIPPED coordinate space -- a
# widget scrolled out of (or simply clipped inside) a scrollable ancestor
# still has a plausible region the ancestor never paints. Only the
# compositor (``App.get_widget_at``) answers what a live terminal actually
# renders (see lessons-live-verification.md).


def assert_painted_at_own_region(host, widget) -> None:
    """Fail unless the compositor paints ``widget`` at its own top-left."""
    region = widget.region
    try:
        hit_widget, _hit_region = host.get_widget_at(region.x + 1, region.y)
    except Exception as exc:  # textual.errors.NoWidget
        pytest.fail(
            f"nothing is painted at {widget!r}'s own region {region!r}: {exc}"
        )
    assert hit_widget is widget, (
        f"the compositor paints {hit_widget!r} at {region!r}, not {widget!r} "
        "itself -- the widget's display chain is all-True but it is not "
        "actually visible on screen"
    )


def painted_at_own_center(host, widget) -> bool:
    """Center-probe variant: True when the compositor paints ``widget``
    (or one of its descendants -- Select/TextArea paint through children)
    at the widget's own center cell."""
    region = widget.region
    if region.height <= 0 or region.width <= 0:
        return False
    cx, cy = region.center
    try:
        target, _ = host.get_widget_at(int(cx), int(cy))
    except Exception:
        return False
    return target is widget or widget in list(target.ancestors)


def rendered_row_cells(table, row_index: int = 0) -> list[str]:
    """The cell text a `DataTable` will actually PAINT for one row.

    Routes the stored row through the widget's own
    `_get_row_renderables` -> `default_cell_formatter`, which is where a
    `str` cell gets run through `rich.text.Text.from_markup` and a
    bracket token can be silently eaten (task 6 round 2, D8).

    `get_cell_at()` returns the STORED value and therefore passes whether
    or not the content survives rendering -- the same self-confirming
    shape as the round-1 `TabPane.label` badge test. Assert through here
    whenever the point of the test is that content renders literally.
    """
    return [str(cell) for cell in table._get_row_renderables(row_index).cells]


# --- settling the workbench --------------------------------------------------


async def settle_schedules_workbench(pilot, workbench=None) -> None:
    """Drain the workbench's background work, debounce timer included.

    ``pilot.app.workers.wait_for_complete()`` does NOT cover a pending
    ``set_timer`` callback, and `SchedulesWorkbench.on_mount` arms one: the
    catch-up results pull (`_schedule_catch_up_results_pull`, 0.3 s). Its
    worker ends in `_request_tasks_refresh`, which re-feeds the detail
    pane -- so a test that has painted or pushed a detail pane can have it
    cleared out from under an assertion by a reload landing after the test
    believed the screen was settled.

    redesign PR-4 task 5 hit this in `test_runs_on_dropdown_refusal_
    renders_inline_with_health_reason` and fixed it inline; task 6's
    pushed-pane tests are the second occurrence, which is where a shared
    helper earns its keep (task-5 review's own ruling). Only the
    results-pull timer is stopped -- the queue filter's debounce is
    deliberately driven by the tests that use it.

    Args:
        pilot: The running `Pilot`.
        workbench: The workbench screen; defaults to the app's current
            screen.
    """
    app = pilot.app
    target = workbench if workbench is not None else app.screen
    await app.workers.wait_for_complete()
    timer = getattr(target, "_results_pull_debounce_timer", None)
    if timer is not None:
        timer.stop()
        target._results_pull_debounce_timer = None
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()
