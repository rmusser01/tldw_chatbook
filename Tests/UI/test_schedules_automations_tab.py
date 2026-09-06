"""Automation-definition behavior on the Schedules workbench (task-18940
slice 2).

Definitions come from two halves -- the server's (ADR-077, dispatched
through the server control plane) and this device's own local-owned
`recurring_question` rows (dispatched through `SchedulingService.run_
automation_now`; never the server client, and never both).

redesign PR-4 task 5 (the retirement): the file keeps its `automations_
tab` name -- the same judgment the plan applied to `results_tab.py`/
`conflicts_tab.py`, renaming buys churn, not clarity -- but the TAB it
was written against is gone. Its listing, its per-definition detail pane
and its run-now/edit actions all relocated onto the unified queue
(`#scheduling-task-table` + `#scheduling-queue-definition-detail`), and
its audit-trail pane relocated to the pushed `DefinitionAuditView`
(task 3). Every test below is re-pointed at whichever of those now owns
the behaviour, with the exceptions individually cited where a surface
(and its behaviour) genuinely went away rather than moved.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from textual.widgets import Button, DataTable, Input, Select, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
    MockServerClient,
    rendered_row_cells,
)
from tldw_chatbook.Scheduling.events import ViewDefinitionResultsRequested
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientNotFoundError,
    ServerClientValidationError,
)
from tldw_chatbook.Widgets.detail_value_row import DetailValueRow
from tldw_chatbook.UI.Screens.scheduling.definition_audit_view import (
    DefinitionAuditView,
)
from tldw_chatbook.UI.Screens.scheduling.definition_detail import DefinitionDetail
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
)
from tldw_chatbook.UI.Screens.scheduling.workbench_host_screen import (
    WorkbenchHostScreen,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench


class AutomationsServerClient:
    """Stub scheduling server client with the automation control plane.

    The definition items carry `"owner_id": "1"` because that is what a
    real tldw_server puts on the wire (live-verified task 6, D1 -- its
    raw user id, NOT the client's `server:<id>` scope convention). Do not
    "fix" this to a prefixed value: the ingestion boundary derives the
    scope from the connection, and a prefixed fixture is exactly the
    drift that hid D1 for a whole slice. Same rule as
    `Tests/Scheduling/fixtures/server_responses/README.md`'s drift guard.
    """

    def __init__(self, notifications_service=None) -> None:
        self.notifications_service = notifications_service or object()
        # task-3 (schedules UAT remediation ruling 5): the capabilities
        # handshake -- defaults to "present" so every OTHER test in this
        # file (which predate the handshake) keeps exercising the real
        # audit call unhindered; the two `get_capabilities`-specific
        # tests below override this per case.
        self.get_capabilities = AsyncMock(return_value={"items": []})
        self.list_automation_definitions = AsyncMock(
            return_value={
                "items": [
                    {
                        "id": "def-1",
                        "owner_id": "1",
                        "name": "Morning brief",
                        "family": "recurring_question",
                        "lifecycle": "configured",
                        "health": "ready",
                    },
                    {
                        "id": "def-2",
                        "owner_id": "1",
                        "name": "Paused one",
                        "family": "recurring_question",
                        "lifecycle": "paused",
                        "health": "ready",
                    },
                ],
                "total": 2,
            }
        )
        self.run_automation_definition_now = AsyncMock(
            return_value={
                "definition_id": "def-1",
                "run_slot_utc": "slot-1",
                "job_id": 42,
                "deduped": False,
            }
        )
        self.list_automation_definition_audit = AsyncMock(
            return_value={
                "items": [
                    {
                        "id": "evt-2",
                        "definition_id": "def-1",
                        "event_type": "run_succeeded",
                        "actor": "automation:consumer",
                        "summary": "Run succeeded.",
                        "after": {"run_id": "run-2", "status": "succeeded"},
                        "created_at": "2026-08-30T00:30:00+00:00",
                    },
                    {
                        "id": "evt-1",
                        "definition_id": "def-1",
                        "event_type": "run_queued",
                        "actor": "automation:feed",
                        "summary": "Run queued.",
                        "after": {"run_id": "run-2"},
                        "created_at": "2026-08-30T00:29:50+00:00",
                    },
                ],
                "total": 2,
            }
        )


class AutomationsMockService(MockSchedulingServiceMixin):
    """Scheduling service whose server client knows automations."""

    def __init__(
        self,
        server_client,
        local_definitions=None,
        automation_runs=None,
        automation_results=None,
    ) -> None:
        self.owner_id = "local"
        self.server_client = server_client
        self.db = MockSchedulingDB(
            automation_definitions=local_definitions or [],
            automation_runs=automation_runs or [],
            automation_results=automation_results or [],
        )
        self.sync_engine = None
        # task-5 fix round: the PR-2 local run-now seam
        # (SchedulingService.run_automation_now) -- overridable per test.
        self.run_automation_now = AsyncMock(
            return_value={"run_id": "run-local-1", "deduped": False}
        )

    async def list_tasks(self, owner_id=None, include_projections=True):
        return []


def _local_definition(**overrides):
    """A local `automation_definitions` row (task-5 fix round fixtures)."""
    row = {
        "id": "local-def-1",
        "server_id": None,
        "owner_id": "local",
        "family": "recurring_question",
        "name": "Local digest",
        "lifecycle": "configured",
        # DB placeholder (never trusted -- the tab must recompute this via
        # automation_health.compute_local_health, not read it verbatim).
        "health": "execution_unavailable",
        "schedule": {"kind": "one_time", "run_at": "2099-01-01T00:00:00+00:00"},
        "input": {"question": "What's new?"},
        "config": {},
        "visibility_policy": {"mode": "findings_only"},
        "notification_policy": {},
        "approval_policy": {},
        "version": 1,
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": None,
    }
    row.update(overrides)
    return row


class _ServerRuntimeState:
    active_server_id = "server-1"


class _ServerRuntimePolicy:
    state = _ServerRuntimeState()


class AutomationsTestApp(ConsolidatedCSSApp):
    def __init__(self, service, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scheduling_service = service
        self.runtime_policy = _ServerRuntimePolicy()


class LocalOnlyTestApp(ConsolidatedCSSApp):
    scheduling_service = None


# -- queue-list readers (redesign PR-4 task 5) -------------------------------
#
# The retired Automations table was a flat, definitions-only listing whose
# row order matched the loader's own merge order, so tests indexed it
# directly. The unified queue mixes reminders in and SORTS (`sort_rows`),
# so a definition is found by id, never by a hard-coded position. Its
# columns are (glyph, Title, Details), so the Name cell the old table put
# at index 0 is index 1 here.


def _queue_table(workbench) -> DataTable:
    return workbench.query_one("#scheduling-task-table", DataTable)


def _queue_titles(workbench) -> list[str]:
    table = _queue_table(workbench)
    return [rendered_row_cells(table, i)[1] for i in range(table.row_count)]


def _queue_definitions(workbench) -> list[dict]:
    return [
        row.source_row for row in workbench._visible_rows if row.kind == "definition"
    ]


def _queue_row_index(workbench, definition_id: str) -> int:
    return next(
        index
        for index, row in enumerate(workbench._visible_rows)
        if row.kind == "definition"
        and str(row.source_row.get("id")) == definition_id
    )


async def _select_queue_definition(pilot, workbench, definition_id: str):
    """Put the queue cursor on `definition_id` and let its detail load."""
    table = _queue_table(workbench)
    table.cursor_coordinate = (_queue_row_index(workbench, definition_id), 0)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench.query_one(
        "#scheduling-queue-definition-detail", DefinitionDetail
    )


async def _settled_workbench(pilot):
    """Push the workbench and let its mount-time loads finish."""
    await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


@pytest.mark.asyncio
async def test_server_definitions_load_into_the_queue():
    """redesign PR-4 task 5: the server fetch lands in the unified queue.

    Two things changed with the retirement. The fetch count drops 2 -> 1:
    PR-2 noted that `load_tasks` had started fetching both definition
    halves ALONGSIDE the tab's own loader, on its own cadence -- with the
    tab's loader deleted, one fetch per mount is all that is left. And
    the selection the run-now/edit actions read is the unified row's
    (`_selected_row_id`), not the retired table's `_selected_automation_
    id`.

    The pane notice this used to assert ("2 automations on the server")
    is DELETED with `_automations_notice_text`: the queue's own notice
    (`#scheduling-pane-notice`) reports hidden panes/marks/glyphs, not a
    per-owner definition census, and inventing one here would be new
    copy, not a relocation.
    """
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)

        assert server_client.list_automation_definitions.await_count == 1
        assert _queue_table(workbench).row_count == 2

        # Highlighting a row records the selection the actions act on.
        await _select_queue_definition(pilot, workbench, "def-1")
        assert workbench._selected_row_id == "definition:def-1"
        assert workbench._selected_queue_definition()["id"] == "def-1"


@pytest.mark.asyncio
async def test_server_rows_are_rebound_to_the_connection_owner_scope():
    """Live verification task 6, D1: the server sends its own raw user id
    as `owner_id` ("1"), which is NOT the client's `server:<id>` scope.
    Every row from the server fetch must be rebound to the CONNECTION's
    scope at ingestion, or the whole downstream stack reads it as local:
    the Name cell says "[This device]", run-now routes to the local
    executor, and "move to this device" refuses as not-found.
    """
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)

        assert [
            row["owner_id"] for row in _queue_definitions(workbench)
        ] == ["server:server-1", "server:server-1"]

        first_cell = _queue_titles(workbench)[
            _queue_row_index(workbench, "def-1")
        ]
        assert first_cell == "[server-1] Morning brief"
        assert "This device" not in first_cell


@pytest.mark.asyncio
async def test_owner_prefix_and_bracket_name_render_literally():
    """Live verification task 6 round 2, D8.

    `DataTable` formats string cells with `rich.text.Text.from_markup`,
    whose tag regex matches `\\[[a-z#/@]...]`. A real server id is a base
    URL, so the Name cell reads `[http://127.0.0.1:8020] ...` -- `http`
    starts with a lowercase letter, the whole prefix parsed as a markup
    tag, and server rows rendered with NO ownership prefix at all while
    the pane's count line still said "1 automation on the server".

    Asserted on the PAINTED cells: `get_cell_at` returns the stored
    string and passes either way. Note the old `server:42` fixture value
    would have been eaten identically -- no fixture shape could have
    caught this, only a render-level assertion.
    """
    server_client = AutomationsServerClient()
    server_client.list_automation_definitions = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "def-1",
                    "owner_id": "1",
                    # A name with a lowercase tag token, which the same
                    # parser would eat independently of the prefix.
                    "name": "Nightly [bold] digest",
                    "family": "recurring_question",
                    "lifecycle": "configured",
                    "health": "ready",
                    "input": {
                        "provider": "custom-openai-api",
                        # A lowercase tag token -- `[2.5]` would NOT
                        # discriminate, rich only eats `[a-z#/@]...`.
                        "model": "[deprecated] Qwen2.5",
                    },
                },
            ],
            "total": 1,
        }
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    # The live shape: an active server id that IS a base URL.
    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_server_id="http://127.0.0.1:8020")
    )
    async with app.run_test(size=(200, 50)) as pilot:
        workbench = await _settled_workbench(pilot)

        assert _queue_titles(workbench) == [
            "[http://127.0.0.1:8020] Nightly [bold] digest"
        ]
        # redesign PR-4 task 5: the retired table's Model COLUMN carried
        # server-derived text too, and its escaping had to be pinned for
        # the same reason. That reading now lives on the definition pane's
        # Model row, so the second half of this claim follows it there
        # rather than being dropped.
        detail = await _select_queue_definition(pilot, workbench, "def-1")
        assert (
            _detail_text(detail, "scheduling-automation-detail-model")
            == "custom-openai-api/[deprecated] Qwen2.5"
        )


# redesign PR-4 task 5, two DELETIONS in this block, each because the
# surface under test is gone rather than moved:
#
# `test_automations_tab_shows_notice_without_server` pinned the retired
# pane notice's no-server wording ("Server automations need a connected
# server"), composed by `_automations_notice_text`. Both are deleted --
# the queue's own `#scheduling-pane-notice` is about hidden panes, marks
# and the glyph legend, and giving it a definitions census would be new
# copy rather than a relocation. The behaviour that mattered underneath
# (no server -> no server rows, local rows still listed) is still pinned
# by `test_local_automation_appears_with_recomputed_health` below, which
# runs with `notifications_service=None`.
#
# `test_run_now_on_automations_tab_dispatches_server_side` pinned `r` on
# the Automations tab dispatching server-side. Its claim is now covered
# verbatim by `test_run_now_on_queue_tab_routes_a_definition_row_to_its_
# owner` below -- same fixture, same `def-1`, same
# `run_automation_definition_now` assertion, reached through the surface
# that survived. Keeping both would be one test twice.


@pytest.mark.asyncio
async def test_server_run_now_refusal_surfaces_without_raising():
    """A `ServerClientValidationError` refusal is surfaced, not raised out
    of the action. (redesign PR-4 task 5: driven from the queue row --
    the Automations tab's own selection state is retired, so the
    still-selected assertion reads the unified row id.)"""
    server_client = AutomationsServerClient()
    server_client.run_automation_definition_now = AsyncMock(
        side_effect=ServerClientValidationError("definition_paused")
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        await _select_queue_definition(pilot, workbench, "def-1")

        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        server_client.run_automation_definition_now.assert_awaited_once()
        # The refusal path must not raise out of the action handler.
        assert workbench._selected_row_id == "definition:def-1"


@pytest.mark.asyncio
async def test_run_now_on_queue_tab_routes_a_definition_row_to_its_owner():
    """redesign PR-4, task 3 (supersedes this test's old pin): Queue
    definition rows now have their own run-now -- PR-2 ruling 1 made them
    action-less, but PR-4 ruling 1 gives every family a home on the Queue
    and task 3 wires run-now/edit onto those rows. `r` on the Queue's
    first row (the server-owned "Morning brief" definition, `def-1`)
    reaches the SAME server-run seam the Automations tab's own `r`
    uses, routed by owner exactly like `_run_automation_now` already
    routes there."""
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        # Queue tab is active by default; its first row is the
        # server-owned "Morning brief" definition (def-1).
        workbench.action_run_task_now()
        await pilot.pause()
        server_client.run_automation_definition_now.assert_awaited_once_with("def-1")


# redesign PR-4 task 5: the audit trail's three states move from the
# retired Automations-tab third pane (a DataTable + notice + title that
# reloaded on every row selection) to the pushed `DefinitionAuditView`
# (task 3), which fetches once on mount. The states themselves --
# events + count line, an honest empty state, and the no-server refusal
# -- are unchanged: both surfaces always shared `fetch_definition_audit`
# and `audit_notice_text`. The three tests below carry them over.
#
# `test_successful_run_now_refreshes_the_audit_trail` is DELETED, not
# carried over: it pinned the server run-now's immediate + 5s-delayed
# re-fetch INTO the always-mounted history pane, and that pane is what
# made the re-fetch necessary. A pushed view fetches on mount, so it can
# never be showing a trail that a just-dispatched run has outdated --
# there is nothing left to poke, and the workbench no longer calls the
# audit seam at all outside a push. What the run-now path does now
# instead (refresh the definitions) is pinned by
# `test_server_run_now_marks_definitions_stale` in
# `test_schedules_unified_list.py`.


async def _push_audit_view(pilot, service, definition):
    """Push a `DefinitionAuditView` the way `_push_definition_audit_
    overlay` does, and let its on-mount fetch finish."""
    await pilot.app.push_screen(
        WorkbenchHostScreen(
            lambda: DefinitionAuditView(service, dict(definition)),
            title="Run history",
        )
    )
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen.query_one(DefinitionAuditView)


@pytest.mark.asyncio
async def test_audit_view_lists_the_events_and_counts_them():
    server_client = AutomationsServerClient()
    service = AutomationsMockService(server_client)
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await _settled_workbench(pilot)
        server_client.list_automation_definition_audit.reset_mock()

        overlay = await _push_audit_view(
            pilot,
            service,
            {"id": "def-1", "name": "Morning brief", "owner_id": "server:server-1"},
        )

        server_client.list_automation_definition_audit.assert_awaited_once_with(
            "def-1"
        )
        table = overlay.query_one("#scheduling-audit-view-table", DataTable)
        notice = overlay.query_one("#scheduling-audit-view-notice")
        assert table.row_count == 2
        assert "2 events" in str(notice.content)


@pytest.mark.asyncio
async def test_audit_view_shows_empty_state_without_events():
    server_client = AutomationsServerClient()
    server_client.list_automation_definition_audit = AsyncMock(
        return_value={"items": [], "total": 0}
    )
    service = AutomationsMockService(server_client)
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await _settled_workbench(pilot)

        overlay = await _push_audit_view(
            pilot,
            service,
            {"id": "def-2", "name": "Paused one", "owner_id": "server:server-1"},
        )

        table = overlay.query_one("#scheduling-audit-view-table", DataTable)
        notice = overlay.query_one("#scheduling-audit-view-notice")
        assert table.row_count == 0
        assert "No recorded events" in str(notice.content)


@pytest.mark.asyncio
async def test_audit_view_shows_notice_when_capabilities_absent():
    """task-3 handshake, shape 1: the capabilities probe itself comes
    back absent (`get_capabilities` returns `None`, root-causes.md #7's
    "server predates Scheduled Tasks automation entirely") -- an honest
    notice, and the real audit call is never even attempted."""
    server_client = AutomationsServerClient()
    server_client.get_capabilities = AsyncMock(return_value=None)
    service = AutomationsMockService(server_client)
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await _settled_workbench(pilot)

        overlay = await _push_audit_view(
            pilot,
            service,
            {"id": "def-1", "name": "Morning brief", "owner_id": "server:server-1"},
        )

        server_client.list_automation_definition_audit.assert_not_awaited()
        notice = overlay.query_one("#scheduling-audit-view-notice")
        assert "does not support scheduled task automation" in str(notice.content)


@pytest.mark.asyncio
async def test_audit_view_shows_notice_when_audit_route_missing_despite_capabilities():
    """task-3 handshake, shape 2 (the actual UAT repro): capabilities are
    PRESENT (a mid-rollout server, new enough for the handshake, too old
    for this one route) -- a probe alone can't predict this, so the
    honest degrade comes from the real call's own 404 instead of a raw
    `scheduled_task_not_found` (UAT Minor 24)."""
    server_client = AutomationsServerClient()
    server_client.list_automation_definition_audit = AsyncMock(
        side_effect=ServerClientNotFoundError("scheduled_task_not_found")
    )
    service = AutomationsMockService(server_client)
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await _settled_workbench(pilot)

        overlay = await _push_audit_view(
            pilot,
            service,
            {"id": "def-1", "name": "Morning brief", "owner_id": "server:server-1"},
        )

        notice = overlay.query_one("#scheduling-audit-view-notice")
        assert "does not provide run history" in str(notice.content)


@pytest.mark.asyncio
async def test_audit_view_without_server_shows_notice():
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        overlay = await _push_audit_view(
            pilot,
            None,
            {"id": "def-1", "name": "Morning brief", "owner_id": "server:server-1"},
        )

        table = overlay.query_one("#scheduling-audit-view-table", DataTable)
        notice = overlay.query_one("#scheduling-audit-view-notice")
        assert table.row_count == 0
        assert "needs a connected server" in str(notice.content)


def test_execution_target_label_matrix():
    # redesign PR-4 task 5: imported from `definition_detail`, this
    # helper's real home. `schedules_workbench` used to re-export it
    # purely so this test's original import kept working, and that
    # re-export was the last consumer of the name there once the retired
    # Automations table (its only real caller) went -- so the import
    # moves and the re-export is deleted rather than kept alive with a
    # `noqa`.
    from tldw_chatbook.UI.Screens.scheduling.definition_detail import (
        automation_execution_target_label,
    )

    assert (
        automation_execution_target_label(
            {"input": {"provider": "openai", "model": "gpt-5"}}
        )
        == "openai/gpt-5"
    )
    assert automation_execution_target_label({"input": {"model": "gpt-5"}}) == "gpt-5"
    assert automation_execution_target_label({"input": {"provider": "mlx"}}) == "mlx"
    assert automation_execution_target_label({"input": {}}) == "auto"
    assert automation_execution_target_label({}) == "auto"
    # Blank strings and non-dict input fall through to the default chain.
    assert (
        automation_execution_target_label({"input": {"provider": "  ", "model": ""}})
        == "auto"
    )
    assert automation_execution_target_label({"input": "redacted"}) == "auto"


# redesign PR-4 task 5: `test_definitions_table_shows_the_model_column`
# is DELETED. It pinned the retired table's five-column shape
# (Name/Family/Lifecycle/Health/Model) and the Model cell's
# provider/model vs "auto" rendering. The unified queue is deliberately
# (glyph, Title, Details) -- PR-2 spec S4, "a single primitive's column
# set no longer fits a mixed reminder+definition list" -- so the columns
# themselves have no successor to assert. The READING survives in two
# places that already cover it: `automation_execution_target_label`'s own
# matrix (`test_execution_target_label_matrix` above, including the
# "auto" fallback) and the definition pane's Model row (asserted by
# `test_selecting_a_local_definition_paints_its_details_and_counts` and
# `test_owner_prefix_and_bracket_name_render_literally`).


# --- task-5 fix round: merged local + server listing ------------------------


@pytest.mark.asyncio
async def test_local_automation_appears_with_recomputed_health():
    """Local rows appear even without a connected server, and their Health
    cell is the freshly COMPUTED value (automation_health), never the DB's
    unreliable create-time placeholder."""
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)

        assert _queue_titles(workbench) == ["[This device] Local digest"]
        # No `library_rag_search_service` on the bare test app ->
        # capability_unavailable, NOT the DB row's stored
        # "execution_unavailable" placeholder. redesign PR-4 task 5: the
        # retired table had a Health COLUMN to read this from; the queue
        # does not, so the recomputed value is read off the row the queue
        # actually built (`_device_only_automations` stamps it, and that
        # stamping is what this test is about).
        assert _queue_definitions(workbench)[0]["health"] == (
            "capability_unavailable"
        )


@pytest.mark.asyncio
async def test_merged_list_shows_both_local_and_server_rows_with_owner_prefix():
    server_client = AutomationsServerClient()
    service = AutomationsMockService(
        server_client, local_definitions=[_local_definition(name="Local one")]
    )
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)

        # redesign PR-4 task 5: the queue SORTS its rows (`sort_rows`)
        # rather than preserving the loader's merge order, so the merge is
        # asserted as a set of painted titles rather than by position.
        assert sorted(_queue_titles(workbench)) == sorted(
            [
                "[This device] Local one",
                "[server-1] Morning brief",
                "[server-1] Paused one",
            ]
        )


@pytest.mark.asyncio
async def test_offline_server_owned_row_is_listed_as_pending_sync():
    """Final review I5: a "Runs on: Server" automation saved while offline
    is owned by `server:*` but has no `server_id` yet -- it used to fall
    between the local half (owner-filtered) and the server half (which has
    never heard of it) and so appeared in NEITHER."""
    pending_row = _local_definition(
        id="local-def-pending",
        owner_id="server:server-1",
        server_id=None,
        name="Queued digest",
    )
    server_client = AutomationsServerClient()
    service = AutomationsMockService(server_client, local_definitions=[pending_row])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)

        assert "[server-1 · pending sync] Queued digest" in _queue_titles(
            workbench
        )
        # Its `id` is the LOCAL one; editing must not treat it as a server
        # id and mirror it back as a second row.
        listed = next(
            row
            for row in _queue_definitions(workbench)
            if row["id"] == "local-def-pending"
        )
        assert (
            await workbench._resolve_local_definition_id(service, listed)
            == "local-def-pending"
        )


@pytest.mark.asyncio
async def test_run_now_routes_local_automation_through_the_service_seam():
    """Local AND server rows both present -- selecting the local one must
    route through the local seam and never touch the server client."""
    server_client = AutomationsServerClient()
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        assert _queue_table(workbench).row_count == 3  # 1 local + 2 server
        await _select_queue_definition(pilot, workbench, "local-def-1")

        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service.run_automation_now.assert_awaited_once_with("local-def-1")
        server_client.run_automation_definition_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_run_now_refusal_surfaces_without_raising():
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    service.run_automation_now = AsyncMock(return_value=None)
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        await _select_queue_definition(pilot, workbench, "local-def-1")

        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service.run_automation_now.assert_awaited_once()
        assert workbench._selected_row_id == "definition:local-def-1"


@pytest.mark.asyncio
async def test_local_automation_history_says_not_available_yet():
    """Local dispatch keeps no durable audit trail, and the view says so
    rather than showing an empty table. (redesign PR-4 task 5: read off
    the pushed `DefinitionAuditView`, which owns this notice now.)"""
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await _settled_workbench(pilot)

        overlay = await _push_audit_view(pilot, service, _local_definition())

        notice = overlay.query_one("#scheduling-audit-view-notice")
        assert "isn't available yet" in str(notice.content)


@pytest.mark.asyncio
async def test_refresh_after_local_save_shows_the_new_row():
    """The exact gap the review named: a local save must not read as a no-op."""
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        assert _queue_table(workbench).row_count == 0

        # Simulate what save_definition("local") just wrote to the DB.
        service.db._automation_definitions.append(_local_definition(name="Just saved"))

        from types import SimpleNamespace

        workbench._on_automation_form_result(
            SimpleNamespace(status="saved", definition_id="local-def-1")
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _queue_titles(workbench) == ["[This device] Just saved"]


# --- task-5 fix round: edit affordance ---------------------------------------


@pytest.mark.asyncio
async def test_edit_action_opens_form_prefilled_for_a_local_row():
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        await _select_queue_definition(pilot, workbench, "local-def-1")

        workbench.action_edit_task()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(pilot.app.screen, AutomationDefinitionForm)
        form = pilot.app.screen
        assert form.query_one("#automation-name", Input).value == "Local digest"
        runs_on = form.query_one("#automation-runs-on", Select)
        assert runs_on.disabled
        assert runs_on.value == "local"
        assert form._definition_id == "local-def-1"


@pytest.mark.asyncio
async def test_edit_action_refuses_agent_task_rows():
    server_client = AutomationsServerClient()
    server_client.list_automation_definitions = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "def-agent",
                    "name": "Agent one",
                    "family": "agent_task",
                    "lifecycle": "configured",
                    "health": "ready",
                    "owner_id": "server:1",
                }
            ],
            "total": 1,
        }
    )
    service = AutomationsMockService(server_client)
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        await _select_queue_definition(pilot, workbench, "def-agent")

        workbench.action_edit_task()
        await pilot.pause()

        assert isinstance(pilot.app.screen, SchedulesWorkbench)
        notifications = list(pilot.app._notifications)
        assert any("recurring-question" in n.message for n in notifications)


@pytest.mark.asyncio
async def test_edit_action_mirrors_a_server_only_row_on_demand():
    """A server row with no local shadow yet gets one created on the fly
    (via the same upsert the sync pull uses) so `save_definition`'s
    LOCAL-id contract for `definition_id` is honored."""
    server_client = AutomationsServerClient()
    service = AutomationsMockService(server_client)  # no local rows yet
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        workbench = await _settled_workbench(pilot)
        await _select_queue_definition(pilot, workbench, "def-1")

        assert service.db._automation_definitions == []

        workbench.action_edit_task()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(pilot.app.screen, AutomationDefinitionForm)
        assert len(service.db._automation_definitions) == 1
        mirrored = service.db._automation_definitions[0]
        assert mirrored["server_id"] == "def-1"
        assert mirrored["owner_id"] == "server:server-1"

        form = pilot.app.screen
        assert form._definition_id == mirrored["id"]
        runs_on = form.query_one("#automation-runs-on", Select)
        assert runs_on.disabled
        assert runs_on.value == "server:server-1"


@pytest.mark.asyncio
async def test_edit_save_reports_updated_not_created():
    """A save from the edit flow must say "updated", not the create-mode
    "created" wording -- both routes share one result handler."""
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        from types import SimpleNamespace

        workbench._on_automation_form_result(
            SimpleNamespace(status="saved", definition_id="local-def-1"),
            was_edit=True,
        )
        await pilot.pause()

        notifications = list(pilot.app._notifications)
        assert any(n.message == "Recurring question updated." for n in notifications)
        assert not any(
            n.message == "Recurring question created." for n in notifications
        )


# --- Definitions detail pane (schedules-redesign PR-1, Task 4) ------------
#
# `DefinitionDetail.set_definition` is a pure paint method (no I/O), so
# most field-value coverage lives in bare-widget unit tests that call it
# directly -- same shape as `test_schedules_workbench.py`'s
# `_BareTaskDetailApp`/`_frequency_reminder` precedent for Task 3's
# `TaskDetail` regrammar. The remaining tests drive a real row highlight
# through `SchedulesWorkbench` to prove the wiring end to end: both owner
# labels, the Task 2 count seams, and the off-thread read discipline.


class _BareDefinitionDetailApp(ConsolidatedCSSApp):
    """Bare app mounting one `DefinitionDetail`, matching
    `test_schedules_workbench.py`'s `_BareTaskDetailApp` pattern. `CSS_PATH`
    is pinned to the app bundle so `DetailValueRow`/`DetailGroup`'s real
    `css/features/_scheduling.tcss` styling resolves.
    """

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self):
        yield DefinitionDetail()


def _frequency_definition(**overrides) -> dict:
    """A representative local `recurring_question` definition covering
    every Details/Frequency row: a cron schedule, a non-UTC timezone, an
    explicit source scope, and a non-default generation mode/finding
    policy."""
    row = {
        "id": "def-freq",
        "owner_id": "local",
        "name": "Weekly digest",
        "family": "recurring_question",
        "lifecycle": "configured",
        "health": "ready",
        "input": {
            "question": "What changed this week?",
            "provider": "openai",
            "model": "gpt-5",
        },
        "schedule": {
            "kind": "cron",
            "cron": "0 9 * * 1",
            "timezone": "America/New_York",
        },
        "config": {
            "scope": {"mode": "sources", "sources": ["media_db", "notes"]},
            "generation_mode": "required",
        },
        "finding_policy": {"preset": "high_confidence_only"},
        # This client's writer emits booleans from the form's one "Notify
        # me about results" checkbox; the server sends per-outcome channel
        # strings instead (see the fixture-shaped test below).
        "notification_policy": {"on_success": True, "on_failure": True},
    }
    row.update(overrides)
    return row


def _detail_text(detail: DefinitionDetail, widget_id: str) -> str:
    return detail.query_one(f"#{widget_id}", Static).render_line(0).text.strip()


@pytest.mark.asyncio
async def test_definition_detail_renders_every_details_and_frequency_value():
    """Every Details/Frequency value paints through the grouped rows for a
    representative recurring definition (task-4 brief AC)."""
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_frequency_definition())
        await pilot.pause()

        assert (
            _detail_text(detail, "scheduling-automation-detail-question")
            == "What changed this week?"
        )
        assert _detail_text(detail, "scheduling-automation-detail-runs-on") == "This device"
        assert _detail_text(detail, "scheduling-automation-detail-model") == "openai/gpt-5"
        assert (
            _detail_text(detail, "scheduling-automation-detail-generation")
            == "Always generate a draft"
        )
        assert (
            _detail_text(detail, "scheduling-automation-detail-finding-policy")
            == "High confidence only"
        )
        assert (
            _detail_text(detail, "scheduling-automation-detail-sources")
            == "Media, Notes"
        )
        assert _detail_text(detail, "scheduling-automation-detail-repeat") == "Recurring"
        assert (
            _detail_text(detail, "scheduling-automation-detail-at")
            == "Weekly on Monday at 09:00 America/New_York"
        )
        assert (
            _detail_text(detail, "scheduling-automation-detail-timezone")
            == "America/New_York"
        )
        # Final review F8: spec §5 gives BOTH columns this Frequency row.
        assert (
            _detail_text(detail, "scheduling-automation-detail-notifications") == "On"
        )


@pytest.mark.asyncio
async def test_definition_detail_runs_on_shows_transfer_badge_when_in_flight():
    """'Runs on' appends the existing in-flight transfer badge text to the
    owner label -- same wording as the reminder detail pane's own suffix
    (mirrors `test_schedules_workbench.py`'s
    `test_task_detail_runs_on_shows_transfer_badge_when_in_flight`; fix
    round 1 finding 1: this case had no dedicated test)."""
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            _frequency_definition(transfer_state="to_server_pending")
        )
        await pilot.pause()

        assert (
            _detail_text(detail, "scheduling-automation-detail-runs-on")
            == "This device (Moving to server\u2026)"
        )


@pytest.mark.asyncio
async def test_definition_detail_renders_one_time_schedule_and_absent_keys():
    """A one-time schedule renders through the same rows; a definition
    with no `config`/`finding_policy`/`notification_policy` at all says
    so rather than blanking, crashing, or (final review F2) presenting
    the create/edit form's DEFAULTS as if they were readings."""
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            {
                "id": "def-one-time",
                "owner_id": "local",
                "name": "One-off check",
                "input": {"question": "Did the migration finish?"},
                "schedule": {"kind": "one_time", "run_at": "2026-09-10T14:30:00+00:00"},
            }
        )
        await pilot.pause()

        assert _detail_text(detail, "scheduling-automation-detail-repeat") == "One-time"
        assert (
            _detail_text(detail, "scheduling-automation-detail-at")
            == "One-time at 2026-09-10 14:30 UTC"
        )
        assert _detail_text(detail, "scheduling-automation-detail-timezone") == "UTC"
        assert _detail_text(detail, "scheduling-automation-detail-model") == "auto"
        assert (
            _detail_text(detail, "scheduling-automation-detail-generation") == "Not set"
        )
        assert (
            _detail_text(detail, "scheduling-automation-detail-finding-policy")
            == "Not set"
        )
        assert _detail_text(detail, "scheduling-automation-detail-sources") == "Not set"
        assert (
            _detail_text(detail, "scheduling-automation-detail-notifications")
            == "Not set"
        )


@pytest.mark.asyncio
async def test_definition_detail_question_card_renders_brackets_literally():
    """A bracket-bearing question must never be interpreted as Rich markup
    (task-4 brief: escape discipline, bracket-bearing test required)."""
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            _frequency_definition(input={"question": "What's the [bold] status?"})
        )
        await pilot.pause()

        assert (
            _detail_text(detail, "scheduling-automation-detail-question")
            == "What's the [bold] status?"
        )


@pytest.mark.asyncio
async def test_definition_detail_shows_counts_and_last_run_outcome():
    """Run count / unread-results count / last-run outcome paint from the
    Task 2 seams, and the History group starts collapsed."""
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(
            _frequency_definition(),
            run_count=2,
            last_run={
                "status": "failed",
                "started_at": "2026-08-20T09:00:00+00:00",
                "ended_at": "2026-08-20T09:00:05+00:00",
            },
            unread_count=3,
        )
        await pilot.pause()

        history_group = detail.query_one("#scheduling-automation-detail-group-history")
        assert history_group.collapsed is True
        history_group.collapsed = False
        await pilot.pause()

        assert _detail_text(detail, "scheduling-automation-detail-run-count") == "2"
        assert (
            _detail_text(detail, "scheduling-automation-detail-unread-results") == "3"
        )
        last_run_text = _detail_text(detail, "scheduling-automation-detail-last-run")
        assert "failed" in last_run_text
        assert "2026-08-20 09:00" in last_run_text
        # redesign PR-4, task 2: the separate "Results -- See Results
        # tab" row (a dangling pointer once the tab bar retires) is gone
        # -- the unread-results row itself is now the live activation
        # (see test_unread_row_activation_requests_definition_results
        # below).
        assert detail._unread_row.affordance is True


class _CapturingDefinitionDetailApp(_BareDefinitionDetailApp):
    """`_BareDefinitionDetailApp` + captures `ViewDefinitionResultsRequested`
    messages bubbled up to the App (redesign PR-4, task 2) -- there is no
    `SchedulesWorkbench` mounted here to route them further, this only
    proves `DefinitionDetail` posts the right message."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.captured: list[ViewDefinitionResultsRequested] = []

    def on_view_definition_results_requested(
        self, event: ViewDefinitionResultsRequested
    ) -> None:
        self.captured.append(event)


@pytest.mark.asyncio
async def test_unread_row_activation_requests_definition_results():
    """redesign PR-4, task 2: activating the `Unread results` row posts
    `ViewDefinitionResultsRequested` carrying the currently-painted
    definition -- the live replacement for the retired "See Results tab"
    pointer."""
    app = _CapturingDefinitionDetailApp()
    async with app.run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        definition = _frequency_definition()
        detail.set_definition(definition, unread_count=2)
        await pilot.pause()

        row = detail._unread_row
        assert row.affordance is True
        assert row.can_focus is True
        row.post_message(DetailValueRow.Activated(row))
        await pilot.pause()

        assert len(pilot.app.captured) == 1
        assert pilot.app.captured[0].definition["id"] == definition["id"]


@pytest.mark.asyncio
async def test_definition_detail_clears_to_empty_state_for_none():
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(_frequency_definition())
        await pilot.pause()
        body = detail.query_one("#scheduling-automation-detail-body")
        assert body.display is True

        detail.set_definition(None)
        await pilot.pause()
        assert body.display is False
        empty_state = detail.query_one(
            "#scheduling-automation-detail-empty-state", Static
        )
        assert empty_state.display is True


@pytest.mark.asyncio
async def test_selecting_a_local_definition_paints_its_details_and_counts():
    """Integration path: a real row highlight through `SchedulesWorkbench`
    feeds the detail pane, off-thread reads included (task-4 brief AC)."""
    server_client = AutomationsServerClient()
    definition = _frequency_definition(id="local-def-freq")
    app = AutomationsTestApp(
        AutomationsMockService(
            server_client,
            local_definitions=[definition],
            automation_runs=[
                {
                    "id": "run-1",
                    "owner_id": "local",
                    "definition_id": "local-def-freq",
                    "status": "succeeded",
                    "created_at": "2026-08-20T09:00:00+00:00",
                    "ended_at": "2026-08-20T09:00:05+00:00",
                },
                {
                    "id": "run-2",
                    "owner_id": "local",
                    "definition_id": "local-def-freq",
                    "status": "failed",
                    "created_at": "2026-08-13T09:00:00+00:00",
                    "ended_at": "2026-08-13T09:00:05+00:00",
                },
            ],
            automation_results=[
                {
                    "id": "res-1",
                    "owner_id": "local",
                    "definition_id": "local-def-freq",
                    "review_state": "unread",
                    "created_at": "2026-08-20T09:00:10+00:00",
                },
            ],
        )
    )
    # Wide terminal (task-4 fix round): a pane that paints at a zero
    # region legitimately blanks `render_line(0)` (proven empty, not a
    # stored-attribute false pass: `.content` still reads correctly, only
    # the PAINT is absent -- see `test_detail_value_row.py`'s own
    # painted-not-stored discipline). redesign PR-4 task 5: the tab
    # activation that used to give the pane a region is retired; the
    # queue's detail pane is on screen from mount, so selecting the row
    # is enough.
    async with app.run_test(size=(200, 50)) as pilot:
        workbench = await _settled_workbench(pilot)
        detail = await _select_queue_definition(
            pilot, workbench, "local-def-freq"
        )

        assert _detail_text(detail, "scheduling-automation-detail-runs-on") == "This device"
        assert _detail_text(detail, "scheduling-automation-detail-model") == "openai/gpt-5"

        # History starts collapsed (spec §5) -- its rows paint at zero
        # region until expanded, same discipline Task 3's own
        # `last_fire_row` assertion required.
        history_group = detail.query_one("#scheduling-automation-detail-group-history")
        history_group.collapsed = False
        await pilot.pause()
        assert _detail_text(detail, "scheduling-automation-detail-run-count") == "2"
        assert (
            _detail_text(detail, "scheduling-automation-detail-unread-results") == "1"
        )
        assert "succeeded" in _detail_text(detail, "scheduling-automation-detail-last-run")


@pytest.mark.asyncio
async def test_selecting_a_server_definition_shows_its_server_owner_label():
    """A server-mirrored row's 'Runs on' shows the SERVER's owner label
    (task-4 brief AC), and never claims local run history it cannot have
    (`automation_runs` is local-only)."""
    server_client = AutomationsServerClient()
    server_client.list_automation_definitions = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "def-server-freq",
                    "owner_id": "1",
                    "name": "Server digest",
                    "family": "recurring_question",
                    "lifecycle": "configured",
                    "health": "ready",
                    "input": {
                        "question": "What shipped?",
                        "provider": "anthropic",
                        "model": "claude",
                    },
                    # WIRE shape: the real server sends `expression`, not
                    # `cron` (final review F1, recorded fixture
                    # `automation_definition_list.json`). Do not "fix"
                    # this to the client's own key -- that drift is what
                    # let `At: -` ship for every server definition.
                    "schedule": {
                        "kind": "cron",
                        "expression": "0 9 * * 1",
                        "timezone": "UTC",
                    },
                    "config": {"scope": {"mode": "all_searchable_library"}},
                },
            ],
            "total": 1,
        }
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    # Wide terminal -- see the local-definition test above.
    async with app.run_test(size=(200, 50)) as pilot:
        workbench = await _settled_workbench(pilot)
        detail = await _select_queue_definition(
            pilot, workbench, "def-server-freq"
        )

        assert _detail_text(detail, "scheduling-automation-detail-runs-on") == "server-1"
        assert (
            _detail_text(detail, "scheduling-automation-detail-model")
            == "anthropic/claude"
        )
        # Final review F1: the wire's `expression` key reaches the At row.
        assert (
            _detail_text(detail, "scheduling-automation-detail-at")
            == "Weekly on Monday at 09:00 UTC"
        )

        # History starts collapsed -- expand before reading its rows.
        history_group = detail.query_one("#scheduling-automation-detail-group-history")
        history_group.collapsed = False
        await pilot.pause()
        # Final review F3: `automation_runs` is local-only, so a server
        # row has no local counts to report -- and "Never run"/"0" here
        # contradicted the run-history pane beside it, which was listing
        # that same definition's server audit trail.
        assert (
            _detail_text(detail, "scheduling-automation-detail-run-count")
            == "Kept on the server"
        )
        assert (
            _detail_text(detail, "scheduling-automation-detail-last-run")
            == "Kept on the server — see Run history"
        )


@pytest.mark.asyncio
async def test_definition_detail_counts_are_read_off_the_event_loop():
    """schedules-redesign PR-1, task-4 AC: the pane's DB reads go through
    `asyncio.to_thread`, the same discipline `_load_local_automations`
    already uses for a `service.db.*` call made from inside a worker
    coroutine.

    `DataTable` auto-highlights row 0 the moment rows land under its
    default (0, 0) cursor, so the FIRST detail load happens during the
    initial mount/`load_automations()` pass -- the patch must wrap that
    pass, not a later re-assignment of the same coordinate (which is a
    no-op: `_on_automations_row_highlighted` skips an unchanged id).
    """
    server_client = AutomationsServerClient()
    definition = _frequency_definition(id="local-def-freq")
    app = AutomationsTestApp(
        AutomationsMockService(server_client, local_definitions=[definition])
    )
    async with app.run_test() as pilot:
        with patch(
            "tldw_chatbook.UI.Screens.scheduling.schedules_workbench.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as spy:
            await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
            await pilot.pause()
            await pilot.pause()

        spy.assert_awaited()


def _recorded_server_definition() -> dict:
    """Item 0 of this repo's own RECORDED real-server definition list,
    scoped the way `_load_server_automations` scopes it (`owner_id`
    stamped from the connection, everything else passed through raw).

    Hand-written dicts are what hid final review F1/F2 for a whole task:
    every branch test wrote the CLIENT's payload shape (`schedule.cron`,
    a full `config`), so the pane looked correct against a payload the
    server never sends.
    """
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "Scheduling/fixtures/server_responses/automation_definition_list.json"
    )
    definition = json.loads(path.read_text())["items"][0]
    definition["owner_id"] = "server:server-1"
    return definition


@pytest.mark.asyncio
async def test_definition_detail_reads_the_recorded_server_payload_honestly():
    """Final review F1 + F2, against the recorded fixture rather than a
    hand-written dict.

    F1: the server sends `schedule.expression`; reading only `cron`
    rendered `At: -` for EVERY server-owned definition.
    F2: the server's `config` carries none of the create-form's keys, and
    substituting that form's defaults presented a guess as a reading --
    a definition actually configured `high_confidence_only` would have
    rendered "Finding policy: Balanced findings".
    """
    definition = _recorded_server_definition()
    # Guard the premise: if the recorded payload ever grows these keys,
    # this test is no longer testing what it claims to.
    assert definition["schedule"]["expression"] == "0 9 * * 1-5"
    assert "cron" not in definition["schedule"]
    assert "generation_mode" not in definition["config"]
    assert "scope" not in definition["config"]
    assert "finding_policy" not in definition

    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.set_definition(definition)
        await pilot.pause()

        assert _detail_text(detail, "scheduling-automation-detail-repeat") == "Recurring"
        assert (
            _detail_text(detail, "scheduling-automation-detail-at")
            == "Weekdays at 09:00 UTC"
        )
        assert _detail_text(detail, "scheduling-automation-detail-timezone") == "UTC"
        # The server's per-outcome channel strings, not this client's bools.
        assert (
            _detail_text(detail, "scheduling-automation-detail-notifications")
            == "silent on success · toast on failure"
        )
        for row_id in ("generation", "finding-policy", "sources"):
            assert (
                _detail_text(detail, f"scheduling-automation-detail-{row_id}")
                == "Not set"
            ), row_id


@pytest.mark.asyncio
async def test_definition_detail_history_is_owner_honest():
    """Final review F3: local counts are the truth only for a LOCAL row.

    `automation_runs` has one writer (local dispatch) and no server
    mirror, so a server-owned definition's local counts are structurally
    zero -- "Never run"/"0" was a claim, not a reading, and it sat beside
    a run-history pane showing the server's real audit trail.
    """
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        # History starts collapsed (spec §5) and its rows paint at a zero
        # region until expanded -- the same discipline every other History
        # assertion in this file follows.
        detail.query_one(
            "#scheduling-automation-detail-group-history"
        ).collapsed = False

        detail.set_definition(_frequency_definition(), run_count=0, last_run=None)
        await pilot.pause()
        assert (
            _detail_text(detail, "scheduling-automation-detail-last-run") == "Never run"
        )
        assert _detail_text(detail, "scheduling-automation-detail-run-count") == "0"

        detail.set_definition(
            _frequency_definition(owner_id="server:server-1"),
            run_count=0,
            last_run=None,
        )
        await pilot.pause()
        assert (
            _detail_text(detail, "scheduling-automation-detail-last-run")
            == "Kept on the server — see Run history"
        )
        assert (
            _detail_text(detail, "scheduling-automation-detail-run-count")
            == "Kept on the server"
        )

        # Authored offline against the server scope: the server has never
        # heard of it, so it has no history there either.
        detail.set_definition(
            _frequency_definition(owner_id="server:server-1", pending_sync=True),
            run_count=0,
            last_run=None,
        )
        await pilot.pause()
        assert (
            _detail_text(detail, "scheduling-automation-detail-last-run")
            == "Not synced to the server yet"
        )


@pytest.mark.asyncio
async def test_definition_detail_says_so_when_the_history_read_failed():
    """Final review F14: a failed count read must not be indistinguishable
    from a genuinely empty history."""
    async with _BareDefinitionDetailApp().run_test() as pilot:
        detail = pilot.app.query_one(DefinitionDetail)
        detail.query_one(
            "#scheduling-automation-detail-group-history"
        ).collapsed = False
        detail.set_definition(_frequency_definition(), history_error=True)
        await pilot.pause()

        for row_id in ("last-run", "run-count", "unread-results"):
            assert (
                _detail_text(detail, f"scheduling-automation-detail-{row_id}")
                == "Couldn't load — see the log"
            ), row_id


@pytest.mark.asyncio
async def test_a_failed_count_read_paints_the_error_not_zeros():
    """Same as above, driven through `SchedulesWorkbench`'s own read path
    (the `except` branch of `_fetch_definition_detail_counts`, reached via
    `_load_queue_definition_detail` -- redesign PR-4 task 5 retired the
    Automations-tab loader that was the other caller)."""
    server_client = AutomationsServerClient()
    definition = _frequency_definition(id="local-def-freq")
    service = AutomationsMockService(server_client, local_definitions=[definition])

    def _boom(*args, **kwargs):
        raise RuntimeError("sqlite is unhappy")

    service.db.count_automation_runs = _boom
    app = AutomationsTestApp(service)
    async with app.run_test(size=(200, 50)) as pilot:
        workbench = await _settled_workbench(pilot)
        detail = await _select_queue_definition(
            pilot, workbench, "local-def-freq"
        )

        history_group = detail.query_one("#scheduling-automation-detail-group-history")
        history_group.collapsed = False
        await pilot.pause()
        assert (
            _detail_text(detail, "scheduling-automation-detail-run-count")
            == "Couldn't load — see the log"
        )


@pytest.mark.asyncio
async def test_editing_the_selected_definition_refreshes_the_detail_pane():
    """Final review F4: a table refresh that lands on the SAME row id must
    still re-feed the detail pane -- the row's DATA may have changed.

    The `RowHighlighted` handler early-returns on an unchanged id, so
    after an edit-save the table cell updated while the pane beside it
    kept painting the pre-edit model.
    """
    server_client = AutomationsServerClient()
    definition = _frequency_definition(id="local-def-freq")
    service = AutomationsMockService(server_client, local_definitions=[definition])
    app = AutomationsTestApp(service)
    async with app.run_test(size=(200, 50)) as pilot:
        workbench = await _settled_workbench(pilot)
        detail = await _select_queue_definition(
            pilot, workbench, "local-def-freq"
        )
        assert _detail_text(detail, "scheduling-automation-detail-model") == "openai/gpt-5"

        # Edit-and-save shape: the stored row changes, the id does not.
        service.db._automation_definitions[0]["input"] = {
            "question": "What changed this week?",
            "provider": "anthropic",
            "model": "claude-x",
        }
        # redesign PR-4 task 5: `load_automations()` (the retired tab's
        # own reloader) is deleted; the equivalent authoritative repaint
        # is the queue's own detail re-feed, which is what every
        # definition mutation path now calls after a successful save.
        await workbench._repaint_queue_definition_detail(
            service, "local-def-freq", service.db._automation_definitions[0]
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert (
            _detail_text(detail, "scheduling-automation-detail-model")
            == "anthropic/claude-x"
        )


# redesign PR-4 task 5: `test_automations_panes_stay_on_screen_across_
# the_detail_hide_boundary` is DELETED. It pinned final review F5 -- the
# Automations tab's own THREE-pane split (list | definition detail | run
# history) laying panes off-screen in the 84-89 band because 3 x
# min-width 30 could not fit. All three panes are retired: the list
# merged into the queue, the detail pane is the queue's own sibling, and
# the history pane became the pushed `DefinitionAuditView`. The geometry
# claim has no subject left here -- the queue's own three-pane fit at
# that boundary is separately pinned (its `hide_detail`/`hide_inspector`
# thresholds), and the responsive floor as a whole is Task 6's brief.


# --- redesign PR-4, task 3: Queue definition-row actions + all-families
# listing + audit-view relocation ------------------------------------------


def _isolated_local_service(definitions):
    """An `AutomationsMockService` whose server side returns NO
    definitions -- isolates the Queue's first row to a single LOCAL
    definition so a test can assert on a deterministic cursor position
    without also contending with the server fixture's own two rows."""
    server_client = AutomationsServerClient()
    server_client.list_automation_definitions = AsyncMock(
        return_value={"items": [], "total": 0}
    )
    return AutomationsMockService(server_client, local_definitions=definitions), server_client


@pytest.mark.asyncio
async def test_run_now_on_queue_tab_dispatches_a_local_definition_locally():
    """PR-4 task 3: the Queue's own run-now routes a LOCAL definition
    through `SchedulingService.run_automation_now` (never the server
    client) -- the same owner routing `_run_automation_now` already
    applies for the Automations tab's own `r`."""
    service, server_client = _isolated_local_service([_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service.run_automation_now.assert_awaited_once_with("local-def-1")
        server_client.run_automation_definition_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_on_queue_tab_opens_the_form_for_a_recurring_question_row():
    """PR-4 task 3: `e` on a Queue definition row opens the SAME
    `AutomationDefinitionForm` the Automations tab's own `e` opens,
    prefilled for that row."""
    service, _server_client = _isolated_local_service([_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_edit_task()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(pilot.app.screen, AutomationDefinitionForm)


@pytest.mark.asyncio
async def test_edit_on_queue_tab_refuses_a_non_recurring_question_row():
    """PR-4 task 3: an `agent_task` Queue row's `e` refuses honestly
    (the SAME copy the Automations tab's own family gate already gives)
    instead of opening a form that only knows how to author `recurring_
    question`."""
    agent_def = _local_definition(
        id="local-agent-1", family="agent_task", name="Nightly agent run"
    )
    service, _server_client = _isolated_local_service([agent_def])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_edit_task()
        await pilot.pause()

        assert pilot.app.screen is workbench
        notifications = list(pilot.app._notifications)
        assert any("recurring-question" in n.message for n in notifications), [
            n.message for n in notifications
        ]


@pytest.mark.asyncio
async def test_agent_task_queue_row_is_visible_and_read_only_with_honest_note():
    """PR-4 ruling 1 + task 3: an `agent_task` definition now has a home
    on the Queue (it was invisible there entirely before PR-4) and
    renders with `DefinitionDetail`'s existing `_UNSUPPORTED_FAMILY_NOTE`
    fallback -- no editable row lights up for it, the same as on the
    Automations tab (`test_non_recurring_question_definition_exposes_no_
    editors`)."""
    agent_def = _local_definition(
        id="local-agent-1", family="agent_task", name="Nightly agent run"
    )
    service, _server_client = _isolated_local_service([agent_def])
    app = AutomationsTestApp(service)
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        table = workbench.query_one("#scheduling-task-table", DataTable)
        # Column 0 is the glyph, column 1 is "Title" (`add_columns("",
        # "Title", "Details")`) -- prefixed with the owner label
        # (`automation_name_cell`).
        assert "Nightly agent run" in rendered_row_cells(table, 0)[1]
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        detail = workbench.query_one(
            "#scheduling-queue-definition-detail", DefinitionDetail
        )
        assert detail._definition["family"] == "agent_task"
        assert [
            row.row_key for row in detail._editable_rows() if row.affordance
        ] == []
        why = detail.query_one("#scheduling-automation-detail-why", Static)
        assert "isn't a recurring question" in why.render_line(0).text


@pytest.mark.asyncio
async def test_last_run_row_activation_pushes_audit_view_with_painted_events():
    """PR-4 task 3 (audit-view relocation): the Queue's own
    `DefinitionDetail` sibling's `Last run` row activation pushes a
    `DefinitionAuditView` scoped to the highlighted (server-owned)
    definition, reusing the SAME `list_automation_definition_audit` seam
    the retiring Automations-tab pane already calls."""
    server_client = AutomationsServerClient()
    service = AutomationsMockService(server_client)
    app = AutomationsTestApp(service)
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        detail = workbench.query_one(
            "#scheduling-queue-definition-detail", DefinitionDetail
        )
        row = detail._last_run_row
        assert row.affordance is True
        row.post_message(DetailValueRow.Activated(row))
        await pilot.pause()

        assert isinstance(pilot.app.screen, WorkbenchHostScreen)
        assert str(pilot.app.screen.title).startswith("Run history — Morning brief")
        overlay = pilot.app.screen.query_one(DefinitionAuditView)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        audit_table = overlay.query_one("#scheduling-audit-view-table", DataTable)
        assert audit_table.row_count == 2
        assert rendered_row_cells(audit_table, 0)[2] == "Run succeeded."


@pytest.mark.asyncio
async def test_run_now_button_on_queue_definition_pane_dispatches_locally():
    """PR-4 task 3, ruling 2: the retired Automations-tab `r` key
    relocates to a `Run now` button beside Pause/Resume on the pane
    itself -- pressing it on the Queue's own `DefinitionDetail` sibling
    reaches the SAME local dispatch seam the key used to."""
    service, _server_client = _isolated_local_service([_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        table = workbench.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        detail = workbench.query_one(
            "#scheduling-queue-definition-detail", DefinitionDetail
        )
        run_now = detail.query_one("#scheduling-automation-run-now", Button)
        detail.on_button_pressed(Button.Pressed(run_now))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service.run_automation_now.assert_awaited_once_with("local-def-1")
