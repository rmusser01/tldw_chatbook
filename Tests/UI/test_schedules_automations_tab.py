"""Automations tab behavior on the Schedules workbench (task-18940 slice 2).

The tab surfaces both the server's automation definitions (ADR-077, ``r``
dispatches through the server control plane) and, since task-5's fix
round, this device's own local-owned `recurring_question` definitions
(``r`` routes those through `SchedulingService.run_automation_now`
instead -- never the server client, and never both).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from textual.widgets import DataTable, Input, Select, Static, TabbedContent

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
    MockServerClient,
    rendered_row_cells,
)
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientValidationError,
)
from tldw_chatbook.UI.Screens.scheduling.definition_detail import DefinitionDetail
from tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form import (
    AutomationDefinitionForm,
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

    async def list_tasks(self, owner_id=None):
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


async def _mounted_workbench(app):
    """Run the app and return (pilot context, screen) with the workbench up."""
    return app


@pytest.mark.asyncio
async def test_automations_tab_loads_server_definitions():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        notice = workbench.query_one("#scheduling-automations-notice")

        # redesign PR-2, Task 2: `load_tasks` (the Queue tab's own unified
        # loader) now ALSO fetches both definition halves on every mount
        # (its own separate cadence, per the brief -- not a shared cache
        # with this tab's `load_automations`), so the server fetch is
        # awaited twice, not once, at mount.
        assert server_client.list_automation_definitions.await_count == 2
        assert table.row_count == 2
        assert "2 automations on the server" in str(notice.content)
        # Highlighting a row records the selection Run-now will act on.
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        assert workbench._selected_automation_id == "def-1"


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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        assert [row["owner_id"] for row in workbench._automations] == [
            "server:server-1",
            "server:server-1",
        ]

        table = workbench.query_one("#scheduling-automations-table", DataTable)
        first_cell = rendered_row_cells(table, 0)[0]
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
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        table = pilot.app.screen.query_one(
            "#scheduling-automations-table", DataTable
        )

        cells = rendered_row_cells(table, 0)
        assert cells[0] == "[http://127.0.0.1:8020] Nightly [bold] digest"
        # The Model column carries server-derived text too.
        assert cells[4] == "custom-openai-api/[deprecated] Qwen2.5"


@pytest.mark.asyncio
async def test_automations_tab_shows_notice_without_server():
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        notice = workbench.query_one("#scheduling-automations-notice")
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        assert table.row_count == 0
        assert "need a connected server" in str(notice.content)


@pytest.mark.asyncio
async def test_run_now_on_automations_tab_dispatches_server_side():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_run_task_now()
        await pilot.pause()

        server_client.run_automation_definition_now.assert_awaited_once_with("def-1")


@pytest.mark.asyncio
async def test_run_now_refusal_surfaces_without_raising():
    server_client = AutomationsServerClient()
    server_client.run_automation_definition_now = AsyncMock(
        side_effect=ServerClientValidationError("definition_paused")
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_run_task_now()
        await pilot.pause()

        server_client.run_automation_definition_now.assert_awaited_once()
        # The refusal path must not raise out of the action handler.
        assert workbench._selected_automation_id == "def-1"


@pytest.mark.asyncio
async def test_run_now_on_queue_tab_never_reaches_server_client():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen

        # Queue tab is active by default; r must not reach the server client.
        workbench.action_run_task_now()
        await pilot.pause()
        server_client.run_automation_definition_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_selecting_a_definition_loads_its_audit_trail():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        history = workbench.query_one(
            "#scheduling-automation-history-table", DataTable
        )
        notice = workbench.query_one("#scheduling-automation-history-notice")
        title = workbench.query_one("#scheduling-automation-history-title")

        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        server_client.list_automation_definition_audit.assert_awaited_once_with(
            "def-1"
        )
        assert history.row_count == 2
        assert "Run history — Morning brief" in str(title.content)
        assert "2 events" in str(notice.content)


@pytest.mark.asyncio
async def test_audit_trail_shows_empty_state_without_events():
    server_client = AutomationsServerClient()
    server_client.list_automation_definition_audit = AsyncMock(
        return_value={"items": [], "total": 0}
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        history = workbench.query_one(
            "#scheduling-automation-history-table", DataTable
        )
        notice = workbench.query_one("#scheduling-automation-history-notice")

        table.cursor_coordinate = (1, 0)
        await pilot.pause()

        assert history.row_count == 0
        assert "No recorded events" in str(notice.content)


@pytest.mark.asyncio
async def test_audit_trail_without_server_shows_notice():
    app = LocalOnlyTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        history = workbench.query_one(
            "#scheduling-automation-history-table", DataTable
        )
        notice = workbench.query_one("#scheduling-automation-history-notice")

        assert history.row_count == 0
        assert "needs a connected server" in str(notice.content)


@pytest.mark.asyncio
async def test_successful_run_now_refreshes_the_audit_trail():
    server_client = AutomationsServerClient()
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        assert server_client.list_automation_definition_audit.await_count == 1

        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.pause()

        # Selection load + post-dispatch refresh.
        assert server_client.list_automation_definition_audit.await_count == 2
        assert (
            server_client.list_automation_definition_audit.await_args.args[-1]
            == "def-1"
        )


def test_execution_target_label_matrix():
    from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
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


@pytest.mark.asyncio
async def test_definitions_table_shows_the_model_column():
    server_client = AutomationsServerClient()
    server_client.list_automation_definitions = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "def-1",
                    "name": "Pinned",
                    "family": "recurring_question",
                    "lifecycle": "configured",
                    "health": "ready",
                    "input": {"provider": "anthropic", "model": "claude-x"},
                },
                {
                    "id": "def-2",
                    "name": "Default",
                    "family": "recurring_question",
                    "lifecycle": "configured",
                    "health": "ready",
                },
            ],
            "total": 2,
        }
    )
    app = AutomationsTestApp(AutomationsMockService(server_client))
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)

        assert [c.label.plain for c in table.columns.values()] == [
            "Name",
            "Family",
            "Lifecycle",
            "Health",
            "Model",
        ]
        assert rendered_row_cells(table, 0)[4] == "anthropic/claude-x"
        assert rendered_row_cells(table, 1)[4] == "auto"


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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        notice = workbench.query_one("#scheduling-automations-notice")

        assert table.row_count == 1
        assert rendered_row_cells(table, 0)[0] == "[This device] Local digest"
        # No `library_rag_search_service` on the bare test app -> capability_unavailable,
        # NOT the DB row's stored "execution_unavailable" placeholder.
        assert rendered_row_cells(table, 0)[3] == "capability_unavailable"
        assert "1 on this device" in str(notice.content)


@pytest.mark.asyncio
async def test_merged_list_shows_both_local_and_server_rows_with_owner_prefix():
    server_client = AutomationsServerClient()
    service = AutomationsMockService(
        server_client, local_definitions=[_local_definition(name="Local one")]
    )
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)

        assert table.row_count == 3
        names = [rendered_row_cells(table, i)[0] for i in range(3)]
        assert names[0] == "[This device] Local one"
        assert names[1] == "[server-1] Morning brief"
        assert names[2] == "[server-1] Paused one"


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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)

        names = [rendered_row_cells(table, i)[0] for i in range(table.row_count)]
        assert names[0] == "[server-1 · pending sync] Queued digest"
        # Its `id` is the LOCAL one; editing must not treat it as a server
        # id and mirror it back as a second row.
        listed = workbench._automations[0]
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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        assert table.row_count == 3  # local row first, then the 2 server rows
        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        assert workbench._selected_automation_id == "local-def-1"

        workbench.action_run_task_now()
        await pilot.pause()
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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.pause()

        service.run_automation_now.assert_awaited_once()
        assert workbench._selected_automation_id == "local-def-1"


@pytest.mark.asyncio
async def test_local_automation_history_says_not_available_yet():
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        notice = workbench.query_one("#scheduling-automation-history-notice")

        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        assert "isn't available yet" in str(notice.content)


@pytest.mark.asyncio
async def test_refresh_after_local_save_shows_the_new_row():
    """The exact gap the review named: a local save must not read as a no-op."""
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        assert table.row_count == 0

        # Simulate what save_definition("local") just wrote to the DB.
        service.db._automation_definitions.append(_local_definition(name="Just saved"))

        from types import SimpleNamespace

        workbench._on_automation_form_result(
            SimpleNamespace(status="saved", definition_id="local-def-1")
        )
        await pilot.pause()
        await pilot.pause()

        assert table.row_count == 1
        assert rendered_row_cells(table, 0)[0] == "[This device] Just saved"


# --- task-5 fix round: edit affordance ---------------------------------------


@pytest.mark.asyncio
async def test_edit_action_opens_form_prefilled_for_a_local_row():
    server_client = MockServerClient(notifications_service=None)
    service = AutomationsMockService(server_client, local_definitions=[_local_definition()])
    app = AutomationsTestApp(service)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        workbench.action_edit_task()
        await pilot.pause()
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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table.cursor_coordinate = (0, 0)  # def-1, "Morning brief"
        await pilot.pause()

        assert service.db._automation_definitions == []

        workbench.action_edit_task()
        await pilot.pause()
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
        assert any(n.message == "Automation updated." for n in notifications)
        assert not any(n.message == "Automation created." for n in notifications)


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
        assert (
            _detail_text(detail, "scheduling-automation-detail-view-results")
            == "See Results tab"
        )


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
    # Wide terminal (task-4 fix round): an inactive `TabPane`'s content
    # paints at a zero region -- `render_line(0)` legitimately blanks
    # there (proven empty, not a stored-attribute false pass: `.content`
    # still reads correctly, only the PAINT is absent -- see
    # `test_detail_value_row.py`'s own painted-not-stored discipline), so
    # this switches to the Automations tab before asserting, same as
    # `test_run_now_on_automations_tab_dispatches_server_side` above.
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        detail = workbench.query_one("#scheduling-automation-detail", DefinitionDetail)

        row_index = next(
            i
            for i, row in enumerate(workbench._automations)
            if row["id"] == "local-def-freq"
        )
        table.cursor_coordinate = (row_index, 0)
        await pilot.pause()
        await pilot.pause()

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
    # Wide terminal + active tab -- see the local-definition test above.
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
        tabs.active = "scheduling-automations-tab"
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        detail = workbench.query_one("#scheduling-automation-detail", DefinitionDetail)

        table.cursor_coordinate = (0, 0)
        await pilot.pause()
        await pilot.pause()

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
    (the `except` branch of `_load_automation_detail`)."""
    server_client = AutomationsServerClient()
    definition = _frequency_definition(id="local-def-freq")
    service = AutomationsMockService(server_client, local_definitions=[definition])

    def _boom(*args, **kwargs):
        raise RuntimeError("sqlite is unhappy")

    service.db.count_automation_runs = _boom
    app = AutomationsTestApp(service)
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        workbench.query_one("#scheduling-tabs", TabbedContent).active = (
            "scheduling-automations-tab"
        )
        detail = workbench.query_one("#scheduling-automation-detail", DefinitionDetail)
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        row_index = next(
            i
            for i, row in enumerate(workbench._automations)
            if row["id"] == "local-def-freq"
        )
        table.cursor_coordinate = (row_index, 0)
        await pilot.pause()
        await pilot.pause()

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
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        workbench.query_one("#scheduling-tabs", TabbedContent).active = (
            "scheduling-automations-tab"
        )
        detail = workbench.query_one("#scheduling-automation-detail", DefinitionDetail)
        table = workbench.query_one("#scheduling-automations-table", DataTable)
        row_index = next(
            i
            for i, row in enumerate(workbench._automations)
            if row["id"] == "local-def-freq"
        )
        table.cursor_coordinate = (row_index, 0)
        await pilot.pause()
        await pilot.pause()
        assert _detail_text(detail, "scheduling-automation-detail-model") == "openai/gpt-5"

        # Edit-and-save shape: the stored row changes, the id does not.
        service.db._automation_definitions[0]["input"] = {
            "question": "What changed this week?",
            "provider": "anthropic",
            "model": "claude-x",
        }
        await workbench.load_automations()
        await pilot.pause()
        await pilot.pause()

        assert (
            _detail_text(detail, "scheduling-automation-detail-model")
            == "anthropic/claude-x"
        )


@pytest.mark.asyncio
async def test_automations_panes_stay_on_screen_across_the_detail_hide_boundary():
    """Final review F5: three panes x min-width 30 could not fit the 84-89
    band, and Textual's fr layout then laid the detail and history panes
    out entirely off-screen (the split has `overflow: hidden`, so nothing
    hinted at it).

    84 is the width at which the detail pane is still shown -- the same
    `hide_detail` threshold the Queue tab uses.
    """

    class _CssApp(AutomationsTestApp):
        # The default harness does not load `_scheduling.tcss`, so
        # geometry assertions there measure NOTHING (this exact trap made
        # the reviewer's first probe read clean when it was not).
        CSS_PATH = str(BUNDLED_STYLESHEET)

    app = _CssApp(AutomationsMockService(AutomationsServerClient()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        workbench.query_one("#scheduling-tabs", TabbedContent).active = (
            "scheduling-automations-tab"
        )
        await pilot.pause()

        for width in (84, 90):
            await pilot.resize_terminal(width, 40)
            await pilot.pause()
            await pilot.pause()
            panes = [
                workbench.query_one(f"#{pane_id}")
                for pane_id in (
                    "scheduling-automations-pane",
                    "scheduling-automations-detail-pane",
                    "scheduling-automation-history-pane",
                )
            ]
            for pane in panes:
                assert not pane.has_class("pane-hidden"), (width, pane.id)
                assert pane.region.width > 0, (width, pane.id)
                assert pane.region.right <= width, (
                    f"W={width}: {pane.id} runs off-screen "
                    f"(x={pane.region.x}, width={pane.region.width})"
                )
