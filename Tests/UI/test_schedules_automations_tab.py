"""Automations tab behavior on the Schedules workbench (task-18940 slice 2).

The tab surfaces both the server's automation definitions (ADR-077, ``r``
dispatches through the server control plane) and, since task-5's fix
round, this device's own local-owned `recurring_question` definitions
(``r`` routes those through `SchedulingService.run_automation_now`
instead -- never the server client, and never both).
"""

from unittest.mock import AsyncMock

import pytest
from textual.widgets import DataTable, Input, Select, TabbedContent

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import (
    MockSchedulingDB,
    MockSchedulingServiceMixin,
    MockServerClient,
)
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientValidationError,
)
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

    def __init__(self, server_client, local_definitions=None) -> None:
        self.owner_id = "local"
        self.server_client = server_client
        self.db = MockSchedulingDB(automation_definitions=local_definitions or [])
        self.sync_engine = None
        # task-5 fix round: the PR-2 local run-now seam
        # (SchedulingService.run_automation_now) -- overridable per test.
        self.run_automation_now = AsyncMock(
            return_value={"run_id": "run-local-1", "deduped": False}
        )

    async def list_tasks(self):
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

        server_client.list_automation_definitions.assert_awaited_once()
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
        first_cell = str(table.get_cell_at((0, 0)))
        assert first_cell == "[server-1] Morning brief"
        assert "This device" not in first_cell


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
        assert table.get_cell_at((0, 4)) == "anthropic/claude-x"
        assert table.get_cell_at((1, 4)) == "auto"


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
        assert table.get_cell_at((0, 0)) == "[This device] Local digest"
        # No `library_rag_search_service` on the bare test app -> capability_unavailable,
        # NOT the DB row's stored "execution_unavailable" placeholder.
        assert table.get_cell_at((0, 3)) == "capability_unavailable"
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
        names = [table.get_cell_at((i, 0)) for i in range(3)]
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

        names = [table.get_cell_at((i, 0)) for i in range(table.row_count)]
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
        assert table.get_cell_at((0, 0)) == "[This device] Just saved"


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
