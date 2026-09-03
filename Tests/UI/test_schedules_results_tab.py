"""Results tab behavior on the Schedules workbench (schedules-handoff PR-6
task 3).

Two layers, matching `test_schedules_transfer_actions.py`'s split:

- Pure-function pins on the small rendering/gating helpers in
  `results_tab.py` -- fast, exact, no widget machinery.
- Workbench-level, over a REAL `SchedulingService` + tmp_path
  `ScheduledTasksDB` (never a hand-rolled fake for `review_automation_
  result`/`resolve_definition`'s own business logic -- Task 1/Task 2's
  suites already prove those; this file only proves the tab calls them
  right and renders what they say, avoiding the drifted-fake trap this
  repo has hit before). A bare-widget layer covers `ResultsTab.populate`
  rendering that doesn't need a service at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import DataTable, TabbedContent

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import rendered_row_cells
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import SchedulingServerClient
from tldw_chatbook.UI.Screens.scheduling.results_tab import (
    ResultsTab,
    _format_result_created,
    _result_kind_cell,
    _result_owner_suffix,
    _review_state_cell,
    solved_eligibility,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench

# ---------------------------------------------------------------------------
# Pure-function pins
# ---------------------------------------------------------------------------


def test_kind_cell_glyphs_and_failure_styling():
    finding = _result_kind_cell("finding")
    failure = _result_kind_cell("failure")
    assert "●" in str(finding)
    assert "✕" in str(failure)
    # Failure rows get the same red-toned style status_badge_text uses
    # for BLOCKED/CONFLICT (task_detail.py) -- there is no Rich-usable
    # `$error` CSS token, so this pins the literal-style substitute.
    assert failure.style == "bold white on red"
    assert finding.style != failure.style


def test_owner_suffix_only_for_server_scoped_rows():
    assert _result_owner_suffix({"owner_id": "local"}) == ""
    assert _result_owner_suffix({"owner_id": None}) == ""
    assert _result_owner_suffix({"owner_id": "server:example.com"}) == (
        " (server: example.com)"
    )


def test_created_relative_formats_past_and_future():
    now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    five_min_ago = (now - timedelta(minutes=5)).isoformat()
    two_days_ago = (now - timedelta(days=2)).isoformat()
    assert _format_result_created(five_min_ago, now=now) == "5m ago"
    assert _format_result_created(two_days_ago, now=now) == "2d ago"
    assert _format_result_created(None, now=now) == "-"


def test_review_state_cell_bolds_unread_with_a_dot():
    unread = _review_state_cell("unread")
    read = _review_state_cell("read")
    assert str(unread) == "● unread"
    assert unread.style == "bold"
    assert str(read) == "read"
    assert read.style != "bold"


def test_solved_eligibility_gates_kind_and_resolution_state():
    definitions = {"def-1": {"id": "def-1", "resolution_state": "open"}}
    finding = {"kind": "finding", "definition_id": "def-1"}
    failure = {"kind": "failure", "definition_id": "def-1"}
    unknown_def = {"kind": "finding", "definition_id": "missing"}

    eligible, reason = solved_eligibility(finding, definitions)
    assert eligible is True
    assert reason is None

    eligible, reason = solved_eligibility(failure, definitions)
    assert eligible is False
    assert reason == "Only findings can be marked solved."

    eligible, reason = solved_eligibility(unknown_def, definitions)
    assert eligible is False
    assert "could not be found" in reason

    definitions["def-1"]["resolution_state"] = "solved"
    eligible, reason = solved_eligibility(finding, definitions)
    assert eligible is False
    assert "already marked solved" in reason


# ---------------------------------------------------------------------------
# Bare-widget rendering (no service)
# ---------------------------------------------------------------------------


class _BareResultsApp(ConsolidatedCSSApp):
    def compose(self):
        yield ResultsTab(id="scheduling-results")


@pytest.mark.asyncio
async def test_detail_pane_renders_answer_evidence_and_review_metadata():
    """Pins the fixture-shaped row from
    Tests/Scheduling/fixtures/server_responses/automation_results_list.json."""
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        tab.populate(
            [
                {
                    "id": "res_01J5RHPQWXYZ1234567890AB",
                    "owner_id": "server:42",
                    "definition_id": "def_01J5RHPQWXYZ1234567890AB",
                    "kind": "finding",
                    "title": "Daily stand-up summary",
                    "answer": (
                        "The team is blocked on CI flakiness and an API "
                        "review."
                    ),
                    "source_refs": [{"source_type": "message", "source_id": "msg_1"}],
                    "review_state": "unread",
                    "reviewed_at": None,
                    "reviewed_by": None,
                    "created_at": "2026-08-30T09:00:05+00:00",
                },
            ]
        )
        await pilot.pause()

        table = tab.query_one("#scheduling-results-table", DataTable)
        assert table.row_count == 1
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        detail_text = str(
            tab.query_one("#scheduling-results-detail").render()
        )
        assert "Daily stand-up summary" in detail_text
        assert "The team is blocked on CI flakiness" in detail_text
        assert "message: msg_1" in detail_text
        assert "Review: unread" in detail_text


async def _detail_text_for(pilot, tab, result: dict) -> str:
    """Populate a bare ResultsTab with one row, select it, and return the
    rendered detail-pane text."""
    tab.populate([result])
    await pilot.pause()
    table = tab.query_one("#scheduling-results-table", DataTable)
    table.cursor_coordinate = (0, 0)
    await pilot.pause()
    return str(tab.query_one("#scheduling-results-detail").render())


def _base_result(**overrides) -> dict:
    row = {
        "id": "res-1",
        "owner_id": "local",
        "definition_id": "def-1",
        "kind": "finding",
        "title": "Digest",
        "answer": "fine",
        "source_refs": [],
        "review_state": "unread",
        "reviewed_at": None,
        "reviewed_by": None,
        "created_at": "2026-08-30T09:00:05+00:00",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_non_list_source_refs_degrades_instead_of_crashing():
    """Review fix round 1, finding 1 (HIGH): a syntactically-valid but
    wrong-shaped source_refs (e.g. an int from a malformed server
    payload) must render degraded, never raise."""
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        detail_text = await _detail_text_for(
            pilot, tab, _base_result(source_refs=5)
        )
        assert "unparsed" in detail_text
        assert "5" in detail_text


@pytest.mark.asyncio
async def test_non_dict_str_evidence_entries_degrade_per_item():
    """A source_refs list mixing well-shaped and malformed entries renders
    the good ones normally and each bad one degraded, in place."""
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        detail_text = await _detail_text_for(
            pilot,
            tab,
            _base_result(
                source_refs=[
                    {"source_type": "message", "source_id": "m1"},
                    42,
                    ["nested", "list"],
                ]
            ),
        )
        assert "message: m1" in detail_text
        assert detail_text.count("unparsed") == 2


@pytest.mark.asyncio
async def test_non_str_answer_degrades_instead_of_crashing():
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        detail_text = await _detail_text_for(
            pilot, tab, _base_result(answer={"nested": "dict"})
        )
        assert "unparsed" in detail_text


@pytest.mark.asyncio
async def test_bracket_tokens_in_answer_and_title_render_literally():
    """Review fix round 1, finding 2 (upgraded INFO): this pane renders
    LLM-generated content, so a `[bold]`-shaped token is realistic, not
    hypothetical -- it must render as literal text, never be parsed as
    Rich markup (which would corrupt the render or silently eat the
    token)."""
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        detail_text = await _detail_text_for(
            pilot,
            tab,
            _base_result(
                title="Digest [urgent]",
                answer="Status: [bold]blocked[/bold] on review",
                source_refs=[{"source_type": "message", "source_id": "[x]"}],
            ),
        )
        assert "Digest [urgent]" in detail_text
        assert "[bold]blocked[/bold]" in detail_text
        assert "message: [x]" in detail_text


@pytest.mark.asyncio
async def test_uppercase_bracket_tokens_survive_both_render_paths():
    """Live verification task 6: a literal `[PR-6]` in a real result
    answer VANISHED from the detail pane.

    `rich.markup.escape` only escapes tags matching `[a-z#/@]...`, but the
    parser this pane renders through (`Static.update(str)` ->
    `Content.from_markup`) consumes ANY `[...]` token. The existing
    bracket test above passed only because it used a lowercase `[bold]`,
    which rich does escape. This pins an uppercase token on both the
    normal and the degraded (`(unparsed — ...)`) branches.
    """
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()

        normal = await _detail_text_for(
            pilot,
            tab,
            _base_result(
                title="Digest [PR-6]",
                answer="the note [PR-6] describes the inbox",
                source_refs=[{"source_type": "note", "source_id": "[PR-6]"}],
            ),
        )
        assert "Digest [PR-6]" in normal
        assert "the note [PR-6] describes the inbox" in normal
        assert "note: [PR-6]" in normal

        degraded = await _detail_text_for(
            pilot,
            tab,
            _base_result(
                id="res-2",
                answer={"text": "Draft answer about [PR-6] behaviour."},
                source_refs=42,
            ),
        )
        assert "unparsed" in degraded
        assert "[PR-6]" in degraded


@pytest.mark.asyncio
async def test_table_title_cell_renders_brackets_literally():
    """Task 6 round 2, D8 -- the follow-up round 1 filed and deferred.

    The TABLE cell goes through a different parser from the detail pane:
    `DataTable` formats string cells with `rich.text.Text.from_markup`,
    which eats lowercase tags. So `[PR-6]` survived here in round 1 while
    a `[bold]` in the same title would not have, even though the detail
    pane escapes both correctly. Asserted on the painted cell, not the
    stored one.
    """
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        tab.populate(
            [
                _base_result(
                    title="Digest [bold] and [PR-6]",
                    owner_id="server:http://127.0.0.1:8020",
                )
            ]
        )
        await pilot.pause()
        table = tab.query_one("#scheduling-results-table", DataTable)
        assert rendered_row_cells(table, 0)[1] == (
            "Digest [bold] and [PR-6] (server: http://127.0.0.1:8020)"
        )


@pytest.mark.asyncio
async def test_empty_state_shown_when_no_results():
    app = _BareResultsApp()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()
        tab.populate([])
        await pilot.pause()
        table = tab.query_one("#scheduling-results-table", DataTable)
        assert not table.display
        empty = tab.query_one("#scheduling-results-empty")
        assert empty.display != "none"


# ---------------------------------------------------------------------------
# Workbench-level: real SchedulingService + tmp_path DB
# ---------------------------------------------------------------------------


@pytest.fixture
def results_db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


class ResultsWorkbenchTestApp(ConsolidatedCSSApp):
    """A real Textual test app wired to a REAL `SchedulingService` over a
    tmp_path DB (matches `test_schedules_transfer_actions.py`'s
    `TransferWorkbenchTestApp`) -- `review_automation_result`/`resolve_
    definition` run for real; this app only proves the workbench calls
    them right and renders what they say.
    """

    def __init__(self, db, *args, server_client=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(
                active_server_id="1" if server_client is not None else None
            )
        )
        self.scheduling_service = SchedulingService(
            db=db,
            server_client=server_client,
            runtime_source="local",
            app_getter=lambda: self,
        )


def _seed_definition(
    db: ScheduledTasksDB,
    *,
    owner_id: str = "local",
    server_id: str | None = None,
    resolution_state: str = "open",
    name: str = "Daily digest",
) -> str:
    kwargs: dict = {}
    if server_id is not None:
        kwargs["server_id"] = server_id
    if resolution_state != "open":
        kwargs["resolution_state"] = resolution_state
    return db.create_automation_definition(
        owner_id=owner_id, family="recurring_question", name=name, **kwargs
    )


def _seed_result(
    db: ScheduledTasksDB,
    *,
    definition_id: str,
    owner_id: str = "local",
    server_id: str | None = None,
    kind: str = "finding",
    review_state: str = "unread",
    dedupe_key: str,
) -> str:
    kwargs: dict = {
        "answer": "The team is blocked on CI flakiness.",
        "source_refs": [{"source_type": "message", "source_id": "msg-1"}],
        "review_state": review_state,
    }
    if server_id is not None:
        kwargs["server_id"] = server_id
    result_id = db.create_automation_result(
        owner_id=owner_id,
        definition_id=definition_id,
        run_id="run-1",
        kind=kind,
        title="Daily stand-up summary",
        summary="Two blockers reported.",
        dedupe_key=dedupe_key,
        **kwargs,
    )
    assert result_id is not None
    return result_id


def _rendered_tab_title(workbench, pane_id: str) -> str:
    """The text the tab bar actually shows for one pane.

    Goes through `TabbedContent.get_tab()` -> the `Tab` widget's own
    rendered visual, so it can only pass if the label really reached the
    widget that paints the tab bar.
    """
    tab = workbench.query_one("#scheduling-tabs", TabbedContent).get_tab(pane_id)
    return str(tab.render())


async def _open_results_tab(pilot, *, row: int = 0):
    workbench = pilot.app.screen
    tabs = workbench.query_one("#scheduling-tabs", TabbedContent)
    tabs.active = "scheduling-results-tab"
    await pilot.pause()
    table = workbench.query_one("#scheduling-results-table", DataTable)
    if table.row_count:
        table.cursor_coordinate = (row, 0)
        await pilot.pause()
    return workbench


@pytest.mark.asyncio
async def test_badge_and_table_span_every_owner(results_db):
    db = results_db
    local_def = _seed_definition(db, owner_id="local")
    server_def = _seed_definition(db, owner_id="server:1", server_id="srv-def-1")
    _seed_result(
        db, definition_id=local_def, owner_id="local", dedupe_key="d1"
    )  # unread
    _seed_result(
        db,
        definition_id=server_def,
        owner_id="server:1",
        server_id="srv-res-1",
        dedupe_key="d2",
    )  # unread
    _seed_result(
        db,
        definition_id=local_def,
        owner_id="local",
        kind="failure",
        review_state="read",
        dedupe_key="d3",
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        # Assert the RENDERED tab title through the real widget tree, not
        # an attribute the code just set. The previous assertion read back
        # `TabPane.label` -- an attribute Textual 8.x's TabPane does not
        # have (it stores `_title`), so the assignment created an inert
        # Python attribute, this test confirmed itself, and the badge was
        # invisible on screen (live verification task 6, D2).
        assert _rendered_tab_title(pilot.app.screen, "scheduling-results-tab") == (
            "Results (2)"
        )

        table = pilot.app.screen.query_one("#scheduling-results-table", DataTable)
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_conflicts_badge_renders_too(results_db):
    """The Conflicts badge is the one PR-6's Results badge was copied from
    and was equally inert (live verification task 6, D2). Both go through
    the same `_set_tab_label` seam now, so both are pinned on the render.
    """
    db = results_db
    db.record_conflict(
        local_id="l1",
        primitive="reminder_task",
        owner_id="local",
        server_state={"title": "Server"},
        local_state={"title": "Local"},
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()

        assert _rendered_tab_title(
            pilot.app.screen, "scheduling-conflicts-tab"
        ) == "Conflicts (1)"
        # No results seeded -> the Results badge stays a bare label.
        assert _rendered_tab_title(
            pilot.app.screen, "scheduling-results-tab"
        ) == "Results"


@pytest.mark.asyncio
async def test_mark_solved_resolves_a_server_keyed_result(results_db):
    """Live verification task 6, D3: a synced result's `definition_id` is
    the SERVER's id, but the tab indexed definitions by their LOCAL id and
    `resolve_definition` takes a LOCAL id -- so `o` refused ("definition
    could not be found") on exactly the rows the feature exists for.
    """
    db = results_db
    local_definition_id = _seed_definition(
        db, owner_id="local", server_id="srv-def-1"
    )
    _seed_result(
        db,
        # The server's id for the definition, as a mirrored row carries it.
        definition_id="srv-def-1",
        owner_id="local",
        dedupe_key="d1",
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        # The eligibility gate resolves through the server id space...
        results_tab = workbench.query_one("#scheduling-results", ResultsTab)
        detail_text = str(
            results_tab.query_one("#scheduling-results-detail").render()
        )
        assert "Solve: eligible" in detail_text

        # ...and so does the action, which must hand `resolve_definition`
        # the LOCAL id it contracts for.
        workbench.action_mark_result_solved()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_automation_definition(local_definition_id)
        assert row["resolution_state"] == "solved"


@pytest.mark.asyncio
async def test_read_action_marks_local_result_read(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        # r reuses action_run_task_now, routed to "read" on this tab.
        workbench.action_run_task_now()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_automation_result(result_id)
        assert row["review_state"] == "read"
        # Badge clears once nothing is unread -- asserted on the render,
        # same reason as the badge test above (task 6, D2).
        assert _rendered_tab_title(workbench, "scheduling-results-tab") == "Results"


@pytest.mark.asyncio
async def test_dismiss_action_queues_server_pushback_mutation(results_db):
    db = results_db
    definition_id = _seed_definition(
        db, owner_id="server:1", server_id="srv-def-1"
    )
    result_id = _seed_result(
        db,
        definition_id=definition_id,
        owner_id="server:1",
        server_id="srv-res-1",
        dedupe_key="d1",
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        # d reuses action_delete, routed to "dismissed" on this tab.
        workbench.action_delete()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_automation_result(result_id)
        assert row["review_state"] == "dismissed"

        mutations = db.get_pending_mutations(
            owner_id="server:1", primitive="automation_result_review"
        )
        assert len(mutations) == 1
        assert mutations[0]["payload"]["review_state"] == "dismissed"
        assert mutations[0]["payload"]["server_result_id"] == "srv-res-1"


@pytest.mark.asyncio
async def test_mark_all_read_fans_out_per_row(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    r1 = _seed_result(db, definition_id=definition_id, dedupe_key="d1")
    r2 = _seed_result(db, definition_id=definition_id, dedupe_key="d2")
    r3 = _seed_result(
        db, definition_id=definition_id, review_state="read", dedupe_key="d3"
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        workbench.action_mark_all_results_read()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_automation_result(r1)["review_state"] == "read"
        assert db.get_automation_result(r2)["review_state"] == "read"
        assert db.get_automation_result(r3)["review_state"] == "read"
        notifications = list(pilot.app._notifications)
        assert any("Marked 2 result" in n.message for n in notifications)


@pytest.mark.asyncio
async def test_mark_solved_local_round_trip(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        workbench.action_mark_result_solved()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "solved"
        assert definition["resolved_result_id"] == result_id
        assert definition["resolved_by"] == "local"


@pytest.mark.asyncio
async def test_mark_solved_refuses_failure_row_without_reaching_the_service(
    results_db,
):
    db = results_db
    definition_id = _seed_definition(db)
    _seed_result(
        db, definition_id=definition_id, kind="failure", dedupe_key="d1"
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        workbench.action_mark_result_solved()
        await pilot.pause()

        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "open"
        notifications = list(pilot.app._notifications)
        assert any(
            "Only findings can be marked solved." in n.message
            for n in notifications
        )


@pytest.mark.asyncio
async def test_mark_solved_server_round_trip(results_db):
    db = results_db
    definition_id = _seed_definition(
        db, owner_id="server:1", server_id="srv-def-1"
    )
    result_id = _seed_result(
        db,
        definition_id=definition_id,
        owner_id="server:1",
        server_id="srv-res-1",
        dedupe_key="d1",
    )

    notifications_service = SimpleNamespace(
        mark_scheduled_automation_definition_solved=AsyncMock(
            return_value={"id": "srv-def-1", "resolution_state": "solved"}
        )
    )
    server_client = SchedulingServerClient(notifications_service=notifications_service)

    app = ResultsWorkbenchTestApp(db, server_client=server_client)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        workbench.action_mark_result_solved()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        notifications_service.mark_scheduled_automation_definition_solved.assert_awaited_once_with(
            "srv-def-1", result_id="srv-res-1"
        )
        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "solved"
        assert result_id  # sanity: the local->server id translation ran


@pytest.mark.asyncio
async def test_mark_solved_refused_offline_surfaces_the_reason(results_db):
    db = results_db
    definition_id = _seed_definition(
        db, owner_id="server:1", server_id="srv-def-1"
    )
    _seed_result(
        db,
        definition_id=definition_id,
        owner_id="server:1",
        server_id="srv-res-1",
        dedupe_key="d1",
    )

    # No server_client wired -- SchedulingService defaults to a
    # disconnected SchedulingServerClient(), so the server branch hits a
    # real ServerUnavailableError (never a hand-rolled refusal fake).
    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = await _open_results_tab(pilot)

        workbench.action_mark_result_solved()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "open"
        notifications = list(pilot.app._notifications)
        assert any(
            "requires a server connection" in n.message for n in notifications
        )


@pytest.mark.asyncio
async def test_results_actions_refuse_off_the_results_tab(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen  # Queue tab is active by default

        workbench.action_mark_result_solved()
        workbench.action_mark_all_results_read()
        await pilot.pause()

        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "open"
        notifications = list(pilot.app._notifications)
        assert len(notifications) == 2
        assert all("Results tab" in n.message for n in notifications)
