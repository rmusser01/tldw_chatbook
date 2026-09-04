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
from textual.widgets import Button, DataTable

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.schedules_test_helpers import rendered_row_cells
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import SchedulingServerClient
from tldw_chatbook.UI.Screens.scheduling.results_tab import (
    RESULTS_HEADING,
    ResultsHostScreen,
    ResultsTab,
    _format_result_created,
    _result_kind_cell,
    _result_owner_suffix,
    _review_state_cell,
    index_definitions_by_id,
    solved_eligibility,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    RESULTS_INBOX_LIMIT,
    SchedulesWorkbench,
)

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


def _rendered_badge_label(workbench, button_id: str) -> str:
    """The text a status/rail badge Button actually shows.

    redesign PR-4 task 5 replaces `_rendered_tab_title`, which read the
    same counts off the retired `TabbedContent`'s own `Tab` widgets. The
    reason that helper went through the rendered widget rather than an
    attribute survives verbatim and is why this one does too: the
    previous assertion read back `TabPane.label`, an attribute Textual
    8.x's `TabPane` does not have, so the badge was invisible on screen
    while the test confirmed itself (live verification task 6, D2). A
    `Button`'s `label` IS its rendered content, so reading it is reading
    the paint.
    """
    return str(workbench.query_one(button_id, Button).label)


async def _open_results_overlay(pilot, *, row: int = 0):
    """Open the results view the way a user does -- the rail's
    `Results (N)` button -- and select `row`.

    redesign PR-4 task 5 replaces `_open_results_tab`, which flipped
    `TabbedContent.active` to the retired Results tab. Every scenario
    that used it is re-pointed here rather than deleted: the listing, the
    detail rendering and the read/dismiss/mark-solved verbs all still
    exist, on the pushed surface (task 2) instead of the tab. Returns
    `(workbench, screen)` because several of those scenarios assert on
    BOTH -- the pushed view's effect, and the rail badge behind it.
    """
    workbench = pilot.app.screen
    workbench.query_one("#scheduling-results-badge", Button).press()
    await pilot.pause()
    screen = pilot.app.screen
    assert isinstance(screen, ResultsHostScreen), screen
    table = screen.query_one("#scheduling-results-table", DataTable)
    if table.row_count:
        table.cursor_coordinate = (row, 0)
        await pilot.pause()
    return workbench, screen


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

        # redesign PR-4 task 5: the unread count is read off the rail's
        # `Results (N)` button, the badge that survived the retirement --
        # `_refresh_results_badge` mirrors the same single
        # `count_unread_results` the tab label used to.
        assert _rendered_badge_label(
            pilot.app.screen, "#scheduling-results-badge"
        ) == "Results (2)"

        # ...and the all-owners listing is the pushed view's, opened from
        # that same button.
        _workbench, screen = await _open_results_overlay(pilot)
        table = screen.query_one("#scheduling-results-table", DataTable)
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_conflicts_badge_renders_too(results_db):
    """The Conflicts badge is the one PR-6's Results badge was copied from
    and was equally inert (live verification task 6, D2), so both are
    pinned on the render.

    redesign PR-4 task 5: both counts used to be mirrored onto a tab
    label AND a badge Button; `_set_tab_label` is deleted with the
    `TabbedContent` and the Buttons are the only home left. Same two
    counts, same two `service.db` reads.
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

        assert _rendered_badge_label(
            pilot.app.screen, "#scheduling-conflicts-badge"
        ) == "Conflicts (1)"
        # No results seeded -> the Results badge stays a bare label.
        assert _rendered_badge_label(
            pilot.app.screen, "#scheduling-results-badge"
        ) == "Results"


@pytest.mark.asyncio
async def test_mark_solved_resolves_a_server_keyed_result(results_db):
    """Live verification task 6, D3: a synced result's `definition_id` is
    the SERVER's id, but the view indexed definitions by their LOCAL id
    and `resolve_definition` takes a LOCAL id -- so `o` refused
    ("definition could not be found") on exactly the rows the feature
    exists for.

    redesign PR-4 task 5: driven through the pushed view's own `o`
    (`ResultsHostScreen.action_mark_solved`) now that the tab and its
    tab-gated `action_mark_result_solved` are retired. Both always shared
    the same `solved_eligibility` gate and `mark_selected_result_solved`
    orchestration (task 2 factored them out for exactly this), so the D3
    regression this pins is unmoved -- only its caller is.
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
        _workbench, screen = await _open_results_overlay(pilot)

        # The eligibility gate resolves through the server id space...
        results_tab = screen.query_one(ResultsTab)
        detail_text = str(
            results_tab.query_one("#scheduling-results-detail").render()
        )
        assert "Solve: eligible" in detail_text

        # ...and so does the action, which must hand `resolve_definition`
        # the LOCAL id it contracts for.
        screen.action_mark_solved()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_automation_definition(local_definition_id)
        assert row["resolution_state"] == "solved"


@pytest.mark.asyncio
async def test_read_action_marks_local_result_read(results_db):
    """redesign PR-4 task 5: `r` used to be `action_run_task_now` routed
    by active tab; the routing and the tab are retired, so this drives
    the pushed view's own `r` (`ResultsHostScreen.action_review_read`),
    which has always called the same `review_selected_result`. The badge
    half of the claim -- the count clears once nothing is unread -- is
    re-asserted on the rail button, and now also proves the pop-time
    `dismissed` refresh actually runs."""
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench, screen = await _open_results_overlay(pilot)
        assert _rendered_badge_label(
            workbench, "#scheduling-results-badge"
        ) == "Results (1)"

        screen.action_review_read()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = db.get_automation_result(result_id)
        assert row["review_state"] == "read"

        # Closing the view runs its `dismissed` hook, which re-counts the
        # rail badge: it clears once nothing is unread.
        await pilot.press("escape")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert pilot.app.screen is workbench
        assert _rendered_badge_label(
            workbench, "#scheduling-results-badge"
        ) == "Results"


@pytest.mark.asyncio
async def test_dismiss_action_queues_server_pushback_mutation(results_db):
    """redesign PR-4 task 5: `d` used to be `action_delete` routed by
    active tab; driven through the pushed view's own `d` now (same
    `review_selected_result` orchestration). The claim -- a server-owned
    result's dismissal queues its pushback mutation with the SERVER's
    result id -- is untouched and has no other home."""
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
        _workbench, screen = await _open_results_overlay(pilot)

        screen.action_review_dismiss()
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
    """redesign PR-4 task 5: no tab to activate first --
    `action_mark_all_results_read` is ungated now (it was Results-tab-only,
    with a byte-identical ungated twin behind the rail button; the two
    collapsed into one when the gate went). The fan-out claim is
    unchanged."""
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
        workbench = pilot.app.screen

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
async def test_mark_all_read_reaches_unread_rows_beyond_the_inbox_window(
    results_db,
):
    """HIGH (Qodo): `_unread_result_ids` used to read the results view's
    own CAPPED listing (`RESULTS_INBOX_LIMIT`, 200 newest rows) instead
    of querying the DB directly, so an older unread result outside that
    window survived a mark-all-read -- while the rail button's
    visibility is gated on the FULL-table `count_unread_results` (final
    review F2), so it could hide with unread work still sitting there.
    Seeds one old server-mirrored unread result first (oldest, so it
    falls outside the newest-`RESULTS_INBOX_LIMIT` window once
    `RESULTS_INBOX_LIMIT` more local ones are seeded after it), then
    proves mark-all-read reaches it anyway AND still queues its server
    pushback mutation.
    """
    db = results_db
    server_definition_id = _seed_definition(
        db, owner_id="server:1", server_id="srv-def-1"
    )
    old_result_id = _seed_result(
        db,
        definition_id=server_definition_id,
        owner_id="server:1",
        server_id="srv-res-old",
        dedupe_key="old",
    )
    local_definition_id = _seed_definition(db)
    for index in range(RESULTS_INBOX_LIMIT):
        _seed_result(db, definition_id=local_definition_id, dedupe_key=f"d{index}")
    total_unread = RESULTS_INBOX_LIMIT + 1
    assert db.count_unread_results(owner_id=None) == total_unread

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench, screen = await _open_results_overlay(pilot)

        # Sanity: the loaded table window is capped, so the old result is
        # not among the currently-rendered rows -- this is the exact
        # shape that used to leave it unreached. (redesign PR-4 task 5:
        # that capped listing is the PUSHED view's now; the fan-out under
        # test still reads the DB, which is the whole point.)
        table = screen.query_one("#scheduling-results-table", DataTable)
        assert table.row_count == RESULTS_INBOX_LIMIT
        await pilot.press("escape")
        await pilot.pause()

        workbench.action_mark_all_results_read()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.count_unread_results(owner_id=None) == 0
        assert db.get_automation_result(old_result_id)["review_state"] == "read"
        notifications = list(pilot.app._notifications)
        assert any(
            f"Marked {total_unread} result" in n.message for n in notifications
        )

        mutations = db.get_pending_mutations(
            owner_id="server:1", primitive="automation_result_review"
        )
        assert any(
            mutation["payload"].get("server_result_id") == "srv-res-old"
            and mutation["payload"].get("review_state") == "read"
            for mutation in mutations
        )


@pytest.mark.asyncio
async def test_rail_mark_all_read_button_fans_out_and_hides_itself(results_db):
    """redesign PR-2, Task 3: the rail's `Mark all read` button reuses the
    per-row fan-out (`_dispatch_mark_all_results_read`), then refreshes
    the Queue's own unread dots so the button hides itself again once
    nothing is unread.

    redesign PR-4 task 5 (renamed): "without switching tabs" was the
    point of the button when `a` was Results-tab-only -- there are no
    tabs and no gate now, so the button and `a` are one path and the
    name says what is still true."""
    db = results_db
    definition_id = _seed_definition(db)
    r1 = _seed_result(db, definition_id=definition_id, dedupe_key="d1")
    r2 = _seed_result(db, definition_id=definition_id, dedupe_key="d2")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        workbench = pilot.app.screen

        button = workbench.query_one("#scheduling-mark-all-read", Button)
        assert button.display is True

        button.press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_automation_result(r1)["review_state"] == "read"
        assert db.get_automation_result(r2)["review_state"] == "read"
        notifications = list(pilot.app._notifications)
        assert any("Marked 2 result" in n.message for n in notifications)

        assert button.display is False


@pytest.mark.asyncio
async def test_mark_solved_local_round_trip(results_db):
    """redesign PR-4 task 5: driven through the pushed view's `o`
    (`ResultsHostScreen.action_mark_solved`) -- same
    `mark_selected_result_solved` orchestration the retired tab-gated
    `action_mark_result_solved` called."""
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        _workbench, screen = await _open_results_overlay(pilot)

        screen.action_mark_solved()
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
    """redesign PR-4 task 5: the SYNCHRONOUS eligibility refusal (no
    worker round-trip) that `solved_eligibility` owns, re-pinned on the
    pushed view's `o` -- `ResultsHostScreen.action_mark_solved` keeps the
    same caller-side gate the retired tab action had (results_tab.py's
    own docstring: the gates stay in each CALLER)."""
    db = results_db
    definition_id = _seed_definition(db)
    _seed_result(
        db, definition_id=definition_id, kind="failure", dedupe_key="d1"
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        _workbench, screen = await _open_results_overlay(pilot)

        screen.action_mark_solved()
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
    """redesign PR-4 task 5: driven through the pushed view's `o`; the
    local->server id translation under test is unchanged."""
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
        _workbench, screen = await _open_results_overlay(pilot)

        screen.action_mark_solved()
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
    """redesign PR-4 task 5: driven through the pushed view's `o`; the
    real `ServerUnavailableError` refusal under test is unchanged."""
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
        _workbench, screen = await _open_results_overlay(pilot)

        screen.action_mark_solved()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "open"
        notifications = list(pilot.app._notifications)
        assert any(
            "requires a server connection" in n.message for n in notifications
        )


# redesign PR-4 task 5: `test_results_actions_refuse_off_the_results_tab`
# is DELETED, not relocated. It pinned `_is_results_tab_active`'s two
# refusals ("Switch to the Results tab to ...") -- the behaviour of a
# gate whose only purpose was to stop tab-scoped keys acting on a tab the
# user was not looking at. With one surface there is no tab to be off of:
# `o` lives only on the pushed view (a screen underneath never receives
# the key at all -- structural, not gated), and `a` is deliberately
# ungated because its target, the rail's `Mark all read` button, is
# always on screen. There is no behaviour left to assert; keeping the
# test would mean re-introducing the gate to satisfy it. The half that
# survives -- `a` doing the right thing from the rail -- is
# `test_rail_mark_all_read_button_fans_out_and_hides_itself` above.


@pytest.mark.asyncio
async def test_inbox_lists_the_sync_window_and_says_what_it_hides(results_db):
    """HIGH (Qodo): the refresh took the DB's default `limit=50` while the
    badge counted EVERY unread result, so the badge could read "Results
    (201)" over 50 rows with nothing saying the rest existed.

    The listing is the sync-mirrored window -- `RESULTS_INBOX_LIMIT`,
    i.e. exactly the newest-pages walk `SyncEngine._pull_results` does --
    and once it bites, the heading says so. Deliberately no pagination:
    saying what is hidden is the fix.

    redesign PR-4 task 5: the listing and its cap line moved to the
    pushed view with the retirement (`_push_results_overlay` runs the
    same `RESULTS_INBOX_LIMIT` query the tab refresh used to). Both
    halves of the honesty claim are still asserted together, which is the
    point of the test: the capped listing AND the uncapped badge count.
    """
    db = results_db
    definition_id = _seed_definition(db)
    total = RESULTS_INBOX_LIMIT + 1
    for index in range(total):
        _seed_result(db, definition_id=definition_id, dedupe_key=f"d{index}")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench, screen = await _open_results_overlay(pilot)

        table = screen.query_one("#scheduling-results-table", DataTable)
        assert table.row_count == RESULTS_INBOX_LIMIT

        heading = screen.query_one("#scheduling-results-heading")
        assert (
            f"showing newest {RESULTS_INBOX_LIMIT} of {total}"
            in str(heading.render())
        )
        # The badge still counts every unread result -- that honesty is
        # the whole reason the truncation has to be stated.
        assert f"Results ({total})" in _rendered_badge_label(
            workbench, "#scheduling-results-badge"
        )


@pytest.mark.asyncio
async def test_inbox_heading_stays_plain_when_everything_fits(results_db):
    """The count line is truncation-only -- an inbox that fits must not
    grow a permanent "showing newest 2 of 2" tail. (redesign PR-4 task 5:
    read off the pushed view, which owns the listing now.)"""
    db = results_db
    definition_id = _seed_definition(db)
    for index in range(2):
        _seed_result(db, definition_id=definition_id, dedupe_key=f"d{index}")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        _workbench, screen = await _open_results_overlay(pilot)

        heading = screen.query_one("#scheduling-results-heading")
        assert str(heading.render()).strip() == RESULTS_HEADING


# ---------------------------------------------------------------------------
# redesign PR-4, task 2: Results relocation onto the pushed surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_results_self_populate_with_custom_heading():
    """`ConflictsTab.initial_conflicts`'s own self-populate-on-mount idiom
    (task 1), plus the optional `heading` override a definition-filtered
    pushed view uses -- the SAME "showing newest N of TOTAL" cap-line math
    renders under whatever heading text it is given."""

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield ResultsTab(
                id="scheduling-results-overlay",
                initial_results=[_base_result()],
                initial_total=5,
                heading="Automation results — Weekly digest",
            )

    app = _App()
    async with app.run_test() as pilot:
        tab = pilot.app.query_one(ResultsTab)
        await pilot.pause()

        table = tab.query_one("#scheduling-results-table", DataTable)
        assert table.row_count == 1
        heading = str(tab.query_one("#scheduling-results-heading").render()).strip()
        assert heading == "Automation results — Weekly digest — showing newest 1 of 5"


def _push_results_host_closures(service, *, definition_id: str | None = None):
    """`(query, unread_ids)` closures matching `SchedulesWorkbench._push_
    results_overlay`'s own shape, scoped to a single `definition_id`
    string when given. These tests never mix local/server id spaces for
    one definition -- that merge is `_definition_results_query`'s own
    concern, exercised directly below."""

    def query():
        results = service.db.list_automation_results(
            owner_id=None, definition_id=definition_id, limit=RESULTS_INBOX_LIMIT
        )
        total = service.db.count_automation_results(
            owner_id=None, definition_id=definition_id
        )
        definitions_by_id = index_definitions_by_id(
            service.db.list_automation_definitions(owner_id=None)
        )
        return results, definitions_by_id, total

    def unread_ids():
        total_unread = service.db.count_unread_results(
            owner_id=None, definition_id=definition_id
        )
        if not total_unread:
            return []
        rows = service.db.list_automation_results(
            owner_id=None,
            definition_id=definition_id,
            review_state="unread",
            limit=total_unread,
        )
        return [row["id"] for row in rows]

    return query, unread_ids


async def _push_results_host_screen(pilot, service, *, definition_id: str | None = None):
    """Push a `ResultsHostScreen` directly (no `SchedulesWorkbench`
    involved) -- exercises the pushed view's own r/d/o/a binding surface
    and the shared service orchestration in isolation."""
    query, unread_ids = _push_results_host_closures(service, definition_id=definition_id)
    results, definitions_by_id, total = query()

    def _factory() -> ResultsTab:
        return ResultsTab(
            id="scheduling-results-overlay",
            initial_results=results,
            initial_definitions_by_id=definitions_by_id,
            initial_total=total,
        )

    await pilot.app.push_screen(
        ResultsHostScreen(
            _factory, title="Results", service=service, query=query, unread_ids=unread_ids
        )
    )
    await pilot.pause()
    return pilot.app.screen


@pytest.mark.asyncio
async def test_hosted_read_action_updates_the_selected_result(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        service = pilot.app.scheduling_service
        screen = await _push_results_host_screen(pilot, service)
        table = screen.query_one("#scheduling-results-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        await pilot.press("r")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_automation_result(result_id)["review_state"] == "read"


@pytest.mark.asyncio
async def test_hosted_dismiss_action_updates_the_selected_result(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        service = pilot.app.scheduling_service
        screen = await _push_results_host_screen(pilot, service)
        table = screen.query_one("#scheduling-results-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        await pilot.press("d")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_automation_result(result_id)["review_state"] == "dismissed"


@pytest.mark.asyncio
async def test_hosted_mark_solved_action_resolves_the_definition(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    result_id = _seed_result(db, definition_id=definition_id, dedupe_key="d1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        service = pilot.app.scheduling_service
        screen = await _push_results_host_screen(pilot, service)
        table = screen.query_one("#scheduling-results-table", DataTable)
        table.cursor_coordinate = (0, 0)
        await pilot.pause()

        await pilot.press("o")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        definition = db.get_automation_definition(definition_id)
        assert definition["resolution_state"] == "solved"
        assert definition["resolved_result_id"] == result_id


@pytest.mark.asyncio
async def test_hosted_mark_all_read_action_fans_out(results_db):
    db = results_db
    definition_id = _seed_definition(db)
    r1 = _seed_result(db, definition_id=definition_id, dedupe_key="d1")
    r2 = _seed_result(db, definition_id=definition_id, dedupe_key="d2")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        service = pilot.app.scheduling_service
        await _push_results_host_screen(pilot, service)

        await pilot.press("a")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert db.get_automation_result(r1)["review_state"] == "read"
        assert db.get_automation_result(r2)["review_state"] == "read"


@pytest.mark.asyncio
async def test_definition_results_query_merges_local_and_server_id_spaces(results_db):
    """redesign PR-4 task 2: a definition mirrored from the server can
    carry results in BOTH id spaces -- a locally-created one
    (`definition_id` = the local row id) and a server-mirrored one
    (`definition_id` = the server's id, `index_definitions_by_id`'s own
    documented split). `_definition_results_query` must see both, and
    count both toward its cap-line total -- neither single-id-space DB
    query alone sees more than half. A result belonging to a different
    definition must not leak in."""
    db = results_db
    definition_id = _seed_definition(db, owner_id="server:1", server_id="srv-def-1")
    local_result_id = _seed_result(db, definition_id=definition_id, dedupe_key="local-1")
    server_result_id = _seed_result(
        db, definition_id="srv-def-1", server_id="srv-res-1", dedupe_key="server-1"
    )
    other_definition_id = _seed_definition(db, name="Other")
    _seed_result(db, definition_id=other_definition_id, dedupe_key="other-1")

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        service = pilot.app.scheduling_service

        definition = db.get_automation_definition(definition_id)
        results, total = workbench._definition_results_query(service, definition)

        assert total == 2
        assert {row["id"] for row in results} == {local_result_id, server_result_id}


@pytest.mark.asyncio
async def test_definition_unread_result_ids_merges_both_id_spaces(results_db):
    """`_definition_unread_result_ids` counterpart of the query merge test
    above -- the `a` action inside a definition-filtered pushed view must
    reach unread results in BOTH id spaces, and none belonging to another
    definition."""
    db = results_db
    definition_id = _seed_definition(db, owner_id="server:1", server_id="srv-def-1")
    local_result_id = _seed_result(db, definition_id=definition_id, dedupe_key="local-1")
    server_result_id = _seed_result(
        db, definition_id="srv-def-1", server_id="srv-res-1", dedupe_key="server-1"
    )
    other_definition_id = _seed_definition(db, name="Other")
    other_result_id = _seed_result(
        db, definition_id=other_definition_id, dedupe_key="other-1"
    )

    app = ResultsWorkbenchTestApp(db)
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        workbench = pilot.app.screen
        service = pilot.app.scheduling_service

        definition = db.get_automation_definition(definition_id)
        ids = workbench._definition_unread_result_ids(service, definition)

        assert set(ids) == {local_result_id, server_result_id}
        assert other_result_id not in ids
