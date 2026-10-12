"""Actual Schedules and Watchlists controls display durable names literally."""

import pytest
from rich.text import Text
from textual.content import Content
from textual.widgets import Button, DataTable, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind, TaskStatus
from tldw_chatbook.UI.Screens.scheduling.task_detail import TaskDetail, TaskInspector
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

NAMES = ("Ordinary name", "[/unmatched]", "[@click=app.quit]x[/]")


class NameHarness(ConsolidatedCSSApp):
    CSS_PATH = list(APP_STYLESHEETS)

    def __init__(self, surface, name):
        super().__init__()
        self.surface, self.runtime_name = surface, name

    def compose(self):
        if self.surface == "reminder":
            yield TaskDetail()
        elif self.surface == "conflict":
            yield TaskInspector()
        elif self.surface == "rule":
            pane = RulesPane()
            pane.set_reactive(
                RulesPane.rules,
                [
                    {
                        "id": "1",
                        "name": self.runtime_name,
                        "condition_type": "no_items",
                        "severity": "warning",
                        "enabled": True,
                    }
                ],
            )
            yield pane
        else:
            yield WatchlistTree(
                watchlists=[],
                counts={},
                source_rows_loader=lambda _: [],
                all_source_rows=[{"id": 1, "name": self.runtime_name}],
                expanded_root_kinds=("all",),
            )


def plain(value):
    if isinstance(value, str):
        value = Text.from_markup(value)
    assert isinstance(value, (Text, Content))
    assert not any(span.style for span in value.spans)
    return value.plain


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("surface", ["reminder", "conflict", "rule", "source"])
async def test_runtime_name_rows_are_literal(surface, name):
    app = NameHarness(surface, name)
    async with app.run_test(size=(120, 50)) as pilot:
        if surface in ("reminder", "conflict"):
            task = ReminderTask(
                id="literal",
                title=name,
                schedule_kind=ScheduleKind.RECURRING,
                cron="0 9 * * *",
                timezone="UTC",
                last_status=TaskStatus.CONFLICT,
            )
            if surface == "reminder":
                app.query_one(TaskDetail).set_task(task)
                await pilot.pause()
                assert (
                    plain(
                        app.query_one(
                            "#scheduling-task-detail-title", Static
                        ).renderable
                    )
                    == name
                )
                assert (
                    plain(app.query_one("#schedules-follow-in-console", Button).label)
                    == f"Follow '{name}' in Console"
                )
            else:
                app.query_one(TaskInspector).set_task(task)
                await pilot.pause()
                assert (
                    plain(app.query_one("#scheduling-conflict-text", Static).renderable)
                    == "Conflict detected\n" + name
                )
        elif surface == "rule":
            await pilot.pause()
            assert plain(app.query_one(DataTable).get_row_at(0)[0]) == name
        else:
            await pilot.pause()
            assert (
                plain(app.query_one("#wl-tree-node-source-all-1", Button).label).strip()
                == name
            )
