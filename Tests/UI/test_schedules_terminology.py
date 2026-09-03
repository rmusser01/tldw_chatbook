"""task-23106: one noun for user-created items -- "scheduled task".

User-facing copy mixed "Schedules" (nav), "New Scheduled Task" (form),
"Reminder created." (toast), "Only reminder tasks can be edited here."
(guard). Rows managed by other systems must say what they are and where
to edit them instead of exposing the internal "reminder" noun.
"""

from datetime import datetime, timezone

import pytest

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Scheduling.models import (
    ReminderTask,
    ScheduledTask,
    ScheduleKind,
    TaskStatus,
)
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)
from tldw_chatbook.UI.Screens.scheduling.task_detail import (
    _managed_elsewhere_notice,
)


def _projection(task_type: str = "watchlist_job") -> ScheduledTask:
    return ScheduledTask(
        id=f"{task_type}:1",
        title="Watchlist Title",
        type=task_type,
        status=TaskStatus.WAITING,
        schedule_summary="Every 1h",
        next_run_at=datetime(2099, 7, 20, 11, 0, tzinfo=timezone.utc),
    )


def test_managed_elsewhere_notice_names_the_owning_screen():
    assert _managed_elsewhere_notice(_projection("watchlist_job")) == (
        "Managed by Watchlists — edit it there."
    )
    assert _managed_elsewhere_notice(_projection("briefing_job")) == (
        "Managed by Watchlists — edit it there."
    )
    assert "reminder" not in _managed_elsewhere_notice(
        _projection("mystery_job")
    ).lower()


from Tests.UI.schedules_test_helpers import MockSchedulingServiceMixin


class _MixedService(MockSchedulingServiceMixin):
    def __init__(self) -> None:
        self.created: list[dict] = []

    async def list_tasks(self, owner_id=None, include_projections=True):
        return [
            ReminderTask(
                id="task-1",
                title="Reminder",
                schedule_kind=ScheduleKind.ONE_TIME,
                run_at=datetime(2099, 7, 20, 10, 0, tzinfo=timezone.utc),
                next_run_at=datetime(2099, 7, 20, 10, 0, tzinfo=timezone.utc),
            ),
            _projection(),
        ]

    async def create_reminder(self, payload: dict, *, owner_id: str | None = None):
        self.created.append(payload)


class _App(ConsolidatedCSSApp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduling_service = _MixedService()


async def _mounted_workbench(pilot):
    workbench = SchedulesWorkbench(app_instance=pilot.app)
    await pilot.app.push_screen(workbench)
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return workbench


# redesign PR-2, Task 2: the three tests that used to live here (edit
# guard / toggle guard / detail-pane ownership line, all triggered by
# selecting a watchlist projection AT ROW 1 of the queue table) pinned a
# scenario the redesign retires -- watchlist/briefing projections no
# longer enter the unified Queue list at all (spec S2 locked decision 2,
# Task 1's report), so there is no longer a route to reach the
# `_managed_elsewhere_notice` guard FROM this table with a projection.
# The guard's own copy generation stays covered by
# `test_managed_elsewhere_notice_names_the_owning_screen` above (a
# direct, workbench-free unit test of `_managed_elsewhere_notice` itself);
# `TaskDetail.set_task`'s "#scheduling-task-detail-managed" ownership-line
# RENDERING for a `ScheduledTask` has no dedicated test anywhere else in
# the suite and genuinely loses coverage here -- the code path is now
# unreachable from this screen (a `ScheduledTask` is never fed to
# `TaskDetail` from the Queue table any more) but is left in place,
# unmodified, in case another future caller needs it.


@pytest.mark.asyncio
async def test_create_toast_uses_the_scheduled_task_noun():
    app = _App()
    async with app.run_test(size=(160, 48)) as pilot:
        workbench = await _mounted_workbench(pilot)
        workbench._on_reminder_form_result(
            {
                "title": "T",
                "body": "",
                "schedule_kind": "one_time",
                "run_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
                "cron": None,
                "timezone": None,
            }
        )
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        messages = [n.message for n in pilot.app._notifications]
        assert "Scheduled task created." in messages, messages


def _leftmost_call_name(node) -> str | None:
    """Resolve the leftmost Name in a call chain like logger.opt(...).warning."""
    import ast

    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None


def _sentence_copy_offenders(module) -> list[tuple[int, str]]:
    """All string literals in ``module`` that read as sentence copy using
    the internal "reminder" noun.

    task-23106 review round (F12): the previous line-regex sweep excluded
    any line containing "notify(" or "logger" and missed copy ending at a
    quote boundary; this walks the AST instead, so formatting cannot hide
    an offender. Deliberate allowances:
    - docstrings (internal documentation, not user-facing copy)
    - string arguments anywhere inside a ``logger.*(...)`` call (log
      lines, not screen copy)
    - strings without spaces (ids, group names, selectors -- not
      sentences)
    - DEFAULT_CSS / BUNDLED_CSS class attributes (stylesheet selectors
      like ``#reminder-cron-group``, not copy)
    """
    import ast
    import re
    from pathlib import Path

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    excluded_ids: set[int] = set()
    for node in ast.walk(tree):
        # Docstrings: the first statement of a module/class/function body.
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                excluded_ids.add(id(body[0].value))
        # Logger calls: every string constant inside the call subtree.
        if isinstance(node, ast.Call) and _leftmost_call_name(node.func) == "logger":
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    excluded_ids.add(id(child))
        # Stylesheet class attributes: selectors, not user-facing copy.
        if isinstance(node, ast.Assign):
            targets = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if targets & {"DEFAULT_CSS", "BUNDLED_CSS", "BUNDLED_SCREEN_CSS"}:
                if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str
                ):
                    excluded_ids.add(id(node.value))

    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in excluded_ids:
            continue
        value = node.value
        if " " not in value:
            continue  # identifiers, ids, group names -- not sentences
        if re.search(r"\breminders?\b", value, re.IGNORECASE):
            offenders.append((node.lineno, value))
    return offenders


def test_no_bare_reminder_noun_in_schedules_screen_copy():
    """AST sweep: sentence copy must not expose the "reminder" noun.

    The internal identifiers (ReminderTask, create_reminder, ...) are
    deliberately untouched (task-23106) -- this checks string literals
    on the modules that own the Schedules screen's user-facing copy,
    including f-string fragments and implicit concatenations.
    """
    import tldw_chatbook.UI.Screens.scheduling.forms.reminder_form as form
    import tldw_chatbook.UI.Screens.scheduling.schedules_workbench as wb
    import tldw_chatbook.UI.Screens.scheduling.task_detail as td

    for module in (wb, td, form):
        offenders = _sentence_copy_offenders(module)
        assert not offenders, (module.__name__, offenders)
