"""Schedules overlay and mark-read queries run off the UI thread."""

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.scheduling import schedules_workbench as module
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
    SchedulesWorkbench,
)

pytestmark = pytest.mark.bootstrap_profile


class _RecordingDB:
    def __init__(self) -> None:
        self.read_threads: list[int] = []

    def _record(self, value):
        self.read_threads.append(threading.get_ident())
        return value

    def list_automation_results(self, **_kwargs):
        return self._record(([],))

    def count_automation_results(self, **_kwargs):
        return self._record(0)

    def list_automation_definitions(self, **_kwargs):
        return self._record([])

    def get_conflicts(self, *_args, **_kwargs):
        return self._record([])


class _OffloopWorkbench(SchedulesWorkbench):
    """Bare shell: app is a read-only property on real widgets."""

    app = None

    def __init__(self) -> None:  # skip the heavy real workbench init
        pass


def _workbench(db: _RecordingDB):
    workbench = _OffloopWorkbench()
    service = SimpleNamespace(db=db, owner_id=None)
    workbench._service = lambda: service
    pushes = []
    workbench.app = SimpleNamespace(push_screen=lambda *args, **kw: pushes.append(args))
    return workbench, pushes


@pytest.mark.asyncio
async def test_results_overlay_reads_run_off_the_loop(monkeypatch):
    monkeypatch.setattr(module, "ResultsHostScreen", lambda *args, **kwargs: object())
    db = _RecordingDB()
    workbench, pushes = _workbench(db)

    await workbench._push_results_overlay()
    assert pushes, "the overlay must still be pushed"
    assert db.read_threads, "the three inbox reads must run"
    assert all(
        thread != threading.get_ident() for thread in db.read_threads
    ), "reads must not run on the event-loop thread"


@pytest.mark.asyncio
async def test_conflicts_overlay_read_runs_off_the_loop(monkeypatch):
    monkeypatch.setattr(
        module, "WorkbenchHostScreen", lambda *args, **kwargs: object()
    )
    db = _RecordingDB()
    workbench, pushes = _workbench(db)
    workbench._service = lambda: SimpleNamespace(
        db=db, owner_id=None, sync_engine=None
    )

    await workbench._push_conflicts_overlay()
    assert pushes
    assert db.read_threads and all(
        thread != threading.get_ident() for thread in db.read_threads
    )
