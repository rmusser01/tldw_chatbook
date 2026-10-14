"""The Settings agents-panel definition read runs off the event loop."""

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.settings_agents_panel import AgentsSettingsPanel

pytestmark = pytest.mark.bootstrap_profile


class _RecordingDB:
    def __init__(self, rows) -> None:
        self._rows = rows
        self.read_threads: list[int] = []

    def list_agent_definitions(self):
        self.read_threads.append(threading.get_ident())
        return self._rows


class _RecordingListView:
    def __init__(self) -> None:
        self.cleared = 0
        self.batches: list[list] = []

    async def clear(self):
        self.cleared += 1

    async def extend(self, items):
        self.batches.append(list(items))


class _Panel(AgentsSettingsPanel):
    """Bare shell: only _reload_list's collaborators are stubbed."""

    def __init__(self) -> None:  # skip the heavy real panel init
        pass


@pytest.mark.asyncio
async def test_definition_read_and_row_batching():
    rows = [
        {"id": str(index), "name": f"agent-{index}", "enabled": index % 2 == 0}
        for index in range(7)
    ]
    db = _RecordingDB(rows)
    panel = _Panel()
    panel._runs_db = db
    list_view = _RecordingListView()
    panel.query_one = lambda *a, **k: list_view
    panel._set_status = lambda text: None

    await panel._reload_list()

    assert panel._rows == rows
    assert db.read_threads and all(
        thread != threading.get_ident() for thread in db.read_threads
    ), "the SQLite read must run off the event-loop thread"
    assert len(list_view.batches) == 1, "rows append in ONE batch, not one per row"
    assert len(list_view.batches[0]) == 7
