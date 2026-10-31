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


@pytest.mark.asyncio
async def test_review_team_config_load_and_atomic_save_run_offloop(
    tmp_path, monkeypatch
):
    import os
    from pathlib import Path
    from typing import ClassVar

    from textual.app import App

    from Tests.UI.consolidated_css import APP_STYLESHEETS
    from tldw_chatbook.Agents.code_review_models import ReviewTeam
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Widgets import settings_agents_panel as module

    threads = []

    def load():
        threads.append(("read", threading.get_ident()))
        return ReviewTeam()

    def save(*args, **kwargs):
        threads.append(("write", threading.get_ident()))
        return save_settings_to_cli_config(*args, **kwargs)

    monkeypatch.setattr(module, "get_code_review_team", load)
    monkeypatch.setattr(module, "save_settings_to_cli_config", save)
    path = Path(os.environ["TLDW_CONFIG_PATH"])
    original = path.read_bytes()
    db = AgentRunsDB(tmp_path / "agents.db", client_id="test")
    panel = module.AgentsSettingsPanel(
        SimpleNamespace(app_config={"agents": {"code_review": {}}}), runs_db=db
    )

    class Harness(App):
        CSS_PATH: ClassVar[list[str]] = [str(p) for p in APP_STYLESHEETS]

        def compose(self):
            yield panel

    try:
        async with Harness().run_test(size=(80, 24)):
            await panel._save_review_team()
        assert {kind for kind, _ in threads} == {"read", "write"}
        assert all(t != threading.get_ident() for _, t in threads)
    finally:
        db.close()
        path.write_bytes(original)
