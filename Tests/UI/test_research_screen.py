from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Research_Modules.research_controller import ResearchController
from tldw_chatbook.UI.Research_Window import ResearchWindow
from tldw_chatbook.UI.Screens.research_screen import ResearchScreen


def test_research_screen_composes_research_window():
    app = SimpleNamespace(research_scope_service=object())
    screen = ResearchScreen(app)

    widgets = list(screen.compose_content())

    assert len(widgets) == 1
    assert isinstance(widgets[0], ResearchWindow)


def test_research_screen_round_trips_window_state():
    app = SimpleNamespace(research_scope_service=object())
    screen = ResearchScreen(app)
    window = ResearchWindow(app)
    window.restore_state({"source": "server"})
    screen.query_one = lambda *_args, **_kwargs: window

    state = screen.save_state()
    screen.restore_state({"source": "local"})

    assert state == {"source": "server"}
    assert window.save_state() == {"source": "local"}


class FakeResearchScopeService:
    def __init__(self):
        self.calls = []
        self.runs = {
            "local": [SimpleNamespace(id="local-run", query="Local query", status="draft", phase="planning")],
            "server": [SimpleNamespace(id="server-run", query="Server query", status="running", phase="collecting")],
        }

    async def list_runs(self, *, mode, limit=25):
        self.calls.append(("list_runs", mode, limit))
        return list(self.runs[mode])

    async def create_run(self, *, mode, **payload):
        self.calls.append(("create_run", mode, dict(payload)))
        run = SimpleNamespace(
            id=f"{mode}-created",
            query=payload["query"],
            status="draft" if mode == "local" else "running",
            phase="planning",
        )
        self.runs[mode].insert(0, run)
        return run

    async def pause_run(self, run_id, *, mode):
        self.calls.append(("pause_run", mode, run_id))
        return SimpleNamespace(id=run_id, query="Paused", status="running", control_state="paused")

    async def stream_run_events(self, run_id, *, mode, after_id=0):
        self.calls.append(("stream_run_events", mode, run_id, after_id))
        if mode == "local":
            raise ValueError("Local research live events are not available in this slice.")
        yield {
            "event": "snapshot",
            "id": "3",
            "data": {
                "run": {
                    "id": run_id,
                    "query": "Server query",
                    "status": "running",
                    "phase": "collecting",
                    "control_state": "running",
                    "progress_message": "Collecting sources",
                }
            },
        }
        yield {
            "event": "progress",
            "id": "4",
            "data": {"progress_message": "Synthesizing answer"},
        }


@pytest.mark.asyncio
async def test_research_controller_routes_runs_by_source():
    service = FakeResearchScopeService()
    controller = ResearchController(service)

    local_runs = await controller.load_runs("local")
    server_created = await controller.create_run("server", {"query": "Server query"})

    assert [run.id for run in local_runs] == ["local-run"]
    assert server_created.id == "server-created"
    assert service.calls == [
        ("list_runs", "local", 25),
        ("create_run", "server", {"query": "Server query"}),
    ]


@pytest.mark.asyncio
async def test_research_window_loads_and_selects_runs_without_mixed_sources():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    local_runs = await window.load_runs("local")
    server_runs = await window.switch_source("server")
    window.select_run(server_runs[0])

    assert [run.id for run in local_runs] == ["local-run"]
    assert [run.id for run in server_runs] == ["server-run"]
    assert window.current_source == "server"
    assert window.selected_run.id == "server-run"


@pytest.mark.asyncio
async def test_research_window_watches_selected_server_run_events():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    server_runs = await window.switch_source("server")
    window.select_run(server_runs[0])

    events = await window.watch_selected_run_events(after_id=3)

    assert [event["event"] for event in events] == ["snapshot", "progress"]
    assert ("stream_run_events", "server", "server-run", 3) in service.calls
    assert "Synthesizing answer" in window.status_message


@pytest.mark.asyncio
async def test_research_window_reports_local_live_events_unavailable():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    local_runs = await window.load_runs("local")
    window.select_run(local_runs[0])

    events = await window.watch_selected_run_events()

    assert events == []
    assert "Local research live events" in window.status_message
