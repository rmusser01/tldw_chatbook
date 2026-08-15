from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Research_Modules.research_controller import ResearchController
from tldw_chatbook.UI.Research_Window import ResearchWindow

# NOTE (task-16322, ADR-066): ``ResearchScreen`` is back -- the local
# research execution engine drives launched local runs, so the window is
# reachable from navigation again under the "research" route id (the
# task-255 library alias is reversed). The screen-level behavior is covered
# by Tests/UI/test_screen_navigation.py; this module covers
# ``ResearchWindow``/``ResearchController`` and the engine-start wiring.


class FakeResearchScopeService:
    def __init__(self):
        self.calls = []
        self.runs = {
            "local": [
                SimpleNamespace(
                    id="local-run",
                    query="Local query",
                    status="draft",
                    phase="planning",
                    control_state="paused",
                    latest_checkpoint_id=None,
                )
            ],
            "server": [
                SimpleNamespace(
                    id="server-run",
                    query="Server query",
                    status="running",
                    phase="collecting",
                    control_state="running",
                    latest_checkpoint_id="checkpoint-1",
                )
            ],
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

    async def resume_run(self, run_id, *, mode):
        self.calls.append(("resume_run", mode, run_id))
        resumed = self.runs[mode][0] if self.runs[mode] else None
        if resumed is not None:
            resumed.status = "running"
            resumed.control_state = "running"
        return resumed

    async def pause_run(self, run_id, *, mode):
        self.calls.append(("pause_run", mode, run_id))
        return SimpleNamespace(
            id=run_id, query="Paused", status="running", control_state="paused"
        )

    async def get_bundle(self, run_id, *, mode):
        self.calls.append(("get_bundle", mode, run_id))
        return {
            "report.md": "# Report",
            "sources.json": {"count": 2, "run_id": run_id, "mode": mode},
        }

    async def get_artifact(self, run_id, artifact_name, *, mode):
        self.calls.append(("get_artifact", mode, run_id, artifact_name))
        return SimpleNamespace(
            run_id=run_id,
            artifact_name=artifact_name,
            content_type="text/markdown",
            content=f"# Artifact for {run_id}",
            artifact_version=1,
        )

    async def patch_and_approve_checkpoint(
        self, run_id, checkpoint_id, *, mode, patch_payload=None
    ):
        self.calls.append(
            ("patch_and_approve_checkpoint", mode, run_id, checkpoint_id, patch_payload)
        )
        if mode == "local":
            raise ValueError(
                "Local research checkpoints are not available in this slice."
            )
        return SimpleNamespace(
            id=run_id,
            query="Server query",
            status="running",
            phase="synthesizing",
            control_state="running",
            latest_checkpoint_id=checkpoint_id,
            progress_message="Checkpoint approved",
        )

    async def stream_run_events(self, run_id, *, mode, after_id=0):
        self.calls.append(("stream_run_events", mode, run_id, after_id))
        if mode == "local":
            raise ValueError(
                "Local research live events are not available in this slice."
            )
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
async def test_research_controller_routes_bundle_artifact_and_checkpoint_actions():
    service = FakeResearchScopeService()
    controller = ResearchController(service)

    bundle = await controller.get_bundle("server", "server-run")
    artifact = await controller.get_artifact("server", "server-run", "report.md")
    updated = await controller.patch_and_approve_checkpoint(
        "server",
        "server-run",
        "checkpoint-1",
        {"resolution": "accept"},
    )

    assert bundle["report.md"] == "# Report"
    assert artifact.artifact_name == "report.md"
    assert updated.latest_checkpoint_id == "checkpoint-1"
    assert service.calls == [
        ("get_bundle", "server", "server-run"),
        ("get_artifact", "server", "server-run", "report.md"),
        (
            "patch_and_approve_checkpoint",
            "server",
            "server-run",
            "checkpoint-1",
            {"resolution": "accept"},
        ),
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
    assert len(window.event_log_entries) == 2
    assert "snapshot" in window.event_log_entries[0]
    assert "progress" in window.event_log_entries[1]


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


@pytest.mark.asyncio
async def test_research_window_loads_bundle_and_artifact_for_selected_run():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    server_runs = await window.switch_source("server")
    window.select_run(server_runs[0])

    bundle = await window.load_selected_run_bundle()
    artifact = await window.load_selected_run_artifact("report.md")

    assert bundle["report.md"] == "# Report"
    assert artifact.artifact_name == "report.md"
    assert window.current_bundle == bundle
    assert window.current_artifact == artifact
    assert ("get_bundle", "server", "server-run") in service.calls
    assert ("get_artifact", "server", "server-run", "report.md") in service.calls


@pytest.mark.asyncio
async def test_research_window_approves_selected_server_checkpoint():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    server_runs = await window.switch_source("server")
    window.select_run(server_runs[0])

    updated = await window.approve_selected_checkpoint(
        patch_payload={"resolution": "accept"}
    )

    assert updated.latest_checkpoint_id == "checkpoint-1"
    assert getattr(window.selected_run, "latest_checkpoint_id", None) == "checkpoint-1"
    assert (
        "patch_and_approve_checkpoint",
        "server",
        "server-run",
        "checkpoint-1",
        {"resolution": "accept"},
    ) in service.calls


@pytest.mark.asyncio
async def test_research_window_reports_local_checkpoint_approval_unavailable():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    local_runs = await window.load_runs("local")
    window.select_run(local_runs[0])

    updated = await window.approve_selected_checkpoint()

    assert updated is None
    assert "Local research checkpoints" in window.status_message


# --- local engine start wiring (task-16322, ADR-066) ---------------------------

@pytest.mark.asyncio
async def test_research_window_starts_local_engine_after_local_create(monkeypatch):
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    started = []
    monkeypatch.setattr(window, "_start_local_engine", started.append)

    created = await window.create_run({"query": "Local engine question"})

    assert created.id == "local-created"
    assert started == ["local-created"]


@pytest.mark.asyncio
async def test_research_window_does_not_start_engine_for_server_create(monkeypatch):
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    started = []
    monkeypatch.setattr(window, "_start_local_engine", started.append)
    await window.switch_source("server")

    await window.create_run({"query": "Server question"})

    assert started == []


@pytest.mark.asyncio
async def test_research_window_resume_restarts_local_engine(monkeypatch):
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    started = []
    monkeypatch.setattr(window, "_start_local_engine", started.append)
    await window.load_runs("local")
    window.select_run(window.runs[0])
    window.runs[0].status = "running"
    window.runs[0].control_state = "paused"

    await window.resume_selected_run()

    assert started == ["local-run"]


def test_research_window_engine_start_skips_without_local_service():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    # Not mounted, no worker infrastructure: must not raise, just report.
    window._start_local_engine("local-run")


def test_research_window_engine_start_builds_engine_from_app_service(monkeypatch):
    from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService

    service = FakeResearchScopeService()
    local_service = LocalResearchService(":memory:")
    app = SimpleNamespace(
        research_scope_service=service, local_research_service=local_service
    )
    window = ResearchWindow(app_instance=app)
    engines = []
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.local_research_engine.LocalResearchEngine",
        lambda svc, **kwargs: engines.append(svc) or SimpleNamespace(),
    )

    window._start_local_engine("local-run")

    assert engines == [local_service]
