from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Research_Modules.research_controller import ResearchController
from tldw_chatbook.UI.Research_Window import ResearchWindow, _parse_limits_text

# NOTE (task-16322, ADR-068): ``ResearchScreen`` is back -- the local
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
async def test_research_window_reports_no_pending_local_checkpoint():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service)
    window = ResearchWindow(app)

    local_runs = await window.load_runs("local")
    window.select_run(local_runs[0])

    updated = await window.approve_selected_checkpoint()

    # task-16482: local approval EXISTS now; the unavailable case is "no
    # pending checkpoint" (no local service wired in this test).
    assert updated is None
    assert "No pending checkpoint" in window.status_message


# --- local engine start wiring (task-16322, ADR-068) ---------------------------

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


# --- academic lane toggle (task-16328) ------------------------------------------

def test_window_academic_toggle_defaults_off_and_persists_in_state():
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=None)
    window = ResearchWindow(app_instance=app)

    assert window.academic_enabled is False
    assert window.save_state() == {"source": "local", "academic": False,
                                   "limits": "", "policy": "balanced",
                                   "providers": ""}

    window.academic_enabled = True
    assert window.save_state() == {"source": "local", "academic": True,
                                   "limits": "", "policy": "balanced",
                                   "providers": ""}


def test_window_academic_toggle_restores_from_state():
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=None)
    window = ResearchWindow(app_instance=app)

    window.restore_state({"source": "local", "academic": True})

    assert window.academic_enabled is True


def test_window_academic_toggle_default_comes_from_config(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.UI.Research_Window._academic_lane_default", lambda: True
    )
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=None)
    window = ResearchWindow(app_instance=app)

    assert window.academic_enabled is True


def test_window_engine_start_passes_paper_fn_only_when_toggle_on(monkeypatch):
    from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
    from tldw_chatbook.Research_Interop import academic_providers

    captured = {}

    class FakeEngine:
        def __init__(self, service, **kwargs):
            captured["paper_search_fn"] = kwargs.get("paper_search_fn")

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.local_research_engine.LocalResearchEngine",
        FakeEngine,
    )
    service = FakeResearchScopeService()
    local_service = LocalResearchService(":memory:")
    app = SimpleNamespace(research_scope_service=service,
                          local_research_service=local_service)

    window_off = ResearchWindow(app_instance=app)
    window_off._start_local_engine("run-1")
    assert captured["paper_search_fn"] is None

    window_on = ResearchWindow(app_instance=app)
    window_on.academic_enabled = True
    window_on._start_local_engine("run-2")
    assert captured["paper_search_fn"] is academic_providers.search_papers


# --- budgets + follow-up Q&A in the window (task-16334) ---------------------------


def test_parse_limits_text_numeric_pairs():
    limits, warnings = _parse_limits_text("max_searches=5, max_runtime_seconds=120")
    assert limits == {"max_searches": 5, "max_runtime_seconds": 120.0}
    assert warnings == []


def test_parse_limits_text_invalid_pairs_warn_and_are_excluded():
    limits, warnings = _parse_limits_text("max_searches=five, junk, max_tokens=100")
    assert limits == {"max_tokens": 100}
    assert len(warnings) == 2


def test_parse_limits_text_empty():
    assert _parse_limits_text("") == ({}, [])
    assert _parse_limits_text("   ") == ({}, [])


@pytest.mark.asyncio
async def test_create_run_carries_parsed_limits(monkeypatch):
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    monkeypatch.setattr(window, "_start_local_engine", lambda run_id: None)

    window.limits_text = "max_searches=3, max_fetched_docs=10"
    await window.create_run({"query": "Budgeted question"})

    create_call = [c for c in service.calls if c[0] == "create_run"][0]
    assert create_call[2]["limits_json"] == {
        "max_searches": 3, "max_fetched_docs": 10.0,
    }


def test_limits_text_persists_in_state():
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=None)
    window = ResearchWindow(app_instance=app)
    window.limits_text = "max_searches=3"
    assert window.save_state()["limits"] == "max_searches=3"

    window2 = ResearchWindow(app_instance=app)
    window2.restore_state({"source": "local", "limits": "max_searches=4"})
    assert window2.limits_text == "max_searches=4"


@pytest.mark.asyncio
async def test_ask_follow_up_answers_from_selected_local_run(monkeypatch):
    from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService

    asked = {}

    class FakeEngine:
        def __init__(self, service, **kwargs):
            asked["params"] = kwargs.get("search_params")

        async def answer_follow_up(self, run_id, question, **kwargs):
            asked["run_id"] = run_id
            asked["question"] = question
            return {"status": "answered", "answer": "Because the claims say so.",
                    "question": question}

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.local_research_engine.LocalResearchEngine",
        FakeEngine,
    )
    local_service = LocalResearchService(":memory:")
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service,
                          local_research_service=local_service)
    window = ResearchWindow(app_instance=app)
    await window.load_runs("local")
    window.select_run(window.runs[0])

    result = await window.ask_follow_up("Why is that?")

    assert asked["run_id"] == "local-run"
    assert asked["question"] == "Why is that?"
    assert result["status"] == "answered"
    assert "Because the claims say so." in window.followup_answer_text


@pytest.mark.asyncio
async def test_ask_follow_up_insufficient_verdict_is_displayed_not_faked(monkeypatch):
    from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService

    class FakeEngine:
        def __init__(self, service, **kwargs):
            pass

        async def answer_follow_up(self, run_id, question, **kwargs):
            return {"status": "insufficient_evidence", "answer": None,
                    "reason": "no stored claims",
                    "suggestion": "Launch a new research run."}

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.local_research_engine.LocalResearchEngine",
        FakeEngine,
    )
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=LocalResearchService(":memory:"))
    window = ResearchWindow(app_instance=app)
    await window.load_runs("local")
    window.select_run(window.runs[0])

    result = await window.ask_follow_up("Anything?")

    assert result["status"] == "insufficient_evidence"
    assert "insufficient" in window.followup_answer_text
    assert "Launch a new research run." in window.followup_answer_text


@pytest.mark.asyncio
async def test_ask_follow_up_requires_selection_and_local_source(monkeypatch):
    constructed = {"n": 0}

    class FakeEngine:
        def __init__(self, *a, **k):
            constructed["n"] += 1

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.local_research_engine.LocalResearchEngine",
        FakeEngine,
    )
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=None)
    window = ResearchWindow(app_instance=app)

    result = await window.ask_follow_up("No run selected")  # no selection
    assert result is None
    assert constructed["n"] == 0
    assert "Select a research run" in window.status_message

    await window.load_runs("local")
    window.select_run(window.runs[0])
    # Set the source directly: switch_source() clears the selection, which
    # would trip the no-selection guard instead of the source guard.
    window.current_source = "server"
    result = await window.ask_follow_up("Server run")
    assert result is None
    assert constructed["n"] == 0
    assert "local" in window.status_message


# --- local checkpoint approval (task-16482) ---------------------------------------

@pytest.mark.asyncio
async def test_window_approves_latest_local_checkpoint_and_restarts_engine(monkeypatch):
    from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService

    local_service = LocalResearchService(":memory:")
    run = local_service.launch_run(query="Checkpointed run")
    checkpoint = local_service.create_checkpoint(
        run["id"], checkpoint_type="plan_review", proposed_payload={"query": "q"}
    )
    service = FakeResearchScopeService()
    service.runs["local"] = [
        SimpleNamespace(id=run["id"], query="Checkpointed run", status="running",
                        phase="planning", control_state="awaiting_plan_review",
                        latest_checkpoint_id=checkpoint["id"])
    ]
    app = SimpleNamespace(research_scope_service=service,
                          local_research_service=local_service)
    window = ResearchWindow(app_instance=app)
    restarted = []
    monkeypatch.setattr(window, "_start_local_engine", restarted.append)

    async def approve(scope_run_id, checkpoint_id, patch_payload):
        return local_service.patch_and_approve_checkpoint(
            scope_run_id, checkpoint_id, patch_payload=patch_payload
        )

    async def fake_scope_approve(run_id_arg, checkpoint_id, *, mode, patch_payload):
        assert mode == "local"
        return await approve(run_id_arg, checkpoint_id, patch_payload)

    service.patch_and_approve_checkpoint = fake_scope_approve

    await window.load_runs("local")
    window.select_run(window.runs[0])
    updated = await window.approve_selected_checkpoint(patch_payload={"limits": {}})

    assert updated["status"] == "approved"
    assert local_service.latest_pending_checkpoint(run["id"]) is None
    assert restarted == [run["id"]]


# --- readable bundle inspection (task-16483) --------------------------------------

@pytest.mark.asyncio
async def test_load_bundle_auto_loads_the_report(monkeypatch):
    service = FakeResearchScopeService()

    async def get_bundle(run_id, *, mode):
        return {
            "run": {"id": run_id, "status": "completed", "phase": "completed",
                    "query": "What is RAG?"},
            "artifacts": [
                {"artifact_name": "plan.json", "content_type": "application/json",
                 "content": {"query": "What is RAG?"}},
                {"artifact_name": "report_v1.md", "content_type": "text/markdown",
                 "content": "# Report\nAnswer[1]."},
            ],
        }

    async def get_artifact(run_id, artifact_name, *, mode):
        return {"artifact_name": artifact_name, "content_type": "text/markdown",
                "artifact_version": 1, "content": "# Report\nAnswer[1]."}

    service.get_bundle = get_bundle
    service.get_artifact = get_artifact
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    monkeypatch.setattr(window, "_start_local_engine", lambda run_id: None)
    await window.load_runs("local")
    window.select_run(window.runs[0])

    bundle = await window.load_selected_run_bundle()

    assert bundle is not None
    # The run record is NOT selected as the artifact; the report is.
    assert window.current_artifact is not None
    assert window.current_artifact["artifact_name"] == "report_v1.md"
    assert "Answer[1]." in window.current_artifact["content"]


# --- selected-run auto-refresh (task-16486) ---------------------------------------

@pytest.mark.asyncio
async def test_auto_refresh_updates_non_terminal_local_run_preserving_payload():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    await window.load_runs("local")
    window.select_run(window.runs[0])
    window.runs[0].status = "running"
    window.current_bundle = {"kept": True}

    calls = {"n": 0}

    async def get_run(run_id, *, mode):
        calls["n"] += 1
        return SimpleNamespace(id=run_id, query="Local query", status="running",
                               phase="synthesizing", control_state="running")

    service.get_run = get_run
    await window._auto_refresh_selected_run()

    assert calls["n"] == 1
    assert window.selected_run.phase == "synthesizing"
    assert window.current_bundle == {"kept": True}  # payload state preserved


@pytest.mark.asyncio
async def test_auto_refresh_skips_terminal_and_non_local():
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    await window.load_runs("local")
    window.select_run(window.runs[0])
    window.runs[0].status = "completed"  # terminal: no refresh

    calls = {"n": 0}

    async def get_run(run_id, *, mode):
        calls["n"] += 1
        return window.runs[0]

    service.get_run = get_run
    await window._auto_refresh_selected_run()
    assert calls["n"] == 0

    window.runs[0].status = "running"
    window.current_source = "server"  # server source: not our engine
    await window._auto_refresh_selected_run()
    assert calls["n"] == 0


# --- Qodo remediation (task-16814) ------------------------------------------------

def test_academic_lane_default_parses_string_booleans(monkeypatch):
    from tldw_chatbook.UI.Research_Window import _parse_config_bool

    for raw, expected in [
        ("false", False), ("False", False), ("0", False), ("no", False),
        ("off", False), ("", False),
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        (True, True), (False, False),
    ]:
        assert _parse_config_bool(raw) is expected, f"{raw!r} should parse {expected}"


# --- source policy + providers in the window (task-16791) -------------------------

@pytest.mark.asyncio
async def test_window_policy_and_providers_sent_on_create(monkeypatch):
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    monkeypatch.setattr(window, "_start_local_engine", lambda run_id: None)

    window.source_policy = "academic_only"
    window.providers_text = "arxiv, pubmed"
    await window.create_run({"query": "Policy question"})

    create_call = [c for c in service.calls if c[0] == "create_run"][0]
    payload = create_call[2]
    assert payload["source_policy"] == "academic_only"
    assert payload["provider_overrides"] == {"academic_providers": ["arxiv", "pubmed"]}


def test_window_policy_persists_in_state():
    app = SimpleNamespace(research_scope_service=FakeResearchScopeService(),
                          local_research_service=None)
    window = ResearchWindow(app_instance=app)
    window.source_policy = "web_first"
    window.providers_text = "pubmed"
    state = window.save_state()
    assert state["policy"] == "web_first"
    assert state["providers"] == "pubmed"

    window2 = ResearchWindow(app_instance=app)
    window2.restore_state(state)
    assert window2.source_policy == "web_first"
    assert window2.providers_text == "pubmed"


# --- Qodo remediation on PR 1722 ---------------------------------------------------

@pytest.mark.asyncio
async def test_create_run_reads_limits_and_providers_from_inputs(monkeypatch):
    service = FakeResearchScopeService()
    app = SimpleNamespace(research_scope_service=service, local_research_service=None)
    window = ResearchWindow(app_instance=app)
    monkeypatch.setattr(window, "_start_local_engine", lambda run_id: None)

    # Simulate what compose() + typing produces: the widgets' values must
    # reach the payload even when only the attributes were never set.
    window.limits_text = "max_searches=3"
    window.providers_text = "pubmed, arxiv, typo_lane"
    await window.create_run({"query": "Inputs question"})

    create_call = [c for c in service.calls if c[0] == "create_run"][0]
    payload = create_call[2]
    assert payload["provider_overrides"]["academic_providers"] == ["pubmed", "arxiv"]
    assert payload["limits_json"] == {"max_searches": 3}


def test_parse_provider_tokens_validates_and_dedupes():
    from tldw_chatbook.UI.Research_Window import _parse_provider_tokens

    assert _parse_provider_tokens("pubmed, arxiv, pubmed") == ["pubmed", "arxiv"]
    assert _parse_provider_tokens("  ARXIV ,, biorxiv ") == ["arxiv", "biorxiv"]
    # Dangerous input fails validation -> empty list (caller warns).
    assert _parse_provider_tokens("pubmed, <script>x") == []
