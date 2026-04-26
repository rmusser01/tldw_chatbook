import pytest

from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService


class RecordingDispatcher:
    def __init__(self):
        self.calls = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": len(self.calls), **kwargs}


def test_local_research_service_persists_sessions_runs_events_and_artifacts(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")

    session = service.create_session(title="MCP governance", query="How should MCP approvals work?")
    updated_session = service.update_session(session["id"], expected_version=1, notes="Focus on local/server scope")
    run = service.launch_run(session_id=session["id"], query=session["query"])
    paused = service.pause_run(run["id"])
    resumed = service.resume_run(run["id"])
    service.save_artifact(run["id"], artifact_name="notes.md", content_type="text/markdown", content="# Notes")
    artifact = service.get_artifact(run["id"], "notes.md")
    bundle = service.get_bundle(run["id"])
    events = list(service.list_run_events(run["id"]))

    assert session["record_id"].startswith("local:research_session:")
    assert updated_session["notes"] == "Focus on local/server scope"
    assert updated_session["version"] == 2
    assert run["record_id"].startswith("local:research_run:")
    assert paused["control_state"] == "paused"
    assert resumed["control_state"] == "running"
    assert artifact == {"artifact_name": "notes.md", "content_type": "text/markdown", "content": "# Notes"}
    assert bundle["artifacts"][0]["artifact_name"] == "notes.md"
    assert [event["event"] for event in events] == ["created", "paused", "resumed", "artifact_saved"]


def test_local_research_service_soft_deletes_sessions_and_runs(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    session = service.create_session(title="MCP governance", query="How should MCP approvals work?")
    run = service.launch_run(session_id=session["id"], query=session["query"])

    assert service.delete_run(run["id"], expected_version=1) is True
    assert service.delete_session(session["id"], expected_version=1) is True

    assert service.get_run(run["id"]) is None
    assert service.get_session(session["id"]) is None
    assert service.list_runs() == []
    assert service.list_sessions() == []


def test_local_research_service_rejects_stale_versions(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    session = service.create_session(title="MCP governance", query="How should MCP approvals work?")

    with pytest.raises(ValueError, match="version conflict"):
        service.update_session(session["id"], expected_version=2, notes="Stale")


def test_local_research_service_can_clear_nullable_session_fields(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    session = service.create_session(title="MCP governance", query="How should MCP approvals work?", notes="Draft")

    updated = service.update_session(session["id"], expected_version=1, notes=None)

    assert updated["notes"] is None
    assert updated["version"] == 2


def test_local_research_service_dispatches_terminal_run_notifications(tmp_path):
    dispatcher = RecordingDispatcher()
    app = object()
    service = LocalResearchService(
        tmp_path / "research.db",
        notification_dispatcher=dispatcher,
        notification_app=app,
    )
    session = service.create_session(title="MCP governance", query="How should MCP approvals work?")
    run = service.launch_run(session_id=session["id"])

    completed = service.complete_run(run["id"], progress_message="Final report ready")

    assert completed["status"] == "completed"
    assert completed["control_state"] == "completed"
    assert completed["progress_percent"] == 100.0
    assert dispatcher.calls == [
        {
            "app": app,
            "category": "research",
            "title": "Research run completed",
            "message": "How should MCP approvals work?",
            "severity": "information",
            "source_backend": "local",
            "source_entity_kind": "research_run",
            "source_entity_id": run["id"],
            "payload": {
                "run_id": run["id"],
                "session_id": session["id"],
                "status": "completed",
                "control_state": "completed",
                "query": "How should MCP approvals work?",
            },
        }
    ]
