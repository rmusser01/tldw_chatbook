import sqlite3

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

    session = service.create_session(
        title="MCP governance", query="How should MCP approvals work?"
    )
    updated_session = service.update_session(
        session["id"], expected_version=1, notes="Focus on local/server scope"
    )
    run = service.launch_run(session_id=session["id"], query=session["query"])
    paused = service.pause_run(run["id"])
    resumed = service.resume_run(run["id"])
    service.save_artifact(
        run["id"],
        artifact_name="notes.md",
        content_type="text/markdown",
        content="# Notes",
    )
    artifact = service.get_artifact(run["id"], "notes.md")
    bundle = service.get_bundle(run["id"])
    events = list(service.list_run_events(run["id"]))

    assert session["record_id"].startswith("local:research_session:")
    assert updated_session["notes"] == "Focus on local/server scope"
    assert updated_session["version"] == 2
    assert run["record_id"].startswith("local:research_run:")
    assert paused["control_state"] == "paused"
    assert resumed["control_state"] == "running"
    assert artifact == {
        "artifact_name": "notes.md",
        "content_type": "text/markdown",
        "content": "# Notes",
    }
    assert bundle["artifacts"][0]["artifact_name"] == "notes.md"
    assert [event["event"] for event in events] == [
        "created",
        "paused",
        "resumed",
        "artifact_saved",
    ]


def test_local_research_service_soft_deletes_sessions_and_runs(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    session = service.create_session(
        title="MCP governance", query="How should MCP approvals work?"
    )
    run = service.launch_run(session_id=session["id"], query=session["query"])

    assert service.delete_run(run["id"], expected_version=1) is True
    assert service.delete_session(session["id"], expected_version=1) is True

    assert service.get_run(run["id"]) is None
    assert service.get_session(session["id"]) is None
    assert service.list_runs() == []
    assert service.list_sessions() == []


def test_local_research_service_rejects_stale_versions(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    session = service.create_session(
        title="MCP governance", query="How should MCP approvals work?"
    )

    with pytest.raises(ValueError, match="version conflict"):
        service.update_session(session["id"], expected_version=2, notes="Stale")


def test_local_research_service_can_clear_nullable_session_fields(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    session = service.create_session(
        title="MCP governance", query="How should MCP approvals work?", notes="Draft"
    )

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
    session = service.create_session(
        title="MCP governance", query="How should MCP approvals work?"
    )
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


# --- update_run_progress (task-16322 engine seam) ------------------------------

def test_update_run_progress_sets_fields_records_event_and_bumps_version(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="How do persistent agents checkpoint?")

    updated = service.update_run_progress(
        run["id"],
        phase="collecting",
        progress_percent=45.0,
        progress_message="Collecting sources",
    )

    assert updated["phase"] == "collecting"
    assert updated["progress_percent"] == 45.0
    assert updated["progress_message"] == "Collecting sources"
    assert updated["status"] == "running"  # untouched
    assert updated["version"] == run["version"] + 1
    events = list(service.list_run_events(run["id"]))
    assert events[-1]["event"] == "progress"
    assert events[-1]["data"] == {"phase": "collecting", "progress_percent": 45.0}


def test_update_run_progress_supports_status_and_control_for_engine_start(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    draft = service.create_run(query="Draft question")

    started = service.update_run_progress(
        draft["id"],
        status="running",
        control_state="running",
        phase="planning",
        progress_percent=10.0,
        event="engine_started",
    )

    assert started["status"] == "running"
    assert started["control_state"] == "running"
    assert started["phase"] == "planning"
    events = list(service.list_run_events(draft["id"]))
    assert events[-1]["event"] == "engine_started"


def test_update_run_progress_missing_run_raises(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    with pytest.raises(ValueError, match="research run not found"):
        service.update_run_progress("local:research_run:nope", phase="collecting")


# --- checkpoints (task-16482) -----------------------------------------------------

def _launch_checkpoint_run(service, **kwargs):
    return service.launch_run(query="Checkpoint question", **kwargs)


def test_checkpoint_create_list_latest_pending(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    run = _launch_checkpoint_run(service)

    checkpoint = service.create_checkpoint(
        run["id"], checkpoint_type="plan_review", proposed_payload={"query": "q"}
    )

    assert checkpoint["checkpoint_type"] == "plan_review"
    assert checkpoint["status"] == "pending"
    listed = service.list_checkpoints(run["id"])
    assert [c["id"] for c in listed] == [checkpoint["id"]]
    assert service.latest_pending_checkpoint(run["id"])["id"] == checkpoint["id"]


def test_patch_and_approve_stores_patch_bumps_version_and_records_event(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    run = _launch_checkpoint_run(service)
    checkpoint = service.create_checkpoint(
        run["id"], checkpoint_type="plan_review", proposed_payload={"query": "q"}
    )

    approved = service.patch_and_approve_checkpoint(
        run["id"], checkpoint["id"], patch_payload={"limits": {"max_searches": 3}}
    )

    assert approved["status"] == "approved"
    assert approved["resolution"] == "approved"
    assert approved["user_patch"] == {"limits": {"max_searches": 3}}
    assert approved["version"] == checkpoint["version"] + 1
    assert service.latest_pending_checkpoint(run["id"]) is None
    events = [e["event"] for e in service.list_run_events(run["id"])]
    assert "checkpoint_approved" in events


def test_patch_validation_rejects_unknown_keys_and_bad_inventory(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    run = _launch_checkpoint_run(service)
    plan = service.create_checkpoint(
        run["id"], checkpoint_type="plan_review", proposed_payload={"query": "q"}
    )
    with pytest.raises(ValueError, match="unexpected patch keys"):
        service.patch_and_approve_checkpoint(
            run["id"], plan["id"], patch_payload={"bogus_key": 1}
        )

    sources = service.create_checkpoint(
        run["id"],
        checkpoint_type="sources_review",
        proposed_payload={"source_ids": ["s1", "s2"]},
    )
    with pytest.raises(ValueError, match="not in the proposed inventory"):
        service.patch_and_approve_checkpoint(
            run["id"], sources["id"], patch_payload={"pinned_source_ids": ["sX"]}
        )
    with pytest.raises(ValueError, match="disjoint"):
        service.patch_and_approve_checkpoint(
            run["id"],
            sources["id"],
            patch_payload={"pinned_source_ids": ["s1"], "dropped_source_ids": ["s1"]},
        )


def test_approve_requires_pending_checkpoint(tmp_path):
    service = LocalResearchService(tmp_path / "research.db")
    run = _launch_checkpoint_run(service)
    checkpoint = service.create_checkpoint(
        run["id"], checkpoint_type="plan_review", proposed_payload={"query": "q"}
    )
    service.patch_and_approve_checkpoint(run["id"], checkpoint["id"])

    with pytest.raises(ValueError, match="not pending"):
        service.patch_and_approve_checkpoint(run["id"], checkpoint["id"])


# --- external-DB transaction (task-16814) ------------------------------------------

class FakeExternalResearchDB:
    """Minimal external-DB double: transaction() yields a real sqlite conn
    (delete_run's precedent interface)."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE research_runs (
                id TEXT PRIMARY KEY, session_id TEXT, query TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                phase TEXT NOT NULL DEFAULT 'local_planning',
                control_state TEXT NOT NULL DEFAULT 'running',
                progress_percent REAL, progress_message TEXT,
                source_policy TEXT NOT NULL DEFAULT 'balanced',
                autonomy_mode TEXT NOT NULL DEFAULT 'checkpointed',
                limits_json TEXT NOT NULL DEFAULT '{}',
                provider_overrides_json TEXT NOT NULL DEFAULT '{}',
                chat_handoff_json TEXT NOT NULL DEFAULT '{}',
                follow_up_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                client_id TEXT NOT NULL DEFAULT 'local',
                version INTEGER NOT NULL DEFAULT 1
            );
            """
        )

    def transaction(self):
        return self.conn

    def get_run(self, run_id):
        row = self.conn.execute(
            "SELECT * FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()


def test_update_run_progress_external_db_wraps_in_transaction():
    external = FakeExternalResearchDB()
    now = "2026-08-16T00:00:00Z"
    external.conn.execute(
        "INSERT INTO research_runs (id, query, created_at, updated_at) "
        "VALUES ('ext-run', 'External question', ?, ?)",
        (now, now),
    )
    service = LocalResearchService(external)  # db object -> external mode

    updated = service.update_run_progress(
        "ext-run", phase="collecting", progress_percent=45.0,
        event="progress", data={"phase": "collecting"},
    )

    assert updated["phase"] == "collecting"
    assert updated["progress_percent"] == 45.0
    assert updated["version"] == 2
    external.close()


def test_lease_operations_in_external_db_mode_do_not_raise():
    """task-3 report finding 6: every lease operation called ``_connect()``
    unconditionally, which raises ``RuntimeError`` in external-db mode
    (``self.db is not None``, ``self.db_path is None``) -- so since
    ``execute_run`` now always claims a lease, external-db mode was broken
    outright. ``self.db`` has no lease columns and no lease API of its own
    (the ``FakeExternalResearchDB`` double above matches that: only
    ``transaction``/``get_run``/``close``), so a real, persisted,
    cross-process lease cannot be implemented against it -- the service
    degrades to an in-memory, per-instance lease instead of raising.
    """
    external = FakeExternalResearchDB()
    now = "2026-08-16T00:00:00Z"
    external.conn.execute(
        "INSERT INTO research_runs (id, query, created_at, updated_at) "
        "VALUES ('ext-run', 'External question', ?, ?)",
        (now, now),
    )
    service = LocalResearchService(external)  # db object -> external mode

    lease_id = service.claim_run("ext-run", worker_id="engine-a", lease_seconds=60)
    assert lease_id is not None
    assert service.holds_lease("ext-run", lease_id=lease_id) is True

    # A second claim while the first is live must decline, not raise --
    # "still functions single-executor" per the finding's fix direction.
    declined = service.claim_run("ext-run", worker_id="engine-b", lease_seconds=60)
    assert declined is None

    assert service.renew_lease("ext-run", lease_id=lease_id, lease_seconds=60) is True
    assert service.release_lease("ext-run", lease_id=lease_id) is True
    assert service.holds_lease("ext-run", lease_id=lease_id) is False

    # Released -- a new claim must now succeed.
    second_lease = service.claim_run("ext-run", worker_id="engine-b", lease_seconds=60)
    assert second_lease is not None
    external.close()
