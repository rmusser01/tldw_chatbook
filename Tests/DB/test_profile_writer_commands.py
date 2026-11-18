"""Exercise the closed domain boundary using stock AgentRuns repositories."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB, AgentStepConflictError
from tldw_chatbook.DB import profile_writer_commands as commands


def test_domain_errors_round_trip_with_conflict_identity():
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat.console_context_policy import ContextPolicyError
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDispatchCheckpointValidationError,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_models import (
        MixinPreferenceConflictError,
        MixinSnapshotUnavailableError,
        MixinSourceChangedError,
    )
    from tldw_chatbook.Chat.library_preparation import LibraryPreparationValidationError
    from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError
    from tldw_chatbook.Notes.note_folder_models import NotesOrganizationRepositoryError

    errors = (
        AgentStepConflictError("step already recorded"),
        KeyError("unknown run"),
        sqlite3.IntegrityError("duplicate"),
        RecoveryRequired("maintenance_in_progress"),
        ConflictError("stale", entity="notes", entity_id=17),
        NotesOrganizationRepositoryError("folder_changed", "Folder moved"),
        ConsoleDispatchCheckpointValidationError("Invalid checkpoint"),
        ContextPolicyError("Invalid policy"),
        MixinPreferenceConflictError("Preference changed"),
        MixinSnapshotUnavailableError("Snapshot unavailable"),
        MixinSourceChangedError("Source changed"),
        LibraryPreparationValidationError("Invalid preparation event"),
    )
    for error in errors:
        decoded = commands.decode_profile_writer_error(
            commands.encode_profile_writer_error(error)
        )
        assert type(decoded) is type(error)
        assert decoded.args == error.args
        assert str(decoded) == str(error)
        assert vars(decoded) == vars(error)


def test_agent_commands_are_closed_before_open(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("an invalid command must be rejected before opening storage")

    monkeypatch.setattr(AgentRunsDB, "__init__", unexpected)
    for name, arguments in (
        ("agent_runs._initialize_schema", {}),
        ("agent_runs.transaction", {}),
        ("agent_runs.create_run", {"agent_kind": "primary", "db_path": "other"}),
        ("agent_runs.append_steps", {"run_id": "r", "steps": "SELECT 1"}),
    ):
        with pytest.raises((ValueError, TypeError)):
            commands.execute_profile_writer_command(tmp_path, name, arguments)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows executor presence")
def test_executor_presence_retires_on_process_exit(tmp_path):
    from tldw_chatbook.DB.profile_writer_run_presence import run_is_live

    program = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tldw_chatbook.DB.profile_writer_run_presence import claim_run
claim_run(Path(sys.argv[2]), "paused")
print("ready", flush=True)
sys.stdin.read(1)
os._exit(7)
"""
    assert not run_is_live(tmp_path, "paused")
    with subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            str(Path(__file__).resolve().parents[2]),
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as executor:
        try:
            ready = executor.stdout.readline().strip()
            assert ready == "ready", executor.stderr.read() if not ready else ready
            assert run_is_live(tmp_path, "paused")
        finally:
            executor.communicate(input="x", timeout=10)
        assert executor.returncode == 7
    assert not run_is_live(tmp_path, "paused")


def test_stock_writer_preserves_atomic_steps_and_settings(monkeypatch, tmp_path):
    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import legacy
    from tldw_chatbook import config
    from tldw_chatbook.DB import profile_writer

    profile = tmp_path
    path = profile / "agent_runs.db"
    monkeypatch.setattr(config, "get_chachanotes_db_path", lambda: profile / "notes.db")
    monkeypatch.setattr(config, "get_user_data_dir", lambda: profile)
    monkeypatch.setattr(profile_writer, "in_profile_writer", lambda: True)
    run = commands.execute_profile_writer_command
    legacy(path, version=18)
    run(profile, "agent_runs.open", {})
    db = AgentRunsDB(path, reconcile_on_init=False)
    try:
        assert db.get_run("retained-run")["result"] == "saved result"
        assert db.get_agent_definition("kept")["name"] == "Saved preset"
        run_id = db.create_run(conversation_id="chat", agent_kind="primary")
        definition = AgentDefinition(
            "reader", instructions="Read the file.", tool_allowlist=("read",)
        )
        definition_id = run(
            profile,
            "agent_runs.create_agent_definition",
            {"defn": commands.encode_agent_definition(definition)},
        )
        assert db.get_agent_definition(definition_id)["tool_allowlist"] == ["read"]
        run(
            profile,
            "agent_runs.insert_steps_at_indices",
            {"run_id": run_id, "steps": [[0, {"index": 0, "kind": "start"}]]},
        )
        with pytest.raises(AgentStepConflictError):
            run(
                profile,
                "agent_runs.set_terminal_with_step",
                {
                    "run_id": run_id,
                    "status": "done",
                    "result": "lost",
                    "terminal_step": {"index": 0, "kind": "finish"},
                },
            )
        assert db.get_run(run_id)["status"] == "running"
        assert run(
            profile,
            "agent_runs.set_terminal_with_step",
            {
                "run_id": run_id,
                "status": "done",
                "result": "ok",
                "terminal_step": {"index": 1, "kind": "finish"},
            },
        )
        assert db.get_run(run_id)["result"] == "ok"
        run(
            profile,
            "agent_runs.soft_delete_agent_definition",
            {"definition_id": definition_id},
        )
        assert db.get_agent_definition(definition_id) is None
    finally:
        db.close()


@pytest.mark.skipif(
    sys.platform != "win32", reason="native Windows writer and run ownership"
)
def test_two_profile_instances_can_write_during_a_live_run_and_writer_restart():
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.participants import core_operation
    from tldw_chatbook.DB.profile_writer import drain_profile_writer

    profile = config.get_user_data_dir()
    path = profile / "agent_runs.db"
    db = AgentRunsDB(path)
    run_id = None
    try:
        assert db._writer_profile == profile
        with core_operation(db):
            run_id = db.create_run(conversation_id="live-chat", agent_kind="primary")
            db.insert_steps_at_indices(run_id, [(0, {"index": 0, "kind": "start"})])
        drain_profile_writer(profile)
        # A freshly launched Settings instance opens its normal profile
        # repository and requests orphan repair while the Console is paused.
        program = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from Tests.windows_private_fixture_runner import user_fixture_default_owner
with user_fixture_default_owner():
    from tldw_chatbook import config
    from tldw_chatbook.Agents.agent_models import AgentDefinition
    from tldw_chatbook.Widgets.settings_agents_panel import _derive_runs_db
    selected = config.get_user_data_dir()
    assert selected == Path(sys.argv[2])
    database = _derive_runs_db(SimpleNamespace(chachanotes_db=SimpleNamespace(db_path=config.get_chachanotes_db_path())))
    try:
        assert database._writer_profile == selected
        database.reconcile_orphaned_runs()
        assert database.get_run(sys.argv[3])["status"] == "running"
        definition = AgentDefinition("check-" + sys.argv[3], instructions="Read the file.")
        chosen = database.create_agent_definition(definition)
        database.insert_steps_at_indices(sys.argv[3], [(1, {"index": 1, "kind": "other-instance"})])
        print(json.dumps(chosen), flush=True)
    finally:
        database.close()
"""
        peer = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                program,
                str(Path(__file__).resolve().parents[2]),
                str(profile),
                run_id,
            ],
            capture_output=True,
            text=True,
            timeout=110,
        )
        assert peer.returncode == 0, peer.stderr[-4000:]
        definition_id = json.loads(peer.stdout.splitlines()[-1])
        assert (
            db.get_agent_definition(definition_id)["instructions"] == "Read the file."
        )
        assert [step["index"] for step in db.get_run(run_id)["steps"]] == [0, 1]
        assert db.set_terminal_with_step(
            run_id, "done", "saved", {"index": 2, "kind": "finish"}
        )
        assert db.get_run(run_id)["result"] == "saved"
        db.soft_delete_agent_definition(definition_id)
    finally:
        try:
            if run_id is not None:
                db.set_status(run_id, "cancelled")
            db.close()
        finally:
            drain_profile_writer(profile)
