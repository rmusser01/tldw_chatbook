"""Real, closed native stores shared by legacy consolidation qualification."""

import hashlib
from pathlib import Path

from Tests.Agents.test_goal_models import request
from Tests.DB.test_automatic_wake_attempts import claim, survivor
from tldw_chatbook.Backup_Recovery.mobile_consolidation import CanonicalConsoleStores
from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
from tldw_chatbook.Chat.console_trace_models import TraceContentRef
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def fingerprints(root: Path) -> dict[str, str]:
    """Fingerprint closed isolated fixture files, never a live user database."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.is_file()
    }


def _seed(
    stores: CanonicalConsoleStores,
    *,
    legacy: bool,
    artifact_root: Path | None = None,
    chat_seed=None,
    linked_chat_seed=None,
    runs_seed=None,
    legacy_runs_version=25,
    legacy_chat_version=81,
    legacy_workspace_version=8,
) -> dict:
    for path in stores.paths:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    workspace_db = WorkspaceDB(stores.workspace_path, "mobile-fixture")
    chat = CharactersRAGDB(stores.chat_path, "mobile-fixture")
    execution_owner = ConsoleExecutionOwner.acquire(stores.run_path)
    runs = AgentRunsDB(
        stores.run_path, client_id="mobile-fixture", execution_owner=execution_owner
    )
    try:
        if artifact_root is None:
            artifact_root = (
                stores.run_path.parent.parent / "tool_sandbox"
                if legacy
                else stores.run_path.parent.parent / "canonical-artifacts"
            )
        registry = LocalWorkspaceRegistryService(workspace_db)
        default = registry.ensure_default_workspace()
        workspace = registry.create_workspace(
            workspace_id="conflicting-workspace", name="Shared title"
        )
        # Public record API preserves exclusion metadata without validating live roots.
        from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

        binding = WorkspaceRuntimeBinding(
            binding_id="binding",
            workspace_id=workspace.workspace_id,
            binding_kind="local-filesystem",
            label="Fixture",
            locator=str(artifact_root),
            status="ready",
            metadata={
                "allow_write": True,
                "exclusions": [
                    {"path": "private", "kind": "directory", "added_at": "2026-10-25"}
                ],
            },
        )
        registry.save_runtime_binding(binding)
        conversation = chat.add_conversation(
            {
                "id": "conversation",
                "title": "Shared title",
                "workspace_id": workspace.workspace_id,
            }
        )
        user = chat.add_message(
            {
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "private fixture prompt",
            }
        )
        first = chat.add_message(
            {
                "conversation_id": conversation,
                "sender": "Assistant",
                "role": "assistant",
                "content": "private first branch",
                "parent_message_id": user,
            }
        )
        second = chat.add_message(
            {
                "conversation_id": conversation,
                "sender": "Assistant",
                "role": "assistant",
                "content": "private second branch",
                "parent_message_id": user,
            }
        )
        repo = ConsoleTraceRepository()
        with chat.transaction(immediate=True) as cursor:
            segment = repo.create_segment(cursor)
            owner = repo.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=segment.segment_id
            )
            artifact = repo.store_sanitized_artifact(
                cursor,
                sanitized_bytes=b'"private trace artifact"',
                media_type="application/json",
                normalization_version="test-v1",
            )
            repo.append_surface_node(
                cursor,
                segment_id=segment.segment_id,
                sequence=0,
                predecessor_node_id=None,
                component_kind="native_message",
                reference=TraceContentRef(artifact.artifact_id, "native_message"),
            )
        chain = runs.automatic_work.create_chain(
            conversation, root_submission_id="settled-root"
        )
        runs.automatic_work.reserve(
            chain,
            reservation_id="settled-tokens",
            owner_id="owner",
            kind="tokens",
            amount=20,
        )
        runs.automatic_work.commit("settled-tokens", owner_id="owner")
        runs.automatic_work.settle("settled-tokens", owner_id="owner", actual_amount=12)
        from tldw_chatbook.Agents.agent_models import AgentDefinition

        definition_id = runs.create_agent_definition(
            AgentDefinition(name="fixture-agent", instructions="private definition")
        )
        definition_run = runs.create_run(
            conversation_id=conversation,
            agent_kind="subagent",
            agent_definition="fixture-agent",
        )
        runs.set_status(definition_run, "done", result="retained definition result")
        child = survivor(runs, chain, conversation=conversation)
        runs.append_steps(child, [{"type": "fixture", "content": "retained step"}])
        parent = runs.get_run(child)["parent_run_id"]
        artifact_file = artifact_root / ".agent-runs" / child / "events.jsonl"
        artifact_file.parent.mkdir(parents=True, mode=0o700)
        artifact_file.write_text('{"event":"private retained artifact"}\n')
        artifact_file.chmod(0o600)
        runs.record_change_snapshot(
            run_id=child,
            root=str(artifact_root),
            baseline_sha="baseline",
            end_sha="end",
        )
        if legacy:
            pending_chain = runs.automatic_work.create_chain(
                conversation, root_submission_id="pending-root"
            )
            pending_child = survivor(runs, pending_chain, conversation=conversation)
            claim(runs, pending_chain, [pending_child])
        else:
            pending_chain = pending_child = None
        goal_request = request(root=str(artifact_root))
        goal_request = goal_request.model_copy(
            update={
                "binding": goal_request.binding.model_copy(
                    update={"workspace_id": workspace.workspace_id}
                )
            }
        )
        goal = runs.goal_runs.create(goal_request, launch_id="fixture-goal")
        chat.add_conversation(
            {
                "id": goal.conversation_id,
                "title": "Fixture Goal",
                "workspace_id": workspace.workspace_id,
            }
        )
        if chat_seed is not None:
            chat_seed(chat, legacy)
        result = {
            "conversation_id": conversation,
            "message_ids": (user, first, second),
            "branches": ((first, user), (second, user)),
            "default_workspace_id": default.workspace_id,
            "workspace_id": workspace.workspace_id,
            "binding_id": "binding",
            "excluded_paths": ("private",),
            "definition_id": definition_id,
            "definition_run_id": definition_run,
            "parent_run_id": parent,
            "child_run_id": child,
            "settled_chain_id": chain,
            "pending_chain_id": pending_chain,
            "pending_child_id": pending_child,
            "goal_id": goal.id,
            "artifact_file": artifact_file,
            "trace_owner_id": owner.owner_id,
            "artifact_id": artifact.artifact_id,
        }
        if runs_seed is not None:
            runs_seed(runs, legacy, result)
        if linked_chat_seed is not None:
            linked_chat_seed(chat, legacy, result)
        return result
    finally:
        runs.close()
        chat.close_connection()
        workspace_db.close()
        execution_owner.close()
        if legacy:
            _reconstruct_legacy_stores(
                stores, legacy_chat_version=legacy_chat_version,
                legacy_workspace_version=legacy_workspace_version,
                legacy_runs_version=legacy_runs_version,
            )


def _reconstruct_legacy_stores(
    stores, *, legacy_chat_version=81, legacy_workspace_version=8, legacy_runs_version=25,
):
    """Build exact installed predecessors only after fixture owners close."""
    from tldw_chatbook.DB import recovery_core_schema
    from tldw_chatbook.DB.recovery_operations import (
        _AGENT_RUNS_SCHEMA, _WORKSPACES_SCHEMA,
    )

    assert legacy_runs_version in (24, 25)
    assert legacy_chat_version in (81, 82, 83, 84, 85)
    assert legacy_workspace_version in (8, 9, 10)
    for path, catalog, version in (
        (stores.chat_path, getattr(recovery_core_schema, f"CHACHANOTES_V{legacy_chat_version}_SCHEMAS")[0], legacy_chat_version),
        (stores.workspace_path, next(c for v, c in _WORKSPACES_SCHEMA if v == legacy_workspace_version), legacy_workspace_version),
        (stores.run_path, next(c for v, c in _AGENT_RUNS_SCHEMA if v == legacy_runs_version), legacy_runs_version),
    ):
        _reconstruct_predecessor(path, catalog, version)

def _reconstruct_predecessor(path, catalog, version):
    """Build an exact installed fixture catalog, preserving compatible seed rows."""
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.DB.ChaChaNotes_DB import _install_canvas_revision_payload_validator

    target = path.with_suffix(".predecessor")
    quote = lambda name: '"' + name.replace('"', '""') + '"'
    with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(target)) as dest:
        _install_canvas_revision_payload_validator(dest)
        # Virtual tables create their shadow catalog themselves. Ordinary tables
        # must precede inserts; native triggers are installed only after copying
        # seed rows so source sync logs are not generated a second time.
        virtual = [sql for sql in catalog if sql.startswith("CREATE VIRTUAL TABLE")]
        for sql in virtual:
            dest.execute(sql)
        created = {row[0] for row in dest.execute("SELECT name FROM sqlite_schema")}
        for sql in catalog:
            if not sql.startswith("CREATE TABLE"):
                continue
            name = sql.split("(", 1)[0].split()[-1].strip("\"'`[]")
            if name not in created and name != "sqlite_sequence":
                dest.execute(sql)
        for _, name, kind, *_ in dest.execute("PRAGMA table_list").fetchall():
            if name.startswith("sqlite_") or kind not in ("table", "virtual"):
                continue
            columns = tuple(row[1] for row in dest.execute("PRAGMA table_info(" + quote(name) + ")"))
            names = ",".join(map(quote, columns))
            rows = source.execute("SELECT " + names + " FROM " + quote(name)).fetchall()
            dest.executemany("INSERT INTO " + quote(name) + " (" + names + ") VALUES (" + ",".join("?" for _ in columns) + ")", rows)
        if dest.execute("SELECT 1 FROM sqlite_schema WHERE name='sqlite_sequence'").fetchone():
            dest.execute("DELETE FROM sqlite_sequence")
            dest.executemany("INSERT INTO sqlite_sequence VALUES (?,?)", source.execute("SELECT name,seq FROM sqlite_sequence").fetchall())
        for sql in catalog:
            if not sql.startswith(("CREATE TABLE", "CREATE VIRTUAL TABLE")):
                dest.execute(sql)
        if dest.execute("SELECT 1 FROM sqlite_schema WHERE name='db_schema_version'").fetchone():
            dest.execute("UPDATE db_schema_version SET version=? WHERE schema_name='rag_char_chat_schema'", (version,))
        else:
            dest.execute("DELETE FROM schema_version WHERE version>?", (version,))
        dest.commit()
    target.chmod(0o600)
    target.replace(path)


def seed_legacy_mobile(source_dir: Path, *, artifact_root: Path | None = None) -> dict:
    """Seed closed branched/history/authority fixtures under one isolated source."""
    result = _seed(
        CanonicalConsoleStores(
            source_dir / "chat.db", source_dir / "workspaces.db", source_dir / "runs.db"
        ),
        legacy=True,
        artifact_root=artifact_root,
    )
    result["fingerprints"] = fingerprints(source_dir)
    return result


def seed_canonical_console(destination: CanonicalConsoleStores) -> dict:
    """Seed a conflicting destination without inferring equivalence from titles."""
    return _seed(destination, legacy=False)
