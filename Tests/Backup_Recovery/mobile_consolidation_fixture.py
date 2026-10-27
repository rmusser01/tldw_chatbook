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
    legacy_runs_version=25,
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
        if linked_chat_seed is not None:
            linked_chat_seed(chat, legacy, result)
        return result
    finally:
        if legacy:
            # Produce the installed legacy layout before preview or capture.
            assert legacy_runs_version in (24, 25)
            for database, version in (
                (chat, 81),
                (workspace_db, 8),
                (runs, legacy_runs_version),
            ):
                with database.transaction() as cursor:
                    cursor.execute("DROP TABLE mobile_consolidation_batches")
                    if database is chat:
                        cursor.execute(
                            "UPDATE db_schema_version SET version=81 WHERE schema_name='rag_char_chat_schema'"
                        )
                    else:
                        cursor.execute("DROP TABLE mobile_import_authority_holds")
                        if database is runs and version == 24:
                            cursor.execute("DROP TABLE console_command_receipts")
                        cursor.execute(
                            "DELETE FROM schema_version WHERE version>?", (version,)
                        )
        runs.close()
        chat.close_connection()
        workspace_db.close()
        execution_owner.close()


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
