"""Real populated native owner commits, never caller-provided plan authority."""

import json
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_native_operation import captured_operation


def _small_native_seed(stores, *, legacy, artifact_root=None, source_case=None):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        SemanticRevisionRef,
        TraceCallState,
        TraceContentRef,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    for path in stores.paths:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    chat = CharactersRAGDB(stores.chat_path, "native-writer-test")
    workspace = WorkspaceDB(stores.workspace_path, "native-writer-test")
    lease = ConsoleExecutionOwner.acquire(stores.run_path)
    runs = AgentRunsDB(stores.run_path, "native-writer-test", execution_owner=lease)
    try:
        registry = LocalWorkspaceRegistryService(workspace)
        registry.ensure_default_workspace()
        named = registry.create_workspace(
            workspace_id="same-workspace",
            name="Imported project" if legacy else "Current project",
            assistant_defaults=None,
        )
        if legacy:
            assert chat.update_character_card(
                1,
                {"name": "Imported assistant"},
                expected_version=chat.get_character_card_by_id(1)["version"],
            )
        with chat.transaction() as cursor:
            cursor.execute(
                "INSERT INTO chat_dictionaries(name,content,entries_json) VALUES (?,?,?)",
                (
                    "Imported words" if legacy else "Current words",
                    "Exact dictionary body",
                    "[]",
                ),
            )
            dictionary_id = cursor.lastrowid
        conversation = chat.add_conversation(
            {
                "id": "shared-chat",
                "title": "Retained title",
                "metadata": json.dumps({"active_dictionaries": [dictionary_id]}),
            }
        )
        registry.link_membership(
            named.workspace_id, item_type="conversation", item_id=conversation
        )
        message = chat.add_message(
            {
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "Exact retained body Ω",
            }
        )
        turn = chat.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant" if source_case == "passive" else "AI",
                "role": "assistant",
                "content": "Exact terminal result",
            }
        )
        run = runs.create_run(
            conversation_id=conversation,
            agent_kind="primary",
            assistant_message_id=turn,
        )
        runs.set_status(run, "done", result="Exact terminal result")
        trace = ConsoleTraceRepository()
        with chat.transaction(immediate=True) as cursor:
            segment = trace.create_segment(cursor)
            owner = trace.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=segment.segment_id
            )
            artifact = trace.store_sanitized_artifact(
                cursor,
                sanitized_bytes=b'"same bytes"',
                media_type="application/json",
                normalization_version="test-v1",
            )
            trace.append_surface_node(
                cursor,
                segment_id=segment.segment_id,
                sequence=0,
                predecessor_node_id=None,
                component_kind="native_message",
                reference=TraceContentRef(artifact.artifact_id, "native_message"),
            )
            policy = trace.ensure_policy(
                cursor, FrozenTracePolicy(new_opaque_id(), "test-v1", False, None)
            )
            call = trace.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id=turn,
                run_id=run,
                call_sequence=0,
                idempotency_key="legacy-call" if legacy else "canonical-call",
                policy_id=policy.policy_id,
            )
            trace.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=TraceCallState.NOT_DISPATCHED,
                occurred_at=cursor.execute(
                    "SELECT created_at FROM console_trace_calls WHERE call_id=?",
                    (call.call_id,),
                ).fetchone()[0],
            )
        result = {
            "conversation": conversation,
            "message": message,
            "run": run,
            "artifact": artifact.artifact_id,
            "owner": owner.owner_id,
        }
        if source_case == "consumers":
            from tldw_chatbook.Agents.agent_models import AgentDefinition

            result["definition"] = runs.create_agent_definition(
                AgentDefinition(
                    name="imported-definition" if legacy else "current-definition",
                    instructions="Retained instructions",
                )
            )
            if legacy:
                from Tests.Agents.test_goal_models import request
                from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

                root = stores.run_path.parent.parent / "tool_sandbox"
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
                registry.save_runtime_binding(
                    WorkspaceRuntimeBinding(
                        binding_id="binding",
                        workspace_id=named.workspace_id,
                        binding_kind="local-filesystem",
                        label="Goal project",
                        locator=str(root),
                        status="ready",
                        metadata={"allow_write": True},
                    )
                )
                launch = request(root=str(root))
                launch = launch.model_copy(
                    update={
                        "binding": launch.binding.model_copy(
                            update={"workspace_id": named.workspace_id}
                        )
                    }
                )
                goal = runs.goal_runs.create(launch, launch_id="native-imported-goal")
                goal = runs.goal_runs.control(goal.id)
                chat.add_conversation(
                    {
                        "id": goal.conversation_id,
                        "title": "Retained goal",
                        "workspace_id": named.workspace_id,
                    }
                )
                result["goal"] = goal.id
                result["chain"] = goal.chain_id
                result["note"] = runs.add_change_note(
                    run_id=run,
                    root=str(root),
                    path="file.py",
                    hunk_index=0,
                    hunk_header="@@ -1 +1 @@",
                    hunk_excerpt="Retained self-contained excerpt",
                    note="Pending historical annotation",
                    snapshot_id=42,
                )
        if legacy and source_case == "passive":
            from Tests.Chat.test_console_dispatch_recovery import _acceptance, _insert
            from tldw_chatbook.Agents.fleet_messages import (
                MessageIdentity,
                MessageStore,
            )
            from tldw_chatbook.Chat.console_dispatch_repository import (
                ConsoleDispatchRepository,
            )
            from tldw_chatbook.DB.fleet_progress_repository import (
                FleetProgressRepository,
            )

            with chat.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO console_conversation_library_policy (conversation_id,auto_retrieve_on_send,assistant_library_access,policy_revision,updated_at) VALUES (?,0,0,1,CURRENT_TIMESTAMP)",
                    (conversation,),
                )
            from dataclasses import replace

            from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
            from tldw_chatbook.Chat.response_rules.corrections import (
                MachineFollowupReceipt,
            )

            acceptance = replace(
                _acceptance(
                    conversation, origin="queued", queue_entry_id="historical-queue"
                ),
                parent_message_id=turn,
                continuation_receipt=ContinuationReceipt(
                    "historical-turn", "historical-stop", turn, "historical-chain", 1
                ),
                machine_followup_receipt=MachineFollowupReceipt(
                    "historical-operation",
                    "historical-turn",
                    "historical-settlement",
                    turn,
                    "historical-chain",
                    1,
                    1,
                    ("native", "hook"),
                    chat.get_message_by_id(turn)["version"],
                ),
            )
            checkpoint = _insert(chat, ConsoleDispatchRepository(chat), acceptance)
            child = runs.create_run(
                conversation_id=conversation, agent_kind="subagent", parent_run_id=run
            )
            runs.set_status(child, "done", result="Historical fleet report")
            messenger = MessageStore()
            repository = FleetProgressRepository(chat)
            inbox = messenger.open_inbox(
                "historical-console",
                repository=repository,
                saved_conversation_id=conversation,
                messages=repository.load(conversation),
            )
            try:
                fleet_id = inbox.sender(
                    MessageIdentity("historical-handle", child, run, None, "reader")
                ).send("Historical fleet report")
            finally:
                messenger.close()
            result.update(dispatch=checkpoint.assistant_message_id, fleet=fleet_id)
        if legacy and source_case in {"physical_snapshot", "physical_worktree"}:
            from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

            # External native project: arbitrary project/Git files are not the
            # installed sandbox's owned run-log/script-output children.
            root = stores.run_path.parents[3] / "external-native-project"
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            (root / "retained.txt").write_text("original native project bytes")
            if source_case == "physical_snapshot":
                from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

                service = ShadowRepoService(
                    data_dir=stores.run_path.parents[3] / "external-shadow"
                )
                native_repo = service.repo_for_root(root)
                baseline = native_repo.snapshot("baseline")
                (root / "retained.txt").write_text("changed native project bytes")
                end = native_repo.snapshot("end")
                assert native_repo.has_snapshot(baseline) and native_repo.has_snapshot(
                    end
                )
                runs.record_change_snapshot(
                    run_id=run,
                    root=str(root),
                    baseline_sha=baseline,
                    end_sha=end,
                    files_changed=1,
                )
                result["physical_root"] = root
            else:
                from Tests.Agents.test_agent_worktree_confirmed_recovery import git
                from tldw_chatbook.Agents import agent_worktree
                from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository
                from tldw_chatbook.Workspaces.file_inspector import _binding_fingerprint

                git(root, "init", "-b", "main")
                git(root, "config", "user.name", "Native fixture")
                git(root, "config", "user.email", "native@example.invalid")
                git(root, "add", "retained.txt")
                git(root, "commit", "-m", "native fixture")
                child_run = runs.create_run(
                    conversation_id=conversation,
                    agent_kind="subagent",
                    parent_run_id=run,
                )
                with pytest.MonkeyPatch.context() as isolated:
                    isolated.setattr(
                        agent_worktree,
                        "_worktrees_base",
                        lambda: stores.run_path.parents[3] / "external-native-children",
                    )
                    child = agent_worktree.create_agent_worktree(root, child_run)
                assert isinstance(child, agent_worktree.AgentWorktree)
                binding = WorkspaceRuntimeBinding(
                    binding_id="worktree-binding",
                    workspace_id=named.workspace_id,
                    binding_kind="local-filesystem",
                    label="Native project",
                    locator=str(root),
                    status="ready",
                    metadata={"allow_write": True},
                )
                registry.save_runtime_binding(binding)
                common, common_identity = agent_worktree._git_common_directory_identity(
                    root
                )
                repository = AgentWorktreeRepository(runs)
                repository.record_created(
                    run_id=child_run,
                    workspace_id=named.workspace_id,
                    binding_id=binding.binding_id,
                    locator_fingerprint=_binding_fingerprint(binding),
                    repo_root=str(root),
                    repo_identity=agent_worktree._worktree_root_identity(root),
                    git_common_dir=str(common),
                    git_common_identity=common_identity,
                    child_path=str(child.worktree_path),
                    child_identity=agent_worktree._worktree_root_identity(
                        child.worktree_path
                    ),
                    branch=child.branch,
                    base_sha=child.base_sha,
                    execution_id="native-execution",
                )
                repository.mark_writer_finished(
                    child_run, "native-execution", cleanup_proven=True
                )
                runs.set_status(child_run, "done", result="Native child retained")
                result["physical_root"] = child.worktree_path
        if legacy and source_case == "failed_snapshot":
            root = stores.run_path.parent.parent / "tool_sandbox"
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            runs.record_change_snapshot(
                run_id=run,
                root=str(root),
                baseline_sha="a" * 40,
                end_sha="b" * 40,
                files_changed=2,
                adds=3,
                dels=1,
                tracking_error="Native tracking failed after initial snapshot",
                untracked_oversize=1,
                nested_repos=("nested",),
                kind="turn",
            )
            with runs.connection() as connection:
                result["snapshot"] = dict(
                    connection.execute(
                        "SELECT * FROM change_snapshots WHERE run_id=?", (run,)
                    ).fetchone()
                )
        if legacy and source_case == "history":
            from tldw_chatbook.Agents.run_log import RunLogWriter

            root = stores.run_path.parent.parent / "tool_sandbox"
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            writer = RunLogWriter(
                root=root,
                dir_name=".agent-runs",
                segment_bytes=65536,
                max_record_bytes=32768,
            )
            writer.bind(run)
            try:
                assert (
                    writer.append(
                        run_id=run,
                        kind="primary",
                        type="model",
                        content="Original native frame Ω",
                    )
                    is not None
                )
            finally:
                writer.close()
            result["log_dir"] = writer.log_dir
        if legacy and source_case in {
            "detached",
            "detached_original",
            "detached_current",
            "detached_twice",
            "empty_detached",
        }:
            deleted_conversation = chat.add_conversation(
                {"title": "Native detached trace"}
            )
            with chat.transaction() as cursor:
                detached_segment = trace.create_segment(cursor)
                detached_owner = trace.attach_owner(
                    cursor,
                    conversation_id=deleted_conversation,
                    root_segment_id=detached_segment.segment_id,
                )
                if source_case != "empty_detached":
                    trace.append_surface_node(
                        cursor,
                        segment_id=detached_segment.segment_id,
                        sequence=0,
                        predecessor_node_id=None,
                        component_kind="native_message",
                        reference=TraceContentRef(
                            artifact.artifact_id, "native_message"
                        ),
                    )
                if source_case not in {"detached", "empty_detached"}:
                    revision = trace.ensure_semantic_revision(
                        cursor,
                        source_conversation_id=deleted_conversation,
                        source_message_id="historical-detached-message",
                        revision_sequence=0,
                        normalized_role="user",
                        content_kind="text",
                        creation_reason="message_create",
                        live_locator_retired_at="2026-10-27T00:00:00Z",
                    )
                    trace.append_surface_node(
                        cursor,
                        segment_id=detached_segment.segment_id,
                        sequence=1,
                        predecessor_node_id=cursor.execute(
                            "SELECT node_id FROM console_trace_surface_nodes WHERE segment_id=?",
                            (detached_segment.segment_id,),
                        ).fetchone()[0],
                        component_kind="native_message",
                        reference=SemanticRevisionRef(revision.revision_id),
                    )
                trace.detach_owner(
                    cursor,
                    owner_id=detached_owner.owner_id,
                    detached_at="2026-10-27T01:00:00Z",
                )
                if source_case == "detached":
                    cursor.execute(
                        "DELETE FROM conversations WHERE id=?", (deleted_conversation,)
                    )
                elif source_case in {"detached_current", "detached_twice"}:
                    current_segment = trace.create_segment(cursor)
                    current_owner = trace.attach_owner(
                        cursor,
                        conversation_id=deleted_conversation,
                        root_segment_id=current_segment.segment_id,
                    )
                    trace.append_surface_node(
                        cursor,
                        segment_id=current_segment.segment_id,
                        sequence=0,
                        predecessor_node_id=None,
                        component_kind="native_message",
                        reference=SemanticRevisionRef(revision.revision_id),
                    )
                    if source_case == "detached_twice":
                        trace.detach_owner(
                            cursor,
                            owner_id=current_owner.owner_id,
                            detached_at="2026-10-27T02:00:00Z",
                        )
                        final_segment = trace.create_segment(cursor)
                        trace.attach_owner(
                            cursor,
                            conversation_id=deleted_conversation,
                            root_segment_id=final_segment.segment_id,
                        )
                        trace.append_surface_node(
                            cursor,
                            segment_id=final_segment.segment_id,
                            sequence=0,
                            predecessor_node_id=None,
                            component_kind="native_message",
                            reference=SemanticRevisionRef(revision.revision_id),
                        )
            result["blocked_identity"] = detached_owner.owner_id
        if legacy and source_case == "promoted":
            from dataclasses import replace

            from Tests.Chat.test_console_voice_trace_repository import (
                _completed_pair,
                _post_dispatch_import,
                _seal_post_dispatch_import,
            )

            voice_conversation, voice_user, voice_assistant = _completed_pair(chat)
            voice_run = runs.create_run(
                conversation_id=voice_conversation,
                agent_kind="primary",
                assistant_message_id=voice_assistant,
            )
            runs.set_status(
                voice_run, "done", result="Exact native promoted voice result"
            )
            request = _post_dispatch_import(
                chat,
                conversation_id=voice_conversation,
                user_message_id=voice_user,
                assistant_message_id=voice_assistant,
                call_count=1,
            )
            request = _seal_post_dispatch_import(replace(request, run_id=voice_run))
            trace.import_post_dispatch_trace(chat, request)
            result["blocked_identity"] = request.calls[0].call_id
        return result
    finally:
        if legacy:
            for db in (chat, workspace, runs):
                with db.transaction() as cursor:
                    cursor.execute("DROP TABLE mobile_consolidation_batches")
                    if db is chat:
                        cursor.execute(
                            "UPDATE db_schema_version SET version=81 WHERE schema_name='rag_char_chat_schema'"
                        )
                    else:
                        cursor.execute("DROP TABLE mobile_import_authority_holds")
                        cursor.execute(
                            "DELETE FROM schema_version WHERE version>?",
                            (8 if db is workspace else 25,),
                        )
        runs.close()
        workspace.close()
        chat.close_connection()
        lease.close()


def test_populated_same_operation_commits_three_real_owner_receipts(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture

    seeded = {}

    def seed(stores, **kwargs):
        result = _small_native_seed(stores, **kwargs)
        seeded["legacy" if kwargs["legacy"] else "canonical"] = result
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        captured.consolidate(password=b"test-private-mobile-rollback", cancel=Event())
        for owner_id, batch in (
            ("db.chachanotes.primary", "chat"),
            ("db.workspaces", "workspaces"),
            ("db.agent_runs", "runs"),
        ):
            with captured.owner(owner_id) as owner, owner.transaction() as connection:
                receipts = connection.execute(
                    "SELECT source_fingerprint,batch_key FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                    (captured.journal.operation_id, batch),
                ).fetchall()
                assert [tuple(row) for row in receipts] == [
                    (captured.preview.source_fingerprint, batch)
                ]
                if batch == "chat":
                    assert (
                        connection.execute(
                            "SELECT count(*) FROM conversations WHERE title='Retained title'"
                        ).fetchone()[0]
                        == 2
                    )
                    artifacts = connection.execute(
                        "SELECT artifact_id FROM console_trace_artifacts WHERE sanitized_bytes=?",
                        (b'"same bytes"',),
                    ).fetchall()
                    assert len(artifacts) == 2
                if batch == "runs":
                    rows = connection.execute(
                        "SELECT id,status,result,wake_delivered_at FROM agent_runs ORDER BY id"
                    ).fetchall()
                    assert len(rows) == 2
                    assert all(
                        tuple(row)[1:] == ("done", "Exact terminal result", None)
                        for row in rows
                    )
                    holds = connection.execute(
                        "SELECT authority_kind,authority_id FROM mobile_import_authority_holds"
                    ).fetchall()
                    assert len(holds) == 1 and holds[0][0] == "run"
                    assert holds[0][1] != seeded["canonical"]["run"]


def test_fixed_schedule_preserves_native_owner_before_trace_nodes(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery.mobile_native_rows import _prepare_rows
    from tldw_chatbook.Backup_Recovery.mobile_runs_preflight import _plan_runs
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with captured_operation(tmp_path, monkeypatch) as captured:  # noqa: SIM117 -- Keep the explicit native resource/fault lifetime boundary.
        with captured.sqlite_reads(
            password=b"test-private-mobile-rollback", cancel=Event()
        ) as reads:
            budget = _TraceReadBudget()
            plan = _plan_runs(reads, _read_budget=budget)
            assert not plan.blockers
            ordered = _prepare_rows(plan, budget)
            names = [row.table for row, values, _detach in ordered]
            assert len(ordered) == len(plan.rows)
            assert names.index("conversations") < names.index("messages")
            assert names.index("console_trace_segments") < names.index(
                "console_trace_owners"
            )
            assert names.index("console_trace_owners") < names.index(
                "console_trace_surface_nodes"
            )
            assert names.index("console_trace_artifacts") < names.index(
                "console_trace_surface_nodes"
            )
            assert all(
                tuple(values.items()) == row.values for row, values, _detach in ordered
            )


def test_fixed_rows_preserve_native_reserved_artifact_ids_and_clocks(
    tmp_path, monkeypatch
):
    import pytest

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery.mobile_native_rows import (
        _insert_row,
        _prepare_rows,
    )
    from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES
    from tldw_chatbook.Backup_Recovery.mobile_runs_preflight import _plan_runs
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    class RollbackProbe(Exception):
        pass

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        with captured.sqlite_reads(
            password=b"test-private-mobile-rollback", cancel=Event()
        ) as reads:
            budget = _TraceReadBudget()
            plan = _plan_runs(reads, _read_budget=budget)
            assert not plan.blockers
            ordered = _prepare_rows(plan, budget)
            from tldw_chatbook.Backup_Recovery.mobile_native_receipts import (
                _encode_receipts,
                _receipt_children,
            )
            from tldw_chatbook.DB.console_command_history import (
                _prepare_command_history,
            )

            commands = tuple(
                row
                for row in plan.operational_history
                if row.table == "console_command_receipts"
            )
            passive = _prepare_command_history(
                captured.journal.operation_id,
                captured.preview.source_fingerprint,
                commands,
                budget,
            )
            encoded = _encode_receipts(
                plan,
                reads.versions,
                _receipt_children(plan, budget),
                len(commands),
                budget,
            )
            assert set(encoded) == {"chat", "workspaces", "runs"}
            assert len(passive) == 1
            assert not any(
                "Exact retained body" in raw
                for entries in encoded.values()
                for _, raw in entries
            )
        specs = {spec.table: spec for spec in _TABLES}
        for family, owner_id in (
            ("chat", "db.chachanotes.primary"),
            ("workspaces", "db.workspaces"),
            ("runs", "db.agent_runs"),
        ):
            with captured.owner(owner_id) as owner:
                with pytest.raises(RollbackProbe), owner.transaction() as connection:
                    for row, values, _detach in ordered:
                        spec = specs[row.table]
                        if spec.family != family:
                            continue
                        _insert_row(connection, row, values)
                        predicate = " AND ".join(
                            f"{column}=?" for column in spec.identity_columns
                        )
                        projection = ",".join(
                            f"CASE WHEN 1 THEN {column} END" for column in spec.columns
                        )
                        saved = connection.execute(
                            f"SELECT {projection} FROM {spec.table} WHERE {predicate}",
                            tuple(values[column] for column in spec.identity_columns),
                        ).fetchone()
                        assert tuple(saved) == tuple(
                            values[column] for column in spec.columns
                        ), row.table
                    if family == "chat":
                        assert (
                            connection.execute(
                                "SELECT count(DISTINCT artifact_id) FROM console_trace_artifacts WHERE sanitized_bytes=?",
                                (b'"same bytes"',),
                            ).fetchone()[0]
                            == 2
                        )
                    raise RollbackProbe
                with owner.transaction() as connection:
                    assert (
                        connection.execute(
                            "SELECT count(*) FROM mobile_consolidation_batches"
                        ).fetchone()[0]
                        == 0
                    )


def test_snapshot_frontier_requires_owner_transaction_and_rolls_back(tmp_path):
    import pytest

    from Tests.agent_runs_owner import owned_agent_runs_db
    from tldw_chatbook.Backup_Recovery.mobile_native_rows import (
        _advance_snapshot_allocator,
    )
    from tldw_chatbook.Backup_Recovery.mobile_runs_snapshots import _SnapshotAllocator

    db = owned_agent_runs_db(tmp_path / "allocator.db")
    try:
        allocator = _SnapshotAllocator(None, None, 41)
        with (
            db.connection() as connection,
            pytest.raises(ValueError, match="owner_transaction"),
        ):
            _advance_snapshot_allocator(connection, allocator)
        with (
            pytest.raises(RuntimeError, match="rollback"),
            db.transaction() as connection,
        ):
            _advance_snapshot_allocator(connection, allocator)
            raise RuntimeError("rollback")
        with db.connection() as connection:
            assert (
                connection.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='change_snapshots'"
                ).fetchone()
                is None
            )
        with db.transaction() as connection:
            _advance_snapshot_allocator(connection, allocator)
        db.record_change_snapshot(
            run_id="native-run",
            root="/historical/root",
            baseline_sha="",
            end_sha="",
            tracking_error="retained native error",
        )
        assert db.change_snapshots_for_run("native-run")[0]["id"] == 42
    finally:
        db.close()


@pytest.mark.parametrize(
    "source_case,reason",
    [
        ("detached", "native_detached_trace_original_owner_unproved"),
        ("promoted", "native_promoted_trace_current_voice_authority_required"),
    ],
)
def test_native_unrepresentable_trace_refuses_before_first_effect(
    tmp_path, monkeypatch, source_case, reason
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
    from tldw_chatbook.Backup_Recovery.mobile_native_operation import (
        _NativeOperationBlocked,
    )

    seeded = {}

    def seed(stores, **kwargs):
        result = _small_native_seed(stores, **kwargs, source_case=source_case)
        if kwargs["legacy"]:
            seeded.update(result)
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            records = tuple(captured.journal._records(fd))
        with pytest.raises(_NativeOperationBlocked) as error:
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
        assert any(
            block.reason == reason and block.original == (seeded["blocked_identity"],)
            for block in error.value.blockers
        )
        assert _fingerprints(tmp_path / "profile") == before
        with captured.journal._locked(exclusive=False) as fd:
            assert tuple(captured.journal._records(fd)) == records


def test_real_chat_commit_resumes_receipt_first_after_next_owner_rollback(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery import mobile_native_rows
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _resume_captured_mobile,
    )

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    destination = tmp_path / "archives"
    destination.mkdir(mode=0o700)
    insert = mobile_native_rows._insert_row

    def fail_workspace(connection, row, values, detach=False):
        if row.table == "workspace_records":
            raise OSError("native workspace transaction interrupted")
        return insert(connection, row, values, detach=detach)

    with _capture_fixture(tmp_path, monkeypatch) as (  # noqa: SIM117 -- Native resource lifetime boundary.
        session,
        journal,
        _config,
        _legacy,
        preview,
        inventory,
        authority,
    ):
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            with monkeypatch.context() as fault:
                fault.setattr(mobile_native_rows, "_insert_row", fail_workspace)
                with pytest.raises(OSError, match="workspace transaction interrupted"):
                    captured.consolidate(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    )
            with pytest.raises(ValueError, match="already_attempted"):
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
            with (
                captured.owner("db.chachanotes.primary") as chat,
                chat.transaction() as conn,
            ):
                before = tuple(
                    tuple(row)
                    for row in conn.execute(
                        "SELECT * FROM mobile_consolidation_batches ORDER BY batch_key"
                    )
                )
                assert (
                    before
                    and conn.execute(
                        "SELECT count(*) FROM conversations WHERE title='Retained title'"
                    ).fetchone()[0]
                    == 2
                )
            with (
                captured.owner("db.workspaces") as workspace,
                workspace.transaction() as conn,
            ):
                assert (
                    conn.execute(
                        "SELECT count(*) FROM mobile_consolidation_batches"
                    ).fetchone()[0]
                    == 0
                )
                assert (
                    conn.execute(
                        "SELECT count(*) FROM workspace_records WHERE name='Imported project'"
                    ).fetchone()[0]
                    == 0
                )

    def reject_chat_replay(connection, row, values, detach=False):
        if row.table == "conversations":
            raise AssertionError(
                "committed Chat was replayed before receipt verification"
            )
        return insert(connection, row, values, detach=detach)

    with monkeypatch.context() as fault:
        fault.setattr(mobile_native_rows, "_insert_row", reject_chat_replay)
        with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:  # noqa: SIM117 -- Keep the explicit native resource/fault lifetime boundary.
            with _resume_captured_mobile(
                fresh,
                journal,
                destination=destination,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured:
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
                for owner_id, key in (
                    ("db.chachanotes.primary", "chat"),
                    ("db.workspaces", "workspaces"),
                    ("db.agent_runs", "runs"),
                ):
                    with captured.owner(owner_id) as owner, owner.transaction() as conn:
                        assert (
                            conn.execute(
                                "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                                (journal.operation_id, key),
                            ).fetchone()[0]
                            == 1
                        )
                        if key == "chat":
                            assert (
                                tuple(
                                    tuple(row)
                                    for row in conn.execute(
                                        "SELECT * FROM mobile_consolidation_batches ORDER BY batch_key"
                                    )
                                )
                                == before
                            )
                            assert (
                                conn.execute(
                                    "SELECT count(*) FROM conversations WHERE title='Retained title'"
                                ).fetchone()[0]
                                == 2
                            )


@pytest.mark.parametrize("crash_owner", ["chat", "runs"])
def test_cold_commit_before_retirement_uses_original_capture_and_receipt(
    tmp_path, monkeypatch, crash_owner
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery import profile_paths, storage_admission
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    marker = os.environ.get("TLDW_OWNING_COLD_CRASH_MARKER")
    if marker:
        monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
        destination = tmp_path / "archives"
        destination.mkdir(mode=0o700)
        with _capture_fixture(tmp_path, monkeypatch) as (  # noqa: SIM117 -- Native resource lifetime boundary.
            session,
            journal,
            config,
            _,
            preview,
            inventory,
            authority,
        ):
            with _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=destination,
                password=b"test-private-mobile-rollback",
                cancel=Event(),
            ) as captured:
                retire = storage_admission._MobileScope.retire
                receipts = []

                def die_after_commit(scope):
                    if scope.owner["owner_id"] == "db.chachanotes.primary":
                        with scope.repository.transaction() as conn:
                            receipts.extend(
                                tuple(row)
                                for row in conn.execute(
                                    "SELECT * FROM mobile_consolidation_batches ORDER BY batch_key"
                                )
                            )
                            assert (
                                receipts
                                and conn.execute(
                                    "SELECT count(*) FROM conversations WHERE title='Retained title'"
                                ).fetchone()[0]
                                == 2
                            )
                    if (
                        scope.owner["owner_id"]
                        == {"chat": "db.chachanotes.primary", "runs": "db.agent_runs"}[
                            crash_owner
                        ]
                    ):
                        with scope.repository.transaction() as conn:
                            assert (
                                conn.execute(
                                    "SELECT count(*) FROM mobile_consolidation_batches WHERE batch_key=?",
                                    (crash_owner,),
                                ).fetchone()[0]
                                == 1
                            )
                        Path(marker).write_text(
                            json.dumps(
                                {
                                    "bootstrap": str(authority.control_root.parent),
                                    "control": str(journal.root.parent),
                                    "config": str(config),
                                    "destination": str(destination),
                                    "default": str(profile_paths.default_config_path()),
                                    "effective": str(
                                        profile_paths.effective_config_path()
                                    ),
                                    "source": observe_artifact(captured.source.path),
                                    "canonical": observe_artifact(
                                        captured.canonical.path
                                    ),
                                    "receipts": receipts,
                                    "sandbox_environment": {
                                        key: os.environ[key]
                                        for key in (
                                            "HOME",
                                            "USERPROFILE",
                                            "XDG_DATA_HOME",
                                            "XDG_CONFIG_HOME",
                                            "TLDW_CONFIG_PATH",
                                            "TLDW_TEST_CONFIG_ROOT",
                                            "TLDW_TEST_CONFIG_ROOT_OWNER",
                                        )
                                        if key in os.environ
                                    },
                                }
                            )
                        )
                        os._exit(73)
                    return retire(scope)

                monkeypatch.setattr(
                    storage_admission._MobileScope, "retire", die_after_commit
                )
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
        pytest.fail("actual commit crash was not reached")
    marker = tmp_path / "cold-commit.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_cold_commit_before_retirement_uses_original_capture_and_receipt[{crash_owner}]",
            "-q",
            "--no-cov",
            "--basetemp",
            str(tmp_path / "child"),
        ],
        env={**os.environ, "TLDW_OWNING_COLD_CRASH_MARKER": str(marker)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 73, child.stdout + child.stderr
    evidence = json.loads(marker.read_text())
    import inspect

    script = (
        """
import json, sys, traceback
from pathlib import Path
from threading import Event
import pytest
from Tests import real_profile_guard, network_guard
real_profile_guard.install()
network_guard.install()
"""
        + inspect.getsource(_resume_crashed_import)
        + """
try:
    _resume_crashed_import(json.loads(Path(sys.argv[1]).read_text()))
except BaseException:
    traceback.print_exc()
    raise
finally:
    violations = real_profile_guard.take_violations()
    blocked = network_guard.blocked_attempts()
    assert not violations, violations
    assert not blocked, blocked
    print('resume_native_guards_empty', flush=True)
"""
    )
    resumed = subprocess.run(
        [sys.executable, "-c", script, str(marker)],
        env={**os.environ, **evidence["sandbox_environment"]},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr


def _resume_crashed_import(evidence):
    """Cold native process using the same isolated profile/default locations."""
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        config_adapter,
        mobile_native_rows,
        profile_paths,
    )
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _resume_captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.recovery_copies import _journal

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("TLDW_CONFIG_PATH", evidence["effective"])
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: Path(evidence["bootstrap"])
    )
    monkeypatch.setattr(
        profile_paths, "default_config_path", lambda: Path(evidence["default"])
    )
    monkeypatch.setattr(
        config_adapter, "default_config_path", profile_paths.default_config_path
    )
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    journal = _journal(Path(evidence["control"]), "mobile-test")
    with journal._locked(exclusive=False) as fd:
        assert journal._records(fd)[-1].event == "mobile_owner_opened"
    insert = mobile_native_rows._insert_row

    def reject_replay(conn, row, values, detach=False):
        if row.table == "conversations":
            raise AssertionError("committed Chat insertion replayed")
        return insert(conn, row, values, detach=detach)

    monkeypatch.setattr(mobile_native_rows, "_insert_row", reject_replay)
    with admission_authority(Path(evidence["bootstrap"])).maintenance(  # noqa: SIM117 -- Native resource lifetime boundary.
        ("bootstrap.unbound", "mobile.profile"), 2
    ) as fresh:
        with _resume_captured_mobile(
            fresh,
            journal,
            destination=Path(evidence["destination"]),
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            assert observe_artifact(captured.source.path) == evidence["source"]
            assert observe_artifact(captured.canonical.path) == evidence["canonical"]
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
            for owner_id, key in (
                ("db.chachanotes.primary", "chat"),
                ("db.workspaces", "workspaces"),
                ("db.agent_runs", "runs"),
            ):
                with captured.owner(owner_id) as owner, owner.transaction() as conn:
                    assert (
                        conn.execute(
                            "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                            (journal.operation_id, key),
                        ).fetchone()[0]
                        == 1
                    )
                    if key == "chat":
                        assert [
                            list(row)
                            for row in conn.execute(
                                "SELECT * FROM mobile_consolidation_batches ORDER BY batch_key"
                            )
                        ] == evidence["receipts"]
                        assert (
                            conn.execute(
                                "SELECT count(*) FROM conversations WHERE title='Retained title'"
                            ).fetchone()[0]
                            == 2
                        )


def test_imported_run_cannot_stamp_notes_but_fresh_manual_send_keeps_exact_pending_ids(
    tmp_path,
):
    from Tests.agent_runs_owner import owned_agent_runs_db
    from tldw_chatbook.DB.automatic_work import AutomaticWorkRefused

    db = owned_agent_runs_db(tmp_path / "notes.db")
    try:
        old = db.create_run(conversation_id="chat", agent_kind="primary")
        fresh = db.create_run(conversation_id="chat", agent_kind="primary")
        db.set_status(old, "done", result="historical result")
        with db.transaction() as conn:
            db.hold_imported_authority(conn, "operation", [("run", old)])

        def note(body):
            return db.add_change_note(
                run_id=old,
                root="/historical/root",
                path="file.py",
                hunk_index=0,
                hunk_header="@@ -1 +1 @@",
                hunk_excerpt="self-contained retained excerpt",
                note=body,
                snapshot_id=41,
            )

        first = note("retained annotation")
        included = [row["id"] for row in db.pending_notes_for_conversation("chat")]
        assert included == [first]
        later = note("created after send was prepared")
        before = db.notes_for_run(old)
        with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
            db.mark_notes_delivered(included, delivered_by_run_id=old)
        assert db.notes_for_run(old) == before
        assert db.mark_notes_delivered(included, delivered_by_run_id=fresh) == [first]
        assert [row["id"] for row in db.pending_notes_for_conversation("chat")] == [
            later
        ]
        assert db.notes_for_run(old)[0]["delivered_by_run_id"] == fresh
        assert db.mark_notes_delivered([later]) == [later]
    finally:
        db.close()


@pytest.mark.parametrize(
    "native_module",
    [
        "tldw_chatbook.Chat.console_trace_native_reader",
        "tldw_chatbook.Character_Chat.Chat_Dictionary_Lib",
        "tldw_chatbook.Chat.console_generation_settings_metadata",
        "tldw_chatbook.Chat.console_session_settings",
        "tldw_chatbook.Chat.console_context_compaction",
    ],
)
def test_cold_native_reader_import_does_not_activate_current_config(native_module):
    import subprocess
    import sys

    script = """
import sys
from Tests import real_profile_guard, network_guard
real_profile_guard.install()
network_guard.install()
try:
    import importlib, traceback
    config_import_stacks = []
    def observe_config_import(event, args):
        if event == 'import' and args[0] == 'tldw_chatbook.config' and not config_import_stacks:
            config_import_stacks.append(''.join(traceback.format_stack(limit=12)))
    sys.addaudithook(observe_config_import)
    module = importlib.import_module(sys.argv[1])
    if sys.argv[1].endswith('Chat_Dictionary_Lib'):
        entry = module.ChatDictionary.from_dict({'key': 'term', 'content': 'value', 'enabled': False, 'priority': 3})
        assert entry.to_dict()['content'] == 'value'
        assert entry.enabled is False and entry.priority == 3
        try:
            module.ChatDictionary.from_dict({'content': 'missing key'})
        except KeyError:
            pass
        else:
            raise AssertionError('native required key validation was skipped')
    elif sys.argv[1].endswith('console_generation_settings_metadata'):
        from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
        snapshot = module.snapshot_from_session_settings(ConsoleSessionSettings(provider='openai'))
        valid = module.merge_console_generation_settings({}, snapshot)
        assert module.parse_console_generation_settings(valid).snapshot == snapshot
        valid['console_generation_settings']['unknown'] = True
        assert module.parse_console_generation_settings(valid).status is module.ConsoleGenerationSettingsReadStatus.INVALID
        assert module.parse_console_generation_settings({}).status is module.ConsoleGenerationSettingsReadStatus.ABSENT
        assert module.parse_console_generation_settings({'console_generation_settings': {'version': 1}}).status is module.ConsoleGenerationSettingsReadStatus.INVALID
    elif sys.argv[1].endswith('console_session_settings'):
        import dataclasses, json
        settings = module.ConsoleSessionSettings(provider='openai')
        payload = {'console_session_settings': {'version': 1, **dataclasses.asdict(settings)}}
        assert module.parse_persisted_console_session_settings(json.dumps(payload)) == settings
        payload['console_session_settings']['unknown'] = True
        assert module.parse_persisted_console_session_settings(json.dumps(payload)) is None
        del payload['console_session_settings']['unknown']
        payload['console_session_settings']['temperature'] = True
        assert module.parse_persisted_console_session_settings(json.dumps(payload)) is None
    elif sys.argv[1].endswith('console_context_compaction'):
        first = module.DurableMessageSnapshot('user', 1, 'user', 'retained')
        second = module.DurableMessageSnapshot('reply', 1, 'assistant', 'answer')
        units = module.complete_durable_units((first, second))
        assert len(units) == 1 and units[0].boundary_message_id == 'reply'
        assert module.prefix_digest((first, second)) != module.prefix_digest((first,))
        try:
            module.DurableConversationUnit(())
        except ValueError:
            pass
        else:
            raise AssertionError('empty durable unit accepted')
    assert 'tldw_chatbook.config' not in sys.modules, config_import_stacks
finally:
    assert not real_profile_guard.take_violations()
    assert not network_guard.blocked_attempts()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, native_module],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recovery_anchor_encodes_identical_native_event_once_per_current_check(
    monkeypatch,
):
    from tldw_chatbook.Backup_Recovery import journal
    from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import _owner_observation
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    opened = journal._Event(
        operation_id="native-operation",
        sequence=3,
        previous="a" * 64,
        event="mobile_owner_opened",
        evidence={"owner_id": "db.agent_runs", "path": "/private/native/runs.db"},
    )
    encoded = journal._encoded
    calls = []

    def observe(event):
        calls.append(event)
        return encoded(event)

    monkeypatch.setattr(journal, "_encoded", observe)
    budget = _TraceReadBudget()
    before = budget._remaining
    first = _owner_observation([opened], budget)
    charged = before - budget._remaining
    assert len(calls) == 1 and calls[0] is opened
    assert first[0] == first[2] == opened.sequence and first[3] == opened.previous
    assert first[1] is first[4]
    before = budget._remaining
    assert _owner_observation([opened], budget) == first
    assert len(calls) == 2 and before - budget._remaining == charged
    # A distinct native event at the same owner path is separately encoded.
    tail = journal._Event(
        operation_id=opened.operation_id,
        sequence=4,
        previous="b" * 64,
        event="mobile_mode_observed",
        evidence=dict(opened.evidence),
    )
    later = _owner_observation([opened, tail], budget)
    assert calls[-2] is opened and calls[-1] is tail and len(calls) == 4
    assert later[:2] == first[:2] and later[2:4] == (tail.sequence, tail.previous)
    assert later[4] != first[4]


def test_native_discovery_child_requires_exact_issued_copy_and_lists(
    tmp_path, monkeypatch
):
    from copy import copy

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import (
        mobile_native_operation,
        mobile_runs_preflight,
        storage_admission,
    )

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    checked = []

    class CheckedNativeChild(Exception):
        pass

    def inspect_child(reads, *, _read_budget):
        with reads.session._discovery_reads(private_sqlite=True):
            child = storage_admission._local.discovery_scope
            mobile_native_operation._require_readers_retired(
                reads.session, _reader=reads
            )
            copies = child.sqlite_copies
            for target, attribute, replacement in (
                (storage_admission._local, "discovery_scope", copy(child)),
                (child, "sqlite_copies", copy(copies)),
                (child, "resources", list(child.resources)),
                (copies, "resources", list(copies.resources)),
            ):
                with monkeypatch.context() as changed:
                    changed.setattr(target, attribute, replacement)
                    with pytest.raises((ValueError, RuntimeError)):
                        mobile_native_operation._require_readers_retired(
                            reads.session, _reader=reads
                        )
            admission = storage_admission._mobile_admission(reads.session)
            original_map = admission[3]
            admission[3] = dict(original_map)
            try:
                with pytest.raises((ValueError, RuntimeError)):
                    mobile_native_operation._require_readers_retired(
                        reads.session, _reader=reads
                    )
            finally:
                admission[3] = original_map
            binding = original_map.pop(child)
            try:
                with pytest.raises((ValueError, RuntimeError)):
                    mobile_native_operation._require_readers_retired(
                        reads.session, _reader=reads
                    )
            finally:
                original_map[child] = binding
            custody = storage_admission._CaptureFileDescriptors(child)
            custody.close_failed = True
            try:
                with pytest.raises((ValueError, RuntimeError)):
                    mobile_native_operation._require_readers_retired(
                        reads.session, _reader=reads
                    )
            finally:
                custody.close_failed = False
                custody.retire()
            with pytest.raises((ValueError, RuntimeError)):
                mobile_native_operation._require_readers_retired(reads.session)
            checked.append(child)
        assert not child.active and not copies.active and copies.directory is None
        raise CheckedNativeChild

    monkeypatch.setattr(mobile_runs_preflight, "_plan_runs", inspect_child)
    with captured_operation(tmp_path, monkeypatch) as captured:
        with pytest.raises(CheckedNativeChild):
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
        assert checked
        with captured.owner("db.agent_runs") as owner, owner.transaction() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches"
                ).fetchone()[0]
                == 0
            )


def test_actual_writer_publishes_native_history_and_commits_member_witnesses(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt

    seeded = {}

    def seed(stores, **kwargs):
        result = _small_native_seed(stores, **kwargs, source_case="history")
        seeded["legacy" if kwargs["legacy"] else "canonical"] = result
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        captured.consolidate(password=b"test-private-mobile-rollback", cancel=Event())
        with (
            captured.owner("db.agent_runs") as owner,
            owner.transaction() as connection,
        ):
            receipt = _read_owner_receipt(
                connection,
                captured.journal.operation_id,
                captured.preview.source_fingerprint,
                "runs",
                _TraceReadBudget(),
            )
        assert receipt is not None
        children = receipt[1]
        roots = [row["data"] for row in children if row["kind"] == "runlog_root"]
        members = [row["data"] for row in children if row["kind"] == "runlog"]
        assert roots and members
        for row in roots + members:
            assert observe_artifact(Path(row["artifact"]["path"])) == row["artifact"]
        framed = [row for row in members if row["framed_runs"]]
        assert len(framed) == 1
        assert framed[0]["framed_runs"][0][0] == seeded["legacy"]["run"]
        assert (
            Path(framed[0]["artifact"]["path"]).read_bytes()
            == next(seeded["legacy"]["log_dir"].glob("logs.*.txt")).read_bytes()
        )
        assert (
            len([row for row in children if row["kind"] == "history_publication"]) == 1
        )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(captured.preview.target_selector))
    with pytest.raises(RecoveryRequired, match="recovery_pending"):  # noqa: SIM117 -- Keep the explicit native resource/fault lifetime boundary.
        with acquire_storage(captured.preview.destination.run_path):
            pytest.fail("pending mobile target admitted ordinary runtime storage")


def test_native_publication_failure_resumes_same_objects_before_owner_commits(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _resume_captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_history_files import _HistoryFiles
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt

    def seed(stores, **kwargs):
        return _small_native_seed(stores, **kwargs, source_case="history")

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    destination = tmp_path / "archives"
    destination.mkdir(mode=0o700)
    published = []
    publish = _HistoryFiles.publish

    def fail_after_native_publication(scope):
        result = publish(scope)
        published.extend(
            (member.logical_id, observe_artifact(scope.location / member.location))
            for member in result
        )
        raise OSError("publication completed before owner interruption")

    with _capture_fixture(tmp_path, monkeypatch) as (  # noqa: SIM117 -- Native resource lifetime boundary.
        session,
        journal,
        _config,
        _legacy,
        preview,
        inventory,
        authority,
    ):
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            original_ciphers = (
                observe_artifact(captured.source.path),
                observe_artifact(captured.canonical.path),
            )
            with monkeypatch.context() as fault:
                fault.setattr(_HistoryFiles, "publish", fail_after_native_publication)
                with pytest.raises(OSError, match="publication completed"):
                    captured.consolidate(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    )
            assert published
            with journal._locked(exclusive=False) as fd:
                events = journal._records(fd)
                assert events[-1].event == "mobile_history_verified"
                assert not any(event.event == "mobile_owner_opened" for event in events)
            with pytest.raises(ValueError, match="already_attempted"):
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
    with authority.maintenance(("bootstrap.unbound", "mobile.profile"), 2) as fresh:  # noqa: SIM117 -- Keep the explicit native resource/fault lifetime boundary.
        with _resume_captured_mobile(
            fresh,
            journal,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            assert (
                observe_artifact(captured.source.path),
                observe_artifact(captured.canonical.path),
            ) == original_ciphers
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
            with (
                captured.owner("db.agent_runs") as owner,
                owner.transaction() as connection,
            ):
                receipt = _read_owner_receipt(
                    connection,
                    journal.operation_id,
                    captured.preview.source_fingerprint,
                    "runs",
                    _TraceReadBudget(),
                )
            actual = {
                child["data"]["logical_id"]: child["data"]["artifact"]
                for child in receipt[1]
                if child["kind"] == "runlog"
            }
            assert actual == dict(published)
            for artifact in actual.values():
                assert observe_artifact(Path(artifact["path"])) == artifact


def test_actual_goal_note_definition_owners_preserve_history_and_current_guards(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.automatic_work import AutomaticWorkRefused
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
    from tldw_chatbook.DB.mobile_import_authority import chain_allowed

    seeded = {}

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs, source_case="consumers")
        seeded["legacy" if kwargs["legacy"] else "canonical"] = value
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        captured.consolidate(password=b"test-private-mobile-rollback", cancel=Event())
        with captured.owner("db.agent_runs") as owner:
            with owner.transaction() as connection:
                _master, children = _read_owner_receipt(
                    connection,
                    captured.journal.operation_id,
                    captured.preview.source_fingerprint,
                    "runs",
                    _TraceReadBudget(),
                )
                mapped = {
                    (row["data"]["domain"], row["data"]["original"]): row["data"][
                        "mapped"
                    ]
                    for row in children
                    if row["kind"] == "identity"
                    and type(row["data"]["original"]) in (str, int)
                }
                assert not chain_allowed(
                    connection, mapped["work_chain", seeded["legacy"]["chain"]]
                )
                assert (
                    connection.execute(
                        "SELECT seq FROM sqlite_sequence WHERE name='change_snapshots'"
                    ).fetchone()[0]
                    >= 42
                )
                assert (
                    connection.execute(
                        "SELECT count(*) FROM change_snapshots"
                    ).fetchone()[0]
                    == 0
                )
                note = connection.execute(
                    "SELECT note,hunk_excerpt,snapshot_id,delivered_at FROM change_notes WHERE id=?",
                    (mapped["change_note", seeded["legacy"]["note"]],),
                ).fetchone()
                assert tuple(note) == (
                    "Pending historical annotation",
                    "Retained self-contained excerpt",
                    42,
                    None,
                )
            imported_definition = mapped[
                "agent_definition", seeded["legacy"]["definition"]
            ]
            assert owner.get_agent_definition(imported_definition)["enabled"] == 1
            enabled = {
                row["id"] for row in owner.list_agent_definitions(enabled_only=True)
            }
            assert imported_definition not in enabled
            assert seeded["canonical"]["definition"] in enabled
            goal = owner.goal_runs.get(mapped["goal", seeded["legacy"]["goal"]])
            assert goal.request.objective == "Repair fixture"
            with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
                owner.goal_runs.set_provisioning(goal, status="ready")
            assert owner.goal_runs.get(goal.id) == goal


def test_actual_passive_chat_rows_commit_with_native_consumer_disposition(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleDispatchRecoveryActionId,
        ConsoleDispatchRecoveryKind,
    )
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt

    seeded = {}

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs, source_case="passive")
        seeded["legacy" if kwargs["legacy"] else "canonical"] = value
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        from tldw_chatbook.Backup_Recovery.mobile_native_operation import (
            _NativeOperationBlocked,
        )

        try:
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
        except _NativeOperationBlocked as error:
            pytest.fail(repr(error.blockers))
        with captured.owner("db.chachanotes.primary") as owner:
            with owner.transaction() as cursor:
                _master, children = _read_owner_receipt(
                    cursor.connection,
                    captured.journal.operation_id,
                    captured.preview.source_fingerprint,
                    "chat",
                    _TraceReadBudget(),
                )
                mapped = {
                    (row["data"]["domain"], row["data"]["original"]): row["data"][
                        "mapped"
                    ]
                    for row in children
                    if row["kind"] == "identity"
                    and type(row["data"]["original"]) in (str, int)
                }
                fleet = cursor.execute(
                    "SELECT body FROM fleet_progress_messages WHERE message_id=?",
                    (mapped["fleet_message", seeded["legacy"]["fleet"]],),
                ).fetchone()
                assert fleet[0] == "Historical fleet report"
                for table in (
                    "console_hook_continuation_receipts",
                    "console_machine_followup_receipts",
                ):
                    association = cursor.execute(
                        f"SELECT parent_assistant_message_id,assistant_message_id FROM {table}"
                    ).fetchone()
                    assert (
                        association[1]
                        == mapped["message", seeded["legacy"]["dispatch"]]
                    )
                    assert association[0] != seeded["legacy"]["dispatch"]
            conversation = mapped["conversation", seeded["legacy"]["conversation"]]
            recovery = ConsoleDispatchRepository(owner).reconcile_for_session(
                conversation
            )
            assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
            assert recovery.error_code == "imported_history"
            assert [
                (action.action_id, action.enabled) for action in recovery.actions
            ] == [(ConsoleDispatchRecoveryActionId.DISCARD, True)]
            assert (
                recovery.assistant_message_id
                == mapped["message", seeded["legacy"]["dispatch"]]
            )
            assert recovery.queue_entry_id is recovery.preparation_id is None
            assert FleetProgressRepository(owner).load(conversation) == ()


def test_actual_failed_snapshot_retains_native_history_without_git(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
    from tldw_chatbook.UI.Screens.change_review_screen import (
        AgentRunsChangeReviewProvider,
    )
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

    seeded = {}

    def seed(stores, **kwargs):
        result = _small_native_seed(stores, **kwargs, source_case="failed_snapshot")
        if kwargs["legacy"]:
            seeded.update(result)
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        captured.consolidate(password=b"test-private-mobile-rollback", cancel=Event())
        with (
            captured.owner("db.agent_runs") as owner,
            owner.transaction() as connection,
        ):
            _master, children = _read_owner_receipt(
                connection,
                captured.journal.operation_id,
                captured.preview.source_fingerprint,
                "runs",
                _TraceReadBudget(),
            )
            mapped = {
                (row["data"]["domain"], row["data"]["original"]): row["data"]["mapped"]
                for row in children
                if row["kind"] == "identity"
                and type(row["data"]["original"]) in (str, int)
            }
            original = seeded["snapshot"]
            saved = dict(
                connection.execute(
                    "SELECT * FROM change_snapshots WHERE id=?",
                    (mapped["change_snapshot", original["id"]],),
                ).fetchone()
            )
            assert saved == original | {
                "id": mapped["change_snapshot", original["id"]],
                "run_id": mapped["run", seeded["run"]],
            }
            assert (
                connection.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='change_snapshots'"
                ).fetchone()[0]
                >= saved["id"]
            )
            service = ShadowRepoService(data_dir=tmp_path / "unopened-shadow")

            def no_git(*args, **kwargs):
                pytest.fail("failed snapshot history attempted a Git lookup")

            monkeypatch.setattr(service, "repo_for_root", no_git)
            provider = AgentRunsChangeReviewProvider(
                db=owner,
                service=service,
                conversation_id="unused",
                diff_display_max_lines=2000,
            )
            assert provider.changed_files(saved) == []
            assert not (tmp_path / "unopened-shadow").exists()


def test_actual_publisher_iterator_close_failure_quarantines_native_admission(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    marker = os.environ.get("TLDW_NATIVE_PUBLISHER_CLOSE_PROOF")
    if marker is None:
        proof = tmp_path / "publisher-close.json"
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--show-capture=no",
                "--no-cov",
                "--basetemp",
                str(tmp_path / "child"),
                __file__
                + "::test_actual_publisher_iterator_close_failure_quarantines_native_admission",
            ],
            env={**os.environ, "TLDW_NATIVE_PUBLISHER_CLOSE_PROOF": str(proof)},
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        evidence = json.loads(proof.read_text())
        assert evidence["contender_returncode"] == 23 and evidence["guards_empty"]
        from tldw_chatbook.Backup_Recovery.control_records import admission_authority

        with admission_authority(Path(evidence["control_root"])).maintenance(
            ("bootstrap.unbound", "mobile.profile"), 2
        ):
            pass
        return
    from Tests import network_guard, real_profile_guard
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_native_operation import (
        _assert_retirement_quarantine,
    )
    from tldw_chatbook.Backup_Recovery import journal as journal_module
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.mobile_history_files import _HistoryFiles

    def seed(stores, **kwargs):
        return _small_native_seed(stores, **kwargs, source_case="history")

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    reached = {}
    native_observe, native_scandir = _HistoryFiles._observe, journal_module.os.scandir

    class CloseFault:
        def __init__(self, native):
            self.native, self.close_calls = native, 0

        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("injected native enumeration interruption before exhaustion")

        def close(self):
            self.close_calls += 1
            raise OSError("injected close refusal before native iterator close")

    def observe(scope, path):
        reached["scope"] = scope

        def interrupted(fd):
            iterator = CloseFault(native_scandir(fd))
            reached["iterator"] = iterator
            return iterator

        with monkeypatch.context() as fault:
            fault.setattr(journal_module.os, "scandir", interrupted)
            return native_observe(scope, path)

    with (
        pytest.raises(AdmissionError, match="resources_not_retired"),
        captured_operation(tmp_path, monkeypatch) as captured,
    ):
        with monkeypatch.context() as fault:
            fault.setattr(_HistoryFiles, "_observe", observe)
            with pytest.raises(RecoveryRequired, match="resources_not_retired"):
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
        scope, iterator = reached["scope"], reached["iterator"]
        assert not scope.active and scope.resources
        resource = next(item for item in scope.resources if item.directory_iterators)
        assert resource.close_failed and resource.directory_iterators == [iterator]
        assert iterator.close_calls == 1
        assert resource.fds and all(os.fstat(fd) for fd in resource.fds)
        assert next(
            iterator.native
        ).name  # Actual unexhausted native iterator remains live.
        with (
            pytest.raises(ValueError, match="readers_not_retired"),
            captured.owner("db.agent_runs"),
        ):
            pytest.fail("publisher's unretired physical reader admitted an owner")
        with captured.journal._locked(exclusive=False) as fd:
            records = captured.journal._records(fd)
            assert any(row.event == "mobile_history_intended" for row in records)
            assert not any(row.event == "mobile_owner_opened" for row in records)
        assert iterator.close_calls == 1
    assert storage_admission._failed_capture_holds
    control_root = str(captured.session._control.parent)
    code = _assert_retirement_quarantine(control_root)
    assert not real_profile_guard.take_violations()
    assert not network_guard.blocked_attempts()
    Path(marker).write_text(
        json.dumps(
            {
                "control_root": control_root,
                "contender_returncode": code,
                "guards_empty": True,
            }
        )
    )
    os._exit(0)  # Actual process death, never a retry/flag reset, releases custody.


@pytest.mark.parametrize(
    "source_case,table,reason",
    [
        (
            "physical_snapshot",
            "change_snapshots",
            "native_physical_dependency_unproved",
        ),
        (
            "physical_worktree",
            "agent_worktrees",
            "physical_worktree_dependencies_unproved",
        ),
    ],
)
def test_actual_missing_physical_authority_refuses_before_first_effect(
    tmp_path, monkeypatch, source_case, table, reason
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.mobile_native_operation import (
        _NativeOperationBlocked,
    )

    seeded = {}

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs, source_case=source_case)
        if kwargs["legacy"]:
            seeded.update(value)
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        physical = observe_artifact(seeded["physical_root"])
        with captured.journal._locked(exclusive=False) as fd:
            events = tuple(captured.journal._records(fd))
        with pytest.raises(_NativeOperationBlocked) as error:
            captured.consolidate(
                password=b"test-private-mobile-rollback", cancel=Event()
            )
        assert any(
            block.table == table and block.reason == reason
            for block in error.value.blockers
        )
        assert _fingerprints(tmp_path / "profile") == before
        assert observe_artifact(seeded["physical_root"]) == physical
        with captured.journal._locked(exclusive=False) as fd:
            assert tuple(captured.journal._records(fd)) == events


def test_actual_first_root_recovery_refuses_replacement_unrelated_declaration_and_extra_file(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery.journal import observe_artifact
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
        _resume_captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_history_files import _HistoryFiles

    def seed(stores, **kwargs):
        return _small_native_seed(stores, **kwargs, source_case="history")

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    destination = tmp_path / "archives"
    destination.mkdir(mode=0o700)
    publish = _HistoryFiles.publish

    def interrupt(scope):
        publish(scope)
        raise OSError("native publication completed before owners")

    with _capture_fixture(tmp_path, monkeypatch) as (  # noqa: SIM117 -- Native resource lifetime boundary.
        session,
        journal,
        _config,
        _legacy,
        preview,
        inventory,
        authority,
    ):
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=destination,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            originals = (
                observe_artifact(captured.source.path),
                observe_artifact(captured.canonical.path),
            )
            with monkeypatch.context() as fault:
                fault.setattr(_HistoryFiles, "publish", interrupt)
                with pytest.raises(OSError, match="publication completed"):
                    captured.consolidate(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    )
            with journal._locked(exclusive=False) as fd:
                events = tuple(journal._records(fd))
            intent = next(
                row.evidence for row in events if row.event == "mobile_history_intended"
            )
            assert events[-1].event == "mobile_history_verified"
    publication = Path(intent["publication"])
    original_object = observe_artifact(publication)
    for fault_kind in ("replaced_root", "unrelated_declaration", "extra_file"):
        retained = tmp_path / "original-publication"
        unrelated = Path(intent["parent"]["path"]) / "change_review"
        extra = publication / "unexpected-retained.txt"
        if fault_kind == "replaced_root":
            publication.rename(retained)
            publication.mkdir(mode=0o700)
        elif fault_kind == "unrelated_declaration":
            assert not unrelated.exists()
            unrelated.mkdir(mode=0o700)
        else:
            extra.write_bytes(b"Unknown file must remain untouched")
        try:
            expected = observe_artifact(publication)
            with authority.maintenance(  # noqa: SIM117 -- Native resource lifetime boundary.
                ("bootstrap.unbound", "mobile.profile"), 2
            ) as fresh:
                with _resume_captured_mobile(
                    fresh,
                    journal,
                    destination=destination,
                    password=b"test-private-mobile-rollback",
                    cancel=Event(),
                ) as resumed:
                    assert (
                        observe_artifact(resumed.source.path),
                        observe_artifact(resumed.canonical.path),
                    ) == originals
                    with pytest.raises(
                        ValueError,
                        match="scope_changed|mobile_history_object_changed|artifact_limit",
                    ):
                        resumed.consolidate(
                            password=b"test-private-mobile-rollback", cancel=Event()
                        )
                    with journal._locked(exclusive=False) as fd:
                        assert tuple(journal._records(fd)) == events
            assert observe_artifact(publication) == expected
            if fault_kind == "replaced_root":
                assert observe_artifact(retained) == original_object | {
                    "path": str(retained)
                }
            elif fault_kind == "unrelated_declaration":
                assert unrelated.is_dir()
            else:
                assert extra.read_bytes() == b"Unknown file must remain untouched"
        finally:
            # Undo only this test's injected mutation; originals are never deleted.
            if fault_kind == "replaced_root":
                publication.rmdir()
                retained.rename(publication)
            elif fault_kind == "unrelated_declaration":
                unrelated.rmdir()
            else:
                extra.unlink()
    assert observe_artifact(publication) == original_object
