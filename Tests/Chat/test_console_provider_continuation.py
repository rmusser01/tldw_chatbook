"""Console ownership and recovery for durable provider continuation."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    ContinuationEventContext,
    ToolBatchReady,
    ToolCallExecuting,
    ToolCallFinished,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ContinuationDurabilityResult,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ContinuationResult,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.chat_outbox_producer import ChatSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.crypto import decrypt_sync_payload, generate_dataset_key
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from Tests.console_provider_doubles import provider_resolution


def _active_checkpoint() -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k2",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("PRIVATE-REASONING-CANARY",),
                calls=(
                    ContinuationCall(
                        call_id="PRIVATE-CALL-ID",
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        state="pending",
                    ),
                ),
            ),
        ),
    )


def _complete_k3_checkpoint(content: str) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content=content,
                reasoning_blocks=("PRIVATE-K3-REASONING",),
                calls=(),
            ),
        ),
    )


def test_first_tool_batch_force_creates_preallocated_owner_and_stream_reuses_it() -> (
    None
):
    """Removing the event write would let tool dispatch precede its durable owner."""
    database = CharactersRAGDB(":memory:", "console-continuation-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="Durable continuation")
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Use the calculator",
            persist=True,
        )
        owner = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        checkpoint = _active_checkpoint()

        assert user.persisted_message_id is not None
        assert owner.persisted_message_id is None
        store.persist_provider_continuation_event(
            ToolBatchReady(
                context=ContinuationEventContext(
                    owner_message_id=owner.id,
                    run_id="run-primary",
                    agent_kind="primary",
                    durability="persistent",
                ),
                checkpoint=checkpoint,
                expected_checkpoint_revision=None,
            )
        )

        created = database.get_message_by_id(owner.id)
        assert created is not None
        assert created["content"] == ""
        assert created["provider_continuation_json"] is not None
        assert store.get_message(owner.id).persisted_message_id == owner.id
        assert store.get_message(owner.id).provider_continuation == checkpoint

        store.append_stream_chunk(owner.id, "The answer is 4.")
        store.mark_message_complete(owner.id)

        updated = database.get_message_by_id(owner.id)
        assert updated is not None
        assert updated["content"] == "The answer is 4."
        assert updated["provider_continuation_json"] is not None
        assert (
            sum(
                1
                for message in database.get_messages_for_conversation(
                    session.persisted_conversation_id
                )
                if message["sender"] == "assistant"
            )
            == 1
        )
    finally:
        database.close_connection()


def _store_with_checkpoint(*, content: str = ""):
    database = CharactersRAGDB(":memory:", "console-continuation-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(database))
    session = store.create_session(title="Recovery")
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Use a tool", persist=True
    )
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=content, persist=True
    )
    store.persist_provider_continuation_event(
        ToolBatchReady(
            ContinuationEventContext(owner.id, "run", "primary", "persistent"),
            _active_checkpoint(),
            None,
        )
    )
    return database, store, session, owner


def _enable_sync(tmp_path, database, store):
    repository = SyncStateRepository(tmp_path / "continuation-sync.db")
    repository.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first",
        device_id="device-1",
        dataset_id="dataset-1",
    )
    dataset_key = generate_dataset_key()
    store.sync_v2_chat_producer = ChatSyncV2OutboxProducer(
        state_repository=repository,
        dataset_keys={"dataset-1": dataset_key},
        source=database,
    )
    store.sync_v2_server_profile_id = "server-a"
    store.sync_v2_authenticated_principal_id = "user-a"
    store.sync_v2_workspace_scope = "workspace-1"
    return repository, dataset_key


@pytest.mark.parametrize(
    "terminal_method",
    ["mark_message_complete", "mark_message_stopped", "mark_message_failed"],
)
def test_terminal_owner_mutations_refresh_version_and_project_private_checkpoint(
    tmp_path, terminal_method: str
) -> None:
    database, store, _session, owner = _store_with_checkpoint()
    try:
        repository, dataset_key = _enable_sync(tmp_path, database, store)
        store.append_stream_chunk(owner.id, "visible partial")

        getattr(store, terminal_method)(owner.id)

        current = store.get_message(owner.id)
        row = database.get_message_by_id(owner.id)
        assert row is not None
        assert current.provider_continuation_message_version == row["version"] == 2
        assert row["provider_continuation_json"] is not None
        entries = repository.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
        )
        assert len(entries) == 1
        envelope = entries[0]["envelope"]
        payload = decrypt_sync_payload(
            json.loads(envelope["payload_ciphertext"]), key=dataset_key
        )
        assert envelope["entity_version"] == 2
        assert payload["content"] == "visible partial"
        assert "PRIVATE-REASONING-CANARY" in payload["provider_continuation_json"]
    finally:
        database.close_connection()


def test_visible_discard_projects_explicit_checkpoint_clear(tmp_path) -> None:
    database, store, _session, owner = _store_with_checkpoint(content="Visible")
    try:
        repository, dataset_key = _enable_sync(tmp_path, database, store)
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None

        assert store.discard_provider_continuation(
            owner.id, expected_message_version=version
        )

        current = store.get_message(owner.id)
        assert current.provider_continuation is None
        assert current.provider_continuation_message_version == version + 1
        entries = repository.list_sync_v2_outbox_entries(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            dataset_id="dataset-1",
        )
        assert len(entries) == 1
        payload = decrypt_sync_payload(
            json.loads(entries[0]["envelope"]["payload_ciphertext"]),
            key=dataset_key,
        )
        assert payload == {"content": "Visible", "role": "assistant"}
    finally:
        database.close_connection()


def test_user_edit_clears_complete_k3_checkpoint_and_refreshes_owner_version() -> None:
    database = CharactersRAGDB(":memory:", "k3-edit-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="K3 edit")
        owner = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="original answer",
            persist=True,
        )
        checkpoint = _complete_k3_checkpoint("original answer")
        persisted_id = store.get_message(owner.id).persisted_message_id
        assert persisted_id is not None
        database.update_provider_continuation(
            message_id=persisted_id,
            expected_message_version=1,
            provider_continuation_json=dump_provider_continuation_json(checkpoint),
            content="original answer",
        )
        live_owner = store._message_or_raise(owner.id)
        live_owner.provider_continuation = checkpoint
        live_owner.provider_continuation_message_version = 2
        before = store.get_message(owner.id).provider_continuation_message_version
        assert before is not None

        updated = store.update_message_content(owner.id, "user edit")

        row = database.get_message_by_id(persisted_id)
        assert row is not None
        assert updated.provider_continuation is None
        assert updated.provider_continuation_message_version == before + 1
        assert row["provider_continuation_json"] is None
        assert row["content"] == "user edit"
    finally:
        database.close_connection()


def test_user_edit_preserves_non_k3_checkpoint_and_refreshes_owner_version() -> None:
    database, store, _session, owner = _store_with_checkpoint(content="Visible")
    try:
        before = store.get_message(owner.id).provider_continuation_message_version
        assert before is not None

        updated = store.update_message_content(owner.id, "Edited visible")

        assert updated.persisted_message_id is not None
        row = database.get_message_by_id(updated.persisted_message_id)
        assert row is not None
        assert updated.provider_continuation == _active_checkpoint()
        assert updated.provider_continuation_message_version == before + 1
        assert row["provider_continuation_json"] is not None
        assert row["content"] == "Edited visible"
    finally:
        database.close_connection()


def test_restore_hydrates_blank_owner_without_exposing_private_fields() -> None:
    database, store, session, owner = _store_with_checkpoint()
    try:
        restored = ConsoleChatStore(persistence=ChatPersistenceService(database))
        loaded = restored.restore_persisted_session(
            title="Recovery",
            workspace_id=None,
            persisted_conversation_id=session.persisted_conversation_id,
            all_nodes=[],
            active_leaf_persisted_id=owner.id,
        )
        interrupted = restored.interrupted_provider_continuation_message(loaded.id)
        assert interrupted is not None
        assert interrupted.id == owner.id
        assert interrupted.provider_continuation == _active_checkpoint()
        assert "PRIVATE" not in repr(interrupted)
    finally:
        database.close_connection()


def test_restore_never_infers_remote_state_from_database_client_id() -> None:
    database, _store, session, owner = _store_with_checkpoint()
    try:
        database.get_connection().execute(
            "UPDATE messages SET client_id = ? WHERE id = ?",
            ("legacy-other-client", owner.id),
        )
        database.get_connection().commit()
        restored = ConsoleChatStore(persistence=ChatPersistenceService(database))
        loaded = restored.restore_persisted_session(
            title="Recovery",
            workspace_id=None,
            persisted_conversation_id=session.persisted_conversation_id,
            all_nodes=[],
            active_leaf_persisted_id=owner.id,
        )
        interrupted = restored.interrupted_provider_continuation_message(loaded.id)
        assert interrupted is not None
        assert not interrupted.provider_continuation_remote
    finally:
        database.close_connection()


def test_restore_accepts_only_explicit_trusted_remote_active_marker() -> None:
    database, _store, session, owner = _store_with_checkpoint()
    try:
        restored = ConsoleChatStore(persistence=ChatPersistenceService(database))
        loaded = restored.restore_persisted_session(
            title="Recovery",
            workspace_id=None,
            persisted_conversation_id=session.persisted_conversation_id,
            all_nodes=[],
            active_leaf_persisted_id=owner.id,
            remote_active=True,
        )
        interrupted = restored.interrupted_provider_continuation_message(loaded.id)
        assert interrupted is not None
        assert interrupted.provider_continuation_remote
    finally:
        database.close_connection()


def test_discard_blank_owner_tombstones_and_visible_owner_is_retained() -> None:
    for content, retained in (("", False), ("Visible preface", True)):
        database, store, session, owner = _store_with_checkpoint(content=content)
        try:
            version = store.get_message(owner.id).provider_continuation_message_version
            persisted_id = store.get_message(owner.id).persisted_message_id
            assert version is not None
            assert persisted_id is not None
            assert store.discard_provider_continuation(
                owner.id,
                expected_message_version=version,
            )
            row = database.get_message_by_id(persisted_id)
            if retained:
                assert row is not None and row["content"] == content
                assert store.get_message(owner.id).provider_continuation is None
            else:
                assert row is None
                assert (
                    store.interrupted_provider_continuation_message(session.id) is None
                )
        finally:
            database.close_connection()


def test_discard_stale_version_preserves_checkpoint() -> None:
    database, store, _session, owner = _store_with_checkpoint(content="Visible")
    try:
        persisted_id = store.get_message(owner.id).persisted_message_id
        assert persisted_id is not None
        try:
            store.discard_provider_continuation(owner.id, expected_message_version=99)
        except Exception:
            pass
        row = database.get_message_by_id(persisted_id)
        assert row is not None and row["provider_continuation_json"] is not None
        assert store.get_message(owner.id).provider_continuation is not None
    finally:
        database.close_connection()


def test_successful_discard_clears_prior_recovery_warning() -> None:
    database, store, session, owner = _store_with_checkpoint(content="Visible")
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        store.set_provider_continuation_warning(
            owner.id,
            "Pinned provider settings no longer match. Restore them or Discard.",
        )

        assert store.discard_provider_continuation(
            owner.id,
            expected_message_version=version,
        )
        assert store.provider_continuation_recovery_message(session.id) is None
    finally:
        database.close_connection()


async def test_resume_target_mismatch_blocks_before_bridge_or_tool() -> None:
    database, store, _session, owner = _store_with_checkpoint(content="Visible")

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, _checkpoint):
            return []

        async def resolve_for_send(self, selection):
            self.calls += 1
            return provider_resolution(
                       ready=True,
                       provider="ZAI",
                       model="glm-5",
                       base_url="https://api.z.ai/v1",
                       api_mode="chat_completions",
                   )

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_bridge=object()
    )
    try:
        assert controller.provider_continuation_replay_available()
        assert not ConsoleChatController(
            store=store,
            provider_gateway=gateway,
        ).provider_continuation_replay_available()
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        assert gateway.calls == 1
        assert store.get_message(owner.id).provider_continuation is not None
    finally:
        database.close_connection()


async def test_resume_without_translator_sets_specific_unavailable_warning() -> None:
    database, store, _session, owner = _store_with_checkpoint(content="Visible")

    class Gateway:
        async def resolve_for_send(self, _selection):
            return provider_resolution(
                       ready=True,
                       provider="Moonshot",
                       model="kimi-k2",
                       base_url="https://api.moonshot.ai/v1",
                       api_mode="chat_completions",
                   )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=Gateway(),
        agent_bridge=object(),
    )
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        assert not controller.provider_continuation_replay_available()
        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        assert store.get_message(owner.id).provider_continuation_warning == (
            "Continuation replay support is not enabled for this provider integration. "
            "Enable or configure it, or Discard the interrupted run."
        )
    finally:
        database.close_connection()


async def test_live_continuation_owner_rejects_every_recovery_action() -> None:
    """A checkpoint written by the current run is not an interruption yet."""
    database, store, session, owner = _store_with_checkpoint(content="Visible")

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, _checkpoint):
            raise AssertionError("live continuation must not be translated")

        async def resolve_for_send(self, _selection):
            self.calls += 1
            raise AssertionError("live continuation must not reach the provider")

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=object(),
    )
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        controller._active_assistant_message_ids[session.id] = owner.id
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."),
            session_id=session.id,
        )

        for action in ("resume", "take_over", "discard"):
            assert not await controller.recover_provider_continuation(
                action, owner.id, version
            )

        assert gateway.calls == 0
        assert store.get_message(owner.id).provider_continuation is not None
        assert "still active" in (
            store.get_message(owner.id).provider_continuation_warning or ""
        )
        assert controller.send_refusal_copy(session.id) == (
            "A run is already running in this tab."
        )

        # A stale STREAMING stamp without the exact active owner is not liveness.
        controller._active_assistant_message_ids.pop(session.id)
        assert await controller.recover_provider_continuation(
            "discard", owner.id, version
        )
    finally:
        database.close_connection()


async def test_resume_excludes_visible_owner_and_reports_failed_completion(
    monkeypatch,
) -> None:
    """The canonical restore owns the interrupted assistant turn exactly once."""
    database, store, session, owner = _store_with_checkpoint(
        content="VISIBLE-OWNER-MUST-NOT-BE-DUPLICATED"
    )
    translated = {"role": "assistant", "content": "CANONICAL-OWNER-ONCE"}

    class Gateway:
        def expand_provider_continuation(self, _checkpoint):
            return [translated]

        async def resolve_for_send(self, _selection):
            return provider_resolution(
                       ready=True,
                       provider="Moonshot",
                       model="kimi-k2",
                       base_url="https://api.moonshot.ai/v1",
                       api_mode="chat_completions",
                   )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=Gateway(),
        agent_bridge=object(),
    )
    captured: list[dict] = []

    async def fail_after_acceptance(**kwargs):
        captured.extend(kwargs["provider_messages"])
        captured.extend(
            kwargs["expand_provider_continuation"](
                kwargs["restore_provider_continuation"]
            )
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, "Provider request failed."),
            session_id=session.id,
        )
        return ConsoleSubmitResult(True, True, "Provider request failed.")

    monkeypatch.setattr(controller, "_run_agent_reply", fail_after_acceptance)
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        assert all(
            row.get("content") != "VISIBLE-OWNER-MUST-NOT-BE-DUPLICATED"
            for row in captured
        )
        assert captured.count(translated) == 1
        recovered = store.get_message(owner.id)
        assert recovered.provider_continuation is not None
        assert "failed" in (recovered.provider_continuation_warning or "").lower()
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    ("provider", "model", "protocol", "base_url", "tool_bearing", "expected_prior"),
    [
        (
            "Moonshot",
            "kimi-k3",
            "chat_completions",
            "https://api.moonshot.ai/v1",
            False,
            True,
        ),
        (
            "DeepSeek",
            "deepseek-v4-flash",
            "responses",
            "https://api.deepseek.com/v1",
            True,
            True,
        ),
        (
            "ZAI",
            "glm-5",
            "chat_completions",
            "https://api.z.ai/v1",
            True,
            False,
        ),
        # TASK-19170: keep_all is a versioned-kimi-family policy, not a
        # kimi-k3 literal -- k2.6 returns reasoning_content on the wire
        # (chatcmpl-6a8768a9b5c429b466fbc42f), so its complete checkpoints
        # (tool-bearing or reasoning-only) must survive a resume.
        (
            "Moonshot",
            "kimi-k2",
            "chat_completions",
            "https://api.moonshot.ai/v1",
            True,
            True,
        ),
        (
            "Moonshot",
            "kimi-k2.6",
            "chat_completions",
            "https://api.moonshot.ai/v1",
            False,
            True,
        ),
        # kimi-latest returns no reasoning_content
        # (chatcmpl-6a8768a616ceb0c0ae780f2c): stays outside the retention
        # policy.
        (
            "Moonshot",
            "kimi-latest",
            "chat_completions",
            "https://api.moonshot.ai/v1",
            True,
            False,
        ),
    ],
)
async def test_resume_forwards_only_policy_retained_prior_complete_sidecars(
    monkeypatch,
    provider: str,
    model: str,
    protocol: str,
    base_url: str,
    tool_bearing: bool,
    expected_prior: bool,
) -> None:
    database = CharactersRAGDB(":memory:", "resume-history-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="Resume history")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="initial request",
            persist=True,
        )
        prior = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="prior visible answer",
            persist=True,
        )
        prior_round = ContinuationRound(
            assistant_content="prior visible answer",
            reasoning_blocks=("PRIVATE-PRIOR-REASONING",),
            calls=(
                (
                    ContinuationCall(
                        call_id="prior_call",
                        name="calculator",
                        arguments='{"expression":"1+1"}',
                        state="completed",
                        result=ContinuationResult("2"),
                    ),
                )
                if tool_bearing
                else ()
            ),
        )
        prior_checkpoint = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider=provider.lower(),
            protocol=protocol,
            model=model,
            api_base_url=base_url,
            state="complete",
            rounds=(prior_round,),
        )
        prior_live = store._message_or_raise(prior.id)
        prior_persisted = prior_live.persisted_message_id
        assert prior_persisted is not None
        database.update_provider_continuation(
            message_id=prior_persisted,
            expected_message_version=1,
            provider_continuation_json=dump_provider_continuation_json(
                prior_checkpoint
            ),
            content="prior visible answer",
        )
        prior_live.provider_continuation = prior_checkpoint
        prior_live.provider_continuation_message_version = 2
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="continue",
            persist=True,
        )

        active = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider=provider.lower(),
            protocol=protocol,
            model=model,
            api_base_url=base_url,
            state="active",
            rounds=(
                ContinuationRound(
                    assistant_content="",
                    reasoning_blocks=("PRIVATE-ACTIVE",),
                    calls=(
                        ContinuationCall(
                            call_id="active_call",
                            name="calculator",
                            arguments='{"expression":"2+2"}',
                            state="pending",
                        ),
                    ),
                ),
            ),
        )
        owner = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="active visible",
            persist=True,
        )
        owner_live = store._message_or_raise(owner.id)
        owner_persisted = owner_live.persisted_message_id
        assert owner_persisted is not None
        database.update_provider_continuation(
            message_id=owner_persisted,
            expected_message_version=1,
            provider_continuation_json=dump_provider_continuation_json(active),
            content="active visible",
        )
        owner_live.provider_continuation = active
        owner_live.provider_continuation_message_version = 2

        class Gateway:
            def expand_provider_continuation(self, _checkpoint):
                return [{"role": "assistant", "content": "active translated"}]

            async def resolve_for_send(self, _selection):
                return provider_resolution(
                           ready=True,
                           provider=provider,
                           model=model,
                           base_url=base_url,
                           api_mode=protocol,
                       )

        controller = ConsoleChatController(
            store=store, provider_gateway=Gateway(), agent_bridge=object()
        )
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            store._message_or_raise(owner.id).provider_continuation = None
            return ConsoleSubmitResult(True, True, "done")

        monkeypatch.setattr(controller, "_run_agent_reply", capture)

        assert await controller.recover_provider_continuation(
            "resume", owner.id, expected_message_version=2
        )
        sidecar = captured.get("continuation_sidecar", ())
        assert [item.owner_message_id for item in sidecar] == (
            [prior.id] if expected_prior else []
        )
        assert captured.get("continuation_history_target") == (
            captured["restore_provider_target"] if expected_prior else None
        )
        provider_rows = captured["provider_messages"]
        assert sum(row.get("content") == "prior visible answer" for row in provider_rows) == 1
        assert all(row.get("content") != "active visible" for row in provider_rows)
    finally:
        database.close_connection()


async def test_concurrent_resume_is_serialized_at_controller_session_boundary(
    monkeypatch,
) -> None:
    database, store, session, owner = _store_with_checkpoint(content="Visible")
    resolving = asyncio.Event()
    release = asyncio.Event()

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, _checkpoint):
            return []

        async def resolve_for_send(self, _selection):
            self.calls += 1
            resolving.set()
            await release.wait()
            return provider_resolution(
                       ready=True,
                       provider="Moonshot",
                       model="kimi-k2",
                       base_url="https://api.moonshot.ai/v1",
                       api_mode="chat_completions",
                   )

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=object(),
    )

    async def complete_checkpoint(**_kwargs):
        current_version = store.get_message(
            owner.id
        ).provider_continuation_message_version
        assert current_version is not None
        store.discard_provider_continuation(
            owner.id, expected_message_version=current_version
        )
        return ConsoleSubmitResult(True, True, "done")

    monkeypatch.setattr(controller, "_run_agent_reply", complete_checkpoint)
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        first = asyncio.create_task(
            controller.recover_provider_continuation("resume", owner.id, version)
        )
        await asyncio.wait_for(resolving.wait(), timeout=1)
        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        assert gateway.calls == 1
        release.set()
        assert await first
        assert session.id not in controller._provider_continuation_recovery_sessions
    finally:
        database.close_connection()


@pytest.mark.parametrize("action", ["discard", "resume"])
async def test_stale_variant_recovery_rejects_inactive_owner_before_side_effects(
    monkeypatch,
    action: str,
) -> None:
    """A callout from branch A cannot recover it after branch B becomes active."""
    database, store, _session, owner = _store_with_checkpoint(content="Branch A")

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, _checkpoint):
            return []

        async def resolve_for_send(self, _selection):
            self.calls += 1
            return provider_resolution(
                       ready=True,
                       provider="Moonshot",
                       model="kimi-k2",
                       base_url="https://api.moonshot.ai/v1",
                       api_mode="chat_completions",
                   )

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=object(),
    )
    bridge_calls = 0

    async def unexpected_runtime(**_kwargs):
        nonlocal bridge_calls
        bridge_calls += 1
        return ConsoleSubmitResult(True, True, "unexpected")

    monkeypatch.setattr(controller, "_run_agent_reply", unexpected_runtime)
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        persisted_id = store.get_message(owner.id).persisted_message_id
        assert version is not None
        assert persisted_id is not None
        sibling = store.create_sibling(
            owner.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Branch B",
            persist=True,
        )

        assert store.active_leaf(store.active_session_id or "") == sibling.id
        assert not await controller.recover_provider_continuation(
            action, owner.id, version
        )
        row = database.get_message_by_id(persisted_id)
        assert row is not None
        assert row["version"] == version
        assert row["provider_continuation_json"] is not None
        assert store.get_message(owner.id).provider_continuation is not None
        assert gateway.calls == 0
        assert bridge_calls == 0
    finally:
        database.close_connection()


@pytest.mark.parametrize("ready", [False, True])
async def test_resume_revalidates_active_variant_after_async_resolution(
    monkeypatch,
    ready: bool,
) -> None:
    """A branch switch during readiness resolution cannot enter the runtime."""
    database, store, session, owner = _store_with_checkpoint(content="Branch A")
    sibling = store.create_sibling(
        owner.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Branch B",
        persist=True,
    )
    store.set_active_leaf(session.id, owner.id)

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, _checkpoint):
            return []

        async def resolve_for_send(self, _selection):
            self.calls += 1
            store.set_active_leaf(session.id, sibling.id)
            await asyncio.sleep(0)
            return provider_resolution(
                       ready=ready,
                       provider="Moonshot",
                       model="kimi-k2",
                       base_url="https://api.moonshot.ai/v1",
                       api_mode="chat_completions",
                   )

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=object(),
    )
    runtime_calls = 0

    async def unexpected_runtime(**_kwargs):
        nonlocal runtime_calls
        runtime_calls += 1
        return ConsoleSubmitResult(True, True, "unexpected")

    monkeypatch.setattr(controller, "_run_agent_reply", unexpected_runtime)
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        persisted_id = store.get_message(owner.id).persisted_message_id
        assert version is not None
        assert persisted_id is not None
        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        row = database.get_message_by_id(persisted_id)
        assert row is not None
        assert row["version"] == version
        assert row["provider_continuation_json"] is not None
        stale_owner = store.get_message(owner.id)
        assert stale_owner.provider_continuation is not None
        assert stale_owner.provider_continuation_warning is None
        assert gateway.calls == 1
        assert runtime_calls == 0
    finally:
        database.close_connection()


async def test_recovery_rejects_stale_persisted_owner_version_before_resolution() -> (
    None
):
    database, store, _session, owner = _store_with_checkpoint(content="Visible")

    class Gateway:
        calls = 0

        def expand_provider_continuation(self, _checkpoint):
            return []

        async def resolve_for_send(self, _selection):
            self.calls += 1
            raise AssertionError("stale durable state must not reach resolution")

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=object(),
    )
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        persisted_id = store.get_message(owner.id).persisted_message_id
        assert version is not None
        assert persisted_id is not None
        row = database.get_message_by_id(persisted_id)
        assert row is not None
        database.update_provider_continuation(
            message_id=persisted_id,
            expected_message_version=version,
            provider_continuation_json=row["provider_continuation_json"],
        )

        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        durable = database.get_message_by_id(persisted_id)
        assert durable is not None
        assert durable["version"] == version + 1
        assert durable["provider_continuation_json"] is not None
        assert store.get_message(owner.id).provider_continuation is not None
        assert gateway.calls == 0
    finally:
        database.close_connection()


def test_continuation_warning_clears_only_after_successful_durability(
    monkeypatch,
) -> None:
    database, store, _session, owner = _store_with_checkpoint(content="Visible")
    warning = "Recovery failed. The interrupted run is unchanged."
    context = ContinuationEventContext(owner.id, "run", "primary", "persistent")
    store.set_provider_continuation_warning(owner.id, warning)
    monkeypatch.setattr(
        store,
        "ensure_provider_continuation_durable",
        lambda **_kwargs: ContinuationDurabilityResult(False, "safe failure"),
    )
    try:
        with pytest.raises(RuntimeError, match="safe failure"):
            store.persist_provider_continuation_event(
                ToolCallExecuting(context, "PRIVATE-CALL-ID", 1)
            )
        assert store.get_message(owner.id).provider_continuation_warning == warning

        monkeypatch.setattr(
            store,
            "ensure_provider_continuation_durable",
            lambda **_kwargs: ContinuationDurabilityResult(True, "durable"),
        )
        store.persist_provider_continuation_event(
            ToolCallFinished(
                context,
                "PRIVATE-CALL-ID",
                2,
                "completed",
                ContinuationResult("PRIVATE-RESULT-CANARY"),
            )
        )
        assert store.get_message(owner.id).provider_continuation_warning is None
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    ("status", "visible_copy", "expected_warning"),
    [
        (ConsoleRunStatus.FAILED, "Provider request failed.", "Recovery failed"),
        (ConsoleRunStatus.FAILED, "Response stopped/cancelled.", "Recovery failed"),
        (ConsoleRunStatus.STOPPED, "Response stopped.", "Recovery stopped"),
    ],
)
async def test_accepted_recovery_is_false_while_checkpoint_remains_active(
    monkeypatch,
    status: ConsoleRunStatus,
    visible_copy: str,
    expected_warning: str,
) -> None:
    database, store, session, owner = _store_with_checkpoint(content="Visible")

    class Gateway:
        def expand_provider_continuation(self, _checkpoint):
            return []

        async def resolve_for_send(self, _selection):
            return provider_resolution(
                       ready=True,
                       provider="Moonshot",
                       model="kimi-k2",
                       base_url="https://api.moonshot.ai/v1",
                       api_mode="chat_completions",
                   )

    controller = ConsoleChatController(
        store=store,
        provider_gateway=Gateway(),
        agent_bridge=object(),
    )

    async def accepted_but_incomplete(**_kwargs):
        controller._active_assistant_message_ids[session.id] = owner.id
        controller._active_stream_tasks[session.id] = asyncio.current_task()
        controller._set_run_state(
            ConsoleRunState(status, visible_copy), session_id=session.id
        )
        return ConsoleSubmitResult(True, True, visible_copy)

    monkeypatch.setattr(controller, "_run_agent_reply", accepted_but_incomplete)
    try:
        version = store.get_message(owner.id).provider_continuation_message_version
        assert version is not None
        assert not await controller.recover_provider_continuation(
            "resume", owner.id, version
        )
        assert expected_warning in (
            store.get_message(owner.id).provider_continuation_warning or ""
        )
        assert not controller.provider_continuation_owner_is_live(owner.id)
        assert session.id not in controller._active_assistant_message_ids
    finally:
        database.close_connection()


async def test_new_turn_is_blocked_without_provider_or_tool_dispatch() -> None:
    database, store, session, _owner = _store_with_checkpoint(content="Visible")

    class Gateway:
        calls = 0

        async def resolve_for_send(self, selection):
            self.calls += 1
            raise AssertionError("interrupted work must be recovered explicitly")

    gateway = Gateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    try:
        result = await controller.submit_draft("new text", session_id=session.id)
        assert not result.accepted
        assert result.visible_copy == (
            "Recover the interrupted tool run before sending a new message: "
            "Resume or Discard it first."
        )
        assert gateway.calls == 0
    finally:
        database.close_connection()


def test_ephemeral_event_writes_no_checkpoint_and_offers_no_recovery() -> None:
    database = CharactersRAGDB(":memory:", "console-continuation-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(database))
        session = store.create_session(title="Temporary", ephemeral=True)
        owner = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        store.persist_provider_continuation_event(
            ToolBatchReady(
                ContinuationEventContext(owner.id, "run", "primary", "ephemeral"),
                _active_checkpoint(),
                None,
            )
        )
        assert store.get_message(owner.id).provider_continuation is None
        assert store.interrupted_provider_continuation_message(session.id) is None
        assert (
            database.get_messages_for_conversation(
                session.persisted_conversation_id or "missing"
            )
            == []
        )
    finally:
        database.close_connection()
