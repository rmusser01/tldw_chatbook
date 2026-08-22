import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyDefaults,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


def _defaults(
    *, automatic: bool = True, allowed: bool = True
) -> ConsoleLibraryPolicyDefaults:
    return ConsoleLibraryPolicyDefaults(
        auto_retrieve=(
            ConsoleAutoRetrieve.AUTOMATIC if automatic else ConsoleAutoRetrieve.NEVER
        ),
        assistant_access=(
            ConsoleAssistantLibraryAccess.ALLOWED
            if allowed
            else ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )


def test_new_session_captures_current_defaults_without_following_later_changes():
    store = ConsoleChatStore(library_policy_defaults=_defaults())
    first = store.create_session()

    store.set_library_policy_defaults(_defaults(automatic=False, allowed=False))
    second = store.create_session()

    assert first.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert (
        first.library_policy_holder.snapshot.assistant_access
        is ConsoleAssistantLibraryAccess.ALLOWED
    )
    assert second.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert (
        second.library_policy_holder.snapshot.assistant_access
        is ConsoleAssistantLibraryAccess.BLOCKED
    )


def test_first_persistence_inserts_even_unedited_policy_and_publishes_after_commit(
    tmp_path, monkeypatch
):
    db = CharactersRAGDB(tmp_path / "first-policy.db", "policy-test")
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=service, library_policy_defaults=_defaults())
    session = store.create_session(title="Atomic policy")
    original_insert = service.console_library_policy_repository.insert
    original_publish = store.publish_committed_identity
    publish_observations = []
    attempts = 0

    def fail_once(conversation_id, candidate):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected first-persistence policy failure")
        return original_insert(conversation_id, candidate)

    monkeypatch.setattr(
        service.console_library_policy_repository,
        "insert",
        fail_once,
    )

    def observe_publish(session_id, identity):
        publish_observations.append(
            (
                "before",
                db.get_connection().in_transaction,
                session.persisted_conversation_id,
                session.title,
                identity.conversation_id,
                identity.title,
            )
        )
        original_publish(session_id, identity)
        publish_observations.append(
            (
                "after",
                db.get_connection().in_transaction,
                session.persisted_conversation_id,
                session.title,
                identity.conversation_id,
                identity.title,
            )
        )

    monkeypatch.setattr(store, "publish_committed_identity", observe_publish)

    with pytest.raises(
        RuntimeError, match="injected first-persistence policy failure"
    ):
        store.persist_session_if_needed(session.id)

    assert session.persisted_conversation_id is None
    assert session.title == "Atomic policy"
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM conversations"
    ).fetchone()[0] == 0
    assert publish_observations == []

    conversation_id = store.persist_session_if_needed(session.id)

    assert conversation_id is not None
    row = service.console_library_policy_repository.read(conversation_id)
    assert row.snapshot.source == "durable"
    assert row.snapshot.policy_revision == 1
    assert row.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert row.snapshot.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
    assert session.library_policy_holder.snapshot == row.snapshot
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM conversations"
    ).fetchone()[0] == 1
    assert publish_observations == [
        (
            "before",
            False,
            None,
            "Atomic policy",
            conversation_id,
            "Atomic policy",
        ),
        (
            "after",
            False,
            conversation_id,
            "Atomic policy",
            conversation_id,
            "Atomic policy",
        ),
    ]


def test_restored_missing_policy_is_fail_closed_and_write_free_until_explicit_save(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "missing-policy.db", "policy-test")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(conversation_title="Legacy")
    store = ConsoleChatStore(persistence=service, library_policy_defaults=_defaults())

    session = store.restore_persisted_session(
        title="Legacy",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    asyncio.run(store.hydrate_session_library_policy(session.id))

    snapshot = session.library_policy_holder.snapshot
    assert snapshot.source == "missing"
    assert snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert service.console_library_policy_repository.read(conversation_id).durable_policy is None

    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    result = asyncio.run(store.save_session_library_policy(session.id))

    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert result.snapshot.policy_revision == 1


def test_committed_save_publishes_to_sibling_holders_and_close_unregisters(tmp_path):
    db = CharactersRAGDB(tmp_path / "policy-publication.db", "policy-test")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(conversation_title="Shared")
    assert (
        service.console_library_policy_repository.insert(
            conversation_id,
            ConsoleLibraryPolicyCandidate(
                ConsoleAutoRetrieve.NEVER,
                ConsoleAssistantLibraryAccess.BLOCKED,
            ),
        ).status
        is ConsoleLibraryPolicyWriteStatus.COMMITTED
    )
    store = ConsoleChatStore(persistence=service, library_policy_defaults=_defaults())
    first = store.restore_persisted_session(
        title="Shared",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    second = store.restore_persisted_session(
        title="Shared again",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    asyncio.run(store.hydrate_session_library_policy(first.id))
    asyncio.run(store.hydrate_session_library_policy(second.id))
    store.stage_session_library_policy(
        first.id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )

    result = asyncio.run(store.save_session_library_policy(first.id))

    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert second.library_policy_holder.snapshot == result.snapshot
    store.close_session(first.id)
    assert first.id not in store.library_policy_coordinator._holders
    assert second.id in store.library_policy_coordinator._holders


def test_every_store_creation_reads_current_defaults_without_mutating_existing():
    current = _defaults()

    def provider():
        return current

    store = ConsoleChatStore(library_policy_defaults_provider=provider)
    first = store.ensure_session()
    current = _defaults(automatic=False, allowed=False)
    second = store.create_session()

    assert first.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert second.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER


def test_real_runtime_store_resolves_settings_for_every_creation_entrypoint():
    app = SimpleNamespace(
        app_config={
            "chat_defaults": {"rag_auto_retrieve_on_send": True},
            "console": {"assistant_library_access_default": True},
        }
    )
    store = ConsoleRuntime(app).ensure_chat_store()
    first = store.ensure_session()
    app.app_config["chat_defaults"]["rag_auto_retrieve_on_send"] = False
    app.app_config["console"]["assistant_library_access_default"] = False
    second = store.create_session()

    assert first.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
    assert first.library_policy_holder.snapshot.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
    assert second.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert second.library_policy_holder.snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED


def test_rollback_created_session_unregisters_holder(tmp_path):
    db = CharactersRAGDB(tmp_path / "rollback-holder.db", "policy-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    settings = ConsoleSessionSettings(provider="openai")
    prior = store.create_session(
        settings=settings, canonical_settings_baseline=settings
    )
    created = store.create_session(
        settings=settings, canonical_settings_baseline=settings
    )

    assert store.rollback_created_pristine_session(
        created.id,
        expected_session=created,
        expected_settings=settings,
        prior_active_session_id=prior.id,
    )
    assert created.id not in store.library_policy_coordinator._holders


def test_restore_state_replaces_holder_bindings_without_stale_publication(tmp_path):
    db = CharactersRAGDB(tmp_path / "restore-holders.db", "policy-test")
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=service)
    stale = store.create_session(session_id="overlap")
    removed = store.create_session(session_id="removed")
    replacement = ConsoleChatSession(
        id="overlap",
        title="Replacement",
        persisted_conversation_id=service.create_conversation(
            conversation_title="Replacement"
        ),
    )

    store.restore_state(sessions=[replacement])

    registered = store.library_policy_coordinator._holders
    assert registered["overlap"].holder is store.sessions()[0].library_policy_holder
    assert registered["overlap"].holder is not stale.library_policy_holder
    assert registered["overlap"].conversation_id == replacement.persisted_conversation_id
    assert removed.id not in registered
    store.stage_session_library_policy(
        "overlap",
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    result = asyncio.run(store.save_session_library_policy("overlap"))
    assert result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED
    assert stale.library_policy_holder.snapshot.source == "new_session"
    assert store.sessions()[0].library_policy_holder.snapshot == result.snapshot


def test_restored_policy_starts_fail_closed_and_hydrates_off_loop(tmp_path, monkeypatch):
    db = CharactersRAGDB(tmp_path / "async-hydrate.db", "policy-test")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(conversation_title="Hydrate")
    service.console_library_policy_repository.insert(
        conversation_id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    store = ConsoleChatStore(persistence=service)
    caller_thread = threading.get_ident()
    read_threads = []
    original_read = service.console_library_policy_repository.read

    def recording_read(target):
        read_threads.append(threading.get_ident())
        return original_read(target)

    monkeypatch.setattr(service.console_library_policy_repository, "read", recording_read)
    session = store.restore_persisted_session(
        title="Hydrate",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )

    assert read_threads == []
    assert session.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    asyncio.run(store.hydrate_session_library_policy(session.id))
    assert read_threads and all(thread != caller_thread for thread in read_threads)
    assert session.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC


def test_blocked_hydration_yields_event_loop_and_rebinds_without_stale_publication(
    tmp_path, monkeypatch
):
    async def scenario():
        db = CharactersRAGDB(tmp_path / "hydrate-rebind.db", "policy-test")
        service = ChatPersistenceService(db)
        old_id = service.create_conversation(conversation_title="Old")
        new_id = service.create_conversation(conversation_title="New")
        repository = service.console_library_policy_repository
        repository.insert(
            old_id,
            ConsoleLibraryPolicyCandidate(
                ConsoleAutoRetrieve.AUTOMATIC,
                ConsoleAssistantLibraryAccess.ALLOWED,
            ),
        )
        repository.insert(
            new_id,
            ConsoleLibraryPolicyCandidate(
                ConsoleAutoRetrieve.NEVER,
                ConsoleAssistantLibraryAccess.BLOCKED,
            ),
        )
        store = ConsoleChatStore(persistence=service)
        old = store.restore_persisted_session(
            title="Old",
            workspace_id=None,
            persisted_conversation_id=old_id,
            all_nodes=(),
        )
        started = threading.Event()
        release = threading.Event()
        original_read = repository.read

        def blocking_old_read(conversation_id):
            if conversation_id == old_id:
                started.set()
                assert release.wait(5)
            return original_read(conversation_id)

        monkeypatch.setattr(repository, "read", blocking_old_read)
        hydration = asyncio.create_task(store.hydrate_session_library_policy(old.id))
        assert await asyncio.to_thread(started.wait, 2)
        heartbeat = False

        async def pulse():
            nonlocal heartbeat
            heartbeat = True

        await pulse()
        assert heartbeat
        replacement = ConsoleChatSession(
            id=old.id,
            title="New",
            persisted_conversation_id=new_id,
            library_policy_hydrated=False,
            library_policy_holder=ConsoleLibraryPolicyHolder(
                ConsoleLibraryPolicySnapshot(
                    ConsoleAutoRetrieve.NEVER,
                    ConsoleAssistantLibraryAccess.BLOCKED,
                    None,
                    "missing",
                )
            ),
        )
        store.restore_state(sessions=[replacement], active_session_id=old.id)
        release.set()
        await hydration

        current = store.sessions()[0]
        assert current.persisted_conversation_id == new_id
        assert current.library_policy_holder.snapshot.source == "durable"
        assert current.library_policy_holder.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
        assert current.library_policy_holder.snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
        assert current.library_policy_hydrated is False

    asyncio.run(scenario())
