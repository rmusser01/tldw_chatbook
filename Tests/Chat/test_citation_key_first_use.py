"""First-use custody of the local citation fingerprint key (TASK-33620.8).

Owner decision 2026-10-27: citation records persist by default and the key
is created in the system keyring on first use. The app's boot composition
must not touch the keyring; the first caller that needs the key acquires it,
and a key that saved records depend on is never silently replaced.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from tldw_chatbook.Chat.citation_key_first_use import (
    CitationKeyState,
    acquire_local_fingerprint_codec,
    ensure_fingerprint_key_off_loop,
)
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_service_factory import (
    build_local_citation_conversation_service,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CITATION_FINGERPRINT_KEYRING_SERVICE,
    KeyringCitationFingerprintKeyProvider,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    load_local_citation_identity_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

ON = CitationProvenanceRuntimePolicy(canonical_writes_enabled=True)
OFF = CitationProvenanceRuntimePolicy(canonical_writes_enabled=False)


class _Keyring:
    """A secure-classified keyring that records every call and its thread."""

    __module__ = "keyring.backends.macOS"
    priority = 5

    def __init__(self) -> None:
        self.secrets: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[str, int]] = []

    def get_password(self, service: str, account: str) -> str | None:
        self.calls.append(("get", threading.get_ident()))
        return self.secrets.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.calls.append(("set", threading.get_ident()))
        self.secrets[(service, account)] = value


class _NullKeyring(_Keyring):
    """What the test suite (and a keyring-less desktop) has: not secure."""

    __module__ = "keyring.backends.null"
    priority = -1


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "first-use.sqlite", client_id="first-use")
    yield database
    database.close_connection()


def _compose(db, tmp_path, backend, *, policy=ON, defer_key=True):
    _service, repository, _migration = build_local_citation_conversation_service(
        db,
        sidecar_path=tmp_path / "chat_rag_context.json",
        policy=policy,
        key_provider=KeyringCitationFingerprintKeyProvider(keyring_backend=backend),
        defer_key=defer_key,
    )
    return repository


def test_deferred_composition_touches_no_keyring_until_first_use(db, tmp_path):
    backend = _Keyring()

    repository = _compose(db, tmp_path, backend)

    assert backend.calls == []
    assert repository.fingerprint_key_pending is True
    assert repository.fingerprint_key_state == "pending"
    assert repository.local_citation_writes_ready is False

    assert repository.ensure_local_fingerprint_key() is True

    key_id = load_local_citation_identity_context(db).fingerprint_key_id
    assert (CITATION_FINGERPRINT_KEYRING_SERVICE, key_id) in backend.secrets
    assert repository.fingerprint_key_state == "ready"
    calls = len(backend.calls)
    assert repository.ensure_local_fingerprint_key() is True
    assert len(backend.calls) == calls  # one acquisition per repository


def test_first_use_reuses_the_key_a_previous_run_provisioned(db, tmp_path):
    backend = _Keyring()
    first = _compose(db, tmp_path, backend)
    assert first.ensure_local_fingerprint_key() is True
    secrets = dict(backend.secrets)

    restarted = _compose(db, tmp_path, backend)

    assert restarted.ensure_local_fingerprint_key() is True
    assert backend.secrets == secrets
    assert [kind for kind, _thread in backend.calls].count("set") == 1


def test_chatbook_ownership_wiring_composes_without_the_key(db, tmp_path, monkeypatch):
    """The app's second composition site defers the key as well.

    ``_wire_citation_artifact_ownership`` builds its own repository whenever
    it runs without one for the current database (the Chatbook wiring can
    run before the chat services are wired, or after the database changes).
    The mounted Console test pins the other site
    (``_wire_chat_conversation_services``), the one the constructor reaches.
    """
    import keyring

    from types import SimpleNamespace

    from tldw_chatbook import app_service_wiring

    backend = _Keyring()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(app_service_wiring, "get_user_data_dir", lambda: tmp_path)
    # Records on, as shipped, without reading this process's config file.
    monkeypatch.setattr(
        CitationProvenanceRuntimePolicy, "from_config", classmethod(lambda _cls: ON)
    )
    monkeypatch.setattr(
        app_service_wiring,
        "CitationArtifactOwnershipCoordinator",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    app = SimpleNamespace(
        chachanotes_db=db,
        local_chatbook_service=SimpleNamespace(
            set_citation_ownership_coordinator=lambda _coordinator: None
        ),
    )

    app_service_wiring.ServiceWiringMixin._wire_citation_artifact_ownership(app)

    assert app.citation_trace_repository.db is db
    assert app.citation_trace_repository.fingerprint_key_state == "pending"
    assert backend.calls == []


def test_eager_composition_keeps_acquiring_at_compose_time(db, tmp_path):
    """Chatbook export/import keep the factory's eager default."""
    backend = _Keyring()

    repository = _compose(db, tmp_path, backend, defer_key=False)

    assert backend.calls
    assert repository.local_citation_writes_ready is True


def test_records_off_never_reads_the_keyring(db, tmp_path):
    backend = _Keyring()
    repository = _compose(db, tmp_path, backend, policy=OFF)

    assert repository.ensure_local_fingerprint_key() is False
    assert backend.calls == []
    assert repository.fingerprint_key_pending is False
    assert repository.fingerprint_key_state == CitationKeyState.WRITES_DISABLED.value


def test_insecure_keyring_fails_closed_without_provisioning(db, tmp_path):
    backend = _NullKeyring()
    repository = _compose(db, tmp_path, backend)

    assert repository.ensure_local_fingerprint_key() is False
    assert backend.secrets == {}
    assert repository.fingerprint_key_state == (
        CitationKeyState.KEYRING_UNAVAILABLE.value
    )
    assert repository.fingerprint_key_pending is False
    assert (
        repository.create_local_trace_builder(
            request_id="request-1", generation_id="generation-1"
        )
        is None
    )


def test_missing_key_for_existing_records_is_never_replaced(db, tmp_path):
    identity = load_local_citation_identity_context(db)
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_payload_tombstones VALUES (
                ?, 'local_payload_v1', 'prior-payload', 'prior-scope',
                'source_revoked', 'policy-1',
                '2026-07-27T00:00:00Z', '2027-07-27T00:00:00Z'
            )
            """,
            (identity.profile_id,),
        )
    backend = _Keyring()
    repository = _compose(db, tmp_path, backend)

    assert repository.ensure_local_fingerprint_key() is False
    assert backend.secrets == {}
    assert repository.fingerprint_key_state == CitationKeyState.KEY_UNREADABLE.value


def test_unexpected_failure_fails_closed_and_is_retried(db, tmp_path, monkeypatch):
    backend = _Keyring()
    repository = _compose(db, tmp_path, backend)
    failures = iter([RuntimeError("database is locked")])

    def rows_exist():
        error = next(failures, None)
        if error is not None:
            raise error
        return False

    monkeypatch.setattr(repository, "fingerprint_bearing_rows_exist", rows_exist)

    assert repository.ensure_local_fingerprint_key() is False
    assert repository.fingerprint_key_state == CitationKeyState.UNAVAILABLE.value
    assert repository.fingerprint_key_pending is True
    assert repository.ensure_local_fingerprint_key() is True
    assert repository.fingerprint_key_state == CitationKeyState.READY.value


def test_acquire_reports_writes_disabled_before_any_keyring_call(db, tmp_path):
    backend = _Keyring()
    repository = _compose(db, tmp_path, backend, policy=OFF)

    codec, state = acquire_local_fingerprint_codec(
        repository, KeyringCitationFingerprintKeyProvider(keyring_backend=backend)
    )

    assert codec is None
    assert state is CitationKeyState.WRITES_DISABLED
    assert backend.calls == []


def test_off_loop_acquisition_runs_on_a_worker_for_a_file_database(db, tmp_path):
    backend = _Keyring()
    repository = _compose(db, tmp_path, backend)
    loop_thread = threading.get_ident()

    asyncio.run(ensure_fingerprint_key_off_loop(repository))

    assert repository.local_citation_writes_ready is True
    assert backend.calls
    assert all(thread != loop_thread for _kind, thread in backend.calls)


def test_off_loop_acquisition_is_a_no_op_once_settled(db, tmp_path):
    backend = _Keyring()
    repository = _compose(db, tmp_path, backend)
    assert repository.ensure_local_fingerprint_key() is True
    calls = len(backend.calls)

    asyncio.run(ensure_fingerprint_key_off_loop(repository))
    asyncio.run(ensure_fingerprint_key_off_loop(None))

    assert len(backend.calls) == calls


def test_keyring_never_runs_inside_a_write_transaction(db, tmp_path):
    transactions = []

    class ObservedKeyring(_Keyring):
        __module__ = "keyring.backends.macOS"

        def get_password(self, service, account):
            connection = getattr(db._local, "conn", None)
            transactions.append(bool(connection and connection.in_transaction))
            return super().get_password(service, account)

        def set_password(self, service, account, value):
            connection = getattr(db._local, "conn", None)
            transactions.append(bool(connection and connection.in_transaction))
            return super().set_password(service, account, value)

    backend = ObservedKeyring()
    repository = _compose(db, tmp_path, backend)
    old_id = repository.identity_context.fingerprint_key_id
    assert repository.ensure_local_fingerprint_key()
    assert transactions and not any(transactions)
    assert repository.identity_context.fingerprint_key_id != old_id
    assert (CITATION_FINGERPRINT_KEYRING_SERVICE, old_id) not in backend.secrets


@pytest.mark.asyncio
async def test_cancelled_count_wait_does_not_drain_interactive_keyring(db, tmp_path):
    from tldw_chatbook.UI.Console_Modules.citation_counts import read_citation_counts

    entered, release = threading.Event(), threading.Event()

    class HeldKeyring(_Keyring):
        __module__ = "keyring.backends.macOS"

        def get_password(self, service, account):
            entered.set()
            assert release.wait(10)
            return super().get_password(service, account)

    repository = _compose(db, tmp_path, HeldKeyring())
    original_identity = repository.identity_context
    read = []
    task = asyncio.create_task(
        read_citation_counts(
            repository,
            lambda *_: read.append(True) or {},
            (),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.15)
        assert task in done, "Footer drain still owns the interactive keyring wait"
        assert task.cancelled()
        assert read == []
        assert repository.fingerprint_key_pending
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        operation = getattr(repository, "_key_acquisition", None)
        if operation is not None:
            await asyncio.to_thread(operation.thread.join, 5)
            assert not operation.thread.is_alive()
    assert operation.future.done() and not operation.future.cancelled()
    assert repository.identity_context == original_identity
    assert load_local_citation_identity_context(db) == original_identity
    assert repository.fingerprint_key_pending
    await ensure_fingerprint_key_off_loop(repository)
    assert repository._key_acquisition is operation
    assert repository.local_citation_writes_ready


@pytest.mark.asyncio
async def test_cancellation_during_issued_cas_drains_its_actual_commit(
    db, tmp_path, monkeypatch
):
    from tldw_chatbook.Chat import citation_key_first_use as first_use

    repository = _compose(db, tmp_path, _Keyring())
    original = repository.identity_context
    entered, release = threading.Event(), threading.Event()
    publish = first_use._publish
    committed = []

    def held_publish(*args):
        entered.set()
        assert release.wait(10)
        result = publish(*args)
        committed.append(load_local_citation_identity_context(db))
        return result

    monkeypatch.setattr(first_use, "_publish", held_publish)
    task = asyncio.create_task(ensure_fingerprint_key_off_loop(repository))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(3):
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.03)
            assert task not in done
        assert not committed
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert committed == [repository.identity_context]
    assert repository.identity_context != original
    assert load_local_citation_identity_context(db) == repository.identity_context
    assert repository.local_citation_writes_ready


def test_pending_key_edit_rolls_back_instead_of_invalidating_owner(db, tmp_path):
    from Tests.Chat.test_console_chat_fork_persistence import _active_citation_snapshot
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.citation_trace_repository import (
        CitationPersistenceUnavailable,
    )

    _active_citation_snapshot(db)
    repository = _compose(db, tmp_path, _Keyring())
    service = ChatPersistenceService(db, citation_repository=repository)
    before = dict(db.get_message_by_id("source-assistant"))
    owners = [
        tuple(row)
        for row in db.get_connection().execute("SELECT * FROM rag_message_trace_owners")
    ]
    with pytest.raises(CitationPersistenceUnavailable, match="fingerprint_key_pending"):
        service.update_message_content(
            message_id="source-assistant",
            content="Answer [S1].",
            image_data=None,
            image_mime_type=None,
        )
    assert dict(db.get_message_by_id("source-assistant")) == before
    assert [
        tuple(row)
        for row in db.get_connection().execute("SELECT * FROM rag_message_trace_owners")
    ] == owners


@pytest.mark.asyncio
async def test_concurrent_waiters_share_one_key_operation_despite_cancellation(
    db, tmp_path
):
    entered, release = threading.Event(), threading.Event()

    class HeldKeyring(_Keyring):
        __module__ = "keyring.backends.macOS"

        def get_password(self, service, account):
            entered.set()
            assert release.wait(10)
            return super().get_password(service, account)

    backend = HeldKeyring()
    repository = _compose(db, tmp_path, backend)
    first = asyncio.create_task(ensure_fingerprint_key_off_loop(repository))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        operation = repository._key_acquisition
        second = asyncio.create_task(ensure_fingerprint_key_off_loop(repository))
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert not operation.future.done() and operation.thread.is_alive()
        release.set()
        await second
        assert repository._key_acquisition is operation
        assert repository.local_citation_writes_ready
        assert [kind for kind, _ in backend.calls].count("set") == 1
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (first, second) if task), return_exceptions=True
        )
        await asyncio.to_thread(repository._key_acquisition.thread.join, 5)
