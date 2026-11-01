"""Real-process publication orderings; no OS keyring or user profile access."""

import asyncio
import base64
import subprocess
import sys
import threading

import pytest

from Tests.Chat.test_citation_key_first_use import _Keyring, _compose
from Tests.Chat.test_citation_trace_repository import _repository
from tldw_chatbook.Chat.citation_key_first_use import ensure_fingerprint_key_off_loop
from tldw_chatbook.Chat.citation_trace_identity import (
    CITATION_FINGERPRINT_KEYRING_SERVICE,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    load_local_citation_identity_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_STALE_WRITER = r"""
import sys
from Tests.windows_private_fixture_runner import user_fixture_default_owner
with user_fixture_default_owner():
    from Tests.Chat.test_citation_trace_repository import _repository, _sealed_write
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Chat.citation_trace_repository import CitationPersistenceUnavailable
    db = CharactersRAGDB(sys.argv[1], client_id="independent-writer")
    repository = _repository(db)
    prepared = repository.prepare_write(_sealed_write())
    print("READY", flush=True)
    assert sys.stdin.readline().strip() == "write"
    try:
        with db.transaction(immediate=True) as cursor:
            db.add_conversation({"id": "race-chat", "title": "Race"})
            db.add_message({"id": "race-reply", "conversation_id": "race-chat",
                            "sender": "assistant", "content": "Answer [S1].",
                            "client_id": db.client_id})
            repository.write_prepared(cursor, prepared, message_id="race-reply",
                                      message_revision=1, message_body="Answer [S1].")
        active = repository.get_active_trace_for_current_message("race-reply", "Answer [S1].")
        assert repository.verify_active_trace_result(active)
        print("COMMITTED", flush=True)
    except CitationPersistenceUnavailable as exc:
        assert str(exc) == "citation_identity_context_mismatch", str(exc)
        print("STALE", flush=True)
    finally:
        db.close()
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_first", [True, False])
async def test_independent_loaded_writer_and_fresh_key_cas_orderings(
    tmp_path, writer_first
):
    path = tmp_path / "race.sqlite"
    db = CharactersRAGDB(path, client_id="publisher")
    old = load_local_citation_identity_context(db)
    entered, release = threading.Event(), threading.Event()

    class HeldProvision(_Keyring):
        __module__ = "keyring.backends.macOS"

        def set_password(self, service, account, value):
            super().set_password(service, account, value)
            entered.set()
            assert release.wait(15)

    backend = HeldProvision()
    repo = _compose(db, tmp_path, backend)
    process = task = None
    with (tmp_path / "writer.stderr").open("w+") as errors:
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-X", "utf8", "-c", _STALE_WRITER, str(path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
            )
            assert await asyncio.to_thread(process.stdout.readline) == "READY\n"
            task = asyncio.create_task(ensure_fingerprint_key_off_loop(repo))
            assert await asyncio.to_thread(entered.wait, 10)
            # A separate process commits while the real provider is held.
            if writer_first:
                process.stdin.write("write\n")
                process.stdin.flush()
                assert await asyncio.to_thread(process.stdout.readline) == "COMMITTED\n"
            release.set()
            await task
            adopted = load_local_citation_identity_context(db)
            if writer_first:
                assert adopted == old
                assert repo.fingerprint_key_state == "key_unreadable"
                reopened = _repository(db)
                active = reopened.get_active_trace_for_current_message(
                    "race-reply", "Answer [S1]."
                )
                assert reopened.verify_active_trace_result(active)
            else:
                assert adopted.fingerprint_key_id != old.fingerprint_key_id
                assert repo.identity_context == adopted
                assert repo.local_citation_writes_ready
                process.stdin.write("write\n")
                process.stdin.flush()
                assert await asyncio.to_thread(process.stdout.readline) == "STALE\n"
                assert db.get_message_by_id("race-reply") is None
                assert not repo.fingerprint_bearing_rows_exist()
            assert (
                CITATION_FINGERPRINT_KEYRING_SERVICE,
                old.fingerprint_key_id,
            ) not in backend.secrets
            assert await asyncio.to_thread(process.wait, 10) == 0
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if repo._key_acquisition is not None:
                await asyncio.to_thread(repo._key_acquisition.thread.join, 5)
                assert not repo._key_acquisition.thread.is_alive()
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                process.stdin.close()
                process.stdout.close()
            db.close()


@pytest.mark.asyncio
async def test_original_key_with_protected_owner_remains_valid_after_deferred_load(
    tmp_path,
):
    from Tests.Chat.test_console_chat_fork_persistence import (
        _active_citation_snapshot,
        _commit,
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    db = CharactersRAGDB(tmp_path / "existing.sqlite", client_id="existing")
    try:
        snapshot, _ = _active_citation_snapshot(db)
        identity = load_local_citation_identity_context(db)
        backend = _Keyring()
        backend.secrets[
            (CITATION_FINGERPRINT_KEYRING_SERVICE, identity.fingerprint_key_id)
        ] = base64.b64encode(b"k" * 32).decode()
        repo = _compose(db, tmp_path, backend)
        await ensure_fingerprint_key_off_loop(repo)
        assert repo.identity_context == identity and repo.local_citation_writes_ready
        assert all(kind != "set" for kind, _ in backend.calls)
        service = ChatPersistenceService(db, citation_repository=repo)
        _commit(service, snapshot)
        service.update_message_content(
            message_id="source-assistant",
            content="Answer [S1].",
            image_data=None,
            image_mime_type=None,
        )
        reopened = _compose(db, tmp_path, backend)
        await ensure_fingerprint_key_off_loop(reopened)
        for message_id in ("source-assistant", "fork-assistant"):
            active = reopened.get_active_trace_for_current_message(
                message_id, "Answer [S1]."
            )
            assert reopened.verify_active_trace_result(active)
            assert active.summary.namespace.trace_id == "trace-1"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_two_process_provisioners_keep_the_winning_identity(tmp_path):
    db = CharactersRAGDB(tmp_path / "contenders.sqlite", client_id="first")
    entered, release = threading.Event(), threading.Event()

    class HeldProvision(_Keyring):
        __module__ = "keyring.backends.macOS"

        def set_password(self, service, account, value):
            super().set_password(service, account, value)
            entered.set()
            assert release.wait(15)

    backend = HeldProvision()
    repo = _compose(db, tmp_path, backend)
    original = repo.identity_context
    task = asyncio.create_task(ensure_fingerprint_key_off_loop(repo))
    code = r"""
import sys
from pathlib import Path
from unittest.mock import patch
from Tests.windows_private_fixture_runner import user_fixture_default_owner
with user_fixture_default_owner():
    from Tests.Chat.test_citation_key_first_use import _Keyring, _compose
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    db = CharactersRAGDB(sys.argv[1], client_id="second")
    repo = _compose(db, Path(sys.argv[1]).parent, _Keyring())
    with patch("tldw_chatbook.Chat.citation_trace_identity.secrets.token_bytes", return_value=b"k" * 32):
        assert repo.ensure_local_fingerprint_key()
    print(repo.identity_context.fingerprint_key_id, flush=True)
    db.close()
"""
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        unpublished = next(iter(backend.secrets))[1]
        with (tmp_path / "contender.stderr").open("w") as errors:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    "-c",
                    code,
                    str(tmp_path / "contenders.sqlite"),
                ],
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                timeout=10,
            )
        assert result.returncode == 0
        winner = result.stdout.strip()
        assert winner != unpublished and winner != original.fingerprint_key_id
        # Model the shared OS backend becoming visible; only the opaque ID
        # crosses the process pipe. Both test processes know this fixture key.
        backend.secrets[(CITATION_FINGERPRINT_KEYRING_SERVICE, winner)] = (
            base64.b64encode(b"k" * 32).decode()
        )
        release.set()
        await task
        assert load_local_citation_identity_context(db).fingerprint_key_id == winner
        assert repo.identity_context.fingerprint_key_id == winner
        assert repo.local_citation_writes_ready
        assert [kind for kind, _ in backend.calls].count("set") == 1
        assert (CITATION_FINGERPRINT_KEYRING_SERVICE, unpublished) in backend.secrets
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(repo._key_acquisition.thread.join, 5)
        db.close()


def test_stale_legacy_terminal_journal_cannot_follow_fresh_key_publication(tmp_path):
    from tldw_chatbook.Chat.citation_legacy_migration import (
        CitationLegacyMigrationService,
        LegacyMigrationState,
    )
    from tldw_chatbook.Chat.citation_trace_repository import (
        CitationPersistenceUnavailable,
    )

    db = CharactersRAGDB(tmp_path / "journal.sqlite", client_id="journal")
    try:
        stale = _repository(db)
        migration = CitationLegacyMigrationService(
            db=db, repository=stale, sidecar_path=tmp_path / "sidecar.json"
        )
        winner = _compose(db, tmp_path, _Keyring())
        assert winner.ensure_local_fingerprint_key()
        assert winner.identity_context != stale.identity_context
        with pytest.raises(
            CitationPersistenceUnavailable, match="citation_identity_context_mismatch"
        ):
            migration._record_terminal_state(
                "missing-conversation",
                source_fingerprint="old-key-fingerprint",
                state=LegacyMigrationState.FAILED,
                reason_code="legacy_source_unreadable",
                expected_journal=None,
            )
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM rag_legacy_migration_journal")
            .fetchone()[0]
            == 0
        )
    finally:
        db.close()
