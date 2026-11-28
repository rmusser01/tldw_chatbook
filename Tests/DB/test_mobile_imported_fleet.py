"""Passive retained fleet rows do not consume actual live delivery capacity."""

import pytest

from Tests.Agents.test_durable_fleet_messages import durable, source
from Tests.Chat.test_mobile_imported_dispatch import _install_chat_history
from tldw_chatbook.Agents.fleet_messages import (
    MAX_CHILD_PENDING,
    MAX_INBOX_PENDING,
    MessageError,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.fleet_progress_repository import FleetProgressRepository


def test_imported_stored_fleet_saturation_leaves_fresh_native_capacity(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.db", "mobile-fleet-test")
    try:
        conversation = db.add_conversation({"title": "history"})
        store, inbox = durable(db, conversation)
        retained = []
        for index in range(MAX_INBOX_PENDING // MAX_CHILD_PENDING):
            sender = inbox.sender(source(index))
            retained.extend(
                sender.send("Historical Ω report") for _ in range(MAX_CHILD_PENDING)
            )
        _install_chat_history(db, {"fleet_progress_messages": retained})
        store.close()
        current, fresh = durable(db, conversation, owner="fresh-owner")
        assert fresh.snapshot() == ()
        sender = fresh.sender(source(100))
        new_ids = [sender.send("Fresh report") for _ in range(MAX_CHILD_PENDING)]
        with pytest.raises(MessageError, match="queue_full"):
            sender.send("Over child limit")
        assert {row.message_id for row in fresh.snapshot()} == set(new_ids)
        for index in range(1, MAX_INBOX_PENDING // MAX_CHILD_PENDING):
            sender = fresh.sender(source(100 + index))
            new_ids.extend(
                sender.send("Fresh Ω report") for _ in range(MAX_CHILD_PENDING)
            )
        with pytest.raises(MessageError, match="queue_full"):
            fresh.sender(source(999)).send("Over global limit")
        assert db.get_connection().execute(
            "SELECT count(*) FROM fleet_progress_messages"
        ).fetchone()[0] == len(retained) + len(new_ids)
        repository = FleetProgressRepository(db)
        repository.remove(conversation, retained)
        assert {row.message_id for row in repository.load(conversation)} == set(new_ids)
        current.close()
    finally:
        db.close()


def test_invalid_fleet_census_cannot_free_live_capacity(tmp_path):
    db = CharactersRAGDB(tmp_path / "invalid.db", "mobile-fleet-test")
    try:
        conversation = db.add_conversation({"title": "history"})
        store, inbox = durable(db, conversation)
        retained = inbox.sender(source()).send("Retain me")
        _install_chat_history(db, {"fleet_progress_messages": (retained,)})
        with db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key='chat'"
            )
        repository = FleetProgressRepository(db)
        with pytest.raises(MessageError, match="durable_unavailable"):
            repository.load(conversation)
        from tldw_chatbook.Agents.fleet_messages import ProgressMessage

        with pytest.raises(ValueError, match="receipt"):
            repository.append(
                conversation, ProgressMessage("new-id", source(20), "Current")
            )
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM fleet_progress_messages")
            .fetchone()[0]
            == 1
        )
        store.close()
    finally:
        db.close()
