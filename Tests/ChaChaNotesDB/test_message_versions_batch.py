"""Batch message-version projection (task-32804.12 [D2]).

``_durable_context_snapshots`` used to read one ``get_message_by_id_without_blob``
per active-path message on the event loop, on every dispatch. It now hydrates
the versions in one chunked read via ``get_message_versions_by_ids``. These
tests pin that method's per-row contract: the current positive version for each
live row, deleted rows omitted, unknown ids absent -- matching what the point
read + ``get_message_version`` validity check returned per message.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(chachanotes_template_db, tmp_path):
    path = tmp_path / "versions.sqlite"
    shutil.copyfile(chachanotes_template_db, path)
    database = CharactersRAGDB(path, "versions-test")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture
def conv_id(db):
    card_id = db.add_character_card({"name": "Batch Version Char"})
    return db.add_conversation({"character_id": card_id, "title": "conv"})


def _add(db, conv_id: str, content: str) -> str:
    return db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": content}
    )


def test_batch_returns_current_versions_excludes_deleted_and_unknown(db, conv_id):
    live = _add(db, conv_id, "kept")
    bumped = _add(db, conv_id, "will edit")
    gone = _add(db, conv_id, "will delete")

    # Bump `bumped` to version 2 so the batch must report the CURRENT version.
    db.update_message(bumped, {"content": "edited"}, expected_version=1)
    # Soft-delete `gone`; it must drop out of the result entirely.
    db.soft_delete_message(gone, expected_version=1)

    versions = db.get_message_versions_by_ids([live, bumped, gone, "does-not-exist"])

    assert versions == {live: 1, bumped: 2}
    # Parity with the per-message reader for each live id.
    assert db.get_message_by_id_without_blob(live)["version"] == 1
    assert db.get_message_by_id_without_blob(bumped)["version"] == 2


def test_batch_empty_and_falsy_input_returns_empty(db):
    assert db.get_message_versions_by_ids([]) == {}
    # Falsy/None ids are collapsed away, not queried.
    assert db.get_message_versions_by_ids([None, ""]) == {}
