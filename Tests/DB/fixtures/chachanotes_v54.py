"""Genuine v54 ChaChaNotes fixture built by the production migration chain."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@contextmanager
def genuine_v54_database(path: Path) -> Iterator[CharactersRAGDB]:
    """Yield a real v54 database with one ordinary pre-v55 message."""

    with chachanotes_db_at_version(
        path, 54, client_id="semantic-mutation-v54-fixture"
    ) as database:
        conversation_id = database.add_conversation({"title": "v54 fixture"})
        assert conversation_id is not None
        message_id = database.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "ordinary pre-v55 body",
            }
        )
        assert message_id is not None
        yield database
