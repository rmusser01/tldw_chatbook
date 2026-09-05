"""The source database owner drains Keyword worker-thread SQLite handles."""

from concurrent.futures import ThreadPoolExecutor

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_keyword_worker_handles_close_through_source_owner_quiescence(tmp_path):
    database = CharactersRAGDB(tmp_path / "keyword-owner.sqlite", client_id="owner")
    service = CharacterConversationNavigationService(database)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(service.ensure_keyword_index).result()
        database.close_connection()
        assert database.registered_connection_count() == 1
        with database.quiesce_connections(timeout_seconds=2.0):
            pass
        assert database.registered_connection_count() == 0
    finally:
        with database.quiesce_connections(timeout_seconds=2.0):
            pass
