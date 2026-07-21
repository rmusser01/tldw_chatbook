"""UserStatistics DB API regression test.

Every helper in ``UserStatistics`` used to call
``db.get_or_create_connection()`` — a method that no longer exists on
``CharactersRAGDB`` — so each stat swallowed an ``AttributeError`` and
reported 0/"Unknown" against real data. The module must use the current
public ``get_connection()`` accessor.
"""

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Stats.user_statistics import UserStatistics


def test_user_statistics_counts_real_data(tmp_path):
    db = CharactersRAGDB(tmp_path / "stats.db", client_id="test-client")
    conv_id = db.add_conversation({"title": "Stats Conversation", "character_id": None})
    db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "Hello there"}
    )
    db.add_message({"conversation_id": conv_id, "sender": "ai", "content": "Hi!"})

    stats = UserStatistics(db).get_all_stats()

    assert stats["total_conversations"] == 1
    assert stats["total_messages"] == 2
