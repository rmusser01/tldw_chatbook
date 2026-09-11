"""Exercise archive scopes through real file-backed and in-memory services."""

import threading

import pytest

from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [False, True])
async def test_archive_scope_and_reader_route_without_losing_db_or_blocking_loop(
    tmp_path, monkeypatch, memory
):
    db = CharactersRAGDB(
        ":memory:" if memory else tmp_path / "scope.db", client_id="scope-test"
    )
    local = ChatConversationService(db)
    scope = ChatConversationScopeService(local_service=local, server_service=None)
    cid = db.add_conversation({"title": "Saved content"})
    mid = db.add_message(
        {"conversation_id": cid, "sender": "user", "content": "retained"}
    )
    version = db.get_conversation_by_id(cid)["version"]
    owner = threading.get_ident()
    worker_threads = []
    original = db.set_conversations_archived

    def observed(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        try:
            return original(*args, **kwargs)
        finally:
            if not memory:
                db.close_connection()

    monkeypatch.setattr(db, "set_conversations_archived", observed)
    try:
        result = await scope.set_conversations_archived(
            [cid], archived=True, expected_versions={cid: version}
        )
        assert result["changed"] == {cid: version + 1}
        assert (worker_threads[0] == owner) is memory
        assert await scope.get_conversation_archive_states([cid]) == {cid: True}
        assert (await scope.list_conversations())["pagination"]["total"] == 0
        assert (await scope.list_conversations(archive_scope="all"))["pagination"][
            "total"
        ] == 1
        assert (
            await scope.get_library_user_content_evidence()
            is LibraryContentEvidence.HAS_USER_CONTENT
        )
        assert (await scope.get_library_conversation_messages(cid))["messages"][0][
            "id"
        ] == mid
        with pytest.raises(ValueError, match="local"):
            await scope.set_conversations_archived(
                [cid],
                archived=False,
                expected_versions=result["changed"],
                mode="server",
            )
    finally:
        db.close_connection()
