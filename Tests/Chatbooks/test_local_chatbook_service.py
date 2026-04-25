import pytest

from tldw_chatbook.Chatbooks import LocalChatbookService


@pytest.mark.asyncio
async def test_local_chatbook_service_persists_record_crud(tmp_path):
    registry_path = tmp_path / "chatbooks.json"
    service = LocalChatbookService(db_paths={}, registry_path=registry_path)

    created = await service.create_chatbook(
        name="Research Pack",
        description="Curated notes and chats",
        file_path="/tmp/research.chatbook.zip",
        tags=["research", "offline"],
        categories=["project"],
        metadata={"purpose": "handoff"},
    )
    listed = await service.list_chatbooks()
    fetched = await service.get_chatbook(created["chatbook_id"])
    updated = await service.update_chatbook(
        created["chatbook_id"],
        name="Research Pack v2",
        tags=["research"],
    )
    reloaded = LocalChatbookService(db_paths={}, registry_path=registry_path)
    persisted = await reloaded.get_chatbook(created["chatbook_id"])
    deleted = await reloaded.delete_chatbook(created["chatbook_id"])

    assert created["id"] == "1"
    assert created["name"] == "Research Pack"
    assert created["file_path"] == "/tmp/research.chatbook.zip"
    assert listed[0]["chatbook_id"] == created["chatbook_id"]
    assert fetched["metadata"] == {"purpose": "handoff"}
    assert updated["name"] == "Research Pack v2"
    assert updated["description"] == "Curated notes and chats"
    assert updated["tags"] == ["research"]
    assert persisted["name"] == "Research Pack v2"
    assert deleted is True
    with pytest.raises(KeyError):
        await reloaded.get_chatbook(created["chatbook_id"])


@pytest.mark.asyncio
async def test_local_chatbook_service_lists_with_query_limit_and_offset(tmp_path):
    service = LocalChatbookService(db_paths={}, registry_path=tmp_path / "chatbooks.json")
    await service.create_chatbook(name="Alpha Pack", description="first")
    await service.create_chatbook(name="Beta Pack", description="second")
    await service.create_chatbook(name="Gamma Notes", description="third")

    results = await service.list_chatbooks(q="pack", limit=1, offset=1)

    assert [item["name"] for item in results] == ["Beta Pack"]
