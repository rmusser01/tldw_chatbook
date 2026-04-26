from __future__ import annotations

import pytest

from tldw_chatbook.Chat_Grammars_Interop.local_chat_grammars_service import LocalChatGrammarsService


@pytest.mark.asyncio
async def test_local_chat_grammars_service_persists_crud_and_archival(tmp_path):
    service = LocalChatGrammarsService(store_path=tmp_path / "grammars.json")

    created = await service.create_grammar(
        name="JSON object",
        description="Strict JSON shape",
        grammar_text="root ::= object",
    )

    assert created["id"] == "local-grammar-1"
    assert created["version"] == 1
    assert created["validation_status"] == "unchecked"

    listed = await service.list_grammars()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == "local-grammar-1"

    fetched = await service.get_grammar("local-grammar-1")
    assert fetched["grammar_text"] == "root ::= object"

    updated = await service.update_grammar(
        "local-grammar-1",
        version=1,
        name="Strict JSON",
        validation_status="valid",
    )
    assert updated["name"] == "Strict JSON"
    assert updated["version"] == 2
    assert updated["validation_status"] == "valid"

    reloaded = LocalChatGrammarsService(store_path=tmp_path / "grammars.json")
    reloaded_fetched = await reloaded.get_grammar("local-grammar-1")
    assert reloaded_fetched["name"] == "Strict JSON"
    assert reloaded_fetched["version"] == 2

    deleted = await reloaded.delete_grammar("local-grammar-1")
    assert deleted is True
    assert await reloaded.list_grammars() == {"items": [], "total": 0, "limit": 100, "offset": 0}
    archived = await reloaded.get_grammar("local-grammar-1", include_archived=True)
    assert archived["is_archived"] is True


@pytest.mark.asyncio
async def test_local_chat_grammars_service_hard_deletes(tmp_path):
    service = LocalChatGrammarsService(store_path=tmp_path / "grammars.json")
    await service.create_grammar(name="JSON object", grammar_text="root ::= object")

    assert await service.delete_grammar("local-grammar-1", hard_delete=True) is True

    with pytest.raises(ValueError, match="local_chat_grammar_not_found"):
        await service.get_grammar("local-grammar-1", include_archived=True)


@pytest.mark.asyncio
async def test_local_chat_grammars_service_rejects_version_conflict(tmp_path):
    service = LocalChatGrammarsService(store_path=tmp_path / "grammars.json")
    await service.create_grammar(name="JSON object", grammar_text="root ::= object")

    with pytest.raises(ValueError, match="local_chat_grammar_version_conflict"):
        await service.update_grammar("local-grammar-1", version=99, name="Conflict")
