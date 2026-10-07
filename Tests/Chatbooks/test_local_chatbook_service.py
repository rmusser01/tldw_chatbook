import json
import os

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
    assert created["artifact_revision"] == 1
    assert created["name"] == "Research Pack"
    assert created["file_path"] == "/tmp/research.chatbook.zip"
    assert listed[0]["chatbook_id"] == created["chatbook_id"]
    assert fetched["metadata"] == {"purpose": "handoff"}
    assert updated["name"] == "Research Pack v2"
    assert updated["description"] == "Curated notes and chats"
    assert updated["tags"] == ["research"]
    assert persisted["name"] == "Research Pack v2"
    assert (
        json.loads(registry_path.read_text(encoding="utf-8"))["provenance_outbox"] == []
    )
    assert deleted is True
    with pytest.raises(KeyError):
        await reloaded.get_chatbook(created["chatbook_id"])


@pytest.mark.asyncio
async def test_local_chatbook_service_lists_with_query_limit_and_offset(tmp_path):
    service = LocalChatbookService(
        db_paths={}, registry_path=tmp_path / "chatbooks.json"
    )
    await service.create_chatbook(name="Alpha Pack", description="first")
    await service.create_chatbook(name="Beta Pack", description="second")
    await service.create_chatbook(name="Gamma Notes", description="third")

    results = await service.list_chatbooks(q="pack", limit=1, offset=1)

    assert [item["name"] for item in results] == ["Beta Pack"]


def _install_registry_parse_spy(monkeypatch):
    """Count registry file parse (read + json + validation) passes.

    Returns the list of service instances that triggered each parse.
    """

    calls = []
    original = LocalChatbookService._parse_registry

    def spy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(LocalChatbookService, "_parse_registry", spy)
    return calls


@pytest.mark.asyncio
async def test_list_chatbooks_parses_registry_once_while_stat_unchanged(
    tmp_path, monkeypatch
):
    registry_path = tmp_path / "chatbooks.json"
    writer = LocalChatbookService(db_paths={}, registry_path=registry_path)
    await writer.create_chatbook(name="Alpha Pack", description="first")
    parse_calls = _install_registry_parse_spy(monkeypatch)
    reader = LocalChatbookService(db_paths={}, registry_path=registry_path)

    first = await reader.list_chatbooks()
    second = await reader.list_chatbooks()

    assert [item["name"] for item in first] == ["Alpha Pack"]
    assert second == first
    assert len(parse_calls) == 1, (
        "two list_chatbooks calls with an unchanged registry file must perform "
        f"exactly one file read/parse, performed {len(parse_calls)}"
    )


@pytest.mark.asyncio
async def test_registry_cache_reparses_after_stat_change(tmp_path, monkeypatch):
    registry_path = tmp_path / "chatbooks.json"
    writer = LocalChatbookService(db_paths={}, registry_path=registry_path)
    await writer.create_chatbook(name="Alpha Pack", description="first")
    parse_calls = _install_registry_parse_spy(monkeypatch)
    reader = LocalChatbookService(db_paths={}, registry_path=registry_path)

    await reader.list_chatbooks()
    assert len(parse_calls) == 1

    stat = registry_path.stat()
    registry_path.write_text(
        json.dumps(
            {
                "next_id": 2,
                "records": [
                    {
                        "id": "1",
                        "chatbook_id": 1,
                        "name": "External Edit",
                        "description": "written by another writer",
                        "file_path": None,
                        "tags": [],
                        "categories": [],
                        "metadata": {},
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "artifact_revision": 1,
                    }
                ],
                "provenance_outbox": [],
                "provenance_reconcile_cursor": 0,
            }
        ),
        encoding="utf-8",
    )
    # Force an mtime_ns difference even on coarse-grained filesystems.
    os.utime(registry_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    listed = await reader.list_chatbooks()

    assert len(parse_calls) == 2, "a touched registry file must trigger a re-parse"
    assert [item["name"] for item in listed] == ["External Edit"]


@pytest.mark.asyncio
async def test_list_chatbooks_copies_stay_independent_between_calls(tmp_path):
    service = LocalChatbookService(
        db_paths={}, registry_path=tmp_path / "chatbooks.json"
    )
    await service.create_chatbook(name="Alpha Pack", tags=["research"])

    first = await service.list_chatbooks()
    first[0]["name"] = "Mutated In Place"
    first[0]["tags"].append("mutated")
    first[0]["metadata"]["injected"] = True

    second = await service.list_chatbooks()

    assert second[0]["name"] == "Alpha Pack"
    assert second[0]["tags"] == ["research"]
    assert second[0]["metadata"] == {}


@pytest.mark.asyncio
async def test_registry_cache_reflects_writes_from_another_service_instance(
    tmp_path, monkeypatch
):
    registry_path = tmp_path / "chatbooks.json"
    service_a = LocalChatbookService(db_paths={}, registry_path=registry_path)
    await service_a.create_chatbook(name="Alpha Pack", description="first")
    await service_a.list_chatbooks()  # prime service_a's cache
    parse_calls = _install_registry_parse_spy(monkeypatch)
    service_b = LocalChatbookService(db_paths={}, registry_path=registry_path)

    await service_b.create_chatbook(name="Beta Pack", description="second")

    listed = await service_a.list_chatbooks()

    assert [item["name"] for item in listed] == ["Alpha Pack", "Beta Pack"]
    service_a_parses = [call for call in parse_calls if call is service_a]
    assert len(service_a_parses) == 1, (
        "service_b's write changes the stat signature, so service_a's next read "
        "re-parses exactly once and observes the new record"
    )


@pytest.mark.asyncio
async def test_local_chatbook_service_home_artifact_snapshot_lists_latest_console_saved_artifacts(
    tmp_path,
):
    service = LocalChatbookService(
        db_paths={}, registry_path=tmp_path / "chatbooks.json"
    )
    await service.create_chatbook(
        name="Generic Pack",
        description="Imported pack",
        metadata={"artifact_source": "import"},
    )
    older = await service.create_chatbook(
        name="Older Console Answer",
        description="Saved from Console assistant response.",
        metadata={
            "artifact_source": "console",
            "artifact_kind": "assistant-response",
            "message_id": "msg-old",
            "content": "Older saved answer.",
            "content_truncated": False,
        },
    )
    newer = await service.create_chatbook(
        name="Newer Console Answer",
        description="Saved from Console assistant response.",
        metadata={
            "artifact_source": "console",
            "artifact_kind": "assistant-response",
            "message_id": "msg-new",
            "content": "Newer saved answer.",
            "content_truncated": False,
        },
    )

    snapshot = service.list_home_artifact_snapshot(limit=2)

    assert [record["chatbook_id"] for record in snapshot] == [
        newer["chatbook_id"],
        older["chatbook_id"],
    ]
    assert snapshot[0]["metadata"]["message_id"] == "msg-new"
