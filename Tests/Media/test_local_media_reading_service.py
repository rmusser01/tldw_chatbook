import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.notification_dispatch_service import NotificationDispatchService
from tldw_chatbook.tldw_api.media_reading_schemas import ItemsBulkRequest, ReadingSaveRequest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Media" / "local_media_reading_service.py"
_SPEC = spec_from_file_location("local_media_reading_service_test_module", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
LocalMediaReadingService = _MODULE.LocalMediaReadingService


@pytest.fixture
def memory_db_factory():
    created_dbs = []

    def _create_db(client_id="test_client"):
        db = Database(db_path=":memory:", client_id=client_id)
        created_dbs.append(db)
        return db

    yield _create_db

    for db in created_dbs:
        try:
            db.close_connection()
        except Exception:
            pass


class SpyReadItLaterDb:
    def __init__(self):
        self.saved_filters = []
        self.search_calls = []

    def list_read_it_later_media_ids(self, *, include_deleted=False, include_trash=False):
        self.saved_filters.append(
            {
                "include_deleted": include_deleted,
                "include_trash": include_trash,
            }
        )
        return [2, 3]

    def get_media_read_it_later_state(self, media_id):
        return {
            "media_id": media_id,
            "is_read_it_later": True,
            "saved_at": "2026-04-21T10:00:00Z",
        }

    def search_media_db(
        self,
        *,
        search_query=None,
        search_fields=None,
        media_types=None,
        date_range=None,
        must_have_keywords=None,
        must_not_have_keywords=None,
        sort_by="last_modified_desc",
        media_ids_filter=None,
        page=1,
        results_per_page=20,
        include_trash=False,
        include_deleted=False,
    ):
        self.search_calls.append(
            {
                "search_query": search_query,
                "media_ids_filter": media_ids_filter,
                "include_trash": include_trash,
                "include_deleted": include_deleted,
            }
        )
        return ([{"id": media_id} for media_id in (media_ids_filter or [])], len(media_ids_filter or []))


def test_local_service_search_media_uses_db_backed_saved_filter_spy():
    db = SpyReadItLaterDb()
    service = LocalMediaReadingService(db)

    payload = service.search_media(read_it_later_only=True, media_ids_filter=["1", 2, "4"])

    assert db.saved_filters == [{"include_deleted": False, "include_trash": False}]
    assert db.search_calls[0]["media_ids_filter"] == [2]
    assert [item["id"] for item in payload["items"]] == [2]
    assert payload["items"][0]["is_read_it_later"] is True
    assert payload["items"][0]["saved_at"] == "2026-04-21T10:00:00Z"


def test_local_service_search_media_uses_db_backed_saved_filter(memory_db_factory):
    db = memory_db_factory()
    kept_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    other_id, _, _ = db.add_media_with_keywords(title="Drop", content="B", media_type="article", keywords=[])
    db.save_media_to_read_it_later(kept_id)

    service = LocalMediaReadingService(db)
    payload = service.search_media(read_it_later_only=True)

    assert [item["id"] for item in payload["items"]] == [kept_id]
    assert all(item["id"] != other_id for item in payload["items"])


def test_local_service_search_media_read_it_later_only_enriches_saved_state(memory_db_factory):
    db = memory_db_factory()
    kept_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    db.save_media_to_read_it_later(kept_id)

    service = LocalMediaReadingService(db)
    payload = service.search_media(read_it_later_only=True)

    assert payload["items"][0]["id"] == kept_id
    assert payload["items"][0]["is_read_it_later"] is True
    assert payload["items"][0]["saved_at"] is not None


def test_local_service_search_media_enriches_saved_state_on_normal_browse(memory_db_factory):
    db = memory_db_factory()
    kept_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    db.save_media_to_read_it_later(kept_id)

    service = LocalMediaReadingService(db)
    payload = service.search_media()

    assert payload["items"][0]["id"] == kept_id
    assert payload["items"][0]["is_read_it_later"] is True
    assert payload["items"][0]["saved_at"] is not None


def test_local_service_get_media_detail_enriches_saved_state(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    db.save_media_to_read_it_later(media_id)

    service = LocalMediaReadingService(db)
    detail = service.get_media_detail(media_id)

    assert detail["id"] == media_id
    assert detail["is_read_it_later"] is True
    assert detail["saved_at"] is not None


def test_local_service_save_and_remove_read_it_later_round_trips(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    service = LocalMediaReadingService(db)

    saved = service.save_to_read_it_later(media_id)
    removed = service.remove_from_read_it_later(media_id)

    assert saved["is_read_it_later"] is True
    assert saved["saved_at"] is not None
    assert removed["is_read_it_later"] is False
    assert removed["saved_at"] is None
    assert db.get_media_read_it_later_state(media_id) is None


def test_local_service_reading_highlights_round_trip(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Annotated",
        content="Important sentence in the document",
        media_type="article",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    created = service.create_reading_highlight(
        media_id,
        quote="Important sentence",
        start_offset=0,
        end_offset=18,
        color="yellow",
        note="Check this",
    )
    listed = service.list_reading_highlights(media_id)
    updated = service.update_reading_highlight(
        created["id"],
        color="blue",
        note="Updated",
        state="active",
    )
    deleted = service.delete_reading_highlight(created["id"])

    assert created["media_id"] == media_id
    assert created["quote"] == "Important sentence"
    assert created["anchor_strategy"] == "fuzzy_quote"
    assert listed == [created]
    assert updated["color"] == "blue"
    assert updated["note"] == "Updated"
    assert deleted is True
    assert service.list_reading_highlights(media_id) == []


def test_local_service_ingestion_sources_round_trip(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    created = service.create_ingestion_source(
        source_type="local_directory",
        sink_type="media",
        policy="canonical",
        enabled=True,
        schedule_enabled=True,
        schedule={"interval": "daily"},
        config={"path": "/tmp/media"},
    )
    listed = service.list_ingestion_sources()
    detail = service.get_ingestion_source(created["id"])
    patched = service.patch_ingestion_source(
        created["id"],
        enabled=False,
        policy="import_only",
        schedule={"interval": "weekly"},
        config={"path": "/tmp/media-v2"},
    )
    items = service.list_ingestion_source_items(created["id"])
    deleted = service.delete_ingestion_source(created["id"])

    assert created["source_type"] == "local_directory"
    assert created["sink_type"] == "media"
    assert created["schedule_config"] == {"interval": "daily"}
    assert created["config"] == {"path": "/tmp/media"}
    assert listed == [created]
    assert detail == created
    assert patched["enabled"] is False
    assert patched["policy"] == "import_only"
    assert patched["schedule_config"] == {"interval": "weekly"}
    assert patched["config"] == {"path": "/tmp/media-v2"}
    assert items == []
    assert deleted is True
    assert service.list_ingestion_sources() == []


def test_local_service_reading_saved_searches_round_trip(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    created = service.create_reading_saved_search(
        name="Saved AI Reads",
        query={"status": ["saved"], "q": "ai"},
        sort="updated_desc",
    )
    listed = service.list_reading_saved_searches(limit=10, offset=0)
    updated = service.update_reading_saved_search(
        created["id"],
        name="Updated AI Reads",
        query={"status": ["reading"]},
    )
    deleted = service.delete_reading_saved_search(created["id"])

    assert created["name"] == "Saved AI Reads"
    assert created["query"] == {"status": ["saved"], "q": "ai"}
    assert created["sort"] == "updated_desc"
    assert listed["items"] == [created]
    assert listed["total"] == 1
    assert updated["name"] == "Updated AI Reads"
    assert updated["query"] == {"status": ["reading"]}
    assert deleted == {"ok": True}
    assert service.list_reading_saved_searches()["items"] == []


def test_local_service_reading_note_links_round_trip(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(title="Linked", content="A", media_type="article", keywords=[])
    service = LocalMediaReadingService(db)

    linked = service.link_reading_item_note(media_id, note_id="note-uuid-1")
    listed = service.list_reading_item_note_links(media_id)
    unlinked = service.unlink_reading_item_note(media_id, "note-uuid-1")

    assert linked["item_id"] == media_id
    assert linked["note_id"] == "note-uuid-1"
    assert linked["created_at"] is not None
    assert listed == {"item_id": media_id, "links": [linked]}
    assert unlinked == {"ok": True}
    assert service.list_reading_item_note_links(media_id) == {"item_id": media_id, "links": []}


def test_local_service_exports_saved_reading_items_jsonl(memory_db_factory):
    db = memory_db_factory()
    kept_id, _, _ = db.add_media_with_keywords(
        title="Saved RAG Article",
        content="RAG notes for export",
        media_type="article",
        keywords=["ai"],
    )
    db.add_media_with_keywords(title="Unsaved", content="Ignored", media_type="article", keywords=["ai"])
    db.save_media_to_read_it_later(kept_id)
    service = LocalMediaReadingService(db)
    service.create_reading_highlight(kept_id, quote="RAG notes", start_offset=0, end_offset=9)
    service.link_reading_item_note(kept_id, note_id="note-uuid-1")

    exported = service.export_reading_items(
        status=["saved"],
        tags=["ai"],
        include_text=True,
        include_highlights=True,
        include_notes=True,
        format="jsonl",
    )

    rows = [json.loads(line) for line in exported.decode("utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == kept_id
    assert rows[0]["title"] == "Saved RAG Article"
    assert rows[0]["content"] == "RAG notes for export"
    assert rows[0]["highlights"][0]["quote"] == "RAG notes"
    assert rows[0]["note_links"][0]["note_id"] == "note-uuid-1"


def test_local_service_lists_and_gets_unified_items(memory_db_factory):
    db = memory_db_factory()
    kept_id, _, _ = db.add_media_with_keywords(
        title="Saved Local Article",
        content="Local unified item",
        media_type="article",
        keywords=["ai"],
        url="https://example.com/local",
    )
    db.add_media_with_keywords(title="Unsaved", content="Ignored", media_type="article", keywords=["ai"])
    db.save_media_to_read_it_later(kept_id)
    service = LocalMediaReadingService(db)

    listing = service.list_unified_items(status=["saved"], tags=["ai"], page=1, size=10)
    detail = service.get_unified_item(kept_id)

    assert listing["total"] == 1
    assert listing["items"][0]["id"] == kept_id
    assert listing["items"][0]["content_item_id"] == kept_id
    assert listing["items"][0]["media_id"] == kept_id
    assert listing["items"][0]["origin"] == "media"
    assert listing["items"][0]["type"] == "media"
    assert listing["items"][0]["media_type"] == "article"
    assert listing["items"][0]["status"] == "saved"
    assert listing["items"][0]["domain"] == "example.com"
    assert listing["items"][0]["tags"] == ["ai"]
    assert detail["id"] == kept_id
    assert detail["status"] == "saved"


def test_local_service_saves_reading_url_to_local_media(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    saved = service.save_reading_item(
        ReadingSaveRequest(
            url="https://example.com/local",
            title="Local Saved URL",
            tags=[" ai ", "reading"],
            content="Locally saved content",
        )
    )

    detail = service.get_media_detail(saved["media_id"])
    read_it_later = service.list_reading_item_note_links(saved["media_id"])

    assert saved["id"] == saved["media_id"]
    assert saved["title"] == "Local Saved URL"
    assert saved["url"] == "https://example.com/local"
    assert saved["status"] == "saved"
    assert saved["favorite"] is False
    assert saved["tags"] == ["ai", "reading"]
    assert detail["content"] == "Locally saved content"
    assert detail["is_read_it_later"] is True
    assert read_it_later == {"item_id": saved["media_id"], "links": []}


def test_local_service_bulk_updates_reading_items(memory_db_factory):
    db = memory_db_factory()
    first_id, _, _ = db.add_media_with_keywords(title="First", content="A", media_type="article", keywords=["old"])
    second_id, _, _ = db.add_media_with_keywords(title="Second", content="B", media_type="article", keywords=[])
    service = LocalMediaReadingService(db)

    saved = service.bulk_update_reading_items(
        item_ids=[first_id, second_id],
        action="set_status",
        status="saved",
    )
    tagged = service.bulk_update_reading_items(
        item_ids=[first_id],
        action="replace_tags",
        tags=["new"],
    )

    assert saved["succeeded"] == 2
    assert service.get_media_detail(first_id)["is_read_it_later"] is True
    assert service.get_media_detail(second_id)["is_read_it_later"] is True
    assert tagged["succeeded"] == 1
    assert db.fetch_keywords_for_media_batch([first_id]) == {first_id: ["new"]}

    deleted = service.bulk_update_reading_items(
        item_ids=[second_id],
        action="delete",
    )

    assert deleted["succeeded"] == 1
    assert db.get_media_by_id(second_id) is None


def test_local_service_bulk_updates_unified_items(memory_db_factory):
    db = memory_db_factory()
    first_id, _, _ = db.add_media_with_keywords(title="First", content="A", media_type="article", keywords=["old"])
    second_id, _, _ = db.add_media_with_keywords(title="Second", content="B", media_type="article", keywords=[])
    service = LocalMediaReadingService(db)

    saved = service.bulk_update_unified_items(
        ItemsBulkRequest(item_ids=[first_id, second_id], action="set_status", status="saved")
    )
    tagged = service.bulk_update_unified_items(
        ItemsBulkRequest(item_ids=[first_id], action="replace_tags", tags=["new"])
    )

    assert saved["succeeded"] == 2
    assert service.get_unified_item(first_id)["status"] == "saved"
    assert service.get_unified_item(second_id)["status"] == "saved"
    assert tagged["succeeded"] == 1
    assert service.get_unified_item(first_id)["tags"] == ["new"]


def test_local_service_uploads_archive_snapshot_source(memory_db_factory, tmp_path):
    db = memory_db_factory()
    source = db.create_local_ingestion_source(
        source_type="archive_snapshot",
        sink_type="media",
        policy="canonical",
    )
    archive_path = tmp_path / "reading-archive.zip"
    archive_path.write_bytes(b"archive bytes")
    service = LocalMediaReadingService(db)

    uploaded = service.upload_ingestion_source_archive(source["id"], str(archive_path))
    items = service.list_ingestion_source_items(source["id"])

    assert uploaded["status"] == "tracked"
    assert uploaded["source_id"] == source["id"]
    assert uploaded["item"]["normalized_relative_path"] == "reading-archive.zip"
    assert uploaded["item"]["sync_status"] == "tracked"
    assert uploaded["item"]["binding"]["archive_path"] == str(archive_path)
    assert uploaded["item"]["binding"]["size_bytes"] == len(b"archive bytes")
    assert items == [uploaded["item"]]


def test_local_service_syncs_local_directory_source(memory_db_factory, tmp_path):
    source_dir = tmp_path / "library"
    source_dir.mkdir()
    (source_dir / "article.md").write_text("Article", encoding="utf-8")
    nested = source_dir / "nested"
    nested.mkdir()
    removed_file = nested / "old.txt"
    removed_file.write_text("Old", encoding="utf-8")

    db = memory_db_factory()
    source = db.create_local_ingestion_source(
        source_type="local_directory",
        sink_type="media",
        policy="canonical",
        config={"path": str(source_dir)},
    )
    service = LocalMediaReadingService(db)

    first_sync = service.trigger_ingestion_source_sync(source["id"])
    removed_file.unlink()
    second_sync = service.trigger_ingestion_source_sync(source["id"])
    items = {
        item["normalized_relative_path"]: item
        for item in service.list_ingestion_source_items(source["id"])
    }

    assert first_sync["status"] == "completed"
    assert first_sync["items_scanned"] == 2
    assert second_sync["items_scanned"] == 1
    assert second_sync["items_missing"] == 1
    assert items["article.md"]["present_in_source"] is True
    assert items["article.md"]["sync_status"] == "tracked"
    assert items["nested/old.txt"]["present_in_source"] is False
    assert items["nested/old.txt"]["sync_status"] == "missing"


def test_local_service_creates_reading_archive_as_local_media(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Readable Article",
        url="https://example.com/readable",
        content="Article body",
        media_type="article",
        keywords=["source"],
    )
    service = LocalMediaReadingService(db)

    archive = service.create_reading_archive(
        media_id,
        format="md",
        source="text",
        title="Readable Archive",
    )
    archived_media = db.get_media_by_id(archive["output_id"])

    assert archive["title"] == "Readable Archive"
    assert archive["format"] == "md"
    assert archive["storage_path"] == f"local://media/{archive['output_id']}"
    assert archived_media["type"] == "reading_archive"
    assert "# Readable Archive" in archived_media["content"]
    assert "Article body" in archived_media["content"]


def test_local_service_summarizes_reading_item_extractively(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Readable Article",
        url="https://example.com/readable",
        content=(
            "First sentence explains the local article. "
            "Second sentence adds the important detail. "
            "Third sentence gives a useful caveat. "
            "Fourth sentence should be omitted."
        ),
        media_type="article",
        keywords=["source"],
    )
    service = LocalMediaReadingService(db)

    summary = service.summarize_reading_item(media_id, provider="remote-ignored", model="ignored")

    assert summary["item_id"] == media_id
    assert summary["provider"] == "local"
    assert summary["model"] == "extractive"
    assert summary["summary"] == (
        "First sentence explains the local article. "
        "Second sentence adds the important detail. "
        "Third sentence gives a useful caveat."
    )
    assert summary["citations"] == [
        {
            "item_id": media_id,
            "url": "https://example.com/readable",
            "title": "Readable Article",
            "source": "local",
        }
    ]


class FakeTTSService:
    def __init__(self):
        self.calls = []

    async def generate_audio_stream(self, request, internal_model_id):
        self.calls.append((request, internal_model_id))
        yield b"audio-"
        yield b"bytes"


@pytest.mark.asyncio
async def test_local_service_generates_reading_tts_from_local_tts_service(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="TTS Target",
        content="Alpha beta gamma.",
        media_type="article",
        url="https://example.test/tts",
        keywords=[],
    )
    tts_service = FakeTTSService()
    service = LocalMediaReadingService(db, tts_service=tts_service)

    audio = await service.tts_reading_item(
        media_id,
        model="local-kokoro",
        voice="af_heart",
        response_format="wav",
        stream=False,
        speed=1.25,
        max_chars=10,
        text_source="text",
    )

    request, internal_model_id = tts_service.calls[0]
    assert audio == b"audio-bytes"
    assert internal_model_id == "local-kokoro"
    assert request.model == "local-kokoro"
    assert request.input == "Alpha beta"
    assert request.voice == "af_heart"
    assert request.response_format == "wav"
    assert request.stream is False
    assert request.speed == 1.25


@pytest.mark.asyncio
async def test_local_service_coerces_reading_tts_stream_string_false(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="TTS Target",
        content="Alpha beta gamma.",
        media_type="article",
        url="https://example.test/tts",
        keywords=[],
    )
    tts_service = FakeTTSService()
    service = LocalMediaReadingService(db, tts_service=tts_service)

    await service.tts_reading_item(media_id, stream="false")

    request, _ = tts_service.calls[0]
    assert request.stream is False


def test_local_service_persists_reading_digest_schedules_and_outputs(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    created = service.create_reading_digest_schedule(
        name="Morning Digest",
        cron="0 8 * * *",
        timezone="UTC",
        enabled=True,
        require_online=False,
        format="md",
        retention_days=30,
        filters={"status": ["saved"], "limit": 10},
    )
    output = db.create_local_reading_digest_output(
        schedule_id=created["id"],
        title="Morning Digest - 2026-04-24",
        format="md",
        download_url=f"local://reading_digest/{created['id']}/1",
        item_count=2,
        metadata={"item_ids": [1, 2]},
    )

    schedules = service.list_reading_digest_schedules(limit=10, offset=0)
    fetched = service.get_reading_digest_schedule(created["id"])
    updated = service.update_reading_digest_schedule(created["id"], enabled=False, name="Updated Digest")
    outputs = service.list_reading_digest_outputs(schedule_id=created["id"], limit=5, offset=0)
    deleted = service.delete_reading_digest_schedule(created["id"])

    assert schedules[0]["id"] == created["id"]
    assert fetched["filters"] == {"status": ["saved"], "limit": 10}
    assert updated["enabled"] is False
    assert updated["name"] == "Updated Digest"
    assert outputs["items"][0]["output_id"] == output["output_id"]
    assert outputs["items"][0]["schedule_id"] == str(created["id"])
    assert outputs["items"][0]["metadata"] == {"item_ids": [1, 2]}
    assert deleted == {"ok": True}
    assert service.list_reading_digest_schedules(limit=10, offset=0) == []


def test_local_service_imports_pocket_reading_items(memory_db_factory, tmp_path):
    import_file = tmp_path / "pocket.json"
    import_file.write_text(
        json.dumps(
            {
                "list": {
                    "1": {
                        "resolved_url": "https://example.com/one",
                        "resolved_title": "One",
                        "tags": {"ai": {}, "reading": {}},
                        "status": "0",
                    },
                    "2": {
                        "given_url": "https://example.com/two",
                        "given_title": "Two",
                        "tags": [{"tag": "archive"}],
                        "status": "1",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    job = service.import_reading_items(str(import_file), source="pocket", merge_tags=True)
    listed = service.list_reading_import_jobs()
    fetched = service.get_reading_import_job(job["job_id"])
    reloaded_service = LocalMediaReadingService(db)
    reloaded = reloaded_service.get_reading_import_job(job["job_id"])
    saved = db.get_media_by_url("https://example.com/one")
    archived = db.get_media_by_url("https://example.com/two")

    assert job["status"] == "completed"
    assert job["result"] == {
        "source": "pocket",
        "imported": 2,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }
    assert listed["total"] == 1
    assert fetched["job_id"] == job["job_id"]
    assert reloaded["result"] == job["result"]
    assert saved["title"] == "One"
    assert archived["title"] == "Two"
    assert service.get_media_detail(saved["id"])["is_read_it_later"] is True
    assert service.get_media_detail(archived["id"]).get("is_read_it_later") is not True
    assert db.fetch_keywords_for_media_batch([saved["id"], archived["id"]]) == {
        saved["id"]: ["ai", "reading"],
        archived["id"]: ["archive"],
    }


def test_local_service_imports_instapaper_reading_items(memory_db_factory, tmp_path):
    import_file = tmp_path / "instapaper.csv"
    import_file.write_text(
        "URL,Title,Folder,Tags,Notes\n"
        "https://example.com/current,Current,Unread,\"ai,notes\",Read later\n"
        "https://example.com/archive,Archived,Archive,archive,Already read\n",
        encoding="utf-8",
    )
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    job = service.import_reading_items(str(import_file), source="auto")
    current = db.get_media_by_url("https://example.com/current")
    archived = db.get_media_by_url("https://example.com/archive")

    assert job["status"] == "completed"
    assert job["result"] == {
        "source": "instapaper",
        "imported": 2,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }
    assert current["title"] == "Current"
    assert archived["title"] == "Archived"
    assert service.get_media_detail(current["id"])["is_read_it_later"] is True
    assert service.get_media_detail(archived["id"]).get("is_read_it_later") is not True
    assert db.fetch_keywords_for_media_batch([current["id"], archived["id"]]) == {
        current["id"]: ["ai", "notes"],
        archived["id"]: ["archive"],
    }


def test_local_service_submits_and_persists_media_ingest_jobs(memory_db_factory, tmp_path):
    local_file = tmp_path / "article.md"
    local_file.write_text("# Local Article\n\nBody text", encoding="utf-8")
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    submitted = service.submit_media_ingest_jobs(
        media_type="document",
        urls=["https://example.com/remote-doc"],
        file_paths=[str(local_file)],
        title="Imported Document",
        tags=["local", "ingest"],
    )
    jobs = submitted["jobs"]
    listed = service.list_media_ingest_jobs(batch_id=submitted["batch_id"], limit=10)
    fetched = service.get_media_ingest_job(jobs[0]["job_id"])
    reloaded = LocalMediaReadingService(db).get_media_ingest_job(jobs[1]["job_id"])
    file_media = db.get_media_by_id(jobs[0]["result"]["media_id"])
    url_media = db.get_media_by_id(jobs[1]["result"]["media_id"])
    cancel = service.cancel_media_ingest_job(jobs[0]["job_id"], reason="duplicate")
    batch_cancel = service.cancel_media_ingest_jobs_batch(batch_id=submitted["batch_id"], reason="duplicate")

    assert submitted["batch_id"].startswith("local-batch-")
    assert [job["source_kind"] for job in jobs] == ["file", "url"]
    assert [job["status"] for job in jobs] == ["completed", "completed"]
    assert listed["batch_id"] == submitted["batch_id"]
    assert [job["job_id"] for job in listed["jobs"]] == [jobs[0]["job_id"], jobs[1]["job_id"]]
    assert fetched["source"] == str(local_file)
    assert reloaded["source"] == "https://example.com/remote-doc"
    assert file_media["title"] == "Imported Document"
    assert "# Local Article" in file_media["content"]
    assert url_media["url"] == "https://example.com/remote-doc"
    assert "Imported local media URL" in url_media["content"]
    assert db.fetch_keywords_for_media_batch([file_media["id"], url_media["id"]]) == {
        file_media["id"]: ["ingest", "local"],
        url_media["id"]: ["ingest", "local"],
    }
    assert cancel["success"] is False
    assert cancel["status"] == "completed"
    assert batch_cancel["already_terminal"] == 2


@pytest.mark.asyncio
async def test_local_service_streams_media_ingest_job_snapshot(memory_db_factory, tmp_path):
    local_file = tmp_path / "article.md"
    local_file.write_text("Body text", encoding="utf-8")
    service = LocalMediaReadingService(memory_db_factory())
    submitted = service.submit_media_ingest_jobs(media_type="document", file_paths=[str(local_file)])

    events = [
        event
        async for event in service.stream_media_ingest_job_events(batch_id=submitted["batch_id"], after_id=0)
    ]

    assert events[0]["event"] == "snapshot"
    assert events[0]["data"]["batch_id"] == submitted["batch_id"]
    assert events[0]["data"]["jobs"][0]["job_id"] == submitted["jobs"][0]["job_id"]
    assert events[1]["event"] == "job"
    assert events[1]["data"]["event_type"] == "job.completed"


def test_local_service_dispatches_media_ingest_job_notifications(memory_db_factory, tmp_path):
    db = memory_db_factory()
    notifications = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(store=notifications)
    service = LocalMediaReadingService(db, notification_dispatch_service=dispatcher)

    result = service.submit_media_ingest_jobs(
        media_type="article",
        urls=["https://example.com/ok"],
        file_paths=[str(tmp_path / "missing.txt")],
    )

    rows = notifications.list_notifications(limit=10, category="media")
    assert len(rows) == 1
    assert rows[0]["source_backend"] == "local"
    assert rows[0]["source_entity_kind"] == "media_ingest_batch"
    assert rows[0]["source_entity_id"] == result["batch_id"]
    assert rows[0]["severity"] == "warning"
    assert rows[0]["payload"]["completed"] == 1
    assert rows[0]["payload"]["failed"] == 1


def test_local_service_provides_document_workspace_helpers(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        url="local://document/workspace",
        title="Workspace Doc",
        media_type="document",
        content=(
            "# Intro\n"
            "Intro body.\n\n"
            "## Details\n"
            "Details body.\n\n"
            "# References\n"
            "Doe 2024. DOI:10.1000/example\n"
        ),
        keywords=["workspace"],
        overwrite=True,
    )
    service = LocalMediaReadingService(db)

    outline = service.get_document_outline(media_id)
    figures = service.get_document_figures(media_id)
    navigation = service.get_media_navigation(media_id, max_depth=2)
    section = service.get_media_navigation_content(media_id, "heading-1", content_format="markdown")
    references = service.get_document_references(media_id, search="doi")
    insights = service.generate_document_insights(media_id, categories=["summary"], max_content_length=40)
    created = service.create_document_annotation(
        media_id,
        location="heading-1",
        text="Intro body.",
        color="green",
        note="Keep this",
    )
    listed = service.list_document_annotations(media_id)
    updated = service.update_document_annotation(media_id, created["id"], text="Updated", color="yellow")
    synced = service.sync_document_annotations(
        media_id,
        annotations=[{"location": "heading-2", "text": "Details body."}],
        client_ids=["client-1"],
    )
    deleted = service.delete_document_annotation(media_id, created["id"])

    assert outline["has_outline"] is True
    assert [entry["title"] for entry in outline["outline"]] == ["Intro", "Details", "References"]
    assert figures == {"media_id": media_id, "has_figures": False, "figures": [], "total_count": 0}
    assert navigation["available"] is True
    assert navigation["nodes"][0]["id"] == "heading-1"
    assert section["title"] == "Intro"
    assert "Intro body." in section["content"]
    assert references["has_references"] is True
    assert references["references"][0]["raw_text"] == "Doe 2024. DOI:10.1000/example"
    assert insights["insights"][0]["category"] == "summary"
    assert "Intro body." in insights["insights"][0]["content"]
    assert created["id"].startswith("local-highlight-")
    assert listed["total_count"] == 1
    assert updated["text"] == "Updated"
    assert synced["id_mapping"] == {"client-1": synced["annotations"][0]["id"]}
    assert deleted == {"deleted": True}


def test_local_service_extracts_document_figures_from_markdown_and_html(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        url="local://document/figures",
        title="Figure Doc",
        media_type="document",
        content=(
            "# Figures\n"
            "![Architecture](images/arch.png \"System architecture\")\n\n"
            "<img src=\"https://example.com/chart.jpg\" alt=\"Chart\" width=\"120\" height=\"80\">\n"
            "<img src=\"https://example.com/tiny.gif\" alt=\"Tiny\" width=\"20\" height=\"20\">\n"
        ),
        keywords=["workspace"],
        overwrite=True,
    )
    service = LocalMediaReadingService(db)

    figures = service.get_document_figures(media_id, min_size=50)

    assert figures["has_figures"] is True
    assert figures["total_count"] == 2
    assert figures["figures"][0] == {
        "id": "local-figure-1",
        "page": 1,
        "width": 50,
        "height": 50,
        "format": "png",
        "data_url": None,
        "caption": "System architecture",
        "source_url": "images/arch.png",
        "alt_text": "Architecture",
        "line": 2,
    }
    assert figures["figures"][1]["source_url"] == "https://example.com/chart.jpg"
    assert figures["figures"][1]["width"] == 120
    assert figures["figures"][1]["height"] == 80
    assert figures["figures"][1]["format"] == "jpg"


def test_local_service_ingests_web_content_without_server(memory_db_factory, monkeypatch):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    def fake_fetch(url, *, timeout=15, user_agent=None):
        return {
            "url": url,
            "title": "Fetched Article",
            "content": "Fetched body text",
            "author": "Ada",
            "metadata": {"content_type": "text/html"},
            "extraction_successful": True,
        }

    monkeypatch.setattr(service, "_fetch_web_content_url", fake_fetch, raising=False)

    result = service.ingest_web_content(
        urls=["https://example.com/article"],
        titles=["Override Title"],
        authors=["Grace"],
        keywords=["web", "local"],
        perform_analysis=False,
        perform_chunking=False,
    )
    media = db.get_media_by_id(result["media_ids"][0])

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["results"][0]["title"] == "Override Title"
    assert result["results"][0]["author"] == "Grace"
    assert result["results"][0]["extraction_successful"] is True
    assert media["url"] == "https://example.com/article"
    assert media["title"] == "Override Title"
    assert media["content"] == "Fetched body text"
    assert db.fetch_keywords_for_media_batch([media["id"]]) == {media["id"]: ["local", "web"]}


def test_local_service_dispatches_web_content_ingest_notifications(memory_db_factory, tmp_path, monkeypatch):
    db = memory_db_factory()
    notifications = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(store=notifications)
    service = LocalMediaReadingService(db, notification_dispatch_service=dispatcher)

    def fake_fetch(url, *, timeout=15, user_agent=None):
        if url.endswith("/bad"):
            raise RuntimeError("fetch failed")
        return {
            "url": url,
            "title": "Fetched Article",
            "content": "Fetched body text",
            "metadata": {},
            "extraction_successful": True,
        }

    monkeypatch.setattr(service, "_fetch_web_content_url", fake_fetch, raising=False)

    result = service.ingest_web_content(
        urls=["https://example.com/good", "https://example.com/bad"],
        keywords=["web"],
    )

    rows = notifications.list_notifications(limit=10, category="media")
    assert result["count"] == 1
    assert len(rows) == 1
    assert rows[0]["source_backend"] == "local"
    assert rows[0]["source_entity_kind"] == "web_content_ingest"
    assert rows[0]["severity"] == "warning"
    assert rows[0]["payload"]["requested"] == 2
    assert rows[0]["payload"]["completed"] == 1
    assert rows[0]["payload"]["failed"] == 1


def test_local_service_processes_legacy_web_scraping_request(memory_db_factory, monkeypatch):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    monkeypatch.setattr(
        service,
        "_fetch_web_content_url",
        lambda url, **kwargs: {
            "url": url,
            "title": "Legacy Article",
            "content": "Legacy body",
            "metadata": {},
            "extraction_successful": True,
        },
        raising=False,
    )

    result = service.process_web_scraping(
        {
            "url_input": "https://example.com/legacy",
            "custom_titles": "Legacy Override",
            "keywords": "legacy,local",
            "mode": "persist",
        }
    )

    media = db.get_media_by_id(result["media_ids"][0])
    assert result["status"] == "success"
    assert result["results"][0]["title"] == "Legacy Override"
    assert media["url"] == "https://example.com/legacy"
    assert db.fetch_keywords_for_media_batch([media["id"]]) == {media["id"]: ["legacy", "local"]}


def test_local_service_crawls_internal_web_links_with_depth_and_page_limits(memory_db_factory, monkeypatch):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    fetched_urls: list[str] = []

    pages = {
        "https://example.com/root": {
            "title": "Root",
            "content": "Root body",
            "raw_html": """
                <html><body>
                    <a href="/first#fragment">First</a>
                    <a href="https://external.example/out">External</a>
                </body></html>
            """,
        },
        "https://example.com/first": {
            "title": "First",
            "content": "First body",
            "raw_html": '<html><body><a href="/second">Second</a></body></html>',
        },
        "https://example.com/second": {
            "title": "Second",
            "content": "Second body",
            "raw_html": "<html><body>Second</body></html>",
        },
    }

    def fake_fetch(url, *, timeout=15, user_agent=None):
        fetched_urls.append(url)
        page = pages[url]
        return {
            "url": url,
            "title": page["title"],
            "content": page["content"],
            "raw_html": page["raw_html"],
            "metadata": {},
            "extraction_successful": True,
        }

    monkeypatch.setattr(service, "_fetch_web_content_url", fake_fetch, raising=False)

    result = service.ingest_web_content(
        urls=["https://example.com/root"],
        scrape_method="url_level",
        max_depth=1,
        max_pages=5,
        include_external=False,
        keywords=["crawl"],
    )

    assert [item["url"] for item in result["results"]] == [
        "https://example.com/root",
        "https://example.com/first",
    ]
    assert fetched_urls == ["https://example.com/root", "https://example.com/first"]
    assert result["count"] == 2
    assert all(db.get_media_by_id(media_id)["type"] == "web" for media_id in result["media_ids"])


def test_local_service_processes_legacy_web_scraping_crawl_options(memory_db_factory, monkeypatch):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    pages = {
        "https://example.com/root": {
            "title": "Root",
            "content": "Root body",
            "raw_html": '<html><body><a href="https://external.example/out">External</a></body></html>',
        },
        "https://external.example/out": {
            "title": "External",
            "content": "External body",
            "raw_html": "<html><body>External</body></html>",
        },
    }

    monkeypatch.setattr(
        service,
        "_fetch_web_content_url",
        lambda url, **kwargs: {
            "url": url,
            "title": pages[url]["title"],
            "content": pages[url]["content"],
            "raw_html": pages[url]["raw_html"],
            "metadata": {},
            "extraction_successful": True,
        },
        raising=False,
    )

    result = service.process_web_scraping(
        {
            "scrape_method": "url_level",
            "url_input": "https://example.com/root",
            "url_level": 1,
            "max_pages": 2,
            "include_external": True,
            "keywords": "legacy,crawl",
        }
    )

    assert [item["url"] for item in result["results"]] == [
        "https://example.com/root",
        "https://external.example/out",
    ]
    assert result["count"] == 2
