import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database
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
