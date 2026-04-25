import asyncio
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import zipfile

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database


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


def test_local_service_persists_saved_searches_and_note_links(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    service = LocalMediaReadingService(db)

    created = service.create_saved_search(name=" Morning ", query={"q": "ai"}, sort="updated_desc")
    listed = service.list_saved_searches(limit=25, offset=0)
    updated = service.update_saved_search(created["id"], name="Updated", query={"q": "ml"})
    linked = service.link_note(media_id, "note-1")
    links = service.list_note_links(media_id)
    unlinked = service.unlink_note(media_id, "note-1")
    deleted = service.delete_saved_search(created["id"])

    assert created["name"] == "Morning"
    assert created["query"] == {"q": "ai"}
    assert listed["items"][0]["id"] == created["id"]
    assert listed["total"] == 1
    assert updated["name"] == "Updated"
    assert updated["query"] == {"q": "ml"}
    assert updated["sort"] == "updated_desc"
    assert linked["item_id"] == media_id
    assert linked["note_id"] == "note-1"
    assert linked["created_at"] is not None
    assert links["links"] == [linked]
    assert unlinked == {"deleted": True, "item_id": media_id, "note_id": "note-1"}
    assert service.list_note_links(media_id)["links"] == []
    assert deleted == {"deleted": True, "id": created["id"]}


def test_local_service_exports_saved_reading_items(memory_db_factory):
    db = memory_db_factory()
    saved_id, _, _ = db.add_media_with_keywords(
        title="Saved",
        content="Saved text",
        media_type="article",
        url="https://example.com/saved",
        keywords=["ai"],
    )
    other_id, _, _ = db.add_media_with_keywords(
        title="Other",
        content="Other text",
        media_type="article",
        url="https://example.org/other",
        keywords=["ml"],
    )
    db.save_media_to_read_it_later(saved_id)
    service = LocalMediaReadingService(db)

    exported = service.export_reading_items(format="jsonl", include_metadata=True, include_text=True)
    zip_export = service.export_reading_items(format="zip", include_metadata=False)

    rows = [json.loads(line) for line in exported["content"].decode("utf-8").splitlines()]
    assert exported["content_type"] == "application/x-ndjson"
    assert exported["filename"].endswith(".jsonl")
    assert rows[0]["id"] == saved_id
    assert rows[0]["title"] == "Saved"
    assert rows[0]["status"] == "saved"
    assert rows[0]["text"] == "Saved text"
    assert "metadata" in rows[0]
    assert all(row["id"] != other_id for row in rows)

    with zipfile.ZipFile(io.BytesIO(zip_export["content"]), "r") as archive:
        zipped_rows = [
            json.loads(line)
            for line in archive.read("reading_export.jsonl").decode("utf-8").splitlines()
        ]
    assert zip_export["content_type"] == "application/zip"
    assert zip_export["filename"].endswith(".zip")
    assert zipped_rows[0]["id"] == saved_id
    assert "metadata" not in zipped_rows[0]


def test_local_service_creates_durable_reading_archive_snapshot(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Saved",
        content="Saved text",
        media_type="article",
        url="https://example.com/saved",
        keywords=["ai"],
    )
    service = LocalMediaReadingService(db)

    archive = service.create_reading_archive(media_id, format="md", source="text", title="Snapshot")

    assert archive["output_id"] > 0
    assert archive["title"].startswith("Snapshot (archive ")
    assert archive["format"] == "md"
    assert archive["storage_path"].endswith(".md")
    assert archive["download_url"].startswith("local://reading-archives/")

    row = db.get_connection().execute(
        "SELECT * FROM local_reading_archives WHERE id = ?",
        (archive["output_id"],),
    ).fetchone()
    assert row is not None
    assert row["item_id"] == media_id
    assert row["content"].startswith("# Snapshot\n")
    assert "Saved text" in row["content"]


def test_local_service_bulk_updates_reading_status_and_tags(memory_db_factory):
    db = memory_db_factory()
    first_id, _, _ = db.add_media_with_keywords(
        title="First",
        content="First text",
        media_type="article",
        keywords=["old"],
    )
    second_id, _, _ = db.add_media_with_keywords(
        title="Second",
        content="Second text",
        media_type="article",
        keywords=["old"],
    )
    service = LocalMediaReadingService(db)

    saved = service.bulk_update_reading_items(
        item_ids=[first_id, second_id, first_id],
        action="set_status",
        status="saved",
    )
    tagged = service.bulk_update_reading_items(
        item_ids=[first_id],
        action="replace_tags",
        tags=["AI", " research "],
    )

    assert saved == {
        "total": 2,
        "succeeded": 2,
        "failed": 0,
        "results": [
            {"item_id": first_id, "success": True, "error": None},
            {"item_id": second_id, "success": True, "error": None},
        ],
    }
    assert db.get_media_read_it_later_state(first_id)["is_read_it_later"] is True
    assert db.get_media_read_it_later_state(second_id)["is_read_it_later"] is True
    assert tagged["succeeded"] == 1
    assert db.fetch_keywords_for_media_batch([first_id])[first_id] == ["ai", "research"]
    archived = service.bulk_update_reading_items(
        item_ids=[first_id],
        action="set_status",
        status="archived",
    )
    assert archived["succeeded"] == 1
    assert db.get_media_read_it_later_state(first_id) is None
    assert db.get_media_read_it_later_state(second_id)["is_read_it_later"] is True


def test_local_service_generates_extractive_reading_summary(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Long Read",
        content="First sentence explains the topic. Second sentence adds context. Third sentence has details.",
        media_type="article",
        url="https://example.com/long",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    summary = service.summarize_reading_item(media_id)

    assert summary["item_id"] == media_id
    assert summary["provider"] == "local-extractive"
    assert summary["model"] == "first-passages"
    assert summary["summary"].startswith("First sentence explains the topic.")
    assert summary["citations"] == [
        {
            "item_id": media_id,
            "url": "https://example.com/long",
            "canonical_url": "https://example.com/long",
            "title": "Long Read",
            "source": "reading",
        }
    ]
    assert summary["generated_at"] is not None


@pytest.mark.asyncio
async def test_local_service_generates_reading_tts_with_injected_generator(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Listen",
        content="First sentence. Second sentence.",
        media_type="article",
        url="https://example.com/listen",
        keywords=[],
    )
    calls = []

    async def fake_tts_generator(**kwargs):
        calls.append(kwargs)
        return b"audio-bytes"

    service = LocalMediaReadingService(db, tts_audio_generator=fake_tts_generator)

    audio = await service.tts_reading_item(
        media_id,
        model="kokoro",
        voice="af_heart",
        response_format="wav",
        max_chars=15,
    )

    assert calls[0]["text"] == "First sentence."
    assert calls[0]["model"] == "kokoro"
    assert calls[0]["voice"] == "af_heart"
    assert audio == {
        "item_id": media_id,
        "content": b"audio-bytes",
        "content_type": "audio/wav",
        "content_disposition": f"attachment; filename=reading_tts_{media_id}.wav",
        "filename": f"reading_tts_{media_id}.wav",
    }


def test_local_service_executes_csv_reading_import_jobs(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    import_path = tmp_path / "pocket.csv"
    import_path.write_text("title,url\nSaved,https://example.com\n", encoding="utf-8")

    submitted = service.import_reading_items(str(import_path), source="pocket", merge_tags=False)
    listed = service.list_reading_import_jobs(status="completed", limit=25, offset=0)
    detail = service.get_reading_import_job(submitted["job_id"])

    assert submitted["job_id"] == detail["job_id"]
    assert submitted["status"] == "completed"
    assert submitted["job_uuid"]
    assert listed["total"] == 1
    assert listed["jobs"][0]["job_id"] == submitted["job_id"]
    assert detail["status"] == "completed"
    assert detail["progress_percent"] == 100
    assert detail["progress_message"] == "Completed"
    assert detail["result"]["source"] == "pocket"
    assert detail["result"]["imported"] == 1


def test_local_service_executes_jsonl_reading_import_and_materializes_saved_items(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    import_path = tmp_path / "reading.jsonl"
    import_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "title": "Saved A",
                        "url": "https://example.com/a",
                        "text": "Alpha body",
                        "tags": ["AI", "research"],
                        "status": "saved",
                    }
                ),
                json.dumps(
                    {
                        "title": "Saved B",
                        "url": "https://example.com/b",
                        "text": "Beta body",
                        "tags": ["notes"],
                        "status": "saved",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    submitted = service.import_reading_items(str(import_path), source="jsonl", merge_tags=True)
    detail = service.get_reading_import_job(submitted["job_id"])
    saved = service.search_media(read_it_later_only=True, limit=10)

    assert submitted["status"] == "completed"
    assert detail["status"] == "completed"
    assert detail["progress_percent"] == 100
    assert detail["result"] == {
        "source": "jsonl",
        "imported": 2,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }
    assert {item["title"] for item in saved["items"]} == {"Saved A", "Saved B"}
    first = db.get_media_by_url("https://example.com/a")
    assert first is not None
    assert first["content"] == "Alpha body"
    assert db.fetch_keywords_for_media_batch([first["id"]])[first["id"]] == ["ai", "research"]
    assert db.get_media_read_it_later_state(first["id"])["is_read_it_later"] is True


def test_local_service_persists_ingestion_sources_and_sync_jobs(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    source_path = tmp_path / "source"
    source_path.mkdir()

    created = service.create_ingestion_source(
        source_type="local_directory",
        sink_type="media",
        policy="canonical",
        config={"path": str(source_path)},
    )
    listed = service.list_ingestion_sources()
    detail = service.get_ingestion_source(created["id"])
    patched = service.patch_ingestion_source(created["id"], enabled=False, schedule_enabled=True)
    items = service.list_ingestion_source_items(created["id"])
    synced = service.trigger_ingestion_source_sync(created["id"])
    job = service.get_ingest_job(synced["job_id"])
    jobs = service.list_ingest_jobs(job["batch_id"])
    cancelled = service.cancel_ingest_job(job["id"], reason="user requested")
    deleted = service.delete_ingestion_source(created["id"])

    assert listed[0]["id"] == created["id"]
    assert detail["source_type"] == "local_directory"
    assert patched["enabled"] is False
    assert patched["schedule_enabled"] is True
    assert items == []
    assert synced["status"] == "completed"
    assert job["source_id"] == created["id"]
    assert job["status"] == "completed"
    assert job["result"]["scanned"] == 0
    assert jobs["jobs"][0]["id"] == job["id"]
    assert cancelled["success"] is False
    assert cancelled["status"] == "completed"
    assert deleted["deleted"] is True
    with pytest.raises(KeyError):
        service.get_ingestion_source(created["id"])


def test_local_service_syncs_local_directory_ingestion_source_items(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    source_path = tmp_path / "source"
    nested_path = source_path / "nested"
    nested_path.mkdir(parents=True)
    (source_path / "alpha.md").write_text("# Alpha\n", encoding="utf-8")
    (nested_path / "beta.txt").write_text("Beta\n", encoding="utf-8")
    source = service.create_ingestion_source(
        source_type="local_directory",
        sink_type="media",
        policy="canonical",
        config={"path": str(source_path)},
    )

    synced = service.trigger_ingestion_source_sync(source["id"])
    job = service.get_ingest_job(synced["job_id"])
    items = service.list_ingestion_source_items(source["id"])

    assert synced["status"] == "completed"
    assert job["status"] == "completed"
    assert job["result"] == {
        "source_id": source["id"],
        "source_type": "local_directory",
        "scanned": 2,
        "created": 2,
        "updated": 0,
        "missing": 0,
        "errors": [],
    }
    assert {item["normalized_relative_path"] for item in items} == {"alpha.md", "nested/beta.txt"}
    assert {item["sync_status"] for item in items} == {"pending"}
    assert all(item["content_hash"] for item in items)
    detail = service.get_ingestion_source(source["id"])
    assert detail["active_job_id"] == str(job["id"])
    assert detail["last_sync_status"] == "completed"
    assert detail["last_sync_completed_at"] is not None


def test_local_service_uploads_archive_snapshot_into_source_items(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    archive_path = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("alpha.md", "# Alpha\n")
        archive.writestr("nested/beta.txt", "Beta\n")
    source = service.create_ingestion_source(
        source_type="archive_snapshot",
        sink_type="media",
        policy="canonical",
        config={},
    )

    uploaded = service.upload_ingestion_source_archive(source["id"], str(archive_path))
    job = service.get_ingest_job(uploaded["job_id"])
    items = service.list_ingestion_source_items(source["id"])

    assert uploaded["status"] == "completed"
    assert uploaded["snapshot_status"] == "materialized"
    assert job["status"] == "completed"
    assert job["result"]["source_type"] == "archive_snapshot"
    assert job["result"]["scanned"] == 2
    assert {item["normalized_relative_path"] for item in items} == {"alpha.md", "nested/beta.txt"}
    assert {item["sync_status"] for item in items} == {"pending"}
    assert all(item["content_hash"] for item in items)


def test_local_service_syncs_git_repository_source_items(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()
    (repo_path / "README.md").write_text("# Repo\n", encoding="utf-8")
    (repo_path / "docs").mkdir()
    (repo_path / "docs" / "guide.txt").write_text("Guide\n", encoding="utf-8")
    source = service.create_ingestion_source(
        source_type="git_repository",
        sink_type="media",
        policy="canonical",
        config={"repo_url": str(repo_path)},
    )

    synced = service.trigger_ingestion_source_sync(source["id"])
    job = service.get_ingest_job(synced["job_id"])
    items = service.list_ingestion_source_items(source["id"])

    assert synced["status"] == "completed"
    assert job["status"] == "completed"
    assert job["result"]["source_type"] == "git_repository"
    assert job["result"]["scanned"] == 2
    assert {item["normalized_relative_path"] for item in items} == {"README.md", "docs/guide.txt"}
    assert all(".git" not in item["normalized_relative_path"] for item in items)


def test_local_service_reattaches_detached_ingestion_source_item(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    source = service.create_ingestion_source(
        source_type="local_directory",
        sink_type="notes",
        policy="canonical",
        config={"path": "/tmp/source"},
    )
    now = db._get_current_utc_timestamp_str()
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO local_ingestion_source_items (
                source_id, normalized_relative_path, content_hash, sync_status,
                binding_json, present_in_source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source["id"],
                "note.md",
                "old-hash",
                "conflict_detached",
                '{"note_id": "note-1", "current_version": 2}',
                1,
                now,
                now,
            ),
        )
        item_id = cursor.lastrowid

    reattached = service.reattach_ingestion_source_item(source["id"], item_id)

    assert reattached["id"] == item_id
    assert reattached["sync_status"] == "sync_managed"
    assert reattached["content_hash"] is None
    assert reattached["binding"]["note_id"] == "note-1"
    assert reattached["binding"]["sync_status"] == "sync_managed"


def test_local_service_submit_ingest_jobs_executes_url_article_and_file_jobs(memory_db_factory, tmp_path):
    db = memory_db_factory()
    def fake_scraper(url, *, custom_cookies=None):
        return {
            "url": url,
            "title": "Remote Article",
            "author": "Grace",
            "content": "Remote article body",
            "date": "2026-04-25T00:00:00Z",
            "extraction_successful": True,
        }

    service = LocalMediaReadingService(db, url_article_scraper=fake_scraper)
    file_path = tmp_path / "doc.txt"
    file_path.write_text("text placeholder", encoding="utf-8")

    submitted = service.submit_ingest_jobs(
        media_type="article",
        urls=["https://example.com/a.txt"],
        file_paths=[str(file_path)],
        keywords=["paper"],
    )

    assert submitted["batch_id"].startswith("local-batch-")
    assert [job["source_kind"] for job in submitted["jobs"]] == ["url", "file"]
    assert service.get_ingest_job(submitted["jobs"][0]["id"])["status"] == "completed"
    assert service.get_ingest_job(submitted["jobs"][1]["id"])["status"] == "completed"


def test_local_service_executes_url_article_ingest_jobs_with_injected_scraper(memory_db_factory):
    db = memory_db_factory()
    calls = []

    def fake_scraper(url, *, custom_cookies=None):
        calls.append((url, custom_cookies))
        return {
            "url": url,
            "title": "Saved URL",
            "author": "Ada",
            "content": "URL article content",
            "summary": "URL summary",
            "date": "2026-04-25T00:00:00Z",
            "extraction_successful": True,
        }

    service = LocalMediaReadingService(db, url_article_scraper=fake_scraper)

    submitted = service.submit_ingest_jobs(
        media_type="article",
        urls=["https://example.com/article"],
        keywords=["Read Later"],
        custom_cookies=[{"name": "session", "value": "abc"}],
    )
    job = service.get_ingest_job(submitted["jobs"][0]["id"])
    media = db.get_media_by_url("https://example.com/article")

    assert calls == [("https://example.com/article", [{"name": "session", "value": "abc"}])]
    assert submitted["jobs"][0]["status"] == "completed"
    assert job["result"]["source_kind"] == "url"
    assert job["result"]["imported"] == 1
    assert job["result"]["media_id"] == media["id"]
    assert media["title"] == "Saved URL"
    assert media["content"] == "URL article content"
    assert db.fetch_keywords_for_media_batch([media["id"]])[media["id"]] == ["read later"]


def test_local_service_executes_url_file_download_ingest_jobs_with_injected_downloader(memory_db_factory, tmp_path):
    db = memory_db_factory()
    calls = []

    def fake_downloader(url, *, media_type, options):
        calls.append((url, media_type, options.get("keywords")))
        downloaded = tmp_path / "downloaded.txt"
        downloaded.write_text("Downloaded text body", encoding="utf-8")
        return {"path": str(downloaded), "cleanup": False}

    service = LocalMediaReadingService(db, url_file_downloader=fake_downloader)

    submitted = service.submit_ingest_jobs(
        media_type="plaintext",
        urls=["https://example.com/downloaded.txt"],
        keywords=["Download"],
    )
    job = service.get_ingest_job(submitted["jobs"][0]["id"])
    media = db.get_media_by_url("https://example.com/downloaded.txt")

    assert calls == [("https://example.com/downloaded.txt", "plaintext", ["Download"])]
    assert submitted["jobs"][0]["status"] == "completed"
    assert job["result"]["source_kind"] == "url"
    assert job["result"]["downloaded_path"] == str(tmp_path / "downloaded.txt")
    assert job["result"]["media_id"] == media["id"]
    assert media["url"] == "https://example.com/downloaded.txt"
    assert media["content"] == "Downloaded text body"
    assert db.fetch_keywords_for_media_batch([media["id"]])[media["id"]] == ["download"]


@pytest.mark.asyncio
async def test_local_service_default_url_scraper_is_safe_from_async_scope(memory_db_factory, monkeypatch):
    from tldw_chatbook.Web_Scraping import Article_Extractor_Lib

    db = memory_db_factory()

    def fake_scrape_article_sync(url, *, custom_cookies=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("scrape_article_sync was called inside the active event loop")
        return {
            "url": url,
            "title": "Async Safe URL",
            "author": "Ada",
            "content": "Async safe URL article content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(Article_Extractor_Lib, "scrape_article_sync", fake_scrape_article_sync)
    service = LocalMediaReadingService(db)

    submitted = service.submit_ingest_jobs(
        media_type="article",
        urls=["https://example.com/async-safe"],
    )
    media = db.get_media_by_url("https://example.com/async-safe")

    assert submitted["jobs"][0]["status"] == "completed"
    assert media["title"] == "Async Safe URL"


def test_local_service_executes_local_file_ingest_jobs(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    file_path = tmp_path / "note.txt"
    file_path.write_text("Standalone file body", encoding="utf-8")

    submitted = service.submit_ingest_jobs(
        media_type="plaintext",
        file_paths=[str(file_path)],
        keywords=["Offline"],
    )
    job = service.get_ingest_job(submitted["jobs"][0]["id"])
    media = db.get_media_by_url(f"file://{file_path.absolute()}")

    assert submitted["jobs"][0]["status"] == "completed"
    assert job["status"] == "completed"
    assert job["result"]["source_kind"] == "file"
    assert job["result"]["imported"] == 1
    assert job["result"]["media_id"] == media["id"]
    assert media["title"] == "note"
    assert media["content"] == "Standalone file body"
    assert db.fetch_keywords_for_media_batch([media["id"]])[media["id"]] == ["offline"]
