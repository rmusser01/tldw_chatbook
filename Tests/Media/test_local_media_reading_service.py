from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
    assert synced["status"] == "queued"
    assert job["source_id"] == created["id"]
    assert jobs["jobs"][0]["id"] == job["id"]
    assert cancelled["status"] == "cancelled"
    assert deleted["deleted"] is True
    with pytest.raises(KeyError):
        service.get_ingestion_source(created["id"])


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


def test_local_service_submit_ingest_jobs_queues_url_and_file_jobs(memory_db_factory, tmp_path):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("pdf placeholder", encoding="utf-8")

    submitted = service.submit_ingest_jobs(
        media_type="pdf",
        urls=["https://example.com/a.pdf"],
        file_paths=[str(file_path)],
        keywords=["paper"],
    )

    assert submitted["batch_id"].startswith("local-batch-")
    assert [job["source_kind"] for job in submitted["jobs"]] == ["url", "file"]
    assert service.get_ingest_job(submitted["jobs"][0]["id"])["status"] == "queued"
