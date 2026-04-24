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
