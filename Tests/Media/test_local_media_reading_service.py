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


def test_local_service_search_media_uses_db_backed_saved_filter(memory_db_factory):
    db = memory_db_factory()
    kept_id, _, _ = db.add_media_with_keywords(title="Keep", content="A", media_type="article", keywords=[])
    other_id, _, _ = db.add_media_with_keywords(title="Drop", content="B", media_type="article", keywords=[])
    db.save_media_to_read_it_later(kept_id)

    service = LocalMediaReadingService(db)
    payload = service.search_media(read_it_later_only=True)

    assert [item["id"] for item in payload["items"]] == [kept_id]
    assert all(item["id"] != other_id for item in payload["items"])
