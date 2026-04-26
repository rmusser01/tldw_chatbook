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


def test_local_service_direct_media_management_round_trips(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        url="https://example.com/report.md",
        title="Report",
        content="Body text with DOI 10.1234/example",
        media_type="document",
        keywords=["draft", "research"],
        author="Ada",
    )
    other_id, _, _ = db.add_media_with_keywords(
        title="Other",
        content="Other body",
        media_type="article",
        keywords=["misc"],
    )
    service = LocalMediaReadingService(db)

    listed = service.list_media_items(page=1, results_per_page=10, include_keywords=True)
    detail = service.get_media_item(media_id, include_content=False)
    updated = service.update_media_item(media_id, title="Renamed", keywords=["reviewed"])
    keyword_suggestions = service.list_media_keywords(query="view", limit=5)
    metadata_matches = service.search_media_metadata(field="title", value="Renamed", per_page=5)
    identifier_matches = service.get_media_by_identifier(url="https://example.com/report.md")
    deleted = service.delete_media_item(media_id)
    trash = service.list_media_trash(page=1, results_per_page=10, include_keywords=True)
    restored = service.restore_media_item(media_id, include_content=False)
    deleted_again = service.delete_media_item(media_id)
    permanent = service.permanently_delete_media_item(media_id)
    after_permanent = service.get_media_by_identifier(url="https://example.com/report.md")

    assert listed["pagination"]["total_items"] == 2
    assert any(item["id"] == media_id and item["keywords"] == ["draft", "research"] for item in listed["items"])
    assert any(item["id"] == other_id for item in listed["items"])
    assert "content" not in detail
    assert detail["keywords"] == ["draft", "research"]
    assert updated["title"] == "Renamed"
    assert updated["keywords"] == ["reviewed"]
    assert keyword_suggestions == {"keywords": ["reviewed"]}
    assert [item["id"] for item in metadata_matches["items"]] == [media_id]
    assert identifier_matches["items"][0]["id"] == media_id
    assert deleted == {"ok": True, "media_id": media_id}
    assert trash["items"][0]["id"] == media_id
    assert trash["items"][0]["keywords"] == ["reviewed"]
    assert restored["id"] == media_id
    assert restored["is_trash"] in {0, False}
    assert deleted_again == {"ok": True, "media_id": media_id}
    assert permanent == {"ok": True, "media_id": media_id}
    assert after_permanent["items"] == []


def test_local_service_downloads_local_media_files_and_stored_content(memory_db_factory, tmp_path):
    db = memory_db_factory()
    source_file = tmp_path / "source.md"
    source_file.write_text("# Stored file\n\nBody", encoding="utf-8")
    file_media_id, _, _ = db.add_media_with_keywords(
        url=source_file.as_uri(),
        title="Stored File",
        content="indexed copy",
        media_type="document",
        keywords=[],
    )
    content_media_id, _, _ = db.add_media_with_keywords(
        title="Stored Content",
        content="Only in database",
        media_type="document",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    file_check = service.check_media_file(file_media_id)
    file_download = service.download_media_file(file_media_id)
    content_check = service.check_media_file(content_media_id)
    content_download = service.download_media_file(content_media_id)

    assert file_check["available"] is True
    assert file_check["source"] == "file_path"
    assert file_download["content"] == b"# Stored file\n\nBody"
    assert file_download["filename"] == "source.md"
    assert content_check["available"] is True
    assert content_check["source"] == "stored_content"
    assert content_download["content"] == b"Only in database"
    assert content_download["content_type"] == "text/plain; charset=utf-8"


def test_local_service_add_media_persists_url_content_and_files(memory_db_factory, tmp_path):
    db = memory_db_factory()
    source_file = tmp_path / "source.md"
    source_file.write_text("# File body\n\nStored locally", encoding="utf-8")
    service = LocalMediaReadingService(db)

    result = service.add_media(
        media_type="document",
        urls=["https://example.com/report.md"],
        file_paths=[str(source_file)],
        title="Research Report",
        author="Ada",
        keywords="AI, Research",
        content="URL supplied body",
        overwrite_existing=True,
    )

    assert result["status"] == "success"
    assert result["backend"] == "local"
    assert result["processed_count"] == 2
    assert result["failed_count"] == 0
    assert [item["source"] for item in result["items"]] == ["url", "file_path"]
    url_detail = service.get_media_item(result["items"][0]["media_id"], include_content=True)
    file_detail = service.get_media_detail(result["items"][1]["media_id"])
    assert url_detail["url"] == "https://example.com/report.md"
    assert url_detail["content"] == "URL supplied body"
    assert url_detail["keywords"] == ["ai", "research"]
    assert file_detail["url"] == source_file.as_uri()
    assert file_detail["content"] == "# File body\n\nStored locally"
    assert service.check_media_file(result["items"][1]["media_id"])["source"] == "file_path"


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


def test_local_service_saves_direct_reading_item_with_content(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    item = service.save_reading_item(
        url="https://example.com/local-direct",
        title="Local Direct",
        tags=["AI", "reading"],
        status="saved",
        content="Direct local reading body",
    )

    stored = db.get_media_by_url("https://example.com/local-direct")
    assert stored is not None
    assert item["id"] == stored["id"]
    assert item["title"] == "Local Direct"
    assert item["media_type"] == "article"
    assert item["url"] == "https://example.com/local-direct"
    assert item["is_read_it_later"] is True
    assert item["saved_at"] is not None
    assert db.get_media_read_it_later_state(stored["id"])["is_read_it_later"] is True


def test_local_service_saves_direct_reading_item_with_injected_scraper(memory_db_factory):
    db = memory_db_factory()
    calls = []

    def fake_scraper(url, *, custom_cookies=None):
        calls.append((url, custom_cookies))
        return {
            "url": url,
            "title": "Scraped Title",
            "content": "Scraped local reading body",
            "author": "Local Author",
            "keywords": ["scraped"],
        }

    service = LocalMediaReadingService(db, url_article_scraper=fake_scraper)

    item = service.save_reading_item(
        url="https://example.com/scraped",
        title="Caller Title",
        tags=["manual"],
    )

    stored = db.get_media_by_url("https://example.com/scraped")
    assert stored is not None
    assert calls == [("https://example.com/scraped", None)]
    assert item["id"] == stored["id"]
    assert item["title"] == "Caller Title"
    assert item["author"] == "Local Author"
    assert item["is_read_it_later"] is True


def test_local_service_direct_reading_item_archived_status_clears_saved_state(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    saved = service.save_reading_item(
        url="https://example.com/status",
        title="Saved First",
        status="saved",
        content="Saved body",
    )
    archived = service.save_reading_item(
        url="https://example.com/status",
        title="Archived Next",
        status="archived",
        content="Archived body",
    )

    assert saved["is_read_it_later"] is True
    assert archived["id"] == saved["id"]
    assert archived["is_read_it_later"] is False
    assert db.get_media_read_it_later_state(archived["id"]) is None


def test_local_service_persists_reading_highlights(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Highlighted",
        content="Important local content",
        media_type="article",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    created = service.create_highlight(
        media_id,
        quote="Important",
        start_offset=0,
        end_offset=9,
        color="yellow",
        note="review",
    )
    listed = service.list_highlights(media_id)
    updated = service.update_highlight(created["id"], color="blue", note="done", state="stale")
    deleted = service.delete_highlight(created["id"])

    assert created["item_id"] == media_id
    assert created["quote"] == "Important"
    assert created["start_offset"] == 0
    assert created["end_offset"] == 9
    assert created["anchor_strategy"] == "fuzzy_quote"
    assert created["state"] == "active"
    assert listed == [created]
    assert updated["color"] == "blue"
    assert updated["note"] == "done"
    assert updated["state"] == "stale"
    assert deleted == {"success": True}
    assert service.list_highlights(media_id) == []


def test_local_service_persists_document_annotations(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Annotated",
        content="Annotated local content",
        media_type="document",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    created = service.create_annotation(
        media_id,
        location="page:12",
        text="selected text",
        color="yellow",
        note="remember",
        annotation_type="highlight",
        chapter_title="Chapter 1",
        percentage=42.5,
    )
    listed = service.list_annotations(media_id)
    updated = service.update_annotation(media_id, created["id"], text="updated", color="blue", note="done")
    deleted = service.delete_annotation(media_id, created["id"])
    after_delete = service.list_annotations(media_id)
    synced = service.sync_annotations(
        media_id,
        annotations=[{"location": "page:13", "text": "offline note", "annotation_type": "page_note"}],
        client_ids=["client-1"],
    )

    assert created["id"].startswith("local-ann-")
    assert created["media_id"] == media_id
    assert created["location"] == "page:12"
    assert created["text"] == "selected text"
    assert created["color"] == "yellow"
    assert created["note"] == "remember"
    assert created["annotation_type"] == "highlight"
    assert created["chapter_title"] == "Chapter 1"
    assert created["percentage"] == 42.5
    assert listed["media_id"] == media_id
    assert listed["annotations"] == [created]
    assert listed["total_count"] == 1
    assert updated["text"] == "updated"
    assert updated["color"] == "blue"
    assert updated["note"] == "done"
    assert deleted == {}
    assert after_delete["total_count"] == 0
    assert synced["media_id"] == media_id
    assert synced["synced_count"] == 1
    assert synced["annotations"][0]["text"] == "offline note"
    assert synced["id_mapping"] == {"client-1": synced["annotations"][0]["id"]}


def test_local_service_builds_media_navigation_from_markdown_headings(memory_db_factory):
    db = memory_db_factory()
    content = (
        "# Opening\n"
        "Opening body.\n\n"
        "## First Beat\n"
        "Beat body.\n\n"
        "# Closing\n"
        "Closing body."
    )
    media_id, _, _ = db.add_media_with_keywords(
        title="Navigable",
        content=content,
        media_type="document",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    navigation = service.get_media_navigation(media_id, max_depth=1)
    beat_content = service.get_media_navigation_content(
        media_id,
        "heading-1",
        format="markdown",
        include_alternates=True,
    )

    assert navigation["media_id"] == media_id
    assert navigation["available"] is True
    assert navigation["source_order_used"] == ["local_markdown_headings"]
    assert [node["id"] for node in navigation["nodes"]] == ["heading-0", "heading-1", "heading-2"]
    assert navigation["nodes"][0] == {
        "id": "heading-0",
        "parent_id": None,
        "level": 0,
        "title": "Opening",
        "order": 0,
        "path_label": "Opening",
        "target_type": "char_range",
        "target_start": 0,
        "target_end": content.index("# Closing"),
        "target_href": None,
        "source": "local_markdown_headings",
        "confidence": 1.0,
    }
    assert navigation["nodes"][1]["parent_id"] == "heading-0"
    assert navigation["nodes"][1]["level"] == 1
    assert navigation["nodes"][2]["parent_id"] is None
    assert navigation["stats"] == {
        "returned_node_count": 3,
        "node_count": 3,
        "max_depth": 1,
        "truncated": False,
    }
    assert beat_content["node_id"] == "heading-1"
    assert beat_content["title"] == "First Beat"
    assert beat_content["content_format"] == "markdown"
    assert beat_content["available_formats"] == ["markdown", "plain"]
    assert beat_content["content"] == "## First Beat\nBeat body."
    assert beat_content["alternate_content"] == {"plain": "First Beat\nBeat body."}
    assert beat_content["target"]["target_type"] == "char_range"


def test_local_service_builds_generated_media_navigation_from_chunks(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Chunked",
        content="Alpha text.\n\nBeta text.",
        media_type="article",
        keywords=[],
        chunks=[
            {"text": "Alpha text.", "start_char": 0, "end_char": 11, "chunk_type": "section"},
            {"text": "Beta text.", "start_char": 13, "end_char": 23, "chunk_type": "section"},
        ],
    )
    service = LocalMediaReadingService(db)

    without_fallback = service.get_media_navigation(media_id)
    with_fallback = service.get_media_navigation(media_id, include_generated_fallback=True)

    assert without_fallback["available"] is False
    assert without_fallback["nodes"] == []
    assert with_fallback["available"] is True
    assert with_fallback["source_order_used"] == ["local_chunks"]
    assert [node["title"] for node in with_fallback["nodes"]] == ["Alpha text.", "Beta text."]
    assert [node["id"] for node in with_fallback["nodes"]] == ["chunk-0", "chunk-1"]


def test_local_service_persists_reading_digest_schedules(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    created = service.create_reading_digest_schedule(
        name="Morning",
        cron="0 8 * * *",
        timezone="America/Los_Angeles",
        enabled=True,
        require_online=False,
        format="md",
        retention_days=14,
        filters={"status": ["saved"], "tags": ["research"]},
    )
    listed = service.list_reading_digest_schedules(limit=10, offset=0)
    detail = service.get_reading_digest_schedule(created["id"])
    updated = service.update_reading_digest_schedule(
        created["id"],
        name="Evening",
        cron="0 18 * * *",
        enabled=False,
        filters={"status": ["archived"]},
    )
    outputs = service.list_reading_digest_outputs(schedule_id=created["id"], limit=5, offset=0)
    deleted = service.delete_reading_digest_schedule(created["id"])
    after_delete = service.list_reading_digest_schedules()

    assert created["id"].startswith("local-digest-")
    assert created["name"] == "Morning"
    assert created["cron"] == "0 8 * * *"
    assert created["timezone"] == "America/Los_Angeles"
    assert created["enabled"] is True
    assert created["require_online"] is False
    assert created["format"] == "md"
    assert created["retention_days"] == 14
    assert created["filters"] == {"status": ["saved"], "tags": ["research"]}
    assert listed["items"] == [created]
    assert listed["total"] == 1
    assert detail == created
    assert updated["name"] == "Evening"
    assert updated["cron"] == "0 18 * * *"
    assert updated["enabled"] is False
    assert updated["filters"] == {"status": ["archived"]}
    assert outputs == {"items": [], "total": 0, "limit": 5, "offset": 0}
    assert deleted == {"ok": True, "id": created["id"]}
    assert after_delete["items"] == []


def test_local_service_runs_due_reading_digest_schedules(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)
    media_id, _, _ = db.add_media_with_keywords(
        url="https://example.com/read",
        title="Saved Article",
        media_type="article",
        content="Saved article body. Second sentence.",
        keywords=["research"],
    )
    service.save_to_read_it_later(media_id)
    schedule = service.create_reading_digest_schedule(
        name="Morning",
        cron="0 8 * * *",
        timezone="UTC",
        filters={"status": ["saved"], "tags": ["research"]},
    )

    result = service.run_due_reading_digest_schedules(now="2026-04-25T08:00:00+00:00")
    duplicate = service.run_due_reading_digest_schedules(now="2026-04-25T08:00:30+00:00")
    outputs = service.list_reading_digest_outputs(schedule_id=schedule["id"])

    assert result["executed_count"] == 1
    assert result["skipped_count"] == 0
    assert result["results"][0]["output"]["schedule_id"] == schedule["id"]
    assert "Saved Article" in result["results"][0]["output"]["content"]
    assert "https://example.com/read" in result["results"][0]["output"]["content"]
    assert result["results"][0]["output"]["metadata"]["item_count"] == 1
    assert duplicate["executed_count"] == 0
    assert duplicate["skipped_count"] == 1
    assert duplicate["results"][0]["reason"] == "already_executed_for_current_minute"
    assert outputs["total"] == 1


def test_local_service_persists_file_artifacts_and_reference_images(memory_db_factory):
    db = memory_db_factory()
    service = LocalMediaReadingService(db)

    created = service.create_file_artifact(
        file_type="reference_image",
        payload={"mime_type": "image/png", "width": 640, "height": 480, "alt": "Figure"},
        title="Figure 1",
        export={"format": "md", "content": "![Figure 1](local://figure-1.png)", "filename": "figure.md"},
        options={"persist": True},
    )
    detail = service.get_file_artifact(created["artifact"]["file_id"])
    reference_images = service.list_reference_images()
    exported = service.export_file_artifact(created["artifact"]["file_id"], format="md")
    deleted = service.delete_file_artifact(created["artifact"]["file_id"], hard=False, delete_file=False)
    after_delete = service.list_reference_images()
    purged = service.purge_file_artifacts()

    assert created["artifact"]["file_type"] == "reference_image"
    assert created["artifact"]["title"] == "Figure 1"
    assert detail["artifact"]["structured"]["alt"] == "Figure"
    assert reference_images["total"] == 1
    assert reference_images["items"][0]["file_id"] == created["artifact"]["file_id"]
    assert reference_images["items"][0]["mime_type"] == "image/png"
    assert exported["content"] == b"![Figure 1](local://figure-1.png)"
    assert exported["filename"] == "figure.md"
    assert deleted == {"success": True, "file_deleted": False}
    assert after_delete == {"items": [], "total": 0}
    assert purged == {"removed": 1, "files_deleted": 0}


def test_local_service_processes_text_like_files_without_persisting(memory_db_factory, tmp_path):
    db = memory_db_factory()
    notes = tmp_path / "notes.txt"
    doc = tmp_path / "doc.md"
    code = tmp_path / "main.py"
    notes.write_text("Plain text body", encoding="utf-8")
    doc.write_text("# Heading\n\nDocument body", encoding="utf-8")
    code.write_text("print('hello')\n", encoding="utf-8")
    service = LocalMediaReadingService(db)

    plaintext = service.process_plaintext(file_paths=[str(notes)], perform_chunking=True, chunk_size=6, chunk_overlap=0)
    document = service.process_document(file_paths=[str(doc)])
    code_result = service.process_code(file_paths=[str(code)], chunk_method="lines")

    assert plaintext["processed_count"] == 1
    assert plaintext["results"][0]["media_type"] == "plaintext"
    assert plaintext["results"][0]["content"] == "Plain text body"
    assert [chunk["text"] for chunk in plaintext["results"][0]["chunks"]] == ["Plain ", "text b", "ody"]
    assert document["results"][0]["title"] == "doc.md"
    assert document["results"][0]["media_type"] == "document"
    assert code_result["results"][0]["media_type"] == "code"
    assert code_result["results"][0]["content"] == "print('hello')\n"
    assert service.list_media_items()["pagination"]["total_items"] == 0


def test_local_service_processes_pdf_and_ebook_files_without_persisting(memory_db_factory, tmp_path):
    db = memory_db_factory()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nPDF body text\n%%EOF")
    epub_path = tmp_path / "book.epub"
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("OPS/chapter.xhtml", "<html><body><h1>Chapter</h1><p>EPUB body text</p></body></html>")
    service = LocalMediaReadingService(db)

    pdf = service.process_pdf(file_paths=[str(pdf_path)], perform_chunking=True, chunk_size=8, chunk_overlap=0)
    ebook = service.process_ebook(file_paths=[str(epub_path)], perform_chunking=False)

    assert pdf["processed_count"] == 1
    assert pdf["results"][0]["media_type"] == "pdf"
    assert "PDF body text" in pdf["results"][0]["content"]
    assert [chunk["text"] for chunk in pdf["results"][0]["chunks"][:2]] == ["PDF body", " text\n"]
    assert ebook["processed_count"] == 1
    assert ebook["results"][0]["media_type"] == "ebook"
    assert "Chapter" in ebook["results"][0]["content"]
    assert "EPUB body text" in ebook["results"][0]["content"]
    assert service.list_media_items()["pagination"]["total_items"] == 0


def test_local_service_processes_email_files_without_persisting(memory_db_factory, tmp_path):
    db = memory_db_factory()
    email_path = tmp_path / "message.eml"
    email_path.write_text(
        "From: sender@example.com\n"
        "To: reader@example.com\n"
        "Subject: Local Mail\n"
        "Date: Sat, 25 Apr 2026 08:00:00 +0000\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "Email body text",
        encoding="utf-8",
    )
    service = LocalMediaReadingService(db)

    email = service.process_emails(file_paths=[str(email_path)], title="Inbox")

    assert email["processed_count"] == 1
    assert email["results"][0]["media_type"] == "email"
    assert email["results"][0]["title"] == "Local Mail"
    assert email["results"][0]["subject"] == "Local Mail"
    assert email["results"][0]["from"] == "sender@example.com"
    assert email["results"][0]["to"] == "reader@example.com"
    assert email["results"][0]["content"] == "Email body text"
    assert service.list_media_items()["pagination"]["total_items"] == 0


def test_local_service_processes_web_scraping_urls_without_persisting(memory_db_factory):
    db = memory_db_factory()
    calls = []

    def fake_scraper(url, *, custom_cookies=None):
        calls.append((url, custom_cookies))
        return {
            "url": url,
            "title": "Scraped Post",
            "content": "Scraped body",
            "author": "Web Author",
            "keywords": ["scraped"],
        }

    service = LocalMediaReadingService(db, url_article_scraper=fake_scraper)

    web = service.process_web_scraping(
        scrape_method="individual",
        url_input="https://example.com/post",
        mode="ephemeral",
        keywords="ai,reading",
    )

    assert calls == [("https://example.com/post", None)]
    assert web["status"] == "success"
    assert web["backend"] == "local"
    assert web["persisted"] is False
    assert web["count"] == 1
    assert web["results"][0]["url"] == "https://example.com/post"
    assert web["results"][0]["title"] == "Scraped Post"
    assert web["results"][0]["content"] == "Scraped body"
    assert service.list_media_items()["pagination"]["total_items"] == 0


def test_local_service_processes_audio_and_video_without_persisting(memory_db_factory, tmp_path):
    db = memory_db_factory()
    audio_path = tmp_path / "clip.mp3"
    video_path = tmp_path / "clip.mp4"
    audio_path.write_bytes(b"audio")
    video_path.write_bytes(b"video")
    calls = []

    class FakeAudioProcessor:
        def process_audio_files(self, **kwargs):
            calls.append(("audio", kwargs))
            return {
                "processed_count": 1,
                "errors_count": 0,
                "errors": [],
                "results": [
                    {
                        "status": "Success",
                        "input_ref": str(audio_path),
                        "media_type": "audio",
                        "content": "audio transcript",
                    }
                ],
            }

    class FakeVideoProcessor:
        def process_videos(self, **kwargs):
            calls.append(("video", kwargs))
            return {
                "processed_count": 1,
                "errors_count": 0,
                "errors": [],
                "results": [
                    {
                        "status": "Success",
                        "input_ref": str(video_path),
                        "media_type": "video",
                        "content": "video transcript",
                    }
                ],
            }

    service = LocalMediaReadingService(
        db,
        audio_processor_factory=FakeAudioProcessor,
        video_processor_factory=FakeVideoProcessor,
    )

    audio = service.process_audio(
        file_paths=[str(audio_path)],
        transcription_model="tiny",
        perform_analysis=False,
    )
    video = service.process_video(
        file_paths=[str(video_path)],
        transcription_model="tiny",
        perform_analysis=False,
    )

    assert audio["backend"] == "local"
    assert audio["persisted"] is False
    assert audio["results"][0]["backend"] == "local"
    assert audio["results"][0]["persisted"] is False
    assert video["backend"] == "local"
    assert video["persisted"] is False
    assert calls[0] == (
        "audio",
        {
            "inputs": [str(audio_path)],
            "transcription_model": "tiny",
            "perform_analysis": False,
        },
    )
    assert calls[1] == (
        "video",
        {
            "inputs": [str(video_path)],
            "download_video_flag": False,
            "transcription_model": "tiny",
            "perform_analysis": False,
        },
    )
    assert service.list_media_items()["pagination"]["total_items"] == 0


@pytest.mark.asyncio
async def test_local_service_processes_mediawiki_dump_without_persisting(memory_db_factory, tmp_path):
    db = memory_db_factory()
    dump_path = tmp_path / "wiki.xml"
    dump_path.write_text(
        """
        <mediawiki>
          <page>
            <title>Main Page</title>
            <ns>0</ns>
            <revision><text>Main body</text></revision>
          </page>
          <page>
            <title>Talk Page</title>
            <ns>1</ns>
            <revision><text>Talk body</text></revision>
          </page>
        </mediawiki>
        """,
        encoding="utf-8",
    )
    service = LocalMediaReadingService(db)

    pages = [
        page
        async for page in service.process_mediawiki_dump(
            dump_file_path=str(dump_path),
            wiki_name="Demo",
            namespaces_str="0",
        )
    ]

    assert pages == [
        {
            "status": "Success",
            "backend": "local",
            "persisted": False,
            "wiki_name": "Demo",
            "title": "Main Page",
            "namespace": "0",
            "content": "Main body",
            "media_type": "mediawiki_dump",
            "input_ref": str(dump_path),
        }
    ]
    assert service.list_media_items()["pagination"]["total_items"] == 0


def test_local_service_extracts_document_intelligence_from_local_content(memory_db_factory):
    db = memory_db_factory()
    media_id, _, _ = db.add_media_with_keywords(
        title="Intelligence",
        content=(
            "# Introduction\n"
            "This local paper describes a useful offline result.\n\n"
            "![Architecture Diagram](https://example.com/figure.png)\n\n"
            "## Methods\n"
            "The method is deterministic and local.\n\n"
            "References\n"
            "Smith J. 2024. Local Paper. https://doi.org/10.1234/local. https://example.com/paper\n"
        ),
        media_type="document",
        keywords=[],
    )
    service = LocalMediaReadingService(db)

    outline = service.get_document_outline(media_id)
    figures = service.get_document_figures(media_id, min_size=80)
    references = service.get_document_references(media_id, search="Smith", limit=10)
    insights = service.generate_document_insights(media_id, categories=["summary"], max_content_length=5000)

    assert outline["media_id"] == media_id
    assert outline["has_outline"] is True
    assert outline["entries"] == [
        {"level": 1, "title": "Introduction", "page": 1},
        {"level": 2, "title": "Methods", "page": 1},
    ]
    assert figures["has_figures"] is True
    assert figures["total_count"] == 1
    assert figures["figures"][0]["id"] == "local-fig-1"
    assert figures["figures"][0]["width"] == 80
    assert figures["figures"][0]["height"] == 80
    assert figures["figures"][0]["caption"] == "Architecture Diagram"
    assert references["has_references"] is True
    assert references["returned_count"] == 1
    assert references["references"][0]["doi"] == "10.1234/local"
    assert references["references"][0]["url"] == "https://example.com/paper"
    assert insights["model_used"] == "local-extractive"
    assert insights["cached"] is False
    assert insights["insights"][0]["category"] == "summary"
    assert "offline result" in insights["insights"][0]["content"]


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
    highlight = service.create_highlight(saved_id, quote="Saved", color="yellow")

    exported = service.export_reading_items(
        format="jsonl",
        include_metadata=True,
        include_text=True,
        include_highlights=True,
    )
    zip_export = service.export_reading_items(format="zip", include_metadata=False)

    rows = [json.loads(line) for line in exported["content"].decode("utf-8").splitlines()]
    assert exported["content_type"] == "application/x-ndjson"
    assert exported["filename"].endswith(".jsonl")
    assert rows[0]["id"] == saved_id
    assert rows[0]["title"] == "Saved"
    assert rows[0]["status"] == "saved"
    assert rows[0]["text"] == "Saved text"
    assert rows[0]["highlights"] == [highlight]
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
