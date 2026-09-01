"""Coherent Local persistence for authority-scoped Collections captures."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.collections_capture_models import (
    CAPTURE_SORTS,
    CaptureHighlightDraft,
    CaptureIdentity,
    CapturePageRequest,
    CaptureSaveRequest,
    CollectionsCaptureError,
    ExternalNoteReference,
)
from tldw_chatbook.Library.collections_capture_repository import (
    CollectionsCaptureRepository,
)


@pytest.fixture
def repository(tmp_path: Path) -> CollectionsCaptureRepository:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    repo = CollectionsCaptureRepository(db, authority_key="local:profile-a")
    yield repo
    db.close()


def _save(
    repository: CollectionsCaptureRepository,
    suffix: str,
    **changes: object,
):
    values: dict[str, object] = {
        "authority_key": repository.authority_key,
        "submitted_url": f"https://example.org/{suffix}",
        "title": f"Capture {suffix}",
    }
    values.update(changes)
    return repository.save_capture(CaptureSaveRequest(**values))


def test_save_is_authority_scoped_and_canonical_upsert_preserves_omissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.db"
    db_a = LibraryCollectionsDB(path)
    db_b = LibraryCollectionsDB(path)
    repo_a = CollectionsCaptureRepository(db_a, authority_key="local:profile-a")
    repo_b = CollectionsCaptureRepository(db_b, authority_key="local:profile-b")

    first = repo_a.save_capture(
        CaptureSaveRequest(
            authority_key="local:profile-a",
            submitted_url="https://EXAMPLE.org:443/article#section",
            title="Original title",
            tags=("Zeta", "alpha"),
            favorite=True,
            text_content="Original text",
            clean_html="<p>Original text</p>",
        )
    )
    second = repo_a.save_capture(
        CaptureSaveRequest(
            authority_key="local:profile-a",
            submitted_url="https://example.org/article",
            tags=("Beta", "ALPHA"),
            status="archived",
        )
    )
    other = repo_b.save_capture(
        CaptureSaveRequest(
            authority_key="local:profile-b",
            submitted_url="https://example.org/article",
            title="Other profile",
        )
    )

    assert first.created is True
    assert second.created is False
    assert second.capture is not None
    assert first.capture is not None
    assert second.capture.identity == first.capture.identity
    assert second.capture.revision == first.capture.revision + 1
    assert second.capture.canonical_url == "https://example.org/article"
    assert second.capture.title == "Original title"
    assert second.capture.favorite is True
    assert second.capture.status == "archived"
    assert second.capture.tags == ("alpha", "Beta", "Zeta")
    assert second.capture.text_content == "Original text"
    assert second.capture.clean_html == "<p>Original text</p>"
    assert second.capture.content_hash == first.capture.content_hash
    assert other.capture is not None
    assert other.capture.identity != first.capture.identity
    assert repo_a.list_page(CapturePageRequest("local:profile-a")).total == 1
    assert repo_b.list_page(CapturePageRequest("local:profile-b")).total == 1
    db_a.close()
    db_b.close()


def test_content_fields_replace_independently_and_hash_is_deterministic(
    repository: CollectionsCaptureRepository,
) -> None:
    saved = _save(
        repository,
        "content",
        text_content="Text one",
        clean_html="<p>HTML one</p>",
    ).capture
    assert saved is not None

    text_changed = repository.update_capture(
        saved.identity,
        expected_revision=saved.revision,
        changes={"text_content": "Text two"},
    )
    assert text_changed.clean_html == "<p>HTML one</p>"
    assert text_changed.content_hash != saved.content_hash

    html_cleared = repository.update_capture(
        saved.identity,
        expected_revision=text_changed.revision,
        changes={"clean_html": None},
    )
    assert html_cleared.text_content == "Text two"
    assert html_cleared.clean_html is None
    assert html_cleared.content_hash != text_changed.content_hash

    restored = repository.update_capture(
        saved.identity,
        expected_revision=html_cleared.revision,
        changes={"clean_html": "<p>HTML one</p>"},
    )
    repeated = repository.update_capture(
        saved.identity,
        expected_revision=restored.revision,
        changes={"text_content": "Text two"},
    )
    assert repeated.content_hash == restored.content_hash


def test_update_uses_revision_cas_and_replaces_mutable_reading_state(
    repository: CollectionsCaptureRepository,
) -> None:
    saved = _save(
        repository,
        "state",
        tags=("Old", "Keep"),
        favorite=True,
        freeform_note="Old note",
    ).capture
    assert saved is not None

    with pytest.raises(CollectionsCaptureError) as caught:
        repository.update_capture(
            saved.identity,
            expected_revision=saved.revision + 1,
            changes={"status": "read"},
        )
    assert caught.value.reason == "revision_conflict"

    updated = repository.update_capture(
        saved.identity,
        expected_revision=saved.revision,
        changes={
            "status": "read",
            "favorite": False,
            "tags": ("Replacement",),
            "freeform_note": None,
        },
    )
    assert updated.status == "read"
    assert updated.read_at is not None
    assert updated.favorite is False
    assert updated.tags == ("Replacement",)
    assert updated.freeform_note is None

    archived = repository.update_capture(
        saved.identity,
        expected_revision=updated.revision,
        changes={"status": "archived"},
    )
    restored = repository.update_capture(
        saved.identity,
        expected_revision=archived.revision,
        changes={"status": "saved"},
    )
    assert archived.status == "archived"
    assert restored.status == "saved"


def test_fts_and_exact_filters_are_authority_bounded(
    repository: CollectionsCaptureRepository,
) -> None:
    _save(
        repository,
        "alpha",
        title="Rust field guide",
        summary="Ownership and borrowing",
        tags=("Systems", "Reference"),
        status="reading",
        favorite=True,
        published_at="2026-02-10",
    )
    _save(
        repository,
        "beta",
        title="Garden notes",
        summary="Spring planting",
        tags=("Home",),
        status="saved",
        favorite=False,
        published_at="2025-06-01",
    )
    with repository.db.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_items SET created_at = published_at "
            "WHERE authority_key = ?",
            (repository.authority_key,),
        )

    assert repository.list_page(
        CapturePageRequest(repository.authority_key, search="borrowing")
    ).total == 1
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, tags=("systems", "reference"))
    ).total == 1
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, statuses=("reading",))
    ).total == 1
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, favorite=True)
    ).total == 1
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, domain="example.org")
    ).total == 2
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, date_from="2026-01-01")
    ).total == 1
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, date_to="2025-12-31")
    ).total == 1
    assert repository.list_page(
        CapturePageRequest(repository.authority_key, search='Rust: "guide"')
    ).total == 1


@pytest.mark.parametrize("sort", CAPTURE_SORTS)
def test_every_sort_is_stable(
    repository: CollectionsCaptureRepository,
    sort: str,
) -> None:
    first = _save(repository, "sort-a", title="Same", summary="shared token").capture
    second = _save(repository, "sort-b", title="Same", summary="shared token").capture
    assert first is not None and second is not None
    with repository.db.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_items "
            "SET created_at = ?, updated_at = ?, published_at = ? "
            "WHERE authority_key = ?",
            ("2026-01-01", "2026-01-01", "2026-01-01", repository.authority_key),
        )

    request = CapturePageRequest(
        repository.authority_key,
        search="shared" if sort == "relevance" else "",
        sort=sort,
    )
    page = repository.list_page(request)
    ids = [item.identity.capture_id for item in page.items]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    descending = sort in {"saved_desc", "updated_desc", "title_desc"}
    assert ids == sorted(ids, reverse=descending)
    assert ids == [
        item.identity.capture_id
        for item in repository.list_page(request).items
    ]


def test_pagination_returns_exact_totals_and_no_overlap(
    repository: CollectionsCaptureRepository,
) -> None:
    for index in range(45):
        _save(repository, f"page-{index:02d}")
    with repository.db.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_items SET created_at = ? WHERE authority_key = ?",
            ("2026-01-01", repository.authority_key),
        )

    pages = [
        repository.list_page(CapturePageRequest(repository.authority_key, page=page))
        for page in (1, 2, 3)
    ]
    assert [page.total for page in pages] == [45, 45, 45]
    assert [len(page.items) for page in pages] == [20, 20, 5]
    ids = [item.identity.capture_id for page in pages for item in page.items]
    assert len(ids) == len(set(ids)) == 45
    assert ids == sorted(ids, reverse=True)


def test_count_and_rows_share_one_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.db"
    reader_db = LibraryCollectionsDB(path)
    writer_db = LibraryCollectionsDB(path)
    reader = CollectionsCaptureRepository(reader_db, authority_key="local:profile-a")
    writer = CollectionsCaptureRepository(writer_db, authority_key="local:profile-a")
    for index in range(21):
        _save(reader, f"before-{index:02d}")

    count_observed = threading.Event()
    writer_done = threading.Event()
    writer_failures: list[BaseException] = []

    def after_count() -> None:
        count_observed.set()
        assert writer_done.wait(5)

    def write_after_count() -> None:
        try:
            assert count_observed.wait(5)
            _save(writer, "after")
        except BaseException as exc:  # noqa: BLE001 - relayed to test thread
            writer_failures.append(exc)
        finally:
            writer_db.close()
            writer_done.set()

    reader._after_page_count = after_count
    thread = threading.Thread(target=write_after_count)
    thread.start()
    page = reader.list_page(CapturePageRequest(reader.authority_key))
    thread.join(5)

    assert not thread.is_alive()
    assert not writer_failures
    assert page.total == 21
    assert all(item.canonical_url != "https://example.org/after" for item in page.items)
    assert reader.list_page(CapturePageRequest(reader.authority_key)).total == 22
    reader_db.close()
    writer_db.close()


def test_saved_search_crud_is_scoped_and_revision_guarded(
    repository: CollectionsCaptureRepository,
) -> None:
    request = CapturePageRequest(
        repository.authority_key,
        statuses=("reading",),
        tags=("reference",),
    )
    saved = repository.create_saved_search("Reading", request)
    page = repository.list_saved_searches(page=1)
    assert page.total == 1
    assert page.items == (saved,)

    with pytest.raises(CollectionsCaptureError) as caught:
        repository.update_saved_search(
            saved.search_id,
            name="Changed",
            request=request,
            expected_revision=99,
        )
    assert caught.value.reason == "revision_conflict"

    updated = repository.update_saved_search(
        saved.search_id,
        name="Changed",
        request=CapturePageRequest(repository.authority_key, favorite=True),
        expected_revision=saved.revision,
    )
    deleted = repository.delete_saved_search(
        saved.search_id,
        expected_revision=updated.revision,
    )
    assert updated.name == "Changed"
    assert updated.created_at == saved.created_at
    assert deleted.success is True
    assert repository.list_saved_searches(page=1).total == 0


def test_highlights_note_links_and_hard_delete_tombstone(
    repository: CollectionsCaptureRepository,
) -> None:
    capture = _save(repository, "relations").capture
    assert capture is not None
    highlight = repository.save_highlight(
        capture.identity,
        CaptureHighlightDraft("Quoted text", note="My note", anchor_json='{"p":1}'),
    )
    assert repository.list_highlights(capture.identity) == (highlight,)

    with pytest.raises(CollectionsCaptureError) as caught:
        repository.delete_highlight(
            capture.identity,
            highlight.highlight_id,
            expected_revision=99,
        )
    assert caught.value.reason == "revision_conflict"
    assert repository.delete_highlight(
        capture.identity,
        highlight.highlight_id,
        expected_revision=highlight.revision,
    ).success

    note = ExternalNoteReference("notes:profile-a", "note-1")
    link = repository.link_note(capture.identity, note)
    assert repository.link_note(capture.identity, note) == link
    assert repository.list_note_links(capture.identity) == (link,)
    assert repository.unlink_note(capture.identity, link.link_id).success

    result = repository.hard_delete(
        capture.identity,
        expected_revision=capture.revision,
    )
    assert result.success is True
    assert repository.get_detail(capture.identity) is None
    with repository.db.connection() as connection:
        row = connection.execute(
            "SELECT purge_state FROM collection_capture_items "
            "WHERE authority_key = ? AND capture_id = ?",
            (capture.identity.authority_key, capture.identity.capture_id),
        ).fetchone()
    assert row is not None and row[0] == "pending"


def test_repository_refuses_cross_authority_inputs(
    repository: CollectionsCaptureRepository,
) -> None:
    with pytest.raises(CollectionsCaptureError) as caught:
        repository.list_page(CapturePageRequest("local:profile-b"))
    assert caught.value.reason == "authority_mismatch"

    with pytest.raises(CollectionsCaptureError) as caught:
        repository.get_detail(CaptureIdentity("local:profile-b", "capture-1"))
    assert caught.value.reason == "authority_mismatch"
