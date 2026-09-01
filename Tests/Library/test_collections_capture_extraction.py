"""Commit-first extraction lifecycle for Local Collections captures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.collections_capture_models import (
    CaptureSaveRequest,
    CollectionsCaptureError,
)
from tldw_chatbook.Library.collections_capture_repository import (
    CollectionsCaptureRepository,
)


@pytest.fixture
def repository(tmp_path: Path) -> CollectionsCaptureRepository:
    database = LibraryCollectionsDB(tmp_path / "collections.db")
    repo = CollectionsCaptureRepository(database, authority_key="local:profile-a")
    yield repo
    database.close()


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
    outcome = repository.save_capture(CaptureSaveRequest(**values))
    assert outcome.capture is not None
    return outcome


def test_save_commits_queued_capture_before_extraction_claim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collections.db"
    database = LibraryCollectionsDB(path)
    repository = CollectionsCaptureRepository(
        database,
        authority_key="local:profile-a",
    )

    outcome = _save(repository, "commit-first")

    assert outcome.extraction_pending is True
    assert outcome.capture.processing_state == "queued"
    observer_database = LibraryCollectionsDB(path)
    observer = CollectionsCaptureRepository(
        observer_database,
        authority_key="local:profile-a",
    )
    assert observer.get_detail(outcome.capture.identity) == outcome.capture
    observer_database.close()
    database.close()


def test_claim_and_complete_store_only_inert_reader_text(
    repository: CollectionsCaptureRepository,
) -> None:
    capture = _save(repository, "hostile").capture
    claimed = repository.claim_extraction(
        capture.identity,
        expected_revision=capture.revision,
    )
    assert claimed.processing_state == "processing"
    assert claimed.revision == capture.revision + 1

    completed = repository.complete_extraction(
        capture.identity,
        expected_revision=claimed.revision,
        result={
            "content": (
                "<article><h1>Reader title</h1><p>Useful body</p>"
                "<script>script_secret()</script><style>body{display:none}</style>"
                '<img src="https://tracker.invalid/pixel" '
                'onerror="handler_secret()" alt="diagram">'
                '<a href="javascript:link_target()" '
                'onclick="handler_secret()">link</a>'
                "</article>\x1b]8;;https://phish.invalid\x07"
            ),
            "clean_html": '<iframe src="https://active.invalid"></iframe>',
            "title": "<b>Extracted title</b>\x1b[31m",
            "author": "<span onmouseover='steal()'>Ada</span>",
        },
    )

    assert completed.processing_state == "ready"
    assert completed.last_fetch_error is None
    assert completed.title.startswith("Extracted title")
    assert "\x1b" not in completed.title
    assert completed.byline == "Ada"
    assert completed.clean_html is None
    assert completed.text_content is not None
    assert "Useful body" in completed.text_content
    assert "[image: diagram]" in completed.text_content
    assert "javascript:link_target()" in completed.text_content
    assert "<" not in completed.text_content
    assert "script_secret" not in completed.text_content
    assert "handler_secret" not in completed.text_content
    assert "tracker.invalid" not in completed.text_content
    assert "\x1b" not in completed.text_content
    assert completed.word_count == len(completed.text_content.split())
    assert completed.content_hash is not None


def test_failed_extraction_is_bounded_and_retry_preserves_reading_state(
    repository: CollectionsCaptureRepository,
) -> None:
    capture = _save(
        repository,
        "retry",
        status="reading",
        favorite=True,
    ).capture
    claimed = repository.claim_extraction(
        capture.identity,
        expected_revision=capture.revision,
    )
    failed = repository.fail_extraction(
        capture.identity,
        expected_revision=claimed.revision,
        reason="fetch_failed",
    )

    assert failed.processing_state == "failed"
    assert failed.last_fetch_error == "fetch_failed"
    assert failed.status == "reading"
    assert failed.favorite is True

    retried = repository.retry_extraction(
        capture.identity,
        expected_revision=failed.revision,
    )
    assert retried.processing_state == "queued"
    assert retried.last_fetch_error is None
    assert retried.status == "reading"
    assert retried.favorite is True

    with pytest.raises(CollectionsCaptureError) as caught:
        repository.fail_extraction(
            capture.identity,
            expected_revision=retried.revision,
            reason="https://private.invalid/?token=secret",
        )
    assert caught.value.reason == "invalid_extraction_failure_reason"


def test_startup_interrupts_only_this_authoritys_processing_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collections.db"
    database_a = LibraryCollectionsDB(path)
    database_b = LibraryCollectionsDB(path)
    repo_a = CollectionsCaptureRepository(database_a, authority_key="local:profile-a")
    repo_b = CollectionsCaptureRepository(database_b, authority_key="local:profile-b")
    queued_a = _save(repo_a, "a").capture
    queued_b = _save(repo_b, "b").capture
    processing_a = repo_a.claim_extraction(
        queued_a.identity,
        expected_revision=queued_a.revision,
    )
    processing_b = repo_b.claim_extraction(
        queued_b.identity,
        expected_revision=queued_b.revision,
    )

    interrupted = repo_a.interrupt_stale_extractions()

    assert interrupted == 1
    detail_a = repo_a.get_detail(processing_a.identity)
    detail_b = repo_b.get_detail(processing_b.identity)
    assert detail_a is not None
    assert detail_a.processing_state == "interrupted"
    assert detail_a.last_fetch_error == "interrupted"
    assert detail_a.revision == processing_a.revision + 1
    assert detail_b is not None
    assert detail_b.processing_state == "processing"
    database_a.close()
    database_b.close()


def test_extraction_transitions_require_current_revision_and_valid_state(
    repository: CollectionsCaptureRepository,
) -> None:
    capture = _save(repository, "guards").capture
    with pytest.raises(CollectionsCaptureError) as caught:
        repository.complete_extraction(
            capture.identity,
            expected_revision=capture.revision,
            result={"content": "Body"},
        )
    assert caught.value.reason == "invalid_extraction_state"

    claimed = repository.claim_extraction(
        capture.identity,
        expected_revision=capture.revision,
    )
    with pytest.raises(CollectionsCaptureError) as caught:
        repository.complete_extraction(
            capture.identity,
            expected_revision=capture.revision,
            result={"content": "Body"},
        )
    assert caught.value.reason == "revision_conflict"
    assert caught.value.conflict.current == claimed

    with pytest.raises(CollectionsCaptureError) as caught:
        repository.complete_extraction(
            capture.identity,
            expected_revision=claimed.revision,
            result={"content": "<script>only active content</script>"},
        )
    assert caught.value.reason == "empty_extraction_content"
    assert repository.get_detail(capture.identity) == claimed
