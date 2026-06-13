from tldw_chatbook.Study_Interop.study_normalizers import (
    merge_review_outcome_record,
    normalize_study_deck_record,
    normalize_study_flashcard_record,
    normalize_study_review_candidate,
)


def test_normalize_local_study_deck_record_sets_local_defaults():
    record = normalize_study_deck_record(
        "local",
        {
            "id": "deck-local-1",
            "name": "Biology",
            "description": "Cell review",
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:01:00Z",
            "version": 2,
            "metadata": None,
        },
    )

    assert record["record_id"] == "local:study_deck:deck-local-1"
    assert record["workspace_id"] is None
    assert record["scheduler_type"] is None
    assert record["metadata"] == {}


def test_normalize_server_study_flashcard_record_preserves_due_and_scheduler_detail():
    record = normalize_study_flashcard_record(
        "server",
        {
            "uuid": "card-1",
            "deck_id": 7,
            "front": "Question",
            "back": "Answer",
            "notes": "Notes",
            "extra": "Extra",
            "tags": ["science"],
            "model_type": "basic_reverse",
            "due_at": "2026-04-22T00:00:00Z",
            "last_reviewed_at": "2026-04-20T00:00:00Z",
            "interval_days": 4,
            "repetitions": 2,
            "ef": 2.7,
            "queue_state": "review",
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:01:00Z",
            "deleted": False,
            "version": 3,
            "client_id": "server-client",
            "next_intervals": {"again": "10m", "good": "4d"},
        },
    )

    assert record["record_id"] == "server:study_flashcard:card-1"
    assert record["deck_record_id"] == "server:study_deck:7"
    assert record["card_kind"] == "basic_reverse"
    assert record["review_detail_available"] is True
    assert record["metadata"]["next_intervals"]["good"] == "4d"


def test_normalize_local_study_flashcard_record_splits_tag_strings_and_coarsens_queue_state():
    record = normalize_study_flashcard_record(
        "local",
        {
            "id": "card-local-1",
            "deck_id": "deck-local-1",
            "front": "Question",
            "back": "Answer",
            "tags": "science biology",
            "type": "basic",
            "next_review": None,
            "last_review": None,
            "interval": 0,
            "repetitions": 0,
            "ease_factor": 2.5,
            "is_suspended": 0,
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:01:00Z",
            "version": 1,
            "metadata": None,
        },
    )

    assert record["record_id"] == "local:study_flashcard:card-local-1"
    assert record["tags"] == ["science", "biology"]
    assert record["queue_state"] == "new"
    assert record["review_detail_available"] is False


def test_normalize_review_candidate_wraps_card_and_selection_reason():
    card = normalize_study_flashcard_record(
        "server",
        {
            "uuid": "card-1",
            "deck_id": 7,
            "front": "Question",
            "back": "Answer",
            "tags": [],
            "is_cloze": False,
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "queue_state": "new",
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:01:00Z",
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
            "model_type": "basic",
            "reverse": False,
            "scheduler_type": "fsrs",
            "next_intervals": {"again": "10m"},
        },
    )

    candidate = normalize_study_review_candidate(
        "server",
        card=card,
        selection_reason="new",
        review_session={"review_session_id": 41},
    )

    assert candidate["card"]["record_id"] == "server:study_flashcard:card-1"
    assert candidate["selection_reason"] == "new"
    assert candidate["detail_available"] is True
    assert candidate["review_session"]["review_session_id"] == 41


def test_merge_review_outcome_record_merges_server_review_result_into_current_card():
    current_card = normalize_study_flashcard_record(
        "server",
        {
            "uuid": "card-1",
            "deck_id": 7,
            "front": "Question",
            "back": "Answer",
            "tags": ["science"],
            "is_cloze": False,
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "queue_state": "new",
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:01:00Z",
            "deleted": False,
            "client_id": "server-client",
            "version": 1,
            "model_type": "basic",
            "reverse": False,
            "scheduler_type": "fsrs",
        },
    )

    outcome = merge_review_outcome_record(
        "server",
        current_card=current_card,
        review_response={
            "uuid": "card-1",
            "ef": 2.7,
            "interval_days": 5,
            "repetitions": 2,
            "lapses": 0,
            "due_at": "2026-04-25T00:00:00Z",
            "last_reviewed_at": "2026-04-20T00:05:00Z",
            "last_modified": "2026-04-20T00:05:00Z",
            "version": 2,
            "scheduler_type": "fsrs",
            "queue_state": "review",
            "next_intervals": {"again": "10m", "good": "5d"},
            "review_session_id": 41,
        },
        rating=4,
    )

    assert outcome["card"]["front"] == "Question"
    assert outcome["card"]["interval_days"] == 5
    assert outcome["rating"] == 4
    assert outcome["review_session"]["review_session_id"] == 41
    assert outcome["next_intervals"]["good"] == "5d"

