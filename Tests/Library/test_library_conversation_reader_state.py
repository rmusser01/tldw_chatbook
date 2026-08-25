"""Pure state transitions for the bounded Conversations reader."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

import tldw_chatbook.Library.library_conversation_reader_state as reader_state
from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationFindMatch,
    ConversationMessageView,
    ConversationReaderRequest,
    ConversationReaderState,
    mark_conversation_deleted,
    project_conversation_multiselect,
    retry_conversation,
    select_conversation,
    set_conversation_find_query,
    set_conversation_reader_mode,
    settle_conversation_continuation,
    settle_conversation_error,
    settle_conversation_page,
    settle_conversation_unavailable,
)


def test_reader_state_is_a_frozen_empty_read_model() -> None:
    state = ConversationReaderState()

    assert state.selected_id is None
    assert state.loaded_id is None
    assert state.messages == ()
    assert state.mode == "read"
    with pytest.raises(FrozenInstanceError):
        state.mode = "info"  # type: ignore[misc]


def _loaded_state(*, mode: str = "read") -> ConversationReaderState:
    return ConversationReaderState(
        selected_id="conversation-a",
        selected_version=4,
        loaded_id="conversation-a",
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        mode=mode,  # type: ignore[arg-type]
        messages=(
            ConversationMessageView(
                message_id="message-a",
                sender="user",
                timestamp="2026-08-24T10:00:00Z",
                revision="revision-a",
                total_chars=5,
                text="hello",
            ),
        ),
        message_total=1,
        complete=True,
    )


def test_selecting_new_item_requests_next_generation_and_keeps_loaded_preview() -> None:
    loaded = _loaded_state(mode="info")

    selected, request = select_conversation(loaded, "conversation-b", version=7)

    assert request == ConversationReaderRequest(
        destination="conversations",
        conversation_id="conversation-b",
        version=7,
        generation=3,
    )
    assert selected.selected_id == "conversation-b"
    assert selected.selected_version == 7
    assert selected.loaded_id == "conversation-a"
    assert selected.messages == loaded.messages
    assert selected.mode == "read"
    assert selected.loading is True
    assert selected.loaded_actions_eligible is False


def test_refreshing_same_identity_preserves_mode_but_fences_loaded_actions() -> None:
    loaded = _loaded_state(mode="info")

    refreshing, request = select_conversation(loaded, "conversation-a", version=4)

    assert request.generation == 3
    assert refreshing.mode == "info"
    assert refreshing.loaded_generation == 2
    assert refreshing.loaded_actions_eligible is False


def test_read_and_info_mode_changes_preserve_reader_identity() -> None:
    state = _loaded_state()

    info = set_conversation_reader_mode(state, "info")

    assert info.mode == "info"
    assert info.loaded_id == state.loaded_id
    assert info.loaded_actions_eligible is True


def _message(
    message_id: str,
    text: str,
    *,
    revision: str | None = None,
    total_chars: int | None = None,
    char_start: int = 0,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "sender": "user",
        "timestamp": f"2026-08-24T10:00:0{message_id[-1]}Z",
        "revision": revision or f"revision-{message_id}",
        "total_chars": len(text) if total_chars is None else total_chars,
        "char_start": char_start,
        "returned_chars": len(text),
        "has_more": char_start + len(text)
        < (len(text) if total_chars is None else total_chars),
        "text": text,
    }


def _page(
    conversation_id: str,
    version: int,
    messages: list[dict[str, Any]],
    *,
    offset: int = 0,
    total: int | None = None,
) -> dict[str, Any]:
    message_total = len(messages) if total is None else total
    return {
        "id": conversation_id,
        "title": conversation_id,
        "version": version,
        "message_total": message_total,
        "message_offset": offset,
        "returned_message_count": len(messages),
        "has_more": offset + len(messages) < message_total,
        "next_message_offset": (
            offset + len(messages) if offset + len(messages) < message_total else None
        ),
        "include_rag_context": False,
        "messages": messages,
    }


def test_initial_page_settles_exact_total_and_complete_transcript() -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )

    settled = settle_conversation_page(
        pending,
        request,
        _page(
            "conversation-a",
            4,
            [_message("message-1", "one"), _message("message-2", "two")],
        ),
    )

    assert [message.message_id for message in settled.messages] == [
        "message-1",
        "message-2",
    ]
    assert settled.message_total == 2
    assert settled.complete is True
    assert settled.loading is False
    assert settled.loaded_id == "conversation-a"
    assert settled.loaded_version == 4
    assert settled.loaded_generation == request.generation
    assert settled.loaded_actions_eligible is True


def test_subsequent_page_appends_in_order_and_duplicate_callback_is_idempotent() -> (
    None
):
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    first_request = replace(request, message_limit=2)
    first = settle_conversation_page(
        pending,
        first_request,
        _page(
            "conversation-a",
            4,
            [_message("message-1", "one"), _message("message-2", "two")],
            total=3,
        ),
    )
    second_page = _page(
        "conversation-a",
        4,
        [_message("message-3", "three")],
        offset=2,
        total=3,
    )
    second_request = replace(request, message_offset=2, message_limit=2)

    complete = settle_conversation_page(first, second_request, second_page)
    duplicate = settle_conversation_page(complete, second_request, second_page)

    assert [message.message_id for message in complete.messages] == [
        "message-1",
        "message-2",
        "message-3",
    ]
    assert complete.complete is True
    assert duplicate is complete


def test_overlapping_page_cannot_duplicate_or_replace_an_existing_message() -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    first = settle_conversation_page(
        pending,
        replace(request, message_limit=2),
        _page(
            "conversation-a",
            4,
            [_message("message-1", "one"), _message("message-2", "two")],
            total=3,
        ),
    )
    conflicting_overlap = _page(
        "conversation-a",
        4,
        [
            _message("message-2", "changed", revision="different"),
            _message("message-3", "three"),
        ],
        offset=1,
        total=3,
    )

    defended = settle_conversation_page(
        first,
        replace(request, message_offset=1, message_limit=2),
        conflicting_overlap,
    )

    assert defended is first
    assert [message.text for message in defended.messages] == ["one", "two"]


@pytest.mark.parametrize(
    ("response", "message_offset", "message_limit"),
    [
        (
            _page(
                "conversation-a",
                4,
                [_message("message-1", "one"), _message("message-2", "two")],
                total=1,
            ),
            0,
            2,
        ),
        (_page("conversation-a", 4, [], total=1), 0, 1),
        (
            {
                **_page(
                    "conversation-a",
                    4,
                    [_message("message-1", "one")],
                    total=2,
                ),
                "has_more": False,
            },
            0,
            1,
        ),
        (
            {
                **_page(
                    "conversation-a",
                    4,
                    [_message("message-1", "one")],
                    total=2,
                ),
                "next_message_offset": 2,
            },
            0,
            1,
        ),
        (
            _page(
                "conversation-a",
                4,
                [_message("message-2", "two")],
                offset=1,
                total=2,
            ),
            0,
            1,
        ),
        (
            _page(
                "conversation-a",
                4,
                [_message("message-1", "one")],
                total=3,
            ),
            0,
            2,
        ),
    ],
    ids=[
        "oversized",
        "empty-non-final",
        "has-more",
        "next-offset",
        "request-offset",
        "requested-cardinality",
    ],
)
def test_page_rejects_malformed_service_envelope_before_loading(
    response: dict[str, Any], message_offset: int, message_limit: int
) -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    page_request = replace(
        request,
        message_offset=message_offset,
        message_limit=message_limit,
    )

    with pytest.raises(ValueError, match="invalid coordinates"):
        settle_conversation_page(pending, page_request, response)

    assert pending.loaded_id is None
    assert pending.loaded_actions_eligible is False


def test_page_rejects_non_mapping_message_with_declared_validation_error() -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    response = _page("conversation-a", 4, [_message("message-1", "placeholder")])
    response["messages"] = ["not-a-message"]
    response["returned_message_count"] = 1

    with pytest.raises(ValueError, match="message mapping"):
        settle_conversation_page(pending, request, response)


@pytest.mark.parametrize("fence", ["destination", "id", "version", "generation"])
def test_page_settlement_rejects_each_stale_request_fence(fence: str) -> None:
    pending, current = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    values: dict[str, Any] = {
        "destination": current.destination,
        "conversation_id": current.conversation_id,
        "version": current.version,
        "generation": current.generation,
    }
    request_field = "conversation_id" if fence == "id" else fence
    values[request_field] = {
        "destination": "media",
        "id": "conversation-b",
        "version": 5,
        "generation": current.generation + 1,
    }[fence]
    stale = ConversationReaderRequest(**values)  # type: ignore[arg-type]

    result = settle_conversation_page(
        pending,
        stale,
        _page("conversation-a", 4, [_message("message-1", "one")]),
    )

    assert result is pending


def test_long_message_continuation_reassembles_once_and_completes_transcript() -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    initial = settle_conversation_page(
        pending,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message(
                    "message-1",
                    "Hello ",
                    revision="revision-1",
                    total_chars=11,
                )
            ],
        ),
    )
    continuation = _page(
        "conversation-a",
        4,
        [
            _message(
                "message-1",
                "world",
                revision="revision-1",
                total_chars=11,
                char_start=6,
            )
        ],
        total=1,
    )

    complete = settle_conversation_continuation(initial, request, continuation)
    duplicate = settle_conversation_continuation(complete, request, continuation)

    assert len(complete.messages) == 1
    assert complete.messages[0].text == "Hello world"
    assert complete.messages[0].complete is True
    assert complete.complete is True
    assert complete.loading is False
    assert duplicate is complete


@pytest.mark.parametrize(
    ("revision", "char_start"),
    [("stale-revision", 6), ("revision-1", 7)],
)
def test_continuation_rejects_revision_change_and_noncontiguous_window(
    revision: str, char_start: int
) -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    initial = settle_conversation_page(
        pending,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message(
                    "message-1",
                    "Hello ",
                    revision="revision-1",
                    total_chars=11,
                )
            ],
        ),
    )
    continuation = _page(
        "conversation-a",
        4,
        [
            _message(
                "message-1",
                "world",
                revision=revision,
                total_chars=11,
                char_start=char_start,
            )
        ],
        total=1,
    )

    defended = settle_conversation_continuation(initial, request, continuation)

    assert defended is initial
    assert defended.messages[0].text == "Hello "


def test_find_reports_partial_matches_as_incomplete_until_full_body_arrives() -> None:
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    initial = settle_conversation_page(
        pending,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message(
                    "message-1",
                    "Needle ",
                    revision="revision-1",
                    total_chars=20,
                )
            ],
        ),
    )

    finding = set_conversation_find_query(initial, "needle")

    assert finding.find_matches == ()
    assert finding.find_complete is False

    completed = settle_conversation_continuation(
        finding,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message(
                    "message-1",
                    "needle NEEDLE",
                    revision="revision-1",
                    total_chars=20,
                    char_start=7,
                )
            ],
            total=1,
        ),
    )

    assert [match.message_offset for match in completed.find_matches] == [0, 7, 14]
    assert completed.find_complete is True


def test_find_scans_once_only_after_many_continuations_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_count = 0
    find_matches = reader_state._find_matches

    def counted_find_matches(
        messages: tuple[ConversationMessageView, ...], query: str
    ) -> tuple[ConversationFindMatch, ...]:
        nonlocal scan_count
        scan_count += 1
        return find_matches(messages, query)

    monkeypatch.setattr(reader_state, "_find_matches", counted_find_matches)
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    finding = set_conversation_find_query(pending, "needle")
    current = settle_conversation_page(
        finding,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message(
                    "message-1",
                    "n",
                    revision="revision-1",
                    total_chars=6,
                )
            ],
        ),
    )

    assert scan_count == 0
    for char_start, text in enumerate("eedl", start=1):
        current = settle_conversation_continuation(
            current,
            request,
            _page(
                "conversation-a",
                4,
                [
                    _message(
                        "message-1",
                        text,
                        revision="revision-1",
                        total_chars=6,
                        char_start=char_start,
                    )
                ],
                total=1,
            ),
        )
        assert current.complete is False
        assert scan_count == 0

    complete = settle_conversation_continuation(
        current,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message(
                    "message-1",
                    "e",
                    revision="revision-1",
                    total_chars=6,
                    char_start=5,
                )
            ],
            total=1,
        ),
    )

    assert complete.complete is True
    assert scan_count == 1
    assert [match.message_offset for match in complete.find_matches] == [0]


def test_find_normalizes_case_and_reports_stable_message_and_transcript_offsets() -> (
    None
):
    pending, request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    loaded = settle_conversation_page(
        pending,
        request,
        _page(
            "conversation-a",
            4,
            [
                _message("message-1", "Needle needle"),
                _message("message-2", "ＮＥＥＤＬＥ"),
            ],
        ),
    )

    finding = set_conversation_find_query(loaded, " needle ")

    assert finding.find_matches == (
        ConversationFindMatch("message-1", 0, 0, 0, 6),
        ConversationFindMatch("message-1", 0, 7, 7, 6),
        ConversationFindMatch("message-2", 1, 0, 14, 6),
    )
    assert finding.find_complete is True


@pytest.mark.parametrize(
    "change",
    [
        {"selected_id": "conversation-b"},
        {"selected_version": 5},
        {"loaded_generation": 1},
    ],
)
def test_loaded_actions_require_matching_identity_version_and_generation(
    change: dict[str, Any],
) -> None:
    mismatched = replace(_loaded_state(), **change)

    assert mismatched.loaded_actions_eligible is False


def test_initial_error_is_retryable_and_stale_error_is_ignored() -> None:
    pending, first_request = select_conversation(
        ConversationReaderState(), "conversation-a", version=4
    )
    stale_request = replace(first_request, generation=first_request.generation + 1)

    stale = settle_conversation_error(pending, stale_request, "old error")
    failed = settle_conversation_error(pending, first_request, "  read failed  ")
    retrying, retry_request = retry_conversation(failed)

    assert stale is pending
    assert failed.error == "read failed"
    assert failed.loading is False
    assert failed.loaded_id is None
    assert retry_request.generation == first_request.generation + 1
    assert retrying.error is None
    assert retrying.loading is True


def test_stale_refresh_error_retains_labelled_preview_until_retry_succeeds() -> None:
    loaded = _loaded_state(mode="info")
    refreshing, request = select_conversation(loaded, "conversation-a", version=4)

    failed = settle_conversation_error(refreshing, request, "temporarily unavailable")

    assert failed.loaded_id == "conversation-a"
    assert failed.messages == loaded.messages
    assert failed.mode == "info"
    assert failed.loaded_actions_eligible is False

    retrying, retry_request = retry_conversation(failed)
    recovered = settle_conversation_page(
        retrying,
        retry_request,
        _page("conversation-a", 4, [_message("message-1", "hello")]),
    )

    assert recovered.error is None
    assert recovered.mode == "info"
    assert recovered.loaded_actions_eligible is True


def test_unavailable_new_selection_preserves_last_single_loaded_preview() -> None:
    selected, request = select_conversation(
        _loaded_state(), "conversation-b", version=7
    )

    unavailable = settle_conversation_unavailable(selected, request)

    assert unavailable.selected_id == "conversation-b"
    assert unavailable.loaded_id == "conversation-a"
    assert unavailable.messages == _loaded_state().messages
    assert unavailable.unavailable is True
    assert unavailable.loading is False
    assert unavailable.loaded_actions_eligible is False


def test_deleting_current_loaded_identity_clears_transcript_and_invalidates_request() -> (
    None
):
    loaded = _loaded_state(mode="info")
    request = ConversationReaderRequest(
        "conversations", "conversation-a", version=4, generation=2
    )

    deleted = mark_conversation_deleted(loaded, request)

    assert deleted.selected_id == "conversation-a"
    assert deleted.loaded_id is None
    assert deleted.messages == ()
    assert deleted.message_total == 0
    assert deleted.complete is False
    assert deleted.unavailable is True
    assert deleted.error == "Conversation deleted."
    assert deleted.generation == loaded.generation + 1
    assert deleted.loaded_actions_eligible is False


@pytest.mark.parametrize(
    "change",
    [
        {"destination": "media"},
        {"conversation_id": "conversation-b"},
        {"version": 5},
        {"generation": 3},
    ],
)
def test_deletion_rejects_each_wrong_request_fence(change: dict[str, Any]) -> None:
    loaded = _loaded_state()
    request = replace(
        ConversationReaderRequest(
            "conversations", "conversation-a", version=4, generation=2
        ),
        **change,
    )

    result = mark_conversation_deleted(loaded, request)

    assert result is loaded


def test_late_deletion_from_prior_generation_cannot_clear_newer_load() -> None:
    first_pending, first_request = select_conversation(
        _loaded_state(), "conversation-b", version=7
    )
    current_pending, current_request = select_conversation(
        first_pending, "conversation-b", version=7
    )
    current = settle_conversation_page(
        current_pending,
        current_request,
        _page("conversation-b", 7, [_message("message-2", "newer")]),
    )

    stale = mark_conversation_deleted(current, first_request)

    assert stale is current
    assert stale.loaded_id == "conversation-b"
    assert stale.messages[0].text == "newer"


def test_bulk_projection_is_read_only_and_preserves_last_single_preview() -> None:
    loaded = _loaded_state()

    bulk = project_conversation_multiselect(loaded, selected_count=3)

    assert bulk.bulk_selected_count == 3
    assert bulk.loaded_id == loaded.loaded_id
    assert bulk.messages == loaded.messages
    assert bulk.generation == loaded.generation
    assert bulk.loaded_generation == loaded.loaded_generation
    assert bulk.loaded_actions_eligible is False

    single = project_conversation_multiselect(bulk, selected_count=0)
    assert single.loaded_actions_eligible is True


def test_multiselect_invalidates_every_late_single_detail_settlement() -> None:
    loaded = _loaded_state()
    pending, request = select_conversation(loaded, "conversation-b", version=7)

    bulk = project_conversation_multiselect(pending, selected_count=2)
    page = _page("conversation-b", 7, [_message("message-2", "late")])

    assert bulk.generation == request.generation + 1
    assert bulk.loaded_generation == loaded.loaded_generation
    assert bulk.loading is False
    assert settle_conversation_page(bulk, request, page) is bulk
    assert settle_conversation_continuation(bulk, request, page) is bulk
    assert settle_conversation_error(bulk, request, "late error") is bulk
    assert settle_conversation_unavailable(bulk, request) is bulk
    assert mark_conversation_deleted(bulk, request) is bulk
    assert bulk.loaded_id == "conversation-a"
    assert bulk.messages == _loaded_state().messages
    assert bulk.bulk_selected_count == 2
    assert bulk.loaded_actions_eligible is False

    single = project_conversation_multiselect(bulk, selected_count=0)
    assert single.loaded_generation == loaded.loaded_generation
    assert single.loaded_actions_eligible is False

    reselected, reselect_request = select_conversation(
        single, "conversation-a", version=4
    )
    restored = settle_conversation_page(
        reselected,
        reselect_request,
        _page("conversation-a", 4, [_message("message-1", "hello")]),
    )
    assert restored.loaded_actions_eligible is True
