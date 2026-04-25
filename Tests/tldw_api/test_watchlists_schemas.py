import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceListResponse,
    SourceResponse,
    SourceUpdateRequest,
)


def test_source_create_request_allows_first_slice_types_and_strips_tags():
    request = SourceCreateRequest(
        name="AI News",
        url="https://example.com/feed.xml",
        source_type="rss",
        tags=[" ai ", "", None, "ml"],
    )

    assert request.source_type == "rss"
    assert request.tags == ["ai", "ml"]
    assert request.model_dump(exclude_none=True, mode="json") == {
        "name": "AI News",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "active": True,
        "tags": ["ai", "ml"],
        "settings": {},
    }


def test_source_create_request_rejects_deferred_forum_type():
    with pytest.raises(ValidationError):
        SourceCreateRequest(
            name="Forum",
            url="https://example.com/forum",
            source_type="forum",
        )


def test_source_update_request_forbids_group_ids_in_first_slice():
    with pytest.raises(ValidationError):
        SourceUpdateRequest(group_ids=[1])


def test_source_list_and_delete_responses_preserve_restore_metadata():
    listed = SourceListResponse.model_validate(
        {
            "items": [
                {
                    "id": 17,
                    "name": "AI News",
                    "url": "https://example.com/feed.xml",
                    "source_type": "rss",
                    "group_ids": [3, 5],
                    "settings": {"rss": {"limit": 50}},
                }
            ],
            "total": 1,
        }
    )
    deleted = SourceDeleteResponse(
        success=True,
        source_id=17,
        restore_window_seconds=10,
        restore_expires_at="2026-04-21T12:00:00Z",
    )

    assert isinstance(listed.items[0], SourceResponse)
    assert listed.items[0].group_ids == [3, 5]
    assert listed.items[0].settings == {"rss": {"limit": 50}}
    assert deleted.restore_window_seconds == 10
