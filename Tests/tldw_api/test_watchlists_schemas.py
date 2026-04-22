from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.watchlists_schemas import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceResponse,
)


def test_source_delete_response_keeps_restore_window_metadata():
    payload = SourceDeleteResponse(
        success=True,
        source_id=17,
        restore_window_seconds=10,
        restore_expires_at="2026-04-21T12:00:00Z",
    )
    assert payload.restore_window_seconds == 10


def test_source_create_request_only_allows_first_slice_source_types():
    request = SourceCreateRequest(
        name="AI",
        url="https://example.com/feed.xml",
        source_type="rss",
    )

    assert request.source_type == "rss"

    with pytest.raises(ValidationError):
        SourceCreateRequest(
            name="AI",
            url="https://example.com/feed.xml",
            source_type="forum",
        )


def test_source_response_exposes_read_only_group_ids():
    response = SourceResponse(
        id=17,
        name="AI",
        url="https://example.com/feed.xml",
        source_type="rss",
    )

    assert response.group_ids == []
