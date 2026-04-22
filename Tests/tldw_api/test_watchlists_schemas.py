from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.watchlists_schemas import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceResponse,
    SourceUpdateRequest,
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

    with pytest.raises(ValidationError):
        SourceCreateRequest(
            name="AI",
            url="https://example.com/feed.xml",
            source_type="rss",
            group_ids=[1],
        )

    with pytest.raises(ValidationError):
        SourceCreateRequest(
            name="AI",
            url="https://example.com/feed.xml",
            source_type="rss",
            settings={"legacy": True},
        )


def test_source_update_request_keeps_first_slice_surface():
    request = SourceUpdateRequest(
        name="AI",
        url="https://example.com/feed.xml",
        source_type="site",
        active=False,
    )

    assert request.source_type == "site"
    assert "group_ids" not in SourceUpdateRequest.model_fields
    assert "settings" not in SourceUpdateRequest.model_fields

    with pytest.raises(ValidationError):
        SourceUpdateRequest(
            name="AI",
            url="https://example.com/feed.xml",
            source_type="forum",
        )

    with pytest.raises(ValidationError):
        SourceUpdateRequest(
            name="AI",
            url="https://example.com/feed.xml",
            source_type="rss",
            group_ids=[1],
        )

    with pytest.raises(ValidationError):
        SourceUpdateRequest(
            name="AI",
            url="https://example.com/feed.xml",
            source_type="rss",
            settings={"legacy": True},
        )


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (
            SourceCreateRequest,
            {"name": "", "url": "https://example.com/feed.xml", "source_type": "rss"},
        ),
        (
            SourceUpdateRequest,
            {"name": "", "url": "https://example.com/feed.xml", "source_type": "site"},
        ),
    ],
)
def test_source_request_rejects_blank_names(model_cls, kwargs):
    with pytest.raises(ValidationError):
        model_cls(**kwargs)


def test_source_response_exposes_read_only_group_ids():
    response = SourceResponse(
        id=17,
        name="AI",
        url="https://example.com/feed.xml",
        source_type="forum",
        last_scraped_at="2026-04-21T12:00:00Z",
        status="active",
        created_at="2026-04-20T12:00:00Z",
        updated_at="2026-04-21T12:00:00Z",
    )

    assert response.group_ids == []
    assert response.source_type == "forum"
    assert response.last_scraped_at == "2026-04-21T12:00:00Z"
    assert response.status == "active"
    assert response.created_at == "2026-04-20T12:00:00Z"
    assert response.updated_at == "2026-04-21T12:00:00Z"
