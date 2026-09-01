"""Source-neutral contracts for the Collections capture reader."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given, strategies as st

from tldw_chatbook.Library.collections_capture_models import (
    CAPTURE_CAPABILITY_NAMES,
    CAPTURE_PAGE_SIZE,
    CapabilityState,
    CaptureActionResult,
    CaptureAuthority,
    CaptureCapabilities,
    CaptureCapability,
    CaptureConflict,
    CaptureContentResult,
    CaptureDetail,
    CaptureHighlight,
    CaptureHighlightDraft,
    CaptureIdentity,
    CaptureNoteLink,
    CaptureOfflineCopy,
    CapturePage,
    CapturePageRequest,
    CaptureSaveOutcome,
    CaptureSaveRequest,
    CaptureSavedSearchPage,
    CaptureSummary,
    CollectionsCaptureError,
    ExternalMediaReference,
    ExternalNoteReference,
    ExternalReferenceAvailability,
    ResolvedCaptureDetail,
    SavedCaptureSearch,
)


def _summary(
    capture_id: str,
    *,
    authority_key: str = "local:alpha",
) -> CaptureSummary:
    return CaptureSummary(
        identity=CaptureIdentity(authority_key, capture_id),
        canonical_url=f"https://example.org/{capture_id}",
        title=f"Capture {capture_id}",
        created_at="2026-09-01T10:00:00Z",
        updated_at="2026-09-01T10:00:00Z",
    )


def test_page_request_normalizes_exact_filters_and_is_frozen() -> None:
    request = CapturePageRequest.from_mapping(
        {
            "authority_key": " local:alpha ",
            "search": "  exact   phrase  ",
            "statuses": [" READ ", "saved", "read"],
            "favorite": True,
            "tags": [" Topic ", "topic", "Other"],
            "domain": " Example.COM ",
            "date_from": " 2026-01-01 ",
            "date_to": " 2026-12-31 ",
            "sort": "title_asc",
            "page": 2,
            "size": 20,
        }
    )

    assert request.authority_key == "local:alpha"
    assert request.search == "exact phrase"
    assert request.statuses == ("read", "saved")
    assert request.tags == ("other", "topic")
    assert request.domain == "example.com"
    assert request.date_from == "2026-01-01"
    assert request.date_to == "2026-12-31"
    with pytest.raises(FrozenInstanceError):
        request.page = 3  # type: ignore[misc]


@pytest.mark.parametrize("size", [0, 1, 19, 21, 100])
def test_page_request_requires_fixed_twenty_row_size(size: int) -> None:
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePageRequest(authority_key="local:alpha", size=size)
    assert caught.value.reason == "invalid_page_size"


@pytest.mark.parametrize("page", [-10, -1, 0])
def test_page_request_is_one_based(page: int) -> None:
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePageRequest(authority_key="local:alpha", page=page)
    assert caught.value.reason == "invalid_page"


@pytest.mark.parametrize(
    "sort",
    [
        "saved_desc",
        "saved_asc",
        "updated_desc",
        "updated_asc",
        "title_asc",
        "title_desc",
    ],
)
def test_page_request_accepts_source_neutral_sorts(sort: str) -> None:
    assert CapturePageRequest(authority_key="local:alpha", sort=sort).sort == sort


def test_relevance_sort_requires_nonblank_search() -> None:
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePageRequest(authority_key="local:alpha", sort="relevance")
    assert caught.value.reason == "relevance_requires_search"
    assert (
        CapturePageRequest(
            authority_key="local:alpha", search="reader", sort="relevance"
        ).sort
        == "relevance"
    )


@given(
    sort=st.text(min_size=1).filter(
        lambda value: value.strip()
        not in {
            "saved_desc",
            "saved_asc",
            "updated_desc",
            "updated_asc",
            "title_asc",
            "title_desc",
            "relevance",
        }
    )
)
def test_page_request_rejects_unknown_sorts(sort: str) -> None:
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePageRequest(authority_key="local:alpha", search="q", sort=sort)
    assert caught.value.reason == "invalid_sort"


@given(key=st.text(min_size=1).filter(lambda value: value not in {"search", "q"}))
def test_saved_search_mapping_rejects_unknown_and_nested_keys(key: str) -> None:
    with pytest.raises(CollectionsCaptureError):
        CapturePageRequest.from_mapping(
            {"authority_key": "local:alpha", key: {"nested": "expression"}}
        )


def test_page_rejects_duplicate_ids_and_authority_mismatch() -> None:
    request = CapturePageRequest(authority_key="local:alpha")
    duplicate = _summary("same")
    with pytest.raises(CollectionsCaptureError) as duplicate_error:
        CapturePage(applied=request, items=(duplicate, duplicate), total=2)
    assert duplicate_error.value.reason == "duplicate_capture_id"

    with pytest.raises(CollectionsCaptureError) as authority_error:
        CapturePage(
            applied=request,
            items=(_summary("other", authority_key="server:beta"),),
            total=1,
        )
    assert authority_error.value.reason == "page_authority_mismatch"


@given(extra=st.integers(min_value=1, max_value=30))
def test_page_rejects_oversized_rows(extra: int) -> None:
    request = CapturePageRequest(authority_key="local:alpha")
    items = tuple(_summary(str(index)) for index in range(CAPTURE_PAGE_SIZE + extra))
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePage(applied=request, items=items, total=len(items))
    assert caught.value.reason == "oversized_page"


@given(row_count=st.integers(min_value=0, max_value=CAPTURE_PAGE_SIZE - 1))
def test_page_rejects_undersized_nonfinal_rows(row_count: int) -> None:
    request = CapturePageRequest(authority_key="local:alpha")
    items = tuple(_summary(str(index)) for index in range(row_count))
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePage(applied=request, items=items, total=row_count + 1)
    assert caught.value.reason == "undersized_nonfinal_page"


def test_page_rejects_impossible_total_for_later_page() -> None:
    request = CapturePageRequest(authority_key="local:alpha", page=3)
    with pytest.raises(CollectionsCaptureError) as caught:
        CapturePage(applied=request, items=(_summary("41"),), total=40)
    assert caught.value.reason == "impossible_total"


def test_page_accepts_exact_final_page_and_source_revision() -> None:
    request = CapturePageRequest(authority_key="local:alpha", page=3)
    page = CapturePage(
        applied=request,
        items=tuple(_summary(str(index)) for index in range(41, 46)),
        total=45,
        source_revision="epoch-7",
    )
    assert page.total == 45
    assert page.items[-1].identity.capture_id == "45"


def test_capabilities_require_every_known_action_and_are_immutable() -> None:
    capabilities = CaptureCapabilities(
        {
            action: CaptureCapability(CapabilityState.SUPPORTED)
            for action in CAPTURE_CAPABILITY_NAMES
        }
    )
    assert capabilities.for_action("capture").state is CapabilityState.SUPPORTED
    with pytest.raises(TypeError):
        capabilities.values["capture"] = CaptureCapability(  # type: ignore[index]
            CapabilityState.UNSUPPORTED
        )

    missing = {
        action: CaptureCapability(CapabilityState.UNKNOWN)
        for action in CAPTURE_CAPABILITY_NAMES
        if action != "hard_delete"
    }
    with pytest.raises(CollectionsCaptureError) as caught:
        CaptureCapabilities(missing)
    assert caught.value.reason == "invalid_capability_set"

    complete = dict(capabilities.values)
    complete["made_up"] = CaptureCapability(CapabilityState.SUPPORTED)
    with pytest.raises(CollectionsCaptureError):
        CaptureCapabilities(complete)


def test_saved_search_page_validates_scope_and_paging() -> None:
    request = CapturePageRequest(authority_key="local:alpha", statuses=("saved",))
    search = SavedCaptureSearch(
        authority_key="local:alpha",
        search_id="search-1",
        name="Saved",
        request=request,
        created_at="2026-09-01",
        updated_at="2026-09-01",
        revision=1,
    )
    page = CaptureSavedSearchPage(items=(search,), total=1, page=1)
    assert page.size == CAPTURE_PAGE_SIZE

    with pytest.raises(CollectionsCaptureError) as caught:
        SavedCaptureSearch(
            authority_key="server:beta",
            search_id="search-2",
            name="Wrong authority",
            request=request,
            created_at="2026-09-01",
            updated_at="2026-09-01",
            revision=1,
        )
    assert caught.value.reason == "saved_search_authority_mismatch"


def test_remaining_contracts_keep_external_references_authority_qualified() -> None:
    authority = CaptureAuthority("local", "local:alpha", "a1b2c3d4")
    identity = CaptureIdentity("local:alpha", "capture-1")
    media = ExternalMediaReference("media:profile-a", "media-9")
    note = ExternalNoteReference("notes:profile-a", "note-4")
    offline = CaptureOfflineCopy(
        identity=identity,
        file_id="file-1",
        state="ready",
        content_hash="sha256:abc",
        size=12,
        media_type="text/html",
        revision=1,
    )
    detail = CaptureDetail(
        identity=identity,
        canonical_url="https://example.org/capture-1",
        submitted_url="https://example.org/capture-1?source=inbox",
        title="Capture",
        media_reference=media,
        offline_copy=offline,
        revision=2,
    )
    link = CaptureNoteLink(identity, "link-1", note, "2026-09-01")
    availability = ExternalReferenceAvailability("available")
    resolved = ResolvedCaptureDetail(
        capture=detail,
        media=availability,
        note_links=((link, availability),),
    )

    assert authority.key == identity.authority_key
    assert resolved.capture.media_reference == media
    assert resolved.note_links[0][0].note_reference == note
    assert CaptureHighlightDraft("quote").quote == "quote"
    assert CaptureHighlight(
        identity,
        "highlight-1",
        "quote",
        None,
        None,
        False,
        "2026-09-01",
        "2026-09-01",
        1,
    ).revision == 1
    request = CaptureSaveRequest(
        authority_key="local:alpha",
        submitted_url="https://example.org/capture-1",
        tags=("Research",),
    )
    assert request.tags == ("Research",)
    assert CaptureSaveOutcome(detail, created=True).capture == detail
    assert CaptureConflict(identity, expected_revision=1, current=detail).actual_revision == 2
    assert CaptureActionResult(identity, success=True, revision=3).success is True
    assert CaptureContentResult(identity, "summary", text="Short summary").text


def test_typed_error_exposes_bounded_reason_and_retryability() -> None:
    error = CollectionsCaptureError("server_offline", retryable=True)
    assert error.reason == "server_offline"
    assert error.retryable is True
