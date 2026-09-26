"""Dreams ingest action: one story URL into read-it-later (Task 8).

Binds to the REAL capture seam: ``CollectionsCaptureBackend.save_capture``
(the async Protocol member ``LocalCollectionsCaptureService`` implements),
submitting the story as a read-it-later ``CaptureSaveRequest`` attributed
to Dreams through its ``freeform_note`` field. The fake implements that
real Protocol method (never the brief sketch's invented name) and carries
the same ``authority`` surface the concrete services expose, because the
request requires the backend's authority key.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Dreams.ingest_action import ingest_story_url
from tldw_chatbook.Library.collections_capture_models import (
    CaptureAuthority,
    CaptureDetail,
    CaptureIdentity,
    CaptureSaveOutcome,
    CaptureSaveRequest,
    CollectionsCaptureError,
)


class FakeCaptureBackend:
    """Test seam: the Protocol's real ``save_capture(request)`` member.

    Implements the same ``authority`` attribute the concrete services
    (``LocalCollectionsCaptureService`` / the Server adapter) carry, since
    ``CaptureSaveRequest`` requires the authority key and the wrapper
    resolves it off the backend rather than a second parameter.
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        self.authority = CaptureAuthority(
            "local", "local:dreams-test", "dreams-test"
        )
        self.requests: list[CaptureSaveRequest] = []
        self.error = error
        self.unknown_outcome = False

    async def save_capture(
        self, request: CaptureSaveRequest
    ) -> CaptureSaveOutcome:
        if self.error is not None:
            raise self.error
        self.requests.append(request)
        if self.unknown_outcome:
            return CaptureSaveOutcome(None, None, outcome_unknown=True)
        return CaptureSaveOutcome(
            CaptureDetail(
                CaptureIdentity(self.authority.key, f"cap-{len(self.requests)}"),
                request.submitted_url,
            ),
            True,
        )


async def test_ingest_submits_url_with_dreams_attribution():
    backend = FakeCaptureBackend()
    ident = await ingest_story_url(
        backend,
        url="https://example.com/flights",
        title="Cheap flights",
        source_note="via Dreams 2026-09-22",
    )
    assert ident == "cap-1", "the backend's item identifier comes back"
    (request,) = backend.requests
    assert request.submitted_url == "https://example.com/flights"
    assert request.title == "Cheap flights"
    assert request.freeform_note == "via Dreams 2026-09-22"
    assert request.authority_key == backend.authority.key


async def test_ingest_propagates_backend_failure():
    backend = FakeCaptureBackend(
        error=CollectionsCaptureError("capture_queue_full")
    )
    with pytest.raises(CollectionsCaptureError, match="capture_queue_full"):
        await ingest_story_url(
            backend, url="https://example.com/2", title="t", source_note="n"
        )
    assert backend.requests == [], "a failed save submits nothing"


async def test_ingest_rejects_non_http_and_whitespace_urls():
    backend = FakeCaptureBackend()
    for bad_url in (
        "ftp://example.com/file",  # wrong scheme
        "https://ex ample.com/",  # embedded whitespace
        "  https://example.com/",  # leading whitespace
        "",  # empty
        "https://user:pass@example.com/",  # embedded credentials
    ):
        with pytest.raises(ValueError):
            await ingest_story_url(
                backend, url=bad_url, title="t", source_note="n"
            )
    assert backend.requests == [], "no rejected URL may reach the backend"


async def test_ingest_returns_unknown_identifier_when_outcome_unknown():
    backend = FakeCaptureBackend()
    backend.unknown_outcome = True
    ident = await ingest_story_url(
        backend, url="https://example.com/3", title="t", source_note="n"
    )
    assert ident == "unknown", (
        "an outcome-unknown save must not invent a capture id"
    )


async def test_ingest_fails_closed_without_a_backend_authority():
    backend = FakeCaptureBackend()
    del backend.authority  # a bare Protocol object carries no authority
    with pytest.raises(CollectionsCaptureError, match="capture_authority"):
        await ingest_story_url(
            backend, url="https://example.com/4", title="t", source_note="n"
        )
    assert backend.requests == []
