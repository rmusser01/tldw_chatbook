"""[dreams] One story URL into the read-it-later capture queue.

``ingest_story_url`` binds the modal's Ingest action to the Collections
capture seam's REAL public entry: ``CollectionsCaptureBackend.save_capture``
-- the async Protocol member ``LocalCollectionsCaptureService`` implements
(above ``_start_extraction``) and the app instantiates as
``local_collections_capture_service``. It submits the story as a
read-it-later ``CaptureSaveRequest`` attributed to Dreams through the
request's ``freeform_note`` field (the caller's ``source_note``).

Two shape adaptations, both forced by the real API:

- The entry is async, so the wrapper is async; its public contract
  otherwise stays the brief's ``-> str`` item identifier.
- ``CaptureSaveRequest`` requires the backend's authority key, which the
  Protocol itself does not carry. The wrapper resolves it off the backend's
  own surface (``authority`` on ``LocalCollectionsCaptureService`` and the
  Server adapter, ``active_authority`` on the scope service) and fails
  closed with the seam's typed error when neither is present.

URL shape goes through the shared ``validate_url`` seam (http/https only,
no whitespace/credentials/backslashes) -- the same validation the Library
quick-capture form applies to this exact operation. Backend failures
propagate unchanged for the caller to surface as a notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tldw_chatbook.Library.collections_capture_models import (
    CaptureSaveRequest,
    CollectionsCaptureError,
)
from tldw_chatbook.Utils.input_validation import validate_url

if TYPE_CHECKING:
    from tldw_chatbook.Library.collections_capture_service import (
        CollectionsCaptureBackend,
    )

#: Returned when the backend cannot tell whether the save landed
#: (``CaptureSaveOutcome.outcome_unknown`` -- a server-mode timeout shape;
#: the Local backend never produces it). Inventing a capture id there would
#: be a hallucinated receipt, so the caller gets this short status string.
UNKNOWN_OUTCOME_IDENTIFIER = "unknown"


def _authority_key(backend: Any) -> str:
    """The backend's own authority key, or the seam's typed error."""
    authority = getattr(backend, "authority", None) or getattr(
        backend, "active_authority", None
    )
    key = getattr(authority, "key", None)
    if not isinstance(key, str) or not key.strip():
        raise CollectionsCaptureError("capture_authority_unavailable")
    return key


async def ingest_story_url(
    backend: CollectionsCaptureBackend,
    *,
    url: str,
    title: str,
    source_note: str,
) -> str:
    """Submit one story URL as a read-it-later capture attributed to Dreams.

    Args:
        backend: Any ``CollectionsCaptureBackend`` implementation (the
            app's ``local_collections_capture_service``, the Server
            adapter, or the scope service) whose ``save_capture`` member
            is the real entry.
        url: The story's source URL; must be an http/https URL with no
            whitespace (shared ``validate_url`` seam).
        title: The story title, stored on the capture item.
        source_note: Dreams attribution text, stored as the item's
            freeform note.

    Returns:
        The saved item's ``capture_id`` identifier, or
        ``UNKNOWN_OUTCOME_IDENTIFIER`` when the backend reports an
        outcome-unknown save.

    Raises:
        ValueError: The URL failed shape validation (never submitted).
        CollectionsCaptureError: No authority key could be resolved off
            the backend, or the backend raised its typed error.
    """
    if not validate_url(str(url or "")):
        raise ValueError("invalid story URL: expected an http(s) URL without whitespace")
    outcome = await backend.save_capture(
        CaptureSaveRequest(
            _authority_key(backend),
            str(url),
            title=str(title or "") or None,
            freeform_note=str(source_note or "") or None,
        )
    )
    capture = getattr(outcome, "capture", None)
    if capture is None:
        return UNKNOWN_OUTCOME_IDENTIFIER
    return str(capture.identity.capture_id)
