"""Neutral Library content evidence contracts."""

from collections.abc import Collection
from enum import Enum
from typing import Any

LIBRARY_CONTENT_SOURCES = (
    "notes",
    "media",
    "conversations",
    "prompts",
    "skills",
    "collections",
    "artifacts",
)


class LibraryContentEvidence(str, Enum):
    """Source-owned evidence of eligible user content."""

    UNKNOWN = "unknown"
    EMPTY = "empty"
    HAS_USER_CONTENT = "has_user_content"


class LibraryEvidenceStatus(str, Enum):
    """Presentation status for a Library evidence request."""

    LOADING = "loading"
    SETTLED = "settled"
    PARTIAL_FAILURE = "partial_failure"


async def get_library_artifact_content_evidence(
    *,
    subscriptions_db: Any = None,
    chachanotes_db: Any = None,
    local_chatbook_service: Any = None,
    unavailable_sources: Collection[str] = (),
) -> LibraryContentEvidence:
    """Read artifact evidence from existing local owners without creating them.

    Invoke through the screen's isolated service worker: the DB methods and
    the async registry method perform blocking local reads. Each owner returns
    at most one row; the JSON registry still parses its existing file.

    Args:
        subscriptions_db: Existing local live-report owner, when configured.
        chachanotes_db: Existing local kept-report owner, when configured.
        local_chatbook_service: Existing local registry owner, when configured.
        unavailable_sources: Configured owners unavailable to the caller, named
            ``live_reports``, ``kept_reports``, or ``chatbooks``. Absent owners
            otherwise mean unconfigured, which contributes empty evidence.

    Returns:
        Positive evidence from any owner wins. All known-empty or unconfigured
        owners yield EMPTY; a failed or incomplete read otherwise yields UNKNOWN.
    """
    unknown = bool(unavailable_sources)
    for owner, method_name in (
        (subscriptions_db, "list_recent_briefings"),
        (chachanotes_db, "list_kept_briefings"),
        (local_chatbook_service, "list_chatbooks"),
    ):
        if owner is None:
            continue
        try:
            read = getattr(owner, method_name)
            rows = read(limit=1)
            if method_name == "list_chatbooks":
                rows = await rows
            if not isinstance(rows, list):
                unknown = True
            elif rows:
                return LibraryContentEvidence.HAS_USER_CONTENT
        except Exception:  # noqa: BLE001 - unavailable owners must remain unknown
            unknown = True
    return LibraryContentEvidence.UNKNOWN if unknown else LibraryContentEvidence.EMPTY
