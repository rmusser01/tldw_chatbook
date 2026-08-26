"""Neutral Library content evidence contracts."""

from enum import Enum


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
