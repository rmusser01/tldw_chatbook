"""Repository-only result and metadata types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    quarantine_id: str
    object_type: str
    object_id: str
    version_id: str | None
    reason_code: str
    created_at: str
