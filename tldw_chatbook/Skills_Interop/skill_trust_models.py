"""Local skill trust state models and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TRUST_STATUS_TRUSTED = "trusted"
TRUST_STATUS_UNINITIALIZED = "trust_uninitialized"
TRUST_STATUS_LOCKED = "trust_locked"
TRUST_STATUS_QUARANTINED_MODIFIED = "quarantined_modified"
TRUST_STATUS_QUARANTINED_ADDED = "quarantined_added"
TRUST_STATUS_QUARANTINED_DELETED = "quarantined_deleted"
TRUST_STATUS_QUARANTINED_MANIFEST_ERROR = "quarantined_manifest_error"
TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH = "quarantined_unsupported_path"

TRUST_REASON_SKILL_MODIFIED = "skill_modified"
TRUST_REASON_SKILL_ADDED = "skill_added"
TRUST_REASON_SKILL_DELETED = "skill_deleted"
TRUST_REASON_UNSUPPORTED_PATH = "unsupported_path"
TRUST_REASON_TRUST_LOCKED = "trust_locked"
TRUST_REASON_TRUST_UNINITIALIZED = "trust_uninitialized"
TRUST_REASON_MANIFEST_INVALID = "manifest_invalid"
TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE = "rollback_marker_unavailable"
TRUST_REASON_SNAPSHOT_MISMATCH = "snapshot_mismatch"
TRUST_REASON_STORE_WRITE_FAILED = "trust_store_write_failed"
TRUST_REASON_HISTORY_UNRECOVERABLE = "trust_history_unrecoverable"


@dataclass(frozen=True, slots=True)
class SkillFileFingerprint:
    """Stable fingerprint metadata for one local skill file."""

    relative_path: str
    file_type: str
    byte_length: int
    sha256: str

    def as_manifest_entry(self) -> dict[str, Any]:
        """Return the JSON-safe manifest representation for this fingerprint."""

        return {
            "relative_path": self.relative_path,
            "file_type": self.file_type,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SkillDirectorySnapshot:
    """Current or trusted snapshot of a Chatbook-managed local skill directory."""

    skill_name: str
    fingerprints: tuple[SkillFileFingerprint, ...]
    text_files: dict[str, str]
    unsupported_paths: tuple[str, ...] = ()

    @property
    def fingerprint_map(self) -> dict[str, SkillFileFingerprint]:
        """Return fingerprints keyed by relative path."""

        return {item.relative_path: item for item in self.fingerprints}


@dataclass(frozen=True, slots=True)
class SkillTrustStatus:
    """Trust posture exposed by local skill list/detail responses."""

    skill_name: str
    trust_status: str
    trust_reason_code: str | None
    trust_blocked: bool
    changed_files: tuple[str, ...]
    manifest_generation: int | None
    last_verified_at: str | None

    def response_fields(self) -> dict[str, Any]:
        """Return JSON-safe trust fields for API/UI response payloads."""

        return {
            "trust_status": self.trust_status,
            "trust_reason_code": self.trust_reason_code,
            "trust_blocked": self.trust_blocked,
            "trust_changed_files": list(self.changed_files),
            "trust_manifest_generation": self.manifest_generation,
            "trust_last_verified_at": self.last_verified_at,
        }


class SkillTrustBlockedError(RuntimeError):
    """Raised when a trust-blocked local skill is staged or executed."""

    def __init__(
        self,
        *,
        skill_name: str,
        reason_code: str,
        trust_status: str,
        changed_files: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"Local skill {skill_name} is trust-blocked: {reason_code}")
        self.skill_name = skill_name
        self.reason_code = reason_code
        self.trust_status = trust_status
        self.changed_files = changed_files
