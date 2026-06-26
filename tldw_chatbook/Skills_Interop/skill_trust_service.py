"""Orchestration service for Chatbook-managed local skill trust."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tldw_api.skills_schemas import _normalize_skill_name as _normalize_api_skill_name
from .skill_trust_crypto import (
    SkillTrustKeys,
    canonical_json,
    derive_skill_trust_keys,
    sha256_hex,
)
from .skill_trust_models import (
    SkillDirectorySnapshot,
    SkillTrustBlockedError,
    SkillTrustStatus,
    TRUST_REASON_MANIFEST_INVALID,
    TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE,
    TRUST_REASON_SKILL_ADDED,
    TRUST_REASON_SKILL_DELETED,
    TRUST_REASON_SKILL_MODIFIED,
    TRUST_REASON_TRUST_LOCKED,
    TRUST_REASON_TRUST_UNINITIALIZED,
    TRUST_REASON_UNSUPPORTED_PATH,
    TRUST_STATUS_LOCKED,
    TRUST_STATUS_QUARANTINED_ADDED,
    TRUST_STATUS_QUARANTINED_DELETED,
    TRUST_STATUS_QUARANTINED_MANIFEST_ERROR,
    TRUST_STATUS_QUARANTINED_MODIFIED,
    TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH,
    TRUST_STATUS_TRUSTED,
    TRUST_STATUS_UNINITIALIZED,
)
from .skill_trust_scanner import scan_skill_directory
from .skill_trust_store import SkillTrustMarkerUnavailable, SkillTrustStore


_SKILL_FILENAME = "SKILL.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillTrustService:
    """Coordinate trust bootstrap, classification, review, and approval."""

    def __init__(
        self,
        *,
        skills_dir: Path,
        trust_store: SkillTrustStore,
        key_cache: Any | None = None,
        keyring_convenience_enabled: bool = False,
        reduced_rollback_protection: bool = False,
    ) -> None:
        self.skills_dir = skills_dir
        self.trust_store = trust_store
        self.key_cache = key_cache
        self.keyring_convenience_enabled = keyring_convenience_enabled
        self.reduced_rollback_protection = reduced_rollback_protection
        self._keys: SkillTrustKeys | None = None
        self._salt: bytes | None = None
        self._reviews: dict[str, dict[str, Any]] = {}

    def unlock_with_passphrase(self, passphrase: str, *, salt: bytes | None = None) -> None:
        """Derive in-memory trust keys from a passphrase and manifest salt."""

        if salt is None:
            salt = self.trust_store.load_salt()
        self._keys = derive_skill_trust_keys(passphrase, salt=salt)
        self._salt = salt

    def enable_keyring_convenience(self) -> None:
        """Persist derived trust keys in secure keyring storage, never a passphrase."""

        if self.key_cache is None:
            raise SkillTrustMarkerUnavailable("No secure OS-backed key cache is available.")
        keys = self._require_keys()
        salt = self._require_salt()
        self.key_cache.save_keys(keys, salt=salt)
        self.keyring_convenience_enabled = True

    def unlock_from_keyring_convenience(self) -> bool:
        """Load derived trust keys from a salt-bound secure keyring cache."""

        if self.key_cache is None or not self.trust_store.has_manifest():
            return False
        try:
            salt = self.trust_store.load_salt()
            keys = self.key_cache.load_keys(expected_salt=salt)
        except (OSError, ValueError, SkillTrustMarkerUnavailable):
            return False
        if keys is None:
            return False
        self._keys = keys
        self._salt = salt
        self.keyring_convenience_enabled = True
        return True

    def overall_status(self) -> str:
        """Return a global Settings-friendly trust posture from live files."""

        if not self.trust_store.has_manifest():
            missing_status = self._manifest_missing_status("<all>")
            return missing_status.trust_status
        if self._keys is None:
            return TRUST_STATUS_LOCKED
        try:
            manifest = self._load_valid_manifest()
        except SkillTrustMarkerUnavailable:
            return TRUST_STATUS_QUARANTINED_MANIFEST_ERROR
        except ValueError:
            return TRUST_STATUS_QUARANTINED_MANIFEST_ERROR
        try:
            skill_names = self._known_and_current_skill_names(manifest)
        except ValueError:
            return TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH
        for skill_name in sorted(skill_names):
            status = self.status_for_skill(skill_name)
            if status.trust_blocked:
                return status.trust_status
        return TRUST_STATUS_TRUSTED

    def bootstrap_trust(self, passphrase: str | None = None, *, salt: bytes | None = None) -> None:
        """Trust the current local skill directories as the initial baseline."""

        if passphrase is not None:
            self.unlock_with_passphrase(passphrase, salt=salt or secrets.token_bytes(32))
        keys = self._require_keys()
        manifest_salt = self._require_salt()
        generation = 1
        skills: dict[str, Any] = {}

        for normalized_name, skill_dir in self._iter_skill_dirs():
            snapshot = scan_skill_directory(normalized_name, skill_dir)
            if snapshot.unsupported_paths:
                raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
            snapshot_id = self._snapshot_id(normalized_name, generation)
            self.trust_store.save_snapshot(
                snapshot_id,
                {"files": dict(snapshot.text_files)},
                keys,
                generation=generation,
            )
            skills[normalized_name] = self._manifest_skill_entry(
                snapshot=snapshot,
                snapshot_id=snapshot_id,
                snapshot_generation=generation,
            )

        manifest = {
            "version": 1,
            "generation": generation,
            "skills": skills,
            "audit": [
                {
                    "event": "trust_bootstrap",
                    "at": _now_iso(),
                    "skill_count": len(skills),
                }
            ],
        }
        self.trust_store.save_manifest(manifest, keys, salt=manifest_salt)

    def status_for_skill(self, skill_name: str) -> SkillTrustStatus:
        """Return visible trust status without raising for locked or bad manifests."""

        try:
            normalized_name = self._normalize_skill_name(skill_name)
        except ValueError:
            return self._unsupported_name_status(skill_name)
        if not self.trust_store.has_manifest():
            return self._manifest_missing_status(normalized_name)
        if self._keys is None:
            return self._locked_status(normalized_name)
        try:
            manifest = self._load_valid_manifest()
        except SkillTrustMarkerUnavailable:
            return self._manifest_error_status(
                normalized_name,
                TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE,
            )
        except ValueError as exc:
            return self._manifest_error_status(
                normalized_name,
                self._manifest_error_reason(exc),
            )

        generation = int(manifest["generation"])
        current = self._scan_skill(normalized_name)
        if current.unsupported_paths:
            return SkillTrustStatus(
                skill_name=normalized_name,
                trust_status=TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH,
                trust_reason_code=TRUST_REASON_UNSUPPORTED_PATH,
                trust_blocked=True,
                changed_files=tuple(current.unsupported_paths),
                manifest_generation=generation,
                last_verified_at=_now_iso(),
            )

        trusted = manifest["skills"].get(normalized_name)
        if trusted is None:
            return SkillTrustStatus(
                skill_name=normalized_name,
                trust_status=TRUST_STATUS_QUARANTINED_ADDED,
                trust_reason_code=TRUST_REASON_SKILL_ADDED,
                trust_blocked=True,
                changed_files=tuple(item.relative_path for item in current.fingerprints),
                manifest_generation=generation,
                last_verified_at=_now_iso(),
            )

        try:
            trusted_files = self._trusted_file_map(trusted)
        except ValueError as exc:
            return self._manifest_error_status(
                normalized_name,
                self._manifest_error_reason(exc),
            )
        current_files = {
            item.relative_path: item.as_manifest_entry() for item in current.fingerprints
        }
        missing = set(trusted_files) - set(current_files)
        added = set(current_files) - set(trusted_files)
        modified = {
            path
            for path in trusted_files.keys() & current_files.keys()
            if trusted_files[path] != current_files[path]
        }
        changed = tuple(sorted(missing | added | modified))
        if not changed:
            return SkillTrustStatus(
                skill_name=normalized_name,
                trust_status=TRUST_STATUS_TRUSTED,
                trust_reason_code=None,
                trust_blocked=False,
                changed_files=(),
                manifest_generation=generation,
                last_verified_at=_now_iso(),
            )
        if missing:
            status = TRUST_STATUS_QUARANTINED_DELETED
            reason = TRUST_REASON_SKILL_DELETED
        elif added:
            status = TRUST_STATUS_QUARANTINED_ADDED
            reason = TRUST_REASON_SKILL_ADDED
        else:
            status = TRUST_STATUS_QUARANTINED_MODIFIED
            reason = TRUST_REASON_SKILL_MODIFIED
        return SkillTrustStatus(
            skill_name=normalized_name,
            trust_status=status,
            trust_reason_code=reason,
            trust_blocked=True,
            changed_files=changed,
            manifest_generation=generation,
            last_verified_at=_now_iso(),
        )

    def ensure_skill_trusted(self, skill_name: str) -> None:
        """Raise only at use time when a local skill is trust-blocked."""

        status = self.status_for_skill(skill_name)
        if not status.trust_blocked:
            return
        raise SkillTrustBlockedError(
            skill_name=skill_name,
            reason_code=status.trust_reason_code or "trust_blocked",
            trust_status=status.trust_status,
            changed_files=status.changed_files,
        )

    def verify_skill_content(
        self,
        skill_name: str,
        *,
        skill_content: str,
        supporting_files: dict[str, str] | None,
    ) -> None:
        """Verify the exact in-memory skill files match the trusted manifest."""

        self.ensure_skill_trusted(skill_name)
        try:
            normalized_name = self._normalize_skill_name(skill_name)
            manifest = self._load_valid_manifest()
            trusted = manifest["skills"].get(normalized_name)
            if trusted is None:
                raise SkillTrustBlockedError(
                    skill_name=normalized_name,
                    reason_code=TRUST_REASON_SKILL_ADDED,
                    trust_status=TRUST_STATUS_QUARANTINED_ADDED,
                    changed_files=(_SKILL_FILENAME,),
                )
            trusted_files = self._trusted_file_map(trusted)
        except SkillTrustBlockedError:
            raise
        except SkillTrustMarkerUnavailable as exc:
            raise SkillTrustBlockedError(
                skill_name=skill_name,
                reason_code=TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE,
                trust_status=TRUST_STATUS_QUARANTINED_MANIFEST_ERROR,
            ) from exc
        except ValueError as exc:
            raise SkillTrustBlockedError(
                skill_name=skill_name,
                reason_code=self._manifest_error_reason(exc),
                trust_status=TRUST_STATUS_QUARANTINED_MANIFEST_ERROR,
            ) from exc

        current_files = self._fingerprint_in_memory_files(
            skill_content=skill_content,
            supporting_files=supporting_files,
        )
        missing = set(trusted_files) - set(current_files)
        added = set(current_files) - set(trusted_files)
        modified = {
            path
            for path in trusted_files.keys() & current_files.keys()
            if trusted_files[path] != current_files[path]
        }
        changed = tuple(sorted(missing | added | modified))
        if not changed:
            return
        if missing:
            status = TRUST_STATUS_QUARANTINED_DELETED
            reason = TRUST_REASON_SKILL_DELETED
        elif added:
            status = TRUST_STATUS_QUARANTINED_ADDED
            reason = TRUST_REASON_SKILL_ADDED
        else:
            status = TRUST_STATUS_QUARANTINED_MODIFIED
            reason = TRUST_REASON_SKILL_MODIFIED
        raise SkillTrustBlockedError(
            skill_name=normalized_name,
            reason_code=reason,
            trust_status=status,
            changed_files=changed,
        )

    def capture_review(self, skill_name: str) -> dict[str, Any]:
        """Capture a JSON-safe review snapshot for the current skill files."""

        normalized_name = self._normalize_skill_name(skill_name)
        status = self.status_for_skill(normalized_name)
        current = self._scan_skill(normalized_name)
        review_id = secrets.token_hex(16)
        review = {
            "review_id": review_id,
            "skill_name": normalized_name,
            "manifest_generation": status.manifest_generation,
            "current_digest": self._fingerprints_digest(current),
            "current_files": dict(current.text_files),
            "current_fingerprints": [item.as_manifest_entry() for item in current.fingerprints],
            "changed_files": list(status.changed_files),
            "captured_at": _now_iso(),
        }
        self._reviews[review_id] = {
            "review_id": review_id,
            "skill_name": normalized_name,
            "manifest_generation": status.manifest_generation,
            "current_digest": review["current_digest"],
            "changed_files": list(status.changed_files),
        }
        return dict(review)

    def discard_review(self, review_id: str) -> None:
        """Forget a captured review without changing trust state."""

        self._reviews.pop(review_id, None)

    def trust_reviewed_snapshot(self, review_id: str) -> None:
        """Approve a captured review if live files still match that review."""

        review = self._reviews[review_id]
        try:
            skill_name = self._normalize_skill_name(str(review["skill_name"]))
        except ValueError as exc:
            self._reviews.pop(review_id, None)
            raise ValueError(TRUST_REASON_UNSUPPORTED_PATH) from exc

        try:
            manifest = self._load_valid_manifest()
        except SkillTrustMarkerUnavailable as exc:
            self._reviews.pop(review_id, None)
            raise ValueError("snapshot_mismatch") from exc
        review_generation = review.get("manifest_generation")
        if not isinstance(review_generation, int):
            self._reviews.pop(review_id, None)
            raise ValueError("snapshot_mismatch")
        if int(manifest["generation"]) != review_generation:
            self._reviews.pop(review_id, None)
            raise ValueError("snapshot_mismatch")
        current = self._scan_skill(skill_name)
        if self._fingerprints_digest(current) != review["current_digest"]:
            self._reviews.pop(review_id, None)
            raise ValueError("snapshot_mismatch")
        self._reviews.pop(review_id, None)
        self.trust_current_skill(skill_name, audit_event="trust_approved", snapshot=current)

    def trust_current_skill(
        self,
        skill_name: str,
        *,
        audit_event: str = "trust_chatbook_mutation",
        snapshot: SkillDirectorySnapshot | None = None,
    ) -> None:
        """Trust the live files for one skill after an explicit approval path."""

        normalized_name = self._normalize_skill_name(skill_name)
        keys = self._require_keys()
        manifest = self._load_valid_manifest()
        generation = int(manifest["generation"]) + 1
        current = snapshot or self._scan_skill(normalized_name)
        if current.unsupported_paths:
            raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
        if not current.fingerprints:
            raise ValueError(TRUST_REASON_SKILL_DELETED)

        snapshot_id = self._snapshot_id(normalized_name, generation)
        self.trust_store.save_snapshot(
            snapshot_id,
            {"files": dict(current.text_files)},
            keys,
            generation=generation,
        )
        manifest["generation"] = generation
        manifest.setdefault("skills", {})[normalized_name] = self._manifest_skill_entry(
            snapshot=current,
            snapshot_id=snapshot_id,
            snapshot_generation=generation,
        )
        manifest.setdefault("audit", []).append(
            {
                "event": audit_event,
                "at": _now_iso(),
                "skill_name": normalized_name,
            }
        )
        self.trust_store.save_manifest(manifest, keys, salt=self._require_salt())

    def _iter_skill_dirs(self) -> list[tuple[str, Path]]:
        if not self.skills_dir.exists():
            return []
        skill_dirs: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for child in sorted(self.skills_dir.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
            if child.is_dir():
                normalized_name = self._normalize_skill_name(child.name)
                if normalized_name in seen:
                    raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
                seen.add(normalized_name)
                skill_dirs.append((normalized_name, child))
        return skill_dirs

    def _known_and_current_skill_names(self, manifest: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for skill_name in manifest["skills"]:
            names.add(self._normalize_skill_name(skill_name))
        if not self.skills_dir.exists():
            return names
        current_names: set[str] = set()
        for child in sorted(self.skills_dir.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
            if child.is_dir():
                normalized_name = self._normalize_skill_name(child.name)
                if normalized_name in current_names:
                    raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
                current_names.add(normalized_name)
                names.add(normalized_name)
        return names

    def _load_valid_manifest(self) -> dict[str, Any]:
        manifest = self.trust_store.load_manifest(self._require_keys())
        if manifest.get("version") != 1:
            raise ValueError("manifest schema invalid")
        if not isinstance(manifest.get("generation"), int):
            raise ValueError("manifest schema invalid")
        if not isinstance(manifest.get("skills"), dict):
            raise ValueError("manifest schema invalid")
        if not isinstance(manifest.get("audit"), list):
            raise ValueError("manifest schema invalid")
        return manifest

    def _require_keys(self) -> SkillTrustKeys:
        if self._keys is None:
            raise SkillTrustBlockedError(
                skill_name="<all>",
                reason_code=TRUST_REASON_TRUST_LOCKED,
                trust_status=TRUST_STATUS_LOCKED,
            )
        return self._keys

    def _require_salt(self) -> bytes:
        if self._salt is None:
            raise ValueError("skill trust salt missing")
        return self._salt

    def _scan_skill(self, skill_name: str) -> SkillDirectorySnapshot:
        normalized_name = self._normalize_skill_name(skill_name)
        try:
            skill_dir = self._skill_dir_for_normalized_name(normalized_name)
        except ValueError:
            return SkillDirectorySnapshot(
                skill_name=normalized_name,
                fingerprints=(),
                text_files={},
                unsupported_paths=(normalized_name,),
            )
        if skill_dir.is_symlink():
            return SkillDirectorySnapshot(
                skill_name=normalized_name,
                fingerprints=(),
                text_files={},
                unsupported_paths=(normalized_name,),
            )
        if not skill_dir.exists():
            return SkillDirectorySnapshot(
                skill_name=normalized_name,
                fingerprints=(),
                text_files={},
            )
        if not skill_dir.is_dir():
            return SkillDirectorySnapshot(
                skill_name=normalized_name,
                fingerprints=(),
                text_files={},
                unsupported_paths=(normalized_name,),
            )
        return scan_skill_directory(normalized_name, skill_dir)

    def _normalize_skill_name(self, skill_name: str) -> str:
        try:
            return _normalize_api_skill_name(skill_name)
        except (AttributeError, ValueError) as exc:
            raise ValueError(TRUST_REASON_UNSUPPORTED_PATH) from exc

    def _skill_dir_for_normalized_name(self, normalized_name: str) -> Path:
        direct = self.skills_dir / normalized_name
        if direct.exists() or direct.is_symlink() or not self.skills_dir.exists():
            return direct
        matches: list[Path] = []
        for child in sorted(self.skills_dir.iterdir(), key=lambda item: item.name):
            if not (child.is_dir() or child.is_symlink()):
                continue
            try:
                child_name = self._normalize_skill_name(child.name)
            except ValueError:
                continue
            if child_name == normalized_name:
                matches.append(child)
        if len(matches) > 1:
            raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)
        if matches:
            return matches[0]
        return direct

    def _manifest_missing_status(self, skill_name: str) -> SkillTrustStatus:
        try:
            marker = self.trust_store.marker_store.load_marker()
        except SkillTrustMarkerUnavailable:
            return self._manifest_error_status(
                skill_name,
                TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE,
            )
        except Exception:
            return self._manifest_error_status(skill_name, TRUST_REASON_MANIFEST_INVALID)
        if marker:
            return self._manifest_error_status(skill_name, TRUST_REASON_MANIFEST_INVALID)
        return SkillTrustStatus(
            skill_name=skill_name,
            trust_status=TRUST_STATUS_UNINITIALIZED,
            trust_reason_code=TRUST_REASON_TRUST_UNINITIALIZED,
            trust_blocked=True,
            changed_files=(),
            manifest_generation=None,
            last_verified_at=None,
        )

    def _locked_status(self, skill_name: str) -> SkillTrustStatus:
        return SkillTrustStatus(
            skill_name=skill_name,
            trust_status=TRUST_STATUS_LOCKED,
            trust_reason_code=TRUST_REASON_TRUST_LOCKED,
            trust_blocked=True,
            changed_files=(),
            manifest_generation=None,
            last_verified_at=_now_iso(),
        )

    def _manifest_error_status(self, skill_name: str, reason_code: str) -> SkillTrustStatus:
        return SkillTrustStatus(
            skill_name=skill_name,
            trust_status=TRUST_STATUS_QUARANTINED_MANIFEST_ERROR,
            trust_reason_code=reason_code,
            trust_blocked=True,
            changed_files=(),
            manifest_generation=None,
            last_verified_at=_now_iso(),
        )

    def _unsupported_name_status(self, skill_name: Any) -> SkillTrustStatus:
        return SkillTrustStatus(
            skill_name=str(skill_name),
            trust_status=TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH,
            trust_reason_code=TRUST_REASON_UNSUPPORTED_PATH,
            trust_blocked=True,
            changed_files=(),
            manifest_generation=None,
            last_verified_at=_now_iso(),
        )

    def _manifest_error_reason(self, exc: ValueError) -> str:
        message = str(exc)
        if "marker" in message:
            return TRUST_REASON_ROLLBACK_MARKER_UNAVAILABLE
        return TRUST_REASON_MANIFEST_INVALID

    def _fingerprints_digest(self, snapshot: SkillDirectorySnapshot) -> str:
        return sha256_hex(
            canonical_json([item.as_manifest_entry() for item in snapshot.fingerprints])
        )

    def _trusted_file_map(self, trusted: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(trusted, dict):
            raise ValueError("manifest schema invalid")
        files = trusted.get("files")
        if not isinstance(files, list):
            raise ValueError("manifest schema invalid")
        result: dict[str, dict[str, Any]] = {}
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
                raise ValueError("manifest schema invalid")
            result[str(item["relative_path"])] = dict(item)
        return result

    def _fingerprint_in_memory_files(
        self,
        *,
        skill_content: str,
        supporting_files: dict[str, str] | None,
    ) -> dict[str, dict[str, Any]]:
        files = {
            _SKILL_FILENAME: self._content_manifest_entry(
                relative_path=_SKILL_FILENAME,
                content=skill_content,
            )
        }
        for relative_path, content in sorted((supporting_files or {}).items()):
            files[str(relative_path)] = self._content_manifest_entry(
                relative_path=str(relative_path),
                content=str(content),
            )
        return files

    def _content_manifest_entry(self, *, relative_path: str, content: str) -> dict[str, Any]:
        raw = content.encode("utf-8")
        return {
            "relative_path": relative_path,
            "file_type": "skill" if relative_path == _SKILL_FILENAME else "supporting_text",
            "byte_length": len(raw),
            "sha256": sha256_hex(raw),
        }

    def _manifest_skill_entry(
        self,
        *,
        snapshot: SkillDirectorySnapshot,
        snapshot_id: str,
        snapshot_generation: int,
    ) -> dict[str, Any]:
        return {
            "files": [item.as_manifest_entry() for item in snapshot.fingerprints],
            "snapshot_id": snapshot_id,
            "snapshot_generation": snapshot_generation,
            "trusted_at": _now_iso(),
        }

    def _snapshot_id(self, skill_name: str, generation: int) -> str:
        return f"{skill_name}-{generation}"
