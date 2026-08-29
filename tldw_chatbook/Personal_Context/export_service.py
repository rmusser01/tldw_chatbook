"""Explicit, private publication of Personal Context export artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    RecordState,
    ScopeKind,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_bytes,
    open_private_binary,
)

if TYPE_CHECKING:
    from .service import PersonalContextService


_RECOVERY_FORMAT = "tldw-personal-context-recovery-v1"
_RECOVERY_AAD = b"tldw-chatbook:personal-context:recovery:v1\x00"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


class PersonalContextExportError(ValueError):
    """A stable path- and content-free export failure."""


@dataclass(frozen=True, slots=True)
class ExportRequest:
    destination: str | os.PathLike[str] = field(repr=False)
    confirm_plaintext: bool
    scope_ids: tuple[str, ...] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class RecoveryExportRequest:
    destination: str | os.PathLike[str] = field(repr=False)
    passphrase: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    manifest: ProfileManifest = field(repr=False)
    scopes: tuple[ProfileScope, ...] = field(repr=False)
    records: tuple[ProfileRecord, ...] = field(repr=False)
    proposals: tuple[ProfileProposal, ...] = field(repr=False)


def _destination(
    value: str | os.PathLike[str],
) -> tuple[Path, PrivateFileWritePrecondition]:
    try:
        destination = Path(value)
        if not destination.is_absolute():
            raise PersonalContextExportError("Export destination must be absolute.")
        if destination.name in {"", ".", ".."} or not destination.parent.is_dir():
            raise PersonalContextExportError("Export destination is invalid.")
        if destination.parent.resolve(strict=True) != destination.parent:
            raise PersonalContextExportError("Export destination is invalid.")
        try:
            existing = os.lstat(destination)
        except FileNotFoundError:
            precondition = PrivateFileWritePrecondition.missing()
        else:
            if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                raise PersonalContextExportError("Export destination is invalid.")
            precondition = PrivateFileWritePrecondition(
                (existing.st_dev, existing.st_ino)
            )
        return destination, precondition
    except PersonalContextExportError:
        raise
    except (OSError, TypeError, ValueError):
        raise PersonalContextExportError("Export destination is invalid.") from None


def _snapshot(
    service: "PersonalContextService",
    scope_ids: tuple[str, ...] | None,
    *,
    include_all_current_records: bool = False,
) -> RecoverySnapshot:
    manifest, available_scopes, current_records, current_proposals = (
        service.snapshot_for_export()
    )
    selected_ids = (
        tuple(scope.scope_id for scope in available_scopes)
        if scope_ids is None
        else tuple(scope_ids)
    )
    selected = set(selected_ids)
    if not selected or not selected.issubset(
        {scope.scope_id for scope in available_scopes}
    ):
        raise PersonalContextExportError("Export scope selection is invalid.")
    scopes = tuple(scope for scope in available_scopes if scope.scope_id in selected)
    now = service.clock()
    records = tuple(
        record
        for record in current_records
        if record.scope_id in selected
        and (
            include_all_current_records
            or (
                record.state is not RecordState.DELETED
                and (record.expires_at is None or record.expires_at > now)
            )
        )
    )
    proposals = tuple(
        proposal for proposal in current_proposals if proposal.scope_id in selected
    )
    return RecoverySnapshot(manifest, scopes, records, proposals)


def _snapshot_dict(snapshot: RecoverySnapshot, *, format_name: str) -> dict[str, Any]:
    return {
        "format": format_name,
        "manifest": snapshot.manifest.model_dump(mode="json"),
        "scopes": [scope.model_dump(mode="json") for scope in snapshot.scopes],
        "records": [record.model_dump(mode="json") for record in snapshot.records],
        "proposals": [
            proposal.model_dump(mode="json") for proposal in snapshot.proposals
        ],
    }


def export_plaintext(service: "PersonalContextService", request: ExportRequest) -> Path:
    if not isinstance(request, ExportRequest):
        raise PersonalContextExportError("Plaintext export request is invalid.")
    if request.confirm_plaintext is not True:
        raise PersonalContextExportError("Explicit plaintext confirmation is required.")
    destination, precondition = _destination(request.destination)
    snapshot = _snapshot(service, request.scope_ids)
    payload = (
        json.dumps(
            _snapshot_dict(snapshot, format_name="tldw-personal-context-plaintext-v1"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        atomic_private_write_bytes(
            destination, payload, target_precondition=precondition
        )
    except Exception:
        raise PersonalContextExportError(
            "Plaintext export could not be written."
        ) from None
    return destination


def _recovery_key(passphrase: str, salt: bytes) -> bytes:
    if not isinstance(passphrase, str) or not passphrase or len(passphrase) > 4096:
        raise PersonalContextExportError("A bounded non-empty passphrase is required.")
    domain_salt = hashlib.sha256(_RECOVERY_AAD + salt).digest()
    try:
        return hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=domain_salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=32,
        )
    except (UnicodeError, ValueError):
        raise PersonalContextExportError("Recovery key derivation failed.") from None


def export_recovery(
    service: "PersonalContextService", request: RecoveryExportRequest
) -> Path:
    if not isinstance(request, RecoveryExportRequest):
        raise PersonalContextExportError("Recovery export request is invalid.")
    destination, precondition = _destination(request.destination)
    snapshot = _snapshot(service, None, include_all_current_records=True)
    plaintext = canonical_json_bytes(
        _snapshot_dict(snapshot, format_name="tldw-personal-context-snapshot-v1")
    )
    salt = os.urandom(32)
    nonce = os.urandom(12)
    ciphertext = AESGCM(_recovery_key(request.passphrase, salt)).encrypt(
        nonce, plaintext, _RECOVERY_AAD
    )
    envelope = canonical_json_bytes(
        {
            "format": _RECOVERY_FORMAT,
            "version": 1,
            "kdf": {"name": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P},
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
    )
    try:
        atomic_private_write_bytes(
            destination, envelope, target_precondition=precondition
        )
    except Exception:
        raise PersonalContextExportError(
            "Recovery export could not be written."
        ) from None
    return destination


def _decode_snapshot(plaintext: bytes) -> RecoverySnapshot:
    try:
        body = json.loads(plaintext)
        if set(body) != {"format", "manifest", "scopes", "records", "proposals"}:
            raise ValueError
        if body["format"] != "tldw-personal-context-snapshot-v1":
            raise ValueError
        manifest = ProfileManifest.model_validate(body["manifest"])
        scopes = tuple(ProfileScope.model_validate(value) for value in body["scopes"])
        records = tuple(
            ProfileRecord.model_validate(value) for value in body["records"]
        )
        proposals = tuple(
            ProfileProposal.model_validate(value) for value in body["proposals"]
        )
        if len({scope.scope_id for scope in scopes}) != len(scopes):
            raise ValueError
        if sum(scope.kind is ScopeKind.GLOBAL for scope in scopes) != 1:
            raise ValueError
        if any(scope.profile_id != manifest.profile_id for scope in scopes):
            raise ValueError
        scope_ids = {scope.scope_id for scope in scopes}
        if len({record.record_id for record in records}) != len(records):
            raise ValueError
        if len({proposal.proposal_id for proposal in proposals}) != len(proposals):
            raise ValueError
        if any(
            record.profile_id != manifest.profile_id or record.scope_id not in scope_ids
            for record in records
        ):
            raise ValueError
        if any(
            proposal.profile_id != manifest.profile_id
            or proposal.scope_id not in scope_ids
            for proposal in proposals
        ):
            raise ValueError
        current_record_versions = {record.version_id for record in records}
        if any(
            record.parent_version_id in current_record_versions
            for record in records
            if record.parent_version_id is not None
        ):
            raise ValueError

        proposed_records = tuple(
            proposal.proposed_record
            for proposal in proposals
            if proposal.proposed_record is not None
        )
        version_ids = (
            (manifest.current_version_id,)
            + tuple(scope.version_id for scope in scopes)
            + tuple(record.version_id for record in records)
            + tuple(record.version_id for record in proposed_records)
        )
        if len(set(version_ids)) != len(version_ids):
            raise ValueError
        if any(
            record.parent_version_id == record.version_id for record in proposed_records
        ):
            raise ValueError

        semantic_keys = [
            (record.scope_id, record.kind, record.semantic_key)
            for record in records
            if record.state is RecordState.ACTIVE
            and record.semantic_key is not None
            and (record.expires_at is None or record.expires_at > manifest.updated_at)
        ]
        if len(set(semantic_keys)) != len(semantic_keys):
            raise ValueError

        return RecoverySnapshot(manifest, scopes, records, proposals)
    except (KeyError, TypeError, ValueError):
        raise PersonalContextExportError("Recovery snapshot is invalid.") from None


def load_recovery_export(
    path: str | os.PathLike[str], passphrase: str
) -> RecoverySnapshot:
    try:
        with open_private_binary(path) as opened:
            encoded = opened.stream.read()
        envelope = json.loads(encoded)
        if set(envelope) != {"format", "version", "kdf", "salt", "nonce", "ciphertext"}:
            raise ValueError
        if (
            envelope["format"] != _RECOVERY_FORMAT
            or envelope["version"] != 1
            or envelope["kdf"]
            != {"name": "scrypt", "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P}
        ):
            raise ValueError
        salt = base64.b64decode(envelope["salt"], validate=True)
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
        if len(salt) != 32 or len(nonce) != 12:
            raise ValueError
        plaintext = AESGCM(_recovery_key(passphrase, salt)).decrypt(
            nonce, ciphertext, _RECOVERY_AAD
        )
        return _decode_snapshot(plaintext)
    except PersonalContextExportError:
        raise
    except (InvalidTag, KeyError, OSError, TypeError, ValueError):
        raise PersonalContextExportError(
            "Recovery export could not be unlocked."
        ) from None
