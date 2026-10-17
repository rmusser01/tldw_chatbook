# tldw_chatbook/Web_Server/webauthn_store.py
"""Disk-backed persistence for WebAuthn passkey credentials (ADR-235).

Framework-neutral on purpose and a byte-for-byte sibling of
``Web_Server/session_store.py`` in discipline: atomic tmp+rename writes,
0600 permissions, and a fail-safe load that yields an empty mapping for a
missing, corrupt, or non-object file.  Records carry public material only --
the base64 COSE public key and the hex credential id; nothing private is ever
written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

_FIELDS = (
    "credential_id_hex",
    "public_key_b64",
    "rp_id",
    "sign_count",
    "label",
    "aaguid",
    "created_at",
    "last_used_at",
)


def _is_well_typed_record(value: object) -> bool:
    """Whether one decoded record carries the scalar types it claims.

    A wrong-typed ``sign_count`` (e.g. ``"soon"`` from a hand-edited file)
    must be skipped like a malformed one, mirroring ``session_store``'s
    discipline: letting it through would surface later as a ``TypeError``
    inside the service's numeric sign-count comparison (#3130).
    """

    if not isinstance(value, dict):
        return False
    sign_count = value.get("sign_count")
    return isinstance(sign_count, int) and not isinstance(sign_count, bool)


@dataclass(frozen=True)
class StoredCredential:
    credential_id_hex: str
    public_key_b64: str  # base64 of the COSE CBOR bytes
    rp_id: str
    sign_count: int
    label: str
    aaguid: str
    created_at: float
    last_used_at: float


class WebAuthnCredentialStore:
    """Single-owner JSON credential store with atomic, private writes."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, StoredCredential]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            logger.warning(
                "discarding unreadable webauthn credential store: %s", error
            )
            return {}
        if not isinstance(raw, dict):
            logger.warning(
                "discarding non-object webauthn credential store payload: %s",
                type(raw).__name__,
            )
            return {}
        records: dict[str, StoredCredential] = {}
        for key, value in raw.items():
            if not _is_well_typed_record(value):
                continue  # skip wrong-typed entries, keep the rest
            try:
                records[key] = StoredCredential(
                    **{name: value[name] for name in _FIELDS}
                )
            except (KeyError, TypeError):
                continue  # skip malformed entries, keep the rest
        return records

    def save(self, creds: Mapping[str, StoredCredential]) -> None:
        payload = {key: asdict(record) for key, record in creds.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=".webauthn_credentials-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self._path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
