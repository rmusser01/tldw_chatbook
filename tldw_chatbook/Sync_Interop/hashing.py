"""Canonical, versioned payload hashing shared by Sync v2 clients.

The server stores ``object_hash = payload_hash`` verbatim, so single-client push is
safe with any deterministic hash. This canonical form exists for cross-client parity
(chat.message dedupe, restore/preview local-inventory comparison): all chatbook
clients must hash identical payloads to identical digests.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

HASH_VERSION = 1


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON encoding of ``payload``.

    Canonical form: UTF-8 JSON with sorted keys and compact separators.
    """
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
