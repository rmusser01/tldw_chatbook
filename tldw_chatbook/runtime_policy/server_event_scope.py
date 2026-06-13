"""Helpers for durable server event state scope identity."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any


def event_principal_id_from_active_context(context: Any) -> str | None:
    """Return a stable non-secret event principal scope from an active server context."""

    auth_token = getattr(context, "auth_token", None)
    if not auth_token:
        return None
    token = str(auth_token)
    jwt_subject = _jwt_subject(token)
    if jwt_subject:
        return f"jwt-sub:{jwt_subject}"

    credential_source = str(getattr(context, "credential_source", None) or "unknown")
    fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    return f"credential-fingerprint:{credential_source}:{fingerprint}"


def _jwt_subject(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    try:
        payload_bytes = base64.urlsafe_b64decode(_pad_base64(parts[1]))
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (binascii.Error, TypeError, ValueError, json.JSONDecodeError):
        return None
    subject = payload.get("sub") or payload.get("user_id") or payload.get("uid")
    if subject is None:
        return None
    subject_text = str(subject).strip()
    return subject_text or None


def _pad_base64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return f"{value}{padding}".encode("ascii")
