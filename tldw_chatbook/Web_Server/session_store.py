# tldw_chatbook/Web_Server/session_store.py
"""Disk-backed persistence for served web sessions (ADR-233).

Framework-neutral on purpose: this module must not import
``tldw_chatbook.Canvas.web_auth`` — the manager imports the store, never the
reverse. Records carry cookie *digests* (hex) only; raw cookie values are
never written to disk.
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
    "digest_hex",
    "session_id",
    "csrf_token",
    "client_ip",
    "device_label",
    "created_at",
    "last_seen_at",
    "absolute_expires_at",
)


@dataclass(frozen=True)
class StoredSession:
    digest_hex: str
    session_id: str
    csrf_token: str
    client_ip: str
    device_label: str
    created_at: float
    last_seen_at: float
    absolute_expires_at: float


class WebSessionStore:
    """Single-owner JSON session store with atomic, private writes."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, StoredSession]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            logger.warning("discarding unreadable web session store: %s", error)
            return {}
        if not isinstance(raw, dict):
            logger.warning(
                "discarding non-object web session store payload: %s",
                type(raw).__name__,
            )
            return {}
        records: dict[str, StoredSession] = {}
        for key, value in raw.items():
            try:
                records[key] = StoredSession(**{name: value[name] for name in _FIELDS})
            except (KeyError, TypeError):
                continue  # skip malformed entries, keep the rest
        return records

    def save(self, sessions: Mapping[str, StoredSession]) -> None:
        payload = {key: asdict(record) for key, record in sessions.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=".web_sessions-", suffix=".tmp"
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
