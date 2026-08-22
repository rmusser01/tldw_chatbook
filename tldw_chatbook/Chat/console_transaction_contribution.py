"""Caller-owned transaction extension point for Console sidecar persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Protocol


class ConsoleTransactionContribution(Protocol):
    """Write one sidecar through an existing atomic Console transaction."""

    def write(
        self,
        *,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Write through the caller-owned cursor without committing."""
