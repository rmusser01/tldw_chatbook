"""Bounded JSONL log of MCP tool executions (Hub tests now; chat/agents later).

Append-only with two-generation size rotation (crash-safe: a torn final line
is skipped on read). Arguments are redacted before they ever reach disk.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    atomic_private_write_bytes,
    open_private_binary,
    open_private_text_append,
    secure_private_directory,
)

RESULT_EXCERPT_LIMIT = 500


@dataclass(frozen=True)
class ExecutionRecord:
    ts: str
    server_key: str
    tool_name: str
    initiator: str
    decision: str
    ok: bool
    duration_ms: int
    error: str | None = None
    arguments: dict[str, Any] | None = None
    result_excerpt: str | None = None


def build_record(
    *,
    server_key: str,
    tool_name: str,
    initiator: str,
    ok: bool,
    duration_ms: int,
    error: str | None = None,
    arguments: dict[str, Any] | None = None,
    result_excerpt: str | None = None,
    decision: str = "allowed",
    capture_args: bool = True,
) -> ExecutionRecord:
    """Build a redacted, timestamped execution record.

    Args:
        server_key: Hub server key ("local:<id>" / "builtin:tldw_chatbook").
        tool_name: Tool invoked.
        initiator: "test" (Hub-initiated) — "chat"/"agent" in later phases.
        ok: Whether the execution succeeded.
        duration_ms: Wall-clock duration.
        error: Error summary on failure (caller-truncated).
        arguments: Call arguments; redacted here, dropped when capture_args
            is False.
        result_excerpt: Caller-provided excerpt; truncated to 500 chars.
        decision: Permission decision ("allowed" for user-initiated tests).
        capture_args: The [mcp] log_tool_arguments setting value.

    Returns:
        A frozen ExecutionRecord safe to persist.
    """
    kept_arguments: dict[str, Any] | None = None
    if capture_args and isinstance(arguments, dict):
        kept_arguments = redact_mapping(arguments)
    excerpt = None
    if result_excerpt is not None:
        excerpt = str(result_excerpt)[:RESULT_EXCERPT_LIMIT]
    return ExecutionRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        server_key=server_key,
        tool_name=tool_name,
        initiator=initiator,
        decision=decision,
        ok=ok,
        duration_ms=int(duration_ms),
        error=(str(error)[:300] if error else None),
        arguments=kept_arguments,
        result_excerpt=excerpt,
    )


class MCPExecutionLog:
    """Two-generation bounded JSONL store for ExecutionRecords."""

    def __init__(self, path: Path, *, max_records_per_file: int = 500) -> None:
        self.path = Path(path)
        if max_records_per_file < 1:
            raise ValueError("max_records_per_file must be positive")
        self.max_records_per_file = max_records_per_file
        self._lock = threading.RLock()

    def append(self, record: ExecutionRecord) -> None:
        """Append one record, rotating generations at the size cap.

        Args:
            record: The execution record to persist. Dict-shaped
                ``arguments`` are defensively re-redacted before the
                record reaches disk.

        Raises:
            OSError: If the log file or its parent directory cannot be
                written (callers treat recording as best-effort).
        """
        payload = asdict(record)
        if isinstance(payload.get("arguments"), dict):
            payload["arguments"] = redact_mapping(payload["arguments"])
        encoded_line = (json.dumps(payload, default=str) + "\n").encode("utf-8")
        rotated = self.path.with_name(self.path.name + ".1")
        with self._lock:
            self._secure_parent()
            self._verify_existing(rotated)
            active_payload = self._read_bytes(self.path)
            line_count = (
                len(active_payload.splitlines()) if active_payload is not None else 0
            )
            if line_count >= self.max_records_per_file:
                atomic_private_write_bytes(
                    rotated,
                    active_payload or b"",
                    application_owned_directory=self.path.parent,
                )
                atomic_private_write_bytes(
                    self.path,
                    encoded_line,
                    application_owned_directory=self.path.parent,
                )
                return
            with open_private_text_append(
                self.path,
                application_owned_directory=self.path.parent,
            ) as handle:
                handle.write(encoded_line.decode("utf-8"))

    def read_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return recent records, newest first, across both generations.

        Args:
            limit: Maximum number of records to return.

        Returns:
            Up to ``limit`` record dicts, newest first. Torn or corrupt
            JSONL lines are skipped rather than raising.
        """
        if limit <= 0:
            return []
        rows: list[dict[str, Any]] = []
        rotated = self.path.with_name(self.path.name + ".1")
        with self._lock:
            try:
                self._secure_parent()
            except PrivatePathError as exc:
                logger.warning(
                    "MCP execution log read disabled (status={}).",
                    exc.result.status.value,
                )
                return []
            for source in (rotated, self.path):  # oldest generation first
                try:
                    raw = self._read_bytes(source)
                except PrivatePathError as exc:
                    logger.warning(
                        "MCP execution-log generation skipped "
                        "(status={}, generation={}).",
                        exc.result.status.value,
                        "rotated" if source == rotated else "active",
                    )
                    continue
                if raw is None:
                    continue
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        decoded = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn/corrupt line — skip, never crash
                    if isinstance(decoded, dict):
                        rows.append(decoded)
        rows.reverse()
        return rows[:limit]

    def _secure_parent(self) -> None:
        secure_private_directory(
            self.path.parent,
            create=True,
            application_owned=True,
        )

    @staticmethod
    def _read_bytes(path: Path) -> bytes | None:
        try:
            with open_private_binary(path) as pinned:
                return pinned.stream.read()
        except FileNotFoundError:
            return None

    @staticmethod
    def _verify_existing(path: Path) -> None:
        try:
            with open_private_binary(path):
                pass
        except FileNotFoundError:
            pass
