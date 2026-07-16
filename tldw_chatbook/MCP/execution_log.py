"""Bounded JSONL log of MCP tool executions (Hub tests now; chat/agents later).

Append-only with two-generation size rotation (crash-safe: a torn final line
is skipped on read). Arguments are redacted before they ever reach disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_chatbook.MCP.redaction import redact_mapping

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


def build_record(*, server_key: str, tool_name: str, initiator: str, ok: bool,
                 duration_ms: int, error: str | None = None,
                 arguments: dict[str, Any] | None = None,
                 result_excerpt: str | None = None,
                 decision: str = "allowed",
                 capture_args: bool = True) -> ExecutionRecord:
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
        server_key=server_key, tool_name=tool_name, initiator=initiator,
        decision=decision, ok=ok, duration_ms=int(duration_ms),
        error=(str(error)[:300] if error else None),
        arguments=kept_arguments, result_excerpt=excerpt,
    )


class MCPExecutionLog:
    """Two-generation bounded JSONL store for ExecutionRecords."""

    def __init__(self, path: Path, *, max_records_per_file: int = 500) -> None:
        self.path = Path(path)
        self.max_records_per_file = max_records_per_file

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._count_lines(self.path) >= self.max_records_per_file:
            rotated = self.path.with_name(self.path.name + ".1")
            self.path.replace(rotated)
        payload = asdict(record)
        if isinstance(payload.get("arguments"), dict):
            payload["arguments"] = redact_mapping(payload["arguments"])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def read_recent(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return recent records, newest first, across both generations.

        Args:
            limit: Maximum number of records to return.

        Returns:
            Up to ``limit`` record dicts, newest first. Torn or corrupt
            JSONL lines are skipped rather than raising.
        """
        rows: list[dict[str, Any]] = []
        rotated = self.path.with_name(self.path.name + ".1")
        for source in (rotated, self.path):  # oldest generation first
            if not source.exists():
                continue
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn/corrupt line — skip, never crash
        rows.reverse()
        return rows[:limit]

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
