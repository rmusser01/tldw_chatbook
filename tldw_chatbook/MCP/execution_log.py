"""Bounded, metadata-only JSONL log of MCP tool executions."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    atomic_private_write_bytes,
    open_private_binary,
    open_private_text_append,
    secure_private_directory,
)
from tldw_chatbook.Utils.persistent_diagnostics import safe_metadata_token

APPROVED_SESSION_DECISION = "approved-session"


@dataclass(frozen=True)
class ExecutionRecord:
    ts: str
    server_key: str
    tool_name: str
    initiator: str
    decision: str
    ok: bool
    status: str
    duration_ms: int
    error_category: str | None
    exception_type: str | None
    status_code: int | None
    argument_names: tuple[str, ...]
    unknown_argument_count: int
    result_type: str
    result_size: int


def _result_metadata(result: Any) -> tuple[str, int]:
    if result is None:
        return "none", 0
    result_type = safe_metadata_token(type(result).__name__)
    try:
        result_size = len(result)
    except (AttributeError, TypeError):
        result_size = 0
    return result_type, max(0, int(result_size))


def _registered_argument_metadata(
    arguments: dict[str, Any] | None,
    registered_argument_names: set[str] | tuple[str, ...] | list[str] | None,
) -> tuple[tuple[str, ...], int]:
    supplied = set(arguments) if isinstance(arguments, dict) else set()
    registered = (
        set(registered_argument_names)
        if registered_argument_names is not None
        else set()
    )
    kept = tuple(
        sorted(
            token
            for name in supplied & registered
            if (token := safe_metadata_token(name)) != "invalid"
        )
    )
    unknown_count = len(supplied - registered) + len(
        (supplied & registered) - set(kept)
    )
    return kept, unknown_count


def build_record(
    *,
    server_key: str,
    tool_name: str,
    initiator: str,
    ok: bool,
    duration_ms: int,
    status: str | None = None,
    error_category: str | None = None,
    exception_type: str | None = None,
    status_code: int | None = None,
    arguments: dict[str, Any] | None = None,
    registered_argument_names: set[str] | tuple[str, ...] | list[str] | None = None,
    result: Any = None,
    decision: str = "allowed",
) -> ExecutionRecord:
    """Build a timestamped record containing operational metadata only.

    Args:
        server_key: Hub server key ("local:<id>" / "builtin:tldw_chatbook").
        tool_name: Tool invoked.
        initiator: "test" (Hub-initiated) — "chat"/"agent" in later phases.
        ok: Whether the execution succeeded.
        duration_ms: Wall-clock duration.
        status: Bounded outcome category.
        error_category: Sanitized error category, never exception text.
        exception_type: Exception class name, never ``str(exception)``.
        status_code: Optional HTTP status.
        arguments: Call arguments used only for their keys.
        registered_argument_names: Schema-approved argument names.
        result: Result used only for its type and top-level size.
        decision: Permission decision (for example, "allowed", "approved",
            or "approved-session").

    Returns:
        A frozen metadata-only ExecutionRecord safe to persist.
    """
    argument_names, unknown_argument_count = _registered_argument_metadata(
        arguments, registered_argument_names
    )
    result_type, result_size = _result_metadata(result)
    return ExecutionRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        server_key=safe_metadata_token(server_key),
        tool_name=safe_metadata_token(tool_name),
        initiator=safe_metadata_token(initiator),
        decision=safe_metadata_token(decision),
        ok=bool(ok),
        status=safe_metadata_token(status or ("success" if ok else "error")),
        duration_ms=max(0, int(duration_ms)),
        error_category=(
            safe_metadata_token(error_category)
            if error_category is not None
            else None
        ),
        exception_type=(
            safe_metadata_token(exception_type)
            if exception_type is not None
            else None
        ),
        status_code=(
            max(0, int(status_code)) if status_code is not None else None
        ),
        argument_names=argument_names,
        unknown_argument_count=unknown_argument_count,
        result_type=result_type,
        result_size=result_size,
    )


class MCPExecutionLog:
    """Two-generation bounded JSONL store for ExecutionRecords."""

    def __init__(self, path: Path, *, max_records_per_file: int = 500) -> None:
        self.path = Path(path)
        if max_records_per_file < 1:
            raise ValueError("max_records_per_file must be positive")
        self.max_records_per_file = max_records_per_file
        self._lock = threading.RLock()
        #: TASK-21134: identity fingerprint -> sanitized bytes, for each
        #: generation this instance has already scrubbed. See
        #: ``_migrate_generation``.
        self._migrated: dict[str, tuple[tuple[int, ...], bytes]] = {}

    def append(self, record: ExecutionRecord) -> None:
        """Append one record, rotating generations at the size cap.

        Args:
            record: The execution record to persist. Every identity field is
                defensively sanitized again before it reaches disk.

        Raises:
            OSError: If the log file or its parent directory cannot be
                written (callers treat recording as best-effort).
        """
        payload = self._metadata_only_payload(asdict(record))
        encoded_line = (json.dumps(payload) + "\n").encode("utf-8")
        rotated = self.path.with_name(self.path.name + ".1")
        with self._lock:
            self._secure_parent()
            self._migrate_generation(rotated)
            active_payload = self._migrate_generation(self.path)
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
                # Both generations were just written from bytes that are
                # already sanitized; re-deriving them on the next append
                # would reproduce them exactly (TASK-21134).
                self._remember_migration(rotated, active_payload or b"")
                self._remember_migration(self.path, encoded_line)
                return
            with open_private_text_append(
                self.path,
                application_owned_directory=self.path.parent,
            ) as handle:
                handle.write(encoded_line.decode("utf-8"))
            self._remember_migration(
                self.path, (active_payload or b"") + encoded_line
            )

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
                    raw = self._migrate_generation(source)
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
                        continue
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
    def _nonnegative_int(value: Any, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return default

    @classmethod
    def _metadata_only_payload(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize current and legacy rows to the payload-free public schema."""

        ts = str(raw.get("ts") or "invalid")
        try:
            datetime.fromisoformat(ts)
        except ValueError:
            ts = "invalid"

        argument_names = []
        raw_argument_names = raw.get("argument_names")
        if isinstance(raw_argument_names, (list, tuple)):
            argument_names = sorted(
                {
                    token
                    for value in raw_argument_names
                    if (token := safe_metadata_token(value)) != "invalid"
                }
            )

        unknown_argument_count = cls._nonnegative_int(
            raw.get("unknown_argument_count")
        )
        legacy_arguments = raw.get("arguments")
        if isinstance(legacy_arguments, dict):
            # Legacy rows lack the schema required to distinguish registered
            # names. Count every supplied key and discard every key/value.
            unknown_argument_count = max(
                unknown_argument_count, len(legacy_arguments)
            )
            argument_names = []

        error_category = raw.get("error_category")
        if error_category is None and raw.get("error") is not None:
            error_category = "legacy_error"

        result_type = raw.get("result_type")
        result_size = cls._nonnegative_int(raw.get("result_size"))
        legacy_excerpt = raw.get("result_excerpt")
        if result_type is None and legacy_excerpt is not None:
            result_type = "legacy"
            result_size = len(legacy_excerpt) if isinstance(legacy_excerpt, str) else 0

        ok = raw.get("ok") is True
        status = raw.get("status") or ("success" if ok else "error")
        raw_status_code = raw.get("status_code")
        status_code = (
            cls._nonnegative_int(raw_status_code)
            if raw_status_code is not None
            else None
        )
        return {
            "ts": ts,
            "server_key": safe_metadata_token(raw.get("server_key")),
            "tool_name": safe_metadata_token(raw.get("tool_name")),
            "initiator": safe_metadata_token(raw.get("initiator")),
            "decision": safe_metadata_token(raw.get("decision")),
            "ok": ok,
            "status": safe_metadata_token(status),
            "duration_ms": cls._nonnegative_int(raw.get("duration_ms")),
            "error_category": (
                safe_metadata_token(error_category)
                if error_category is not None
                else None
            ),
            "exception_type": (
                safe_metadata_token(raw.get("exception_type"))
                if raw.get("exception_type") is not None
                else None
            ),
            "status_code": status_code,
            "argument_names": argument_names,
            "unknown_argument_count": unknown_argument_count,
            "result_type": safe_metadata_token(result_type or "none"),
            "result_size": result_size,
        }

    @staticmethod
    def _identity(path: Path) -> tuple[int, ...] | None:
        """Fingerprint a generation file, or ``None`` if it cannot be read.

        ``lstat`` deliberately: a symlink in the leaf's place reports the
        link's own inode, which can never match the fingerprint of the
        regular file this instance last wrote, so a swapped path is a cache
        MISS and falls through to the guarded read. ``None`` (missing, or any
        other stat failure) is also a miss, never a decision -- every failure
        mode is handed to ``_read_bytes`` and its private-path guards to
        classify, exactly as before this cache existed. The fingerprint
        decides staleness only.
        """
        try:
            stat = os.lstat(path)
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _remember_migration(self, path: Path, sanitized: bytes) -> None:
        """Record ``sanitized`` as this generation's content, if it still is.

        The size check is the multi-process window: another writer can append
        between our read (or our own append) and this stat, and caching then
        would pin bytes that are already short of the file while the
        fingerprint says they are current. A mismatch simply means no cache
        entry, i.e. the next call re-reads exactly as it did before.
        """
        identity = self._identity(path)
        if identity is None or identity[2] != len(sanitized):
            self._migrated.pop(str(path), None)
            return
        self._migrated[str(path)] = (identity, sanitized)

    def _migrate_generation(self, path: Path) -> bytes | None:
        """Scrub legacy payload rows and torn lines before further use.

        The scrub is idempotent, so re-running it on a file this instance
        already sanitized and has not seen change is pure waste. TASK-21134
        measured that waste at one full-file parse + re-serialize per tool
        invocation -- 4.6 ms at the 500-record cap, on every MCP tool call,
        purely to re-derive bytes identical to what the previous call wrote.
        A cached identity fingerprint (device/inode/size/mtime) skips it. Any
        change by another process, or a replaced path, misses the cache and
        takes the full scrub exactly as before.
        """

        cached = self._migrated.get(str(path))
        if cached is not None:
            identity = self._identity(path)
            if identity is not None and identity == cached[0]:
                return cached[1]
            self._migrated.pop(str(path), None)

        raw = self._read_bytes(path)
        if raw is None:
            self._migrated.pop(str(path), None)
            return None
        rows: list[dict[str, Any]] = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                rows.append(self._metadata_only_payload(decoded))
        sanitized = b"".join(
            (json.dumps(row) + "\n").encode("utf-8") for row in rows
        )
        if sanitized != raw:
            atomic_private_write_bytes(
                path,
                sanitized,
                application_owned_directory=self.path.parent,
            )
        self._remember_migration(path, sanitized)
        return sanitized
