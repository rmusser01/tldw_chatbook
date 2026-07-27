"""Segmented, append-only run log writer.

Impure by design (filesystem + config). Path resolution reuses the file
tools' own chain -- ``allowed_file_roots`` -> ``is_within`` ->
``is_sensitive_path`` -- so this writer can never become a path-validation
bypass. See the design spec §3.3, §7, §8, §9.2.
"""

from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger

from .run_log_format import RunLogRecord, encode_record

#: Directory created inside the resolved root. Deliberately UNDOTTED: a
#: dotted directory is excluded by `_is_hidden_within`, which would hide
#: the log from the very tools meant to read it.
DEFAULT_DIR_NAME = "agent-runs"
DEFAULT_SEGMENT_BYTES = 4_000_000
DEFAULT_MAX_RECORD_BYTES = 1_000_000
MANIFEST_NAME = "MANIFEST"


def _setting(key: str, default):
    """Read one ``[agents]`` config key. Test seam: monkeypatched wholesale.

    Args:
        key: Key name within the ``[agents]`` section.
        default: Value returned when unset or unreadable.

    Returns:
        The configured value, or ``default``.
    """
    try:
        from tldw_chatbook.config import get_cli_setting

        value = get_cli_setting("agents", key, default)
    except Exception:
        return default
    return default if value is None else value


def resolve_log_root() -> Path | None:
    """Return the directory the log tree is created under, or ``None``.

    Prefers the run's first read-write workspace folder root so the log is
    a user-visible artifact; falls back to the tool sandbox root when no
    such folder is bound. Any failure resolves to ``None`` (logging off)
    rather than to a wider or unvalidated location.

    Returns:
        The chosen root directory, or ``None`` when none is usable.
    """
    try:
        from tldw_chatbook.Tools.file_operation_tools import _tool_sandbox_root
        from tldw_chatbook.Tools.workspace_file_roots import allowed_file_roots

        sandbox = _tool_sandbox_root()
        roots = allowed_file_roots(write=True, sandbox_root=sandbox)
    except Exception:
        logger.opt(exception=True).warning("run log: cannot resolve any root")
        return None
    if not roots:
        return None
    # allowed_file_roots returns (sandbox, *workspace_folders); prefer a
    # bound workspace folder, fall back to the sandbox.
    for candidate in roots[1:]:
        return candidate
    return roots[0]


class RunLogWriter:
    """Appends records for ONE run tree to a segmented log.

    Constructed unbound (the run id does not exist until ``_run_one`` calls
    ``create_run``), then bound once by the primary run. Child runs share
    the instance, and therefore the record counter, so parent and child
    record numbers can never collide.
    """

    def __init__(
        self,
        *,
        dir_name: str | None = None,
        segment_bytes: int | None = None,
        max_record_bytes: int | None = None,
    ) -> None:
        """Build an UNBOUND writer. Explicit args override ``[agents]`` config.

        Args:
            dir_name: Directory name; defaults to ``[agents] run_log_dir_name``.
            segment_bytes: Roll threshold; defaults to
                ``[agents] run_log_segment_bytes``.
            max_record_bytes: Per-record ceiling; defaults to
                ``[agents] run_log_max_record_bytes``.
        """
        self._dir_name = dir_name or str(_setting("run_log_dir_name", DEFAULT_DIR_NAME))
        self._segment_bytes = int(
            segment_bytes
            if segment_bytes is not None
            else _setting("run_log_segment_bytes", DEFAULT_SEGMENT_BYTES)
        )
        self._max_record_bytes = int(
            max_record_bytes
            if max_record_bytes is not None
            else _setting("run_log_max_record_bytes", DEFAULT_MAX_RECORD_BYTES)
        )
        self._lock = threading.Lock()
        self._counter = 0
        self._segment_index = 1
        self._segment_size = 0
        self._active = False
        self.log_dir: Path | None = None

    @property
    def is_active(self) -> bool:
        """Whether records are currently being written."""
        return self._active

    def bind(self, run_id: str) -> None:
        """Bind to ``run_id`` and create its directory. Idempotent.

        Args:
            run_id: The PRIMARY run's id. Later calls are ignored so a
                child run never rebinds its parent's writer.
        """
        if self.log_dir is not None:
            return
        if not _setting("run_log_enabled", True):
            self._active = False
            return
        root = resolve_log_root()
        if root is None:
            self._active = False
            return
        try:
            base = root / self._dir_name
            base.mkdir(parents=True, exist_ok=True)
            gitignore = base / ".gitignore"
            if not gitignore.exists():
                # Created only if absent: writing into a user's repository
                # is itself a mutation.
                gitignore.write_text("*\n", encoding="utf-8")
            run_dir = base / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.opt(exception=True).warning(
                "run log: cannot create log directory; logging disabled"
            )
            self._active = False
            return
        self.log_dir = run_dir
        self._active = True

    def _segment_path(self) -> Path:
        assert self.log_dir is not None
        return self.log_dir / f"logs.{self._segment_index:04d}.txt"

    def _write_bytes(self, path: Path, payload: bytes, *, sync: bool = False) -> None:
        """Append ``payload`` to ``path``.

        ``flush()`` on every record survives a process crash. ``fsync`` is
        reserved for segment rolls and run end (``sync=True``): calling it
        per record into a user's project directory is wasteful.

        Args:
            path: Target file, opened in append-binary mode.
            payload: Bytes to append.
            sync: Whether to force an ``fsync`` after flushing.
        """
        import os

        with open(path, "ab") as handle:
            handle.write(payload)
            handle.flush()
            if sync:
                os.fsync(handle.fileno())

    def append(
        self,
        *,
        run_id: str,
        kind: str,
        type: str,
        content: str,
        tool: str = "",
        status: str = "",
        call_id: str = "",
    ) -> int | None:
        """Append one record and return its number.

        Args:
            run_id: Id of the run this record belongs to (parent or child).
            kind: ``primary`` or ``subagent``.
            type: ``model``, ``tool_call``, ``tool_result``, or ``spawn``.
            content: Full, untruncated text.
            tool: Tool name, when applicable.
            status: ``ok`` / ``error``, when applicable.
            call_id: Provider ``tool_call_id``, when applicable.

        Returns:
            The assigned record number, or ``None`` when the writer is
            inactive or the write failed. Never raises.
        """
        if not self._active or self.log_dir is None:
            return None
        with self._lock:
            truncated_from = 0
            body = content.encode("utf-8")
            if len(body) > self._max_record_bytes:
                truncated_from = len(body)
                # Cut on a character boundary, then re-encode.
                body = body[: self._max_record_bytes]
                content = body.decode("utf-8", "ignore")
            self._counter += 1
            record = RunLogRecord(
                number=self._counter,
                run_id=run_id,
                kind=kind,
                type=type,
                ts=_now_iso(),
                content=content,
                tool=tool,
                status=status,
                call_id=call_id,
                truncated_from=truncated_from,
            )
            payload = encode_record(record)
            # Roll BEFORE writing: a record must never span segments, or
            # bytes=-exact parsing (which assumes one file) breaks.
            if self._segment_size and self._segment_size + len(payload) > (
                self._segment_bytes
            ):
                try:
                    # fsync the segment being retired; it will not be
                    # appended to again.
                    self._write_bytes(self._segment_path(), b"", sync=True)
                except Exception:  # noqa: BLE001 — durability is best-effort
                    logger.opt(exception=True).warning("run log: segment fsync failed")
                self._segment_index += 1
                self._segment_size = 0
            try:
                self._write_bytes(self._segment_path(), payload)
            except Exception:
                logger.opt(exception=True).warning(
                    "run log: append failed; logging disabled for this run"
                )
                self._active = False
                return None
            self._segment_size += len(payload)
            return record.number

    def write_manifest(self, metadata: dict) -> None:
        """Write run-level convenience metadata. Never raises.

        The manifest is deliberately NOT load-bearing: segment discovery is
        glob + sort (``run_log_search.load_records``), so a crashed run that
        never reaches this call is still fully readable.

        Args:
            metadata: Run-level fields (model, budget, status, supersession).
        """
        if self.log_dir is None:
            return
        import json

        payload = dict(metadata)
        try:
            payload["segments"] = [p.name for p in sorted(self.log_dir.glob("logs.*.txt"))]
            payload["record_count"] = self._counter
            self._write_bytes(
                self.log_dir / MANIFEST_NAME,
                json.dumps(payload, indent=2, default=str).encode("utf-8"),
                sync=True,
            )
        except Exception:  # noqa: BLE001 — convenience metadata only
            logger.opt(exception=True).warning("run log: manifest write failed")

    def close(self) -> None:
        """Flush the final segment to disk. Idempotent and always safe."""
        if not self._active or self.log_dir is None:
            return
        try:
            self._write_bytes(self._segment_path(), b"", sync=True)
        except Exception:  # noqa: BLE001 — best-effort durability
            logger.opt(exception=True).warning("run log: final fsync failed")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
