"""Bounded, content-free breadcrumbs for Console sends (TASK-31977)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import sys
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor

_SEND_DIAGNOSTIC_EVENT_LIMIT = 64

# Exact code-owned categories only; never derive a token from arbitrary error text.
_TRACE_FAILURE_CATEGORIES = frozenset(
    {
        "trace_provenance_unavailable",
        "trace_policy_unavailable",
        "trace_route_unavailable",
        "trace_route_mismatch",
        "trace_owner_unavailable",
        "trace_prefill_unavailable",
        "trace_turn_unavailable",
        "trace_revision_unavailable",
        "trace_primary_run_already_started",
        "trace_fresh_turn_requires_owned_recovery",
        "trace_tool_chain_unavailable",
        "surface_provenance_mismatch",
        "surface_prefix_mismatch",
        "unsupported_surface_change",
        "semantic_revision_value_mismatch",
    }
)


@dataclass
class SendDiagnostic:
    monitor: UIResponsivenessMonitor
    # A random correlation ID, never a credential or stored conversation ID.
    # A *_token log label makes the live-view redactor remove the whole tail.
    attempt_id: str = field(default_factory=lambda: uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    phase: str = "controller_submit"
    outcome: str = "completed"
    count: int = 0
    submission_started: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def record(self, phase: str, status: str = "entered", **fields: object) -> None:
        """Record bounded metadata for this attempt in its submitting context.

        Args:
            phase: Code-owned lifecycle stage, never user-provided text.
            status: Code-owned outcome; ``failed`` retains cached metadata.
            **fields: Metadata allowed by the persistent diagnostic schema.
                Exclude content, credentials, paths and private identifiers.
        """
        self.phase = phase
        self.metadata.update(
            {
                key: value
                for key, value in fields.items()
                if key
                in {
                    "app_version",
                    "python_version",
                    "sqlite_version",
                    "capture_enabled",
                }
            }
        )
        if status == "failed":
            fields = {**self.metadata, **fields}
        self.count += 1
        if self.count > _SEND_DIAGNOSTIC_EVENT_LIMIT:
            if self.count == _SEND_DIAGNOSTIC_EVENT_LIMIT + 1:
                self.monitor.record_diagnostic(
                    "console", "diagnostic_events_dropped", item_count=1
                )
            return
        self.monitor.record_diagnostic(
            "console",
            "console_send_stage",
            phase=phase,
            status=status,
            attempt_id=self.attempt_id,
            level=logging.ERROR if status == "failed" else logging.INFO,
            duration_ms=int((time.monotonic() - self.started) * 1000),
            **fields,
        )


_CURRENT: ContextVar[SendDiagnostic | None] = ContextVar(
    "console_send_diagnostic", default=None
)


def record_send_stage(
    phase: str,
    status: str = "entered",
    *,
    error: BaseException | None = None,
    **fields: object,
) -> None:
    """Record a stage without changing delivery if diagnostic emission fails.

    Call within ``send_diagnostic_scope`` or its inherited worker context;
    without an active scope this is a no-op. No file I/O runs on the caller.

    Args:
        phase: Code-owned lifecycle stage, never user-provided text.
        status: Code-owned outcome; ``failed`` emits at ERROR severity.
        error: Optional failure to classify without logging its message.
        **fields: Metadata allowed by the persistent diagnostic schema, such
            as runtime versions and capture state. Never pass private content,
            credentials, paths or database identifiers.
    """
    with contextlib.suppress(Exception):
        current = _CURRENT.get()
        if current is None:
            return
        if error is not None:
            # Follow generic wrappers to the cause, but never stringify it.
            seen: set[int] = set()
            cause = error
            while id(cause) not in seen and len(seen) < 8:
                if isinstance(cause, TimeoutError):
                    break  # asyncio timeouts chain an internal cancellation.
                seen.add(id(cause))
                next_cause = cause.__cause__ or cause.__context__
                if next_cause is None or id(next_cause) in seen:
                    break
                cause = next_cause
            fields["exception_type"] = type(cause).__name__
            fields["error_category"] = (
                "timeout"
                if isinstance(cause, TimeoutError)
                else "database"
                if isinstance(cause, sqlite3.Error)
                else "storage"
                if isinstance(cause, OSError)
                else "validation"
                if isinstance(cause, (ValueError, TypeError))
                else "internal"
            )
            if (
                type(cause) is ValueError
                and len(cause.args) == 1
                and type(cause.args[0]) is str
                and cause.args[0] in _TRACE_FAILURE_CATEGORIES
            ):
                fields["error_category"] = cause.args[0]
            code = getattr(cause, "sqlite_errorcode", None)
            if isinstance(cause, sqlite3.Error) and type(code) is int:
                fields["sqlite_code"] = code
        current.record(phase, status, **fields)


@contextlib.asynccontextmanager
async def send_diagnostic_scope(
    phase: str, monitor: UIResponsivenessMonitor | None = None
) -> AsyncIterator[SendDiagnostic]:
    """Share an attempt across nested async scopes and inherited worker contexts.

    Subsequent controller submissions get their own token and event budget.
    A scope that creates its own monitor drains it off-loop on exit; the app
    remains responsible for closing a supplied monitor during teardown.

    Args:
        phase: Code-owned entry stage; ``controller_submit`` marks a submission.
        monitor: App-owned monitor for an outer UI scope. Nested scopes reuse
            the current monitor; standalone scopes create a temporary one.

    Yields:
        The active diagnostic attempt; callers may set its terminal outcome.

    Raises:
        BaseException: Re-raises exceptions from the wrapped send unchanged.
    """
    current = _CURRENT.get()
    owns_monitor = current is None and monitor is None
    token = None
    if current is None or (phase == "controller_submit" and current.submission_started):
        if current is not None:
            monitor = current.monitor
        from tldw_chatbook import __version__

        current = SendDiagnostic(
            monitor if monitor is not None else UIResponsivenessMonitor(enabled=False)
        )
        token = _CURRENT.set(current)
        record_send_stage(
            phase,
            app_version=__version__,
            python_version=".".join(str(value) for value in sys.version_info[:3]),
            sqlite_version=sqlite3.sqlite_version,
        )
    else:
        record_send_stage(phase)
    if phase == "controller_submit":
        current.submission_started = True
    try:
        yield current
    except BaseException as error:
        record_send_stage(current.phase, "failed", error=error)
        current.outcome = (
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
        )
        raise
    finally:
        record_send_stage(phase, current.outcome)
        if token is not None:
            _CURRENT.reset(token)
        if owns_monitor:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(current.monitor.close)
