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

from tldw_chatbook.Chat.console_trace_errors import (
    TraceReservationRefused,
    TraceSurfaceChangeRefused,
)
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor

_SEND_DIAGNOSTIC_EVENT_LIMIT = 64
# Rolled-back refusals carry their code, so the cause walk stops at them.
_TRACE_REFUSAL_TYPES = (TraceReservationRefused, TraceSurfaceChangeRefused)

# Exact code-owned categories only; never derive a token from arbitrary error text.
# Every literal refusal code raised in console_trace_service / console_trace_runtime
# (TASK-42471; Tests/Chat/test_console_trace_missing_revision_diagnostic.py scans
# both modules and fails on a code missing here).
_TRACE_FAILURE_CATEGORIES = frozenset(
    {
        "available_bundle_omission",
        "completed_replay_checkpoint",
        "completed_replay_disclosure",
        "completed_replay_policy",
        "completed_replay_terminal",
        "completed_replay_value",
        "completed_tool_turn_unavailable",
        "continuation_trace_policy",
        "continuation_trace_projection_changed",
        "credential_source",
        "duplicate_components",
        "duplicate_handler_components",
        "endpoint_identity",
        "inherited_surface_head_unavailable",
        "live_native_call",
        "live_native_custody",
        "live_native_history_append",
        "live_native_history_lineage",
        "live_native_history_prefix",
        "live_native_owner",
        "live_native_retry_surface",
        "model_name",
        "provider_name",
        "rendered_system_slot_mismatch",
        "rendered_system_slot_unavailable",
        "replacement_chain_gap",
        "replacement_component_domain",
        "replacement_component_ordinal",
        "replacement_predecessor_mismatch",
        "replacement_range_mismatch",
        "replacement_range_noncontiguous",
        "replacement_range_order",
        "replacement_value",
        "request_header_component_invalid",
        "request_header_component_unavailable",
        "request_header_unavailable",
        "request_route_unavailable",
        "required_components",
        "revision_owner_mismatch",
        "rewritten_tail_unavailable",
        "semantic_revision_value_mismatch",
        "surface_checkpoint_identity",
        "surface_child_binding",
        "surface_delta_alignment",
        "surface_delta_identity",
        "surface_delta_shape",
        "surface_owner_mismatch",
        "surface_predecessor_mismatch",
        "surface_prefix_mismatch",
        "surface_provenance_mismatch",
        "surface_reference",
        "surface_replacement_checkpoint_unavailable",
        "surface_replacement_span",
        "surface_replacement_target_inactive",
        "surface_replacement_target_unavailable",
        "surface_segment_unavailable",
        "surface_value_unavailable",
        "surface_values_unavailable",
        "surface_verified_bundle",
        "system_component_alignment",
        "system_composition_invalid",
        "system_composition_span",
        "system_revision_span",
        "system_revision_value_mismatch",
        "tool_values_unavailable",
        "trace_child_owner",
        "trace_child_segment_unavailable",
        "trace_current_transform_owner",
        "trace_fresh_turn_requires_owned_recovery",
        "trace_owner_unavailable",
        "trace_policy_mismatch",
        "trace_policy_unavailable",
        "trace_prefill_unavailable",
        "trace_primary_run_already_started",
        "trace_provenance_unavailable",
        "trace_recovery_owner",
        "trace_recovery_reservation",
        "trace_recovery_revision",
        "trace_revision_unavailable",
        "trace_route_mismatch",
        "trace_route_unavailable",
        "trace_tool_chain_source_mismatch",
        "trace_tool_chain_unavailable",
        "trace_turn_unavailable",
        "unavailable_bundle_content",
        "unavailable_bundle_reason",
        "unsupported_surface_change",
        "unsupported_surface_descriptor",
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
    #: TASK-33621.23: a code-owned token naming why the outcome is what it
    #: is (a blocked submit's blocker), written as the final line's
    #: ``error_category``. Never user content.
    outcome_category: str | None = None
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


def _trace_code(error: BaseException) -> str | None:
    """Return a trace refusal's code-owned reason, never other error text."""
    if (
        type(error) in (ValueError, *_TRACE_REFUSAL_TYPES)
        and len(error.args) == 1
        and type(error.args[0]) is str
        and error.args[0] in _TRACE_FAILURE_CATEGORIES
    ):
        return error.args[0]
    return None


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
    without an active scope this records nothing, except that a provider
    request still ends its own send's start (``console_send_start``). No file
    I/O runs on the caller.

    Args:
        phase: Code-owned lifecycle stage, never user-provided text.
        status: Code-owned outcome; ``failed`` emits at ERROR severity.
        error: Optional failure to classify without logging its message.
        **fields: Metadata allowed by the persistent diagnostic schema, such
            as runtime versions and capture state. Never pass private content,
            credentials, paths or database identifiers.
    """
    with contextlib.suppress(Exception):
        if phase == "provider_entry" and status == "entered":
            starts = sys.modules.get("tldw_chatbook.Chat.console_send_start")
            if starts is not None:  # Loaded by any send's acceptance.
                starts.note_provider_request()  # TASK-36610: this send's own.
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
                if type(cause) in _TRACE_REFUSAL_TYPES or _trace_code(cause):
                    break  # TASK-42471: the refusal names the failure, not its cause.
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
            if code := _trace_code(cause):
                fields["error_category"] = code
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
        category = current.outcome_category
        record_send_stage(
            phase, current.outcome, **({"error_category": category} if category else {})
        )
        if token is not None:
            _CURRENT.reset(token)
        if owns_monitor:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(current.monitor.close)
