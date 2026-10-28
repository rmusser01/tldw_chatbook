"""Read-only, bounded evidence collection for the bug report modal."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import platform
import re
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity
from tldw_chatbook.Utils.bug_report import BugReportEvidence, BugReportItem
from tldw_chatbook.Utils.log_sanitizer import sanitize_string

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


@dataclass(frozen=True)
class BugReportContext:
    runtime_identity: RuntimeIdentity = field(repr=False)
    workspace_id: str | None = field(repr=False)
    session_id: str | None = field(repr=False)
    session_epoch: int | None = field(repr=False)
    conversation_id: str | None = field(repr=False)
    origin_route: str
    collection_started_at: str


@dataclass(frozen=True)
class BugReportCapture:
    context: BugReportContext = field(repr=False)
    evidence: BugReportEvidence = field(repr=False)
    tool_refs: tuple = field(default=(), repr=False)
    valid: bool = True


def capture_bug_report_context(app: TldwCli) -> BugReportContext:
    """Freeze already-existing selection without initializing any Console state."""
    identity = app._current_runtime_identity()
    runtime = getattr(app, "console_runtime", None)
    store = getattr(runtime, "chat_store", None)
    active = getattr(store, "active_session_id", None)
    session = (
        next((item for item in store.sessions() if item.id == active), None)
        if store is not None
        else None
    )
    workspace = getattr(
        getattr(store, "workspace_context", None), "active_workspace_id", None
    )
    if workspace is None:
        workspace = getattr(session, "workspace_id", None)
    eligible = (
        identity.active_source == "local"
        and session is not None
        and getattr(session, "workspace_id", None) == workspace
    )
    return BugReportContext(
        identity,
        workspace,
        active,
        store.active_session_epoch() if store else None,
        getattr(session, "persisted_conversation_id", None) if eligible else None,
        str(getattr(app, "current_tab", "unknown")),
        datetime.now(timezone.utc).isoformat(),
    )


class BugReportCollector:
    """Own source tasks until retirement, including after a presentation timeout."""

    presentation_timeout = 5.0

    def __init__(self, app: TldwCli, context: BugReportContext) -> None:
        self.app = app
        self.context = context
        runtime = getattr(app, "console_runtime", None)
        store = getattr(runtime, "chat_store", None)
        self.chat_db = getattr(getattr(store, "persistence", None), "db", None)
        self.agent_db = getattr(app, "agent_runs_db", None)
        if self.agent_db is None:
            bridge = getattr(runtime, "agent_bridge", None)
            self.agent_db = getattr(bridge, "runs_db", None) or getattr(
                runtime, "_agent_runs_db", None
            )
        self.log_handler = getattr(app, "_persistent_log_handler", None)
        self._cancelled = threading.Event()
        self._pending: dict[str, asyncio.Task] = {}
        self._aliases: dict[str, str] = {}
        self._deadline = float("inf")
        self._last_capture: BugReportCapture | None = None

    @property
    def pending_sources(self) -> tuple[str, ...]:
        """Sources whose underlying worker still owns unfinished work."""
        return tuple(name for name, task in self._pending.items() if not task.done())

    def cancel(self) -> None:
        """Request cooperative retirement without abandoning thread-backed tasks."""
        self._cancelled.set()

    async def wait_retired(self) -> None:
        """Wait for already-owned tasks, used by teardown after cancellation."""
        await asyncio.gather(
            *(asyncio.shield(task) for task in tuple(self._pending.values())),
            return_exceptions=True,
        )

    def is_current(self) -> bool:
        current = capture_bug_report_context(self.app)
        return (
            not self._cancelled.is_set()
            and current.runtime_identity == self.context.runtime_identity
            and current.workspace_id == self.context.workspace_id
            and current.session_id == self.context.session_id
            and current.session_epoch == self.context.session_epoch
        )

    async def _source(self, name: str, read, *args):
        owners = getattr(self.app, "_bug_report_collectors", None)
        if owners is None:
            owners = set()
            self.app._bug_report_collectors = owners
        if any(owner is not self and name in owner.pending_sources for owner in owners):
            return None
        previous = self._pending.get(name)
        if previous is not None and not previous.done():
            return None
        if self._cancelled.is_set():
            return None
        task = asyncio.create_task(asyncio.to_thread(read, *args))
        self._pending[name] = task
        owners.add(self)
        try:
            done, _ = await asyncio.wait({task}, timeout=self.presentation_timeout)
            if not done:
                return None
            return task.result()
        except asyncio.CancelledError:
            self.cancel()
            raise
        except Exception:
            return None  # Source failures are controlled states, never raw error text.
        finally:
            # Retrieve late exceptions without cancelling the backing thread.
            def retired(done):
                if not done.cancelled():
                    done.exception()
                if not self.pending_sources:
                    owners.discard(self)

            task.add_done_callback(retired)

    def _environment(self) -> tuple[tuple[str, str], ...]:
        from importlib.metadata import PackageNotFoundError, version
        from tldw_chatbook.Utils.about_text import get_app_version

        try:
            textual_version = version("textual")
        except PackageNotFoundError:
            textual_version = "unavailable"
        return (
            ("Collected (UTC)", self.context.collection_started_at),
            ("App version", get_app_version()),
            ("Python", platform.python_version()),
            ("Textual", textual_version),
            ("SQLite", sqlite3.sqlite_version),
            (
                "OS (app server)",
                f"{platform.system()} {platform.release()} {platform.machine()}",
            ),
            (
                "Mode",
                "served (server environment)"
                if getattr(self.app, "is_web", False)
                else "terminal",
            ),
            ("Origin route", self.context.origin_route),
        )

    def _read_logs(self) -> BugReportEvidence:
        if self.log_handler is None:
            return BugReportEvidence(
                inventory=("Application-wide session logs unavailable.",)
            )
        sequence, rows = self.log_handler.snapshot_report_tail()
        logs = tuple(BugReportItem("logs", row[2]) for row in rows)
        exceptions = tuple(
            BugReportItem("exceptions", item.text)
            for item in logs
            if "Traceback (most recent call last)" in item.text
        )
        clock = datetime.now().astimezone().strftime("%Z %z")
        return BugReportEvidence(
            logs=logs,
            exceptions=exceptions,
            inventory=(
                f"Application-wide session logs: {len(logs)} retained source entries through sequence {sequence}.",
                f"Log clock: captured app-server local time; offset at collection {clock}; historical offset not inferred.",
                "Only current-session log tail is available; original entries may already be truncated to 2,000 characters.",
            ),
        )

    def _alias(self, value: object, domain: str = "event") -> str:
        raw = str(value)
        if raw not in self._aliases:
            self._aliases[raw] = f"{domain}-{len(self._aliases) + 1:04d}"
        return self._aliases[raw]

    def _fact(self, source: str, key: str, row: dict) -> BugReportItem:
        from tldw_chatbook.Chat.trajectory import redact_local_paths

        values = [self._alias(key), f"source={source}"]
        for name in (
            "event_kind",
            "state",
            "status",
            "tool_name",
            "provider",
            "model",
            "seq",
            "step_started_at",
            "first_token_at",
            "completed_at",
            "created_at",
            "dispatch_started_at",
            "response_started_at",
            "settled_at",
            "outcome",
        ):
            value = row.get(name)
            if value is not None:
                values.append(
                    f"{name}={redact_local_paths(sanitize_string(str(value)))}"
                )
        for name in ("run_id", "turn_id", "parent_run_id"):
            if row.get(name):
                values.append(
                    f"{name}={self._alias(row[name], name.removesuffix('_id'))}"
                )
        return BugReportItem("trace", " ".join(values), event_key=key)

    def _read_trace(self) -> tuple[BugReportEvidence, tuple]:
        if self.chat_db is None or not self.context.conversation_id:
            return BugReportEvidence(
                inventory=(
                    "Console trace unavailable: no selected local durable conversation.",
                )
            ), ()
        from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
        from tldw_chatbook.DB.base_db import operation_owned_connection

        streams: list[list[BugReportItem]] = []
        notes: list[str] = []
        events = ()
        with operation_owned_connection(self.chat_db):
            connection = self.chat_db.get_connection()
            deadline = self._deadline
            previous_busy = int(connection.execute("PRAGMA busy_timeout").fetchone()[0])
            connection.execute("PRAGMA busy_timeout = 200")
            work = 0

            def stop_read():
                nonlocal work
                work += 1000
                return int(
                    self._cancelled.is_set()
                    or time.monotonic() > deadline
                    or work >= 3_000_000
                )

            connection.set_progress_handler(stop_read, 1000)
            try:
                with self.chat_db.transaction() as cursor:
                    repo = ConsoleTraceRepository()
                    tail = repo.read_report_event_tail(
                        cursor, self.context.conversation_id
                    )
                    events = tail.events
                    native = []
                    for event in events:
                        if self._cancelled.is_set() or time.monotonic() > deadline:
                            notes.append(
                                "Native metadata read stopped at its work/deadline budget."
                            )
                            break
                        call = repo.get_call(cursor, event.call_id or "")
                        row = {"event_kind": event.event_type, "seq": event.sequence}
                        if call is not None:
                            row.update(
                                state=call.state.value,
                                provider=call.provider_name,
                                model=call.model_name,
                                dispatch_started_at=call.dispatch_started_at,
                                response_started_at=call.response_started_at,
                                settled_at=call.settled_at,
                                outcome=call.outcome,
                                run_id=call.run_id,
                                turn_id=call.turn_id,
                            )
                        native.append(self._fact("native", event.event_id, row))
                    streams.append(native)
                    notes.append(
                        f"Console native excerpt: {len(events)} events; older={tail.has_older}; lineage coverage limited={tail.lineage_truncated}."
                    )
                rows = self.chat_db.get_report_trajectory_tail(
                    self.context.conversation_id
                )
                streams.append(
                    [
                        self._fact(
                            "trajectory",
                            f"trajectory:{row['message_id']}:{row['seq']}",
                            row,
                        )
                        for row in rows
                    ]
                )
            finally:
                connection.set_progress_handler(None, 0)
                connection.execute(f"PRAGMA busy_timeout = {previous_busy}")
        if (
            self.agent_db is not None
            and not self._cancelled.is_set()
            and time.monotonic() <= self._deadline
        ):
            try:
                rows = self.agent_db.get_report_event_tail(self.context.conversation_id)
                streams.append(
                    [
                        self._fact("agent", f"agent:{row['run_id']}:{row['seq']}", row)
                        for row in rows
                    ]
                )
            except Exception:
                notes.append("Agent metadata unavailable.")
        # Independent source streams have no trustworthy shared clock. Select
        # recent facts fairly while preserving each source's causal order.
        selected = []
        while len(selected) < 500 and any(streams):
            for stream in streams:
                if stream and len(selected) < 500:
                    selected.append(stream.pop())
        selected.reverse()
        keys = {item.event_key for item in selected}
        refs = tuple(event for event in events if event.event_id in keys)
        notes.append(
            "Source-local order preserved; cross-source order is not proof of causality. Missing parents/timing remain unavailable."
        )
        return BugReportEvidence(trace=tuple(selected), inventory=tuple(notes)), refs

    def _read_tools(self, refs: tuple) -> tuple[BugReportItem, ...]:
        if self.chat_db is None or not self.context.conversation_id:
            return ()
        from tldw_chatbook.Chat.console_trace_native_reader import (
            ConsoleTraceNativeReader,
        )
        from tldw_chatbook.Chat.trajectory import redact_local_paths
        from tldw_chatbook.DB.base_db import operation_owned_connection

        with operation_owned_connection(self.chat_db):
            connection = self.chat_db.get_connection()
            previous_busy = int(connection.execute("PRAGMA busy_timeout").fetchone()[0])
            connection.execute("PRAGMA busy_timeout = 200")
            deadline = self._deadline
            connection.set_progress_handler(
                lambda: int(self._cancelled.is_set() or time.monotonic() > deadline),
                1000,
            )
            try:
                rows = ConsoleTraceNativeReader(self.chat_db).read_report_tool_fields(
                    self.context.conversation_id, refs
                )
            finally:
                connection.set_progress_handler(None, 0)
                connection.execute(f"PRAGMA busy_timeout = {previous_busy}")
        return tuple(
            BugReportItem(
                "tools",
                f"{self._alias(row['event_key'])}: {redact_local_paths(sanitize_string(str(row.get('text') or row.get('unavailable'))))}",
                event_key=str(row["event_key"]),
            )
            for row in rows
        )

    async def collect(self) -> BugReportCapture:
        self._deadline = time.monotonic() + self.presentation_timeout
        logs, trace, environment = await asyncio.gather(
            self._source("logs", self._read_logs),
            self._source("trace", self._read_trace),
            self._source("environment", self._environment),
        )
        log_evidence = logs or BugReportEvidence(
            inventory=("Session log read unavailable or timed out.",)
        )
        trace_evidence, refs = (
            trace
            if trace is not None
            else (
                BugReportEvidence(
                    inventory=("Console trace read unavailable or timed out.",)
                ),
                (),
            )
        )
        valid = self.is_current()
        log_items = tuple(
            replace(item, text=self._alias_log(item.text)) for item in log_evidence.logs
        )
        exceptions = tuple(
            replace(item, text=self._alias_log(item.text))
            for item in log_evidence.exceptions
        )
        evidence = BugReportEvidence(
            environment=environment or (("Environment", "unavailable"),),
            logs=log_items,
            exceptions=exceptions,
            trace=trace_evidence.trace if valid else (),
            inventory=log_evidence.inventory
            + trace_evidence.inventory
            + (
                f"Source collection finished (UTC): {datetime.now(timezone.utc).isoformat()}",
            )
            + (
                ()
                if valid
                else (
                    "Selection changed or collection cancelled; Console evidence excluded.",
                )
            ),
        )
        capture = BugReportCapture(self.context, evidence, refs if valid else (), valid)
        self._last_capture = capture
        return capture

    def _alias_log(self, text: str) -> str:
        for original, alias in self._aliases.copy().items():
            if len(original) >= 16:
                text = re.sub(
                    r"(?<![\w-])" + re.escape(original) + r"(?![\w-])",
                    lambda _match: alias,
                    text,
                )
        return text

    async def collect_tools(self, capture: BugReportCapture) -> BugReportCapture:
        if not capture.valid or not self.is_current():
            return replace(capture, valid=False)
        self._deadline = time.monotonic() + self.presentation_timeout
        tools = await self._source("tools", self._read_tools, capture.tool_refs)
        if not self.is_current():
            return replace(capture, valid=False)
        evidence = replace(
            capture.evidence,
            tools=tools or (),
            inventory=capture.evidence.inventory
            + (
                ()
                if tools is not None
                else ("Governed tool read unavailable or timed out.",)
            ),
        )
        return replace(capture, evidence=evidence)
