"""MobileChatService — parent-process Console engine binding (ADR-234).

Binds the served mobile companion surface (Phase 1b) to the real Console
engine inside the parent process: one lazily constructed
``ConsoleChatController`` + persisted ``ConsoleChatStore`` + real
``ConsoleAgentBridge`` per service instance, with a duck-typed
``HeadlessAppContext`` standing in for the Textual App the controller's
agent/approval paths read (construction recipe: the 2026-10-15
mobile-engine-binding spike, Docs/superpowers/spikes/).

No Textual anywhere on this module's own paths: ``call_from_thread`` is a
direct invocation (there is no UI loop to marshal onto), the approval
capture marshals worker-thread deliveries into the asyncio loop via
``loop.call_soon_threadsafe``, and ``UnifiedMCPControlPlaneService`` is
built store-less on purpose — with no permission store every tool resolves
fail-closed to ``"ask"``, which is exactly the approval surface the phone
answers (spike condition 6).

Session posture: mobile v1 ships without project-instruction dispatch, so
every created session gets ``ProjectInstructionControlState.legacy_disabled()``
(the startup ``AGENTS.md`` flow needs UI confirm callbacks mobile does not
wire).

Logging discipline (ADR-234): session ids, run statuses, and event types
only — never message content, never credentials.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

logger = logging.getLogger(__name__)

#: Run statuses from which a new send is accepted (i.e. NOT active).
_SEND_ALLOWED_STATUSES = frozenset(
    {
        ConsoleRunStatus.IDLE,
        ConsoleRunStatus.BLOCKED,
        ConsoleRunStatus.COMPLETED,
        ConsoleRunStatus.FAILED,
        ConsoleRunStatus.STOPPED,
    }
)

#: Message statuses that end a message's lifecycle (``message_done`` fires
#: only on a transition INTO one of these — a ``pending``→``streaming``
#: hop is progress, not completion).
_TERMINAL_MESSAGE_STATUSES = frozenset(
    {"complete", "failed", "stopped", "discarded"}
)

#: Delta pump cadence (seconds) while any session has an active run.
_POLL_INTERVAL_SECONDS = 0.5


class MobileSendRefused(RuntimeError):
    """The engine refused a mobile send before a run started."""


@dataclass(frozen=True)
class MobileEvent:
    """WS wire shape for one mobile event (JSON-serializable via to_wire).

    Attributes:
        type: One of ``run_started`` | ``delta`` | ``message_done`` |
            ``approval_requested`` | ``run_completed`` | ``run_failed`` |
            ``session_updated``.
        session_id: Owning Console session id ("" when service-wide).
        payload: Type-specific dict — ``delta``: ``{"message_id", "offset",
            "text"}``; ``approval_requested``: the captured round payload
            (``round_id``, ``calls[...]`` with ``llm_name``/``call_id``/
            ``options``/``arguments``, ``timeout_seconds``); terminal events
            carry the final status.
        seq: Per-service monotonic sequence number.
    """

    type: str
    session_id: str
    payload: dict[str, Any]
    seq: int

    def to_wire(self) -> dict[str, Any]:
        """Return the JSON-ready projection of this event."""
        return {
            "type": self.type,
            "session_id": self.session_id,
            "payload": self.payload,
            "seq": self.seq,
        }


class HeadlessAppContext:
    """Duck-typed stand-in for the Textual App the controller reads.

    The agent path touches App attributes via ``getattr(..., None)`` and
    tolerates absence; the two that MUST exist are ``unified_mcp_service``
    (permission-store owner for the local ``fs_*`` tools — store-less means
    fail-closed ``"ask"``) and ``workspace_registry_service`` (admitted
    workspace roots — without it the ``fs_*`` family is never advertised).
    ``call_from_thread`` is the only METHOD the approval path calls; off
    the UI loop a direct invocation is correct (spike note).
    """

    def __init__(
        self, workspace_registry_service: LocalWorkspaceRegistryService
    ) -> None:
        self.unified_mcp_service = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=None,
            server_service=None,
        )
        self.workspace_registry_service = workspace_registry_service

    def call_from_thread(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke directly — there is no UI loop to marshal onto."""
        return fn(*args, **kwargs)


@dataclass
class _Engine:
    """The lazily constructed engine graph (all real, App-free)."""

    registry: LocalWorkspaceRegistryService
    workspace_db: WorkspaceDB
    store: ConsoleChatStore
    agent_runs_db: AgentRunsDB
    bridge: ConsoleAgentBridge
    controller: ConsoleChatController
    app_context: HeadlessAppContext


@dataclass
class _StreamTrack:
    """Per-session delta-pump state for the current assistant message."""

    message_id: str
    offset: int
    status: str


@dataclass
class _EngineState:
    """Mutable per-service bookkeeping (loop-thread only)."""

    #: session_id -> last observed run status (missing = IDLE).
    run_status: dict[str, ConsoleRunStatus] = field(default_factory=dict)
    #: session_id -> current assistant-message delta state.
    stream_track: dict[str, _StreamTrack] = field(default_factory=dict)
    #: round_id -> captured approval payload (cleared by the None payload).
    pending_approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Mount order of currently-mounted rounds (the card slot is one deep —
    #: the None clear pops the newest, mirroring the TUI card lifecycle).
    mounted_rounds: list[str] = field(default_factory=list)
    #: Sessions with a background send-turn task scheduled but not yet
    #: finished (non-blocking send, live-check round 1). Drives the genuine
    #: pre-submit refusal: the engine admits one run per session, so a
    #: second send while a turn is pending is refused, not queued.
    turns_in_flight: set[str] = field(default_factory=set)
    #: Sessions with a send-initiated run whose terminal event has not been
    #: emitted yet. ``send`` is the authoritative terminal emitter (it knows
    #: a run started AND ended — a sub-poll-cadence run can otherwise flip
    #: the observed status terminal→terminal between two pump ticks and
    #: vanish); the pump emits first whenever it observes the transition,
    #: and whichever emits discards the flag, so each run announces exactly
    #: one ``run_completed``/``run_failed``.
    pending_terminal: set[str] = field(default_factory=set)


class MobileChatService:
    """Parent-process Console engine binding for the mobile surface.

    One service instance owns one engine graph over ``data_dir`` SQLite
    databases. Sends are non-blocking: :meth:`send` pre-validates, schedules
    the turn on a background task, and returns; the turn is serialized per
    session via an ``asyncio.Lock`` and its streaming/outcome is observed by
    subscribing to :meth:`events`. Approval rounds land in
    ``pending_approvals`` via the ``set_pending_approval`` capture
    (worker-thread deliveries are marshalled into the loop) and are decided
    through :meth:`decide_approval` → ``controller.resolve_pending_approval``.
    """

    def __init__(self, *, data_dir: Path, provider_gateway: Any) -> None:
        """Initialize the service (engine construction is lazy).

        Args:
            data_dir: Directory for the chat / workspace / agent-runs
                SQLite databases (created when missing).
            provider_gateway: ``ConsoleProviderGatewayProtocol``
                implementation used for every turn.
        """
        self._data_dir = Path(data_dir)
        self._gateway = provider_gateway
        self._engine: _Engine | None = None
        self._hook_permissions: Any | None = None
        self._state = _EngineState()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._turn_tasks: dict[str, asyncio.Task[None]] = {}
        self._subscribers: set[asyncio.Queue[MobileEvent]] = set()
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    async def _ensure_started(self) -> _Engine:
        """Build the engine graph once, bind the loop, start the pump."""
        if self._closed:
            raise RuntimeError("MobileChatService is shut down")
        if self._engine is None:
            self._build_engine()
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(
                self._pump(), name="mobile-chat-service-pump"
            )
        return self._engine

    def _build_engine(self) -> None:
        """Construct the engine graph verbatim per the spike recipe."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        workspace_db = WorkspaceDB(
            str(self._data_dir / "workspaces.db"), client_id="mobile-service"
        )
        registry = LocalWorkspaceRegistryService(workspace_db)
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(
                CharactersRAGDB(str(self._data_dir / "chat.db"), "mobile-service"),
                workspace_registry=registry,
            )
        )
        agent_runs_db = AgentRunsDB(
            str(self._data_dir / "runs.db"), client_id="mobile-service"
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=agent_runs_db,
            store=store,
            provider_gateway=self._gateway,
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=self._gateway,
            provider="openai",
            model="gpt-4o-mini",
            agent_bridge=bridge,
            agent_runtime_enabled=True,
            # Hooks admission owner, mirroring ConsoleRuntime's
            # ``ensure_hook_permissions``: with hooks configured the send
            # gate fails closed until the definitions are reviewed (mobile
            # v1 has no hook-review surface), while an all-reviewed or
            # hooks-free profile admits sends. No ``ensure_run_hooks`` is
            # wired: mobile v1 does not fire run hooks (spike shape).
            hook_permissions_accessor=lambda: self._ensure_hook_permissions(),
        )
        app_context = HeadlessAppContext(workspace_registry_service=registry)
        controller.app = app_context
        controller.set_pending_approval = self._on_pending_approval
        self._engine = _Engine(
            registry=registry,
            workspace_db=workspace_db,
            store=store,
            agent_runs_db=agent_runs_db,
            bridge=bridge,
            controller=controller,
            app_context=app_context,
        )
        logger.info("mobile engine constructed over data_dir=%s", self._data_dir)

    def _ensure_hook_permissions(self):
        """Return the app-owned hook consent owner (built once)."""
        if self._hook_permissions is None:
            from tldw_chatbook.Agents.hook_permissions import HookPermissions

            self._hook_permissions = HookPermissions()
        return self._hook_permissions

    def _require_engine(self) -> _Engine:
        """Return the engine, building it (sync parts) when necessary."""
        if self._closed:
            raise RuntimeError("MobileChatService is shut down")
        if self._engine is None:
            self._build_engine()
        return self._engine

    @property
    def workspace_registry(self) -> LocalWorkspaceRegistryService:
        """The service's workspace registry (fixture/support surface)."""
        return self._require_engine().registry

    # ------------------------------------------------------------------ #
    # Read surfaces                                                       #
    # ------------------------------------------------------------------ #

    @property
    def sessions(self) -> list[dict[str, Any]]:
        """Sessions newest-first: ``[{id, title, workspace_id, has_folder_tools}]``."""
        engine = self._engine
        if engine is None:
            return []
        return [self._session_dict(s) for s in reversed(engine.store.sessions())]

    def _session_dict(self, session: Any) -> dict[str, Any]:
        return {
            "id": session.id,
            "title": session.title,
            "workspace_id": session.workspace_id,
            "has_folder_tools": self._has_folder_tools(session.workspace_id),
        }

    def _has_folder_tools(self, workspace_id: str) -> bool:
        """Whether the workspace admits local folder bindings.

        Default-workspace (``global``) sessions never do: an empty admitted
        ``admitted_roots`` sequence drops the whole ``fs_*``/``git_*`` path
        family (spike finding; the mobile UX must not offer folder tools
        for those sessions).
        """
        engine = self._engine
        if engine is None or workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
            return False
        try:
            return bool(engine.registry.list_folder_bindings(workspace_id))
        except Exception:  # noqa: BLE001 - listing must never break reads
            logger.warning(
                "folder-binding lookup failed for workspace_id=%s",
                workspace_id,
            )
            return False

    def session_messages(
        self, session_id: str, *, after_message_id: str | None = None
    ) -> dict[str, Any]:
        """Return transcript rows, optionally only those after a cursor.

        Args:
            session_id: Console session id.
            after_message_id: Cursor — only messages strictly after this
                message id are returned.

        Returns:
            ``{"messages": [{id, role, content, status, created_at}],
            "cursor": last_message_id}`` — ``created_at`` is ``None``: the
            in-memory ``ConsoleChatMessage`` model carries no per-row
            timestamp (documented deviation from the plan sketch).

        Raises:
            KeyError: Unknown ``session_id``.
            ValueError: ``after_message_id`` is not in the transcript.
        """
        engine = self._require_engine()
        messages = engine.store.messages_for_session(session_id)
        if after_message_id is not None:
            ids = [m.id for m in messages]
            try:
                index = ids.index(after_message_id)
            except ValueError as exc:
                raise ValueError(
                    f"cursor message not found in session {session_id}"
                ) from exc
            messages = messages[index + 1 :]
        wire = [
            {
                "id": m.id,
                "role": getattr(m.role, "value", m.role),
                "content": m.content,
                "status": getattr(m.status, "value", m.status),
                "created_at": None,
            }
            for m in messages
        ]
        cursor = wire[-1]["id"] if wire else (after_message_id or None)
        return {"messages": wire, "cursor": cursor}

    # ------------------------------------------------------------------ #
    # Write surfaces                                                      #
    # ------------------------------------------------------------------ #

    async def create_session(
        self, *, title: str, workspace_id: str | None = None
    ) -> dict[str, Any]:
        """Create a Console session in the mobile posture.

        Args:
            title: Session title.
            workspace_id: Workspace to bind. ``None`` means the Console
                default (``global``) — those sessions get no folder tools.

        Returns:
            The new session's ``{id, title, workspace_id, has_folder_tools}``.
        """
        engine = await self._ensure_started()
        session = engine.store.create_session(title=title, workspace_id=workspace_id)
        # Mobile v1: a workspace-bound chat WITHOUT project-instruction
        # dispatch — the startup AGENTS.md flow wants UI confirm callbacks
        # this surface does not wire (spike condition 4).
        engine.store.set_session_project_instruction_state(
            session.id, ProjectInstructionControlState.legacy_disabled()
        )
        self._emit(
            "session_updated",
            session.id,
            {"session_id": session.id, "reason": "created"},
        )
        logger.info(
            "mobile session created id=%s workspace_id=%s",
            session.id,
            session.workspace_id,
        )
        return self._session_dict(session)

    async def send(self, session_id: str, draft: str) -> None:
        """Schedule one draft's turn and return once it is submitted.

        Non-blocking (live-check round 1, spec §4.2): the caller's 202 must
        not wait on the turn, because ``controller.submit_draft`` spans the
        whole run — including a tool-approval wait that only the phone can
        resolve. After the synchronous pre-validation below, the turn runs
        on a background task (:meth:`_run_turn`) that owns the lock, the
        exactly-once terminal announcement, and error capture; the turn's
        streaming and outcome are observed via :meth:`events`.

        Pre-validation refusals (raised before anything is scheduled, so the
        REST send route maps them to 404/409):
        - unknown ``session_id`` (``KeyError`` — the engine has no such
          session);
        - a turn already pending or a run still active for the session
          (``MobileSendRefused``) — the engine's own admission rule is one
          run per session, so this mirrors a genuine pre-submit refusal.

        Raises:
            KeyError: Unknown ``session_id``.
            MobileSendRefused: A turn is already in flight for the session.
        """
        engine = await self._ensure_started()
        # Cheap existence probe: raises KeyError for an unknown session
        # without materializing messages.
        engine.store.message_count(session_id)
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        status = engine.controller.run_state_for(session_id).status
        if (
            session_id in self._state.turns_in_flight
            or lock.locked()
            or status not in _SEND_ALLOWED_STATUSES
        ):
            logger.info("mobile send refused session_id=%s", session_id)
            raise MobileSendRefused("a turn is already in flight")
        self._state.turns_in_flight.add(session_id)
        task = asyncio.create_task(
            self._run_turn(session_id, draft),
            name=f"mobile-send-turn-{session_id[:8]}",
        )
        self._turn_tasks[session_id] = task
        task.add_done_callback(
            lambda finished, sid=session_id: self._turn_done(sid, finished)
        )
        logger.info("mobile send scheduled session_id=%s", session_id)

    async def _run_turn(self, session_id: str, draft: str) -> None:
        """Run one scheduled turn to its terminal event (background task).

        The old blocking-``send`` body: activate the session (the approval
        rounds of the ACTIVE session mount immediately, and the controller's
        stop entrypoint targets the active session), submit the draft, then
        flush this turn's not-yet-delivered content (delta + message_done)
        BEFORE the terminal announce — a run whose whole life fits between
        two pump ticks would otherwise be silently baselined and the phone
        would never see its text as events.
        """
        # Head start for the caller's 202: this task is queued the moment
        # ``send`` schedules it, and the engine's submit pre-flight (tool
        # loading, provider composition) runs synchronously for its first
        # stretch. Yielding first lets the pending response write (and the
        # REST middleware unwind) complete before that stretch monopolizes
        # the loop — measured live at ~3 s of starvation without this.
        await asyncio.sleep(0.05)
        engine = self._require_engine()
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            self._activate_session(engine, session_id)
            self._state.pending_terminal.add(session_id)
            try:
                result = await engine.controller.submit_draft(
                    draft,
                    session_id=session_id,
                    origin=ConsoleSubmissionOrigin.MANUAL,
                )
            except BaseException:
                self._state.pending_terminal.discard(session_id)
                raise
            if not result.accepted:
                # A refusal that only surfaced at submit time (e.g. hooks
                # not reviewed): the caller already holds its 202, so the
                # refusal must arrive as an event for the phone to resync
                # and re-enable its composer. The distinguishable
                # ``send_refused`` status lets the phone drop its
                # optimistic user bubble (nothing was persisted).
                self._state.pending_terminal.discard(session_id)
                logger.info("mobile send refused session_id=%s", session_id)
                self._emit("run_failed", session_id, {"status": "send_refused"})
                self._emit(
                    "session_updated",
                    session_id,
                    {"session_id": session_id, "reason": "run_failed"},
                )
                self._state.run_status[session_id] = ConsoleRunStatus.FAILED
                return
            logger.info("mobile send accepted session_id=%s", session_id)
            self._scan_stream(engine, session_id, active=False, flush=True)
            self._announce_run_terminal(engine, session_id)

    def _turn_done(self, session_id: str, task: asyncio.Task[None]) -> None:
        """Reap one finished turn task; convert its failure into events."""
        self._state.turns_in_flight.discard(session_id)
        if self._turn_tasks.get(session_id) is task:
            self._turn_tasks.pop(session_id, None)
        if task.cancelled():
            return  # shutdown: no terminal event for a deliberately killed turn
        error = task.exception()
        if error is None:
            return
        # The turn died mid-run (engine-side failure). The phone already
        # holds its 202: announce a terminal event so the composer
        # re-enables and the resync shows the persisted reality.
        logger.error(
            "mobile turn failed session_id=%s type=%s",
            session_id,
            type(error).__name__,
        )
        self._emit_terminal_events(session_id, ConsoleRunStatus.FAILED)

    def cancel(self, session_id: str) -> bool:
        """Stop the session's active run at the next safe boundary.

        Uses the controller's Stop entrypoint (``stop_active_run`` — the
        same one the TUI Stop button calls), which by contract targets only
        the store's ACTIVE session; the target session is activated first
        via the store's public ``switch_session`` (the same call a TUI tab
        switch makes).

        Args:
            session_id: Session whose run should stop.

        Returns:
            True when the session had an active run and it was stopped.

        Raises:
            KeyError: Unknown ``session_id``.
        """
        engine = self._require_engine()
        self._activate_session(engine, session_id)
        stopped = engine.controller.stop_active_run()
        logger.info(
            "mobile cancel session_id=%s stopped=%s", session_id, stopped
        )
        return stopped

    def _activate_session(self, engine: _Engine, session_id: str) -> None:
        """Make ``session_id`` the active session when it is not already."""
        if engine.store.active_session_id != session_id:
            engine.store.switch_session(session_id)

    # ------------------------------------------------------------------ #
    # Approvals                                                           #
    # ------------------------------------------------------------------ #

    def _on_pending_approval(self, payload: dict[str, Any] | None) -> None:
        """Capture callback for ``controller.set_pending_approval``.

        Runs on the agent worker thread (delivered through
        ``HeadlessAppContext.call_from_thread``); every mutation of
        loop-owned state is marshalled into the event loop.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if payload is None:
            loop.call_soon_threadsafe(self._clear_mounted_approval)
        else:
            loop.call_soon_threadsafe(self._mount_approval, payload)

    def _mount_approval(self, payload: dict[str, Any]) -> None:
        """Record a mounted round and announce it (loop thread)."""
        round_id = str(payload.get("round_id") or "")
        if not round_id:
            return
        self._state.pending_approvals[round_id] = payload
        self._state.mounted_rounds.append(round_id)
        self._emit(
            "approval_requested",
            str(payload.get("session_id") or ""),
            payload,
        )
        logger.info(
            "mobile approval round mounted round_id=%s", round_id
        )

    def _clear_mounted_approval(self) -> None:
        """Drop the newest mounted round (the card slot is one deep)."""
        while self._state.mounted_rounds:
            round_id = self._state.mounted_rounds.pop()
            if self._state.pending_approvals.pop(round_id, None) is not None:
                logger.info("mobile approval round cleared round_id=%s", round_id)
                break

    def list_approvals(self) -> list[dict[str, Any]]:
        """Pending approval rounds across mobile sessions."""
        return [dict(payload) for payload in self._state.pending_approvals.values()]

    def decide_approval(self, round_id: str, decisions: dict[str, str]) -> bool:
        """Resolve one pending round.

        Args:
            round_id: The captured round's id.
            decisions: Verdict-key (``call_id`` or ``llm_name``) to
                decision string (``approve_once`` / ``approve_session`` /
                ``always_allow`` / ``deny``).

        Returns:
            True when the round was known and the decision was delivered;
            False for an unknown round (nothing was resolved).
        """
        engine = self._require_engine()
        if round_id not in self._state.pending_approvals:
            logger.info(
                "mobile approval decision for unknown round_id=%s", round_id
            )
            return False
        engine.controller.resolve_pending_approval(
            dict(decisions), round_id=round_id
        )
        logger.info(
            "mobile approval decision delivered round_id=%s calls=%d",
            round_id,
            len(decisions),
        )
        return True

    # ------------------------------------------------------------------ #
    # Events                                                              #
    # ------------------------------------------------------------------ #

    async def events(self) -> AsyncIterator[MobileEvent]:
        """Subscribe to the service's event fan-out (one queue per caller)."""
        await self._ensure_started()
        queue: asyncio.Queue[MobileEvent] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    def _emit(self, type_: str, session_id: str, payload: dict[str, Any]) -> None:
        """Fan one event out to every subscriber (loop thread only)."""
        self._seq += 1
        event = MobileEvent(
            type=type_, session_id=session_id, payload=payload, seq=self._seq
        )
        for queue in list(self._subscribers):
            queue.put_nowait(event)
        logger.debug(
            "mobile event type=%s session_id=%s seq=%d",
            type_,
            session_id,
            event.seq,
        )

    async def _pump(self) -> None:
        """Delta pump: 0.5 s poll while any session has an active run."""
        try:
            while True:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                engine = self._engine
                if engine is None or self._closed:
                    continue
                try:
                    self._poll_once(engine)
                except Exception:  # noqa: BLE001 - one bad tick must not kill the pump
                    logger.exception("mobile delta pump tick failed")
        except asyncio.CancelledError:
            raise

    def _announce_run_terminal(self, engine: _Engine, session_id: str) -> None:
        """Emit this run's terminal event exactly once (loop thread).

        Called by :meth:`send` when its run has ended and by the pump when
        it observes the transition; the ``pending_terminal`` flag makes the
        two exactly-once.
        """
        if session_id not in self._state.pending_terminal:
            return
        self._state.pending_terminal.discard(session_id)
        status = engine.controller.run_state_for(session_id).status
        self._emit_terminal_events(session_id, status)

    def _emit_terminal_events(
        self, session_id: str, status: ConsoleRunStatus
    ) -> None:
        """Emit run_completed/run_failed + session_updated for a run end."""
        if status == ConsoleRunStatus.COMPLETED:
            event_type = "run_completed"
        else:
            event_type = "run_failed"
        self._emit(event_type, session_id, {"status": status.value})
        self._emit(
            "session_updated",
            session_id,
            {"session_id": session_id, "reason": event_type},
        )
        self._state.run_status[session_id] = status

    def _poll_once(self, engine: _Engine) -> None:
        """One pump tick: run-status transitions + message deltas."""
        for session in engine.store.sessions():
            session_id = session.id
            state = engine.controller.run_state_for(session_id)
            status = state.status
            active = status not in _SEND_ALLOWED_STATUSES
            self._scan_stream(engine, session_id, active=active)
            previous = self._state.run_status.get(session_id, ConsoleRunStatus.IDLE)
            if status != previous:
                was_active = previous not in _SEND_ALLOWED_STATUSES
                if status in (
                    ConsoleRunStatus.COMPLETED,
                    ConsoleRunStatus.FAILED,
                    ConsoleRunStatus.STOPPED,
                    ConsoleRunStatus.BLOCKED,
                ):
                    # Observed the transition (or its residue) before send()
                    # returned: this emitter wins, and send()'s announce
                    # becomes a no-op via the pending flag.
                    self._state.pending_terminal.discard(session_id)
                    self._emit_terminal_events(session_id, status)
                elif active and not was_active:
                    self._emit(
                        "run_started", session_id, {"status": status.value}
                    )
            self._state.run_status[session_id] = status

    def _scan_stream(
        self,
        engine: _Engine,
        session_id: str,
        *,
        active: bool,
        flush: bool = False,
    ) -> None:
        """Emit deltas beyond the last offset and terminal ``message_done``.

        Args:
            engine: The engine graph.
            session_id: Session to scan.
            active: Whether the session's run is currently active.
            flush: Send-time flush — deliver the current message's full
                remaining content as deltas (no silent first-observation
                baseline) and a ``message_done`` when its status is already
                terminal. Exactly-once is preserved: the offset tracking
                and the status-change guard make a pump-delivered piece or
                event a no-op here.
        """
        messages = engine.store.messages_for_session(session_id)
        assistant = None
        for message in reversed(messages):
            if getattr(message.role, "value", message.role) == "assistant":
                assistant = message
                break
        track = self._state.stream_track.get(session_id)
        if assistant is None:
            if track is not None:
                self._state.stream_track.pop(session_id, None)
            return
        content = assistant.content or ""
        status = getattr(assistant.status, "value", assistant.status)
        if track is None or track.message_id != assistant.id:
            if (
                track is not None
                and track.status not in _TERMINAL_MESSAGE_STATUSES
            ):
                # The tracked message ended without a terminal status ever
                # being observed; report its last seen status.
                self._emit(
                    "message_done",
                    session_id,
                    {"message_id": track.message_id, "status": track.status},
                )
            offset = 0
            if track is None and not active and not flush:
                # First observation of an INACTIVE session: baseline the
                # existing history silently (events are live-only).
                offset = len(content)
            if content[offset:]:
                self._emit(
                    "delta",
                    session_id,
                    {
                        "message_id": assistant.id,
                        "offset": offset,
                        "text": content[offset:],
                    },
                )
            if flush and status in _TERMINAL_MESSAGE_STATUSES:
                self._emit(
                    "message_done",
                    session_id,
                    {"message_id": assistant.id, "status": status},
                )
            self._state.stream_track[session_id] = _StreamTrack(
                message_id=assistant.id, offset=len(content), status=status
            )
            return
        if len(content) > track.offset:
            self._emit(
                "delta",
                session_id,
                {
                    "message_id": assistant.id,
                    "offset": track.offset,
                    "text": content[track.offset:],
                },
            )
            track.offset = len(content)
        if status != track.status:
            track.status = status
            if status in _TERMINAL_MESSAGE_STATUSES:
                self._emit(
                    "message_done",
                    session_id,
                    {"message_id": assistant.id, "status": status},
                )

    # ------------------------------------------------------------------ #
    # Teardown                                                            #
    # ------------------------------------------------------------------ #

    async def shutdown(self) -> None:
        """Stop the pump, stop every run, quiesce and close the databases.

        Idempotent. Mirrors the spike's teardown discipline:
        ``controller.shutdown()`` (bounded at 10 s), store connection
        quiesce + close, then ``AgentRunsDB.close()``.
        """
        if self._closed:
            return
        self._closed = True
        pump = self._pump_task
        if pump is not None:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        # Outstanding send-turn tasks are deliberately killed; the engine
        # teardown below stops their runs.
        turn_tasks = list(self._turn_tasks.values())
        self._turn_tasks.clear()
        self._state.turns_in_flight.clear()
        for task in turn_tasks:
            task.cancel()
        if turn_tasks:
            await asyncio.gather(*turn_tasks, return_exceptions=True)
        engine = self._engine
        if engine is None:
            return
        try:
            await asyncio.wait_for(engine.controller.shutdown(), timeout=10)
        except Exception:  # noqa: BLE001 - teardown must finish regardless
            logger.exception("mobile controller shutdown failed")
        try:
            db = getattr(getattr(engine.store, "persistence", None), "db", None)
            if db is not None:
                with db.quiesce_connections(timeout_seconds=5):
                    pass
                db.close()
        except Exception:  # noqa: BLE001 - teardown must finish regardless
            logger.exception("mobile store quiesce/close failed")
        for closer in (
            engine.agent_runs_db.close,
            getattr(engine.workspace_db, "close", None),
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception:  # noqa: BLE001 - teardown must finish regardless
                logger.exception("mobile database close failed")
        logger.info("mobile engine shut down")
