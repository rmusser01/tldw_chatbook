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

Restart rehydration (ADR-234 recorded follow-up): ``sessions`` also lists
persisted conversations from the isolated mobile chat DB — lazily, no tree
hydration; the persisted conversation id is the session identity. Touching an
unknown id on ``session_messages``/``send`` attempts a lazy restore through
the SAME hydration machinery the TUI resume and the launch wake use
(``load_console_conversation_tree`` + ``hydrate_console_generation_settings``
+ ``hydrate_console_session`` → ``ConsoleChatStore.restore_persisted_session``);
an id still unknown after the restore attempt raises ``KeyError`` as before.

Reading surface (Phase 2b, read-only): notes and library browsing through
lazily opened handles over the app's own chatbook databases
(``config.get_chachanotes_db_path`` / ``config.get_media_db_path`` — the
same helpers ``app.py`` constructs them from). Strictly read paths: the
only calls made are ``list_notes`` / ``get_note_by_id`` /
``search_media_db`` / ``get_media_transcripts``.

Desktop approval relay (Phase 2c, ADR-236): ``attach_relay`` subscribes
the service to the parent's loopback canvas control broker. A drainer
task reads each bound child's broker event queue (the same per-child
queue ``canvas.events`` push frames would land in — production canvas
delivery never reads it, it is pull-based via ``canvas.events.request``,
so this drainer is that queue's only consumer and cannot steal frames
from the served canvas gateway). Desktop approval rounds merge into
``list_approvals`` marked ``surface: "desktop"`` and phone decisions
route back through ``broker.request``; a child that disconnects loses
its rounds fail-closed. With no broker attached (canvas disabled) the
relay is inert and desktop approvals stay invisible on the phone.

Logging discipline (ADR-234): session ids, run statuses, and event types
only — never message content, never credentials.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable

from tldw_chatbook.Canvas.control_protocol import (
    DEFAULT_MAX_QUEUED_EVENTS,
    ControlProtocolError,
)
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import (
    ConversationLoadFailed,
    ConversationServiceUnavailable,
    hydrate_console_generation_settings,
    hydrate_console_session,
    load_console_conversation_tree,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase, get_media_transcripts
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

#: Delta pump cadence (seconds) — one scan tick over the sessions in scope
#: (active run, turn slot, or pending stream track; see ``_poll_once``).
_POLL_INTERVAL_SECONDS = 0.5

#: Preview length for mobile notes listings (Phase 2b wire contract).
_NOTES_PREVIEW_CHARS = 200

#: Page size for the persisted-conversation listing that backs ``sessions``
#: (restart rehydration). One page, newest-first — the mobile surface's own
#: list is small; the conversations route gains real pagination as its own
#: follow-up.
_PERSISTED_LISTING_LIMIT = 500

#: Relay supervisor cadence (seconds): spawn/stop per-child drain tasks and
#: run the fail-closed disconnect probes (ADR-236).
_RELAY_SCAN_INTERVAL_SECONDS = 0.25

#: Per-child ``next_event`` wait inside a drain task (seconds): purely a
#: re-check cadence; cancels land at this granularity.
_RELAY_DRAIN_TIMEOUT_SECONDS = 1.0

#: Connectivity probe timeout (seconds) for children that still hold
#: desktop rounds — a failed probe fails their rounds closed.
_RELAY_PROBE_TIMEOUT_SECONDS = 0.1

#: Parent→child decision routing budget (seconds) for
#: ``relay.decide.request`` (plan bound: 2 s).
_RELAY_DECIDE_TIMEOUT_SECONDS = 2.0


def _wire_safe(value: Any) -> Any:
    """Project one read value onto JSON-safe wire types, recursively.

    Both reading DBs hand back ``datetime`` objects for their DATETIME
    columns (``detect_types``); the mobile wire carries them as ISO 8601
    strings. Everything else passes through untouched.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _wire_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_safe(item) for item in value]
    return value


class MobileSendRefused(RuntimeError):
    """The engine refused a mobile send before a run started."""


@dataclass(frozen=True)
class MobileEvent:
    """WS wire shape for one mobile event (JSON-serializable via to_wire).

    Attributes:
        type: One of ``run_started`` | ``delta`` | ``message_done`` |
            ``approval_requested`` | ``run_status`` | ``run_completed`` |
            ``run_failed`` | ``session_updated``.
        session_id: Owning Console session id ("" when service-wide).
        payload: Type-specific dict — ``delta``: ``{"message_id", "offset",
            "text"}``; ``approval_requested``: the captured round payload
            (``round_id``, ``calls[...]`` with ``llm_name``/``call_id``/
            ``options``/``arguments``, ``timeout_seconds``; desktop-relayed
            rounds add ``child_id`` and ``surface``); ``run_status``
            (desktop relay, ADR-236): ``{"status", "surface": "desktop"}``;
            terminal events carry the final status.
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

    The persisted-session resume machinery additionally reads
    ``chachanotes_db`` and ``chat_conversation_scope_service`` (the same two
    attributes the TUI resume and the launch wake read off the real App) —
    here they point at the service's own isolated mobile chat DB, so the
    lazy restore hydrates exactly the conversations this surface owns.
    """

    def __init__(
        self,
        workspace_registry_service: LocalWorkspaceRegistryService,
        chachanotes_db: CharactersRAGDB,
        conversation_service: ChatConversationService,
    ) -> None:
        self.unified_mcp_service = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=None,
            server_service=None,
        )
        self.workspace_registry_service = workspace_registry_service
        self.chachanotes_db = chachanotes_db
        # Local-only scope service over the mobile chat DB, same shape the
        # app wires: no server service, no policy gates — mode "local" is
        # the only mode the resume machinery requests.
        self.chat_conversation_scope_service = ChatConversationScopeService(
            local_service=conversation_service,
            server_service=None,
        )

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
    conversation_service: ChatConversationService


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
    #: session_id -> the store's O(1) change fingerprint as observed at the
    #: last MATERIALIZING stream scan (#3128 minor 3). An equal fingerprint
    #: on a terminal-track session proves nothing can be waiting, so the
    #: next scan skips the transcript snapshot build entirely.
    scan_fingerprints: dict[str, tuple[int, int] | None] = field(
        default_factory=dict
    )
    #: session_id -> ((fingerprint, last observed stream-track status),
    #: wire rows) for ``session_messages`` (#3128 minor 3): the phone's 12 s
    #: resync over an unchanged session answers from this cache instead of
    #: re-materializing the transcript. One entry per touched session —
    #: the same order of memory as the store's own transcript copy.
    history_cache: dict[
        str,
        tuple[tuple[tuple[int, int] | None, str | None], list[dict[str, Any]]],
    ] = field(default_factory=dict)


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
    Desktop rounds relayed from served terminal children (ADR-236, after
    :meth:`attach_relay`) join the same approvals surface and route their
    decisions back over the canvas control broker.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        provider_gateway: Any,
        notes_db_path: Path | None = None,
        media_db_path: Path | None = None,
    ) -> None:
        """Initialize the service (engine construction is lazy).

        Args:
            data_dir: Directory for the chat / workspace / agent-runs
                SQLite databases (created when missing).
            provider_gateway: ``ConsoleProviderGatewayProtocol``
                implementation used for every turn.
            notes_db_path: Override for the reading notes database path.
                ``None`` (the default — and the only production value)
                resolves via ``config.get_chachanotes_db_path()`` on first
                read, exactly as the app itself constructs the DB; the
                override exists so tests can point at tmp_path databases
                without loading config.
            media_db_path: Override for the reading media database path
                (``None`` → ``config.get_media_db_path()``, same rule).
        """
        self._data_dir = Path(data_dir)
        self._gateway = provider_gateway
        self._engine: _Engine | None = None
        self._hook_permissions: Any | None = None
        self._state = _EngineState()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._turn_tasks: dict[str, asyncio.Task[None]] = {}
        # Serializes lazy persisted-session restores (restart rehydration):
        # the store's restore path creates a session under an explicit id
        # and refuses duplicates, so two racing first touches of the same
        # unknown id must not both restore.
        self._restore_lock = asyncio.Lock()
        self._reset_subscribers: set[asyncio.Queue[MobileEvent]] = set()
        self._subscribers: set[asyncio.Queue[MobileEvent]] = set()
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._closed = False
        self._notes_db_path_override = (
            Path(notes_db_path) if notes_db_path is not None else None
        )
        self._media_db_path_override = (
            Path(media_db_path) if media_db_path is not None else None
        )
        self._notes_db: CharactersRAGDB | None = None
        self._media_db: MediaDatabase | None = None
        self._reading_dbs_lock = threading.Lock()
        # Desktop approval relay (ADR-236): set once ``attach_relay`` binds
        # the parent's lifecycle control broker; Canvas disablement does
        # not revoke this runtime connection (ADR-244).
        self._relay_broker: Any | None = None
        self._relay_child_ids: Callable[[], Iterable[str]] | None = None
        self._relay_task: asyncio.Task[None] | None = None
        # (child_id, round_id) -> {**relay.approval.raised payload,
        # child_id, surface: "desktop"}; loop-thread only. Round ids are
        # child-local, so the composite key keeps two children colliding
        # on one round id from shadowing or stealing each other's rounds.
        self._desktop_approvals: dict[tuple[str, str], dict[str, Any]] = {}
        self._relay_observed_buffers: dict[str, Any] = {}

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
        from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner

        # Constructors do not start turns. Until publication, this scope owns
        # every DB and the lease, including on a failed graph initialization.
        execution_owner = ConsoleExecutionOwner.acquire(self._data_dir / "runs.db")
        resources = []
        try:
            workspace_db = WorkspaceDB(
                str(self._data_dir / "workspaces.db"), client_id="mobile-service"
            )
            resources.append(workspace_db)
            registry = LocalWorkspaceRegistryService(workspace_db)
            chat_db = CharactersRAGDB(str(self._data_dir / "chat.db"), "mobile-service")
            resources.append(chat_db)
            conversation_service = ChatConversationService(chat_db)
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    chat_db,
                    workspace_registry=registry,
                )
            )
            agent_runs_db = AgentRunsDB(
                str(self._data_dir / "runs.db"),
                client_id="mobile-service",
                execution_owner=execution_owner,
            )
            resources.append(agent_runs_db)
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
            app_context = HeadlessAppContext(
                workspace_registry_service=registry,
                chachanotes_db=chat_db,
                conversation_service=conversation_service,
            )
            controller.app = app_context
            controller.set_pending_approval = self._on_pending_approval
            engine = _Engine(
                registry=registry,
                workspace_db=workspace_db,
                store=store,
                agent_runs_db=agent_runs_db,
                bridge=bridge,
                controller=controller,
                app_context=app_context,
                conversation_service=conversation_service,
            )
        except BaseException:
            retired = True
            for resource in reversed(resources):
                try:
                    resource.close()
                except Exception:  # retire the remaining resources
                    retired = False
                    logger.exception("mobile startup database close failed")
            if retired:
                execution_owner.close()
            raise
        self._engine = engine
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
    def provider_gateway(self) -> Any:
        """The bound provider gateway (introspection surface for wiring tests)."""
        return self._gateway

    @property
    def workspace_registry(self) -> LocalWorkspaceRegistryService:
        """The service's workspace registry (fixture/support surface)."""
        return self._require_engine().registry

    # ------------------------------------------------------------------ #
    # Read surfaces                                                       #
    # ------------------------------------------------------------------ #

    @property
    def sessions(self) -> list[dict[str, Any]]:
        """Sessions newest-first: ``[{id, title, workspace_id, has_folder_tools}]``.

        In-memory sessions lead (newest-first); persisted conversations from
        the isolated mobile chat DB that are not yet in memory follow, also
        newest-first by their ``last_modified`` — every conversation in that
        DB is mobile's. A persisted row is listed under the persisted
        conversation id (the identity the store's restore path reuses), with
        NO tree hydration: reading one later lazily restores it via
        :meth:`session_messages`/:meth:`send`.
        """
        engine = self._require_engine()
        merged: dict[str, dict[str, Any]] = {}
        bound_conversations: set[str] = set()
        for session in reversed(engine.store.sessions()):
            merged[session.id] = self._session_dict(session)
            if session.persisted_conversation_id is not None:
                # Already open under its live session identity — the
                # persisted row below must not double-list it (the same
                # idiom the launch wake uses to skip open conversations).
                bound_conversations.add(str(session.persisted_conversation_id))
        for row in self._persisted_conversation_rows(engine):
            conversation_id = str(row.get("id") or "")
            if (
                not conversation_id
                or conversation_id in merged
                or conversation_id in bound_conversations
            ):
                continue
            workspace_id = self._persisted_row_workspace_id(row)
            merged[conversation_id] = {
                "id": conversation_id,
                "title": str(row.get("title") or "") or "New Chat",
                "workspace_id": workspace_id,
                "has_folder_tools": self._has_folder_tools(workspace_id),
            }
        return list(merged.values())

    def sessions_page(self, *, page: int = 1, per_page: int = 50) -> dict[str, Any]:
        """One page of the sessions listing (reading-route envelope).

        Windows :attr:`sessions` newest-first into the paginated envelope
        the notes/library routes already answer (``{"items", "page",
        "has_more"}``) so the conversations route's contract matches them
        (#3128 minor 7). ``has_more`` is honest relative to the merged
        listing — in-memory sessions plus the persisted-listing page capped
        at ``_PERSISTED_LISTING_LIMIT`` — so far-out pages over a
        long-lived mobile DB may under-report; the mobile list is small
        and the cap is 500.

        Args:
            page: 1-based page number (validated at the route layer).
            per_page: Page size (route layer: default 50, max 100).
        """
        listing = self.sessions
        start = (page - 1) * per_page
        return {
            "items": listing[start : start + per_page],
            "page": page,
            "has_more": start + per_page < len(listing),
        }

    def _persisted_conversation_rows(
        self, engine: _Engine
    ) -> list[dict[str, Any]]:
        """One newest-first page of persisted mobile conversations.

        Scope ``all`` over the isolated DB (no workspace/client filters —
        everything in it is mobile's); listing must never break the read.

        Args:
            engine: The engine graph.

        Returns:
            Normalized conversation rows (may be empty).
        """
        try:
            page = engine.conversation_service.list_conversations(
                limit=_PERSISTED_LISTING_LIMIT, scope_type="all"
            )
        except Exception:  # noqa: BLE001 - listing must never break reads
            logger.warning("mobile persisted conversation listing failed")
            return []
        items = page.get("items")
        return items if isinstance(items, list) else []

    @staticmethod
    def _persisted_row_workspace_id(row: dict[str, Any]) -> str:
        """The workspace a persisted row's restored session would report."""
        if str(row.get("scope_type") or "") == "workspace" and row.get(
            "workspace_id"
        ):
            return str(row["workspace_id"])
        # Global-scope rows are LABELED the Console default workspace by
        # assumption: this mobile store's engine never switches workspaces
        # at runtime (every session here is created under the default),
        # and the listing must stay a cheap row read. The label is proven
        # right — or corrected — when the row is actually opened: the lazy
        # hydrate (``_restore_persisted_session``) resolves the restored
        # session's true workspace via the store's own resume machinery.
        return CONSOLE_GLOBAL_WORKSPACE_ID

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

    async def session_messages(
        self, session_id: str, *, after_message_id: str | None = None
    ) -> dict[str, Any]:
        """Return transcript rows, optionally only those after a cursor.

        An unknown ``session_id`` first attempts a lazy restore of the
        persisted conversation with that id (restart rehydration); an id
        still unknown after the attempt raises ``KeyError`` as before.

        Fingerprint-gated (#3128 minor 3): the wire is rebuilt only when
        the store's O(1) ``session_fingerprint`` seam reports movement (or
        the pump's last observed stream-track status changed — a terminal
        status flip can land with no content change); an unchanged session
        — the phone's 12 s resync over a settled chat — answers from the
        cached projection without materializing the transcript. A key
        match is necessary, not sufficient: the cached wire is served only
        when its own shape (row count + last row's content length) equals
        the seam's values — the projection-equality second line the seam's
        limitations docstring directs (e.g. a quarantined last message
        projects a fixed placeholder while the seam measures the raw
        content; such a wire is rebuilt, never served stale).

        Args:
            session_id: Console session id (a persisted conversation id is
                restored on first touch).
            after_message_id: Cursor — only messages strictly after this
                message id are returned.

        Returns:
            ``{"messages": [{id, role, content, status, created_at}],
            "cursor": last_message_id}`` — ``created_at`` is ``None``: the
            in-memory ``ConsoleChatMessage`` model carries no per-row
            timestamp (documented deviation from the plan sketch).

        Raises:
            KeyError: Unknown ``session_id`` (after the restore attempt).
            ValueError: ``after_message_id`` is not in the transcript.
        """
        engine = self._require_engine()
        try:
            fingerprint: tuple[int, int] | None = (
                engine.store.session_fingerprint(session_id)
            )
        except KeyError:
            if not await self._restore_persisted_session(session_id):
                raise
            fingerprint = engine.store.session_fingerprint(session_id)
        track = self._state.stream_track.get(session_id)
        cache_key = (fingerprint, track.status if track is not None else None)
        cached = self._state.history_cache.get(session_id)
        messages: list[dict[str, Any]] | None = None
        if cached is not None and cached[0] == cache_key:
            # Projection-equality second line (PR-#3142 minor, per the
            # seam's documented caveat): serve the cached wire only when
            # its shape still equals the seam's values — a wire cached
            # under this very fingerprint can still disagree with it (the
            # snapshot projection, e.g. the quarantine placeholder, is
            # not what the seam measures).
            candidate = cached[1]
            last_length = (
                len(candidate[-1]["content"] or "") if candidate else 0
            )
            if len(candidate) == fingerprint[0] and last_length == fingerprint[1]:
                messages = candidate
        if messages is None:
            materialized = engine.store.messages_for_session(session_id)
            messages = [
                {
                    "id": m.id,
                    "role": getattr(m.role, "value", m.role),
                    "content": m.content,
                    "status": getattr(m.status, "value", m.status),
                    "created_at": None,
                }
                for m in materialized
            ]
            self._state.history_cache[session_id] = (cache_key, messages)
        if after_message_id is not None:
            ids = [m["id"] for m in messages]
            try:
                index = ids.index(after_message_id)
            except ValueError as exc:
                raise ValueError(
                    f"cursor message not found in session {session_id}"
                ) from exc
            messages = messages[index + 1 :]
        cursor = messages[-1]["id"] if messages else (after_message_id or None)
        return {"messages": messages, "cursor": cursor}

    async def _restore_persisted_session(self, session_id: str) -> bool:
        """Lazily restore one persisted conversation into the engine store.

        Reuses the store's resume machinery exactly as the TUI resume and
        the launch wake do — ``load_console_conversation_tree`` (full tree,
        raised caps) + ``hydrate_console_generation_settings`` +
        ``hydrate_console_session`` (which builds ``all_nodes`` and calls
        ``ConsoleChatStore.restore_persisted_session``) — over the
        ``HeadlessAppContext``'s mobile-DB conversation service. The
        restored session takes the persisted conversation id as its session
        identity, so later touches need no translation.

        Args:
            session_id: The unknown id being touched (a persisted
                conversation id).

        Returns:
            True when the session exists afterwards (restored here, or
            another task restored it first); False when no persisted
            conversation matched or the restore failed (logged, ids only).
        """
        engine = self._require_engine()
        if self._session_known(engine, session_id):
            return True
        async with self._restore_lock:
            if self._session_known(engine, session_id):
                return True
            try:
                tree = await load_console_conversation_tree(
                    engine.app_context, session_id
                )
            except (
                ConversationServiceUnavailable,
                ConversationLoadFailed,
            ) as exc:
                logger.warning(
                    "mobile persisted-session restore could not load the tree "
                    "session_id=%s error=%s",
                    session_id,
                    type(exc).__name__,
                )
                return False
            if tree is None:
                return False
            conversation = tree.get("conversation")
            if not isinstance(conversation, dict):
                conversation = {}
            try:
                hydration = hydrate_console_generation_settings({}, conversation)
                await hydrate_console_session(
                    app=engine.app_context,
                    store=engine.store,
                    conversation_id=session_id,
                    tree=tree,
                    settings=hydration.settings,
                    generation_durable_snapshot=hydration.durable_snapshot,
                    generation_metadata_status=hydration.metadata_status,
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001 - a failed restore is a 404
                logger.warning(
                    "mobile persisted-session restore failed session_id=%s "
                    "error=%s",
                    session_id,
                    type(exc).__name__,
                )
                return False
            logger.info("mobile persisted session restored id=%s", session_id)
            return True

    @staticmethod
    def _session_known(engine: _Engine, session_id: str) -> bool:
        """Whether the store already holds a session with this id."""
        return any(session.id == session_id for session in engine.store.sessions())

    # ------------------------------------------------------------------ #
    # Reading surfaces (Phase 2b, read-only)                              #
    # ------------------------------------------------------------------ #

    def _ensure_reading_dbs(self) -> tuple[CharactersRAGDB, MediaDatabase]:
        """Return the reading DB handles, opening them lazily (read-only use).

        Paths resolve exactly as the app resolves them — the same config
        helpers ``app.py`` builds these databases from
        (``get_chachanotes_db_path`` / ``get_media_db_path``) — unless a
        test-injected override was supplied. Opening runs each DB's schema
        init/migration path (the constructors own that); this is the
        established Phase 1b lazy-open-post-startup posture, and the reads
        themselves are strictly read-only.

        Returns:
            ``(notes_db, media_db)`` handles.

        Raises:
            RuntimeError: The service is shut down.
        """
        if self._closed:
            raise RuntimeError("MobileChatService is shut down")
        if self._notes_db is not None and self._media_db is not None:
            return self._notes_db, self._media_db
        with self._reading_dbs_lock:
            if self._notes_db is None or self._media_db is None:
                from tldw_chatbook.config import (
                    get_chachanotes_db_path,
                    get_media_db_path,
                )

                notes_path = (
                    self._notes_db_path_override or get_chachanotes_db_path()
                )
                media_path = self._media_db_path_override or get_media_db_path()
                if self._notes_db is None:
                    self._notes_db = CharactersRAGDB(
                        notes_path, client_id="mobile-service"
                    )
                    logger.info("mobile reading notes DB opened path=%s", notes_path)
                if self._media_db is None:
                    self._media_db = MediaDatabase(
                        db_path=media_path, client_id="mobile-service"
                    )
                    logger.info("mobile reading media DB opened path=%s", media_path)
        return self._notes_db, self._media_db

    def notes_page(self, *, page: int = 1, per_page: int = 50) -> dict[str, Any]:
        """List non-deleted notes newest-first with truncated previews.

        Args:
            page: 1-based page number.
            per_page: Page size (the route layer caps this).

        Returns:
            ``{"items": [{id, title, preview, created_at, last_modified}],
            "page": n, "has_more": bool}`` — ``preview`` is the note's
            first ``_NOTES_PREVIEW_CHARS`` characters (never the full
            body); the timestamps pass through from ``list_notes`` as-is.
            ``has_more`` comes from the one-over fetch row.
        """
        notes_db, _ = self._ensure_reading_dbs()
        rows = notes_db.list_notes(
            limit=per_page + 1,
            offset=(page - 1) * per_page,
            preview_chars=_NOTES_PREVIEW_CHARS,
        )
        has_more = len(rows) > per_page
        items = [
            _wire_safe(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "preview": row["preview"],
                    "created_at": row.get("created_at"),
                    "last_modified": row.get("last_modified"),
                }
            )
            for row in rows[:per_page]
        ]
        return {"items": items, "page": page, "has_more": has_more}

    def note_detail(self, note_id: str) -> dict[str, Any] | None:
        """Return one note's full row, or ``None`` when it does not exist.

        Args:
            note_id: Note id.

        Returns:
            The note row (``id`` / ``title`` / ``content`` plus the other
            scalar row fields ``get_note_by_id`` selects; DATETIME columns
            as ISO 8601 strings), or ``None`` for an unknown or
            soft-deleted note.
        """
        notes_db, _ = self._ensure_reading_dbs()
        row = notes_db.get_note_by_id(note_id)
        return _wire_safe(row) if row is not None else None

    def library_page(self, *, page: int = 1, per_page: int = 25) -> dict[str, Any]:
        """List active (non-deleted, non-trashed) media newest-first.

        Args:
            page: 1-based page number.
            per_page: Page size (the route layer caps this).

        Returns:
            ``{"items": [{media_id, title, media_type, ingestion_date,
            last_modified}], "page": n, "has_more": bool}`` — ``has_more``
            derives from ``search_media_db``'s total-matches count.
        """
        _, media_db = self._ensure_reading_dbs()
        # search_media_db returns ``(results_list, total_matches)`` — a
        # tuple, not a dict; extraction adapts to that shape while the
        # wire shape above stays fixed.
        results, total = media_db.search_media_db(
            search_query=None,
            page=page,
            results_per_page=per_page,
            include_trash=False,
            include_deleted=False,
        )
        items = [_wire_safe(self._library_item(row)) for row in results]
        has_more = page * per_page < int(total)
        return {"items": items, "page": page, "has_more": has_more}

    def library_detail(self, media_id: int) -> dict[str, Any] | None:
        """Return one media row joined with its active transcripts.

        Args:
            media_id: Media id.

        Returns:
            ``{media_id, title, media_type, ...remaining row fields,
            transcripts: [...]}`` — the base wire projection plus every
            other scalar row field passed through, plus the
            ``get_media_transcripts`` list; ``None`` when no active row
            matches.
        """
        _, media_db = self._ensure_reading_dbs()
        results, _total = media_db.search_media_db(
            search_query=None,
            media_ids_filter=[int(media_id)],
            page=1,
            results_per_page=1,
            include_trash=False,
            include_deleted=False,
        )
        if not results:
            return None
        row = results[0]
        item = self._library_item(row)
        for key, value in row.items():
            if key not in ("id", "title", "type", "ingestion_date", "last_modified"):
                item[key] = value
        item["transcripts"] = _wire_safe(
            get_media_transcripts(media_db, int(media_id))
        )
        return _wire_safe(item)

    @staticmethod
    def _library_item(row: dict[str, Any]) -> dict[str, Any]:
        """Project one ``search_media_db`` row onto the mobile wire keys."""
        return {
            "media_id": row["id"],
            "title": row["title"],
            "media_type": row["type"],
            "ingestion_date": row.get("ingestion_date"),
            "last_modified": row.get("last_modified"),
        }

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
          session after the lazy restore attempt a restart-surviving id
          triggers);
        - a turn already pending or a run still active for the session
          (``MobileSendRefused``) — the engine's own admission rule is one
          run per session, so this mirrors a genuine pre-submit refusal.

        Raises:
            KeyError: Unknown ``session_id`` (after the restore attempt).
            MobileSendRefused: A turn is already in flight for the session.
        """
        engine = await self._ensure_started()
        # Cheap existence probe: raises KeyError for an unknown session
        # without materializing messages — after one lazy-restore attempt
        # (restart rehydration), so a persisted conversation id sends too.
        try:
            engine.store.message_count(session_id)
        except KeyError:
            if not await self._restore_persisted_session(session_id):
                raise
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
                # Leave ``pending_terminal`` set: ``_turn_done`` owns the
                # crash path's terminal emit and races the pump (which may
                # observe the FAILED transition first) under the same
                # exactly-once flag.
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
        # Outside the lock: clear this turn's admission slot BEFORE the
        # terminal event is observable. The event fan-out is synchronous
        # (``put_nowait``), so a subscriber reacting to ``run_completed``
        # — the phone re-enabling its composer, or a back-to-back send —
        # runs before this task's done-callback if the announce stays
        # inside the lock/turn lifetime, and the send is refused with
        # "a turn is already in flight" in that window.
        self._release_turn_slot(session_id, asyncio.current_task())
        self._announce_run_terminal(engine, session_id)

    def _turn_done(self, session_id: str, task: asyncio.Task[None]) -> None:
        """Reap one finished turn task; convert its failure into events."""
        self._release_turn_slot(session_id, task)
        if task.cancelled():
            # Shutdown deliberately killed the turn: no terminal event, and
            # the flag dies with the closing service.
            self._state.pending_terminal.discard(session_id)
            return
        error = task.exception()
        if error is None:
            return
        # The turn died mid-run (engine-side failure). The phone already
        # holds its 202: announce a terminal event so the composer
        # re-enables and the resync shows the persisted reality. The
        # ``pending_terminal`` flag makes this exactly-once with the pump,
        # which may have observed the FAILED transition first.
        logger.error(
            "mobile turn failed session_id=%s type=%s",
            session_id,
            type(error).__name__,
        )
        if session_id in self._state.pending_terminal:
            self._state.pending_terminal.discard(session_id)
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
        """Pending approval rounds across both surfaces (ADR-236).

        Mobile rounds (this service's engine) gain ``surface: "mobile"``;
        desktop rounds (relayed from served terminal children) carry
        ``surface: "desktop"`` and their owning ``child_id``. Order:
        mobile first, then desktop (insertion order).
        """
        mobile = [
            {**payload, "surface": "mobile"}
            for payload in self._state.pending_approvals.values()
        ]
        if self._relay_broker is not None:
            for child_id in {key[0] for key in self._desktop_approvals}:
                try:
                    events = self._relay_broker.console_events(child_id)
                    if (
                        events is not self._relay_observed_buffers.get(child_id)
                        or events.read(0).reset_required
                    ):
                        self._drop_child_rounds(child_id, "projection_unavailable")
                except ControlProtocolError:
                    self._drop_child_rounds(child_id, "child_unavailable")
        desktop = [dict(entry) for entry in self._desktop_approvals.values()]
        return mobile + desktop

    async def decide_approval(
        self,
        round_id: str,
        decisions: dict[str, str],
        *,
        child_id: str | None = None,
    ) -> bool:
        """Resolve one pending round on whichever surface owns it.

        Args:
            round_id: The captured round's id.
            decisions: Verdict-key (``call_id`` or ``llm_name``) to
                decision string (``approve_once`` / ``approve_session`` /
                ``always_allow`` / ``deny``).
            child_id: Owning desktop child, for cross-child round_id
                collisions (round ids are child-local). An explicit id
                addresses exactly that child's round; without one, a
                round_id owned by exactly one child routes there, and a
                collision routes to the oldest-listed entry
                (deterministic; the wire card carries the ``child_id``
                for exact addressing).

        Returns:
            True when the round was known and the decision was delivered;
            False for an unknown round (nothing was resolved), and — for
            desktop rounds — when the owning child is dead, unknown, or
            failed to acknowledge within the routing budget (the round is
            then dropped from the list; the child's own timeout/deny
            semantics govern, ADR-236).
        """
        key = self._desktop_round_key(round_id, child_id)
        if key is not None:
            return await self._decide_desktop_round(key, decisions)
        if round_id not in self._state.pending_approvals:
            logger.info(
                "mobile approval decision for unknown round_id=%s", round_id
            )
            return False
        engine = self._require_engine()
        engine.controller.resolve_pending_approval(
            dict(decisions), round_id=round_id
        )
        logger.info(
            "mobile approval decision delivered round_id=%s calls=%d",
            round_id,
            len(decisions),
        )
        return True

    def _desktop_round_key(
        self, round_id: str, child_id: str | None
    ) -> tuple[str, str] | None:
        """Locate the desktop entry one decision addresses (loop thread).

        ``None`` means no desktop round matches — the caller falls through
        to the mobile-round surface.
        """
        if child_id is not None:
            key = (child_id, round_id)
            return key if key in self._desktop_approvals else None
        matches = [key for key in self._desktop_approvals if key[1] == round_id]
        if len(matches) > 1:
            logger.info(
                "mobile desktop decision round_id=%s ambiguous across %d "
                "children; routing to the oldest-listed entry",
                round_id,
                len(matches),
            )
        return matches[0] if matches else None

    # ------------------------------------------------------------------ #
    # Desktop approval relay (ADR-236, Phase 2c)                          #
    # ------------------------------------------------------------------ #

    def attach_relay(
        self, broker: Any, *, child_ids: Callable[[], Iterable[str]]
    ) -> None:
        """Observe child lifecycle projections without owning transport draining.

        Each observer has a cursor. Lost projections clear displayed decision
        authority until a native snapshot reconciles the child.
        """
        if self._closed or self._relay_task is not None:
            return
        self._relay_broker = broker
        self._relay_child_ids = child_ids
        self._relay_task = asyncio.create_task(
            self._relay_supervisor(), name="mobile-relay-supervisor"
        )
        logger.info("mobile desktop approval relay attached")

    async def _relay_supervisor(self) -> None:
        """Keep per-child drain tasks aligned with ``child_ids()`` (loop).

        Every scan cycle: spawn drains for new children, stop drains and
        fail rounds closed for children that disappeared from the source,
        and probe children that still hold desktop rounds — a broker-level
        disconnect (the child process died while its browser binding is
        being torn down) also fails its rounds closed. The broker exposes
        no disconnect callback; ``wait_connected`` is its liveness read.
        """
        broker = self._relay_broker
        child_ids = self._relay_child_ids
        assert broker is not None and child_ids is not None
        drains: dict[str, asyncio.Task[None]] = {}
        try:
            while True:
                await asyncio.sleep(_RELAY_SCAN_INTERVAL_SECONDS)
                for child_id, task in list(drains.items()):
                    if task.done():
                        drains.pop(child_id, None)
                        self._relay_observed_buffers.pop(child_id, None)
                try:
                    live = {str(value) for value in child_ids()}
                except Exception:  # noqa: BLE001 - keep draining what we have
                    logger.warning("mobile relay child source failed", exc_info=True)
                    live = set(drains)
                for child_id in sorted(live - drains.keys()):
                    drains[child_id] = asyncio.create_task(
                        self._drain_relay_child(broker, child_id),
                        name=f"mobile-relay-drain-{child_id}",
                    )
                for child_id in [known for known in drains if known not in live]:
                    drains.pop(child_id).cancel()
                    self._drop_child_rounds(child_id, "child_unbound")
                    self._relay_observed_buffers.pop(child_id, None)
                owners = {
                    str(entry.get("child_id") or "")
                    for entry in self._desktop_approvals.values()
                }
                owners.discard("")
                for child_id in owners:
                    try:
                        await asyncio.wait_for(
                            broker.wait_connected(
                                child_id, timeout=_RELAY_PROBE_TIMEOUT_SECONDS
                            ),
                            _RELAY_PROBE_TIMEOUT_SECONDS * 3,
                        )
                    except (ControlProtocolError, TimeoutError):
                        self._drop_child_rounds(child_id, "child_disconnected")
        except asyncio.CancelledError:
            raise
        finally:
            for task in drains.values():
                task.cancel()
            if drains:
                await asyncio.gather(*drains.values(), return_exceptions=True)
            self._desktop_approvals.clear()
            self._relay_observed_buffers.clear()

    async def _drain_relay_child(self, broker: Any, child_id: str) -> None:
        """Observe one child's bounded history; never consume another view."""
        sequence = 0
        observed = None
        while True:
            try:
                events = broker.console_events(child_id)
                if observed is not None and observed is not events:
                    self._drop_child_rounds(child_id, "child_restarted")
                    sequence = 0
                observed = events
                self._relay_observed_buffers[child_id] = events
                batch = events.read(sequence)
                if batch.reset_required:
                    self._drop_child_rounds(child_id, "projection_reset_required")
                    if batch.sequence != sequence:
                        self._emit(
                            "reset", "", {"child_id": child_id, "reset_required": True}
                        )
                else:
                    for event in batch.events:
                        self._handle_relay_event(
                            child_id,
                            SimpleNamespace(
                                message_type=event["kind"], payload=event["payload"]
                            ),
                        )
                sequence = batch.sequence
                await asyncio.wait_for(
                    events.wait_changed(sequence), _RELAY_DRAIN_TIMEOUT_SECONDS
                )
            except TimeoutError:
                continue
            except ControlProtocolError:
                self._drop_child_rounds(child_id, "child_unavailable")
                self._relay_observed_buffers.pop(child_id, None)
                return

    def _handle_relay_event(self, child_id: str, message: Any) -> None:
        """Fold one relayed child event into the desktop surfaces."""
        message_type = message.message_type
        payload = dict(message.payload)
        if message_type == "relay.approval.raised":
            round_id = str(payload.get("round_id") or "")
            if not round_id:
                return
            key = (child_id, round_id)
            self._desktop_approvals[key] = {
                **payload,
                "child_id": child_id,
                "surface": "desktop",
            }
            self._emit(
                "approval_requested",
                str(payload.get("session_id") or ""),
                dict(self._desktop_approvals[key]),
            )
            logger.info(
                "mobile desktop approval mounted round_id=%s child_id=%s",
                round_id,
                child_id,
            )
        elif message_type == "relay.approval.cleared":
            round_id = str(payload.get("round_id") or "")
            if (
                self._desktop_approvals.pop((child_id, round_id), None)
                is not None
            ):
                logger.info(
                    "mobile desktop approval cleared round_id=%s child_id=%s",
                    round_id,
                    child_id,
                )
        elif message_type == "relay.run.status":
            session_id = str(payload.get("session_id") or "")
            self._emit(
                "run_status",
                session_id,
                {
                    "status": str(payload.get("status") or ""),
                    "surface": "desktop",
                },
            )
            logger.debug(
                "mobile desktop run status child_id=%s session_id=%s",
                child_id,
                session_id,
            )
        elif message_type == "canvas.events":
            # Production canvas delivery is pull-based (canvas.events.request
            # through broker.request, answered by the child) and never reads
            # this queue; a pushed canvas frame is drained and dropped here
            # only so the bounded queue cannot backpressure the connection.
            logger.debug(
                "mobile relay drained canvas push frame child_id=%s", child_id
            )

    def _drop_child_rounds(self, child_id: str, reason: str) -> None:
        """Fail every desktop round of one child closed (loop thread)."""
        stale = [
            key
            for key, entry in self._desktop_approvals.items()
            if entry.get("child_id") == child_id
        ]
        for key in stale:
            del self._desktop_approvals[key]
        if stale:
            logger.info(
                "mobile desktop rounds dropped child_id=%s reason=%s count=%d",
                child_id,
                reason,
                len(stale),
            )

    async def _decide_desktop_round(
        self, key: tuple[str, str], decisions: dict[str, str]
    ) -> bool:
        """Route one phone decision to the owning child (ADR-236).

        ``relay.decide.request`` → ``relay.decide.response`` over the
        broker with a 2 s budget. Delivered → True and the round clears.
        Anything else (child not connected, deadline exceeded, child
        answered ``delivered: false`` — the round already ended there)
        drops the round and returns False: the desktop child's own
        timeout/deny semantics own the round from that point.
        """
        broker = self._relay_broker
        entry = self._desktop_approvals.get(key)
        if broker is None or entry is None:
            return False
        child_id = key[0]
        try:
            events = broker.console_events(child_id)
            if (
                events is not self._relay_observed_buffers.get(child_id)
                or events.read(0).reset_required
            ):
                self._drop_child_rounds(child_id, "projection_unavailable")
                return False
        except ControlProtocolError:
            self._drop_child_rounds(child_id, "child_unavailable")
            return False
        round_id = key[1]
        try:
            response = await broker.request(
                child_id,
                "relay.decide.request",
                {"round_id": round_id, "decisions": dict(decisions)},
                timeout=_RELAY_DECIDE_TIMEOUT_SECONDS,
            )
            delivered = (
                response.payload.get("round_id") == round_id
                and response.payload.get("delivered") is True
            )
        except (ControlProtocolError, TimeoutError) as error:
            delivered = False
            logger.info(
                "mobile desktop decision failed round_id=%s code=%s",
                round_id,
                getattr(error, "code", type(error).__name__),
            )
        self._desktop_approvals.pop(key, None)
        return delivered

    # ------------------------------------------------------------------ #
    # Events                                                              #
    # ------------------------------------------------------------------ #

    async def events(self) -> AsyncIterator[MobileEvent]:
        """Observe bounded projections; overflow stays reset until resubscription."""
        await self._ensure_started()
        queue: asyncio.Queue[MobileEvent] = asyncio.Queue(
            maxsize=DEFAULT_MAX_QUEUED_EVENTS
        )
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)
            self._reset_subscribers.discard(queue)

    def _emit(self, type_: str, session_id: str, payload: dict[str, Any]) -> None:
        """Fan one event out to every subscriber (loop thread only)."""
        self._seq += 1
        event = MobileEvent(
            type=type_, session_id=session_id, payload=payload, seq=self._seq
        )
        for queue in list(self._subscribers):
            if queue in self._reset_subscribers:
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(
                    MobileEvent(
                        type="reset",
                        session_id="",
                        payload={"reset_required": True},
                        seq=self._seq,
                    )
                )
                self._reset_subscribers.add(queue)
        logger.debug(
            "mobile event type=%s session_id=%s seq=%d",
            type_,
            session_id,
            event.seq,
        )

    async def _pump(self) -> None:
        """Delta pump: one scoped scan every 0.5 s.

        Each tick polls only the sessions in scope (an active run, a
        scheduled turn's admission slot, or a pending — non-terminal —
        stream track; see ``_poll_once``); a settled idle session is left
        untouched until a new send re-scopes it, and an in-scope session
        whose fingerprint is unchanged skips the transcript materialization
        (``_scan_stream``).
        """
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
        if self._turn_slot_owned(session_id):
            # A scheduled turn still owns the admission slot: hold the
            # announcement. The owning turn announces after releasing the
            # slot (or its done-callback emits on the crash path), so no
            # subscriber can observe a terminal event while sends for the
            # session are still refused.
            return
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
        """One pump tick: run-status transitions + message deltas.

        Scoped (#3128 minor 4) — no forever-scan of every session: only
        sessions with an active run, a scheduled turn's admission slot
        (``turns_in_flight`` or a held session lock), or a pending
        (non-terminal) stream track are scanned. A settled idle session
        cannot change unseen — every transcript mutation on this surface
        rides a send, and a send's turn slot re-scopes its session before
        any content can move — so dropping it from the tick is safe.
        """
        live_ids = {session.id for session in engine.store.sessions()}
        candidates: set[str] = {
            session_id
            for session_id, run_state in engine.controller.run_states().items()
            if session_id in live_ids
            and run_state.status not in _SEND_ALLOWED_STATUSES
        }
        candidates.update(
            session_id
            for session_id in self._state.turns_in_flight
            if session_id in live_ids
        )
        candidates.update(
            session_id
            for session_id, lock in self._session_locks.items()
            if lock.locked() and session_id in live_ids
        )
        candidates.update(
            session_id
            for session_id, track in self._state.stream_track.items()
            if track.status not in _TERMINAL_MESSAGE_STATUSES
            and session_id in live_ids
        )
        for session_id in candidates:
            state = engine.controller.run_state_for(session_id)
            status = state.status
            active = status not in _SEND_ALLOWED_STATUSES
            self._scan_stream(
                engine,
                session_id,
                # A scheduled turn's session stays "live" for scanning even
                # when the engine status already flipped terminal mid-return
                # of ``submit_draft``: the silent first-observation baseline
                # must not swallow the content the owning turn has not
                # flushed yet (fast runs landing between ticks otherwise
                # finish with zero deltas).
                active=active or self._turn_slot_owned(session_id),
            )
            previous = self._state.run_status.get(session_id, ConsoleRunStatus.IDLE)
            if status != previous:
                was_active = previous not in _SEND_ALLOWED_STATUSES
                if status in (
                    ConsoleRunStatus.COMPLETED,
                    ConsoleRunStatus.FAILED,
                    ConsoleRunStatus.STOPPED,
                    ConsoleRunStatus.BLOCKED,
                ) and not self._turn_slot_owned(session_id):
                    # Observed the transition (or its residue) before the
                    # owning turn released its admission slot: hold the
                    # announcement — the turn announces post-release (its
                    # done-callback emits on the crash path) so no
                    # subscriber reacts to a terminal event while sends
                    # for the session are still refused. With the slot
                    # free this emitter wins, and the turn's announce
                    # becomes a no-op via the pending flag.
                    self._state.pending_terminal.discard(session_id)
                    self._emit_terminal_events(session_id, status)
                elif active and not was_active:
                    self._emit(
                        "run_started", session_id, {"status": status.value}
                    )
            self._state.run_status[session_id] = status

    def _release_turn_slot(
        self, session_id: str, task: asyncio.Task[None]
    ) -> None:
        """Clear one finished turn's admission bookkeeping.

        Identity-guarded: when a newer turn already owns the session's
        slot (its task replaced ``_turn_tasks[session_id]``), this turn's
        cleanup must not drop the newer turn's in-flight marker.
        """
        if self._turn_tasks.get(session_id) is not task:
            return
        self._turn_tasks.pop(session_id, None)
        self._state.turns_in_flight.discard(session_id)

    def _turn_slot_owned(self, session_id: str) -> bool:
        """Whether a scheduled turn still owns the session's admission slot."""
        lock = self._session_locks.get(session_id)
        return session_id in self._state.turns_in_flight or (
            lock is not None and lock.locked()
        )

    def _scan_stream(
        self,
        engine: _Engine,
        session_id: str,
        *,
        active: bool,
        flush: bool = False,
    ) -> None:
        """Emit deltas beyond the last offset and terminal ``message_done``.

        Fingerprint-gated (#3128 minor 3): before materializing, the store's
        O(1) ``session_fingerprint`` seam is checked against the last
        MATERIALIZING scan; when the tracked message already ended and the
        fingerprint is unchanged — count and last visible length both
        steady, so no delta, no new message, and no pending status
        transition can be waiting — the snapshot build is skipped. A
        NON-terminal track always materializes: a terminal status flip can
        land with no content change (fingerprint-invisible) and must be
        observed to emit ``message_done``. ``flush`` (the send-time flush)
        always materializes.

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
        track = self._state.stream_track.get(session_id)
        try:
            fingerprint: tuple[int, int] | None = (
                engine.store.session_fingerprint(session_id)
            )
        except KeyError:
            # Fall through: the materializing read below raises the same
            # KeyError for an unknown session (identical behavior).
            fingerprint = None
        if (
            not flush
            and track is not None
            and track.status in _TERMINAL_MESSAGE_STATUSES
            and self._state.scan_fingerprints.get(session_id) == fingerprint
        ):
            return
        messages = engine.store.messages_for_session(session_id)
        self._state.scan_fingerprints[session_id] = fingerprint
        assistant = None
        for message in reversed(messages):
            if getattr(message.role, "value", message.role) == "assistant":
                assistant = message
                break
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
            baselined = track is None and not active and not flush
            if baselined:
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
            if status in _TERMINAL_MESSAGE_STATUSES and not baselined:
                # The message was live-delivered (the send-time flush, or a
                # pump tick on a turn that has not flushed yet): a terminal
                # message status arrives with its content. Exactly-once with
                # the flush: the stored track status makes the other path's
                # transition guard a no-op.
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
        quiesce + close, then ``AgentRunsDB.close()``. The reading DBs
        (Phase 2b) close first — they may be open even when the engine
        never was.
        """
        if self._closed:
            return
        self._closed = True
        for db, label in ((self._notes_db, "notes"), (self._media_db, "media")):
            if db is None:
                continue
            try:
                db.close()
            except Exception:  # noqa: BLE001 - teardown must finish regardless
                logger.exception("mobile reading %s DB close failed", label)
        self._notes_db = None
        self._media_db = None
        relay = self._relay_task
        if relay is not None:
            relay.cancel()
            try:
                await relay
            except asyncio.CancelledError:
                pass
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
        physically_idle = not any(not task.done() for task in turn_tasks)
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
            for session in engine.store.sessions():
                conversation_id = engine.controller.conversation_id_for_session(
                    session.id
                )
                physically_idle &= await asyncio.wait_for(
                    engine.bridge.await_fleet_terminal(conversation_id), timeout=10
                )
        except Exception:  # noqa: BLE001 - teardown must finish regardless
            physically_idle = False
            logger.exception("mobile controller shutdown failed")
        try:
            db = getattr(getattr(engine.store, "persistence", None), "db", None)
            if db is not None:
                with db.quiesce_connections(timeout_seconds=5):
                    pass
                db.close()
        except Exception:  # noqa: BLE001 - teardown must finish regardless
            physically_idle = False
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
                physically_idle = False
                logger.exception("mobile database close failed")
        if physically_idle:
            engine.agent_runs_db._execution_owner.close()
        logger.info("mobile engine shut down")
