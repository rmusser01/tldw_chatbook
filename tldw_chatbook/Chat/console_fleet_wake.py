"""Auto-wake: a finished background sub-agent re-invokes its supervisor.

PR 3a-2 Task 5 (spec §3 invariant 5, corrected 2026-08-11). A fleet
SURVIVOR -- a child that outlived its spawning turn (PR 3a-1) -- settles
on its own daemon thread with its result durable ONLY in
``agent_runs.result`` (Task 1 A3). Before this module, that result sat
there until the user happened to return to the right conversation and
send another message; this module makes the completion wake the
supervisor so it can act.

The delivery contract (every piece is load-bearing):

- **A wake send is never user input.** It writes NO USER transcript row
  and never clears the composer; the transcript shows a SYSTEM-class row
  whose ``MessageMetadata.origin`` is ``"agent_wake"``. The model payload
  carries the notice as a machine-labelled trailing ``user``-role entry
  appended payload-only at the submit seam (see ``submit_draft``'s
  ``AGENT_WAKE`` branch) -- NOT via ``turn_bundle_block`` and NOT via the
  system-prompt fold; the decision record:

  1. ``turn_bundle_block`` is applied only inside
     ``ConsoleAgentBridge.run_reply`` (the agent path); a wake turn taken
     on the plain-provider path (agent runtime toggled off between spawn
     and settle) would deliver NOTHING -- an empty turn whose payload
     ends on an assistant row, which strict providers treat as a prefill.
  2. With the USER echo suppressed, the bundle block's "append to the
     LAST user entry" rule would splice machine text INTO the user's own
     previous utterance -- structurally the exact reads-as-user-input
     failure invariant 5 forbids, and a silent edit of history under the
     payload fingerprint.
  3. The system-prompt fold (task-1531) puts text in the
     instruction-authority slot; a completion result is event data, not
     operator instruction, and the fold too leaves no trailing user-role
     entry for strict providers (task-427's own lesson).
  4. A payload-only trailing user-role entry works identically on BOTH
     the agent and plain paths, ends the payload the way providers
     require, and matches the spec's named precedent (Claude Code's own
     injected task notifications). The entry's text is explicitly
     labelled machine-injected / not-approval; the durable transcript
     never shows a USER row.

- **Exactly-once, coalesced.** The in-memory pending registry (spec
  invariant 3: events are the hot path) holds every undelivered settle
  per conversation; the durable ``FLEET_UNSEEN`` mark is the undelivered
  bit that survives restart. One wake bundles ALL of a conversation's
  undelivered completions. Delivered state is committed ONLY after the
  wake turn was actually accepted: the delivered run ids leave the
  registry and -- only when nothing undelivered remains -- the mark is
  cleared through the named seam (``clear_fleet_unseen_completion``).
  A refused wake changes nothing (retried later); a child settling
  DURING a wake turn joins the registry and rides the NEXT wake.

- **User wins ties.** A wake defers whenever
  ``controller.wake_user_priority_probe`` reports a user claim -- the
  screen wires it to "the Console composer holds a non-empty draft".
  The composer clears only once a manual send is ACCEPTED
  (``_notify_submission_accepted``), so the probe also covers the
  dispatch gap between pressing Send and the run state turning busy; a
  probe that raises defers too (user wins on uncertainty). With no probe
  wired (headless/tests) there is no user claim to lose to.

- **Scheduling.** Fires on the drain (when a live controller resolves
  the conversation to an open session and ``send_refusal_copy`` -- the
  same gate a manual send passes: per-session run state, queue
  ownership, ``max_parallel_runs`` -- allows it); otherwise the pending
  entry waits and is retried on every terminal run-state transition and
  at queue-chain end. With no controller (Console unmounted) the durable
  mark IS the staged wake: the next Console mount calls
  ``seed_from_marks`` BEFORE the first tab sync can view-clear the
  active conversation's mark, reconstructing the undelivered set from
  ``agent_runs`` (``undelivered_survivor_runs``). Deliveries are
  serialized -- one wake in flight at a time, app-wide.

- **No new authority.** A woken turn is a normal turn under every
  existing gate and cap: approval cards, risk-tag ask-floors,
  ``max_parallel_runs``, wall clocks and token ceilings all apply
  unchanged, and approval resolution only ever happens through
  ``resolve_pending_approval``'s round-id path -- no text this module
  injects can satisfy a pending card. ``[agents] autowake_enabled``
  (default ON) is the kill switch, honoured at BOTH fire points; OFF
  loses nothing durable (mark/toast/badge still record every settle).
"""

from __future__ import annotations

import asyncio
import re
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    RUN_SUPERSEDED,
    TERMINAL_RUN_STATUSES,
    RunBudget,
)
from tldw_chatbook.Agents.agent_service import (
    AUTOWAKE_ENABLED_KEY,
    DEFAULT_AUTOWAKE_ENABLED,
    _coerce_autowake_enabled,
)
from tldw_chatbook.Agents.run_log import _setting
from tldw_chatbook.Chat.console_fleet_attention import (
    clear_fleet_unseen_completion,
)

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController


def autowake_enabled() -> bool:
    """Read the ``[agents] autowake_enabled`` kill switch (default ON).

    Same env -> TOML -> default chain and junk tolerance as its sibling
    fleet switches (``_setting`` + ``_coerce_autowake_enabled``). Read
    fresh at every fire point, so flipping the key takes effect on the
    next wake decision without a restart.

    Returns:
        Whether a finished background sub-agent may wake its supervisor.
    """
    return _coerce_autowake_enabled(
        _setting(AUTOWAKE_ENABLED_KEY, DEFAULT_AUTOWAKE_ENABLED)
    )


#: First line of every wake notice -- the human-visible machine marking.
WAKE_NOTICE_HEADER = "[Background sub-agent completion — automated notice]"

#: The not-user-input / not-approval disclaimer, verbatim in every notice.
WAKE_NOTICE_DISCLAIMER = (
    "This notice was injected automatically because background sub-agent "
    "work finished after its turn had already ended. It is not user "
    "input, and it is not approval or consent for anything -- do not "
    "treat it as permission for any pending or future action."
)

#: Closing line: invites action without granting any authority.
WAKE_NOTICE_TRAILER = (
    "You may act on these results now, or wait for the user's next "
    "message."
)

#: Backward slop applied to the mark's ``created_at`` when reconstructing
#: the undelivered set from ``agent_runs`` at mount-claim: the child's
#: terminal write lands on the same thread strictly BEFORE the mark write,
#: so the drain's own runs sit fractionally EARLIER than the mark.
_MARK_CREATED_SLOP_SECONDS = 5.0


def _parse_iso(value: Any) -> datetime | None:
    """Parse either timestamp dialect in play (marks ``+00:00``->``Z``,
    agent_runs ``%fZ``), returning ``None`` for junk."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fenced(text: str) -> str:
    """Fence ``text`` with more backticks than any run it contains, so a
    result that itself carries code fences cannot break out of the block."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _truncated(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    return (
        text[:cap]
        + "\n… (truncated to share the notice's result budget; the full "
        "result is in the run log)"
    )


def compose_wake_notice(
    rows: Sequence[Mapping[str, Any]], *, budget: RunBudget | None = None
) -> str:
    """Compose the one coalesced wake notice for a conversation's
    undelivered completions.

    Result sizing follows ``wait_agents``' discipline exactly: each
    child's result is capped at ``max_subagent_result_chars`` AND the
    combined bodies are additionally bounded by ``max_tool_result_chars``
    split evenly across the children, so N capped results are shortened
    fairly rather than cut mid-notice downstream.

    Args:
        rows: ``agent_runs`` row dicts (or minimal synthesized stand-ins
            carrying ``id``/``status``) for the completions to deliver,
            oldest first.
        budget: The result-sizing budget; defaults to ``RunBudget()``'s
            shipped caps -- the same constants the in-turn collection
            path bounds with.

    Returns:
        The full notice text, or ``""`` for no rows.
    """
    if not rows:
        return ""
    budget = budget or RunBudget()
    headers: list[str] = []
    bodies: list[str] = []
    for row in rows:
        run_id = str(row.get("id") or "?")
        agent = str(row.get("agent_definition") or "").strip() or "sub-agent"
        status = str(row.get("status") or "done")
        task = str(row.get("task") or "").strip()
        header = f"[{run_id}] {agent} — {status}"
        if task:
            shortened = task if len(task) <= 120 else task[:117] + "..."
            header += f" — task: {shortened}"
        result = row.get("result")
        body = (
            str(result)
            if result
            else f"(no result recorded; the run ended '{status}')"
        )
        headers.append(header)
        bodies.append(body)
    per_child_cap = budget.max_subagent_result_chars
    if budget.max_tool_result_chars > 0:
        fixed = (
            len(WAKE_NOTICE_HEADER)
            + len(WAKE_NOTICE_DISCLAIMER)
            + len(WAKE_NOTICE_TRAILER)
            + sum(len(header) + 1 for header in headers)
            # fences + joining blank lines, over-reserved by a constant
            # per entry (harmless; under-reserving is what must not happen)
            + 24 * len(headers)
        )
        per_child_cap = min(
            per_child_cap,
            max(200, (budget.max_tool_result_chars - fixed) // len(headers)),
        )
    blocks = [
        f"{header}\n{_fenced(_truncated(body, per_child_cap))}"
        for header, body in zip(headers, bodies)
    ]
    count = len(rows)
    plural = "s" if count != 1 else ""
    return "\n".join(
        [
            WAKE_NOTICE_HEADER,
            WAKE_NOTICE_DISCLAIMER,
            "",
            f"{count} background sub-agent{plural} finished after the turn "
            "ended. Results:",
            "",
            "\n\n".join(blocks),
            "",
            WAKE_NOTICE_TRAILER,
        ]
    )


def undelivered_survivor_runs(
    runs_db: Any,
    conversation_id: str,
    mark_created_at: str | None,
    *,
    limit: int = 50,
) -> list[dict]:
    """Reconstruct a marked conversation's undelivered survivor runs from
    ``agent_runs`` alone (the mount-claim path; nothing in-memory
    survived).

    Two filters, both stated inferences with their boundaries:

    - **survivor, not within-turn**: a child's terminal ``updated_at`` at
      or after its parent primary run's own terminal ``updated_at``. A
      within-turn child settles strictly before its parent's terminal
      write by turn mechanics; the ``>=`` keeps the restart-reconcile
      case, where one sweep stamps parent and child together. Boundary: a
      parent later re-stamped by ``supersede_run_tree`` (user retried the
      turn) moves its ``updated_at`` forward and can shadow a genuine
      pre-supersede survivor here -- that survivor's live drain already
      delivered (or toasted) it, so the miss is bounded to the
      staged-claim path and accepted.
    - **undelivered, not already handed over**: terminal ``updated_at``
      no earlier than the mark's stable ``created_at`` (minus
      ``_MARK_CREATED_SLOP_SECONDS`` -- the drain's own terminal writes
      land fractionally BEFORE the mark's). An earlier wake's delivery
      cleared the mark, so anything older than the current mark's birth
      was delivered under a previous one.

    Args:
        runs_db: The ``AgentRunsDB`` (or compatible double).
        conversation_id: The marked conversation.
        mark_created_at: The ``FLEET_UNSEEN`` mark row's ``created_at``;
            ``None`` applies no since-when bound (best effort).
        limit: Newest-runs window to inspect.

    Returns:
        Matching run row dicts, oldest first (reading order).
    """
    try:
        rows = runs_db.list_runs(
            conversation_id, agent_kind="subagent", limit=limit
        )
    except Exception:  # noqa: BLE001 -- a claim must never break a mount
        logger.opt(exception=True).warning(
            "undelivered-survivor listing failed for {conversation_id}",
            conversation_id=conversation_id,
        )
        return []
    threshold = _parse_iso(mark_created_at)
    if threshold is not None:
        threshold -= timedelta(seconds=_MARK_CREATED_SLOP_SECONDS)
    parents: dict[str, Any] = {}
    matched: list[dict] = []
    for row in rows:
        status = str(row.get("status") or "")
        if status not in TERMINAL_RUN_STATUSES or status == RUN_SUPERSEDED:
            continue
        settled_at = _parse_iso(row.get("updated_at"))
        if settled_at is None:
            continue
        if threshold is not None and settled_at < threshold:
            continue
        parent_id = str(row.get("parent_run_id") or "")
        if not parent_id:
            continue
        if parent_id not in parents:
            try:
                parents[parent_id] = runs_db.get_run(parent_id)
            except Exception:  # noqa: BLE001
                parents[parent_id] = None
        parent = parents[parent_id]
        parent_at = _parse_iso((parent or {}).get("updated_at"))
        if parent_at is None or settled_at < parent_at:
            continue
        matched.append(row)
    matched.reverse()  # list_runs is newest-first; deliver oldest first
    return matched


_WAKE_AUTHORIZATION_KEY = object()


class AgentWakeAuthorization:
    """Opaque, coordinator-issued authority to cross the wake send gate.

    The queue's ``QueueGenerationAuthorization`` precedent: only the
    coordinator can mint one (module-private key), and ``submit_draft``'s
    ``AGENT_WAKE`` branch refuses anything else with ``PermissionError``
    -- no other code path can fabricate a machine-origin send.
    """

    __slots__ = ("_coordinator", "session_id")

    def __init__(self, coordinator: object, session_id: str, *, _key: object):
        if _key is not _WAKE_AUTHORIZATION_KEY:
            raise PermissionError("wake authority is coordinator-internal")
        self._coordinator = coordinator
        self.session_id = session_id

    def __repr__(self) -> str:  # pragma: no cover -- debug affordance
        return (
            "AgentWakeAuthorization("
            f"session_id={self.session_id!r}, authority=<redacted>)"
        )


class ConsoleFleetWakeCoordinator:
    """Owns wake pending state, gating, composition, and delivery.

    One per controller, constructed in ``ConsoleChatController.__init__``
    and registered on the bridge fan-out as ``"fleet-wake"`` next to the
    usage-reattach consumer (never from ``run_reply``). The drain half
    runs on the CHILD's thread (registry write + a thread-safe hop);
    everything that touches the controller runs on the loop captured at
    registration -- the app loop in production, which outlives the
    screen. Post-teardown drains find ``_shutdown_requested`` set and
    stage via the durable mark instead.
    """

    #: Fan-out registration name (also the replace key).
    NAME = "fleet-wake"

    def __init__(self, controller: "ConsoleChatController"):
        self._controller = controller
        self._app: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._registry_lock = threading.Lock()
        #: conversation_id -> {run_id: terminal status}; the hot-path
        #: undelivered set (the durable mark is the restart-proof bit).
        self._pending: dict[str, dict[str, str]] = {}
        #: Conversation currently being delivered, or None. Serializes
        #: deliveries app-wide (one wake at a time) and anchors
        #: ``authorizes``.
        self._delivering: str | None = None
        self._delivery_tasks: set[asyncio.Task] = set()

    # -- wiring ---------------------------------------------------------------

    def wire(
        self,
        *,
        app: Any | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Attach the app object (marks service / clear seam) and, when
        given or currently running, the loop deliveries hop onto.

        Args:
            app: The application object; kept if non-``None``.
            loop: Explicit loop override (tests); otherwise the running
                loop is captured when there is one.
        """
        if app is not None:
            self._app = app
        if loop is not None:
            self._loop = loop
        else:
            self.capture_loop_if_running()

    def capture_loop_if_running(self) -> None:
        """Capture the current running loop, if any (registration site)."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    # -- authority ------------------------------------------------------------

    def authorizes(
        self, authorization: Any, session_id: str
    ) -> bool:
        """Return whether ``authorization`` is this coordinator's live
        wake token for ``session_id``."""
        return bool(
            isinstance(authorization, AgentWakeAuthorization)
            and authorization._coordinator is self
            and authorization.session_id == session_id
            and self._delivering is not None
        )

    # -- inspection (tests / screen) -----------------------------------------

    def pending_conversation_ids(self) -> tuple[str, ...]:
        """Snapshot of conversations with undelivered completions."""
        with self._registry_lock:
            return tuple(self._pending)

    def has_pending(self, conversation_id: str) -> bool:
        """Whether ``conversation_id`` still owes a wake."""
        with self._registry_lock:
            return bool(self._pending.get(conversation_id))

    # -- the drain half (child thread) ---------------------------------------

    def on_fleet_drained(self, event: Any) -> None:
        """``FleetDrained`` consumer: record undelivered survivors, then
        hop to the loop for an attempt.

        Child-thread safe: registry write under the lock, no controller
        or UI access here. Children with ``run_id is None`` (died before
        ``create_run``; no row, nothing to wake on) and within-turn
        children (their own turn already delivered them) are skipped.
        Recording is deliberately NOT gated on ``autowake_enabled`` --
        the switch gates FIRING, so flipping it on later can still
        deliver what settled while it was off.
        """
        try:
            survivors = [
                child
                for child in (getattr(event, "children", ()) or ())
                if getattr(child, "settled_after_turn", False)
                and getattr(child, "run_id", None)
            ]
            if not survivors:
                return
            conversation_id = str(getattr(event, "conversation_id", "") or "")
            if not conversation_id:
                return
            with self._registry_lock:
                bucket = self._pending.setdefault(conversation_id, {})
                for child in survivors:
                    bucket[str(child.run_id)] = str(
                        getattr(child, "status", "") or "done"
                    )
            self.retry_soon()
        except Exception:  # noqa: BLE001 -- never raise into the fan-out
            logger.opt(exception=True).warning("fleet wake drain intake failed")

    # -- scheduling -----------------------------------------------------------

    def retry_soon(self) -> None:
        """Schedule a delivery attempt for every pending conversation.

        Thread-safe; the retry trigger every deferral relies on. Called
        from: the drain intake, every terminal run-state transition
        (``_set_run_state``), queue-chain end
        (``_publish_queue_chain_terminal``), the screen's composer-empty
        poke, mount-claim seeding, and each delivery's own completion.
        No loop captured (sync harness) means nothing to schedule onto.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._attempt_all)
        except RuntimeError:
            pass  # closed between check and call: staged mark still covers it

    def _attempt_all(self) -> None:
        with self._registry_lock:
            conversations = tuple(self._pending)
        for conversation_id in conversations:
            self._attempt(conversation_id)

    def _attempt(self, conversation_id: str) -> None:
        """One gating pass for one conversation; loop thread only.

        Every deferral leaves pending + mark untouched -- refusal is
        never loss. The gates, in order: kill switch, controller alive,
        one-delivery-at-a-time, an open session for the conversation,
        the manual-send gate (``send_refusal_copy``: own run state, queue
        ownership, ``max_parallel_runs``), then user-wins-ties.
        """
        if not autowake_enabled():
            return
        controller = self._controller
        if controller is None:
            return
        shutdown = getattr(controller, "_shutdown_requested", None)
        if shutdown is not None and shutdown.is_set():
            return
        if self._delivering is not None:
            return
        with self._registry_lock:
            bucket = dict(self._pending.get(conversation_id) or {})
        if not bucket:
            return
        session_id = self._resolve_session_id(conversation_id)
        if session_id is None:
            return  # no open session: the durable mark stays the staged wake
        try:
            if controller.send_refusal_copy(session_id) is not None:
                return
        except Exception:  # noqa: BLE001 -- a broken gate defers, never fires
            logger.opt(exception=True).debug("wake send gate raised; deferring")
            return
        probe = getattr(controller, "wake_user_priority_probe", None)
        if callable(probe):
            try:
                if probe(session_id):
                    return  # user wins ties
            except Exception:  # noqa: BLE001 -- user wins on uncertainty too
                logger.opt(exception=True).debug(
                    "wake user-priority probe raised; deferring"
                )
                return
        rows = self._rows_for(conversation_id, bucket)
        notice = compose_wake_notice(rows)
        if not notice:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        self._delivering = conversation_id
        authorization = AgentWakeAuthorization(
            self, session_id, _key=_WAKE_AUTHORIZATION_KEY
        )
        task = loop.create_task(
            self._deliver(
                conversation_id,
                session_id,
                tuple(bucket),
                notice,
                authorization,
            )
        )
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)

    async def _deliver(
        self,
        conversation_id: str,
        session_id: str,
        delivered_run_ids: tuple[str, ...],
        notice: str,
        authorization: AgentWakeAuthorization,
    ) -> None:
        """Run one wake turn; commit delivered state only on acceptance.

        ``submit_draft`` returns only once the turn has fully run, so
        ``accepted`` here is strictly after real acceptance -- never
        "merely scheduled". On acceptance the delivered ids leave the
        registry, and the durable mark is cleared through the named seam
        ONLY when nothing undelivered remains for the conversation (a
        child that settled during this very turn keeps the mark alive and
        rides the next wake). A refusal or raise commits nothing.
        """
        from tldw_chatbook.Chat.console_chat_models import (
            ConsoleSubmissionOrigin,
        )

        accepted = False
        try:
            result = await self._controller.submit_draft(
                notice,
                session_id=session_id,
                origin=ConsoleSubmissionOrigin.AGENT_WAKE,
                wake_authorization=authorization,
            )
            accepted = bool(getattr(result, "accepted", False))
        except Exception:  # noqa: BLE001 -- a failed wake is a retry, not a crash
            logger.opt(exception=True).warning(
                "wake delivery failed for conversation {conversation_id}",
                conversation_id=conversation_id,
            )
        finally:
            self._delivering = None
        if accepted:
            with self._registry_lock:
                bucket = self._pending.get(conversation_id)
                if bucket is not None:
                    for run_id in delivered_run_ids:
                        bucket.pop(run_id, None)
                    if not bucket:
                        self._pending.pop(conversation_id, None)
                nothing_undelivered = conversation_id not in self._pending
            if nothing_undelivered and self._app is not None:
                clear_fleet_unseen_completion(self._app, conversation_id)
        self.retry_soon()

    # -- mount claim ----------------------------------------------------------

    def seed_from_marks(self) -> int:
        """Reconstruct pending state from durable marks (mount claim).

        MUST run before the first tab sync of a fresh Console mount: the
        view-clear fires on the first sync whose ACTIVE session carries
        the mark (Task 4's stated ordering hazard), and this read is what
        turns the mark into a deliverable pending set first. Honours the
        kill switch itself (the second fire point): OFF seeds nothing and
        the marks keep driving the indicator only.

        Returns:
            How many conversations gained pending completions.
        """
        if not autowake_enabled():
            return 0
        app = self._app
        service = getattr(app, "conversation_local_marks_service", None)
        runs_db = self._runs_db()
        if service is None or runs_db is None:
            return 0
        try:
            marked = service.list_marked_conversation_ids(service.FLEET_UNSEEN)
        except Exception:  # noqa: BLE001 -- a claim must never break a mount
            logger.opt(exception=True).warning("wake mark listing failed")
            return 0
        seeded = 0
        for conversation_id in marked:
            try:
                mark = service.get_mark(conversation_id, service.FLEET_UNSEEN)
            except Exception:  # noqa: BLE001
                continue
            if mark is None:
                continue
            rows = undelivered_survivor_runs(
                runs_db, conversation_id, mark.created_at
            )
            if not rows:
                continue
            with self._registry_lock:
                bucket = self._pending.setdefault(conversation_id, {})
                for row in rows:
                    bucket.setdefault(
                        str(row.get("id")), str(row.get("status") or "done")
                    )
            seeded += 1
        return seeded

    # -- internals ------------------------------------------------------------

    def _resolve_session_id(self, conversation_id: str) -> str | None:
        store = getattr(self._controller, "store", None)
        if store is None:
            return None
        try:
            for session in store.sessions():
                if conversation_id in (
                    session.persisted_conversation_id,
                    session.id,
                ):
                    return session.id
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).debug("wake session resolution failed")
        return None

    def _runs_db(self) -> Any | None:
        bridge = getattr(self._controller, "_agent_bridge", None)
        if bridge is None:
            return None
        return getattr(bridge, "runs_db", None)

    def _rows_for(
        self, conversation_id: str, bucket: Mapping[str, str]
    ) -> list[dict]:
        """Read the pending runs' rows; synthesize an honest stand-in for
        a row that cannot be read (a wiped runs DB must not strand the
        pending entry forever)."""
        runs_db = self._runs_db()
        rows: list[dict] = []
        for run_id, status in bucket.items():
            row = None
            if runs_db is not None:
                try:
                    row = runs_db.get_run(run_id)
                except Exception:  # noqa: BLE001
                    row = None
            rows.append(
                row
                if row is not None
                else {"id": run_id, "status": status, "result": None}
            )
        rows.sort(key=lambda r: str(r.get("updated_at") or ""))
        return rows
