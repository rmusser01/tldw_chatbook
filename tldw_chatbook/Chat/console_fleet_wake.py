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
  per conversation; durability is TWO-layered: the per-run
  ``agent_runs.wake_delivered_at`` ledger
  (``AgentRunsDB.undelivered_wake_runs`` / ``mark_wake_delivered``) is
  the exact delivered/undelivered bit, and the conversation-level
  ``FLEET_UNSEEN`` mark is the cheap staging trigger + indicator the
  badge/mount-claim key off. One wake bundles ALL of a conversation's
  undelivered completions. Delivered state is committed ONLY after the
  wake turn was actually accepted: the delivered runs are stamped in the
  ledger (first-writer-wins), their ids leave the registry, and -- only
  when nothing undelivered remains -- the mark is cleared through the
  named seam (``clear_fleet_unseen_completion``). A refused wake
  changes nothing (retried later); a child settling DURING a wake turn
  joins the registry and rides the NEXT wake; a pending entry whose run
  the ledger already shows delivered (a redelivered drain after a
  restart mid-commit) is dropped at compose time, never re-announced.

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
  at queue-chain end. **task-15860: Console being unmounted is no longer
  one of the reasons to wait.** The runtime (controller + store + bridge)
  is owned by the app and outlives every ``ChatScreen``, so a survivor
  settling with nothing mounted delivers a full wake turn headlessly;
  ``_attempt`` refuses only for a DISPOSED controller (app exit). The
  durable mark remains the staged wake for what a headless delivery
  genuinely cannot reach -- a conversation with no open session, and
  anything owed across a process restart: the next Console mount calls
  ``seed_from_marks`` BEFORE the first tab sync can view-clear the
  active conversation's mark, reconstructing the undelivered set from
  the ledger (``AgentRunsDB.undelivered_wake_runs``). Deliveries are
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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from loguru import logger

from tldw_chatbook.Agents.agent_models import RunBudget, TERMINAL_RUN_STATUSES
from tldw_chatbook.Agents.agent_service import (
    AUTOWAKE_ENABLED_KEY,
    DEFAULT_AUTOWAKE_ENABLED,
    _coerce_autowake_enabled,
)
from tldw_chatbook.Agents.run_log import _setting
from tldw_chatbook.Chat.console_fleet_attention import (
    clear_fleet_unseen_completion,
    set_fleet_unseen_completion,
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
    screen. task-15860: a drain arriving with NO Console mounted now
    DELIVERS -- the runtime, its store and the fan-out all outlive the
    view, so the only teardown ``_attempt`` refuses for is a DISPOSED
    controller (app exit), where the durable mark stays the staged wake
    for the next process.
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
        #: The SESSION the in-flight delivery is running in, or None.
        #: Written and cleared in lockstep with ``_delivering`` (a
        #: conversation id is not a session id -- ``_resolve_session_id``
        #: maps between them). task-15860 Task 4 needs the session id after
        #: the fact: a Console attaching mid-delivery has to re-arm
        #: ``delivery_ui_hook``, which takes the session.
        self._delivering_session: str | None = None
        self._delivery_tasks: set[asyncio.Task] = set()
        #: task-15862: screen-wired hook fired on the loop thread the
        #: moment a delivery is scheduled (``_delivering`` already set).
        #: The screen arms its 0.2s transcript poll here -- a wake turn
        #: enters through this coordinator, never the user-send worker
        #: that normally arms the poll, so without this hook NOTHING
        #: repaints the wake turn's stream, terminal tab glyph, or
        #: composer state (the live 4+ minute freeze in PR3a-2 Task 7's
        #: findings). Best-effort: a raising hook never blocks delivery.
        self.delivery_ui_hook: Callable[[str], None] | None = None

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

    def delivering_conversation_id(self) -> str | None:
        """The conversation a wake turn is delivering into right now.

        task-15862: set synchronously in ``_attempt`` before the delivery
        task is created and cleared in ``_deliver``'s ``finally``, so it
        covers the whole wake turn. The screen reads it to keep the
        transcript poll alive through the turn and to name the wake in the
        composer's blocked-state copy.

        Returns:
            The conversation id being delivered, or ``None``.
        """
        return self._delivering

    def delivering_session_id(self) -> str | None:
        """The session a wake turn is delivering into right now.

        task-15860 Task 4: ``ConsoleRuntime.attach_view`` reads this to
        re-arm ``delivery_ui_hook`` for a Console that opened DURING a
        delivery -- the hook fires once, at delivery start, and a runtime
        that outlives the screen makes "delivery start" and "view attach"
        independent events.

        Returns:
            The session id being delivered into, or ``None``.
        """
        return self._delivering_session

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
        except Exception as exc:  # noqa: BLE001 -- never raise into the fan-out
            logger.warning(
                "fleet wake drain intake failed (exception_type={})",
                type(exc).__name__,
            )

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
        controller not DISPOSED, one-delivery-at-a-time, an open session
        for the conversation, the manual-send gate (``send_refusal_copy``:
        own run state, queue ownership, ``max_parallel_runs``), then
        user-wins-ties.

        **task-15860 (the wake-fires-headless slice): this reads
        ``_disposed``, not ``_shutdown_requested``.** The two used to be
        one signal. ``ConsoleChatController.shutdown()`` was called both
        at app exit AND from ``ChatScreen.on_unmount`` -- i.e. on every
        ordinary navigation away from Console -- so "the cancellation
        Event is set" meant either "the process is going away" or "the
        user switched tabs", and this gate could not tell them apart. It
        therefore refused every wake with no Console mounted, which is
        the limit this task exists to remove (the User Guide's "if
        Console isn't open, no wake fires").

        The lifetime landing separated them: ``leave_console()`` ends ONE
        visit (sets THAT visit's Event, which stays set between visits by
        design so every round armed during it stays denied), while
        ``begin_shutdown()`` latches ``_disposed`` for the real,
        permanent teardown. A visit that merely ended must not stop a
        wake -- the runtime, the store, the bridge fan-out and the app
        loop all outlive it (Task 0's P2 executed that they do). A
        DISPOSED controller must: its provider gateway is closed, every
        session's stream task has been cancelled and awaited, and nothing
        it produced could reach a user.

        A controller double with neither attribute is unchanged: it was
        allowed before (no ``_shutdown_requested``) and is allowed now
        (``_disposed`` defaults False).
        """
        if not autowake_enabled():
            return
        controller = self._controller
        if controller is None:
            return
        if getattr(controller, "_disposed", False):
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
        except Exception as exc:  # noqa: BLE001 -- a broken gate defers, never fires
            logger.debug(
                "wake send gate raised; deferring (exception_type={})",
                type(exc).__name__,
            )
            return
        probe = getattr(controller, "wake_user_priority_probe", None)
        if callable(probe):
            try:
                if probe(session_id):
                    return  # user wins ties
            except Exception as exc:  # noqa: BLE001 -- user wins on uncertainty too
                logger.debug(
                    "wake user-priority probe raised; deferring (exception_type={})",
                    type(exc).__name__,
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
        self._delivering_session = session_id
        authorization = AgentWakeAuthorization(
            self, session_id, _key=_WAKE_AUTHORIZATION_KEY
        )
        # task-15862: tell the screen a wake turn is starting so it can arm
        # the transcript poll. ``_delivering`` is already set, so a poll
        # beat racing the delivery task's first run cannot self-stop in the
        # gap. Never let a UI hook block the delivery itself.
        hook = self.delivery_ui_hook
        if callable(hook):
            try:
                hook(session_id)
            except Exception as exc:  # noqa: BLE001 -- UI freshness is best-effort
                logger.debug(
                    "wake delivery UI hook raised (exception_type={})",
                    type(exc).__name__,
                )
        # Qodo audit minor batch: `_delivering` must be set BEFORE the hook
        # and the task (the poll-beat race above), so a `create_task` that
        # raises -- the loop closing between the `is_closed()` check and
        # this call is the live shape -- must CLEAR the flags on its way
        # out. Leaving them set wedged every future `_attempt` in the
        # process at the `self._delivering is not None` gate: no wake could
        # ever fire again. The bucket is untouched, so the wake is
        # deferred to the next attempt, not lost.
        try:
            task = loop.create_task(
                self._deliver(
                    conversation_id,
                    session_id,
                    tuple(bucket),
                    notice,
                    authorization,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- a failed wake defers, never wedges
            self._delivering = None
            self._delivering_session = None
            logger.warning(
                "wake delivery task could not be scheduled; deferring "
                "(exception_type={})",
                type(exc).__name__,
            )
            return
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
        "merely scheduled". On acceptance the delivered runs are stamped
        in the durable per-run ledger (``mark_wake_delivered``,
        first-writer-wins -- exactly-once across restart), their ids
        leave the registry, and the durable mark is cleared through the
        named seam ONLY when nothing undelivered remains for the
        conversation (a child that settled during this very turn keeps
        the mark alive and rides the next wake). A refusal or raise
        commits nothing.
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
        except Exception as exc:  # noqa: BLE001 -- a failed wake is a retry, not a crash
            logger.warning(
                "wake delivery failed (exception_type={})",
                type(exc).__name__,
            )
        finally:
            self._delivering = None
            self._delivering_session = None
        if accepted:
            runs_db = self._runs_db()
            stamp = getattr(runs_db, "mark_wake_delivered", None)
            if callable(stamp):
                try:
                    stamp(delivered_run_ids)
                except Exception as exc:  # noqa: BLE001 -- a lost stamp risks one
                    # re-announce at a later claim, never a lost result.
                    logger.warning(
                        "wake delivery ledger stamp failed (exception_type={})",
                        type(exc).__name__,
                    )
            with self._registry_lock:
                bucket = self._pending.get(conversation_id)
                if bucket is not None:
                    for run_id in delivered_run_ids:
                        bucket.pop(run_id, None)
                    if not bucket:
                        self._pending.pop(conversation_id, None)
                nothing_undelivered = conversation_id not in self._pending
            if nothing_undelivered and self._app is not None:
                # task-15971 (the coordinator's design ruling): the mark's
                # fate at commit depends on whether the user WATCHED the
                # delivery. In view -> the result is seen; clear through
                # the named seam as always. Off view (a mounted-but-hidden
                # Console delivering, or a non-active session tab) -> the
                # delivery is real but unseen; the mark is SET so the ◈
                # badge points at the delivered result until the user
                # views it (view-clear then applies normally -- nothing
                # is pending any more).
                if self._conversation_in_view(conversation_id, session_id):
                    clear_fleet_unseen_completion(self._app, conversation_id)
                else:
                    set_fleet_unseen_completion(self._app, conversation_id)
        self.retry_soon()

    def _conversation_in_view(
        self, conversation_id: str, session_id: str
    ) -> bool:
        """Whether the delivered conversation is actually being viewed.

        Consults the screen-wired ``wake_conversation_in_view`` probe
        (task-15971). Unwired (controller doubles, the pre-screen rig)
        keeps the historical clear-on-delivery; a RAISING probe reports
        not-in-view -- fail toward the badge: a kept mark on a viewed
        conversation self-heals on the next displayed sync tick, while a
        cleared mark on an unviewed delivery is the live silent-delivery
        failure this exists to prevent.

        task-15860 Task 4: a controller owned by a ``ConsoleRuntime`` is
        never in the "unwired" case above -- attach binds the view's probe
        and detach restores ``viewless_conversation_in_view``, which
        reports NOT in view. The unwired branch survives only for
        controllers built outside the runtime (doubles, the pre-screen
        rig); if it is ever made to mean "unwatched" globally, the test
        that pins the historical clear
        (``test_an_unwired_view_probe_keeps_the_historical_clear``) is the
        one to rewrite alongside it.

        Args:
            conversation_id: The delivered conversation.
            session_id: The session the wake turn ran in.

        Returns:
            True when the user watched the delivery (or no probe is
            wired); False when it landed off-view or visibility is
            uncertain.
        """
        probe = getattr(self._controller, "wake_conversation_in_view", None)
        if not callable(probe):
            return True
        try:
            return bool(probe(conversation_id, session_id))
        except Exception as exc:  # noqa: BLE001 -- uncertainty keeps the badge
            logger.debug(
                "wake view probe raised; keeping the unseen mark "
                "(exception_type={})",
                type(exc).__name__,
            )
            return False

    # -- mount claim ----------------------------------------------------------

    def seed_from_marks(self) -> int:
        """Reconstruct pending state from the durable layers (mount claim).

        MUST run before the first tab sync of a fresh Console mount: the
        view-clear fires on the first sync whose ACTIVE session carries
        the mark (Task 4's stated ordering hazard), and this read is what
        turns the mark into a deliverable pending set first. The mark
        names WHICH conversations to claim; the per-run
        ``wake_delivered_at`` ledger (``AgentRunsDB.undelivered_wake_
        runs``) is the exact definition of WHAT is still owed -- an
        earlier wake's deliveries are stamped there and never
        re-announced, however the timestamps interleave. Honours the kill
        switch itself (the second fire point): OFF seeds nothing and the
        marks keep driving the indicator only.

        Returns:
            How many conversations gained pending completions.
        """
        if not autowake_enabled():
            return 0
        app = self._app
        service = getattr(app, "conversation_local_marks_service", None)
        runs_db = self._runs_db()
        undelivered = getattr(runs_db, "undelivered_wake_runs", None)
        if service is None or not callable(undelivered):
            return 0
        try:
            marked = service.list_marked_conversation_ids(service.FLEET_UNSEEN)
        except Exception as exc:  # noqa: BLE001 -- a claim must never break a mount
            logger.warning(
                "wake mark listing failed (exception_type={})",
                type(exc).__name__,
            )
            return 0
        seeded = 0
        for conversation_id in marked:
            try:
                rows = undelivered(conversation_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "wake ledger read failed (exception_type={})",
                    type(exc).__name__,
                )
                continue
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
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "wake session resolution failed (exception_type={})",
                type(exc).__name__,
            )
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
        pending entry forever). A run the durable ledger already shows
        wake-delivered (a redelivered drain, or a restart racing the
        in-memory commit) is DROPPED -- from the returned rows AND from
        the registry -- rather than re-announced."""
        runs_db = self._runs_db()
        rows: list[dict] = []
        stale: list[str] = []
        for run_id, status in bucket.items():
            row = None
            if runs_db is not None:
                try:
                    row = runs_db.get_run(run_id)
                except Exception:  # noqa: BLE001
                    row = None
            if row is not None and (
                str(row.get("status")) not in TERMINAL_RUN_STATUSES
            ):
                # task-15863: a pending run is terminal BY CONSTRUCTION --
                # it entered the registry from the settle hook (which
                # fires strictly after the terminal DB write; `run_child`'s
                # finally ordering) or from the durable ledger (terminal
                # statuses only). A non-terminal read here is therefore a
                # stale snapshot pinned on this thread's reused held
                # connection (observed live: a minute-old 'done' child
                # announced as 'running'). Re-read through a fresh
                # connection, which cannot inherit the pin.
                fresh_read = getattr(runs_db, "get_run_fresh", None)
                if callable(fresh_read):
                    try:
                        fresh_row = fresh_read(run_id)
                    except Exception:  # noqa: BLE001
                        fresh_row = None
                    if fresh_row is not None:
                        row = fresh_row
                if str(row.get("status")) not in TERMINAL_RUN_STATUSES:
                    # Last honest resort (a double without the fresh-read
                    # seam, or a genuinely unreadable file): the settle/
                    # ledger-recorded terminal word is the child's known
                    # state at delivery -- never announce 'running' for a
                    # settled child.
                    row = {**row, "status": status}
            if row is not None and row.get("wake_delivered_at"):
                stale.append(run_id)
                continue
            rows.append(
                row
                if row is not None
                else {"id": run_id, "status": status, "result": None}
            )
        if stale:
            with self._registry_lock:
                pending_bucket = self._pending.get(conversation_id)
                if pending_bucket is not None:
                    for run_id in stale:
                        pending_bucket.pop(run_id, None)
                    if not pending_bucket:
                        self._pending.pop(conversation_id, None)
        rows.sort(key=lambda r: str(r.get("updated_at") or ""))
        return rows
