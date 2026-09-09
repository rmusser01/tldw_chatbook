"""Deliver saved child results through bounded, durable automatic attempts.

Individual settlements enter a fixed 250 ms coalescing window. Each accepted
wake claims results from one causal work chain before helpers or provider work.
At most two conversations run automatic primaries, leaving one manual slot.
Completion stamps the claimed results atomically; interrupted or ambiguous work
stays saved for review rather than being replayed. Badges are a view projection,
never the authority for delivery or recovery (ADR-134 and ADR-135).
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES, RunBudget
from tldw_chatbook.Agents.agent_service import (
    AUTOWAKE_ENABLED_KEY,
    DEFAULT_AUTOWAKE_ENABLED,
    _coerce_autowake_enabled,
)
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkRefused,
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
    "You may act on these results now, or wait for the user's next message."
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
        text[:cap] + "\n… (truncated to share the notice's result budget; the full "
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
            str(result) if result else f"(no result recorded; the run ended '{status}')"
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
            (
                f"{count} background sub-agent{plural} finished after the turn "
                "ended. Results:"
            ),
            "",
            "\n\n".join(blocks),
            "",
            WAKE_NOTICE_TRAILER,
        ]
    )


_WAKE_AUTHORIZATION_KEY = object()


class AgentWakeAuthorization:
    """Authority for one exact durable attempt; never supplied by model text."""

    __slots__ = (
        "_coordinator",
        "acceptance_started",
        "accepted",
        "attempt_id",
        "context",
        "conversation_id",
        "owner_id",
        "preflight_refused",
        "session_id",
        "work_chain_id",
    )

    def __init__(
        self,
        coordinator,
        session_id,
        *,
        _key,
        work_chain_id=None,
        conversation_id=None,
        attempt_id=None,
        owner_id=None,
        context=None,
    ):
        if _key is not _WAKE_AUTHORIZATION_KEY:
            raise PermissionError("wake authority is coordinator-internal")
        self._coordinator = coordinator
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.work_chain_id = work_chain_id
        self.attempt_id = attempt_id
        self.owner_id = owner_id
        self.context = context
        self.acceptance_started = False
        self.accepted = False
        self.preflight_refused = False

    def __repr__(self):
        return "AgentWakeAuthorization(authority=<redacted>)"


@dataclass
class _WakeDelivery:
    session_id: str
    authorization: AgentWakeAuthorization | None = None


class ConsoleFleetWakeCoordinator:
    """Bounded conversation scheduling with durable, non-replayable attempts."""

    NAME = "fleet-wake"
    RETRY_DELAY_SECONDS = 1.0
    COALESCE_SECONDS = 0.25
    MAX_AUTOMATIC_PRIMARIES = 2

    def __init__(self, controller: ConsoleChatController) -> None:
        self._controller = controller
        self._app = None
        self._loop = None
        self._registry_lock = threading.Lock()
        self._pending = {}
        self._active: dict[str, _WakeDelivery] = {}
        self._owner_id = uuid4().hex
        self._paused: dict[tuple[str, str | None], str] = {}
        self._result_chains: dict[str, str | None] = {}
        self._coalesce_until: dict[str, float] = {}
        self._retry_after = {}
        self._retry_timer = None
        self._delivery_tasks = set()
        self._recovery_task = None
        self._recovery_requested = False
        self._recovery_failure_reason: str | None = None
        self._recovery_ready = True
        self.delivery_ui_hook = None

    def wire(self, *, app=None, loop=None):
        if app is not None:
            self._app = app
        if loop is not None:
            self._loop = loop
        else:
            self.capture_loop_if_running()

    def capture_loop_if_running(self):
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._recovery_requested and self._recovery_task is None:
            self._recovery_task = self._loop.create_task(self.recover())

    def authorizes(self, authorization, session_id):
        if not isinstance(authorization, AgentWakeAuthorization):
            return False
        active = self._active.get(authorization.conversation_id)
        return bool(
            active
            and active.authorization is authorization
            and authorization._coordinator is self
            and active.session_id == session_id == authorization.session_id
            and authorization.owner_id == self._owner_id
        )

    def pending_conversation_ids(self) -> tuple[str, ...]:
        with self._registry_lock:
            return tuple(self._pending)

    def has_pending(self, conversation_id: str) -> bool:
        with self._registry_lock:
            return bool(self._pending.get(conversation_id))

    def delivering_conversation_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    def delivering_session_ids(self) -> tuple[str, ...]:
        return tuple(item.session_id for item in self._active.values())

    def pause_reason(self, conversation_id: str) -> str | None:
        if self._recovery_failure_reason is not None:
            return self._recovery_failure_reason
        return next(
            (
                reason
                for (cid, _), reason in self._paused.items()
                if cid == conversation_id
            ),
            None,
        )

    def on_child_settled(self, event):
        child = getattr(event, "child", None)
        self._intake(str(getattr(event, "conversation_id", "") or ""), (child,))

    def on_fleet_drained(self, event):
        # Compatibility for older producers; the native bridge uses individual
        # settlement. Duplicate result IDs never extend the coalescing window.
        self._intake(
            str(getattr(event, "conversation_id", "") or ""),
            getattr(event, "children", ()) or (),
        )

    def _intake(self, conversation_id, children):
        if not conversation_id:
            return
        survivors = [
            child
            for child in children
            if getattr(child, "settled_after_turn", False)
            and getattr(child, "run_id", None)
        ]
        if not survivors:
            return
        with self._registry_lock:
            bucket = self._pending.setdefault(conversation_id, {})
            new = any(str(child.run_id) not in bucket for child in survivors)
            for child in survivors:
                bucket[str(child.run_id)] = str(child.status or "done")
            if new:
                self._coalesce_until.setdefault(
                    conversation_id, time.monotonic() + self.COALESCE_SECONDS
                )
        self.retry_soon()

    def retry_soon(self):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._attempt_all)
        except RuntimeError:
            pass

    def _attempt_all(self):
        with self._registry_lock:
            conversations = tuple(self._pending)
        self._retry_after = {
            cid: deadline
            for cid, deadline in self._retry_after.items()
            if cid in conversations
        }
        for conversation_id in conversations:
            self._attempt(conversation_id)

    def _schedule_delayed_retry(self, delay):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if self._retry_timer is not None:
            if self._retry_timer.when() <= loop.time() + delay:
                return
            self._retry_timer.cancel()

        def retry():
            self._retry_timer = None
            self.retry_soon()

        self._retry_timer = loop.call_later(max(delay, 0), retry)

    def _priority_allows(self, session_id, *, accepting=False):
        controller = self._controller
        if not autowake_enabled() or getattr(controller, "_disposed", False):
            return False
        if not any(s.id == session_id for s in controller.store.sessions()):
            return False
        try:
            if not accepting and controller.send_refusal_copy(session_id) is not None:
                return False
            queue = getattr(controller, "prompt_queue_coordinator", None)
            if queue is not None and queue.controls_generation(session_id):
                return False
            probe = getattr(controller, "wake_user_priority_probe", None)
            if callable(probe) and probe(session_id):
                return False
            live = getattr(controller, "_live_busy_session_ids", None)
            busy = set(live()) if callable(live) else set()
            busy.update(self.delivering_session_ids())
            busy.discard(session_id)
            cap = getattr(controller, "max_parallel_runs", 3)
            return len(busy) + 1 <= max(0, cap - 1)
        except Exception:  # noqa: BLE001 - bookkeeping and UI fail closed
            return False

    def _attempt(self, conversation_id):
        if not self._recovery_ready or conversation_id in self._active:
            return
        if len(self._active) >= self.MAX_AUTOMATIC_PRIMARIES:
            return
        with self._registry_lock:
            bucket = self._pending.get(conversation_id)
            if not bucket:
                return
            if all(
                run_id in self._result_chains
                and (conversation_id, self._result_chains[run_id]) in self._paused
                for run_id in bucket
            ):
                return
            deadline = max(
                self._retry_after.get(conversation_id, 0),
                self._coalesce_until.get(conversation_id, 0),
            )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            self._schedule_delayed_retry(remaining)
            return
        session_id = self._resolve_session_id(conversation_id)
        if session_id is None or not self._priority_allows(session_id):
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        self._active[conversation_id] = _WakeDelivery(session_id)
        with self._registry_lock:
            self._coalesce_until.pop(conversation_id, None)
        coroutine = self._deliver(conversation_id, session_id)
        try:
            task = loop.create_task(coroutine)
        except Exception as exc:  # noqa: BLE001 - failed scheduling grants no authority
            logger.warning(
                "wake delivery task could not be scheduled; deferring (exception_type={})",
                type(exc).__name__,
            )
            coroutine.close()
            self._active.pop(conversation_id, None)
            return
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)

    def _notify_ui(self, session_id):
        if callable(self.delivery_ui_hook):
            try:
                self.delivery_ui_hook(session_id)
            except Exception:  # noqa: BLE001 - a detached UI cannot prevent cleanup
                return

    async def accept(self, authorization, session_id):
        """Required fence before helpers, models, or tools; only True may run."""
        if not self.authorizes(authorization, session_id):
            raise PermissionError("wake authority is no longer live")
        if not self._priority_allows(session_id, accepting=True):
            authorization.preflight_refused = True
            return False
        authorization.acceptance_started = True
        accepted = await asyncio.to_thread(
            self._runs_db().automatic_work.accept_wake,
            authorization.attempt_id,
            owner_id=self._owner_id,
            limits=AutomaticWorkLimits.from_settings(),
        )
        if not accepted:
            raise AutomaticWorkRefused("attempt_not_prepared")
        await asyncio.to_thread(authorization.context.mark_accepted)
        authorization.accepted = True
        # Join the back after acceptance, giving other ready conversations a turn.
        with self._registry_lock:
            bucket = self._pending.pop(authorization.conversation_id, None)
            if bucket is not None:
                self._pending[authorization.conversation_id] = bucket
        return True

    async def _pause(self, conversation_id, chain_id, reason, *, review=False):
        self._paused[(conversation_id, chain_id)] = reason
        self._retry_after.pop(conversation_id, None)
        if chain_id is not None:
            try:
                await asyncio.to_thread(
                    self._runs_db().automatic_work.pause,
                    chain_id,
                    reason,
                    review_required=review,
                )
            except Exception as exc:  # noqa: BLE001 - refusal stays latched
                logger.warning(
                    "wake pause could not be saved (exception_type={})",
                    type(exc).__name__,
                )
                self._paused[(conversation_id, chain_id)] = "history_unavailable"
        if self._app is not None:
            set_fleet_unseen_completion(self._app, conversation_id)
        session_id = self._resolve_session_id(conversation_id)
        if session_id:
            self._notify_ui(session_id)

    async def _watch_deadline(self, authorization, delivery_task):
        while True:
            await asyncio.sleep(0.25)
            if authorization.accepted:
                try:
                    await asyncio.to_thread(authorization.context.check)
                except AutomaticWorkRefused:
                    self._controller._signal_stop(session_id=authorization.session_id)
                    delivery_task.cancel()
                    return
                except Exception:  # noqa: BLE001 - bookkeeping and UI fail closed
                    await self._pause(
                        authorization.conversation_id,
                        authorization.work_chain_id,
                        "history_unavailable",
                        review=True,
                    )
                    self._controller._signal_stop(session_id=authorization.session_id)
                    delivery_task.cancel()
                    return

    async def _deliver(self, conversation_id, session_id):
        from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext
        from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin

        authorization = None
        returned = False
        watcher = None
        ledger = getattr(self._runs_db(), "automatic_work", None)
        chain_id = None
        try:
            if ledger is None:
                await self._pause(
                    conversation_id, None, "history_unavailable", review=True
                )
                return
            with self._registry_lock:
                bucket = dict(self._pending.get(conversation_id) or {})
            rows = await asyncio.to_thread(self._rows_for, conversation_id, bucket)
            rows.sort(key=lambda row: (row.get("updated_at") or "", str(row["id"])))
            eligible = []
            for row in rows:
                candidate = row.get("work_chain_id")
                self._result_chains[str(row["id"])] = candidate
                if candidate is None:
                    await self._pause(conversation_id, None, "legacy_lineage")
                    continue
                if (conversation_id, candidate) not in self._paused:
                    eligible.append(row)
            if not eligible:
                return
            chain_id = eligible[0]["work_chain_id"]
            selected = [row for row in eligible if row["work_chain_id"] == chain_id][
                :256
            ]
            run_ids = tuple(str(row["id"]) for row in selected)
            attempt_id = uuid4().hex
            await asyncio.to_thread(
                ledger.claim_wake,
                chain_id,
                attempt_id=attempt_id,
                owner_id=self._owner_id,
                session_id=session_id,
                run_ids=run_ids,
                limits=AutomaticWorkLimits.from_settings(),
            )
            context = AutomaticWorkContext(ledger, chain_id, self._owner_id, attempt_id)
            authorization = AgentWakeAuthorization(
                self,
                session_id,
                _key=_WAKE_AUTHORIZATION_KEY,
                conversation_id=conversation_id,
                work_chain_id=chain_id,
                attempt_id=attempt_id,
                owner_id=self._owner_id,
                context=context,
            )
            self._active[conversation_id].authorization = authorization
            self._notify_ui(session_id)
            watcher = asyncio.create_task(
                self._watch_deadline(authorization, asyncio.current_task())
            )
            with context.scope():
                await self._controller.submit_draft(
                    compose_wake_notice(selected),
                    session_id=session_id,
                    origin=ConsoleSubmissionOrigin.AGENT_WAKE,
                    wake_authorization=authorization,
                )
            returned = True
        except AutomaticWorkRefused as exc:
            await self._pause(conversation_id, chain_id, str(exc))
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - preserve uncertain execution
            logger.warning(
                "wake delivery failed (exception_type={})", type(exc).__name__
            )
            await self._pause(
                conversation_id, chain_id, "interrupted_work", review=True
            )
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                if authorization is not None:
                    await self._finish_attempt(authorization, returned)
            finally:
                self._active.pop(conversation_id, None)
                self.retry_soon()

    async def _finish_attempt(self, authorization, returned):
        cid, chain_id = authorization.conversation_id, authorization.work_chain_id
        ledger = self._runs_db().automatic_work
        try:
            attempt = await asyncio.to_thread(
                ledger.read_attempt, authorization.attempt_id, owner_id=self._owner_id
            )
            if attempt.state == "prepared" and authorization.preflight_refused:
                await asyncio.to_thread(
                    ledger.abort_wake, attempt.id, owner_id=self._owner_id
                )
                self._retry_after[cid] = time.monotonic() + self.RETRY_DELAY_SECONDS
                return
            snapshot = await asyncio.to_thread(ledger.snapshot, chain_id)
            if attempt.state == "accepted" and returned and snapshot.status == "active":
                await asyncio.to_thread(
                    ledger.complete_wake, attempt.id, owner_id=self._owner_id
                )
                with self._registry_lock:
                    bucket = self._pending.get(cid, {})
                    for run_id in attempt.run_ids:
                        bucket.pop(run_id, None)
                        self._result_chains.pop(run_id, None)
                    if not bucket:
                        self._pending.pop(cid, None)
                    nothing_pending = cid not in self._pending
                if nothing_pending and self._app is not None:
                    if self._conversation_in_view(cid, authorization.session_id):
                        clear_fleet_unseen_completion(self._app, cid)
                    else:
                        set_fleet_unseen_completion(self._app, cid)
                return
            await self._pause(
                cid,
                chain_id,
                snapshot.pause_reason or "interrupted_work",
                review=snapshot.status == "active"
                or snapshot.status == "review_required",
            )
        except Exception:  # noqa: BLE001 - bookkeeping and UI fail closed
            await self._pause(cid, chain_id, "completion_unrecorded", review=True)

    def seed_from_marks(self) -> int:
        """Compatibility entry point: discover from run history, independently of badges."""
        db = self._runs_db()
        if db is None:
            return 0
        seeded = 0
        for cid in db.pending_wake_conversation_ids():
            rows = db.pending_wake_results(cid)
            if not rows:
                continue
            with self._registry_lock:
                bucket = self._pending.setdefault(cid, {})
                for row in rows:
                    bucket.setdefault(str(row["id"]), str(row["status"]))
            seeded += 1
        return seeded

    def start_recovery(self) -> None:
        """Called once by native runtime creation, never by view/DB attachment."""
        if self._recovery_requested:
            return
        self._recovery_requested = True
        self._recovery_ready = False
        self.capture_loop_if_running()

    async def wait_for_recovery(self) -> bool:
        """Wait for the runtime's single audit; never start recovery on a view read."""
        self.capture_loop_if_running()
        if self._recovery_task is not None:
            await asyncio.shield(self._recovery_task)
        return self._recovery_ready

    async def recover(self) -> None:
        self._recovery_ready = False
        try:
            ledger = self._runs_db().automatic_work
            await asyncio.to_thread(ledger.recover, current_owner_id=self._owner_id)
            await asyncio.to_thread(self.seed_from_marks)
            with self._registry_lock:
                pending = {cid: dict(bucket) for cid, bucket in self._pending.items()}
            for cid, bucket in pending.items():
                rows = await asyncio.to_thread(self._rows_for, cid, bucket)
                self._result_chains.update(
                    (str(row["id"]), row.get("work_chain_id")) for row in rows
                )
                for chain_id in {row.get("work_chain_id") for row in rows}:
                    if chain_id is None:
                        self._paused[(cid, None)] = "legacy_lineage"
                    else:
                        snapshot = await asyncio.to_thread(ledger.snapshot, chain_id)
                        if snapshot.status != "active":
                            self._paused[(cid, chain_id)] = (
                                snapshot.pause_reason or "review_required"
                            )
            self._recovery_failure_reason = None
            self._recovery_ready = True
            self.retry_soon()
        except Exception as exc:  # noqa: BLE001 - incomplete audit forbids dispatch
            logger.warning(
                "wake recovery failed (exception_type={})", type(exc).__name__
            )
            self._recovery_ready = False
            self._recovery_failure_reason = "history_unavailable"
            for session in self._controller.store.sessions():
                self._notify_ui(session.id)

    def _conversation_in_view(self, conversation_id: str, session_id: str) -> bool:
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
                "wake view probe raised; keeping the unseen mark (exception_type={})",
                type(exc).__name__,
            )
            return False

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

    def _rows_for(self, conversation_id: str, bucket: Mapping[str, str]) -> list[dict]:
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
                except Exception:  # noqa: BLE001 - bookkeeping and UI fail closed  # noqa: BLE001
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
                    except Exception:  # noqa: BLE001 - bookkeeping and UI fail closed  # noqa: BLE001
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
