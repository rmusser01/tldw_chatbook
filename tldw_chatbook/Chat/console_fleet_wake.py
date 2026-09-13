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
from collections.abc import Callable, Mapping, Sequence
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
from tldw_chatbook.Agents.run_log import _setting
from tldw_chatbook.Chat.console_fleet_attention import (
    clear_fleet_unseen_completion,
    set_fleet_unseen_completion,
)

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_agent_bridge import FleetChildSettled, FleetDrained
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController


# Match the durable automatic-work child-result eligibility contract.
_WAKE_CHILD_STATUSES = frozenset({"done", "error", "cancelled"})


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
    #: Bound the number of child results selected for one durable wake claim.
    MAX_RESULTS_PER_ATTEMPT = 256

    def __init__(self, controller: ConsoleChatController) -> None:
        self._controller = controller
        self._app = None
        self._loop = None
        self._registry_lock = threading.RLock()
        self._pending = {}
        self._active: dict[str, _WakeDelivery] = {}
        self._owner_id = uuid4().hex
        self._paused: dict[tuple[str, str | None], str] = {}
        self._result_chains: dict[str, str | None] = {}
        self._coalesce_until: dict[str, float] = {}
        self._retry_after = {}
        self._retry_timer = None
        self._delivery_tasks = {}
        self._recovery_task = None
        self._recovery_requested = False
        self._recovery_failure_reason: str | None = None
        self._recovery_ready = True
        self._startup_ready: Callable[[], bool] = lambda: True
        self.delivery_ui_hook = None
        self.buddy_sink = getattr(controller, "_buddy_sink", None)
        self._disposed = False
        self._conversation_fences: dict[str, int] = {}
        self._runtime_submitter = None

    def wire(
        self,
        *,
        app: TldwCli | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        startup_ready: Callable[[], bool] | None = None,
    ) -> None:
        """Attach runtime services and the loop used to schedule wake delivery.

        Omitted services retain their current bindings. Without an explicit
        loop, capture the caller's running loop if one exists.

        Args:
            app: Application owning durable completion marks and attention.
            loop: Explicit event loop override for delivery scheduling.
            startup_ready: Predicate allowing recovery after startup readiness;
                retained when omitted and initially always true.
        """
        if startup_ready is not None:
            self._startup_ready = startup_ready
        if app is not None:
            self._app = app
        if loop is not None:
            self._loop = loop
        else:
            self.capture_loop_if_running()

    def capture_loop_if_running(self) -> None:
        """Capture the caller's running loop and schedule requested recovery.

        Leave the existing loop unchanged when the caller has no running loop.
        If recovery was requested, create its task only when none exists;
        capturing a loop alone does not request a new recovery audit.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._recovery_requested and self._recovery_task is None:
            self._recovery_task = self._loop.create_task(self.recover())

    def bind_runtime_submitter(self, submit_wake: Callable[..., str]) -> None:
        """Route future wake turns through app-owned runtime custody.

        Args:
            submit_wake: Runtime callback accepting wake submission keywords
                and returning the admitted turn ID.
        """

        self._runtime_submitter = submit_wake

    def authorizes(self, authorization: object, session_id: str) -> bool:
        """Check an opaque token against this coordinator's exact live delivery.

        Args:
            authorization: Candidate authority; arbitrary objects are refused.
            session_id: Native session that would receive the automatic turn.

        Returns:
            True only for the active token with matching coordinator, owner,
            and session while no disposal or conversation-close fence applies.
        """
        if not isinstance(authorization, AgentWakeAuthorization):
            return False
        active = self._active.get(authorization.conversation_id)
        return bool(
            not self._disposed
            and authorization.conversation_id not in self._conversation_fences
            and active
            and active.authorization is authorization
            and authorization._coordinator is self
            and active.session_id == session_id == authorization.session_id
            and authorization.owner_id == self._owner_id
        )

    def pending_conversation_ids(self) -> tuple[str, ...]:
        """Snapshot conversations with pending results under the registry lock.

        Returns:
            Conversation IDs in current registry order, including paused work.
        """
        with self._registry_lock:
            return tuple(self._pending)

    def has_pending(self, conversation_id: str) -> bool:
        """Check whether a conversation has any locally pending result IDs.

        Args:
            conversation_id: Durable conversation whose pending work to inspect.

        Returns:
            True when its pending bucket is nonempty, even if delivery is paused.
        """
        with self._registry_lock:
            return bool(self._pending.get(conversation_id))

    def delivering_conversation_ids(self) -> tuple[str, ...]:
        """Snapshot conversations occupying automatic delivery slots.

        Returns:
            Durable conversation IDs from scheduling through delivery cleanup.
        """
        return tuple(self._active)

    def delivering_session_ids(self) -> tuple[str, ...]:
        """Snapshot native sessions targeted by active automatic deliveries.

        Returns:
            Native session IDs used to reconcile mounted delivery UI hooks.
        """
        return tuple(item.session_id for item in self._active.values())

    def pause_reason(self, conversation_id: str) -> str | None:
        """Read the recovery failure or first stored pause for a conversation.

        Args:
            conversation_id: Durable conversation whose pause to inspect.

        Returns:
            Recovery failure code, first matching chain pause code, or None.
        """
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

    def on_child_settled(self, event: FleetChildSettled) -> None:
        """Stage one eligible survivor from the native settlement callback.

        May run on the child thread. Intake records eligible results even when
        automatic waking is disabled, then schedules attempts on the bound loop.
        Within-turn, missing-ID, and ineligible-status children are excluded.

        Args:
            event: Durable child settlement with its owning conversation and
                original within-turn or survivor classification.
        """
        child = getattr(event, "child", None)
        self._intake(str(getattr(event, "conversation_id", "") or ""), (child,))

    def on_fleet_drained(self, event: FleetDrained) -> None:
        """Stage eligible survivors from a legacy fleet-drain callback.

        Supports older producers; the native bridge uses individual settlement.
        May run on a child thread. Duplicate result IDs do not extend the fixed
        coalescing window, and intake uses the same filters as child settlement.

        Args:
            event: Drained conversation and children settled since its previous
                drain, including their original survivor classifications.
        """
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
        excluded = [
            str(child.run_id)
            for child in survivors
            if str(child.status or "done") not in _WAKE_CHILD_STATUSES
        ]
        survivors = [
            child for child in survivors
            if str(child.status or "done") in _WAKE_CHILD_STATUSES
        ]
        if excluded:
            self._discard_pending_results(conversation_id, excluded, clear_attention=True)
        if not survivors:
            return
        with self._registry_lock:
            if self._disposed or conversation_id in self._conversation_fences:
                return
            bucket = self._pending.setdefault(conversation_id, {})
            new = any(str(child.run_id) not in bucket for child in survivors)
            for child in survivors:
                bucket[str(child.run_id)] = str(child.status or "done")
            if new:
                self._coalesce_until.setdefault(
                    conversation_id, time.monotonic() + self.COALESCE_SECONDS
                )
        self._publish_active_wakes(conversation_id, tuple(str(child.run_id) for child in survivors))
        self.retry_soon()

    def retry_soon(self) -> None:
        """Schedule pending delivery checks on the bound loop from any thread.

        Do nothing after disposal or when no usable loop is bound. Delivery
        checks retain responsibility for coalescing, readiness, and admission.
        """
        if self._disposed:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._attempt_all)
        except RuntimeError:
            pass

    def _attempt_all(self):
        with self._registry_lock:
            if self._disposed:
                return
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
        if (not autowake_enabled() or getattr(controller, "_disposed", False)
                or getattr(controller, "_maintenance_paused", False)):
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
        if self._controller is None or getattr(self._controller, "_maintenance_paused", False):
            return
        if not self._recovery_ready or conversation_id in self._active:
            return
        if self._disposed or getattr(self._controller, "_disposed", False) or conversation_id in self._conversation_fences:
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
        with self._registry_lock:
            if self._disposed or conversation_id in self._conversation_fences:
                return
            self._active[conversation_id] = _WakeDelivery(session_id)
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
        cancel_task = False
        with self._registry_lock:
            if (
                self._disposed
                or conversation_id in self._conversation_fences
                or conversation_id not in self._active
                or self._active[conversation_id].session_id != session_id
            ):
                cancel_task = True
            else:
                self._delivery_tasks[task] = conversation_id
        if cancel_task:
            task.cancel()
            return
        task.add_done_callback(self._forget_delivery_task)

    def _notify_ui(self, session_id):
        if callable(self.delivery_ui_hook):
            try:
                self.delivery_ui_hook(session_id)
            except Exception:  # noqa: BLE001 - a detached UI cannot prevent cleanup
                return

    async def accept(
        self, authorization: AgentWakeAuthorization, session_id: str
    ) -> bool:
        """Accept durable wake authority before helpers, models, or tools run.

        Args:
            authorization: Coordinator-issued token for the prepared attempt.
            session_id: Native session receiving the automatic turn.

        Returns:
            True after durable acceptance and owner revalidation; False when
            manual priority refuses the attempt before acceptance.

        Raises:
            PermissionError: The supplied authority is no longer live.
            AutomaticWorkRefused: The prepared attempt or execution owner cannot
                be accepted. Durable ledger errors also propagate to the caller.
        """
        from tldw_chatbook.Agents.automatic_work_budget import (
            AutomaticWorkLimits,
            AutomaticWorkRefused,
        )
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
        if not self.authorizes(authorization, session_id):
            raise AutomaticWorkRefused("wake_owner_released")
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
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
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
        from tldw_chatbook.Agents.automatic_work_budget import (
            AutomaticWorkLimits,
            AutomaticWorkRefused,
        )
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
                :self.MAX_RESULTS_PER_ATTEMPT
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
            with self._registry_lock:
                if self._disposed or conversation_id in self._conversation_fences or conversation_id not in self._active:
                    authorization.preflight_refused = True
                    return
                self._active[conversation_id].authorization = authorization
            self._notify_ui(session_id)
            watcher = asyncio.create_task(
                self._watch_deadline(authorization, asyncio.current_task())
            )
            with context.scope():
                if not self.authorizes(authorization, session_id):
                    authorization.preflight_refused = True
                    return
                if self._runtime_submitter is not None:
                    terminal = asyncio.get_running_loop().create_future()
                    def on_terminal(accepted):
                        if not terminal.done():
                            terminal.set_result(accepted)
                    try:
                        self._runtime_submitter(
                            compose_wake_notice(selected),
                            session_id=session_id,
                            wake_authorization=authorization,
                            on_terminal=on_terminal,
                        )
                    except Exception:
                        if not authorization.acceptance_started:
                            authorization.preflight_refused = True
                            return
                        raise
                    await asyncio.shield(terminal)
                else:
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
                self._release_exact_wakes(cid, attempt.run_ids)
                if nothing_pending and self._app is not None and not self._disposed and cid not in self._conversation_fences:
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
        """Discover pending results from durable history, independently of badges.

        Returns:
            Number of eligible conversations whose results were seeded, including
            conversations already represented in the local registry.
        """
        db = self._runs_db()
        if self._disposed or db is None:
            return 0
        seeded = 0
        for cid in db.pending_wake_conversation_ids():
            rows = db.pending_wake_results(cid)
            if not rows:
                continue
            with self._registry_lock:
                if self._disposed or cid in self._conversation_fences:
                    continue
                bucket = self._pending.setdefault(cid, {})
                for row in rows:
                    bucket.setdefault(str(row["id"]), str(row["status"]))
            seeded += 1
        return seeded

    def start_recovery(self) -> None:
        """Request the runtime's single recovery audit and block wake admission.

        Native runtime creation calls this once. Repeated requests are no-ops;
        without a running loop, a later loop capture schedules the audit.
        """
        if self._recovery_requested:
            return
        self._recovery_requested = True
        self._recovery_ready = False
        self.capture_loop_if_running()

    async def wait_for_recovery(self) -> bool:
        """Await an existing recovery task without starting another audit.

        Returns:
            Current recovery readiness after any existing task finishes. A failed
            audit remains unready; cancellation of this wait does not cancel it.
        """
        if self._recovery_task is not None:
            await asyncio.shield(self._recovery_task)
        return self._recovery_ready

    async def recover(self) -> None:
        """Audit execution ownership and restore pending results after startup.

        Wait for the startup predicate before reading the ledger. Disposal stops
        the wait; audit failures retain the admission fence and a failure reason.
        A successful audit restores pause state and schedules eligible deliveries.
        """
        self._recovery_ready = False
        try:
            while not self._startup_ready():
                if self._disposed:
                    return
                await asyncio.sleep(0.05)
            if self._disposed:
                return
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

    def fence_conversation(self, conversation_id: str, *, generation: int) -> None:
        """Reject stale wake work for one closing conversation.

        Args:
            conversation_id: Durable conversation to fence and cancel.
            generation: Close generation retained until its exact fence releases.
        """

        with self._registry_lock:
            # The first unreleased fence owns this conversation. A later
            # saved-chat incarnation must not replace a timed-out fence with
            # a generation that its own graceful close could release.
            if conversation_id in self._conversation_fences:
                return
            self._conversation_fences[conversation_id] = generation
            run_ids = tuple((self._pending.pop(conversation_id, None) or {}).keys())
            tasks = tuple(
                task
                for task, owner in self._delivery_tasks.items()
                if owner == conversation_id
            )
            for task in tasks:
                self._delivery_tasks.pop(task, None)
        self._release_exact_wakes(conversation_id, run_ids)
        for task in tasks:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                continue
        if self._app is not None:
            try:
                clear_fleet_unseen_completion(self._app, conversation_id)
            except Exception:  # noqa: BLE001, S110 - mark cleanup must not interrupt close
                pass

    def release_conversation_fence(
        self, conversation_id: str, *, generation: int
    ) -> bool:
        """Release an exact provisional fence after all old work drained.

        A stale generation or retained delivery owner fails closed. Timeout
        and app-disposal paths never call this seam, so uncooperative old work
        remains fenced for the process lifetime.

        Args:
            conversation_id: Durable conversation whose fence to release.
            generation: Exact generation that originally established the fence.

        Returns:
            True only when the matching fence is removed with no delivery owners
            remaining and the coordinator has not been disposed.
        """

        with self._registry_lock:
            if self._disposed:
                return False
            if self._conversation_fences.get(conversation_id) != generation:
                return False
            if conversation_id in self._active or any(
                owner == conversation_id for owner in self._delivery_tasks.values()
            ):
                return False
            self._conversation_fences.pop(conversation_id, None)
            return True

    def dispose(self) -> None:
        """Terminally fence producers, then clear membership and leases."""
        with self._registry_lock:
            if self._disposed:
                return
            self._disposed = True
            self._pending.clear()
            tasks = tuple(self._delivery_tasks)
            self._delivery_tasks.clear()
            sink = self.buddy_sink
        for task in tasks:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                continue
        if sink is not None:
            sink.clear_wakes()

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

    def _rows_for(self, conversation_id: str, bucket: Mapping[str, str]) -> list[dict]:
        """Read the pending runs' rows; synthesize an honest stand-in for
        a row that cannot be read (a wiped runs DB must not strand the
        pending entry forever). A run the durable ledger already shows
        wake-delivered (a redelivered drain, or a restart racing the
        in-memory commit) is DROPPED -- from the returned rows AND from
        the registry -- rather than re-announced.

        task-18601 part A (AC#2): every field this method (and
        ``compose_wake_notice``, its only consumer) reads --
        id/agent_definition/status/task/result/wake_delivered_at/
        updated_at -- is metadata, never a step. Reads prefer
        ``get_run_metadata``/``get_run_metadata_fresh`` (metadata-only,
        no step-log parse at all) over ``get_run``/``get_run_fresh``,
        via ``getattr`` so a test double that only implements the older
        methods (pre-dating this change) keeps working unchanged."""
        with self._registry_lock:
            if self._disposed or conversation_id in self._conversation_fences:
                return []
        runs_db = self._runs_db()
        rows: list[dict] = []
        stale: list[str] = []
        excluded: list[str] = []
        for run_id, status in bucket.items():
            row = None
            if runs_db is not None:
                read_row = getattr(runs_db, "get_run_metadata", None)
                if not callable(read_row):
                    read_row = getattr(runs_db, "get_run", None)
                if callable(read_row):
                    try:
                        row = read_row(run_id)
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
                fresh_read = getattr(runs_db, "get_run_metadata_fresh", None)
                if not callable(fresh_read):
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
            if row is not None and str(row.get("status")) not in _WAKE_CHILD_STATUSES:
                stale.append(run_id)
                excluded.append(run_id)
                continue
            if row is None and status not in _WAKE_CHILD_STATUSES:
                stale.append(run_id)
                excluded.append(run_id)
                continue
            if row is not None and row.get("wake_delivered_at"):
                stale.append(run_id)
                continue
            rows.append(
                row
                if row is not None
                else {"id": run_id, "status": status, "result": None}
            )
        if stale:
            self._discard_pending_results(
                conversation_id, stale, clear_attention=len(excluded) == len(stale)
            )
        rows.sort(key=lambda r: str(r.get("updated_at") or ""))
        return rows

    def _discard_pending_results(
        self, conversation_id: str, run_ids: Sequence[str], *, clear_attention: bool = False
    ) -> None:
        """Release excluded/stamped result owners without disturbing live siblings."""
        with self._registry_lock:
            if self._disposed:
                return
            bucket = self._pending.get(conversation_id)
            if bucket is not None:
                for run_id in run_ids:
                    bucket.pop(run_id, None)
                if not bucket:
                    self._pending.pop(conversation_id, None)
            delivery = self._active.get(conversation_id)
            idle = not self._pending.get(conversation_id) and (
                delivery is None or delivery.authorization is None
            )
        self._release_exact_wakes(conversation_id, run_ids)
        if clear_attention and idle and self._app is not None:
            try:
                eligible = self._runs_db().undelivered_wake_runs(conversation_id)
            except Exception:  # noqa: BLE001 - retain attention on an unreadable ledger
                return
            if not eligible:
                clear_fleet_unseen_completion(self._app, conversation_id)

    def _publish_active_wakes(
        self, conversation_id: str, run_ids: Sequence[str]
    ) -> None:
        """Publish active leases externally, then post-fence exact membership."""

        sink = self.buddy_sink
        if sink is None:
            return
        for run_id in run_ids:
            with self._registry_lock:
                live = (
                    not self._disposed
                    and conversation_id not in self._conversation_fences
                    and run_id in self._pending.get(conversation_id, {})
                )
            if not live:
                continue
            sink.wake(conversation_id, run_id, active=True)
            with self._registry_lock:
                still_live = (
                    not self._disposed
                    and conversation_id not in self._conversation_fences
                    and run_id in self._pending.get(conversation_id, {})
                )
            if not still_live:
                sink.wake(conversation_id, run_id, active=False)

    def _release_exact_wakes(
        self, conversation_id: str, run_ids: Sequence[str]
    ) -> None:
        """Release only the wake owners named by a completed callback."""
        if self.buddy_sink is None:
            return
        for run_id in run_ids:
            self.buddy_sink.wake(conversation_id, run_id, active=False)

    def _forget_delivery_task(self, task: asyncio.Task) -> None:
        """Drop one local delivery task without racing a close fence."""

        with self._registry_lock:
            self._delivery_tasks.pop(task, None)
