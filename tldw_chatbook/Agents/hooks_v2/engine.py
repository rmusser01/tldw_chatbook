"""Immutable hook sessions running on one application-owned event loop.

This module executes handlers and reports effects; producers own tool mutation,
required checkpoints, context acceptance and scheduler integration.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock

from .budgets import BudgetExceeded, HookBudgetOwner
from .command_executor import CommandExecutor
from .matching import UnsupportedEventField, matches_handler
from .models import HookEvent, HookHandler, HookResult
from .ownership import HookProcessOwner, HostProcessOwner
from .validation import parse_event


def _envelope(event: HookEvent) -> dict:
    return {
        key: value for key, value in event.model_dump().items() if value is not None
    }


@dataclass(frozen=True)
class HookFailure:
    handler_id: str
    code: str
    blocking: bool = False
    dependency_required: bool = False


@dataclass(frozen=True)
class HookEventOutcome:
    accepted: tuple[tuple[str, HookResult], ...] = ()
    omissions: tuple[str, ...] = ()
    failures: tuple[HookFailure, ...] = ()
    outstanding_cleanup: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return not self.failures and not self.omissions and not self.outstanding_cleanup

    @property
    def allowed(self) -> bool:
        return not any(f.blocking for f in self.failures) and not any(
            result.decision == "deny" for _, result in self.accepted
        )


@dataclass
class _ExecutionState:
    """Private accounting shared by the engine and its read-only handle."""

    issuer: object
    identity: tuple
    deadline: float
    teardown: bool = False
    active_seconds: float = 0.0
    closed: bool = False
    busy: bool = False


@dataclass(frozen=True, init=False, slots=True, eq=False)
class HookEventExecution:
    """Engine-issued, read-only event scope with irreversible retirement.

    The handle carries bounded private accounting, so abandoned caller scopes
    need no retained engine registry. Only engine code updates accounting;
    public construction, replacement and reassignment are unavailable.
    """

    _state: _ExecutionState = field(repr=False)

    def __init__(self) -> None:
        raise TypeError("event scopes are issued by HookEngine.begin_event")

    @classmethod
    def _issue(cls, state: _ExecutionState) -> HookEventExecution:
        scope = object.__new__(cls)
        object.__setattr__(scope, "_state", state)
        return scope

    @property
    def deadline(self) -> float:
        return self._state.deadline

    @property
    def active_seconds(self) -> float:
        return self._state.active_seconds

    @property
    def closed(self) -> bool:
        return self._state.closed

    @property
    def busy(self) -> bool:
        return self._state.busy

    @property
    def teardown(self) -> bool:
        return self._state.teardown

    def close(self) -> None:
        self._state.closed = True


@dataclass(eq=False)
class _Delivery:
    ticket: object
    cancel: asyncio.Event
    teardown: bool
    task: asyncio.Task | None = None
    job: object | None = None


class HookEngine:
    """Execute reviewed definitions using an injected current-authority check.

    ``authority_check(handler, event, stage)`` is called on an agent worker for
    admission, launch and acceptance. ``environment(handler, event)`` resolves
    declared variable references, returning values keyed by reference name.
    ``host_environment`` supplies reserved values last. No environment, output
    or exception text is retained in diagnostics. Dependency state remains with
    the host via ``dependency_required(handler, event)``.
    """

    def __init__(
        self,
        definitions: tuple[HookHandler, ...],
        authority_check: Callable,
        budget_owner: HookBudgetOwner,
        *,
        process_owner: HookProcessOwner | None = None,
        environment: Callable | None = None,
        host_environment: Callable | None = None,
        invalid_admissions: tuple = (),
        dependency_required: Callable | None = None,
        enabled: bool = True,
    ):
        if len(definitions) > 256 or len({h.id for h in definitions}) != len(
            definitions
        ):
            raise ValueError("invalid definition set")
        self.definitions = tuple(definitions)
        self.enabled = enabled
        self.notification_omissions = 0
        self.notification_failures: Counter[str] = Counter()
        self.authority_check = authority_check
        self.budget_owner = budget_owner
        self.loop = budget_owner.loop
        self.processes = CommandExecutor(process_owner or HostProcessOwner())
        self.invalid_admissions = tuple(invalid_admissions)
        self._environment = environment
        self._host_environment = host_environment
        self._dependency_required = dependency_required or (lambda *_: False)
        self._lock = RLock()
        self._scope_issuer = object()
        self._sealed_at: float | None = None
        self._teardown_closed = False
        self._deliveries: set[_Delivery] = set()
        self._close_task: asyncio.Task | None = None
        self._reconcile_task: asyncio.Task | None = None

    @classmethod
    def from_config(cls, config, authority_check, budget_owner, **kwargs):
        """Preserve H1 rejected-batch policy; never activate its valid-looking tail."""
        return cls(
            () if config.v2_invalid else config.v2_handlers,
            authority_check,
            budget_owner,
            invalid_admissions=config.v2_invalid_admissions,
            enabled=config.enabled,
            **kwargs,
        )

    def _env(self, handler, event) -> dict[str, str]:
        result = dict(os.environ)
        # Ambient host-owned roots cannot leak into an unrelated hook launch.
        result.pop("PLUGIN_ROOT", None)
        result.pop("PLUGIN_DATA", None)
        references = self._environment(handler, event) if self._environment else {}
        for key, value in (handler.env or {}).items():
            resolved = (
                value if isinstance(value, str) else references[value["variable"]]
            )
            if not isinstance(resolved, str) or "\0" in resolved:
                raise ValueError("invalid resolved environment")
            result[key] = resolved
        if self._host_environment:
            reserved = self._host_environment(handler, event)
            if set(reserved) - {"PLUGIN_ROOT", "PLUGIN_DATA"}:
                raise ValueError("invalid host environment")
            result.update(reserved)
        return result

    def begin_event(
        self, event: HookEvent, *, teardown: bool = False
    ) -> HookEventExecution:
        value = _envelope(event)
        parse_event(value)  # Bound input before serialization or admission.
        if event.causal_depth > 4:
            raise ValueError("causal_depth")
        identity = tuple((key, item) for key, item in value.items() if key != "data")
        now = time.monotonic()
        deadline = now + ({"Interrupt": 1.0, "SessionEnd": 3.0}.get(event.event, 180.0))
        with self._lock:
            if teardown and event.event not in {"Interrupt", "SessionEnd"}:
                raise ValueError("invalid teardown event")
            if self._sealed_at is not None:
                deadline = min(deadline, self._sealed_at + 3.0)
        return HookEventExecution._issue(
            _ExecutionState(self._scope_issuer, identity, deadline, teardown)
        )

    def _execution_state(self, execution: HookEventExecution) -> _ExecutionState | None:
        if type(execution) is not HookEventExecution:
            return None
        state = getattr(execution, "_state", None)
        if type(state) is not _ExecutionState or state.issuer is not self._scope_issuer:
            return None
        return state

    def _dependency_status(self, handler, event) -> tuple[bool, str | None]:
        try:
            return bool(self._dependency_required(handler, event)), None
        except Exception:  # noqa: BLE001 - retain unknown dependency requirements
            return True, "dependency_check_failed"

    async def _authority_status(self, handler, event, stage) -> tuple[bool, str | None]:
        try:
            return (
                bool(
                    await asyncio.to_thread(self.authority_check, handler, event, stage)
                ),
                None,
            )
        except Exception:  # noqa: BLE001 - callback text may contain private data
            return False, "authority_check_failed"

    def _failure(
        self, handler, event, code, *, dependency: bool | None = None
    ) -> HookFailure:
        blocking = (
            handler.required
            or (
                event.event == "PreToolUse"
                and bool(handler.effects & {"deny", "updated_input"})
            )
            or (
                event.event == "SubagentStart"
                and bool(handler.effects & {"deny", "child_limits"})
            )
        )
        if dependency is None:
            dependency, failure = self._dependency_status(handler, event)
            code = failure or code
        return HookFailure(handler.id, code, blocking, dependency)

    def _configuration_failure(self, event) -> HookEventOutcome | None:
        if not self.invalid_admissions:
            return None
        failures = tuple(
            HookFailure(
                f"config:{r.index}",
                "invalid_configuration",
                r.policy in {"explicit_required", "event_control"}
                and (r.event is None or r.event == event.event),
            )
            for r in self.invalid_admissions
        )
        return HookEventOutcome(failures=failures)

    def _sync(self, coroutine):
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is self.loop:
            coroutine.close()
            raise RuntimeError(
                "synchronous hooks cannot run on the owner loop; use fire_async"
            )
        if self.loop.is_closed() or not self.loop.is_running():
            coroutine.close()
            raise RuntimeError("hook owner loop unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def fire(self, event: HookEvent) -> HookEventOutcome:
        return self._sync(self.fire_async(event))

    def fire_handler(
        self, execution: HookEventExecution, handler_id: str, event: HookEvent
    ) -> HookEventOutcome:
        return self._sync(self.fire_handler_async(execution, handler_id, event))

    async def fire_async(self, event: HookEvent) -> HookEventOutcome:
        return await self._fire(event, teardown=False)

    async def fire_teardown_async(self, event: HookEvent) -> HookEventOutcome:
        """Only Interrupt/SessionEnd, within the fixed post-seal window."""
        return await self._fire(event, teardown=True)

    async def _fire(self, event, *, teardown):
        if self.loop.is_closed() or not self.loop.is_running():
            return HookEventOutcome(
                failures=(HookFailure("", "owner_unavailable", True),)
            )
        if asyncio.get_running_loop() is not self.loop:
            return await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    self._fire(event, teardown=teardown), self.loop
                )
            )
        try:
            execution = self.begin_event(event, teardown=teardown)
        except ValueError:
            return HookEventOutcome(failures=(HookFailure("", "invalid_event", True),))
        outcomes = []
        try:
            for handler in self.definitions:
                if handler.event == event.event:
                    outcomes.append(
                        await self.fire_handler_async(execution, handler.id, event)
                    )
            if not outcomes:
                config = self._configuration_failure(event)
                if config:
                    return config
                with self._lock:
                    if self._sealed_at is not None and not teardown:
                        return HookEventOutcome(
                            failures=(HookFailure("", "admission_closed", True),)
                        )
            return self._merge(outcomes)
        finally:
            execution.close()

    @staticmethod
    def _merge(outcomes):
        return HookEventOutcome(
            *(
                tuple(item for outcome in outcomes for item in getattr(outcome, key))
                for key in ("accepted", "omissions", "failures", "outstanding_cleanup")
            )
        )

    async def fire_handler_async(
        self, execution: HookEventExecution, handler_id: str, event: HookEvent
    ) -> HookEventOutcome:
        if self.loop.is_closed() or not self.loop.is_running():
            return HookEventOutcome(
                failures=(HookFailure(handler_id, "owner_unavailable", True),)
            )
        if asyncio.get_running_loop() is not self.loop:
            return await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    self.fire_handler_async(execution, handler_id, event), self.loop
                )
            )
        handler = next((h for h in self.definitions if h.id == handler_id), None)
        if handler is None:
            return HookEventOutcome(
                failures=(HookFailure(handler_id, "unknown_handler", True),)
            )

        def failed(code, *, dependency=None):
            return HookEventOutcome(
                failures=(self._failure(handler, event, code, dependency=dependency),)
            )

        try:
            value = _envelope(event)
            parse_event(value)
        except ValueError:
            return failed("invalid_event")
        state = self._execution_state(execution)
        if (
            state is None
            or state.closed
            or state.busy
            or state.identity != tuple((k, v) for k, v in value.items() if k != "data")
        ):
            return failed("invalid_event_execution")
        if not self.enabled:
            return failed("disabled")
        configuration = self._configuration_failure(event)
        if configuration:
            return configuration
        try:
            if not matches_handler(handler, event):
                return HookEventOutcome()
        except UnsupportedEventField:
            return failed("unsupported_matcher")
        with self._lock:
            if self._teardown_closed or (
                self._sealed_at is not None and not state.teardown
            ):
                return failed("admission_closed")
            if state.teardown and self._sealed_at is not None:
                state.deadline = min(state.deadline, self._sealed_at + 3.0)
            if time.monotonic() >= state.deadline or state.active_seconds >= 60:
                return failed("event_deadline")
            dependency, dependency_failure = self._dependency_status(handler, event)
            if dependency_failure:
                return failed(dependency_failure, dependency=dependency)
            observation = (
                not handler.effects and not handler.required and not dependency
            )
            try:
                ticket = self.budget_owner.reserve(
                    event.runtime_session_id, observation
                )
            except BudgetExceeded:
                return failed("delivery_capacity")
            delivery = _Delivery(ticket, asyncio.Event(), state.teardown)
            self._deliveries.add(delivery)
            state.busy = True
            delivery.task = asyncio.create_task(
                self._deliver(delivery, handler, event, state, value)
            )
            delivery.task.add_done_callback(self._consume)
        try:
            done, _ = await asyncio.wait(
                {delivery.task},
                timeout=max(0, state.deadline + 5.0 - time.monotonic()),
            )
            if done:
                return delivery.task.result()
            delivery.cancel.set()
            if delivery.job:
                delivery.job.stop()
            execution.close()
            return HookEventOutcome(
                failures=(self._failure(handler, event, "event_deadline"),),
                outstanding_cleanup=(handler.id,),
            )
        except asyncio.CancelledError:
            delivery.cancel.set()
            if delivery.job:
                delivery.job.stop()
            execution.close()
            raise
        finally:
            state.busy = False

    async def _deliver(self, delivery, handler, event, execution, value):
        ticket = delivery.ticket
        failure = None
        job = None
        try:
            acquire = asyncio.create_task(ticket.acquire())
            cancelled = asyncio.create_task(delivery.cancel.wait())
            try:
                done, _ = await asyncio.wait(
                    {acquire, cancelled},
                    timeout=max(0, execution.deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if acquire not in done or delivery.cancel.is_set():
                    failure = (
                        "cancelled" if delivery.cancel.is_set() else "event_deadline"
                    )
                else:
                    await acquire
            finally:
                if not acquire.done():
                    acquire.cancel()
                cancelled.cancel()
                await asyncio.gather(acquire, cancelled, return_exceptions=True)
            if failure is None:
                authorized, callback_failure = await self._authority_status(
                    handler, event, "admission"
                )
                if not authorized:
                    failure = callback_failure or "authority_refused"
            if failure is None and (delivery.cancel.is_set() or execution.closed):
                failure = "cancelled"
            if failure:
                return HookEventOutcome(
                    failures=(self._failure(handler, event, failure),)
                )
            if handler.type != "command":
                return HookEventOutcome(
                    failures=(self._failure(handler, event, "executor_unavailable"),)
                )
            deadline = min(
                execution.deadline,
                time.monotonic()
                + min(handler.timeout_seconds, 60 - execution.active_seconds),
            )
            job = delivery.job = self.processes.start(
                handler,
                event,
                ticket,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                ),
                self.authority_check,
                self._env,
                deadline,
            )
            result = await self.processes.wait(job, deadline)
            execution.active_seconds += result.duration
            if result.result is not None:
                authorized, callback_failure = await self._authority_status(
                    handler, event, "accept"
                )
                if not authorized:
                    result.result = None
                    result.failure = callback_failure or "authority_refused"
            if result.result is not None and (
                delivery.cancel.is_set()
                or execution.closed
                or time.monotonic() >= deadline
            ):
                result.result = None
                result.failure = "cancelled"
            if result.failure:
                return HookEventOutcome(
                    failures=(self._failure(handler, event, result.failure),),
                    outstanding_cleanup=(job.id,) if result.cleanup_pending else (),
                )
            return HookEventOutcome(accepted=((handler.id, result.result),))
        finally:
            if job is None:
                ticket.release()
            with self._lock:
                self._deliveries.discard(delivery)

    def notify(self, event: HookEvent) -> bool:
        return self._notify(event, teardown=False)

    def notify_teardown(self, event: HookEvent) -> bool:
        return self._notify(event, teardown=True)

    def _notify(self, event, *, teardown):
        # Reserve every matching handler synchronously so queue capacity bounds
        # submissions even when called faster than the owner loop can dispatch.
        try:
            execution = self.begin_event(event, teardown=teardown)
            value = _envelope(event)
        except ValueError:
            return False
        admitted = False
        with self._lock:
            if (
                self._teardown_closed
                or (self._sealed_at is not None and not teardown)
                or self.invalid_admissions
                or not self.enabled
            ):
                return False
            if time.monotonic() >= execution.deadline:
                return False
            for handler in self.definitions:
                try:
                    if not matches_handler(handler, event):
                        continue
                    dependency, dependency_failure = self._dependency_status(
                        handler, event
                    )
                    if dependency_failure:
                        self.notification_failures[dependency_failure] += 1
                        continue
                    if handler.effects or handler.required or dependency:
                        continue
                    ticket = self.budget_owner.reserve(event.runtime_session_id, True)
                except (ValueError, BudgetExceeded):
                    self.notification_omissions += 1
                    continue
                delivery = _Delivery(ticket, asyncio.Event(), teardown)
                self._deliveries.add(delivery)

                def submit(d=delivery, h=handler):
                    d.task = asyncio.create_task(
                        self._deliver(
                            d, h, event, self._execution_state(execution), value
                        )
                    )
                    d.task.add_done_callback(self._consume_notification)

                self.loop.call_soon_threadsafe(submit)
                admitted = True
        return admitted

    def _consume_notification(self, task) -> None:
        if task.cancelled():
            self.notification_failures["cancelled"] += 1
            return
        try:
            outcome = task.result()
        except Exception:  # noqa: BLE001 - only the fixed failure code is retained
            self.notification_failures["notification_failed"] += 1
            return
        for failure in outcome.failures:
            self.notification_failures[failure.code] += 1

    @staticmethod
    def _consume(task):
        if not task.cancelled():
            task.exception()

    @property
    def cleanup_pending(self) -> bool:
        with self._lock:
            return bool(self._deliveries or self.processes.records)

    @property
    def teardown_deadline(self) -> float | None:
        with self._lock:
            return None if self._sealed_at is None else self._sealed_at + 3.0

    def begin_close(self) -> None:
        """Synchronous ordinary admission seal; never reset the teardown clock."""
        with self._lock:
            if self._sealed_at is None:
                self._sealed_at = time.monotonic()
            for delivery in tuple(self._deliveries):
                if not delivery.teardown:

                    def stop(d=delivery):
                        d.cancel.set()
                        if d.job:
                            d.job.stop()

                    if not self.loop.is_closed():
                        self.loop.call_soon_threadsafe(stop)

    async def close(self) -> None:
        self.begin_close()
        if self.loop.is_closed() or not self.loop.is_running():
            raise RuntimeError("hook owner loop unavailable; cleanup remains owned")
        if asyncio.get_running_loop() is not self.loop:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(self.close(), self.loop)
            )
            return
        with self._lock:
            self._teardown_closed = True
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._drain())
        await asyncio.shield(self._close_task)

    async def _drain(self):
        # Ensure synchronously reserved notifications have published their tasks.
        await asyncio.sleep(0)
        tasks = {d.task for d in self._deliveries if d.task is not None}
        if tasks:
            await asyncio.wait(
                tasks, timeout=max(0, self._sealed_at + 3.0 - time.monotonic())
            )
        for job in tuple(self.processes.records.values()):
            job.stop()
        tasks |= {
            job.task for job in self.processes.records.values() if job.task is not None
        }
        if tasks:
            await asyncio.wait(
                tasks, timeout=max(0, self._sealed_at + 8.0 - time.monotonic())
            )
        # A callback can be slow or fail. The retained task/records survive the
        # bounded close caller; another wait never extends this close deadline.
        self._reconcile_task = asyncio.create_task(self.processes.reap_pending())
        self._reconcile_task.add_done_callback(self._consume)
        await asyncio.wait(
            {self._reconcile_task},
            timeout=max(0, self._sealed_at + 8.0 - time.monotonic()),
        )
