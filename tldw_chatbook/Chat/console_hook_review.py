"""Detached runtime-owned initial hook-review presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hook_permissions import HookReviewSnapshot


@dataclass(frozen=True, slots=True)
class HookReviewResult:
    kind: Literal["ready", "cancel", "settings"]
    snapshot: HookReviewSnapshot | None = None


@dataclass(frozen=True, slots=True)
class ConsoleHookReviewProjection:
    review_id: str
    session_id: str
    generation: int
    snapshot: HookReviewSnapshot = field(repr=False)
    waiting_for_send: bool
    presentation_token: object = field(repr=False)
    attachment_generation: int
    busy: bool


async def request_initial_hook_review(
    self,
    session_id: str,
    request_id: str,
    generation: int,
    snapshot: HookReviewSnapshot,
    *,
    waiting_for_send: bool = True,
) -> HookReviewResult:
    """Await one resident review without transferring cancellation to its answer."""
    from . import console_runtime as _runtime_owner

    self._raise_if_disposed_or_session_fenced(session_id)
    controller = self.ensure_chat_controller()
    answer = controller._interrupt_host.begin_hook_review(
        session_id,
        request_id,
        generation,
        snapshot,
        waiting_for_send=waiting_for_send,
        owner=self.ensure_hook_permissions(),
        loop=_runtime_owner.asyncio.get_running_loop(),
    )
    return await _runtime_owner.asyncio.shield(answer)


def _hook_review_presentation_current(
    self,
    review_id: str,
    generation: int,
    token: object,
) -> bool:
    view = self.view
    if self._disposed or view is None:
        return False
    # The owning modal suspends the Console and clears reconciliation.
    # Its current token and attachment still identify an answerable review.
    try:
        projection = view.app.screen._console_hook_review_projection
    except (AttributeError, RuntimeError):
        return False
    return bool(
        projection is not None
        and projection.review_id == review_id
        and projection.generation == generation
        and projection.presentation_token is token
        and projection.attachment_generation == self._attached_generation
    )


async def _await_hook_review_work(task):
    """Retain the original finite producer through repeated waiter cancellation."""
    from . import console_runtime as _runtime_owner

    cancelled = None
    while not task.done():
        try:
            await _runtime_owner.asyncio.shield(task)
        except _runtime_owner.asyncio.CancelledError as error:
            cancelled = error
        except BaseException:  # noqa: BLE001 - continue owned shutdown and preserve its existing failure policy.
            break
    if cancelled is not None:
        if not task.cancelled():
            task.exception()
        raise cancelled
    return task.result()


async def _execute_hook_review_operation(self, host, operation, expected, keys):
    from . import console_runtime as _runtime_owner

    snapshot = None
    error = None
    producer = None
    try:
        self._raise_if_disposed_or_session_fenced(operation.session_id)
        if self.ensure_hook_permissions() is not operation.owner:
            raise RuntimeError("Hook review owner changed.")
        if expected is not operation.expected:
            raise RuntimeError("Hook review snapshot changed.")
        owner = operation.owner
        if operation.purpose == "approve":
            call = _runtime_owner.functools.partial(owner.approve, expected, keys)
        elif operation.purpose == "revoke":
            call = _runtime_owner.functools.partial(owner.revoke, expected, keys[0])
        elif operation.purpose == "disable":
            call = _runtime_owner.functools.partial(owner.disable, expected, keys[0])
        elif operation.purpose == "recover":
            call = owner.recover
        elif operation.purpose == "reset":
            call = _runtime_owner.functools.partial(owner.reset_invalid_state, expected)
        elif operation.purpose == "verify":
            call = owner.snapshot
        else:
            raise ValueError("Unknown hook review action.")
        submit_resolved = _runtime_owner.threading.Event()
        submitted = False

        def invoke():
            # A queued item may survive executor thread-start failure.
            submit_resolved.wait()
            return call() if submitted else None

        try:
            context = _runtime_owner.contextvars.copy_context()
            producer = _runtime_owner.asyncio.get_running_loop().run_in_executor(
                None, context.run, invoke
            )
            submitted = True
        finally:
            submit_resolved.set()
        # Keep the native Future private: cancelling all Tasks must not
        # turn cancellation of a to_thread wrapper into physical retirement.
        snapshot = await self._await_hook_review_work(producer)
        return snapshot
    except BaseException as failure:
        error = failure
        raise
    finally:
        # Waiter cancellation cannot change the actual native write outcome.
        if producer is not None and producer.done() and not producer.cancelled():
            error = producer.exception()
            if error is None:
                snapshot = producer.result()
        host.finish_hook_review_operation(operation, snapshot, error=error)


def _start_hook_review_operation(self, host, operation, expected, keys):
    from . import console_runtime as _runtime_owner

    coroutine = self._execute_hook_review_operation(host, operation, expected, keys)
    try:
        # A configurable factory can eagerly issue native work, then raise
        # without returning its handle. This private driver must start lazily.
        task = _runtime_owner.asyncio.Task(
            coroutine, loop=_runtime_owner.asyncio.get_running_loop()
        )
    except BaseException as error:
        coroutine.close()
        host.finish_hook_review_operation(operation, None, error=error)
        raise
    operation.task = task

    def finished(driver):
        if driver.cancelled():
            # Cancellation before the first step never enters its finally.
            # An entered driver drains native work before becoming done.
            host.finish_hook_review_operation(
                operation, None, error=_runtime_owner.asyncio.CancelledError()
            )
        self._consume_task_outcome(driver)

    task.add_done_callback(finished)
    return task


async def apply_hook_review_action(
    self,
    review_id: str,
    generation: int,
    action: Literal["approve", "revoke", "disable", "recover", "reset"],
    expected: HookReviewSnapshot,
    keys: tuple[str, ...] = (),
    *,
    presentation_token: object,
) -> HookReviewSnapshot:
    """Issue one checked consent action under the resident operation owner."""
    if action not in {"approve", "revoke", "disable", "recover", "reset"}:
        raise ValueError("Unknown hook review action.")
    if (action in {"revoke", "disable"} and len(keys) != 1) or (
        action in {"recover", "reset"} and keys
    ):
        raise ValueError("Invalid hook review selection.")
    if not self._hook_review_presentation_current(
        review_id, generation, presentation_token
    ):
        raise RuntimeError("Hook review presentation changed.")
    host = self._chat_controller._interrupt_host
    operation = host.begin_hook_review_operation(
        review_id, generation, presentation_token, action
    )
    if operation is None:
        raise RuntimeError("Hook review changed or is busy.")
    task = self._start_hook_review_operation(host, operation, expected, keys)
    return await self._await_hook_review_work(task)


def resolve_initial_hook_review(
    self,
    review_id: str,
    generation: int,
    result: HookReviewResult,
    *,
    presentation_token: object,
) -> bool:
    """Treat Ready as a fresh verification intent; other answers settle exactly."""
    if not self._hook_review_presentation_current(
        review_id, generation, presentation_token
    ):
        return False
    host = self._chat_controller._interrupt_host
    if result.kind == "ready":
        operation = host.begin_hook_review_operation(
            review_id, generation, presentation_token, "verify"
        )
        if operation is None:
            return False
        self._start_hook_review_operation(host, operation, operation.expected, ())
        return True
    return host.resolve_hook_review(
        review_id, generation, result, presentation_token=presentation_token
    )


async def _drain_hook_review_operations(self, session_id=None) -> bool:
    from . import console_runtime as _runtime_owner

    controller = self._chat_controller
    host = getattr(controller, "_interrupt_host", None)
    retirements = getattr(host, "hook_review_retirements", None)
    if not callable(retirements):
        return False
    cancelled = False
    while completions := retirements(session_id):
        for completion in completions:
            try:
                await self._await_hook_review_work(completion)
            except _runtime_owner.asyncio.CancelledError:
                if completion.cancelled():
                    raise RuntimeError(
                        "Hook review retirement signal was cancelled."
                    ) from None
                cancelled = True
    return cancelled
