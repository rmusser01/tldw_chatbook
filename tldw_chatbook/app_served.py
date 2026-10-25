"""App-owned served control and approval relay wiring.

Read live App dependencies at their original execution points. The App retains
all tasks, callbacks, controllers, publishers and transport ownership.
"""

from typing import Any


def _start_served_canvas_control(self: Any, *, app_globals: dict[str, Any]) -> None:
    """Attach this authoritative child to its parent transport."""

    client = self.served_canvas_control
    # ADR-236: the relay publisher rides the same loop/lifetime as the
    # control client. Run statuses are read from the app-owned Console
    # controller's run-state map — the same read seam the TUI polls for
    # terminal transitions — through the lazy `chat_controller` handle,
    # so the poller tolerates the controller not existing yet.
    # `getattr`: the start/stop pair is driven on partially-specified
    # app doubles in tests (see test_canvas_control_spawn).
    publisher = getattr(self, "served_relay_publisher", None)
    if client is None or self._served_canvas_control_start_task is not None:
        return
    task = app_globals["asyncio"].create_task(
        client.start(), name="start_served_canvas_control"
    )
    self._served_canvas_control_start_task = task

    def observe(
        completed: app_globals["asyncio"].Task[None],  # noqa: F821 -- eager runtime namespace lookup
    ) -> None:
        try:
            completed.result()
            if publisher is not None:
                publisher.attach(run_state_source=self._served_relay_run_states)
        except app_globals["asyncio"].CancelledError:
            pass
        except Exception as error:  # noqa: BLE001 -- preserve optional transport recovery
            self.loguru_logger.warning(  # noqa: PLE1205 -- App owns a brace-format Loguru logger
                "Served Canvas control unavailable type={} code=connection_failed",
                type(error).__name__,
            )

    task.add_done_callback(observe)


async def _handle_served_control_request(self: Any, request: Any) -> Any:
    """Keep Console control independent of Canvas feature/scope availability."""
    from .Canvas.control_protocol import ControlProtocolError

    if request.message_type.startswith("console."):
        raise ControlProtocolError("authority_unavailable")
    return await self.served_canvas_handler.handle(request)


def _served_relay_run_states(self: Any) -> Any:
    """Current ``{session_id: run state}`` for the relay poller, if built."""

    runtime = self.console_runtime
    controller = runtime.chat_controller if runtime is not None else None
    if controller is None:
        return None
    try:
        return controller.run_states
    except Exception:  # noqa: BLE001 - a poll tick must never break the relay
        return None


def _served_relay_wrap_pending_approval(self: Any, original: Any) -> Any:
    """Wrap one ``set_pending_approval`` binding for the relay (served only).

    The Console screen routes its approval hook value through here at
    the binding site (``console_view_hooks``). Native runs have no
    publisher and get the original callable back untouched — including
    identity — so nothing outside served mode changes behavior.
    """

    from .Web_Server.served_relay_publisher import wrap_pending_approval_hook

    if self.served_relay_publisher is None:
        return original
    return wrap_pending_approval_hook(self.served_relay_publisher, original)


def _served_relay_decide(self: Any, round_id: Any, decisions: Any) -> bool:
    """Apply one phone decision on this child's Console controller.

    ADR-236 ``relay.decide.request`` resolver for the served control
    handler: locates the app-owned Console controller (the same lazy
    ``chat_controller`` handle the relay run-status poller reads) and
    forwards through ``resolve_pending_approval`` — the identical
    entrypoint the desktop approval card's ApprovalDecided handler
    uses. Truthiness is the delivery verdict the parent relays back:
    the controller's own ``resolve_pending_approval`` returns ``None``
    by contract, so the armed-round pre-check below (the same
    round-id registry that resolve itself reads) is what
    distinguishes a delivered decision from an unknown/stale round.
    Both are safe: a mismatched round_id is a documented no-op there.
    """

    runtime = getattr(self, "console_runtime", None)
    controller = (
        getattr(runtime, "chat_controller", None) if runtime is not None else None
    )
    if controller is None:
        return False
    rounds = getattr(controller, "_pending_approval_rounds", None)
    if not isinstance(rounds, dict):
        # Unexpected controller shape: report undelivered rather than
        # guessing; the child's own timeout semantics still govern.
        return False
    state = rounds.get(round_id)
    if state is None or state.get("settled"):
        return False
    controller.resolve_pending_approval(dict(decisions), round_id=round_id)
    return True


async def _stop_served_canvas_control(
    self: Any, *, app_globals: dict[str, Any]
) -> None:
    """Close the private child channel without affecting terminal exit."""

    task = self._served_canvas_control_start_task
    self._served_canvas_control_start_task = None
    if task is not None and not task.done():
        task.cancel()
        await app_globals["asyncio"].gather(task, return_exceptions=True)
    publisher = getattr(self, "served_relay_publisher", None)
    if publisher is not None:
        await publisher.aclose()
    client = self.served_canvas_control
    if client is not None:
        await client.aclose()
