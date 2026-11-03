"""App-owned served control and approval relay wiring.

Read live App dependencies at their original execution points. The App retains
all tasks, callbacks, controllers, publishers and transport ownership.
"""

from typing import Any


def _start_served_canvas_control(self: Any, *, app_globals: dict[str, Any]) -> None:
    """Attach this authoritative child to its parent transport."""
    if self.served_canvas_control is None and not getattr(
        self, "_served_canvas_mode", False
    ):
        return
    from .Web_Server import served_console_app

    return served_console_app._start_served_canvas_control(
        self,
        app_globals=app_globals,
    )


async def _retire_after_control_loss(self: Any) -> None:
    """Close native admission and retain execution ownership through retirement."""
    from .Web_Server import served_console_app

    return await served_console_app._retire_after_control_loss(
        self,
    )


async def _watch_served_control(self: Any) -> None:
    """App-owned served control startup, transport loss, and request handling."""
    from .Web_Server import served_console_app

    return await served_console_app._watch_served_control(
        self,
    )


async def _handle_served_control_request(self: Any, request: Any) -> Any:
    """Keep Console control independent of Canvas feature/scope availability."""
    from .Web_Server import served_console_app

    return await served_console_app._handle_served_control_request(
        self,
        request,
    )


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
    """Preserve the native binding; retained runtime state owns served projections."""

    return original


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

    lifetime = getattr(self, "_served_control_lifetime_task", None)
    self._served_control_lifetime_task = None
    if lifetime is not None and not lifetime.done():
        lifetime.cancel()
        await app_globals["asyncio"].gather(lifetime, return_exceptions=True)
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
