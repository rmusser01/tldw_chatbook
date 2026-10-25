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
    if self._served_canvas_control_start_task is not None:
        return
    if client is None:
        if getattr(self, "_served_canvas_mode", False):
            self._served_control_lifetime_task = app_globals["asyncio"].create_task(
                _retire_after_control_loss(self), name="served-control-unavailable"
            )
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
                publisher.attach(
                    projection_source=lambda: (
                        self.console_runtime.ensure_remote_projection(
                            client.host_identity
                        )
                    )
                )
            self._served_control_lifetime_task = app_globals["asyncio"].create_task(
                _watch_served_control(self), name="served-control-lifetime"
            )
        except app_globals["asyncio"].CancelledError:
            pass
        except Exception as error:  # noqa: BLE001 -- preserve optional transport recovery
            self.loguru_logger.warning(  # noqa: PLE1205 -- App owns a brace-format Loguru logger
                "Served Canvas control unavailable type={} code=connection_failed",
                type(error).__name__,
            )
            self._served_control_lifetime_task = app_globals["asyncio"].create_task(
                _retire_after_control_loss(self), name="served-control-refused"
            )

    task.add_done_callback(observe)


async def _retire_after_control_loss(self: Any) -> None:
    """Close native admission and retain execution ownership through retirement."""
    self._shutting_down = True
    self._close_boot_worker_gate("served control lost")
    runtime = getattr(self, "console_runtime", None)
    if runtime is not None:
        view = getattr(runtime, "view", None)
        if view is not None:
            view._console_chat_tearing_down = True
        # dispose closes admission/automatic wakes before its first physical drain.
        await runtime.dispose()
    await self._shutdown_console_runtime()
    self.exit()


async def _watch_served_control(self: Any) -> None:
    from .Canvas.control_protocol import ControlProtocolError

    while True:
        try:
            await self.served_canvas_control.wait_disconnected(timeout=60)
        except ControlProtocolError as error:
            if error.code == "connection_deadline_exceeded":
                continue
            raise
        await _retire_after_control_loss(self)
        return


async def _handle_served_control_request(self: Any, request: Any) -> Any:
    """Keep Console control independent of Canvas feature/scope availability."""
    import asyncio

    from .Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        ControlMessage,
        ControlProtocolError,
    )

    if request.message_type.startswith("console."):
        from .Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget

        runtime = getattr(self, "console_runtime", None)
        client = getattr(self, "served_canvas_control", None)
        if runtime is None or client is None:
            raise ControlProtocolError("authority_unavailable")
        projection = runtime.ensure_remote_projection(client.host_identity)

        def reply(payload):
            return ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                request.message_type.replace(".request", ".response"),
                request.request_id,
                None,
                payload,
            )

        def batch(after_sequence):
            value = projection.subscribe(after_sequence)
            return {
                "sequence": value.sequence,
                "reset_required": value.reset_required,
                "events": list(value.events),
            }

        data = request.payload["request"]
        try:
            if request.message_type in {
                "console.ticket.request",
                "console.command.request",
                "console.receipt.request",
                "console.snapshot.request",
            }:
                await asyncio.to_thread(runtime.ensure_activity_receipt_service)
                if runtime._agent_runs_db is not None:
                    receipts = runtime.ensure_remote_receipts(client.host_identity)
                elif request.message_type != "console.snapshot.request":
                    raise ControlProtocolError("authority_unavailable")
            if request.message_type == "console.ticket.request":
                if set(data) != {"action", "target", "payload"}:
                    raise ValueError("Unexpected ticket fields.")
                target_type = (
                    ConsoleTarget
                    if "session_id" in data["target"]
                    else ConsoleHostIdentity
                )
                ticket = receipts.issue(
                    data["action"],
                    target_type.model_validate(data["target"]),
                    data["payload"],
                )
                return reply(
                    {"ticket": {"token": ticket.token, "command_id": ticket.command_id}}
                )
            if request.message_type == "console.command.request":
                if set(data) != {"ticket", "payload"}:
                    raise ValueError("Unexpected command fields.")
                receipt = await receipts.submit(data["ticket"], data["payload"])
                return reply({"receipt": receipt.public()})
            if request.message_type == "console.receipt.request":
                if set(data) != {"command_id"} or not isinstance(
                    data["command_id"], str
                ):
                    raise ValueError("Unexpected receipt fields.")
                receipt = receipts.query(data["command_id"])
                return reply(
                    {"receipt": receipt.public() if receipt is not None else None}
                )
            if request.message_type == "console.snapshot.request":
                if set(data) - {"target", "cursor", "limit", "receipt_cursor"}:
                    raise ValueError("Unexpected snapshot fields.")
                target = (
                    ConsoleTarget.model_validate(data["target"])
                    if data.get("target") is not None
                    else None
                )
                return reply(
                    {
                        "snapshot": projection.snapshot(
                            target,
                            cursor=data.get("cursor"),
                            limit=data.get("limit", 50),
                            receipt_cursor=data.get("receipt_cursor"),
                        )
                    }
                )
            if request.message_type == "console.events.request":
                operation = data.get("operation", "read")
                if operation == "read" and set(data) <= {"operation", "after_sequence"}:
                    return reply({"batch": batch(data.get("after_sequence", 0))})
                if operation in {"register", "disconnect"} and set(data) == {
                    "operation",
                    "client_id",
                }:
                    method = (
                        projection.register_client
                        if operation == "register"
                        else projection.disconnect_client
                    )
                    method(data["client_id"])
                elif operation == "acknowledge" and set(data) == {
                    "operation",
                    "sequence",
                }:
                    projection.acknowledge_snapshot(data["sequence"])
                elif operation == "answerable" and set(data) == {
                    "operation",
                    "client_id",
                    "target",
                    "answerable",
                }:
                    projection.set_answerable(
                        data["client_id"],
                        ConsoleTarget.model_validate(data["target"]),
                        answerable=data["answerable"],
                    )
                else:
                    raise ValueError("Invalid Console subscription operation.")
                return reply({"batch": batch(projection.events.read(0).sequence)})
            raise ControlProtocolError("authority_unavailable")
        except (ValueError, TypeError, KeyError) as error:
            raise ControlProtocolError("invalid_console_request") from error
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
