"""App-owned served control startup, transport loss, and request handling."""

from .. import app_served as host


def _load_console_configuration_metadata():
    """Load the stock cold-source capsule inside the native preparation read."""
    from importlib import import_module

    return import_module("tldw_chatbook.Widgets.compact_model_bar")


def _start_served_canvas_control(
    self: host.Any, *, app_globals: dict[str, host.Any]
) -> None:
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
                host._retire_after_control_loss(self), name="served-control-unavailable"
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
                host._watch_served_control(self), name="served-control-lifetime"
            )
        except app_globals["asyncio"].CancelledError:
            pass
        except Exception as error:  # noqa: BLE001 -- preserve optional transport recovery
            self.loguru_logger.warning(  # noqa: PLE1205 -- App owns a brace-format Loguru logger
                "Served Canvas control unavailable type={} code=connection_failed",
                type(error).__name__,
            )
            self._served_control_lifetime_task = app_globals["asyncio"].create_task(
                host._retire_after_control_loss(self), name="served-control-refused"
            )

    task.add_done_callback(observe)


async def _retire_after_control_loss(self: host.Any) -> None:
    """Close native admission and retain execution ownership through retirement."""
    self._shutting_down = True
    try:
        self._close_boot_worker_gate("served control lost")
        runtime = getattr(self, "console_runtime", None)
        if runtime is not None:
            view = getattr(runtime, "view", None)
            if view is not None:
                view._console_chat_tearing_down = True
            controller = getattr(runtime, "_chat_controller", None)
            if controller is not None:
                controller.begin_shutdown()
    finally:
        # Native app shutdown retains the original view and execution lease
        # through project-index, selected view-worker and image retirement.
        self.exit()


async def _watch_served_control(self: host.Any) -> None:
    from ..Canvas.control_protocol import ControlProtocolError

    while True:
        try:
            await self.served_canvas_control.wait_disconnected(timeout=60)
        except ControlProtocolError as error:
            if error.code == "connection_deadline_exceeded":
                continue
            raise
        await host._retire_after_control_loss(self)
        return


async def _handle_served_control_request(self: host.Any, request: host.Any) -> host.Any:
    """Keep Console control independent of Canvas feature/scope availability."""
    import asyncio

    from ..Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        ControlMessage,
        ControlProtocolError,
    )

    if request.message_type.startswith("console."):
        from ..Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget

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
                    receipts = runtime.ensure_remote_receipts(
                        client.host_identity,
                        configuration_metadata_loader=_load_console_configuration_metadata,
                    )
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
                return reply(
                    {
                        "receipt": dict(
                            receipt.public(),
                            payload_fingerprint=receipt.payload_fingerprint,
                        )
                    }
                )
            if request.message_type == "console.receipt.request":
                if set(data) != {"command_id"} or not isinstance(
                    data["command_id"], str
                ):
                    raise ValueError("Unexpected receipt fields.")
                receipt = receipts.query(data["command_id"])
                return reply(
                    {
                        "receipt": dict(
                            receipt.public(),
                            payload_fingerprint=receipt.payload_fingerprint,
                        )
                        if receipt is not None
                        else None
                    }
                )
            if request.message_type == "console.snapshot.request":
                if set(data) - {"target", "cursor", "limit", "receipt_cursor"}:
                    raise ValueError("Unexpected snapshot fields.")
                target = (
                    ConsoleTarget.model_validate(data["target"])
                    if data.get("target") is not None
                    else None
                )
                snapshot = projection.snapshot(
                    target,
                    cursor=data.get("cursor"),
                    limit=data.get("limit", 50),
                    receipt_cursor=data.get("receipt_cursor"),
                )
                admitted = runtime._agent_runs_db is not None
                if admitted:
                    runtime._agent_runs_db._require_execution_owner()
                snapshot["execution_admitted"] = admitted
                return reply({"snapshot": snapshot})
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
            from ..Chat.console_remote_receipts import ExpiredConsoleCommand

            raise ControlProtocolError(
                "command_expired"
                if isinstance(error, ExpiredConsoleCommand)
                else "invalid_console_request"
            ) from error
        except (RuntimeError, PermissionError) as error:
            raise ControlProtocolError("authority_unavailable") from error
    return await self.served_canvas_handler.handle(request)
