"""Publish one original rail save through the selected configuration fence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType

from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import config_participants, raw_participants, storage_admission
from tldw_chatbook.MCP import local_permission_operation
from tldw_chatbook.Utils.definition_records import capture_definitions, definitions_current

from .console_domain_services import _definition, _definition_current
from .console_execution_process import ConsoleChannelLost, ConsoleExecutionClient


@dataclass(eq=False)
class ConsoleRailPublication:
    client: ConsoleExecutionClient
    app: object
    view: object
    visit: int
    attachment_id: str
    old_revision: int
    revision: int
    projection_view: object
    epochs: tuple
    key: str
    value: object
    task: asyncio.Task | None = None
    fence: object = None
    refresh: object = None
    applied: object = None
    writer: object = None
    writer_result: bool | None = None
    writer_operation: object = None
    writer_state: object = None
    writer_retired: bool = False


def _positive(receipt):
    if receipt is None:
        return False
    for future in (receipt.received, receipt.applied, receipt.retired):
        if not future.done() or future.cancelled() or future.exception() is not None:
            return False
    return receipt.retired.result() is True


def require_rail_publication(scope, *, applied=False):
    """Check this actual retained writer; observers never gain command authority."""
    if type(scope) is not ConsoleRailPublication or not definitions_current(_DEFINITIONS):
        raise ConsoleChannelLost("Console rail publication definition changed")
    client = scope.client
    if (
        type(client) is not ConsoleExecutionClient
        or getattr(client, "_rail_publication", None) is not scope
        or scope.task not in client._workspace_operations
        or client._app is not scope.app
        or getattr(scope.app, "console_execution_client", None) is not client
        or getattr(scope.app, "console_runtime", None) is not None
        or (client.profile_epoch, client.service_epoch) != scope.epochs
        or client._disposed or client._channel_error is not None
        or client._source_revision not in (scope.old_revision, scope.revision)
    ):
        raise ConsoleChannelLost("Console rail publication owner changed")
    if client._source_revision == scope.revision and (
        scope.fence is None or not scope.fence.applied.done()
        or scope.fence.applied.cancelled() or scope.fence.applied.exception() is not None
        or scope.fence.applied.result() != {"source": "configuration", "revision": scope.revision}
    ):
        raise ConsoleChannelLost("Console rail publication fence changed")
    if applied and (
        asyncio.current_task() is not scope.task
        or client._source_revision != scope.revision or not client._intake_open
        or not all(_positive(receipt) for receipt in (scope.fence, scope.refresh, scope.applied))
        or scope.writer is None or not scope.writer.done() or scope.writer.cancelled()
        or scope.writer.exception() is not None or scope.writer.result() is not True
        or scope.writer_result is not True or scope.writer_retired is not True
        or not _writer_retired(scope)
        or scope.refresh.applied.result() != {"source_revision": scope.revision}
        or scope.applied.applied.result() != {"source_revision": scope.revision}
    ):
        raise ConsoleChannelLost("Console rail publication is not positively applied")
    return scope


def rail_publication_for(client, view, visit):
    """Return only this client's actual operation on the captured visible visit."""
    scope = getattr(client, "_rail_publication", None)
    if scope is None:
        return None
    require_rail_publication(scope)
    return scope if scope.view is view and scope.visit == visit else None


async def _settle(receipt):
    cancelled = None

    async def join(future):
        nonlocal cancelled
        while True:
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError as error:
                if future.cancelled():
                    raise
                cancelled = cancelled or error

    # A refused Received has no original physical receipt to join. Once
    # admitted, even failed Applied and repeated observer cancellation must
    # retain the actual operation until its independent retirement arrives.
    await join(receipt.received)
    try:
        result = await join(receipt.applied)
    finally:
        if await join(receipt.retired) is not True:
            raise ConsoleChannelLost("Console rail operation retirement is unproven")
    if cancelled is not None:
        raise cancelled
    return result


def _ui_callback(name, anchor):
    from tldw_chatbook.UI.Console_Modules import execution_state

    callback = getattr(execution_state, name)
    definition = getattr(execution_state, anchor)
    if callback is not definition[0] or not _definition_current(definition):
        raise ConsoleChannelLost("Console rail publication callback changed")
    return callback


def _writer_retired(scope):
    state, operation = scope.writer_state, scope.writer_operation
    with storage_admission._lock:
        return (
            state is not None and operation is not None and state.source is config
            and operation not in raw_participants._states
            and operation not in storage_admission._raw_operations
            and local_permission_operation._state_resources_retired(state, storage_admission)
        )


def _write_rail_preference(scope):
    """Observe the original setter's exact config owner on its actual worker."""
    require_rail_publication(scope)
    if (not _positive(scope.fence) or scope.client._source_revision != scope.revision
        or scope.client._intake_open):
        raise ConsoleChannelLost("Console rail writer has no retired source fence")
    if not config_participants.first_send_persistence_source_current(config):
        raise ConsoleChannelLost("Original Console rail writer changed")
    try:
        with config_participants.operation(config) as operation:
            scope.writer_operation = operation
            scope.writer_state = raw_participants._states[operation]
            result = config.save_setting_to_cli_config(
                "console.rail_state", scope.key, dict(scope.value),
            )
    finally:
        scope.writer_retired = _writer_retired(scope)
    if not scope.writer_retired:
        raise ConsoleChannelLost("Console rail writer retirement is unproven")
    return result


async def publish_rail_preference(screen, key, value):
    """Retain the real writer independently of the original worker's observer."""
    app = screen.app_instance
    client = app.console_execution_client
    if (
        type(client) is not ConsoleExecutionClient
        or client._loop is not asyncio.get_running_loop()
        or client._shutdown_task is not None or client._workspace_operations
        or getattr(client, "_rail_publication", None) is not None
        or client._project_modal is not None or client._project_modal_transition is not None
        or not client._intake_open
    ):
        raise ConsoleChannelLost("Console rail publication is unavailable")
    scope = ConsoleRailPublication(
        client, app, client._view, client._visit, client._attachment_id,
        client._source_revision, client._source_revision + 1, client.projection_view,
        (client.profile_epoch, client.service_epoch), key, MappingProxyType(dict(value)),
    )
    client._rail_publication = scope

    async def publish():
        await asyncio.sleep(0)  # Retain this exact Task before an eager first entry.
        completed = False
        try:
            require_rail_publication(scope)
            # Newly scheduled passive work sees the retained publication. Join
            # actual earlier readers/edits before closing their source intake.
            prior = tuple(client._projection_reads | client._view_checkpoints | client._input_tasks)
            if prior:
                await asyncio.gather(*(asyncio.shield(task) for task in prior))
            require_rail_publication(scope)
            current_view = client.projection_view
            previous = scope.projection_view
            if previous is not None and (
                current_view is None
                or current_view.active_session_id != previous.active_session_id
                or current_view.session(current_view.active_session_id).incarnation_id
                != previous.session(previous.active_session_id).incarnation_id
            ):
                raise ConsoleChannelLost("Console rail save session changed before its fence")
            scope.projection_view = current_view
            scope.fence = client.submit("FenceSource", {
                "source": "configuration",
                "rail_update": {"key": key, "value": dict(scope.value)},
            })
            if await _settle(scope.fence) != {"source": "configuration", "revision": scope.revision}:
                raise ConsoleChannelLost("Console rail fence result changed")
            require_rail_publication(scope)
            scope.writer = client._loop.run_in_executor(
                None, _write_rail_preference, scope,
            )
            scope.writer_result = await asyncio.shield(scope.writer)
            if scope.writer_result is not True:
                raise ConsoleChannelLost("Original Console rail save did not publish")
            require_rail_publication(scope)
            scope.refresh = client.submit("RefreshConfiguration", {
                "fence_request_id": scope.fence.identity.request_id,
            })
            if await _settle(scope.refresh) != {"source_revision": scope.revision}:
                raise ConsoleChannelLost("Console rail reload result changed")
            scope.applied = client.submit("ApplySourceRevision", {
                "publication_request_id": scope.refresh.identity.request_id,
            })
            await _settle(scope.applied)
            require_rail_publication(scope, applied=True)
            await _ui_callback("finish_selected_rail_publication", "_RAIL_FINISH")(scope)
            completed = True
            return True
        except BaseException as error:
            # An observer cannot turn a live synchronous writer into retirement.
            if scope.writer is not None:
                while not scope.writer.done():
                    try:
                        await asyncio.shield(scope.writer)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                if not scope.writer.cancelled():
                    scope.writer.exception()
            client._fail(error)
            raise
        finally:
            if getattr(client, "_rail_publication", None) is scope:
                client._rail_publication = None
                if completed:
                    _ui_callback("rail_publication_finished", "_RAIL_FINISHED")(scope)

    task = asyncio.create_task(publish(), name="console-rail-publication")
    scope.task = task
    client._workspace_operations.add(task)
    task.add_done_callback(client._workspace_operations.discard)
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    return await asyncio.shield(task)


def save_rail_preference(screen, key, value):
    """Keep the original worker lock/seed order and default local setter."""
    if getattr(screen.app_instance, "console_execution_client", None) is None:
        return config.save_setting_to_cli_config("console.rail_state", key, value)
    return screen.app.call_from_thread(publish_rail_preference, screen, key, value)


_DEFINITIONS = capture_definitions((
    (config, "save_setting_to_cli_config", config.save_setting_to_cli_config),
    (config_participants, "operation", config_participants.operation),
    (config_participants, "first_send_persistence_source_current", config_participants.first_send_persistence_source_current),
    (local_permission_operation, "_state_resources_retired", local_permission_operation._state_resources_retired),
    (ConsoleExecutionClient, "submit", ConsoleExecutionClient.submit),
    (ConsoleExecutionClient, "require_view", ConsoleExecutionClient.require_view),
    *((globals(), name, globals()[name]) for name in (
        "_positive", "require_rail_publication", "rail_publication_for", "_settle", "_ui_callback",
        "publish_rail_preference", "save_rail_preference", "_writer_retired", "_write_rail_preference",
    )),
))
_RAIL_LOOKUP = _definition(rail_publication_for)
_RAIL_REQUIRE = _definition(require_rail_publication)
