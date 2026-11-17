"""Retain the original workspace source publication on its selected client."""

from __future__ import annotations

import asyncio

from .console_execution_process import ConsoleChannelLost, ConsoleExecutionClient


def set_workspace_binding_access(
    client: ConsoleExecutionClient,
    workspace_id: str,
    binding_id: str,
    *,
    allow_write: bool,
) -> asyncio.Task:
    """Retain Fence, original Registry mutation and Apply through client shutdown."""
    if (
        type(client) is not ConsoleExecutionClient
        or client._loop is not asyncio.get_running_loop()
        or client._shutdown_task is not None
        or client._workspace_operations
        or not client._intake_open
        or type(workspace_id) is not str
        or not workspace_id
        or type(binding_id) is not str
        or not binding_id
        or type(allow_write) is not bool
    ):
        raise ConsoleChannelLost("Console workspace access is unavailable")
    app = client._app
    generation = client.profile_epoch, client.service_epoch
    initial_revision = client._source_revision

    def current(revision):
        if (
            getattr(app, "console_execution_client", None) is not client
            or getattr(app, "console_runtime", None) is not None
            or (client.profile_epoch, client.service_epoch) != generation
            or client._source_revision != revision
            or client._disposed
            or client._channel_error is not None
        ):
            raise ConsoleChannelLost("Console workspace source changed")

    current(initial_revision)

    async def result(receipt):
        value = await asyncio.shield(receipt.applied)
        if await asyncio.shield(receipt.retired) is not True:
            raise ConsoleChannelLost("Console workspace operation did not retire")
        return value

    async def publish():
        # The actual task is retained before an eager task factory can issue work.
        await asyncio.sleep(0)
        current(initial_revision)
        fence = client.submit("FenceSource", {"source": "workspace"})
        fenced = await result(fence)
        revision = initial_revision + 1
        current(revision)
        if fenced != {"source": "workspace", "revision": revision}:
            raise ConsoleChannelLost("Console workspace fence changed")
        changed = client.submit(
            "SetWorkspaceBindingAccess",
            {
                "fence_request_id": fence.identity.request_id,
                "binding_id": binding_id,
                "allow_write": allow_write,
            },
        )
        publication = await result(changed)
        current(revision)
        if publication != {
            "binding_id": binding_id,
            "workspace_id": workspace_id,
            "access": "rw" if allow_write else "ro",
            "source_revision": revision,
        }:
            raise ConsoleChannelLost("Console workspace publication changed")
        applied = await result(
            client.submit(
                "ApplySourceRevision",
                {"publication_request_id": changed.identity.request_id},
            )
        )
        current(revision)
        if applied != {"source_revision": revision} or not client._intake_open:
            raise ConsoleChannelLost("Console workspace revision was not applied")
        return publication

    task = asyncio.create_task(publish(), name="console-workspace-binding-access")
    client._workspace_operations.add(task)
    task.add_done_callback(client._workspace_operations.discard)
    task.add_done_callback(
        lambda finished: None if finished.cancelled() else finished.exception()
    )
    return task
