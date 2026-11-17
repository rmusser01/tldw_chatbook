"""Workspace access uses original source publication and retained wire receipts."""

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


async def _until(predicate):
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()


def _client():
    from Tests.Chat.test_console_execution_process import _ready_client

    client = _ready_client()
    client._app = SimpleNamespace(console_runtime=None, console_execution_client=client)
    return client


def _reply(client, receipt, kind, payload):
    from Tests.Chat.test_console_execution_process import _reply as message

    client._receive(
        message(client, receipt, client._received_sequence + 1, kind, payload)
    )


def _applied(client, receipt, payload):
    _reply(client, receipt, "received", {"sequence": receipt.identity.sequence})
    _reply(client, receipt, "applied", payload)


def _retired(client, receipt, retired=True):
    _reply(
        client,
        receipt,
        "retired",
        {
            "operation_id": receipt.identity.request_id,
            "retired": retired,
        },
    )


@pytest.mark.asyncio
async def test_workspace_transport_waits_for_retirement_despite_observer_cancel():
    from tldw_chatbook.Chat.console_execution_workspace import (
        set_workspace_binding_access,
    )

    client = _client()
    operation = set_workspace_binding_access(
        client, "workspace", "binding", allow_write=True
    )

    async def observe():
        return await asyncio.shield(operation)

    observer = asyncio.create_task(observe())
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        fence = next(iter(client._requests.values()))
        _applied(client, fence, {"source": "workspace", "revision": 1})
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert operation in client._workspace_operations and not operation.done()
        assert len(client._channel.messages) == 1
        _retired(client, fence)
        await _until(lambda: len(client._channel.messages) == 2)
        publication = next(iter(client._requests.values()))
        body = {
            "binding_id": "binding",
            "workspace_id": "workspace",
            "access": "rw",
            "source_revision": 1,
        }
        _applied(client, publication, body)
        await asyncio.sleep(0)
        assert len(client._channel.messages) == 2 and not client._intake_open
        _retired(client, publication)
        await _until(lambda: len(client._channel.messages) == 3)
        apply = next(iter(client._requests.values()))
        _applied(client, apply, {"source_revision": 1})
        assert not operation.done()
        _retired(client, apply)
        assert await operation == body
        assert client._intake_open
        assert [message.operation for message in client._channel.messages] == [
            "FenceSource",
            "SetWorkspaceBindingAccess",
            "ApplySourceRevision",
        ]
    finally:
        if not operation.done():
            client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["retirement", "workspace"])
async def test_workspace_transport_never_applies_unretired_or_wrong_publication(
    failure,
):
    from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
    from tldw_chatbook.Chat.console_execution_workspace import (
        set_workspace_binding_access,
    )

    client = _client()
    operation = set_workspace_binding_access(
        client, "workspace", "binding", allow_write=True
    )
    await _until(lambda: bool(client._requests))
    fence = next(iter(client._requests.values()))
    _applied(client, fence, {"source": "workspace", "revision": 1})
    _retired(client, fence)
    await _until(lambda: len(client._channel.messages) == 2)
    publication = next(iter(client._requests.values()))
    _applied(
        client,
        publication,
        {
            "binding_id": "binding",
            "workspace_id": "other" if failure == "workspace" else "workspace",
            "access": "rw",
            "source_revision": 1,
        },
    )
    _retired(client, publication, failure != "retirement")
    with pytest.raises(ConsoleChannelLost):
        await operation
    assert len(client._channel.messages) == 2 and not client._intake_open


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("close_intake", [False, True])
async def test_backend_workspace_wire_uses_original_registry_and_refuses_late_apply(
    request, tmp_path, close_intake
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_process import _SentFrames
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )
    from tldw_chatbook.Chat.console_execution_worker import _BackendService
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    registry = profile.workspace_registry
    folder = tmp_path / "settings bound folder"
    folder.mkdir()
    with project_index_registry_connections(registry.db):
        registry.create_workspace(
            workspace_id="settings-workspace", name="Settings workspace"
        )
        binding = registry.add_folder_binding("settings-workspace", str(folder))
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="workspace"
    )
    channel = _SentFrames()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def command(sequence, operation, payload, revision=0):
        return ConsoleWireMessage(
            WireIdentity(
                "profile",
                "workspace",
                sequence,
                f"workspace-{sequence}",
                source_revision=revision,
            ),
            "command",
            operation,
            payload,
        )

    def messages(request_id):
        return [
            message
            for message in channel.messages
            if message.identity.request_id == request_id
        ]

    def original_setter(frame):
        if frame.f_locals.get("self") is registry:
            calls.append(frame.f_locals["binding_id"])
            if close_intake:
                entered.set()
                assert release.wait(10), "Original workspace setter was not released"

    change_task = None
    try:
        backend.start()
        service = _BackendService(backend, channel)
        await service.accept(command(1, "FenceSource", {"source": "workspace"}))
        assert messages("workspace-1")[-1].payload["retired"] is True
        with _observe(
            LocalWorkspaceRegistryService.set_folder_binding_access.__code__,
            sys.monitoring.events.PY_RETURN,
            original_setter,
        ):
            change_task = service.accept(
                command(
                    2,
                    "SetWorkspaceBindingAccess",
                    {
                        "fence_request_id": "workspace-1",
                        "binding_id": binding.binding_id,
                        "allow_write": True,
                    },
                    1,
                )
            )
            if close_intake:
                await _until(entered.is_set)
                await service.accept(command(3, "CloseIntake", {}, 1))
                assert not any(
                    item.kind == "applied" for item in messages("workspace-2")
                )
            release.set()
            await change_task
        assert calls == [binding.binding_id]
        assert backend.domain.sources.workspace_registry is registry
        with project_index_registry_connections(registry.db):
            assert (
                registry.get_runtime_binding(binding.binding_id).metadata["access"]
                == "rw"
            )
        outcomes = messages("workspace-2")
        if close_intake:
            assert [item.kind for item in outcomes] == ["received", "failed", "retired"]
            assert outcomes[-1].payload["retired"] is False
            assert not backend.controls._intake_open
        else:
            assert [item.kind for item in outcomes] == [
                "received",
                "applied",
                "retired",
            ]
            assert outcomes[1].payload == {
                "binding_id": binding.binding_id,
                "workspace_id": "settings-workspace",
                "access": "rw",
                "source_revision": 1,
            }
            assert outcomes[-1].payload["retired"] is True
            await service.accept(
                command(
                    3,
                    "ApplySourceRevision",
                    {
                        "publication_request_id": "workspace-2",
                    },
                    1,
                )
            )
            assert messages("workspace-3")[-1].payload["retired"] is True
            assert backend.controls._intake_open
    finally:
        release.set()
        if change_task is not None:
            await asyncio.gather(change_task, return_exceptions=True)
        await backend.aclose()
