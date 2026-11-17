"""Actual backend graph controls, separate from transport and mounted UI."""

from __future__ import annotations

import asyncio

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_backend_constructs_one_original_graph_and_permission_owner(request):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="backend"
    )

    def command(sequence, operation, payload, *, session_id="", incarnation_id=""):
        return ConsoleWireMessage(
            WireIdentity(
                "profile",
                "backend",
                sequence,
                f"backend-{sequence}",
                session_id,
                incarnation_id,
            ),
            "command",
            operation,
            payload,
        )

    try:
        backend.start()
        assert backend.runtime._app is None
        assert backend.runtime.domain_services is backend.domain
        assert backend.domain.sources.chat_database is profile.storage.chachanotes
        assert backend.domain.sources.prompt_scope is profile.prompt_scope
        assert (
            backend.controller.prompt_mixin_service().prompt_scope_service
            is profile.prompt_scope
        )
        assert (
            backend.controls._permissions._store
            is profile.control_plane.permission_store
        )
        opened = backend.dispatch(
            command(
                1,
                "OpenSession",
                {
                    "title": "Original backend chat",
                    "workspace_id": None,
                    "conversation_id": None,
                },
            )
        )
        session_id, incarnation = await asyncio.shield(opened.applied)
        assert backend.store._sessions[session_id].incarnation_id == incarnation
        permission = backend.dispatch(
            command(
                2,
                "PermissionAction",
                {
                    "operation": {"kind": "ReadKillSwitch"},
                },
            )
        )
        assert type(await asyncio.shield(permission.applied)) is bool
        assert await asyncio.shield(permission.retired) is True
        projection = backend.dispatch(
            command(
                3,
                "ReadProjection",
                {},
                session_id=session_id,
                incarnation_id=incarnation,
            )
        )
        displayed = await asyncio.shield(projection.applied)
        assert displayed.session_id == session_id
        assert displayed.incarnation_id == incarnation
        assert await asyncio.shield(projection.retired) is True
        assert backend.runtime.ensure_chat_store() is backend.store
        assert backend.runtime.chat_controller is backend.controller
        shutdown = backend.dispatch(command(4, "Shutdown", {}))
        assert await asyncio.shield(shutdown.applied) is True
        assert backend._shutdown_receipt is shutdown
    finally:
        await backend.aclose()
    assert backend.runtime._disposed
    assert backend.closed
    assert backend._shutdown_receipt is shutdown
    assert backend._sequence == 4
