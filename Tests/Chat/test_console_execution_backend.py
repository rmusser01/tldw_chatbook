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


@private_profile_test
@pytest.mark.asyncio
async def test_backend_source_checks_retain_selection_without_native_reacquisition(
    request, monkeypatch, tmp_path
):
    import sys
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        ConsoleBackendProfile,
        open_console_backend_profile,
    )

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="backend"
    )
    entries = []
    original = config.get_user_data_dir

    def entered(frame):
        if frame.f_locals.get("function") is original.__wrapped__:
            entries.append(True)

    try:
        backend.start()
        with _observe(original.__code__, sys.monitoring.events.PY_START, entered):
            for _ in range(3):
                backend.require_sources()
                backend.presentation.require_association(
                    backend.presentation.association
                )
            for section, key in (
                ("paths", "data_dir"),
                ("Paths", "data_dir"),
                ("general", "users_name"),
            ):
                with monkeypatch.context() as change:
                    change.setitem(config._CONFIG_CACHE, section, {key: "changed"})
                    with pytest.raises(
                        RuntimeError, match="source association changed"
                    ):
                        backend.require_sources()
            with monkeypatch.context() as change:
                change.setitem(
                    config.DEFAULT_CONFIG_FROM_TOML,
                    "general",
                    {"users_name": "changed"},
                )
                with pytest.raises(RuntimeError, match="source association changed"):
                    backend.require_sources()
            for name, value in (
                ("_CONFIG_CACHE", dict(config._CONFIG_CACHE)),
                ("_CONFIG_GENERATION", config._CONFIG_GENERATION + 1),
                ("_CONFIG_CACHE_SOURCE", "changed"),
                ("get_user_data_dir", lambda: directory),
            ):
                with monkeypatch.context() as change:
                    change.setattr(config, name, value)
                    with pytest.raises(
                        RuntimeError, match="source association changed"
                    ):
                        backend.require_sources()
            with monkeypatch.context() as change:
                change.setattr(
                    ConsoleBackendProfile, "selection_is_current", lambda self: True
                )
                with pytest.raises(RuntimeError, match="source association changed"):
                    backend.require_sources()
            for name in (
                "TLDW_CONFIG_PATH",
                "HOME",
                "USERPROFILE",
                "HOMEDRIVE",
                "HOMEPATH",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
            ):
                with monkeypatch.context() as change:
                    change.setenv(name, str(tmp_path / name))
                    with pytest.raises(
                        RuntimeError, match="source association changed"
                    ):
                        backend.require_sources()
            with monkeypatch.context() as change:
                change.chdir(tmp_path)
                with pytest.raises(RuntimeError, match="source association changed"):
                    backend.require_sources()
            backend.require_sources()
        assert entries == []
        request.node.user_properties.append(
            ("association_native_getter_entries", len(entries))
        )
    finally:
        await backend.aclose()
    assert backend.closed and backend.runtime._disposed
