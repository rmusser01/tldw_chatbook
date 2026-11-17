"""Actual-owner lifecycle ordering for the qualified fresh-session boundary."""

import sys
from dataclasses import replace

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_actual_fresh_view_supersession_no_seed_and_repeated_guarded_drain(
    request, monkeypatch
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat import console_backend_view_lifecycle as lifecycle_source
    from tldw_chatbook.Chat.console_backend_view_lifecycle import (
        ConsoleBackendViewCheckpoint,
        ConsoleBackendViewLifecycle,
        ConsoleBackendViewSuperseded,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="view-lifecycle"
    )
    visibility = []
    seed_entries = []
    try:
        backend.start()
        lifecycle = backend.presentation.view_lifecycle
        assert type(lifecycle) is ConsoleBackendViewLifecycle
        session = backend.store.create_session(title="Fresh lifecycle")
        assert session.persisted_conversation_id is None

        def visibility_entry(frame):
            assert frame.f_locals["self"] is backend.controller
            visibility.append(frame.f_locals["visible"])

        with (
            _observe(
                ConsoleChatController.on_console_view_visibility_changed.__code__,
                sys.monitoring.events.PY_START,
                visibility_entry,
            ),
            _observe(
                ConsoleFleetWakeCoordinator._seed_owned_history.__code__,
                sys.monitoring.events.PY_START,
                lambda frame: seed_entries.append(frame.f_locals["self"]),
            ),
        ):
            # Compose can be cancelled before it emits a mount checkpoint.
            cancelled_compose = lifecycle.admit("actual-client", 1, "hidden")
            assert lifecycle._seeds == {}
            await lifecycle.prepare(cancelled_compose)
            assert visibility == [False]

            mount = lifecycle.admit("actual-client", 2, "mount")
            hidden = lifecycle.admit("actual-client", 2, "hidden")
            # Admission invalidates old visibility before either task runs.
            with pytest.raises(ConsoleBackendViewSuperseded):
                await lifecycle.prepare(mount)
            assert visibility == [False]
            await lifecycle.prepare(hidden)
            assert visibility == [False, False]

            # Copies and newly constructed values are not retained checkpoints,
            # and must not be misclassified as a harmless superseded request.
            for unissued in (
                replace(hidden),
                ConsoleBackendViewCheckpoint("actual-client", 2, "hidden"),
            ):
                with pytest.raises(RuntimeError) as refusal:
                    await lifecycle.prepare(unissued)
                assert type(refusal.value) is RuntimeError

            mount = lifecycle.admit("actual-client", 3, "mount")
            visible = lifecycle.admit("actual-client", 3, "visible")
            await lifecycle.prepare(visible)
            assert visibility == [False, False, True]
            assert backend.controller._interrupt_host.view_visible
            with pytest.raises(ConsoleBackendViewSuperseded):
                await lifecycle.prepare(mount)
            assert visibility == [False, False, True]
            assert seed_entries == []  # Actual fresh sessions require no seed.
            assert lifecycle._tasks == []

            # Completed/unissued prior visits do not retain their native owners.
            hidden = lifecycle.admit("actual-client", 4, "hidden")
            assert lifecycle._seeds == {}
            await lifecycle.prepare(hidden)
            assert not backend.controller._interrupt_host.view_visible
            assert visibility == [False, False, True, False]

        await lifecycle.drain()
        assert lifecycle._drained
        await lifecycle.drain()
        await backend.aclose()
        assert backend.closed and backend.runtime._disposed
        await lifecycle.drain()  # Exact positive result survives native disposal.
        with monkeypatch.context() as changed:
            changed.setattr(lifecycle_source, "_retire_seed", object())
            with pytest.raises(RuntimeError, match="drain definitions changed"):
                await lifecycle.drain()
    finally:
        await backend.aclose()
