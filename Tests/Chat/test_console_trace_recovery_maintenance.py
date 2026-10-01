"""Parked trace recoveries retain their live preparation until explicit cleanup."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from Tests.Chat.test_console_turn_preparation import _preparation_values
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_runtime_trace_maintenance_preserves_parked_recovery_until_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    activity_checks = []

    class Maintenance:
        def __init__(self, _database, *, provider_active, **_kwargs):
            activity_checks.append(provider_active)
            started.set()

        def run_batch(self):
            return SimpleNamespace(logical_complete=False, admitted=False)

    module = ModuleType("tldw_chatbook.Chat.console_trace_maintenance")
    module.LegacyTraceMaintenance = Maintenance
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime."
        "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS",
        0.0,
    )
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    preparation = ConsoleTurnPreparation(
        **_preparation_values(
            ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TRACE_PROVENANCE,
            session_id=session.id,
        )
    )
    assert store.begin_preparation(preparation) is preparation
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime = ConsoleRuntime(SimpleNamespace(_ui_ready=True))
    runtime.set_chat_controller(controller)
    try:
        assert controller._submit_tasks_snapshot() == {}
        assert controller._active_stream_tasks == {}
        runtime._schedule_legacy_trace_maintenance(object(), object)
        await asyncio.wait_for(started.wait(), timeout=5)
        assert activity_checks[0]() is True
        assert (
            store.remove_preparation(
                session.id,
                preparation.preparation_id,
                expected_states=frozenset({ConsoleTurnPreparationState.PAUSED}),
            )
            is preparation
        )
        assert activity_checks[0]() is False
    finally:
        await runtime.dispose()
