"""Mounted sends expose pre-provider failures through ordinary support logs."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from Tests.Chat.test_console_send_diagnostics import assert_export, sinks  # noqa: F401
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _persist_console_provider_config
from Tests.UI.test_console_rail_refresh_scope import count_compositor_updates
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("failure", ["admission", "resolution", "trace", None])
@pytest.mark.parametrize(
    "entrypoint,size",
    [("direct", (80, 24)), ("enter", (80, 24)), ("enter", (160, 45))],
)
async def test_mounted_send_has_diagnostic_evidence_before_provider_entry(
    monkeypatch,
    tmp_path,
    sinks,  # noqa: F811 - imported shared pytest fixture
    failure,
    entrypoint,
    size,
):
    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "send-diagnostic")
    app.chachanotes_db = database
    # A new conversation in an existing profile, as in the reported incident.
    for index in range(4):
        assert database.add_conversation({"title": f"Existing conversation {index}"})
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": "synthetic-test-key"},
    )
    host = ConsoleHarness(app)
    host.CSS_PATH = TldwCli.CSS_PATH
    calls = []

    def fail_factory(*args):
        raise ValueError("PRIVATE-DRAFT-31977")

    if failure == "trace":
        monkeypatch.setattr(ConsoleTraceBoundaryFactory, "__call__", fail_factory)
    console = None
    try:
        async with host.run_test(size=size) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            controller = console._ensure_console_chat_controller()
            runtime = console._console_runtime()
            turn_tasks = []
            accept_turn = runtime.accept_turn

            def capture_turn(request, **kwargs):
                if failure == "admission":
                    raise ValueError("PRIVATE-DRAFT-31977")
                turn_id = accept_turn(request, **kwargs)
                turn_tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", capture_turn)
            gateway = controller.provider_gateway
            original_resolve = gateway.resolve_for_send

            async def resolve(selection):
                if failure == "resolution":
                    raise ValueError("PRIVATE-DRAFT-31977")
                return replace(await original_resolve(selection), streaming=False)

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)

            def adapter(**kwargs):
                calls.append(True)
                return {"choices": [{"message": {"content": "Synthetic response"}}]}

            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            controller.store.ensure_session()
            composer = console._console_composer_or_none()
            composer.load_draft("PRIVATE-DRAFT-31977")
            if entrypoint == "enter":
                composer.focus()
                await pilot.pause()
                await pilot.press("enter")
                await asyncio.wait_for(host.workers.wait_for_complete(), 10)
            else:
                sent = await asyncio.wait_for(
                    console._send_console_message_from_visible_action(),
                    10,
                )
                assert sent is (failure != "admission")
            if failure == "admission":
                assert turn_tasks == []
                assert console._console_composer_or_none().draft_text() == (
                    "PRIVATE-DRAFT-31977"
                )
                await asyncio.to_thread(app.ui_responsiveness_monitor.close)
                text = assert_export(
                    sinks,
                    "phase=ui_action",
                    "phase=ui_dispatch",
                    "phase=ui_submit",
                    "status=failed",
                    "error_category=validation",
                )
                assert "phase=controller_submit" not in text
                assert calls == []
                return
            assert len(turn_tasks) == 1
            if failure == "resolution":
                with pytest.raises(ValueError, match="PRIVATE-DRAFT-31977"):
                    await asyncio.wait_for(turn_tasks[0], 10)
            else:
                await asyncio.wait_for(turn_tasks[0], 10)
            await asyncio.wait_for(host.workers.wait_for_complete(), 10)
            await pilot.pause(0.3)
            # Count the compositor's actual output, including redraws caused by
            # focus or child layout; screen-recompose counts alone miss those.
            updates = {"full": 0, "partial": 0}
            with count_compositor_updates(updates):
                await pilot.pause(1.2)
            assert updates["full"] == 0, updates
            assert console._console_pending_send is None
            async with asyncio.timeout(5):
                while console._console_transcript_sync_timer is not None:
                    await pilot.pause(0.05)
            if failure == "resolution":
                assert composer.draft_text() == ""
                session_id = controller.store.active_session_id
                (recovery,) = runtime.recoveries_for_session(session_id)
                assert recovery.session_id == session_id
                assert recovery.draft == "PRIVATE-DRAFT-31977"
                await console._prompt_queue.handle_primary_intent(
                    session_id,
                    action=f"turn-recovery:restore:{recovery.turn_id}",
                    expected_revision=controller.lifecycle_impact(
                        session_id=session_id
                    ).revision,
                )
                assert composer.draft_text() == "PRIVATE-DRAFT-31977"
                assert runtime.recoveries_for_session(session_id) == ()
                assert controller.run_state.is_send_allowed
            await asyncio.to_thread(app.ui_responsiveness_monitor.close)
            assert len(calls) == (0 if failure else 1)
            expected_phase = (
                "provider_resolution"
                if failure == "resolution"
                else "trace_reservation"
            )
            text = assert_export(
                sinks,
                "phase=ui_action",
                "phase=ui_dispatch",
                "phase=ui_submit",
                "phase=controller_submit",
                f"phase={expected_phase}",
            )
            if failure:
                assert "status=failed" in text
                assert "error_category=validation" in text
                assert "phase=provider_entry" not in text
            else:
                assert "phase=provider_entry" in text
                assert "status=completed" in text
            if failure != "resolution":
                assert "capture_enabled=true" in text
            assert "synthetic-test-key" not in text
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        try:
            with database.quiesce_connections(timeout_seconds=5):
                assert database.registered_connection_count() == 0
        finally:
            await asyncio.to_thread(app.ui_responsiveness_monitor.close)
            database.close()
