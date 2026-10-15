"""Preserve native cost refusal and fresh retry after live owner/source changes."""

from __future__ import annotations

import asyncio
import sys
import time

import pytest
from tldw_chatbook.UI.Console_Modules.context_spend import ConsoleContextSpendController

from Tests.UI.test_console_cost_chip_screen import (
    WARM_USAGE,
    ConsoleHarness,
    _AnthropicCostGateway,
    _build_test_app,
    _configure_anthropic_ready_console,
    _mount_and_send_warm_reply,
    _settle_successful_full,
    attach_chachanotes_db,
    context_spend_for_test,
)
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import config_participants

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("change", ["owner", "source", "runtime"])
async def test_cost_settings_only_native_postbody_refusal_and_fresh_retry(change):
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, first_id = await _mount_and_send_warm_reply(console, pilot)
        await _settle_successful_full(console, pilot)
        owner = context_spend_for_test(console)
        spend = sys.modules["tldw_chatbook.UI.Console_Modules.console_spend_projection"]
        sync_module = sys.modules["tldw_chatbook.UI.Console_Modules.config_sync"]
        runtime_module = sys.modules["tldw_chatbook.Chat.console_runtime"]
        fixture = sys.modules["Tests.UI.test_console_cost_chip_screen"]
        projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
        runtime = spend._readiness_runtime(console)
        runtime_attr = runtime_module.CONSOLE_RUNTIME_ATTR
        assert type(projection) is spend.ConsoleReadinessConfigProjection
        assert vars(app)[runtime_attr] is runtime is projection._runtime_owner
        assert "_console_runtime_fallback" not in vars(console)
        assert not runtime._disposed and not console._closing
        first = owner._active_native_session()
        second = store.create_session(settings=store.session_settings(first_id), ephemeral=True, activate=False)
        previous_state = owner._last_console_cost_state
        previous_display = (owner._console_cost_cache_state, owner._console_settled_spend_cache,
                            owner._console_cost_projection_tokens_cache)
        timer = owner._console_cost_ttl_timer
        timer_task = None if timer is None else timer._task
        assert previous_state is not None and timer is not None and timer._interval == 10.0
        assert timer_task is not None and not timer_task.done()
        assert owner._last_console_context_control_state is not None
        original_composer = owner._console_composer_or_none
        previous_profile = sys.getprofile()
        phase = "refusal"
        native_rows, presentation_rows, operation_rows, derives, replays, builds, callback_rows = [], [], [], [], [], [], []
        build_code = ConsoleContextSpendController._build_console_cost_state.__code__
        derive_code = ConsoleContextSpendController._build_console_cost_state_under_config.__code__
        native_code = sync_module.run_console_config_sync.__code__
        presentation_code = sync_module.run_console_presentation_sync.__code__
        operation_code = config_participants.operation.__wrapped__.__code__
        replay_code = type(console)._run_coalesced_control_bar_sync.__code__

        def profile(frame, event, value):
            if not (
                (event == "call" and (
                    frame.f_code is native_code
                    or frame.f_code is presentation_code
                    or frame.f_code is operation_code
                    or frame.f_code is derive_code
                    or frame.f_code is replay_code
                ))
                or (event == "return" and frame.f_code is build_code)
            ):
                if previous_profile is not None:
                    previous_profile(frame, event, value)
                return
            parent = frame.f_back
            build_frame = None
            while parent is not None:
                if parent.f_code is build_code and parent.f_locals.get("self") is owner:
                    build_frame = parent
                    break
                parent = parent.f_back
            if event == "call":
                if frame.f_code is native_code and build_frame is not None:
                    native_rows.append((phase, build_frame.f_locals.get("native_key") is not None))
                if frame.f_code is presentation_code and frame.f_locals.get("screen") is console:
                    presentation_rows.append(phase)
                if frame.f_code is operation_code and build_frame is not None:
                    operation_rows.append((phase, frame.f_locals.get("source") is config))
                if frame.f_code is derive_code and frame.f_locals.get("self") is owner:
                    derives.append((phase, frame.f_locals["session"], frame.f_locals["store"]))
                if frame.f_code is replay_code and frame.f_locals.get("self") is console:
                    replays.append(phase)
            if event == "return" and frame.f_code is build_code and frame.f_locals.get("self") is owner:
                builds.append((phase, frame.f_locals.get("native_key"),
                               frame.f_locals.get("native_current"), frame.f_locals.get("accepted"), value))
            if previous_profile is not None:
                previous_profile(frame, event, value)

        def mutate_after_real_derivation():
            result = original_composer()
            if not callback_rows:
                callback_rows.append((owner._console_settled_spend_cache is not previous_display[1],
                                      owner._console_cost_projection_tokens_cache is not previous_display[2]))
                if change == "owner":
                    store.switch_session(second.id)
                elif change == "source":
                    _configure_anthropic_ready_console(app, model="pr3023-current-retry-model")
                    store.switch_session(second.id)
                else:
                    delattr(app, runtime_attr)
                    assert spend._readiness_runtime(console) is None
            return result

        try:
            # Both mutations are genuine Store APIs; only settings revision
            # changes the published projection key. No proof/key is installed.
            message = next(m for m in store.messages_for_session(first_id) if m.role is fixture.ConsoleMessageRole.USER)
            store.update_message_content(message.id, "EDITED EARLIER HISTORY")
            updated, persisted = store.set_session_system_prompt(first_id, "Cost native revision " * 8)
            assert updated is first and persisted
            native_key = projection._key()
            assert spend._same_readiness_key(native_key, projection.key, ignore_settings=True)
            assert not spend._same_readiness_key(native_key, projection.key)
            previous_source = config.current_config_identity()
            owner._console_composer_or_none = mutate_after_real_derivation
            sys.setprofile(profile)
            try:
                console._sync_console_cost_chip()
            finally:
                owner._console_composer_or_none = original_composer
                # Restore the actual withdrawn owner before yielding. No new
                # Runtime, watcher, source, proof, or borrowed copy is created.
                if change == "runtime":
                    setattr(app, runtime_attr, runtime)
            assert native_rows == [("refusal", True)]
            assert "refusal" not in presentation_rows
            assert any(p == "refusal" and stock for p, stock in operation_rows)
            assert callback_rows == [(True, True)]
            assert ("refusal", first, store) in derives
            assert any(p == "refusal" and key is not None and current is False and accepted is True
                       and value is previous_state for p, key, current, accepted, value in builds)
            assert owner._last_console_cost_state is previous_state
            assert owner._console_cost_cache_state is previous_display[0]
            assert owner._console_settled_spend_cache is previous_display[1]
            assert owner._console_cost_projection_tokens_cache is previous_display[2]
            assert owner._console_cost_ttl_timer is timer and timer._task is timer_task and not timer_task.done()
            assert spend._readiness_runtime(console) is runtime
            current_source = config.current_config_identity()
            assert (current_source != previous_source) == (change == "source")
            wanted = first if change == "runtime" else second
            phase = "retry"
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not (
                (phase, wanted, store) in derives and phase in replays
                and any(p == phase and accepted is True and current is True for p, _, current, accepted, _ in builds)
            ):
                await pilot.pause(0.01)
            assert (phase, wanted, store) in derives and phase in replays
            assert any(p == phase and accepted is True and current is True for p, _, current, accepted, _ in builds)
            assert owner._active_native_session() is wanted and owner._current_chat_store() is store
            assert projection.key[0] == current_source
            assert owner._last_console_cost_state is not None
            if change != "runtime":
                assert owner._console_cost_ttl_timer is None and timer._task is None
                await asyncio.gather(timer_task, return_exceptions=True)
                assert timer_task.done()
            assert len(gateway.sent_messages) == 1
        finally:
            try:
                sys.setprofile(previous_profile)
            finally:
                owner._console_composer_or_none = original_composer
                if change == "runtime":
                    setattr(app, runtime_attr, runtime)
