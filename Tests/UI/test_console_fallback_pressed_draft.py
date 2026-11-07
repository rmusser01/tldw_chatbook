"""A custom-capture press must preserve an identically reauthored draft."""

from __future__ import annotations

import sys

import pytest

from Tests.UI.test_console_approval_compact_layout import (
    _wait_for_reconciled_console,
)
from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
    ConsoleComposerBar,
    _wait_for_selector,
    build,
    eager_tasks,
    paint_state,
    press,
    until,
)
from Tests.UI.test_console_send_admission_off_pump import (
    B_DRAFT,
    ENTRY_SECONDS,
    HeldMcpRead,
    _record_notices,
    _second_tab,
    _sent_or_queued,
    _users,
)
from Tests.UI.test_console_send_resend_guard import _idle
from Tests.UI.test_console_turn_resend_ui import _select_ready_llamacpp_console

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("hidden_owner", [False, True], ids=["visible", "hidden"])
async def test_custom_capture_keeps_identically_reauthored_pressed_draft(
    hidden_owner, record_property
):
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.MCP.console_snapshot import (
        _mcp_metadata_is_closed,
        standard_console_sources,
    )
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = host.screen_stack[-1]
            runtime = console._console_runtime()
            service = host.app_instance.unified_mcp_service
            local = service.local_service
            original_manifest = local.manifest_provider
            manifests = []
            hold = None

            def manifest():
                caller = sys._getframe(1)
                try:
                    tree = current_ordinary_tree()
                    observed = {
                        "caller": caller.f_code,
                        "local": caller.f_locals.get("self"),
                        "tree": tree,
                        "configuration": tree.configuration if tree else None,
                        "service": host.app_instance.unified_mcp_service,
                        "released": hold is not None and hold.release.is_set(),
                    }
                finally:
                    del caller
                manifests.append(observed)
                result = original_manifest()
                observed["returned"] = True
                return result

            try:
                await _wait_for_selector(console, pilot, "#console-native-composer")
                await _wait_for_reconciled_console(console, pilot)
                await _select_ready_llamacpp_console(console, pilot)
                composer = console.query_one(
                    "#console-native-composer", ConsoleComposerBar
                )
                store = console._ensure_console_chat_store()
                session_a = store.active_session_id
                session_b = None
                if hidden_owner:
                    session_a, session_b = await _second_tab(console, pilot, ready=True)
                    await _wait_for_reconciled_console(console, pilot)
                    await _select_ready_llamacpp_console(console, pilot)
                composer.load_draft(DRAFT)
                composer.focus()
                await until(
                    lambda: (
                        composer.draft_text() == DRAFT
                        and store.session_draft(session_a) == DRAFT
                        and console._console_visible_draft_session_id == session_a
                        and store.active_session_id == session_a
                        and host.focused is composer
                        and paint_state(host).draft_in_composer
                    )
                )
                original_inputs = store.session_input_snapshot(session_a)
                local.manifest_provider = manifest
                assert service is host.app_instance.unified_mcp_service
                assert standard_console_sources(service)
                assert not _mcp_metadata_is_closed(service)
                async with HeldMcpRead(service) as hold:
                    gateway.validation_release.set()
                    try:
                        press(host, "enter", "\r")
                        await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                        assert not hold.timed_out
                        assert hold.future.running() and not hold.future.done()
                        assert not hold.body_returned
                        assert hold.captured.service is service
                        assert hold.permission is service._permission_store
                        assert not manifests
                        assert gateway.stream_calls == 0
                        press(host, "ctrl+u")
                        await until(lambda: composer.draft_text() == "", timeout=10)
                        for character in DRAFT:
                            press(
                                host,
                                "space" if character == " " else character,
                                character,
                            )
                        await until(
                            lambda: (
                                composer.draft_text() == DRAFT
                                and store.session_draft(session_a) == DRAFT
                                and store.session_input_snapshot(
                                    session_a
                                ).draft_revision
                                > original_inputs.draft_revision
                            ),
                            timeout=10,
                        )
                        reauthored_revision = store.session_input_snapshot(
                            session_a
                        ).draft_revision
                        assert not hold.timed_out
                        if hidden_owner:
                            press(host, "alt+2")
                            await until(
                                lambda: (
                                    console._console_visible_draft_session_id
                                    == session_b
                                    and composer.draft_text() == B_DRAFT
                                ),
                                timeout=10,
                            )
                        assert not manifests
                        assert gateway.stream_calls == 0
                        assert not hold.body_returned and not hold.future.done()
                    finally:
                        hold.release.set()
                    await until(lambda: gateway.stream_calls == 1)
                    await _idle(host, console, pilot, session_a)
                    assert not hold.timed_out
                    assert _sent_or_queued(console, session_a) == [DRAFT]
                    assert _users(console, session_a) == [DRAFT]
                    assert gateway.stream_calls == 1
                    assert store.session_draft(session_a) == DRAFT, {
                        "stored_a": store.session_draft(session_a),
                        "composer": composer.draft_text(),
                        "reauthored_revision": reauthored_revision,
                        "final_revision": store.session_input_snapshot(
                            session_a
                        ).draft_revision,
                    }
                    if hidden_owner:
                        assert store.session_draft(session_b) == B_DRAFT
                        assert store.active_session_id == session_b
                        assert console._console_visible_draft_session_id == session_b
                        assert composer.draft_text() == B_DRAFT
                    else:
                        assert composer.draft_text() == DRAFT
                    assert manifests
                    tree = manifests[0]["tree"]
                    assert tree is not None and tree.app is host.app_instance
                    assert tree.runtime is runtime and tree.session.id == session_a
                    assert tree.footprint.opaque
                    assert tree.original_configuration.mcp_capture_pending
                    assert manifests[0]["configuration"].mcp_capture_pending
                    assert all(
                        row["tree"] is tree
                        and row["service"] is service
                        and row["local"] is local
                        and row["released"]
                        and row.get("returned")
                        and row["caller"]
                        in {
                            LocalMCPControlService.get_inventory.__code__,
                            LocalMCPControlService.catalog_fingerprint.__code__,
                        }
                        for row in manifests
                    ), manifests
                    assert tree.retired.done()
                    record_property(
                        "manifest_calls",
                        [
                            (
                                row["caller"].co_name,
                                row["configuration"].mcp_capture_pending,
                            )
                            for row in manifests
                        ],
                    )
                assert hold.observer_retired
                assert (
                    hold.body_returned
                    and hold.callback_returned
                    and hold.invoke_returned
                )
                assert hold.future.done() and hold.read.retired.done()
                record_property("native_read_physically_returned", True)
            finally:
                if hold is not None:
                    hold.release.set()
                manifest_owned = (
                    local.manifest_provider is manifest
                    or local.manifest_provider is original_manifest
                )
                local.manifest_provider = original_manifest
                gateway.validation_release.set()
                await runtime.dispose()
                assert manifest_owned
            assert local.manifest_provider is original_manifest
            assert runtime._disposed and not runtime._preparation_reads
            assert not runtime._ordinary_turns and not runtime._turn_custody
            assert not runtime.execution_capacity.snapshot().executions
            record_property("runtime_disposed_without_live_owners", True)


async def test_managed_custom_getter_refuses_pressed_draft_without_invoking_it(
    record_property,
):
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = host.screen_stack[-1]
            runtime = console._console_runtime()
            service = host.app_instance.unified_mcp_service
            missing = object()
            original_slot = vars(service).get("get_kill_switch", missing)
            original_getter = service.get_kill_switch
            getter_calls = []

            def get_kill_switch():
                getter_calls.append(True)
                return original_getter()

            try:
                await _wait_for_selector(console, pilot, "#console-native-composer")
                await _wait_for_reconciled_console(console, pilot)
                await _select_ready_llamacpp_console(console, pilot)
                composer = console.query_one(
                    "#console-native-composer", ConsoleComposerBar
                )
                store = console._ensure_console_chat_store()
                session_id = store.active_session_id
                composer.load_draft(DRAFT)
                composer.focus()
                await until(
                    lambda: (
                        composer.draft_text() == DRAFT
                        and store.session_draft(session_id) == DRAFT
                        and console._console_visible_draft_session_id == session_id
                        and host.focused is composer
                        and paint_state(host).draft_in_composer
                    )
                )
                notices = _record_notices(host)
                service.get_kill_switch = get_kill_switch
                assert not standard_console_sources(service)
                gateway.validation_release.set()
                press(host, "enter", "\r")
                refusal = "Console preparation source unavailable; Send again."
                await until(lambda: any(refusal in notice for notice in notices))
                await _idle(host, console, pilot, session_id)
                flight = console._console_send_flight
                assert not flight.tasks and not flight.deferred
                assert not getter_calls
                assert not _sent_or_queued(console, session_id)
                assert gateway.stream_calls == 0
                assert store.session_draft(session_id) == DRAFT
                assert composer.draft_text() == DRAFT
                assert store.active_session_id == session_id
                assert console._console_visible_draft_session_id == session_id
                record_property("completed_managed_refusal", refusal)
                record_property("custom_getter_calls", len(getter_calls))
            finally:
                installed = vars(service).get("get_kill_switch", missing)
                getter_owned = (
                    installed is get_kill_switch or installed is original_slot
                )
                if original_slot is missing:
                    vars(service).pop("get_kill_switch", None)
                else:
                    service.get_kill_switch = original_slot
                gateway.validation_release.set()
                capacity = runtime._execution_capacity
                await runtime.dispose()
                assert getter_owned
            assert vars(service).get("get_kill_switch", missing) is original_slot
            assert standard_console_sources(service)
            assert runtime._disposed and not runtime._preparation_reads
            assert not runtime._ordinary_turns and not runtime._turn_custody
            assert runtime._execution_capacity is capacity
            assert capacity is None or not capacity.snapshot().executions
            record_property("capacity_already_allocated", capacity is not None)
            record_property("runtime_disposed_without_live_owners", True)
