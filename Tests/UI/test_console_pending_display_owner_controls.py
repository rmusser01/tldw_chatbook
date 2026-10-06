"""Focused owner/source refusals for the optional in-memory pending fragment.

The existing held-original-reader test proves the cold count1/deny0 behavior.
These post-candidate controls preserve its actual stock fixture and callbacks.
"""

from dataclasses import replace

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
from Tests.UI.test_console_pending_facts_cold_readiness import _CheckedNavigationFactory
from Tests.UI import test_console_pending_interrupt_projection as original
from tldw_chatbook import config
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_display_state import ConsoleInspectorState
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Screens import chat_screen
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector


async def _prepared(app, pilot):
    console, controller, store, session_id = await original._seed_console(app, pilot)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
    assert await projection.warm()
    assert console._sync_console_rail_and_controls() is not False
    await pilot.pause()
    inspector = console.query_one("#console-run-inspector-state", ConsoleRunInspector)
    owner = chat_screen._console_pending_display_owner(console)
    assert owner is not None and owner[3] is controller and owner[4] is store
    assert owner[5] is store._sessions[session_id]
    assert console._console_pending_display_base[0][5] is owner[5]
    assert inspector.state.pending_approval_count == 0
    return console, controller, store, session_id, inspector, owner


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@private_profile_test
async def test_pending_fragment_current_owner_and_saved_generation_remain_display_only(
    tmp_path, request,
):
    factory = _CheckedNavigationFactory()
    app = factory.build(tmp_path)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            console, _, _, _, inspector, owner = await _prepared(app, pilot)
            base = inspector.state
            assert chat_screen._sync_console_pending_display(console)
            assert inspector.state.pending_approval_count == 0
            assert not inspector.state.has_pending_approval
            # A real supported save invalidates only the config generation.
            # The fragment does not read/reuse a newly checked config mapping.
            assert config.save_setting_to_cli_config("splash_screen", "enabled", False)
            current = chat_screen._console_pending_display_owner(console)
            assert current is not None and current[8] != owner[8]
            assert current[8][1] == owner[8][1]
            assert chat_screen._sync_console_pending_display(console)
            state = inspector.state
            assert state.pending_approval_count == 0 and not state.has_pending_approval
            assert "approvals 0" in next(
                row.value for row in state.rows if row.label == "Run recipe"
            )
            assert next(row for row in state.rows if row.label == "Provider") is next(
                row for row in base.rows if row.label == "Provider"
            )
            assert factory.current()
    finally:
        drain_created_dirs()
        drain_active_service_patches()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@private_profile_test
async def test_pending_fragment_declines_unknown_owner_custom_reader_and_malformed_metadata(
    tmp_path, request,
):
    factory = _CheckedNavigationFactory()
    app = factory.build(tmp_path)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            console, controller, store, session_id, inspector, owner = await _prepared(
                app, pilot
            )
            before = inspector.state
            base = console._console_pending_display_base
            session = store._sessions[session_id]
            # Same ID/fields is a different selected-session owner.
            replacement = replace(session)
            assert replacement is not session and replacement.id == session.id
            try:
                store._sessions[session_id] = replacement
                assert not chat_screen._sync_console_pending_display(console)
                assert inspector.state is before
            finally:
                store._sessions[session_id] = session
            console._console_pending_display_base = (
                owner,
                ConsoleInspectorState(rows=()),
            )
            assert not chat_screen._sync_console_pending_display(console)
            console._console_pending_display_base = object()
            assert not chat_screen._sync_console_pending_display(console)
            console._console_pending_display_base = base
            calls = []
            console._console_pending_approval_count = (
                lambda: calls.append("custom") or 1
            )
            try:
                assert not chat_screen._sync_console_pending_display(console)
                assert calls == [] and inspector.state is before
            finally:
                del console._console_pending_approval_count
            records = controller_module._CONSOLE_PENDING_FACTS_READERS
            try:
                controller_module._CONSOLE_PENDING_FACTS_READERS = (object(),)
                assert chat_screen._console_pending_display_owner(console) is None
                assert not chat_screen._sync_console_pending_display(console)
                assert inspector.state is before
            finally:
                controller_module._CONSOLE_PENDING_FACTS_READERS = records
            # An original function object whose body changes is still customized.
            method = controller_module.ConsoleChatController.pending_round_kinds
            original_code = method.__code__

            def changed(self, selected):
                raise AssertionError("changed reader body must not be invoked")

            try:
                method.__code__ = changed.__code__
                assert not chat_screen._sync_console_pending_display(console)
                assert calls == [] and inspector.state is before
            finally:
                method.__code__ = original_code
            identity = config.current_config_identity
            identity_code = identity.__code__

            def changed_identity():
                raise AssertionError("changed selector body must not be invoked")

            try:
                identity.__code__ = changed_identity.__code__
                assert not chat_screen._sync_console_pending_display(console)
                assert inspector.state is before
            finally:
                identity.__code__ = identity_code
            for name in ("chat_controller", "chat_store"):
                descriptor = vars(ConsoleRuntime)[name]

                def custom_property(_runtime):
                    raise AssertionError("custom Runtime peek must not be invoked")

                try:
                    setattr(ConsoleRuntime, name, property(custom_property))
                    assert not chat_screen._sync_console_pending_display(console)
                    assert inspector.state is before
                finally:
                    setattr(ConsoleRuntime, name, descriptor)
            assert chat_screen._sync_console_pending_display(console)
            assert inspector.state.pending_approval_count == 0
            assert factory.current()
    finally:
        drain_created_dirs()
        drain_active_service_patches()
