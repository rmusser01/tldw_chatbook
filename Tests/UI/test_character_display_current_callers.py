"""Current mounted FULL uses qualified Character presentation observations."""

import pytest

from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _original_calls,
    _retired,
)
from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.test_console_session_tab_close import (
    ProductionConsoleHarness,
    _mounted_console,
    _settle,
)
from Tests.UI.test_console_session_tab_close import (
    _pending_close_app as _character_screen,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# The established fixture discovers this helper's defining factory namespace.
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_actual_full_reuses_successful_character_display(
    request, record_property
):
    with _character_screen(request, "chat_create") as app:
        host = ProductionConsoleHarness(app)
        async with host.run_test(size=(160, 44)) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            database = app.chachanotes_db
            assert type(database) is CharactersRAGDB and not database.is_memory_db
            assert await _settle(
                pilot,
                lambda: (
                    not console._console_sync_in_progress
                    and not getattr(
                        console, "_console_control_bar_replay_whole_sync", False
                    )
                ),
            )
            await console._character_context.refresh()
            result = await console._sync_native_console_chat_ui()
            assert result is True
            with _original_calls(database) as observed:
                assert await console._sync_native_console_chat_ui() is True
                assert await console._sync_native_console_chat_ui() is True
            if observed["connections"]:
                _retired(database, observed)
            record_property("mounted_character_pairs", observed["pairs"])
            assert observed["pairs"] == 0, "FULL bypassed the qualified display memo"


@pytest.mark.asyncio
async def test_queued_stock_character_resume_refuses_closed_actual_view(request):
    from tldw_chatbook.UI.Console_Modules import character_context as module
    from tldw_chatbook.UI.Screens.chat_screen import _console_screen_is_torn_down

    with _character_screen(request, "chat_create") as app:
        host = ProductionConsoleHarness(app)
        async with host.run_test(size=(160, 44)) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            assert await _settle(pilot, lambda: not console._console_sync_in_progress)
            controller = console._character_context
            # The preceding shipping resume constructs exactly this direct work.
            factory = getattr(module, "character_view_resume_work", None)
            work = (
                factory(controller, console)
                if factory is not None
                else controller.refresh_if_scope_changed()
            )
            await console._close_messages(wait=False)
            assert _console_screen_is_torn_down(console)
            with _original_calls(app.chachanotes_db) as observed:
                await work
            if observed["connections"]:
                _retired(app.chachanotes_db, observed)
            assert observed["pairs"] == observed["recent"] == 0, (
                "queued stock resume read after actual view closure"
            )


@pytest.mark.asyncio
async def test_character_resume_preserves_custom_no_argument_callback():
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.character_context import (
        character_view_resume_work,
    )

    calls = []
    expected = object()

    async def custom():
        calls.append(())
        return expected

    assert (
        await character_view_resume_work(
            SimpleNamespace(refresh_if_scope_changed=custom), object()
        )
        is expected
    )
    assert calls == [()]


@pytest.mark.asyncio
async def test_actual_character_resume_declines_custom_and_refuses_queued_source_change(
    request,
):
    from tldw_chatbook.UI.Console_Modules import character_context as module

    with _character_screen(request, "chat_create") as app:
        host = ProductionConsoleHarness(app)
        async with host.run_test(size=(160, 44)) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            assert await _settle(pilot, lambda: not console._console_sync_in_progress)
            controller = console._character_context
            assert module._stock_character_view_resume(controller, console) is not None
            original_accessor = controller._database_accessor
            controller._database_accessor = lambda: app.chachanotes_db
            try:
                work = module.character_view_resume_work(controller, console)
                assert (
                    work.cr_code is type(controller).refresh_if_scope_changed.__code__
                )
                with _original_calls(app.chachanotes_db) as observed:
                    await work
                assert observed["pairs"] >= 1
                _retired(app.chachanotes_db, observed)
            finally:
                controller._database_accessor = original_accessor
            assert module._stock_character_view_resume(controller, console) is not None
            work = module.character_view_resume_work(controller, console)
            assert work.cr_code is module._resume_stock_character_view.__code__
            controller._database_accessor = lambda: app.chachanotes_db
            try:
                with _original_calls(app.chachanotes_db) as observed:
                    await work
                assert observed["pairs"] == observed["recent"] == 0
            finally:
                controller._database_accessor = original_accessor
            assert module._stock_character_view_resume(controller, console) is not None
