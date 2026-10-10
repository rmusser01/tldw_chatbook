"""Original widget removal, host drain and App exit join native Character work."""

import asyncio

import pytest

from Tests.Backup_Recovery.test_finite_db_counted_interval import live_operations
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root
from Tests.Performance.test_console_shutdown_reader_ownership import (
    _held_original_reader,
    _retired,
    _turns,
)
from Tests.UI.test_console_character_context import _CharacterApp, _controller
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
)
from tldw_chatbook.UI.Console_Modules.view_workers import (
    capture_console_view_workers,
    drain_console_view_workers,
)
from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize("terminal", ["remove", "host_drain", "app_shutdown"])
async def test_original_character_terminal_waits_for_native_callback(
    tmp_path, terminal
):
    database = CharactersRAGDB(tmp_path / "character-terminal.db", "terminal-proof")
    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    executor, entered, release, seen = _held_original_reader(
        database,
        ConsoleCharacterContextController._read_database_scope_metadata.__code__,
    )
    loop = asyncio.get_running_loop()
    prior = loop._default_executor
    loop.set_default_executor(executor)
    app = _CharacterApp(controller, controller.state)
    mounted, close = asyncio.Event(), asyncio.Event()
    widgets = []

    async def app_session():
        # Enter/exit in this same Task, preserving the original run_test
        # context ownership. No substitute _shutdown/on_unmount is installed.
        async with app.run_test():
            widgets.append(app.screen.query_one(ConsoleCharacterContext))
            mounted.set()
            await close.wait()

    session = asyncio.create_task(app_session())
    pending = None
    try:
        await asyncio.wait_for(mounted.wait(), 5)
        assert await asyncio.to_thread(entered.wait, 5)
        widget = widgets[0]
        assert seen and live_operations(database) and worker_leases(database)
        original_task = widget._controller_task
        assert original_task is not None and not original_task.done()
        if terminal == "remove":
            pending = asyncio.ensure_future(widget.remove())
        elif terminal == "host_drain":
            captured = capture_console_view_workers(app)
            assert any(
                node is widget and task is original_task
                for worker, node, task, work in captured[-1]
            )
            pending = asyncio.create_task(drain_console_view_workers(captured))
        else:
            close.set()
            pending = session
        await _turns()
        assert (
            not pending.done()
        ), f"{terminal} returned with the actual native callback held"
        assert not original_task.done(), f"{terminal} lost its issued Character Task"
        assert live_operations(database) and worker_leases(database)
        release.set()
        await asyncio.wait_for(asyncio.shield(pending), 5)
        assert original_task.done()
        assert not live_operations(database) and not worker_leases(
            database
        ), f"{terminal} returned before actual native/core retirement"
        if terminal == "remove":
            assert not widget.is_attached
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        close.set()
        await session
        await _retired(database)
        loop._default_executor = prior
        executor.shutdown(wait=True)
        database.close()
