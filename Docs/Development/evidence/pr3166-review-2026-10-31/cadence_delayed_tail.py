"""Give original terminal-run workers one poll interval after fixture quiescence."""
import asyncio


def pytest_runtest_setup(item):
    from Tests.UI import test_console_config_sync_cadence as fixture
    original = fixture._quiesce

    async def quiesce(tick, seen, limit=10):
        await original(tick, seen, limit)
        if not tick.console._console_run_active():
            await asyncio.sleep(0.2)
    fixture._quiesce = quiesce
    item._restore_tail_quiet = lambda: setattr(fixture,"_quiesce",original)


def pytest_runtest_teardown(item):
    item._restore_tail_quiet()
