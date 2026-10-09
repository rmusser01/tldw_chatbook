"""A Console run's start never holds key delivery while it reads MCP (TASK-33620.15.1).

Measured on the base build 46c3959526 (live, Anthropic haiku through the
review proxy, 160x45, ten sends): between a send's acceptance and its
provider call the run composed its MCP provider on the UI loop -- the kill
switch read three times, the local server catalog, the built-in inventory and
the permission states, each through the MCP stores' storage-admission scope.
Main-thread samples put 63-130 ms busy stretches there, and hover waits over
it reached 104-297 ms.

The composition still happens at the same point of the send (run start,
after the durable commit) and reads every authority fresh there; it now runs
in one worker hop, so the UI loop keeps handling input meanwhile. The test
holds the composition's kill-switch read and types a key while it is held:
on the base build the read ran on the loop thread, so the key landed only
after the hold gave up. Negative control (run by hand, task notes): running
the hop's body inline on the loop turns it red.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)

pytestmark = pytest.mark.bootstrap_profile

#: How long the held read waits before giving up: long enough for a free UI
#: loop to deliver a key many times over, even on a loaded host.
HOLD_SECONDS = 10.0
#: The composition reaches its first read quickly; a loaded host is slow.
ENTRY_SECONDS = 30.0
#: Frames that mark a run-start composition (base: on the loop; fix: the hop).
COMPOSE_FRAMES = frozenset({"_compose_mcp_provider", "compose_run_mcp_provider"})


def _called_from_composition() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name in COMPOSE_FRAMES:
            return True
        frame = frame.f_back
    return False


class HeldRunStartRead:
    """Hold the run-start composition's first kill-switch read.

    Wraps the real ``unified_mcp_service.get_kill_switch``. Only a call made
    by the run-start MCP composition is held (on whichever thread it runs);
    any other reader passes straight through.
    """

    def __init__(self, service) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.timed_out = False
        self.thread: str | None = None
        self.armed = True
        self.reads: list[str] = []
        real = service.get_kill_switch

        def get_kill_switch():
            if _called_from_composition():
                self.reads.append(threading.current_thread().name)
                if self.armed:
                    self.armed = False
                    self.thread = threading.current_thread().name
                    self.entered.set()
                    self.timed_out = not self.release.wait(timeout=HOLD_SECONDS)
            return real()

        service.get_kill_switch = get_kill_switch


async def _compose_run_start(console):
    controller = console._ensure_console_chat_controller()
    session_id = console._ensure_console_chat_store().active_session_id
    return await controller._compose_agent_request_providers(
        session_id=session_id,
        project_selection=None,
        project_authority_guard=None,
        turn_context=None,
        admitted_roots=(),
    )


@pytest.mark.asyncio
async def test_a_key_typed_while_run_start_reads_mcp_is_handled_before_the_read_returns():
    """AC#2/#5: keys keep flowing while a run composes its MCP provider."""
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            hold = HeldRunStartRead(host.app_instance.unified_mcp_service)
            composing = asyncio.ensure_future(_compose_run_start(console))
            try:
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "x", "x")
                await until(lambda: composer.draft_text().endswith("x"), timeout=10)
                typed_while_held = not hold.release.is_set() and not hold.timed_out
            finally:
                hold.release.set()
            mcp_provider, builtin_gate, _local, _hook = await composing
            assert typed_while_held, (
                "run start held the UI loop: the typed key was handled only "
                f"after its MCP read (thread {hold.thread!r}) gave up waiting"
            )
            assert hold.thread != threading.main_thread().name, hold.thread
            assert builtin_gate is not None
            if mcp_provider is not None:
                # Tool calls still run on the UI loop, which owns the service.
                assert mcp_provider._main_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_run_start_reads_the_kill_switch_fresh_where_it_always_did():
    """AC#4: a kill switch turned on before run start still drops MCP tools.

    The switch is read at run start, not taken from admission or a cache:
    composition with the switch off offers MCP; once it is on, the next
    composition offers none and publishes the empty inspector counts.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            service = host.app_instance.unified_mcp_service
            app = host.app_instance
            real = service.get_kill_switch
            switch = {"on": False}
            reads: list[str] = []

            def get_kill_switch():
                reads.append(threading.current_thread().name)
                return switch["on"] or real()

            service.get_kill_switch = get_kill_switch
            first, _gate, _local, _hook = await _compose_run_start(console)
            assert reads, "run start did not read the kill switch"
            if first is None:
                pytest.skip("this profile offers no MCP tools to compare")
            assert app.console_mcp_tool_count == len(first.list_catalog())
            switch["on"] = True
            reads.clear()
            second, _gate, local, _hook = await _compose_run_start(console)
            assert reads, "run start reused an earlier kill-switch read"
            assert second is None and local is None
            assert app.console_mcp_tool_count is None
            assert app.console_mcp_not_connected_count is None
