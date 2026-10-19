"""Run start hands the UI loop back between the MCP stores' reads (TASK-33620.15.5).

Measured live on origin/dev 4de437a5c5 (Anthropic haiku, 160x45, run-start
callbacks timed on the UI loop): every later send checks its kept catalog with
the fingerprint's two admission-checked store reads (the permission store, then
the local store) in one callback, the longest run-start block of those sends.
Each read is its own storage-admission scope. Run start stays on the loop
(TASK-33620.15.1 measured a worker hop making first sends reach the provider
97-249 ms later), but it yields between the reads, so a key press waits for
one read, not both.

The stock Console composition reads its stores through worker hops on dev
(``prepare_console_tools``). A composition from any other sources (a plugin
provider, the mounted harness here) still reads the kill switch, the external
catalog, the built-in inventory and the tool states on the loop; it now yields
between those too.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from Tests.UI.test_console_mcp_catalog_reuse import (
    _Console,
    _run_start,
    _with_console,
    trust_new_files as trust_new_files,  # noqa: PLC0414 - autouse fixture
)

pytestmark = pytest.mark.bootstrap_profile

#: Frames that mark a run-start composition.
COMPOSE_FRAMES = frozenset({"_compose_mcp_provider", "_compose_local_provider"})


def _in_composition() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name in COMPOSE_FRAMES:
            return True
        frame = frame.f_back
    return False


class LoopTurns:
    """Record, per run-start store read, which UI-loop iteration it ran in."""

    def __init__(self, c: _Console) -> None:
        self.turn = 0
        self.reads: list[tuple[str, int]] = []
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._loop.call_soon(self._tick)
        service = c.service
        local = service.local_service
        self._wrap(service, "get_kill_switch", "kill_switch")
        self._wrap(service, "local_external_catalog", "external_catalog")
        self._wrap(service, "effective_tool_states", "tool_states")
        self._wrap(local, "get_inventory", "inventory")
        self._wrap(local, "catalog_fingerprint", "local_store_fingerprint")
        self._wrap(service.permission_store, "catalog_fingerprint", "permission_fingerprint")

    def _tick(self) -> None:
        self.turn += 1
        if self._running:
            self._loop.call_soon(self._tick)

    def stop(self) -> None:
        self._running = False

    def _wrap(self, owner, name: str, label: str) -> None:
        real = getattr(owner, name)
        reads = self.reads

        if asyncio.iscoroutinefunction(real):

            async def wrapped(*args, **kwargs):
                if _in_composition():
                    reads.append((label, self.turn))
                return await real(*args, **kwargs)

        else:

            def wrapped(*args, **kwargs):
                if _in_composition():
                    reads.append((label, self.turn))
                return real(*args, **kwargs)

        setattr(owner, name, wrapped)

    def shared(self) -> list[list[str]]:
        """Groups of reads that ran in the same loop iteration."""
        by_turn: dict[int, list[str]] = {}
        for label, turn in self.reads:
            by_turn.setdefault(turn, []).append(label)
        return [labels for labels in by_turn.values() if len(labels) > 1]


@pytest.mark.asyncio
async def test_a_full_composition_reads_each_store_in_its_own_loop_turn():
    """Red on origin/dev: all six reads ran in one callback (the harness composes on the loop)."""

    async def check(c: _Console) -> None:
        turns = LoopTurns(c)
        try:
            provider, *_ = await _run_start(c.console)
        finally:
            turns.stop()
        assert provider is not None, "the harness offers built-in MCP tools"
        labels = [label for label, _turn in turns.reads]
        assert {
            "permission_fingerprint",
            "local_store_fingerprint",
            "kill_switch",
            "external_catalog",
            "inventory",
            "tool_states",
        } <= set(labels), labels
        assert not turns.shared(), f"store reads shared one loop callback: {turns.shared()}"

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_reused_catalog_checks_each_store_in_its_own_loop_turn():
    """Red on origin/dev: the fingerprint's two store reads ran in one callback."""

    async def check(c: _Console) -> None:
        first, *_ = await _run_start(c.console)
        turns = LoopTurns(c)
        try:
            second, *_ = await _run_start(c.console)
        finally:
            turns.stop()
        labels = [label for label, _turn in turns.reads]
        assert labels == ["permission_fingerprint", "local_store_fingerprint"], (
            f"expected the reuse check alone: {labels}"
        )
        assert [e.id for e in second.list_catalog()] == [e.id for e in first.list_catalog()]
        assert not turns.shared(), f"store reads shared one loop callback: {turns.shared()}"

    await _with_console(check)
