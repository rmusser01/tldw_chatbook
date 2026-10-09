"""A Console run reuses its composed MCP catalog while nothing it read changed.

TASK-33620.15.1. Measured live on the base build 46c3959526 (Anthropic haiku,
160x45, ten sends): every run start composed the MCP catalog on the UI loop
(the kill switch read three times, the local server catalog, the built-in
manifest, the permission states), a 63-130 ms stretch per send. 180 of the
265 sampled frames were the stores' storage-admission scopes re-deriving the
selected store generation. A worker hop freed the loop but made first sends
reach the provider 97-249 ms later, so composition is back on the loop: the
first send composes as before (one kill-switch read for MCP and local tools),
and later sends reuse that catalog after one admission-checked read per MCP
store confirms nothing it read has changed.

Each test first checks the reuse (red on the base build, which recomposes
every time) and then that one authority input invalidates it: the next run
start recomposes and reflects the change. Negative controls (run by hand,
task notes): leaving that input out of the reuse check turns its test red.
"""

from __future__ import annotations

import asyncio
import collections
import sys
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    ready_console,
)

pytestmark = pytest.mark.bootstrap_profile

#: Frames that mark a run-start composition.
COMPOSE_FRAMES = frozenset({"_compose_mcp_provider", "_compose_local_provider"})
#: The reads a run-start composition makes (each through a store scope).
READS = ("get_kill_switch", "local_external_catalog", "effective_tool_states")
#: Tests that keep the real settle rule.
REAL_SETTLE = frozenset({"test_a_composition_over_a_just_written_store_is_not_reused"})


def _in_composition() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name in COMPOSE_FRAMES:
            return True
        frame = frame.f_back
    return False


class CompositionReads:
    """Count the composition's store reads on the real service."""

    def __init__(self, service) -> None:
        self.calls: collections.Counter[str] = collections.Counter()
        for name in READS:
            self._wrap(service, name)
        self._wrap(service.local_service, "get_inventory")

    def _wrap(self, owner, name: str) -> None:
        real = getattr(owner, name)
        calls = self.calls
        if asyncio.iscoroutinefunction(real):

            async def wrapped(*args, **kwargs):
                if _in_composition():
                    calls[name] += 1
                return await real(*args, **kwargs)

        else:

            def wrapped(*args, **kwargs):
                if _in_composition():
                    calls[name] += 1
                return real(*args, **kwargs)

        setattr(owner, name, wrapped)

    def total(self) -> int:
        return sum(self.calls.values())

    def reset(self) -> None:
        self.calls.clear()


@pytest.fixture(autouse=True)
def trust_new_files(monkeypatch, request):
    """Trust store files written moments ago, unless a test pins that rule.

    The reuse is stored only when every store file it read was last changed
    at least ``SETTLE_NS`` before (storage admission's own rule before it
    trusts file stamps); the harness writes them at boot.
    """
    try:
        from tldw_chatbook.Chat import console_run_start_tools as tools
    except ImportError:  # the base build has no reuse
        yield
        return
    if hasattr(tools, "forget"):
        tools.forget()
    if request.node.name not in REAL_SETTLE and hasattr(tools, "SETTLE_NS"):
        monkeypatch.setattr(tools, "SETTLE_NS", 0)
    yield
    if hasattr(tools, "forget"):
        tools.forget()


def _ids(provider) -> list[str]:
    return [] if provider is None else [entry.id for entry in provider.list_catalog()]


async def _run_start(console):
    controller = console._ensure_console_chat_controller()
    session_id = console._ensure_console_chat_store().active_session_id
    return await controller._compose_agent_request_providers(
        session_id=session_id,
        project_selection=None,
        project_authority_guard=None,
        turn_context=None,
        admitted_roots=(),
    )


async def _compose(console, **kwargs):
    controller = console._ensure_console_chat_controller()
    session_id = console._ensure_console_chat_store().active_session_id
    return await controller._compose_mcp_provider(session_id, **kwargs)


async def _reused(console, reads, compose=None) -> object:
    """Compose again with nothing changed: no store read, the same catalog."""
    reads.reset()
    provider = await (compose() if compose else _run_start(console))
    provider = provider[0] if isinstance(provider, tuple) else provider
    assert reads.total() == 0, f"recomposed though nothing changed: {dict(reads.calls)}"
    return provider


async def _recomposed(console, reads, compose=None) -> object:
    reads.reset()
    provider = await (compose() if compose else _run_start(console))
    provider = provider[0] if isinstance(provider, tuple) else provider
    assert reads.total() > 0, "reused a catalog after its input changed"
    return provider


class _Console:
    """One mounted Console, its MCP service and a read counter."""

    def __init__(self, host, console) -> None:
        self.host = host
        self.console = console
        self.app = host.app_instance
        self.service = self.app.unified_mcp_service
        self.reads = CompositionReads(self.service)
        self.touched: list[tuple[str, str]] = []

    def set_state(self, tool, state: str) -> None:
        self.touched.append((tool.server_key, tool.name))
        self.service.set_tool_state(tool.server_key, tool.name, state)


async def _with_console(check) -> None:
    """Run ``check`` on a mounted Console, then undo its MCP store changes.

    ``bootstrap_profile`` tests share one profile, so every tool state, kill
    switch and server this file writes is put back.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            mounted = _Console(host, console)
            try:
                await check(mounted)
            finally:
                service = mounted.service
                service.set_kill_switch(False)
                for server_key, name in mounted.touched:
                    service.set_tool_state(server_key, name, None)
                service.local_service.store.delete_profile("reusecheck")


@pytest.mark.asyncio
async def test_run_start_reads_the_kill_switch_once_for_mcp_and_local_tools():
    """AC#4: one read decides MCP and local tools (the base build read it 3x)."""

    async def check(c: _Console) -> None:
        first, _gate, _local, _hook = await _run_start(c.console)
        assert first is not None, "the harness offers built-in MCP tools"
        assert c.reads.calls["get_kill_switch"] == 1, dict(c.reads.calls)

    await _with_console(check)


@pytest.mark.asyncio
async def test_an_unchanged_catalog_is_reused_until_the_kill_switch_changes():
    """AC#4: the kill switch turned on before run start still drops MCP."""

    async def check(c: _Console) -> None:
        first, _gate, _local, _hook = await _run_start(c.console)
        second = await _reused(c.console, c.reads)
        assert second is not first, "each run gets its own provider"
        assert _ids(second) == _ids(first)
        assert second._main_loop is asyncio.get_running_loop()
        assert c.app.console_mcp_tool_count == len(_ids(first))
        c.service.set_kill_switch(True)
        c.reads.reset()
        off, _gate, local, _hook = await _run_start(c.console)
        assert c.reads.total() > 0
        assert off is None and local is None
        assert c.app.console_mcp_tool_count is None
        assert c.app.console_mcp_not_connected_count is None
        c.service.set_kill_switch(False)
        again = await _recomposed(c.console, c.reads)
        assert _ids(again) == _ids(first)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_changed_tool_state_recomposes_the_catalog():
    """AC#4: a tool switched off between sends is not offered."""

    async def check(c: _Console) -> None:
        first, *_ = await _run_start(c.console)
        await _reused(c.console, c.reads)
        llm_name, (tool, _state) = next(iter(first._entry_by_llm_name.items()))
        c.set_state(tool, "deny")
        after = await _recomposed(c.console, c.reads)
        assert llm_name not in _ids(after)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_changed_tool_or_definition_maximum_recomposes_the_catalog():
    """AC#4: the run's tool and definition maxima are part of the reuse key."""

    async def check(c: _Console) -> None:
        everything = await _compose(c.console)
        llm_name, (tool, _state) = next(iter(everything._entry_by_llm_name.items()))
        only = frozenset({tool.tool_id})

        def narrowed():
            return _compose(c.console, maximum_tool_ids=only)

        one = await _recomposed(c.console, c.reads, narrowed)
        assert _ids(one) == [llm_name]
        await _reused(c.console, c.reads, narrowed)

        def wrong_definition():
            return _compose(
                c.console,
                maximum_tool_ids=only,
                maximum_definition_hashes={tool.tool_id: "0" * 64},
            )

        assert await _recomposed(c.console, c.reads, wrong_definition) is None

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_new_server_or_connection_recomposes_the_catalog(monkeypatch):
    """AC#4: the server set (local store) and connections are reuse inputs."""
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

    async def check(c: _Console) -> None:
        await _run_start(c.console)
        await _reused(c.console, c.reads)
        local = c.service.local_service
        local.store.save_profile(
            LocalExternalMCPProfile.from_input_dict(
                {"profile_id": "reusecheck", "command": "echo", "args": ["hi"]}
            )
        )
        local.store.save_discovery_snapshot(
            "reusecheck",
            {"tools": [{"name": "ping", "description": "Ping the server."}]},
        )
        added = await _recomposed(c.console, c.reads)
        assert any("reusecheck" in name for name in _ids(added)), _ids(added)
        assert added.not_connected_count == 1
        await _reused(c.console, c.reads)
        session = SimpleNamespace(_closed=False)
        sessions = {"reusecheck": session}
        monkeypatch.setattr(local, "client", SimpleNamespace(sessions=sessions))
        connected = await _recomposed(c.console, c.reads)
        assert connected.not_connected_count == 0
        await _reused(c.console, c.reads)
        session._closed = True
        closed = await _recomposed(c.console, c.reads)
        assert closed.not_connected_count == 1

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_changed_permission_profile_recomposes_the_catalog():
    """AC#4: the workspace's permission profile is part of the reuse key."""

    async def check(c: _Console) -> None:
        await _compose(c.console)
        await _reused(c.console, c.reads, lambda: _compose(c.console))

        def named():
            return _compose(c.console, profile_id_provider=lambda: "reuse-named")

        await _recomposed(c.console, c.reads, named)
        await _reused(c.console, c.reads, named)
        await _recomposed(c.console, c.reads, lambda: _compose(c.console))

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_failed_fingerprint_composes_from_the_stores(monkeypatch):
    """A refused or failed check never serves the earlier catalog."""

    async def check(c: _Console) -> None:
        first, *_ = await _run_start(c.console)
        await _reused(c.console, c.reads)

        async def refused(*_args, **_kwargs):
            raise RuntimeError("admission refused")

        monkeypatch.setattr(c.service, "run_catalog_fingerprint", refused)
        again = await _recomposed(c.console, c.reads)
        assert _ids(again) == _ids(first)
        await _recomposed(c.console, c.reads)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_store_write_during_composition_is_never_reused(monkeypatch):
    """The stored catalog is the one its fingerprint described.

    A tool is denied while a composition reads the permission states, and
    the file's earlier bytes are then put back. Had that composition been
    kept under its pre-read fingerprint, the next run would match it and
    offer the denied state's catalog for bytes that allow the tool.
    """

    async def check(c: _Console) -> None:
        first, *_ = await _run_start(c.console)
        await _reused(c.console, c.reads)
        names = list(first._entry_by_llm_name.items())
        (_n1, (one, _s1)), (llm_two, (two, _s2)) = names[0], names[1]
        c.set_state(one, "ask")
        path = c.service.permission_store.path
        before = path.read_bytes()
        real = c.service.effective_tool_states
        fired = []

        def deny_meanwhile(tools, **kwargs):
            if not fired:
                fired.append(True)
                c.set_state(two, "deny")
            return real(tools, **kwargs)

        monkeypatch.setattr(c.service, "effective_tool_states", deny_meanwhile)
        during = await _recomposed(c.console, c.reads)
        assert fired and llm_two not in _ids(during)
        monkeypatch.setattr(c.service, "effective_tool_states", real)
        path.write_bytes(before)
        after = await _recomposed(c.console, c.reads)
        assert llm_two in _ids(after)

    await _with_console(check)


@pytest.mark.asyncio
async def test_an_admission_change_during_composition_is_never_reused(monkeypatch):
    """A recovery or admission record written meanwhile drops the composition.

    Storage admission advances its epoch around every write of admission or
    activation records, which can change what the stores read as. Such a
    write while a composition reads leaves nothing reusable behind.
    """
    from tldw_chatbook.Backup_Recovery import bootstrap

    async def check(c: _Console) -> None:
        first, *_ = await _run_start(c.console)
        await _reused(c.console, c.reads)
        _name, (tool, _state) = next(iter(first._entry_by_llm_name.items()))
        c.set_state(tool, "ask")
        real = c.service.effective_tool_states

        def admission_write_meanwhile(tools, **kwargs):
            bootstrap.advance_admission_epoch()
            return real(tools, **kwargs)

        monkeypatch.setattr(c.service, "effective_tool_states", admission_write_meanwhile)
        await _recomposed(c.console, c.reads)
        monkeypatch.setattr(c.service, "effective_tool_states", real)
        await _recomposed(c.console, c.reads)
        await _reused(c.console, c.reads)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_composition_over_a_just_written_store_is_not_reused():
    """A store file changed under two seconds ago is read again next run.

    Storage admission trusts file stamps only once a change has settled; the
    reuse check keeps that rule, so a same-tick second write cannot hide.
    """

    async def check(c: _Console) -> None:
        await asyncio.sleep(2.2)
        first, *_ = await _run_start(c.console)
        await _reused(c.console, c.reads)
        _name, (tool, _state) = next(iter(first._entry_by_llm_name.items()))
        c.set_state(tool, "ask")
        await _recomposed(c.console, c.reads)
        await _recomposed(c.console, c.reads)
        await asyncio.sleep(2.2)
        await _recomposed(c.console, c.reads)
        await _reused(c.console, c.reads)

    await _with_console(check)
