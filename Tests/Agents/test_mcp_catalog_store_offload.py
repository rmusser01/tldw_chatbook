"""Catalog composition keeps blocking custody reads off the main event loop."""

import asyncio
import threading
import time

import pytest

from Tests.Agents.test_mcp_tool_provider import (
    FakeMCPService,
    _catalog_record,
    _tool_dict,
)
from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Backup_Recovery import (
    raw_participants as raw,
    storage_admission as storage,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState, MCPPermissionStore


@pytest.mark.asyncio
async def test_catalog_store_reads_do_not_block_loop_and_async_source_stays_on_loop():
    main = threading.current_thread()
    reads = []
    async_reads = []
    service = FakeMCPService(
        catalog_records=[_catalog_record("sample", [_tool_dict("lookup")])],
    )
    kill = service.get_kill_switch
    states = service.effective_tool_states
    inventory = service.local_service.get_inventory
    catalog = service.local_external_catalog

    def observed(name, function):
        def call(*args, **kwargs):
            reads.append((name, threading.current_thread()))
            return function(*args, **kwargs)

        return call

    async def async_catalog():
        async_reads.append(threading.current_thread())
        return await catalog()

    service.get_kill_switch = observed("kill", kill)
    service.effective_tool_states = observed("states", states)
    service.local_service.get_inventory = observed("inventory", inventory)
    service.local_external_catalog = async_catalog
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    await provider.compose_catalog()

    assert [name for name, _ in reads] == ["kill", "inventory", "states"]
    assert all(thread is not main for _, thread in reads)
    assert async_reads == [main]
    assert len(provider.list_catalog()) == 1


@pytest.mark.asyncio
async def test_each_catalog_composition_resolves_fresh_permissions_and_kill_state():
    service = FakeMCPService(
        catalog_records=[_catalog_record("sample", [_tool_dict("lookup")])],
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    await provider.compose_catalog()
    assert len(provider.list_catalog()) == 1
    service.default_state = EffectiveToolState(state="deny", origin="global_default")
    await provider.compose_catalog()
    assert provider.list_catalog() == []
    service.default_state = EffectiveToolState(state="allow", origin="global_default")
    await provider.compose_catalog()
    assert len(provider.list_catalog()) == 1
    service.kill_switch = True
    await provider.compose_catalog()
    assert provider.list_catalog() == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
async def test_cancelled_native_permission_read_remains_counted_until_worker_retires(
    tmp_path,
    monkeypatch,
):
    """Canceling composition must not release an actually running source read."""
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    target = tmp_path / "config.toml"
    target.write_text(f'[paths]\ndata_dir = "{data.as_posix()}"\n', encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config = install_config_source(monkeypatch)
    selected_data = config.get_user_data_dir()
    permission = MCPPermissionStore(selected_data / "mcp_permissions.json")
    permission.set_kill_switch(False)
    entered, release, finished = (threading.Event() for _ in range(3))
    operations = []
    service = FakeMCPService()
    actual_read = permission.get_kill_switch
    main = threading.current_thread()

    def blocked_read():
        try:
            # This is an actual installed source operation, not a forged token.
            with raw._scope(permission, "mcp_store", writing=True):
                operations.append(raw._local.operation)
                assert raw._states[operations[-1]].source is permission
                assert raw._states[operations[-1]].participant is not None
                entered.set()
                assert threading.current_thread() is not main
                assert release.wait(10)
                return actual_read()
        finally:
            finished.set()

    service.get_kill_switch = blocked_read
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    pending = asyncio.create_task(provider.compose_catalog())
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pause = storage._begin_local_pause()
        assert not pause.drain(time.monotonic() + 0.02)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert operations and all(
            operation in raw._states and operation in storage._raw_operations
            for operation in operations
        )
        assert not pause.drain(time.monotonic() + 0.02)
        assert provider.list_catalog() == []
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        for _ in range(200):
            if not storage._raw_operations:
                break
            await asyncio.sleep(0.005)
        assert all(
            operation not in raw._states and operation not in storage._raw_operations
            for operation in operations
        )
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        if pause is not None:
            pause.resume()
        assert await asyncio.to_thread(finished.wait, 10)


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_controller_mcp_preflight_reads_kill_switch_off_loop():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    main = threading.current_thread()
    reads = []
    service = FakeMCPService(kill_switch=True)
    actual = service.get_kill_switch

    def read():
        reads.append(threading.current_thread())
        return actual()

    service.get_kill_switch = read
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=SimpleNamespace(),
    )
    controller.app = SimpleNamespace(unified_mcp_service=service)
    assert await controller._compose_mcp_provider() is None
    assert reads and all(thread is not main for thread in reads)
    assert controller.app.console_mcp_tool_count is None


@pytest.mark.asyncio
async def test_catalog_resolves_profile_callback_on_loop_and_reads_it_off_loop():
    main = threading.current_thread()
    service = FakeMCPService(
        catalog_records=[_catalog_record("sample", [_tool_dict("lookup")])],
    )
    reads = []
    callbacks = []

    def profile():
        callbacks.append(threading.current_thread())
        return "work"

    def states(tools, *, profile_id):
        reads.append((threading.current_thread(), profile_id))
        return {
            (tool.server_key, tool.name): EffectiveToolState(
                state="deny",
                origin="global_default",
            )
            for tool in tools
        }

    service.effective_tool_states = states
    provider = MCPToolProvider(
        service=service,
        main_loop=asyncio.get_running_loop(),
        profile_id_provider=profile,
    )
    await provider.compose_catalog()
    assert callbacks == [main]
    assert reads and all(
        thread is not main and name == "work" for thread, name in reads
    )
    assert provider.list_catalog() == []


@pytest.mark.asyncio
async def test_catalog_preserves_store_read_failure_identity():
    error = ValueError("actual store refused")
    service = FakeMCPService()

    def fail():
        raise error

    service.get_kill_switch = fail
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    with pytest.raises(ValueError) as caught:
        await provider.compose_catalog()
    assert caught.value is error
    assert provider.list_catalog() == []
