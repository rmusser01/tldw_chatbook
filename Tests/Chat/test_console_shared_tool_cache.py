"""Actual stock reuse retains its checked cold observation and native lifetime."""
import asyncio
import builtins
import dis
import inspect
import io
import os
import sqlite3
import time
from types import SimpleNamespace

import pytest

from Tests.Chat import test_console_shared_tool_preparation as shared
from Tests.Chat.test_console_async_mcp_snapshot import _MaximumProbe
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Chat import console_run_start_tools as cache
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)

mcp_sources = shared.mcp_sources
local_root = shared.local_root
catalog_store = shared.catalog_store
snapshot_case = shared.snapshot_case


class _StockProbe(_MaximumProbe):
    def __init__(self, case):
        super().__init__(case.source, case.permissions)
        self.service = case.service
        self.fingerprint_code = inspect.unwrap(UnifiedMCPControlPlaneService.run_catalog_fingerprint).__code__
        # A coroutine's frame raises a "call" event again at each resume after
        # an await, and the fingerprint yields between its two store reads
        # (TASK-33620.15.5). Count calls, not resumes: only an entry at the
        # code's start (its first RESUME, oparg 0).
        self.fingerprint_entry = next(
            ins.offset
            for ins in dis.get_instructions(self.fingerprint_code)
            if ins.opname == "RESUME" and ins.arg == 0
        )
        self.adoption_code = MCPToolProvider.adopt_console_preparation.__code__
        self.native_calls = 0
        self.admissions = 0
        self.admission_code = storage_admission._acquire_storage.__code__
        self.native_functions = (builtins.open, io.open, os.open, os.read, os.stat, os.lstat, sqlite3.connect)
        self.fingerprints = 0
        self.adoptions = 0

    def observe(self, frame, event, arg):
        if event == "c_call" and any(arg is function for function in self.native_functions):
            self.native_calls += 1
        if event == "call":
            if frame.f_code is self.admission_code:
                self.admissions += 1
            if (
                frame.f_code is self.fingerprint_code
                and frame.f_lasti == self.fingerprint_entry
                and frame.f_locals.get("self") is self.service
            ):
                self.fingerprints += 1
            if frame.f_code is self.adoption_code:
                self.adoptions += 1
        super().observe(frame, event, arg)


async def _settled(case):
    # Wait for the actual original retention predicate, not a patched clock or
    # shortened age threshold. This is a correctness control, not latency UAT.
    deadline = time.monotonic() + 5
    paths = (case.permissions.path, case.source.path)
    while any(cache._stamp(path)[4] > time.time_ns() - cache.SETTLE_NS for path in paths):
        assert time.monotonic() < deadline, "real private store did not settle"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["permission", "manifest", "connection", "profile", "governance"])
async def test_unwrapped_stock_cache_keeps_fresh_checks_and_invalidates(snapshot_case, change):
    case = snapshot_case
    governance, _ = shared._loop_projection(case)
    context = shared._context(case)
    cache.forget()
    try:
        await _settled(case)
        probe = _StockProbe(case)
        with probe.installed():
            first, _, _, _ = await shared._compose(case, context)
            assert first is not None
            cold_reads = len(probe.read_threads)
            assert cold_reads == 1
            warm, _, _, _ = await shared._compose(case, context)
            assert warm is not None
            assert shared._catalog_projection(warm) == shared._catalog_projection(first)
            assert len(probe.read_threads) == cold_reads
            assert probe.fingerprints == 2
            if change == "permission":
                case.permissions.set_tool_state("local:one", "first", "deny")
            elif change == "manifest":
                case.local.manifest_provider = lambda: {"tools": []}
            elif change == "connection":
                case.local.client = SimpleNamespace(sessions={"one": SimpleNamespace(_closed=False)})
            elif change == "profile":
                case.permissions.ensure_profile("focused")
                context = shared._configure(context, tool_policy_profile_id="focused")
            else:
                governance.denied = True
            changed, _, _, _ = await shared._compose(case, context)
            assert probe.fingerprints == 3
            if change == "governance":
                assert changed is None
            else:
                assert len(probe.read_threads) == cold_reads + 1
                assert changed is not None
                if change == "permission":
                    assert [hub.name for hub, _ in changed._entry_by_llm_name.values()] == ["sample_builtin"]
                elif change == "manifest":
                    assert [hub.name for hub, _ in changed._entry_by_llm_name.values()] == ["first"]
                elif change == "connection":
                    assert changed.not_connected_count == 0
    finally:
        cache.forget()


@pytest.mark.asyncio
async def test_stock_cold_adopts_once_and_warm_reuses_without_payload_decode(snapshot_case):
    case = snapshot_case
    shared._loop_projection(case)
    cache.forget()
    try:
        await _settled(case)
        probe = _StockProbe(case)
        with probe.installed():
            first, _, local, _ = await shared._compose(case, shared._context(case))
            assert first is not None and local is not None
            assert probe.adoptions == 1
            assert probe.native_calls > 0 and probe.admissions > 0
            assert len(probe.permission_threads) == 1
            assert all(thread is not probe.loop_thread for thread in probe.permission_threads)
            assert first.composed_servers == (("one", False, False),)
            warm, _, _, _ = await shared._compose(case, shared._context(case))
            assert warm is not None
            assert probe.adoptions == 1
            assert len(probe.permission_threads) == 1
            assert len(probe.read_threads) == 1
            assert probe.fingerprints == 2
    finally:
        cache.forget()


@pytest.mark.asyncio
async def test_incomplete_stock_inventory_never_supplies_retention_receipt(snapshot_case):
    from tldw_chatbook.MCP.console_tool_preparation import prepare_console_tools

    case = snapshot_case
    shared._loop_projection(case)
    def unavailable():
        raise OSError("private inventory unavailable")
    case.local.manifest_provider = unavailable
    prepared = await prepare_console_tools(
        case.service, include_mcp_catalog=True, need_local_switch=True,
        builtin_raw_name_exclusions=frozenset(), owned_profile_ids=frozenset(),
    )
    assert prepared is not None and prepared.composed_servers is None
    assert [tool.name for tool in prepared.tools] == ["first"]


@pytest.mark.asyncio
async def test_detached_extension_consumer_has_isolated_data_and_zero_native_reads(snapshot_case):
    from tldw_chatbook.MCP.console_tool_preparation import prepare_console_tools

    case = snapshot_case
    shared._loop_projection(case)
    prepared = await prepare_console_tools(
        case.service, include_mcp_catalog=True, need_local_switch=True,
        builtin_raw_name_exclusions=frozenset(), owned_profile_ids=frozenset(),
    )
    assert prepared is not None
    # A tiny detached consumer uses the existing result/converter contract,
    # with no new production category, provider, store or permission seam.
    def extension_description(result):
        return [(tool.name, tool.to_hub_tool().input_schema) for tool in result.tools]
    probe = _StockProbe(case)
    with probe.installed():
        first = extension_description(prepared)
        assert first == [("first", None), ("sample_builtin", {"type": "object"})]
        first[1][1]["type"] = "array"
        assert extension_description(prepared)[1][1] == {"type": "object"}
    assert not probe.permission_threads and not probe.read_threads
    assert probe.fingerprints == probe.adoptions == 0
    assert probe.native_calls == probe.admissions == 0
