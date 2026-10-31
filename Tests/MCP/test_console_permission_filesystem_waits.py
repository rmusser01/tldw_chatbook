"""Original permission gates and physical owners under finite handoff scheduling."""

import os
import sys
import threading

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as controls
from tldw_chatbook.Backup_Recovery import generation_witnesses, raw_participants
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import console_snapshot

local_root = controls.local_root
mcp_sources = controls.mcp_sources
catalog_store = controls.catalog_store
snapshot_case = controls.snapshot_case


class _PermissionProbe(controls._MaximumProbe):
    def __init__(self, case, captured):
        super().__init__(case.source, case.permissions)
        self.captured = captured
        self.entry = console_snapshot._CapturedSources.read_permission_payload.__code__
        self.thread = None
        self.interval = None
        self.acquisitions = []
        self.paths = []
        self.parent_fds = []
        self.witnesses = 0
        self.contexts = 0

    def observe(self, frame, event, arg):
        identity = threading.get_ident()
        if (
            event == "call"
            and frame.f_code is self.entry
            and frame.f_locals["self"] is self.captured
        ):
            self.thread = identity
            self.interval = sys.getswitchinterval()
        if identity == self.thread:
            code = frame.f_code
            if event == "call" and code is generation_witnesses._witnesses.__code__:
                self.witnesses += 1
            if (
                event == "call"
                and code is storage.StorageLease.execution_context.__code__
            ):
                self.contexts += 1
            if (
                event == "return"
                and code is storage.acquire_storage.__code__
                and type(arg) is storage.StorageLease
            ):
                self.acquisitions.append(arg)
                if frame.f_back.f_code is raw_participants._scope.__wrapped__.__code__:
                    self.paths.append(arg)
            if (
                event == "return"
                and code is raw_participants._pin_parent.__code__
                and frame.f_locals["state"].source is self.permissions
            ):
                self.parent_fds.append(arg)
            if event == "return" and code is self.entry:
                self.thread = None
        super().observe(frame, event, arg)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persisted,witnesses,contexts", [(False, 12, 15), (True, 14, 19)]
)
async def test_real_permission_read_keeps_every_gate_and_retires_owners(
    snapshot_case, persisted, witnesses, contexts
):
    case = snapshot_case
    if not persisted:
        case.permissions.path.unlink()
    expected = case.permissions.load()
    captured = console_snapshot._CapturedSources(case.service)
    probe = _PermissionProbe(case, captured)
    previous = sys.getswitchinterval()
    with probe.installed():
        payload = await console_snapshot._owned_worker(captured.read_permission_payload)
    assert payload == expected
    assert probe.interval <= 0.001 and sys.getswitchinterval() == previous
    assert probe.witnesses == witnesses and probe.contexts == contexts
    assert len(probe.paths) == 3 and len(probe.acquisitions) == 12
    assert all(lease not in storage._live_leases for lease in probe.acquisitions)
    assert not any(
        state.source is case.permissions for state in raw_participants._states.values()
    )
    assert probe.permission_threads and all(
        thread is not threading.current_thread() for thread in probe.permission_threads
    )
    assert probe.parent_fds
    for fd in probe.parent_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
