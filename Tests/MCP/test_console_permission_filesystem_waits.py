"""Passive native filesystem attribution for a real finite permission read."""

import json
import os
import threading
import time
from collections import defaultdict

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as controls
from tldw_chatbook.Backup_Recovery import generation_witnesses, raw_participants
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import console_snapshot

local_root = controls.local_root
mcp_sources = controls.mcp_sources
catalog_store = controls.catalog_store
snapshot_case = controls.snapshot_case


class _FilesystemProbe(controls._MaximumProbe):
    def __init__(self, case, captured):
        super().__init__(case.source, case.permissions)
        self.captured = captured
        self.entry = console_snapshot._CapturedSources.read_permission_payload.__code__
        self.thread = None
        self.starts = {}
        self.spans = defaultdict(list)
        self.leases = []
        self.paths = []
        self.witnesses = 0
        self.contexts = 0
        self.dependencies = []

    def observe(self, frame, event, arg):
        identity = threading.get_ident()
        if (
            event == "call"
            and frame.f_code is self.entry
            and frame.f_locals["self"] is self.captured
        ):
            self.thread = identity
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
                self.leases.append(arg)
                if frame.f_back.f_code is raw_participants._scope.__wrapped__.__code__:
                    self.paths.append(arg)
            if event == "call" and code is storage._Evidence.observe.__code__:
                evidence = frame.f_locals["self"]
                posture = {p for p, _ in evidence.posture}
                content = {p for p, _ in evidence.content}
                self.dependencies.append(
                    [len(posture), len(content), len(posture & content)]
                )
            if event in {"c_call", "c_return", "c_exception"} and (
                arg is os.stat or arg is os.lstat or arg is getattr(os, "statvfs", None)
            ):
                key = (identity, id(frame), arg.__name__)
                name = "os." + arg.__name__
                start, end = event == "c_call", event != "c_call"
            else:
                key = (identity, id(frame), code)
                name = code.co_qualname
                start, end = event == "call", event == "return"
            if start:
                self.starts[key] = (time.monotonic_ns(), time.thread_time_ns())
            elif end and key in self.starts:
                wall, cpu = self.starts.pop(key)
                self.spans[name].append(
                    (
                        (time.monotonic_ns() - wall) / 1e6,
                        (time.thread_time_ns() - cpu) / 1e6,
                    )
                )
            if event == "return" and code is self.entry:
                self.thread = None
        super().observe(frame, event, arg)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persisted,witnesses,contexts", [(False, 12, 15), (True, 14, 19)]
)
async def test_real_permission_read_keeps_every_gate_and_retires_owners(
    snapshot_case, persisted, witnesses, contexts, record_property
):
    case = snapshot_case
    if not persisted:
        case.permissions.path.unlink()
    expected = case.permissions.load()
    captured = console_snapshot._CapturedSources(case.service)
    probe = _FilesystemProbe(case, captured)
    with probe.installed():
        payload = await console_snapshot._owned_worker(captured.read_permission_payload)
    assert payload == expected
    assert probe.witnesses == witnesses and probe.contexts == contexts
    assert len(probe.paths) == 3 and len(probe.leases) == 12
    assert all(lease not in storage._live_leases for lease in probe.leases)
    assert not any(
        state.source is case.permissions for state in raw_participants._states.values()
    )
    assert probe.permission_threads and all(
        thread is not threading.current_thread() for thread in probe.permission_threads
    )
    selected = {
        "_CapturedSources.read_permission_payload",
        "_execution_selection_for",
        "_Evidence.observe",
        "_posture",
        "_content",
        "Path.resolve",
        "realpath",
        "_joinrealpath",
        "os.stat",
        "os.lstat",
        "os.statvfs",
        "_mount_read_only",
        "_derived_before",
        "_derived_reuse",
        "_ordinary_hold",
    }
    attribution = {
        name: {
            "calls": len(rows),
            "wall_ms": sum(w for w, _ in rows),
            "cpu_ms": sum(c for _, c in rows),
        }
        for name, rows in probe.spans.items()
        if name in selected
    }
    record_property(
        "filesystem_attribution_inclusive_diagnostic_only",
        json.dumps(attribution, sort_keys=True),
    )
    record_property(
        "evidence_posture_content_overlap_counts", json.dumps(probe.dependencies)
    )
