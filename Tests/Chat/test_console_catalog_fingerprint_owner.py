"""Fresh catalog fingerprints use the selected original permission owner."""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import ExitStack, contextmanager

import pytest

from Tests.Chat.test_console_control_trace import _observe
from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case
from Tests.Chat.test_console_first_send_atomicity import _until

domain_case = _domain_case
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _submit(case, controls, sequence):
    """Exercise the real retained command handoff, without a model request."""
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ControlReceipt,
    )
    from tldw_chatbook.MCP.console_snapshot import read_run_catalog_fingerprint

    loop = asyncio.get_running_loop()
    receipt = ControlReceipt(
        ControlIdentity("profile", "service", sequence, f"fingerprint-{sequence}"),
        loop.create_future(),
        retired=loop.create_future(),
    )
    controls._retain_control_work(
        receipt, read_run_catalog_fingerprint(case.sources.control_plane)
    )
    return receipt


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
@pytest.mark.parametrize("mode", ["fresh", "read_error", "cancel"])
async def test_fingerprint_reuses_local_owner_and_retains_actual_read(
    domain_case, mode
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        PermissionAction,
    )
    from tldw_chatbook.Chat.console_control_permissions import SetKillSwitch
    from tldw_chatbook.MCP.local_permission_operation import active
    from tldw_chatbook.MCP.permission_process import PermissionProcess
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    case = domain_case
    operations = case.runtime._execution_service.ensure_permission_operations()
    await operations.set_kill_switch(False)
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    owners, child_starts = [], []
    entered, release = threading.Event(), threading.Event()
    failure = OSError("original fingerprint read failed")

    def permission_entry(frame):
        if frame.f_locals["self"] is not operations._store:
            return
        owner = active(operations._store)
        assert owner is not None
        assert owner.source is operations._store
        assert owner.creator[1] is threading.current_thread()
        assert owner.creator[1] is not threading.main_thread()
        owners.append(owner)
        if mode == "read_error":
            raise failure

    def permission_return(frame):
        if mode == "cancel" and frame.f_locals["self"] is operations._store:
            entered.set()
            assert release.wait(10), "Original fingerprint worker was not released"

    code = MCPPermissionStore.catalog_fingerprint.__wrapped__.__code__
    with ExitStack() as observed:
        observed.enter_context(
            _observe(code, sys.monitoring.events.PY_START, permission_entry)
        )
        observed.enter_context(
            _observe(code, sys.monitoring.events.PY_RETURN, permission_return)
        )
        observed.enter_context(
            _observe(
                PermissionProcess._start.__code__,
                sys.monitoring.events.PY_START,
                lambda _frame: child_starts.append(True),
            )
        )
        receipt = _submit(case, controls, 1)
        try:
            if mode == "cancel":
                assert await _until(entered.is_set)
                assert controls.admitted_effects == ("permission",)
                receipt.work.cancel()
                await asyncio.sleep(0)
                assert not receipt.retired.done()
                assert not owners[0].retired
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await receipt.applied
            elif mode == "read_error":
                with pytest.raises(OSError) as caught:
                    await receipt.applied
                assert caught.value is failure
            else:
                before = await receipt.applied
            assert await receipt.retired is True
            assert len(owners) == 1 and owners[0].retired
            assert not controls.admitted_effects
            if mode != "read_error":
                assert owners[0].observations["raw"]
                assert owners[0].observations["lease"]
            if mode == "fresh":
                mutation = controls.dispatch(
                    PermissionAction(
                        ControlIdentity("profile", "service", 2, "mutate"),
                        SetKillSwitch(value=True),
                    )
                )
                await mutation.applied
                assert await mutation.retired is True
                second = _submit(case, controls, 3)
                after = await second.applied
                assert await second.retired is True
                assert before[0] != after[0]
                assert before[1] == after[1]
                assert len(owners) == 2 and owners[1].retired
            assert not child_starts
            assert not operations._unresolved
        finally:
            release.set()
            await asyncio.gather(receipt.work, return_exceptions=True)


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
async def test_fingerprint_cannot_use_an_old_selected_source(domain_case):
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, FenceSource
    from tldw_chatbook.MCP.permission_process import PermissionProcess
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    case = domain_case
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="profile", service_epoch="service"
    )
    fence = controls.dispatch(
        FenceSource(ControlIdentity("profile", "service", 1, "fence"), "configuration")
    )
    # FenceSource publishes the drained fence itself, as in the wire worker.
    controls._require_source_publication(await fence.applied)
    entered = []
    with ExitStack() as observed:
        for code in (
            MCPPermissionStore.catalog_fingerprint.__wrapped__.__code__,
            PermissionProcess._start.__code__,
        ):
            observed.enter_context(
                _observe(
                    code,
                    sys.monitoring.events.PY_START,
                    lambda _frame: entered.append(True),
                )
            )
        receipt = _submit(
            case, controls, 2
        )  # Captures the original, now-fenced revision.
        with pytest.raises(RuntimeError, match="fenced"):
            await receipt.applied
        assert await receipt.retired is True
        assert not entered
        assert not controls.admitted_effects


async def test_fingerprint_never_returns_after_scope_retirement_refusal(monkeypatch):
    """A narrow ordering control; real native retirement has separate controls."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat import console_control_tools
    from tldw_chatbook.MCP.console_snapshot import _CapturedFingerprint

    calls = []
    failure = RuntimeError("permission retirement is uncertain")

    @contextmanager
    def refused(_service):
        yield
        raise failure

    monkeypatch.setattr(console_control_tools, "service_permission_scope", refused)
    captured = object.__new__(_CapturedFingerprint)
    captured.service = object()
    captured.permission = SimpleNamespace(
        catalog_fingerprint=lambda: calls.append("permission")
    )
    captured.local = SimpleNamespace(
        _catalog_fingerprint_sources=lambda: calls.append("local")
    )
    with pytest.raises(RuntimeError) as caught:
        captured.read_fingerprint_sources()
    assert caught.value is failure
    assert calls == ["permission"]
