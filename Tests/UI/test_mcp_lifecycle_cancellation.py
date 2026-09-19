"""Lifecycle admission remains owned until cancelled work actually settles."""

import asyncio
import gc
import warnings

import pytest
from textual.widgets import Button
from textual.worker import WorkerCancelled

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_workbench import (
    LifecycleApp,
    LifecycleFakeHubService,
    _capture_notifications,
)
from tldw_chatbook.MCP.readiness import ReadinessState
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench


class CleanupService(LifecycleFakeHubService):
    def __init__(self, outcome="cancel"):
        super().__init__()
        self.started = asyncio.Event()
        self.cancelling = asyncio.Event()
        self.release = asyncio.Event()
        self.interrupted = False
        self.outcome = outcome

    async def connect_local_profile(self, profile_id):
        self.lifecycle_calls.append(("connect", profile_id))
        if (
            self.lifecycle_calls.count(("connect", "docs")) == 1
            and profile_id == "docs"
        ):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelling.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.interrupted = True
                    raise
                if self.outcome == "cancel":
                    raise
                if self.outcome == "error":
                    raise RuntimeError("Cleanup failed")
        return {"tools": [{"name": "fixture"}]}


async def ready(app, pilot):
    await pilot.pause()
    await finish(app)
    wb = app.query_one(MCPWorkbench)
    await wb._select_server_key("local:docs")
    return wb


async def finish(app):
    while workers := list(app.workers):
        results = await asyncio.gather(
            *(w.wait() for w in workers), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, WorkerCancelled
            ):
                raise result


@pytest.mark.asyncio
@private_profile_test
async def test_cancel_reserves_server_until_cleanup_then_permits_retry(request):
    app = LifecycleApp()
    service = app.unified_mcp_service = CleanupService()
    async with app.run_test(size=(120, 40)) as pilot:
        wb = await ready(app, pilot)
        notices = _capture_notifications(app)
        wb._start_lifecycle("local:docs", "docs", "connect")
        await service.started.wait()
        original = wb._in_flight["local:docs"]
        wb.on_mcp_inspector_cancel_requested(MCPInspector.CancelRequested("local:docs"))
        await service.cancelling.wait()
        await pilot.pause()
        during = {
            "owner": wb._in_flight.get("local:docs") is original,
            "state": wb._snapshot_for_display("local:docs").state,
            "message": wb._snapshot_for_display("local:docs").message,
            "notices": list(notices),
        }
        cancel = app.query(MCPInspector).first().query("#mcp-inspector-cancel")
        during["cancel_disabled"] = bool(cancel) and cancel.first(Button).disabled
        wb._start_lifecycle("local:docs", "docs", "connect")
        wb._start_lifecycle("local:other", "other", "connect")
        wb.on_mcp_inspector_cancel_requested(MCPInspector.CancelRequested("local:docs"))
        await pilot.pause()
        during["calls"] = list(service.lifecycle_calls)
        during["interrupted"] = service.interrupted
        service.release.set()
        await finish(app)
        await pilot.pause()
        assert during["owner"]
        assert during["state"] is ReadinessState.CHECKING
        assert "Cancelling" in during["message"]
        assert not any("Cancelled" in text for text, _ in during["notices"])
        assert during["cancel_disabled"]
        assert during["calls"] == [("connect", "docs"), ("connect", "other")]
        assert not during["interrupted"]
        assert "local:docs" not in wb._in_flight
        assert sum("Cancelled" in text for text, _ in notices) == 1
        wb._start_lifecycle("local:docs", "docs", "connect")
        await finish(app)
        assert service.lifecycle_calls.count(("connect", "docs")) == 2
        assert "local:docs" not in wb._in_flight


@pytest.mark.parametrize("boundary", ["button", "message"])
@pytest.mark.parametrize("replacement", ["docs", "other"])
@pytest.mark.asyncio
@private_profile_test
async def test_retired_cancel_cannot_stop_a_replacement_operation(
    request, boundary, replacement
):
    app = LifecycleApp()
    service = app.unified_mcp_service
    gates = [asyncio.Event(), asyncio.Event()]
    calls = []
    catalog = service.local_external_catalog

    async def two_profiles():
        rows = await catalog()
        return rows + [dict(rows[0], profile_id="other")]

    async def connect(profile_id):
        index = len(calls)
        calls.append(profile_id)
        await gates[index].wait()
        return {"tools": []}

    service.local_external_catalog = two_profiles
    service.connect_local_profile = connect
    async with app.run_test(size=(120, 40)) as pilot:
        wb = await ready(app, pilot)
        wb._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        inspector = app.query_one(MCPInspector)
        post = inspector.post_message
        held = []

        def hold(message):
            if (
                boundary == "button"
                and isinstance(message, Button.Pressed)
                and message.button.id == "mcp-inspector-cancel"
            ) or (
                boundary == "message"
                and isinstance(message, MCPInspector.CancelRequested)
            ):
                held.append(message)
                return True
            return post(message)

        inspector.post_message = hold
        inspector.query_one("#mcp-inspector-cancel", Button).press()
        await pilot.pause()
        assert len(held) == 1
        inspector.post_message = post
        gates[0].set()
        await finish(app)
        wb._start_lifecycle(f"local:{replacement}", replacement, "connect")
        await wb._select_server_key(f"local:{replacement}")
        await pilot.pause()
        worker = wb._in_flight[f"local:{replacement}"]
        post(held[0])
        await pilot.pause()
        cancelled = worker.is_cancelled
        gates[1].set()
        await finish(app)
        assert not cancelled


@pytest.mark.asyncio
@private_profile_test
async def test_cancel_before_start_does_not_create_service_coroutine(request):
    app = LifecycleApp()
    service = app.unified_mcp_service
    original = service.connect_local_profile
    created = []

    def create(profile_id):
        created.append(profile_id)
        return original(profile_id)

    service.connect_local_profile = create
    async with app.run_test() as pilot:
        wb = await ready(app, pilot)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            wb._start_lifecycle("local:docs", "docs", "connect")
            wb.on_mcp_inspector_cancel_requested(
                MCPInspector.CancelRequested("local:docs")
            )
            await finish(app)
            await pilot.pause()
            gc.collect()
        assert created == []
        assert service.lifecycle_calls == []
        assert not any("never awaited" in str(w.message) for w in caught)
        assert "local:docs" not in wb._in_flight
        wb._start_lifecycle("local:docs", "docs", "connect")
        await finish(app)
        assert created == ["docs"]


@pytest.mark.parametrize("outcome", ["success", "error"])
@pytest.mark.asyncio
@private_profile_test
async def test_cancel_request_does_not_hide_the_actual_terminal_outcome(
    request, outcome
):
    app = LifecycleApp()
    service = app.unified_mcp_service = CleanupService(outcome)
    async with app.run_test() as pilot:
        wb = await ready(app, pilot)
        notices = _capture_notifications(app)
        wb._start_lifecycle("local:docs", "docs", "connect")
        await service.started.wait()
        wb.on_mcp_inspector_cancel_requested(MCPInspector.CancelRequested("local:docs"))
        await service.cancelling.wait()
        service.release.set()
        await finish(app)
        await pilot.pause()
        assert not any("Cancelled" in text for text, _ in notices)
        expected = "connected" if outcome == "success" else "Cleanup failed"
        assert any(expected in text for text, _ in notices)
        assert "local:docs" not in wb._in_flight


@pytest.mark.asyncio
@private_profile_test
async def test_success_keeps_admission_until_final_projection_is_collected(request):
    app = LifecycleApp()
    async with app.run_test() as pilot:
        wb = await ready(app, pilot)
        collect = wb._collect_snapshots
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_collect():
            entered.set()
            await release.wait()
            return await collect()

        wb._collect_snapshots = held_collect
        notices = _capture_notifications(app)
        wb._start_lifecycle("local:docs", "docs", "connect")
        await entered.wait()
        wb.on_mcp_inspector_cancel_requested(MCPInspector.CancelRequested("local:docs"))
        wb._start_lifecycle("local:docs", "docs", "refresh")
        # Collection is deliberately held; a full message-pump drain waits on it.
        for _ in range(10):
            await asyncio.sleep(0)
        calls = list(app.unified_mcp_service.lifecycle_calls)
        release.set()
        await finish(app)
        await pilot.pause()
        assert calls == [("connect", "docs")]
        assert not any(
            "Cancelled" in text or "Cancelling" in text for text, _ in notices
        )
        assert "local:docs" not in wb._in_flight


@pytest.mark.asyncio
@private_profile_test
async def test_cancel_render_keeps_original_operation_across_detail_await(request):
    app = LifecycleApp()
    gates = [asyncio.Event(), asyncio.Event()]
    started = []

    async def connect(profile_id):
        index = len(started)
        started.append(profile_id)
        await gates[index].wait()
        return {"tools": []}

    app.unified_mcp_service.connect_local_profile = connect
    async with app.run_test(size=(120, 40)) as pilot:
        wb = await ready(app, pilot)
        wb._start_lifecycle("local:docs", "docs", "connect")
        await pilot.pause()
        original = wb._in_flight["local:docs"]
        show = wb._show_selected_detail
        entered, release = asyncio.Event(), asyncio.Event()
        inspector = app.query_one(MCPInspector)
        update = inspector.update_readiness
        rendered = []

        async def held_show(canvas, selected):
            if not entered.is_set():
                entered.set()
                await release.wait()
            await show(canvas, selected)

        async def capture_binding(selected, **kwargs):
            if not rendered and selected.state is ReadinessState.CHECKING:
                rendered.append(kwargs.get("cancel_operation"))
            await update(selected, **kwargs)

        wb._show_selected_detail = held_show
        inspector.update_readiness = capture_binding
        rendering = asyncio.create_task(wb._sync_children())
        await entered.wait()
        gates[0].set()
        await original.wait()
        async with asyncio.timeout(5):
            while "local:docs" in wb._in_flight:
                await asyncio.sleep(0)
        wb._start_lifecycle("local:docs", "docs", "connect")
        replacement = wb._in_flight["local:docs"]
        release.set()
        await rendering
        wb.on_mcp_inspector_cancel_requested(
            MCPInspector.CancelRequested("local:docs", rendered[0])
        )
        cancelled = replacement.is_cancelled
        gates[1].set()
        await finish(app)
        assert rendered[0] is original
        assert not cancelled
