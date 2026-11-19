"""Current Runs view owns publication of native interval reads (TASK-37267)."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.timer import Timer
from textual.widgets import ListView, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService
from tldw_chatbook.UI.Screens.research_screen import ResearchScreen


async def _until(predicate):
    async with asyncio.timeout(8):
        while not predicate():
            await asyncio.sleep(0.01)


def _text(window, selector="#research-run-detail"):
    return str(window.query_one(selector, Static).render())


def _record(request, name, value):
    request.node.user_properties.append((name, value))


def _snapshot(window):
    view = window.query_one("#research-run-list", ListView)
    return {
        "source": window.current_source,
        "select": window.query_one("#research-source-select", Select).value,
        "selected": None if window.selected_run is None else window.selected_run["id"],
        "highlighted": None
        if view.highlighted_child is None
        else view.highlighted_child.id,
        "detail": _text(window),
        "selection_revision": getattr(window, "_selection_revision", None),
        "list_owner": id(view),
        "detail_owner": id(window.query_one("#research-run-detail", Static)),
    }


class _Harness(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def __init__(self, scope, local):
        super().__init__()
        self.context = SimpleNamespace(
            research_scope_service=scope, local_research_service=local
        )

    async def on_mount(self):
        await self.push_screen(ResearchScreen(self.context))


class _HeldRead:
    """Hold the actual scope return, after SQLite has freed its executor."""

    def __init__(self, scope, monkeypatch):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.calls = []
        original = scope.get_run

        async def read(run_id, *, mode=None):
            result = await original(run_id, mode=mode)
            task = asyncio.current_task()
            self.calls.append(
                {
                    "id": run_id,
                    "mode": str(mode),
                    "result_id": result["id"],
                    "task": task.get_name(),
                    "coroutine": task.get_coro().__qualname__,
                }
            )
            if run_id == "run-a" and not self.entered.is_set():
                self.entered.set()
                try:
                    await self.release.wait()
                finally:
                    self.finished.set()
            return result

        monkeypatch.setattr(scope, "get_run", read)

    async def wait(self):
        await asyncio.wait_for(self.entered.wait(), 8)
        assert self.calls[0]["result_id"] == "run-a"
        assert self.calls[0]["coroutine"] == "Timer._run_timer"

    async def finish(self):
        self.release.set()
        if self.entered.is_set():
            await asyncio.wait_for(self.finished.wait(), 8)
        # The caller publishes synchronously after this return; let it finish.
        await asyncio.sleep(0)


@asynccontextmanager
async def _mounted(tmp_path, *, server=None, status="running"):
    local = LocalResearchService(tmp_path / "research.sqlite3")
    local.create_run(
        id="run-a",
        query="LOCAL A ALPHA",
        status=status,
        control_state="running",
        progress_message="A initial progress",
    )
    local.create_run(id="run-b", query="LOCAL B BRAVO", status="draft")
    for run_id, content in (("run-a", "A owned report"), ("run-b", "B owned report")):
        local.save_artifact(
            run_id,
            artifact_name="report.md",
            content_type="text/markdown",
            content=content,
        )
        local.record_run_event(
            run_id, "payload_control", {"progress_message": f"{run_id} owned event"}
        )
    scope = ResearchScopeService(local_service=local, server_service=server)
    app = _Harness(scope, local)
    try:
        async with app.run_test(size=(160, 50)) as pilot:
            window = app.screen.query_one("#research-window")
            await window.load_runs("local")
            yield pilot, window, scope, local
    finally:
        local.close()


async def _select(pilot, window, run_id):
    view = window.query_one("#research-run-list", ListView)
    view.focus()
    view.index = next(
        i for i, row in enumerate(view.children) if row.id == f"research-run-{run_id}"
    )
    await pilot.press("enter")
    await _until(
        lambda: window.selected_run is not None and window.selected_run["id"] == run_id
    )
    assert view.highlighted_child.id == f"research-run-{run_id}"


async def _source(pilot, window, source):
    select = window.query_one("#research-source-select", Select)
    select.focus()
    await pilot.press("enter", "down" if source == "server" else "up", "enter")
    await _until(
        lambda: select.value == source
        and window.current_source == source
        and window.status_message != ""
    )


async def _payload(window, run_id):
    # The existing controller artifact loader passes positional arguments to
    # the scope's keyword-only API. Keep that separate defect out of this
    # ownership control: prepare the view from actual scope/SQLite payloads.
    scope = window.controller.scope_service
    window.current_bundle = await window.controller.get_bundle("local", run_id)
    window.current_artifact = await scope.get_artifact(
        run_id=run_id, artifact_name="report.md", mode="local"
    )
    window._render_bundle_detail()
    window._render_artifact_detail()
    await window.watch_selected_run_events()
    assert window.current_bundle["run"]["id"] == run_id
    assert window.current_artifact["content"] == (
        "A owned report" if run_id == "run-a" else "B owned report"
    )
    return window.current_bundle, window.current_artifact, window.event_log_entries


@pytest.mark.asyncio
@private_profile_test
async def test_refresh_preserves_newer_run_selection(request, tmp_path, monkeypatch):
    """An older A tick cannot retarget selected B or attach B payload to A."""
    async with _mounted(tmp_path) as (pilot, window, scope, _local):
        held = _HeldRead(scope, monkeypatch)
        try:
            await _select(pilot, window, "run-a")
            await held.wait()
            before = _snapshot(window)
            await _select(pilot, window, "run-b")
            payload = await _payload(window, "run-b")
            selected = _snapshot(window)
            assert "LOCAL B BRAVO" in selected["detail"]
            await held.finish()
            final = _snapshot(window)
            _record(
                request,
                "publication",
                json.dumps(
                    dict(
                        before=before, selected=selected, final=final, reads=held.calls
                    )
                ),
            )
            assert final["selected"] == "run-b"
            assert final["highlighted"] == "research-run-run-b"
            assert "LOCAL B BRAVO" in final["detail"]
            assert window._selected_run_id() == "run-b"
            assert window.current_bundle is payload[0]
            assert window.current_artifact is payload[1]
            assert window.event_log_entries is payload[2]
            assert "B owned report" in _text(window, "#research-artifact-detail")
            assert "run-b owned event" in _text(window, "#research-event-log")
            assert final["list_owner"] == before["list_owner"]
            assert final["detail_owner"] == before["detail_owner"]
        finally:
            await held.finish()


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("return_local", [False, True], ids=["server", "source-aba"])
async def test_local_refresh_cannot_restore_a_departed_source(
    request, tmp_path, monkeypatch, return_local
):
    """A Local tick loses authority on departure, including Local→Server→Local."""
    async with _mounted(tmp_path) as (pilot, window, scope, local):
        held = _HeldRead(scope, monkeypatch)
        try:
            await _select(pilot, window, "run-a")
            await held.wait()
            before = _snapshot(window)
            await _source(pilot, window, "server")
            departed = _snapshot(window)
            assert departed["selected"] is None
            if return_local:
                await _source(pilot, window, "local")
                await _until(
                    lambda: len(
                        window.query_one("#research-run-list", ListView).children
                    )
                    == 2
                )
                await _select(pilot, window, "run-a")
                # The old tick has already fetched the initial progress. A later
                # native selection sees a distinct, real committed record.
                await scope.pause_run("run-a", mode="local")
                await window.load_runs("local")
                await _select(pilot, window, "run-a")
                assert local.get_run("run-a")["control_state"] == "paused"
                assert window.selected_run["control_state"] == "paused"
                await _payload(window, "run-a")
            current = _snapshot(window)
            await held.finish()
            final = _snapshot(window)
            _record(
                request,
                "publication",
                json.dumps(
                    dict(
                        before=before,
                        departed=departed,
                        current=current,
                        final=final,
                        reads=held.calls,
                    )
                ),
            )
            if return_local:
                assert final["selected"] == "run-a"
                assert window.selected_run["control_state"] == "paused"
                assert "Control: paused" in final["detail"]
                assert window.current_artifact["content"] == "A owned report"
            else:
                assert final["selected"] is None
                with pytest.raises(ValueError, match="No research run"):
                    window._selected_run_id()
                assert "LOCAL A ALPHA" not in final["detail"]
                assert window.current_bundle is None
        finally:
            await held.finish()


class _EmptyServer:
    async def list_runs(self, **_kwargs):
        return []


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("available", [False, True], ids=["unavailable", "empty"])
async def test_source_switch_clears_previous_detail(request, tmp_path, available):
    """Both catalog outcomes paint empty detail, then later valid selection works."""
    async with _mounted(tmp_path, server=_EmptyServer() if available else None) as (
        pilot,
        window,
        _scope,
        _local,
    ):
        await _select(pilot, window, "run-b")
        await _payload(window, "run-b")
        await _source(pilot, window, "server")
        empty = _snapshot(window)
        _record(request, "empty_projection", json.dumps(empty))
        assert empty["selected"] is None
        assert empty["detail"] == "No research run selected."
        assert not window.query_one("#research-run-list", ListView).children
        assert window.current_bundle is None
        assert window.current_artifact is None
        assert window.event_log_entries == []
        await _source(pilot, window, "local")
        await _select(pilot, window, "run-b")
        assert "LOCAL B BRAVO" in _text(window)
        assert window._selected_run_id() == "run-b"


@pytest.mark.asyncio
@private_profile_test
async def test_native_refresh_updates_progress_and_retains_current_payload(
    request, tmp_path, monkeypatch
):
    """A still-current tick publishes real SQLite progress without discarding payload."""
    async with _mounted(tmp_path) as (pilot, window, scope, local):
        await _select(pilot, window, "run-a")
        payload = await _payload(window, "run-a")
        local.update_run_progress(
            "run-a", phase="collecting", progress_message="A new SQLite progress"
        )
        held = _HeldRead(scope, monkeypatch)
        try:
            await held.wait()
            await held.finish()
            await _until(lambda: "A new SQLite progress" in _text(window))
            _record(
                request,
                "ordinary_refresh",
                json.dumps(dict(state=_snapshot(window), reads=held.calls)),
            )
            assert window.selected_run["progress_message"] == "A new SQLite progress"
            assert window.current_bundle is payload[0]
            assert window.current_artifact is payload[1]
            assert window.event_log_entries is payload[2]
            assert "A owned report" in _text(window, "#research-artifact-detail")
            assert "run-a owned event" in _text(window, "#research-event-log")
        finally:
            await held.finish()


class _RunningServer:
    async def list_runs(self, **_kwargs):
        return [{"id": "server-run", "query": "SERVER DELTA", "status": "running"}]


@pytest.mark.asyncio
@private_profile_test
async def test_native_refresh_skips_terminal_and_server_selections(
    request, tmp_path, monkeypatch
):
    """Observe real interval completion; neither skip branch may issue a read."""
    async with _mounted(tmp_path, server=_RunningServer(), status="completed") as (
        pilot,
        window,
        scope,
        _local,
    ):
        ticks, reads = [], []
        original_tick, original_read = Timer._tick, scope.get_run

        async def tick(timer, **kwargs):
            await original_tick(timer, **kwargs)
            if getattr(timer._callback, "__self__", None) is window:
                ticks.append(_snapshot(window))

        async def read(run_id, *, mode=None):
            reads.append((run_id, str(mode)))
            return await original_read(run_id, mode=mode)

        monkeypatch.setattr(Timer, "_tick", tick)
        monkeypatch.setattr(scope, "get_run", read)
        await _select(pilot, window, "run-a")
        await _until(lambda: any(t["selected"] == "run-a" for t in ticks))
        assert window.selected_run["status"] == "completed"
        assert reads == []
        await _source(pilot, window, "server")
        await _select(pilot, window, "server-run")
        await _until(lambda: any(t["selected"] == "server-run" for t in ticks))
        _record(request, "skipped_ticks", json.dumps(dict(ticks=ticks, reads=reads)))
        assert reads == []
        assert window.selected_run["status"] == "running"
        assert window._selected_run_id() == "server-run"
        assert "SERVER DELTA" in _text(window)
