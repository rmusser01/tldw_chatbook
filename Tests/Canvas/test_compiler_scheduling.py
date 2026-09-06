"""Compilation must yield without lending stale work mutation authority."""

import asyncio
import inspect
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.gateway import CanvasGatewayScope
from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController

SOURCE = "<!doctype html><title>Example</title><p>one</p>"


def test_existing_block_replay_does_not_recompile_historical_source(monkeypatch):
    authority, _controller, _live = setup_authority()
    arguments = {
        "session_id": "session",
        "source": SOURCE,
        "source_message_id": "message",
        "origin_message_id": "message",
        "source_turn_id": "turn",
        "block_index": 0,
        "block_identity": "message:canvas-html:0",
    }
    original = authority.import_html(**arguments)

    def unavailable_compiler(_source):
        raise AssertionError("a historical replay must not recompile")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document",
        unavailable_compiler,
    )
    assert authority.import_html(**arguments) == original


def setup_authority():
    scope = CanvasScope("session", "session", ("message",), None, None, "run")
    controller = ConsoleCanvasController()
    controller.activate_session("session")
    live = {"scope": scope, "enabled": True}
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _session: live["scope"],
        canvas_controller=controller,
        enabled_reader=lambda: live["enabled"],
    )
    return authority, controller, live


class DelayedCompiler:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, source):
        self.started.set()
        if not self.release.wait(2):
            raise AssertionError("compiler was not released")
        return compile_canvas_document(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalidate", [None, "branch", "disable", "dispose", "reincarnate"]
)
async def test_native_preview_yields_during_compilation(monkeypatch, invalidate):
    authority, controller, live = setup_authority()
    info = authority.import_html(session_id="session", source=SOURCE)
    scope = authority.gateway_scope(
        session_id="session",
        browser_session_id="browser",
        canvas_id=info.canvas_id,
        revision_id=info.revision_id,
    )
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Canvas.native_authority.compile_canvas_document", delayed
    )

    async def resolve():
        result = authority.resolve_render_plan(scope)
        return await result if inspect.isawaitable(result) else result

    task = asyncio.create_task(resolve())
    try:
        await asyncio.sleep(0.03)
        assert delayed.started.is_set()
        assert not task.done(), "compilation blocked the event loop"
        if invalidate == "branch":
            live["scope"] = replace(live["scope"], active_message_ids=("sibling",))
        elif invalidate == "disable":
            live["enabled"] = False
        elif invalidate == "dispose":
            authority.dispose()
        elif invalidate == "reincarnate":
            controller.activate_session("session")
    finally:
        delayed.release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
    if invalidate:
        assert isinstance(result, RuntimeError)
    else:
        result.source_identity.verify_source(SOURCE)


@pytest.mark.asyncio
async def test_tool_compilation_releases_shared_lock_and_fences_cancel(monkeypatch):
    _authority, controller, live = setup_authority()
    controller.register_run(
        live["scope"], assistant_message_id="assistant", temporary=True
    )
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document", delayed
    )
    task = asyncio.create_task(
        asyncio.to_thread(
            controller.create_canvas,
            live["scope"],
            tool_call_id="tool",
            title="Example",
            html=SOURCE,
        )
    )
    try:
        assert await asyncio.to_thread(delayed.started.wait, 1)
        # This is the same lock acquired synchronously by UI readers and cancellation.
        assert controller._lock.acquire(blocking=False), (
            "compiler holds the UI-visible lock"
        )
        controller._lock.release()
        controller.finish_run("run", "cancelled")
    finally:
        delayed.release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
    assert isinstance(result, RuntimeError)
    assert controller.run_revision_count("run") == 0


@pytest.mark.asyncio
async def test_served_preview_yields_during_compilation(monkeypatch):
    from tldw_chatbook.Web_Server.serve import _ServedCanvasAuthorityProxy

    proxy = _ServedCanvasAuthorityProxy(SimpleNamespace())

    async def read(_scope):
        return {}, {"source": SOURCE, "runtime_profile": "canvas-v1"}

    monkeypatch.setattr(proxy, "_read", read)

    async def request(_scope, _type, _payload):
        return SimpleNamespace(
            payload={
                "session_id": "session",
                "conversation_id": "session",
                "active_message_ids": ["message"],
                "selected_canvas_id": "canvas",
                "selected_revision_id": "revision",
                "selection_generation": "intent-a",
            }
        )

    monkeypatch.setattr(proxy, "_request", request)
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.serve.compile_canvas_document", delayed
    )
    task = asyncio.create_task(
        proxy.resolve_render_plan(
            CanvasGatewayScope("browser", "session", "canvas", "revision")
        )
    )
    try:
        await asyncio.sleep(0.03)
        assert delayed.started.is_set()
        assert not task.done(), "served compilation blocked the event loop"
    finally:
        delayed.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalidate",
    ["branch", "disable", "dispose", "reincarnate", "cancel", "view", "rebind"],
)
async def test_import_preparation_cannot_mutate_after_invalidation(
    monkeypatch, invalidate
):
    authority, controller, live = setup_authority()
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document", delayed
    )
    operation = getattr(authority, "import_html_async", None)
    assert callable(operation), "HTML import has no yielding preparation boundary"
    current_view = [True]
    task = asyncio.create_task(
        operation(
            session_id="session", source=SOURCE, _is_current=lambda: current_view[0]
        )
    )
    try:
        assert await asyncio.to_thread(delayed.started.wait, 1)
        assert not task.done()
        if invalidate == "branch":
            live["scope"] = replace(live["scope"], active_message_ids=("sibling",))
        elif invalidate == "disable":
            live["enabled"] = False
        elif invalidate == "dispose":
            authority.dispose()
        elif invalidate == "reincarnate":
            controller.activate_session("session")
        elif invalidate == "view":
            current_view[0] = False
        elif invalidate == "rebind":
            authority.rebind_view(
                scope_resolver=lambda _session: live["scope"],
                bridge_sink=None,
                auto_open=None,
            )
        else:
            task.cancel()
    finally:
        delayed.release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
    assert isinstance(result, (RuntimeError, asyncio.CancelledError))
    assert controller.list_session_canvases(live["scope"], temporary=True) == ()
    assert not authority._selection


@pytest.mark.asyncio
async def test_cancelled_waiters_keep_actual_worker_admission():
    from tldw_chatbook.Canvas.compilation import CanvasCompilation

    admission = CanvasCompilation()
    delayed = [DelayedCompiler(), DelayedCompiler()]
    tasks = [
        asyncio.create_task(
            admission.run_async(lambda compiler=compiler: compiler(SOURCE))
        )
        for compiler in delayed
    ]
    try:
        for compiler in delayed:
            assert await asyncio.to_thread(compiler.started.wait, 1)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with pytest.raises(RuntimeError, match="canvas_compilation_busy"):
            await admission.run_async(lambda: compile_canvas_document(SOURCE))
    finally:
        for compiler in delayed:
            compiler.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_worker_failure_is_consumed_and_restores_admission(cancel_waiter):
    from tldw_chatbook.Canvas.compilation import CanvasCompilation

    admission = CanvasCompilation()
    started = threading.Event()
    release = threading.Event()
    settled = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(
        lambda _loop, context: unhandled.append(context["message"])
    )

    def fail():
        started.set()
        try:
            assert release.wait(2)
            raise RuntimeError("synthetic_compilation_failure")
        finally:
            loop.call_soon_threadsafe(settled.set)

    task = asyncio.create_task(admission.run_async(fail))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        if cancel_waiter:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        if not cancel_waiter:
            with pytest.raises(RuntimeError, match="synthetic_compilation_failure"):
                await task
        await asyncio.wait_for(settled.wait(), 1)
        # Let the executor completion and shield callbacks drain before checking
        # the loop's unretrieved-exception channel.
        await asyncio.sleep(0.05)
        assert admission.run(lambda: admission.run(lambda: "both slots free")) == (
            "both slots free"
        )
        assert unhandled == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_tool_update_rechecks_parent_after_unlocked_compile(monkeypatch):
    _authority, controller, live = setup_authority()
    controller.register_run(
        live["scope"], assistant_message_id="assistant", temporary=True
    )
    root = controller.create_canvas(
        live["scope"], tool_call_id="create", title="Example", html=SOURCE
    )
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document", delayed
    )
    arguments = {
        "canvas_id": root.revision.canvas_id,
        "expected_parent_revision_id": root.revision.revision_id,
        "html": SOURCE,
    }
    task = asyncio.create_task(
        asyncio.to_thread(
            controller.update_canvas, live["scope"], tool_call_id="delayed", **arguments
        )
    )
    try:
        assert await asyncio.to_thread(delayed.started.wait, 1)
        monkeypatch.setattr(
            "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document",
            compile_canvas_document,
        )
        winner = controller.update_canvas(
            live["scope"], tool_call_id="winner", **arguments
        )
    finally:
        delayed.release.set()
    result = await task
    assert result.current_revision_id == winner.revision.revision_id
    assert controller.run_revision_count("run") == 2


@pytest.mark.asyncio
async def test_served_preview_rechecks_child_branch_after_compilation(monkeypatch):
    from tldw_chatbook.Web_Server.serve import _ServedCanvasAuthorityProxy

    proxy = _ServedCanvasAuthorityProxy(SimpleNamespace())
    branch = ["message"]

    async def read(_scope):
        return {}, {"source": SOURCE, "runtime_profile": "canvas-v1"}

    async def request(_scope, _type, _payload):
        return SimpleNamespace(
            payload={
                "session_id": "session",
                "conversation_id": "session",
                "active_message_ids": list(branch),
                "selected_canvas_id": "canvas",
                "selected_revision_id": "revision",
                "selection_generation": "intent-a",
            }
        )

    monkeypatch.setattr(proxy, "_read", read)
    monkeypatch.setattr(proxy, "_request", request)
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.serve.compile_canvas_document", delayed
    )
    task = asyncio.create_task(
        proxy.resolve_render_plan(
            CanvasGatewayScope("browser", "session", "canvas", "revision")
        )
    )
    try:
        assert await asyncio.to_thread(delayed.started.wait, 1)
        branch[:] = ["sibling"]
    finally:
        delayed.release.set()
    with pytest.raises(RuntimeError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("detach", [False, True])
async def test_chat_screen_html_import_yields_and_checks_view_before_apply(
    monkeypatch, detach
):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    authority, controller, live = setup_authority()
    message = SimpleNamespace(
        id="message", persisted_message_id=None, turn_id="turn", trace_turn_id=None
    )
    store = SimpleNamespace(
        active_session_id="session",
        session_id_for_message=lambda _id: "session",
        active_path_message_ids=lambda _id: ("message",),
        get_message=lambda _id: message,
    )
    opened = []

    async def open_selection(**selection):
        opened.append(selection)
        return selection

    runtime = SimpleNamespace(
        canvas_controller=controller,
        canvas_authority_is_current=lambda candidate: candidate is authority,
    )
    screen = SimpleNamespace(
        is_mounted=True,
        _ensure_console_chat_store=lambda: store,
        _console_canvas_authority=lambda: authority,
        _console_canvas_scope=lambda _session_id: live["scope"],
        _console_runtime=lambda: runtime,
        _open_console_canvas_selection=open_selection,
    )
    reference = SimpleNamespace(
        message_id="message",
        create_new=False,
        block_index=0,
        identity="message:canvas-html:0",
    )
    delayed = DelayedCompiler()
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_canvas_controller.compile_canvas_document", delayed
    )
    task = asyncio.create_task(
        ChatScreen._open_console_canvas_block(screen, reference, SOURCE)
    )
    try:
        assert await asyncio.to_thread(delayed.started.wait, 1)
        assert not task.done()
        if detach:
            screen.is_mounted = False
    finally:
        delayed.release.set()
    result = (await asyncio.gather(task, return_exceptions=True))[0]
    if detach:
        assert isinstance(result, RuntimeError)
        assert opened == []
        assert controller.list_session_canvases(live["scope"], temporary=True) == ()
    else:
        assert opened == [result]
        assert len(controller.list_session_canvases(live["scope"], temporary=True)) == 1


@pytest.mark.asyncio
async def test_near_limit_operations_preserve_source_and_allow_loop_progress(
    monkeypatch,
):
    """Print content-free loop-lag evidence; timing values are not portable CI thresholds."""
    import json
    import statistics
    import time

    from scripts.canvas_runtime_quota_probe import build_synthetic_fixtures
    from tldw_chatbook.Canvas.limits import CanvasLimits
    from tldw_chatbook.Web_Server.serve import _ServedCanvasAuthorityProxy

    source = next(
        item.source
        for item in build_synthetic_fixtures(CanvasLimits())
        if item.identifier == "adversarial-combined-at-limit"
    )
    authority, controller, live = setup_authority()
    info = await authority.import_html_async(session_id="session", source=source)
    scope = authority.gateway_scope(
        session_id="session",
        browser_session_id="browser",
        canvas_id=info.canvas_id,
        revision_id=info.revision_id,
    )
    proxy = _ServedCanvasAuthorityProxy(SimpleNamespace())

    async def read(_scope):
        return {}, {"source": source, "runtime_profile": "canvas-v1"}

    async def request(_scope, _type, _payload):
        return SimpleNamespace(
            payload={
                "session_id": "session",
                "conversation_id": "session",
                "active_message_ids": ["message"],
                "selected_canvas_id": scope.canvas_id,
                "selected_revision_id": scope.revision_id,
                "selection_generation": "intent-a",
            }
        )

    monkeypatch.setattr(proxy, "_read", read)
    monkeypatch.setattr(proxy, "_request", request)

    async def import_document():
        live["scope"] = replace(live["scope"], run_id=live["scope"].run_id + "-next")
        return await authority.import_html_async(
            session_id="session", source=source, create_new=True
        )

    evidence = {}
    for name, operation in (
        ("native_preview", lambda: authority.resolve_render_plan(scope)),
        ("served_preview_compile", lambda: proxy.resolve_render_plan(scope)),
        ("html_import_with_title", import_document),
    ):
        walls, gaps = [], []
        for _ in range(5):
            stopped = asyncio.Event()

            async def heartbeat(stopped=stopped, gaps=gaps):
                previous = time.perf_counter()
                while not stopped.is_set():
                    await asyncio.sleep(0.001)
                    now = time.perf_counter()
                    gaps.append((now - previous) * 1000)
                    previous = now

            pulse = asyncio.create_task(heartbeat())
            await asyncio.sleep(0)
            started = time.perf_counter()
            try:
                result = await operation()
                if name == "html_import_with_title":
                    assert (
                        controller.read_session_canvas(
                            live["scope"], result.canvas_id, temporary=True
                        ).source
                        == source
                    )
                else:
                    result.source_identity.verify_source(source)
            finally:
                walls.append((time.perf_counter() - started) * 1000)
                stopped.set()
                await pulse
        assert len(gaps) > 5
        evidence[name] = {
            "samples": 5,
            "wall_median_ms": round(statistics.median(walls), 3),
            "wall_max_ms": round(max(walls), 3),
            "loop_gap_max_ms": round(max(gaps), 3),
        }
    print("CANVAS_COMPILER_SCHEDULING " + json.dumps(evidence, sort_keys=True))
