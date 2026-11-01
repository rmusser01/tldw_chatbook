"""Thread-affinity controls for the native nested project callback router.

The live first-Send test exercises its connected callers; these extract the actual
nested body to exercise thread/default/error branches without tracing a live run.
"""

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App


def _native_router(owner):
    from tldw_chatbook.Chat import console_chat_controller

    path = Path(console_chat_controller.__file__)
    matches = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "on_owning_loop"
    ]
    assert len(matches) == 1
    namespace = {"asyncio": asyncio, "self": owner}
    exec(  # noqa: S102 - execute only the parsed native routing function, without a live trace.
        compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["on_owning_loop"]


@pytest.mark.asyncio
async def test_project_callback_real_owner_loop_worker_and_exception():
    app = App()
    async with app.run_test():
        owner = SimpleNamespace(app=app, _owner_loop=asyncio.get_running_loop())
        route = _native_router(owner)

        def thread_id():
            return threading.get_ident()

        assert route(thread_id) == app._thread_id
        assert await asyncio.to_thread(route, thread_id) == app._thread_id
        error = RuntimeError("native callback exception")

        def failed():
            raise error

        with pytest.raises(RuntimeError) as raised:
            route(failed)
        assert raised.value is error
        with pytest.raises(RuntimeError) as raised:
            await asyncio.to_thread(route, failed)
        assert raised.value is error
        # A controller with no owner loop preserves the existing App route.
        owner._owner_loop = None
        assert await asyncio.to_thread(route, thread_id) == app._thread_id
        with pytest.raises(RuntimeError, match="different thread"):
            route(thread_id)
        # A skeletal owner without an App keeps its original direct fallback.
        owner.app = None
        assert await asyncio.to_thread(route, thread_id) != app._thread_id
        assert route(thread_id) == app._thread_id
