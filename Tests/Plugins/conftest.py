"""Real local packages, authored independently from the parser."""

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def native_package(tmp_path):
    count = 0

    def create(*, requires=None, extension=None):
        nonlocal count
        count += 1
        root = tmp_path / f"package-{count}"
        shutil.copytree(Path(__file__).parent / "fixtures" / "native", root)
        manifest = json.loads((root / "plugin.json").read_text())
        ext = {"version": 1} if extension is None else extension
        if requires is not None:
            ext["requires"] = requires
        manifest["extensions"]["io.github.rmusser01.chatbook"] = ext
        (root / "plugin.json").write_text(json.dumps(manifest))
        return root

    return create


class PluginStack:
    """Exercise the real stack on its one persistent storage-worker event loop."""

    def __init__(self, root):
        import asyncio
        import threading
        from concurrent.futures import Future

        ready = Future()

        def worker():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                from tldw_chatbook.Plugins.authority_store import (
                    FilePluginMarkerStore,
                    PluginAuthorityStore,
                )
                from tldw_chatbook.Plugins.coordinator import PluginCoordinator
                from tldw_chatbook.Plugins.registry import PluginRegistry
                from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

                self.owner = PluginRuntimeOwner(root / "plugins")
                assert self.owner.try_acquire()
                self.registry = PluginRegistry(
                    self.owner.root / "registry.sqlite3", owner=self.owner
                )
                self.authority = PluginAuthorityStore(
                    root / "trust" / "plugins",
                    FilePluginMarkerStore(root / "marker"),
                    accept_reduced_protection=True,
                )
                self.coordinator = PluginCoordinator(
                    self.registry, self.authority, self.owner
                )
                ready.set_result(True)
                self.loop.run_forever()
            except BaseException as error:
                if not ready.done():
                    ready.set_exception(error)
                else:
                    raise
            finally:
                if hasattr(self, "registry"):
                    self.registry.close()
                if hasattr(self, "owner"):
                    self.owner.close()
                self.loop.close()

        self.thread = threading.Thread(target=worker, name="plugin-storage-test")
        self.thread.start()
        ready.result(timeout=15)
        self.call(lambda: self.coordinator.bootstrap("test passphrase"))

    def call(self, callback):
        import asyncio
        import inspect

        async def invoke():
            result = callback()
            return await result if inspect.isawaitable(result) else result

        return asyncio.run_coroutine_threadsafe(invoke(), self.loop).result(timeout=30)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=15)
        assert not self.thread.is_alive()


@pytest.fixture
def plugin_stack(tmp_path):
    stack = PluginStack(tmp_path / "stack")
    try:
        yield stack
    finally:
        stack.close()
