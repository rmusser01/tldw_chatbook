"""One fresh Runtime owned by an unmounted pending-Close test app."""

import asyncio
import threading


class PreparedCloseRuntimeOwner:
    """Retain exact constructor ownership without adopting database resources."""

    def __init__(self, app):
        from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

        self.app = app
        self.creator = threading.current_thread()
        self.loop = asyncio.get_running_loop()
        self.runtime = getattr(app, "console_runtime", None)
        self.runtime_type = ConsoleRuntime
        if (
            type(self.runtime) is not ConsoleRuntime
            or self.runtime.app is not app
            or self.runtime._disposed
            or self.runtime.chat_store is not None
            or self.runtime.chat_controller is not None
            or self.runtime.provider_gateway is not None
            or self.runtime._agent_runs_db is not None
            or self.runtime.view is not None
        ):
            raise RuntimeError("prepared_close_runtime_not_fresh_owned")
        watcher = self.runtime._canvas_policy_watch_task
        if (
            type(watcher) is not asyncio.Task
            or watcher.get_loop() is not self.loop
            or watcher.done()
        ):
            raise RuntimeError("prepared_close_runtime_watcher_not_owned")
        self.watcher = watcher
        self.dispose_task = None
        self.runtime_terminal = False

    async def dispose_runtime(self):
        """Finish the one original disposal despite repeated awaiter cancellation."""
        if threading.current_thread() is not self.creator:
            raise RuntimeError("prepared_close_runtime_wrong_thread")
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("prepared_close_runtime_wrong_loop")
        if (
            getattr(self.app, "console_runtime", None) is not self.runtime
            or type(self.runtime) is not self.runtime_type
            or self.runtime.app is not self.app
        ):
            raise RuntimeError("prepared_close_runtime_owner_changed")
        if self.dispose_task is None:
            # Invoke the stock class API on the captured instance. An instance
            # field cannot redirect this original owner to a different callable.
            self.dispose_task = self.loop.create_task(
                self.runtime_type.dispose(self.runtime),
                name="prepared-close-non-chat-runtime-dispose",
            )
        cancelled = False
        try:
            while True:
                try:
                    await asyncio.shield(self.dispose_task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if self.dispose_task.done():
                        self.dispose_task.result()
                        break
                    # A second cancellation is another awaiter interruption;
                    # the exact original disposal still owns its original grace.
        finally:
            self.runtime_terminal = (
                self.dispose_task.done()
                and not self.dispose_task.cancelled()
                and self.dispose_task.exception() is None
                and self.runtime._disposed
                and self.watcher.done()
                and self.runtime._canvas_policy_watch_task is None
                and self.runtime._canvas_policy_read_task is None
            )
        if not self.runtime_terminal:
            raise RuntimeError("prepared_close_runtime_not_retired")
        if cancelled:
            raise asyncio.CancelledError
