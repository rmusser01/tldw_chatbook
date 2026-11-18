"""Bounded pre-mount setup on the same contained Console backend."""

from __future__ import annotations

from concurrent.futures import Future
import threading
import time

from tldw_chatbook.MCP.permission_process import read_message

from .console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure, _ConsoleFrameInput
from .console_execution_protocol import decode_message


# Preserve the original Windows journey's setup allowance; its five-second
# target starts at Send, and the separate shutdown allowance is only 30 seconds.
BOOTSTRAP_TIMEOUT_SECONDS = 300.0


class ConsolePermissionBootstrap:
    """Own direct framed setup until the UI adopts the settled transport."""

    def __init__(self, client):
        self.client = client
        self.thread = None
        self.result = None
        self.receipt = None
        self.aborted = threading.Event()
        self.lock = threading.Lock()
        self._containment_closed = False
        self._adopting = False

    def invoke(self, profile, operation) -> None:
        client = self.client
        if not self.lock.acquire(blocking=False):
            raise ConsoleChannelLost("Console pre-mount setup is already running")
        try:
            if (client._permission_bootstrap is not self or client._loop is not None
                or client._start_task is not None or client._shutdown_task is not None
                or client._channel_error is not None or client._view is not None
                or self._adopting
                or self.aborted.is_set()
                or (self.thread is not None and self.thread.is_alive())):
                raise ConsoleChannelLost("Console synchronous setup is no longer available")
            app = client._app
            if app is not None and (
                getattr(app, "console_execution_client", None) is not client
                or getattr(app, "_console_execution_start_task", None) is not None
                or getattr(app, "is_running", False)
            ):
                raise ConsoleChannelLost("Console synchronous setup requires pre-mount ownership")
            if client._profile is not None and client._profile != profile:
                raise ConsoleChannelLost("Console profile changed within its lifetime")
            client._profile = profile
            if client._ready is None:
                client._ready = Future()
            self.result = Future()
            self.receipt = None
            self.thread = threading.Thread(
                target=self._run, args=(operation,), name="console-permission-bootstrap",
                daemon=True,
            )
            self.thread.start()
            self.thread.join(BOOTSTRAP_TIMEOUT_SECONDS)
            if self.thread.is_alive():
                self.aborted.set()
                error = ConsoleChannelLost("Console pre-mount setup timed out; outcome is unknown")
                client._fail(error)
                # Stop the actual tree to release a blocked framed read. A late
                # launch remains retained by this same worker and sees aborted.
                self.stop()
                self.thread.join(5.0)
                raise error
            self.result.result()
        finally:
            self.lock.release()

    def _receive(self) -> None:
        client = self.client
        if self.aborted.is_set() or client._channel_error is not None:
            raise ConsoleChannelLost("Console pre-mount channel is unavailable")
        message = decode_message(read_message(_ConsoleFrameInput(client._process.stdout)))
        if self.aborted.is_set():
            raise ConsoleChannelLost("Console pre-mount setup was interrupted")
        client._receive(message)
        if client._channel_error is not None:
            raise ConsoleChannelLost("Console pre-mount receipt failed") from client._channel_error

    def _run(self, operation) -> None:
        client = self.client
        try:
            if client._process is None:
                client._launch(start_channel=False)
            while not client._ready.done():
                self._receive()
            client._ready.result()
            if self.aborted.is_set():
                raise ConsoleChannelLost("Console pre-mount setup was interrupted")
            self.receipt = client._submit_command("PermissionAction", {"operation": operation})
            while not self.receipt.retired.done():
                self._receive()
            if self.receipt.retired.result() is not True:
                raise ConsoleChannelLost("Console pre-mount permission retirement is unproven")
            self.receipt.received.result()
            applied = self.receipt.applied.result()
            if applied["kind"] != "SetGlobalDefault" or applied["result"] is not None:
                raise ConsoleChannelLost("Console pre-mount permission result changed")
            self.result.set_result(None)
        except BaseException as error:
            # A refused original operation with positive physical retirement may
            # leave this same owner usable. Transport uncertainty never does.
            retired_failure = (
                type(error) is ConsoleRemoteFailure and self.receipt is not None
                and self.receipt.retired.done()
                and self.receipt.retired.exception() is None
                and self.receipt.retired.result() is True
                and client._channel_error is None
            )
            try:
                if not retired_failure:
                    client._fail(error)
                    self.stop()
                    self._close_after_exit()
            except BaseException as cleanup_error:
                client._cleanup_errors.append(cleanup_error)
                client._fail(cleanup_error)
            finally:
                self.result.set_exception(error)

    def stop(self) -> None:
        """Retain containment failure separately from operation retirement."""
        client = self.client
        try:
            if client._tree is not None:
                self._containment_closed = client._tree.close() is True
            elif client._process is not None and not client._initialization_attempted:
                self._containment_closed = client._close_uninitialized_process()
        except BaseException as error:
            client._cleanup_errors.append(error)

    def _close_after_exit(self) -> None:
        client = self.client
        if not self._containment_closed or client._process is None or client._process.poll() is None:
            return
        deadline = time.monotonic() + 5.0
        joined = client._channel is None or client._channel.close_after_exit(deadline)
        if client._monitor is not None and client._monitor.ident is not None:
            client._monitor.join(max(0.0, deadline - time.monotonic()))
            joined = joined and not client._monitor.is_alive()
        client._physically_closed = joined and client._close_pipes()
        if client._physically_closed:
            client._release_profile_writer()

    def require_adoptable(self, profile) -> None:
        client = self.client
        if not self.lock.acquire(blocking=False):
            raise ConsoleChannelLost("Console pre-mount setup is still running")
        try:
            if (client._permission_bootstrap is not self or client._profile != profile
                or self.thread is None or self.thread.is_alive() or self.result is None
                or not self.result.done() or self.aborted.is_set()
                or client._channel_error is not None or client._requests
                or client._process is None or client._process.poll() is not None
                or client._channel is None or client._channel._started
                or client._ready is None or not client._ready.done()):
                raise ConsoleChannelLost("Console pre-mount setup cannot be adopted")
            client._ready.result()
            self._adopting = True
        finally:
            self.lock.release()

    def join(self) -> None:
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join()
