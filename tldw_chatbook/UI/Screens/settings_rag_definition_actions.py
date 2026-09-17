"""Retain the accepted Settings RAG action through native and UI completion."""

import asyncio
import threading

from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant


class QueuedDefinitionAction:
    """One queued/running action; Textual cancellation retires only queued work."""

    def __init__(self, *, parent=None, cohort=participant, on_cancel=None):
        self._token = cohort.reserve(parent=None if parent is None else parent._token)
        self._lock = threading.Lock()
        self._state = "queued"
        self._on_cancel = on_cancel

    def cancel_queued(self, *, deliver=False):
        with self._lock:
            if self._state != "queued":
                return
            self._state = "done"
        try:
            if deliver and self._on_cancel is not None:
                with self._token.scope():
                    self._on_cancel()
        finally:
            self._token.close()

    def attach(self, worker):
        """Observe the installed Textual task, not its reported native lifetime.

        Worker._task is the actual asyncio waiter created synchronously by
        @work. Its completion can prove only that our still-queued callable
        must not start. execute() arbitrates a racing executor entry under the
        same lock; already-running native code owns retirement independently.
        """
        task = getattr(worker, "_task", None)
        if not isinstance(task, asyncio.Task):
            self.cancel_queued()
            raise TypeError("rag_definition_worker_not_started")
        task.add_done_callback(lambda _task: self.cancel_queued(deliver=True))
        return worker

    def enqueue(self, dispatch, *args):
        """Retire rejected enqueue before the caller changes its UI state."""
        try:
            return self.attach(dispatch(*args, _definition_action=self))
        except BaseException:
            self.cancel_queued()
            raise

    def execute(self, function, *args):
        with self._lock:
            if self._state == "done":
                return None
            if self._state != "queued":
                raise RuntimeError("rag_definition_action_already_running")
            self._state = "running"
        try:
            with self._token.scope():
                return function(*args)
        finally:
            with self._lock:
                self._state = "done"
            self._token.close()

    def deliver(self, app, callback, *args):
        """Transfer explicit acceptance to synchronous UI delivery/descendants."""

        def accepted():
            with self._token.scope():
                return callback(*args)

        return app.call_from_thread(accepted)
