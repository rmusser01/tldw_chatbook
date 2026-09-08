"""Two source-owned queued jobs; pending registration grants no IO authority."""

import asyncio
from dataclasses import dataclass
import os
import sys
import threading
import weakref

from . import bootstrap, raw_participants as raw, storage_admission as storage


_jobs = weakref.WeakSet()


@dataclass(frozen=True)
class _Outcome:
    value: object = None
    error: BaseException | None = None
    cancelled: bool = False


class _FileJob:
    """One-shot source dispatch. Native admission is acquired only by its worker.

    The creator keeps this context through event-loop result bookkeeping. Cancelling
    a queued job atomically disables source entry; cancelling a running job waits
    for the actual wrapper, including repeated cancellation, without revoking IO.
    """

    def __init__(self, source, route):
        self._attempt = storage._Acquisition()
        try:
            self.selected, installed = raw._async_source_selection(source, route)
            self._attempt.check()
            if installed and raw._pinned_io_available():
                participant = raw._raw_participant(source)
                if raw._participant_state(participant).closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            self._source = source
            self._route = route
            self._pid = os.getpid()
            self._thread = threading.current_thread()
            self._task = storage._task_identity()
            self._lock = threading.Lock()
            self._state = "new"
            _jobs.add(self)
        except BaseException:
            self._attempt.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self._check_creator()
        with self._lock:
            if self._state in {"queued", "running"}:
                raise RuntimeError("source_job_not_retired")
            self._state = "closed"
        self._attempt.close()

    def _check_creator(self):
        if (
            type(self) is not _FileJob
            or self not in _jobs
            or self._pid != os.getpid()
            or self._thread is not threading.current_thread()
            or self._task is not storage._task_identity()
        ):
            raise bootstrap.RecoveryRequired("source_job_provenance_invalid")

    def _complete(self, outcome):
        if not self._completion.done():
            self._completion.set_result(outcome)

    def _executor_finished(self, future):
        """Executor cancellation is not proof that a running callback stopped."""
        if not future.cancelled():
            error = future.exception()
            if error is None:
                return
        else:
            error = None
        with self._lock:
            self._executor_cancelled = future.cancelled()
            if self._state != "queued":
                return
            # Atomically prevent source entry if the executor cancelled before
            # callback entry (including a dequeued callback waiting on this lock).
            self._state = "cancelled" if future.cancelled() else "finished"
        self._complete(_Outcome(error=error, cancelled=future.cancelled()))

    def _work(self, payload):
        with self._lock:
            if self._state == "cancelled":
                return
            if self._state != "queued" or self._pid != os.getpid():
                return
            self._state = "running"
        try:
            # Unbound class members, never caller-provided callbacks/metadata.
            if self._route == "prompt_history":
                from ..Chat.prompt_history import PromptHistory

                value = PromptHistory._history_io(self._source, self.selected, payload)
            else:
                cls = sys.modules["tldw_chatbook.UI.Screens.chat_screen"].ChatScreen
                value = cls._write_sidebar_state_snapshot(
                    self._source, payload, _selected=self.selected
                )
            outcome = _Outcome(value=value)
        except BaseException as error:
            outcome = _Outcome(error=error)
        # Source IO, publication/cleanup and the native retirement attempt returned.
        # Uncertain native resources remain retained by the raw source registry.
        # This actual callback signal is independent of the executor wrapper's
        # cancellation state. The worker never waits for an event-loop response.
        with self._lock:
            self._state = "finished"
        self._loop.call_soon_threadsafe(self._complete, outcome)

    async def run(self, payload):
        self._check_creator()
        self._attempt.check()
        with self._lock:
            if self._state != "new":
                raise bootstrap.RecoveryRequired("source_job_provenance_invalid")
            self._state = "queued"
            self._executor_cancelled = False
        self._loop = asyncio.get_running_loop()
        self._completion = self._loop.create_future()
        try:
            future = self._loop.run_in_executor(None, self._work, payload)
            future.add_done_callback(self._executor_finished)
        except BaseException:
            with self._lock:
                self._state = "cancelled"
            raise
        cancelled = False
        while True:
            try:
                outcome = await asyncio.shield(self._completion)
            except asyncio.CancelledError:
                cancelled = True
                with self._lock:
                    if self._state == "queued":
                        self._state = "cancelled"
                        # The wrapper cannot enter source code, even if dequeued.
                        future.cancel()
                        self._completion.cancel()
                        raise
            else:
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return _Outcome(
                    outcome.value, outcome.error, cancelled or self._executor_cancelled
                )
