"""Actual local dictionary executor job; pending lifetime grants no IO authority."""

import copy
import os
import asyncio
import threading
import weakref

from . import bootstrap, storage_admission as storage
from . import chat_source_participants as chat
from .async_file_participants import _Outcome

_jobs = weakref.WeakSet()
_METHODS = frozenset(
    {
        "list_dictionaries",
        "create_dictionary",
        "get_dictionary",
        "update_dictionary",
        "delete_dictionary",
        "add_entry",
        "list_entries",
        "update_entry",
        "delete_entry",
        "reorder_entries",
        "process_text",
        "import_markdown",
        "export_markdown",
        "import_json",
        "export_json",
        "list_activity",
        "list_versions",
        "get_version",
        "revert_version",
        "get_statistics",
        "attach_to_conversation",
        "detach_from_conversation",
        "list_dictionary_conversations",
        "attach_to_character",
        "detach_from_character",
        "list_character_dictionaries",
        "summarize_active_dictionaries",
    }
)


def supports(source, method):
    from ..Character_Chat.local_chat_dictionary_service import (
        LocalChatDictionaryService,
    )
    from ..DB.ChaChaNotes_DB import CharactersRAGDB

    if (
        type(source) is not LocalChatDictionaryService
        or type(source.db) is not CharactersRAGDB
        or source.db.is_memory_db
    ):
        return False
    valid = (
        getattr(method, "__self__", None) is source
        and getattr(method, "__name__", None) in _METHODS
        and LocalChatDictionaryService.__dict__.get(method.__name__)
        is getattr(method, "__func__", None)
    )
    if not valid and chat.binding(source) is not None:
        raise bootstrap.RecoveryRequired("dictionary_job_source_invalid")
    return valid


class _DictionaryJob:
    """One-shot source dispatch. Native admission is acquired only by its worker.

    The creator keeps this context through event-loop result bookkeeping. Cancelling
    a queued job atomically disables source entry; cancelling a running job waits
    for the actual wrapper, including repeated cancellation, without revoking IO.
    """

    def __init__(self, source, method):
        self._attempt = storage._Acquisition()
        try:
            if not supports(source, method):
                raise bootstrap.RecoveryRequired("dictionary_job_source_invalid")
            self._source = source
            self._db = source.db
            self._method = method.__name__
            self._function = method.__func__
            chat._validate(source)
            self._attempt.check()
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
            type(self) is not _DictionaryJob
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
        acquired = False
        previous_connection = None
        before = None
        close_failed = False
        try:
            from ..Character_Chat.local_chat_dictionary_service import (
                LocalChatDictionaryService,
            )
            from .participants import _repository_participant

            if (
                self._source.db is not self._db
                or LocalChatDictionaryService.__dict__.get(self._method)
                is not self._function
            ):
                raise bootstrap.RecoveryRequired("dictionary_job_source_changed")
            chat._validate(self._source)
            while not self._source._history_lock.acquire(timeout=0.05):
                self._attempt.check()
            acquired = True
            self._attempt.check()
            before = copy.deepcopy(self._source._history)
            previous_connection = getattr(self._db._local, "conn", None)
            try:
                value = self._function(self._source, *payload[0], **payload[1])
            finally:
                # This exact route returns DTOs and owns only a newly opened
                # worker connection. Preexisting native borrowers remain live.
                if previous_connection is None:
                    close_failed = True
                    self._db.close_connection()
                    if getattr(self._db._local, "conn", None) is not None:
                        raise bootstrap.RecoveryRequired(
                            "dictionary_job_native_not_retired"
                        )
                    participant = _repository_participant(self._db)
                    if threading.current_thread() in participant.retiring_threads:
                        raise bootstrap.RecoveryRequired(
                            "dictionary_job_native_not_retired"
                        )
                    close_failed = False
            outcome = _Outcome(value=value)
        except BaseException as error:
            if before is not None and (
                close_failed
                or getattr(self._db._local, "conn", None) is not previous_connection
            ):
                self._source._history = before
                self._source._chat_persistence_error = "chat_publication_incomplete"
            outcome = _Outcome(error=error)
        finally:
            if acquired:
                self._source._history_lock.release()
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
