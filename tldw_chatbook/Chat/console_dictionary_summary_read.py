"""Observe the original Inspector summary workers without granting IO authority."""

from __future__ import annotations

import asyncio
from contextvars import copy_context
import inspect
import os
import sys
from types import FunctionType

from tldw_chatbook.Backup_Recovery import dictionary_source_job as jobs
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat import world_info_resolver as world
from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

from .console_domain_services import _definition, _definition_current
from .console_execution_retirement import _ReadResources
from .console_preparation_reads import preparation_read_for


class DictionarySummaryRead:
    """One retained outer handoff and its two actual creator observations.

    Create this before submitting the outer preparation read. Keep it even when
    read() raises; dictionary_summary_retired() checks cleanup independently.
    """

    def __init__(self, owner: object) -> None:
        self.owner = owner
        self._registration = (self,)
        self.pid = os.getpid()
        self.handoff = self.outer_producer = self.producer = None
        self.creator_observation = self.worker_observation = None
        self.service = self.local_service = self.policy = None
        self.database = self.job = self.task = None
        self.payload = None
        self.mode = None
        self.issued = self.finished = self.succeeded = False
        self.worker_returned = self.producer_joined = False
        self.cleanup_error = None
        self._native_result = None

    def _begin(self, handoff, mode):
        if self.issued:
            raise RuntimeError("Summary receipt already issued")
        if (
            self._registration[0] is not self
            or self.pid != os.getpid()
            or _sources_current is not _CURRENT[0]
            or not _definition_current(_CURRENT)
            or not _sources_current()
            or handoff is None
            or preparation_read_for(self.owner) is not handoff
            or handoff.creator is not self.owner
            or handoff._producer is None
        ):
            raise RuntimeError("Summary preparation handoff changed")
        self.issued = True
        self.handoff, self.outer_producer, self.mode = handoff, handoff._producer, mode

    def read(self, service, conversation_id, character_id, *, handoff):
        """Run the original local dictionary scope on its original private loop."""
        if (
            type(service) is not ChatDictionaryScopeService
            or type(service.local_service) is not LocalChatDictionaryService
            or type(service.local_service.db) is not CharactersRAGDB
            or service.local_service.db.is_memory_db
        ):
            raise RuntimeError("Original file-backed dictionary summary is required")
        self._begin(handoff, "dictionary")
        self.service, self.database = service, service.local_service.db
        self.local_service, self.policy = service.local_service, service.policy_enforcer
        self.payload = ((conversation_id, character_id), {})
        return self._run()

    def read_world_books(self, database, conversation_id, *, handoff):
        """Run original run_owned_db_call and original world-book summary."""
        if type(database) is not CharactersRAGDB or database.is_memory_db:
            raise RuntimeError("Original file-backed world-book summary is required")
        self._begin(handoff, "world")
        self.database = database
        self.payload = ((database, conversation_id, None), {})
        return self._run()

    def _run(self):
        async def original():
            self.task = asyncio.current_task()
            observation = self.creator_observation = _ReadResources(None, self.handoff)
            binding = (
                jobs._CONSOLE_DICTIONARY_SUMMARY
                if self.mode == "dictionary"
                else base_db._CONSOLE_WORLD_BOOK_READ
            )
            # The outer callback has no task. Observe only after asyncio.run
            # entered this exact Task, where policy and job construction occur.
            with observation.observe():
                token = binding.set((self.task, self))
                try:
                    self._require_inner()
                    if self.mode == "dictionary":
                        value = await self.service.summarize_active_dictionaries(
                            *self.payload[0], mode="local"
                        )
                    else:
                        value = await base_db.run_owned_db_call(
                            self.database,
                            world.summarize_active_world_books,
                            *self.payload[0],
                        )
                    self._require_inner()
                    self.succeeded = True
                    return value
                finally:
                    binding.reset(token)
                    prior_error = sys.exc_info()[1]
                    try:
                        await self._join_producer()
                    except BaseException as error:
                        if not self.producer_joined:
                            self.cleanup_error = error
                        if prior_error is None:
                            raise

        try:
            return asyncio.run(original())
        finally:
            self.finished = True

    def _require_inner(self):
        if (
            self._registration[0] is not self
            or self.pid != os.getpid()
            or asyncio.current_task() is not self.task
            or _sources_current is not _CURRENT[0]
            or not _definition_current(_CURRENT)
            or not _sources_current()
            or any(
                inspect.getattr_static(self, name) is not raw
                for container, name, raw, _record in _METHODS
                if container is DictionarySummaryRead
            )
        ):
            raise RuntimeError("Summary observation source changed")
        if self.mode == "dictionary":
            if (
                self.service.local_service is not self.local_service
                or self.service.policy_enforcer is not self.policy
                or self.local_service.db is not self.database
                or any(
                    inspect.getattr_static(self.service, name) is not raw
                    for container, name, raw, _record in _METHODS
                    if container is ChatDictionaryScopeService
                )
                or any(
                    inspect.getattr_static(self.service.local_service, name) is not raw
                    for container, name, raw, _record in _METHODS
                    if container is LocalChatDictionaryService
                )
            ):
                raise RuntimeError("Summary dictionary receiver changed")

    async def _join_producer(self):
        producer = self.producer
        if producer is None:
            return
        cancelled = None
        while not producer.done():
            try:
                await asyncio.shield(producer)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                break
        # A cancelled future alone cannot prove an entered native callback's
        # return. Keep that outcome negative, independent of optional results.
        if producer.cancelled():
            raise asyncio.CancelledError
        producer.exception()  # Preserve the body outcome independently of joining.
        self.producer_joined = True
        if cancelled is not None:
            raise cancelled

    def _dictionary_work(self, payload):
        observation = self.worker_observation = _ReadResources(None, self.handoff)
        try:
            with observation.observe():
                if not _sources_current() or payload is not self.payload:
                    raise RuntimeError("Dictionary summary callback changed")
                _WORK[0](self.job, payload)
        finally:
            self.worker_returned = True


def submit_dictionary_summary(binding, job, payload):
    """Select only the retained original summary job's existing executor call."""
    task, owner = binding
    if type(owner) is not DictionarySummaryRead or task is not owner.task:
        raise RuntimeError("Dictionary summary task changed")
    owner._require_inner()
    if (
        owner.mode != "dictionary"
        or owner.job is not None
        or type(job) is not jobs._DictionaryJob
        or job not in jobs._jobs
        or job._source is not owner.service.local_service
        or job._db is not owner.database
        or job._function is not _LOCAL_SUMMARY[0]
        or job._method != "summarize_active_dictionaries"
        or job._task is not task
        or job._loop is not asyncio.get_running_loop()
        or payload != owner.payload
        or any(
            inspect.getattr_static(job, name) is not raw
            for container, name, raw, _record in _METHODS
            if container is jobs._DictionaryJob
        )
    ):
        raise RuntimeError("Dictionary summary job changed")
    owner.job, owner.payload = job, payload
    owner.producer = job._loop.run_in_executor(None, owner._dictionary_work, payload)
    return owner.producer


async def observe_world_book_call(binding, database, operation, invoke, args, kwargs):
    """Observe the unchanged original base-db invoke on its actual worker."""
    task, owner = binding
    if type(owner) is not DictionarySummaryRead or task is not owner.task:
        raise RuntimeError("World-book summary task changed")
    owner._require_inner()
    if (
        owner.mode != "world"
        or owner.producer is not None
        or database is not owner.database
        or operation is not _WORLD_SUMMARY[0]
        or args != owner.payload[0]
        or kwargs != {}
        or type(invoke) is not FunctionType
        or invoke.__code__ is not _DB_INVOKE
        or invoke.__globals__ is not vars(base_db)
        or inspect.getclosurevars(invoke).nonlocals
        != {
            "database": database,
            "operation": operation,
            "args": args,
            "kwargs": kwargs,
        }
    ):
        raise RuntimeError("World-book summary invocation changed")

    def original():
        observation = owner.worker_observation = _ReadResources(None, owner.handoff)
        try:
            with observation.observe():
                if not _sources_current():
                    raise RuntimeError("World-book summary source changed")
                return invoke()
        finally:
            owner.worker_returned = True

    # Same context propagation and executor as asyncio.to_thread, with the
    # actual Future retained so observer cancellation cannot detach the work.
    context = copy_context()
    owner.producer = asyncio.get_running_loop().run_in_executor(
        None, context.run, original
    )
    await owner._join_producer()
    return owner.producer.result()


def dictionary_summary_retired(owner: object) -> bool:
    """Check original cleanup only after the enclosing real handoff retired."""
    if (
        type(owner) is not DictionarySummaryRead
        or owner._registration[0] is not owner
        or owner.pid != os.getpid()
        or _sources_current is not _CURRENT[0]
        or not _definition_current(_CURRENT)
        or not _sources_current()
        or not owner.issued
        or not owner.finished
        or owner.handoff is None
        or not owner.handoff.retired.done()
        or owner.handoff.retired.cancelled()
        or owner.handoff.creator is not owner.owner
        or owner.handoff._producer is not owner.outer_producer
        or owner.outer_producer is None
        or not owner.outer_producer.done()
        or owner.outer_producer.cancelled()
    ):
        return False
    if owner._native_result is None:
        creator = owner.creator_observation
        positive = (
            type(creator) is _ReadResources
            and creator.read is owner.handoff
            and creator.entered
            and creator.closed
            and creator.confirmed()
            and owner.cleanup_error is None
        )
        if owner.producer is not None:
            worker = owner.worker_observation
            positive = (
                positive
                and owner.producer_joined
                and owner.worker_returned
                and owner.producer.done()
                and not owner.producer.cancelled()
                and type(worker) is _ReadResources
                and worker.read is owner.handoff
                and worker.entered
                and worker.closed
                and worker.confirmed()
            )
        else:
            positive = positive and owner.job is None and not owner.succeeded
        if owner.job is not None:
            positive = (
                positive and owner.job in jobs._jobs and owner.job._state == "closed"
            )
        owner._native_result = bool(positive)
    return owner._native_result


def _records(container, names):
    for name in names:
        raw = inspect.getattr_static(container, name)
        function = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        yield container, name, raw, _definition(function)


_METHODS = (
    tuple(
        _records(
            ChatDictionaryScopeService,
            (
                "summarize_active_dictionaries",
                "_invoke",
                "_call_backend",
                "_enforce_policy",
                "_normalize_mode",
                "_backend",
                "_statistics_action",
                "_is_file_backed_local",
            ),
        )
    )
    + tuple(
        _records(
            LocalChatDictionaryService, ("summarize_active_dictionaries", "_require_db")
        )
    )
    + tuple(
        _records(
            jobs._DictionaryJob,
            (
                "__init__",
                "__enter__",
                "__exit__",
                "_check_creator",
                "run",
                "_work",
                "_complete",
                "_executor_finished",
            ),
        )
    )
    + tuple(
        _records(
            DictionarySummaryRead,
            (
                "__init__",
                "_begin",
                "read",
                "read_world_books",
                "_run",
                "_require_inner",
                "_join_producer",
                "_dictionary_work",
            ),
        )
    )
)
_FUNCTIONS = (
    tuple(_records(jobs, ("supports",)))
    + tuple(_records(base_db, ("run_owned_db_call",)))
    + tuple(_records(world, ("summarize_active_world_books",)))
)
_WORK = _definition(jobs._DictionaryJob._work)
_LOCAL_SUMMARY = _definition(LocalChatDictionaryService.summarize_active_dictionaries)
_WORLD_SUMMARY = _definition(world.summarize_active_world_books)
_DB_INVOKE = next(
    code
    for code in base_db.run_owned_db_call.__code__.co_consts
    if inspect.iscode(code) and code.co_name == "invoke"
)
_DICTIONARY_SUBMIT = _definition(submit_dictionary_summary)
_WORLD_INVOKE = _definition(observe_world_book_call)
_RETIRE = _definition(dictionary_summary_retired)


def _sources_current():
    return all(
        inspect.getattr_static(container, name) is raw and _definition_current(record)
        for container, name, raw, record in (*_METHODS, *_FUNCTIONS)
    ) and all(
        _definition_current(record)
        for record in (_DICTIONARY_SUBMIT, _WORLD_INVOKE, _RETIRE)
    )


_CURRENT = _definition(_sources_current)
