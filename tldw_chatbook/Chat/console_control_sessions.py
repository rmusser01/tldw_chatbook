"""Selected saved-session opening through the original conversation owners."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager

from . import console_conversation_hydration as hydration
from . import conversation_archive_actions as archive
from .chat_conversation_scope_service import ChatConversationScopeService
from .chat_conversation_service import ChatConversationService
from .console_domain_services import (
    ConsoleDomainServices,
    _definition,
    _definition_current,
)


_saved_open = contextvars.ContextVar("console_saved_open_owner", default=None)
_saved_read = contextvars.ContextVar("console_saved_read_binding", default=None)


class _SavedOpenOwner:
    def __init__(self, controls, sources):
        self.controls, self.sources = controls, sources
        self.reads = []
        self.workers = []
        self.projection_retired = True

    def current(self):
        require_saved_open_sources(
            self.controls._domain,
            self.sources,
            store=self.controls._store,
            database=self.sources.chat_database,
        )

    def require_archive_reader(self, service, method, call):
        self.current()
        scope = self.controls._domain.conversation_scope_service()
        if (
            service is not scope.local_service
            or method != "get_conversation_archive_states"
            or getattr(call, "__self__", None) is not service
            or getattr(call, "__func__", None)
            is not ChatConversationService.get_conversation_archive_states
        ):
            raise RuntimeError("Console archive reader changed")

    def invoke(self, callback):
        from .console_native_commit import (
            _ConsoleNativeBinding,
            _ConsolePlanNativeRetirement,
            _invoke_plan_native,
        )

        self.current()
        store = self.controls._store
        binding = _ConsoleNativeBinding(
            store,
            self,
            store.persistence,
            self.sources.chat_database,
            retirement=_ConsolePlanNativeRetirement(),
        )
        self.reads.append(binding.retirement)
        token = _saved_read.set(binding)
        try:
            return _invoke_plan_native(binding, callback)
        finally:
            _saved_read.reset(token)

    async def run(self, callback, *, threaded=True, observe=True):
        self.current()
        if self.sources.chat_database.is_memory_db:
            threaded = False
        run = (lambda: self.invoke(callback)) if observe else callback
        if not threaded:
            return run()
        coroutine = asyncio.to_thread(run)
        try:
            task = asyncio.create_task(coroutine)
        except BaseException:
            coroutine.close()
            raise
        self.workers.append(task)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        if cancelled:
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        result = task.result()
        self.current()
        return result

    @property
    def retired(self):
        return (
            all(task.done() and not task.cancelled() for task in self.workers)
            and all(read.confirmed for read in self.reads)
            and self.projection_retired
        )


def capture_saved_read_close(database):
    """Retain the actual handle before the original reader closes it."""
    binding = _saved_read.get()
    if binding is None:
        return
    if binding.database is not database:
        raise RuntimeError("Saved-open read database changed")
    from .console_native_commit import _capture_plan_native_connection

    _capture_plan_native_connection(binding)


def saved_projection_owner(persistence):
    owner = _saved_open.get()
    if owner is None:
        return None
    owner.current()
    if (
        owner.controls._store.persistence is not persistence
        or persistence.db is not owner.sources.chat_database
        or persistence.workspace_registry is not owner.sources.workspace_registry
    ):
        raise RuntimeError("Saved-open projection owner changed")
    return owner


@contextmanager
def saved_projection_registry(owner, registry):
    if owner is None:
        yield
        return
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )

    if registry is not owner.sources.workspace_registry:
        raise RuntimeError("Saved-open projection registry changed")
    owner.projection_retired = False
    error = None
    with project_index_registry_connections(registry.db):
        try:
            yield
        except BaseException as caught:
            error = caught
    # Only the original registry scope's actual native close proves retirement.
    owner.projection_retired = True
    if error is not None:
        raise error


def require_conversation_definitions(scope, local):
    for owner, name, record in _CONVERSATION_DEFINITIONS:
        receiver = scope if owner is ChatConversationScopeService else local
        if (
            getattr(owner, name) is not record[0]
            or not _definition_current(record)
            or name in vars(receiver)
        ):
            raise RuntimeError("Original conversation reader changed")
    for name, record in _HYDRATION_DEFINITIONS:
        if getattr(hydration, name) is not record[0] or not _definition_current(record):
            raise RuntimeError("Original conversation hydration changed")


def require_saved_open_sources(domain, sources, *, store, database):
    if (
        type(domain) is not ConsoleDomainServices
        or domain.runtime is None
        or domain.runtime._chat_store is not store
        or sources.chat_database is not database
        or store.persistence.db is not database
    ):
        raise RuntimeError("Saved Console database owner changed")
    domain.require_current(sources)
    domain.conversation_scope_service()


async def open_saved_session(controls, command):
    # The retained control Task is never cancelled by a cancelled receipt; its
    # original off-loop readers finish before this finite operation retires.
    from .console_control_dispatch import _command_source

    domain, store = controls._domain, controls._store
    sources = domain.sources
    database = sources.chat_database
    require_saved_open_sources(domain, sources, store=store, database=database)
    scope = domain.conversation_scope_service()
    effect = controls._begin_effect("saved_open", scope, _command_source.get())
    owner = _SavedOpenOwner(controls, sources)
    token = _saved_open.set(owner)
    try:
        tree = await hydration.load_console_conversation_tree(
            None,
            command.conversation_id,
            _conversation_service=scope,
            _read_owner=owner,
        )
        require_saved_open_sources(domain, sources, store=store, database=database)
        if tree is None:
            raise LookupError("Saved conversation was not found")
        prepared = await hydration.prepare_console_session_data(
            app=None,
            store=store,
            conversation_id=command.conversation_id,
            tree=tree,
            _database=database,
            _read_owner=owner,
        )
        require_saved_open_sources(domain, sources, store=store, database=database)
        generation = hydration.hydrate_console_generation_settings(
            sources.configuration, tree["conversation"]
        )
        session = await hydration.hydrate_console_session(
            app=None,
            store=store,
            conversation_id=command.conversation_id,
            tree=tree,
            settings=generation.settings,
            generation_durable_snapshot=generation.durable_snapshot,
            generation_metadata_status=generation.metadata_status,
            target_workspace_id=command.workspace_id,
            prepared_data=prepared,
            _database=database,
            _domain_sources=(domain, sources),
            _read_owner=owner,
        )
        require_saved_open_sources(domain, sources, store=store, database=database)
        return session.id, session.incarnation_id
    finally:
        _saved_open.reset(token)
        controls._finish_effect(effect, retired=owner.retired)


async def selected_conversation_send_refusal(runtime, record, require_received):
    """Retain archive DB work while preserving the real parent's coordination."""
    from .console_control_dispatch import require_control_source

    source = require_control_source(runtime)
    controls = source.controls
    require_received()
    domain = controls._domain
    scope = domain.conversation_scope_service()
    for name, definition in _ARCHIVE_DEFINITIONS:
        if getattr(archive, name) is not definition[0] or not _definition_current(
            definition
        ):
            raise RuntimeError("Original Console archive admission changed")
    owner = _SavedOpenOwner(controls, source.sources)
    owner.current()
    effect = controls._begin_effect("archive_read", scope.local_service, source)
    parent_checkpoint = None
    try:
        conversation_id = record.archive_conversation_id
        if not conversation_id:
            return None
        port = domain.presentation
        from .console_backend_presentation import ConsoleBackendPresentation
        if type(port) is ConsoleBackendPresentation:
            from . import console_backend_archive_send as backend_archive

            for name, definition in _BACKEND_ARCHIVE_DEFINITIONS:
                if (getattr(backend_archive, name) is not definition[0]
                    or not _definition_current(definition)):
                    raise RuntimeError("Console archive checkpoint source changed")
            parent_checkpoint, refusal = await backend_archive.prepare_archive_send(port, record, read_owner=owner)
            owner.current()
            require_received()
        else:
            refusal = port.archive_send_inflight(record)
        if refusal is not None:
            return refusal
        service = scope.local_service
        owner.require_archive_reader(
            service,
            "get_conversation_archive_states",
            service.get_conversation_archive_states,
        )
        checkpoint = parent_checkpoint if parent_checkpoint is not None else port.capture_archive_send(record)
        states = await archive.storage_call(
            service,
            "get_conversation_archive_states",
            [conversation_id],
            _read_owner=owner,
        )
        owner.require_archive_reader(
            service,
            "get_conversation_archive_states",
            service.get_conversation_archive_states,
        )
        if not owner.retired:
            raise RuntimeError("Console archive read has not physically retired")
        if parent_checkpoint is not None:
            for name, definition in _BACKEND_ARCHIVE_DEFINITIONS:
                if (getattr(backend_archive, name) is not definition[0]
                    or not _definition_current(definition)):
                    raise RuntimeError("Console archive checkpoint source changed")
            result = await backend_archive.finish_archive_send(checkpoint, record, states)
        else:
            result = port.finish_archive_send(checkpoint, record, states)
        owner.current()
        require_received()
        return result
    finally:
        if parent_checkpoint is not None and record.archive_checkpoint is parent_checkpoint:
            record.archive_checkpoint = None
        controls._finish_effect(effect, retired=owner.retired)


_CONVERSATION_DEFINITIONS = tuple(
    (owner, name, _definition(getattr(owner, name)))
    for owner, names in (
        (
            ChatConversationScopeService,
            (
                "get_conversation_tree",
                "_normalize_mode",
                "_enforce_policy",
                "_action_id",
                "_service_for_mode",
                "_is_memory_backed",
                "_maybe_await",
            ),
        ),
        (
            ChatConversationService,
            ("get_conversation_tree", "get_conversation_archive_states"),
        ),
    )
    for name in names
)
_HYDRATION_DEFINITIONS = tuple(
    (name, _definition(getattr(hydration, name)))
    for name in (
        "load_console_conversation_tree",
        "prepare_console_session_data",
        "hydrate_console_session",
        "hydrate_console_generation_settings",
    )
)
_ARCHIVE_DEFINITIONS = tuple(
    (name, _definition(getattr(archive, name)))
    for name in (
        "conversation_send_refusal",
        "storage_call",
        "archive_send_inflight_refusal",
        "publish_archive_send_state",
    )
)

from . import console_backend_archive_send as _backend_archive

_BACKEND_ARCHIVE_DEFINITIONS = tuple(
    (name, _definition(getattr(_backend_archive, name)))
    for name in ("prepare_archive_send", "finish_archive_send")
)

_DOMAIN_SOURCE_DEFINITIONS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "require_conversation_definitions",
        "require_saved_open_sources",
        "open_saved_session",
        "selected_conversation_send_refusal",
    )
)
