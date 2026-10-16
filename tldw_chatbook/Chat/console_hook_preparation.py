"""Hook preparation compatibility names and exact-task session attribution."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .console_preparation_reads import (
    ConsolePreparationRead as ConsoleHookPreparationRead,
)
from .console_preparation_reads import (
    _current_task,
)
from .console_preparation_reads import (
    drain_preparation_reads as drain_hook_preparation_reads,
)
from .console_preparation_reads import (
    observe_preparation_reads as observe_hook_preparation_reads,
)
from .console_preparation_reads import (
    preparation_reads_for as hook_preparation_reads_for,
)
from .console_preparation_reads import (
    preparation_source_for as hook_preparation_source_for,
)
from .console_preparation_reads import (
    run_preparation_read as run_hook_preparation_read,
)

__all__ = (
    "ConsoleHookPreparationRead",
    "bind_hook_preparation_session",
    "drain_hook_preparation_reads",
    "hook_preparation_reads_for",
    "hook_preparation_session_for",
    "hook_preparation_source_for",
    "observe_hook_preparation_reads",
    "run_hook_preparation_read",
)


@dataclass(frozen=True, slots=True)
class _HookSessionBinding:
    creator: object
    session_id: str | None
    task: asyncio.Task[Any]


_session_binding: ContextVar[_HookSessionBinding | None] = ContextVar(
    "console_hook_preparation_session", default=None
)


@contextmanager
def bind_hook_preparation_session(
    creator: object, session_id: str | None
) -> Iterator[None]:
    """Attribute a known zero-argument hook call without changing its contract."""
    task = _current_task()
    if task is None:
        raise RuntimeError("Hook preparation attribution requires its owning task.")
    token = _session_binding.set(_HookSessionBinding(creator, session_id, task))
    try:
        yield
    finally:
        _session_binding.reset(token)


def hook_preparation_session_for(creator: object) -> str | None:
    """Read exact Task attribution; copied child/thread contexts cannot borrow it."""
    binding = _session_binding.get()
    if binding is None or binding.task is not _current_task():
        return None
    if binding.creator is not creator:
        raise RuntimeError("Hook preparation owner changed.")
    return binding.session_id


@dataclass(frozen=True, slots=True)
class _HookPreparationSource:
    """Original sources for one finite runtime hook preparation invocation."""

    app: Any = field(repr=False)
    store: Any = field(repr=False)
    controller: Any = field(repr=False)
    session_id: str
    session: Any = field(repr=False)
    session_identity: tuple[Any, ...] = field(repr=False)
    permissions: Any = field(repr=False)
    persistence: Any = field(repr=False)
    chat_database: Any = field(repr=False)
    registry: Any = field(repr=False)
    workspace_database: Any = field(repr=False)
    consent: Any = field(repr=False)
    authority_reader: Any = field(repr=False)
    controller_app: Any = field(repr=False)
    context_provider: Any = field(repr=False)

def capture_hook_preparation_source(self, session_id, permissions):
    store, controller, app = self._chat_store, self._chat_controller, self._app
    session = (
        next((row for row in store.sessions() if row.id == session_id), None)
        if store is not None
        else None
    )
    identity = (
        getattr(session, "incarnation_id", None),
        getattr(session, "conversation_binding_revision", None),
        getattr(session, "ephemeral", None),
        getattr(session, "workspace_id", None),
    )
    persistence = getattr(store, "persistence", None)
    registry = getattr(app, "workspace_registry_service", None)
    return _HookPreparationSource(
        app,
        store,
        controller,
        session_id,
        session,
        identity,
        permissions,
        persistence,
        getattr(persistence, "db", None),
        registry,
        getattr(registry, "db", None),
        getattr(app, "change_review_consent_service", None),
        getattr(controller, "_hook_authority_values", None),
        getattr(controller, "app", None),
        getattr(controller, "_turn_context_provider", None),
    )


def require_hook_preparation_source(self, source):
    self._raise_if_disposed_or_session_fenced(source.session_id)
    session = (
        next(
            (row for row in source.store.sessions() if row.id == source.session_id),
            None,
        )
        if source.store is not None
        else None
    )
    reader = getattr(source.controller, "_hook_authority_values", None)
    same_reader = reader is source.authority_reader or (
        inspect.ismethod(reader)
        and inspect.ismethod(source.authority_reader)
        and reader.__self__ is source.authority_reader.__self__
        and reader.__func__ is source.authority_reader.__func__
    )
    if (
        self._app is not source.app
        or self._chat_store is not source.store
        or self._chat_controller is not source.controller
        or self._hook_permissions is not source.permissions
        or not same_reader
        or session is not source.session
        or (
            getattr(session, "incarnation_id", None),
            getattr(session, "conversation_binding_revision", None),
            getattr(session, "ephemeral", None),
            getattr(session, "workspace_id", None),
        )
        != source.session_identity
        or getattr(source.store, "persistence", None) is not source.persistence
        or getattr(source.persistence, "db", None) is not source.chat_database
        or getattr(source.app, "workspace_registry_service", None)
        is not source.registry
        or getattr(source.registry, "db", None) is not source.workspace_database
        or getattr(source.app, "change_review_consent_service", None)
        is not source.consent
        or (
            source.controller is not None
            and (
                getattr(source.controller, "store", source.store)
                is not source.store
                or getattr(source.controller, "app", None)
                is not source.controller_app
                or getattr(source.controller, "_turn_context_provider", None)
                is not source.context_provider
                or getattr(source.controller, "_disposed", False)
                or (
                    getattr(source.controller, "_shutdown_requested", None)
                    is not None
                    and source.controller._shutdown_requested.is_set()
                )
            )
        )
    ):
        raise RuntimeError("Console hook preparation owner changed.")


async def read_hook_preparation(self, callback, source):
    current_callback = callback
    observers = ()
    controller_reads = getattr(source.controller, "_preparation_reads", None)
    if controller_reads is not None:
        observers = (controller_reads,)
    return await run_hook_preparation_read(
        current_callback,
        creator=self,
        session_id=source.session_id,
        reads=self._preparation_reads,
        observers=observers,
        require_current=lambda: self._require_hook_preparation_source(source),
        source=source,
    )


def hooks_v2_context_key(self, session_id: str):
    """Capture host workspace/binding authority, without prompt bodies."""
    from .console_runtime import _HOOK_CONTEXT_KEY_ORIGINAL_OWNER, ConsoleRuntime

    source = hook_preparation_source_for(self)
    if isinstance(source, _HookPreparationSource):
        if source.session_id != session_id:
            raise RuntimeError("Console hook preparation session changed.")
        self._require_hook_preparation_source(source)
        store, controller, session = source.store, source.controller, source.session
    else:
        store, controller = self._chat_store, self._chat_controller
        session = (
            next((row for row in store.sessions() if row.id == session_id), None)
            if store is not None
            else None
        )
    if store is None or controller is None:
        return None
    if session is None:
        raise RuntimeError("Console hook preparation session changed.")
    # Only the stock finite callback gains Workspace connection ownership.
    # Custom/injected callbacks retain the original call and cleanup shape.
    from tldw_chatbook.Chat import console_chat_controller as controller_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import registry_service as registry_module
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
    )

    app = source.app if isinstance(source, _HookPreparationSource) else self._app
    registry = (
        source.registry
        if isinstance(source, _HookPreparationSource)
        else getattr(app, "workspace_registry_service", None)
    )
    database = (
        source.workspace_database
        if isinstance(source, _HookPreparationSource)
        else getattr(registry, "db", None)
    )
    consent = (
        source.consent
        if isinstance(source, _HookPreparationSource)
        else getattr(app, "change_review_consent_service", None)
    )
    hook_anchor = controller_module._HOOK_AUTHORITY_VALUES_ORIGINAL
    controller_class, name, function, code, defaults, kwdefaults, closure = (
        hook_anchor
    )
    reader_anchors = registry_module._HOOK_WORKSPACE_READERS
    registry_class = reader_anchors[0][0]

    def bindings_current():
        return (
            controller_module._HOOK_AUTHORITY_VALUES_ORIGINAL is hook_anchor
            and controller_module.ConsoleChatController is controller_class
            and registry_module.LocalWorkspaceRegistryService is registry_class
            and registry_module._HOOK_WORKSPACE_READERS is reader_anchors
            and inspect.getattr_static(type(controller), name) is function
            and name not in vars(controller)
            and function.__code__ is code
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and function.__closure__ is closure
            and function.__globals__ is controller_module.__dict__
            and all(
                owner is registry_class
                and inspect.getattr_static(type(registry), label) is reader
                and label not in vars(registry)
                and reader.__code__ is reader_code
                and reader.__defaults__ is reader_defaults
                and reader.__kwdefaults__ is reader_kwdefaults
                and reader.__closure__ is reader_closure
                and reader.__globals__ is registry_module.__dict__
                for owner, label, reader, reader_code, reader_defaults, reader_kwdefaults, reader_closure in reader_anchors
            )
        )

    stock = (
        type(self) is _HOOK_CONTEXT_KEY_ORIGINAL_OWNER
        and ConsoleRuntime is _HOOK_CONTEXT_KEY_ORIGINAL_OWNER
        and type(store) is ConsoleChatStore
        and type(controller) is controller_class
        and controller.store is store
        and controller.app is app
        and controller._turn_context_provider is None
        and type(registry) is registry_class
        and type(database) is WorkspaceDB
        and not database.is_memory_db
        and type(consent) is ChangeReviewConsentService
        and consent._registry is registry
        and bindings_current()
    )
    if not stock:
        # Preserve the current custom callback's existing borrowed-safe scope.
        with (
            operation_owned_connection(getattr(store.persistence, "db", None)),
            operation_owned_connection(database),
        ):
            if isinstance(source, _HookPreparationSource):
                self._require_hook_preparation_source(source)
                values = controller._run_owned_chat_db_operation(
                    source.authority_reader, session_id
                )
                self._require_hook_preparation_source(source)
            else:
                values = controller._run_owned_chat_db_operation(
                    controller._hook_authority_values, session_id
                )
            return (
                session.workspace_id,
                values["workspace_roots"],
                values["project_authority"],
            )

    chat_database = getattr(store.persistence, "db", None)
    workspace_id = session.workspace_id
    actor = (os.getpid(), threading.current_thread())
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None

    def require_current():
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if (
            self._disposed
            or ConsoleRuntime is not _HOOK_CONTEXT_KEY_ORIGINAL_OWNER
            or self._app is not app
            or self._chat_store is not store
            or self._chat_controller is not controller
            or controller.store is not store
            or controller.app is not app
            or controller._turn_context_provider is not None
            or not bindings_current()
            or getattr(store.persistence, "db", None) is not chat_database
            or getattr(app, "workspace_registry_service", None) is not registry
            or registry.db is not database
            or getattr(app, "change_review_consent_service", None) is not consent
            or consent._registry is not registry
            or not any(row is session for row in store.sessions())
            or session.workspace_id != workspace_id
            or os.getpid() != actor[0]
            or threading.current_thread() is not actor[1]
            or current_task is not task
        ):
            raise RuntimeError("Console hook workspace ownership changed.")

    require_current()
    with (
        operation_owned_connection(chat_database),
        operation_owned_connection(database),
        database.connection(),
    ):
        require_current()
        # Invoke the qualified original instead of a later mutable lookup.
        values = controller._run_owned_chat_db_operation(
            function, controller, session_id
        )
        require_current()
        result = (
            workspace_id,
            values["workspace_roots"],
            values["project_authority"],
        )
    # Retirement can run callbacks; refuse a redirected result afterwards too.
    require_current()
    return result
