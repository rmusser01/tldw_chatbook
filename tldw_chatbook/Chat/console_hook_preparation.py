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
    "ConsoleHookAttemptRead",
    "ConsoleHookPreparationRead",
    "HookCheckOwnerChanged",
    "bind_hook_preparation_session",
    "drain_hook_preparation_reads",
    "hook_preparation_reads_for",
    "hook_preparation_session_for",
    "hook_preparation_source_for",
    "observe_hook_preparation_reads",
    "run_hook_preparation_read",
)


class HookCheckOwnerChanged(RuntimeError):
    """The chat a hook check read for moved; the read is void, not failed.

    TASK-38751: ``console_automatic_submission`` turns it into "send again"
    copy, never hooks advice.
    """


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleHookAttemptRead:
    """One Send attempt's single pre-commit hook authority read (ADR-230 #3).

    The attempt's first full consent read -- received-intent preparation, or
    the controller's first submission admission on entry points without one
    -- produces this. The same attempt then passes it explicitly, as an
    argument, to its later pre-commit preparation consumers (submission
    admission, v2 hook preparation, legacy UserPromptSubmit target selection)
    so they do not repeat the full read. It is never held by an owner, keyed
    by time or handed to another attempt or session.

    It grants nothing: each consumer accepts it only for its own session and
    owner and only while ``HookPermissions.attempt_read_current`` holds,
    otherwise it performs its existing fresh read. Even then it answers only
    "nothing to do" -- admission that it does not refuse, no UserPromptSubmit
    hook to select, no v2 hook to prepare; a consumer with a hook to select or
    prepare reads fresh. The final pre-dispatch admission is always a fresh
    read, and every hook launch keeps its own fresh ``launch_guard``.

    Attributes:
        session_id: The Console session the attempt belongs to.
        authority: The owner's ``HookAuthorityRead`` for that attempt.
    """

    session_id: str
    authority: Any = field(repr=False)

    def authority_for(self, owner: object, session_id: str | None) -> Any:
        """Return the read for this exact session and owner, else ``None``.

        Args:
            owner: The consent owner the consumer would otherwise read.
            session_id: The consumer's session.

        Returns:
            The ``HookAuthorityRead`` when both match; the owner still decides
            whether it stands (``attempt_read_current``).
        """
        if (
            session_id is None
            or session_id != self.session_id
            or getattr(self.authority, "owner", None) is not owner
        ):
            return None
        return self.authority


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
    native = self._native_plugin_hook_preparation
    if native is not None:
        native.pin_resources()
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
    try:
        self._raise_if_disposed_or_session_fenced(source.session_id)
    except RuntimeError as error:
        raise HookCheckOwnerChanged(str(error)) from error
    if not hook_preparation_source_current(self, source):
        raise HookCheckOwnerChanged("Console hook preparation owner changed.")


def hook_preparation_source_current(self, source):
    """Original direct-reference fence, safe at actual worker queue publication."""
    session = source.store._sessions.get(source.session_id) if source.store else None
    reader = getattr(source.controller, "_hook_authority_values", None)
    same_reader = reader is source.authority_reader or (
        inspect.ismethod(reader)
        and inspect.ismethod(source.authority_reader)
        and reader.__self__ is source.authority_reader.__self__
        and reader.__func__ is source.authority_reader.__func__
    )
    return not (
        self._disposed
        or source.session_id in self._admission_fenced_sessions
        or self._app is not source.app
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
    )


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
    from tldw_chatbook.Agents.activation import callable_source_current
    from tldw_chatbook.Chat import console_chat_controller as controller_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.base_db import operation_owned_connection
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import registry_service as registry_module
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
        hook_workspace_readers_current,
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
    controller_class, name, source_record = hook_anchor
    function = source_record[0]
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
            and callable_source_current(source_record)
            and function.__globals__ is controller_module.__dict__
            and hook_workspace_readers_current(
                registry, registry_class, reader_anchors, registry_module.__dict__
            )
        )

    def project_key(values):
        authority = values["project_authority"]
        (
            snapshot_type,
            capture,
            capture_code,
            capture_defaults,
            capture_kwdefaults,
            capture_closure,
        ) = controller_module._HOOK_PROJECT_AUTHORITY_ORIGINAL
        if (
            type(controller) is controller_class
            and controller._turn_context_provider is None
            and type(registry) is registry_class
            and bindings_current()
            and controller_module.capture_project_instruction_authority is capture
            and capture.__code__ is capture_code
            and capture.__defaults__ is capture_defaults
            and capture.__kwdefaults__ is capture_kwdefaults
            and capture.__closure__ is capture_closure
            and type(authority) is snapshot_type
        ):
            # Validated first-folder adoption and consent record these existing
            # choices later. Full effective authority still revokes on change.
            return (
                authority.workspace_id,
                authority.enabled,
                authority.selected,
                authority.options,
            )
        return authority

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
                project_key(values),
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
            project_key(values),
        )
    # Retirement can run callbacks; refuse a redirected result afterwards too.
    require_current()
    return result


def attempt_v2_configuration_for(permissions, session_id, hook_read):
    """Project only a standing read from the same Send and consent owner."""
    shared = None
    authority = (
        hook_read.authority_for(permissions, session_id)
        if hook_read is not None
        else None
    )
    if authority is not None:
        from tldw_chatbook.Agents.hook_permissions import HookPermissions

        # A replaced reader keeps its original call (ADR-230 decision 8).
        if (
            "v2_configuration" not in vars(permissions)
            and type(permissions).v2_configuration is HookPermissions.v2_configuration
        ):
            shared = permissions.attempt_v2_configuration(authority)
    return shared


@contextmanager
def submission_origin(controller, session, configuration, origin):
    """Bind only the original validated submission task, never event labels."""
    from tldw_chatbook.Agents.run_hooks import native_hook_submission as _submission

    token = _submission.set(
        (asyncio.current_task(), controller, session, configuration, origin)
    )
    try:
        yield
    finally:
        _submission.reset(token)


async def prepare_hooks_v2(
    self,
    session_id: str,
    *,
    reason="startup",
    initiator="manual",
    configuration=None,
    _hook_read=None,
    test_only=False,
    _standalone_compaction=False,
):
    """Initialize only at validated execution admission, never at view access.

    ``_hook_read`` is the submitting attempt's own earlier full consent
    read (``ConsoleHookAttemptRead``). While it stands for this session
    and owner, and only when it answers "no v2 hook to prepare"
    (``_hook_read_prepares_nothing``), it replaces the
    ``v2_configuration()`` re-read (ADR-230 decision 3). Everything that
    builds, keeps or replaces an engine still reads fresh, as before.
    """
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.run_hooks import load_hooks_config

    from . import console_runtime as native_host

    self._raise_if_disposed_or_session_fenced(session_id)
    permissions = self.ensure_hook_permissions()
    source = self._capture_hook_preparation_source(session_id, permissions)
    shared = attempt_v2_configuration_for(permissions, session_id, _hook_read)
    if shared is not None and self._hook_read_prepares_nothing(
        session_id, shared[0], configuration
    ):
        # The early return below, reached without the re-read: nothing in
        # the attempt's read or this session's state needs an engine.
        self._require_hook_preparation_source(source)
        self._raise_if_disposed_or_session_fenced(session_id)
        return None
    review, targets = await self._read_hook_preparation(
        permissions.v2_configuration, source
    )
    self._require_hook_preparation_source(source)
    configured = load_hooks_config(
        {"hooks": review.config.section} if review.config.section_present else {}
    )
    engine = self.get_hooks_v2(session_id)
    native = getattr(engine, "native_plugins", None) if configuration is None else None
    plugin_request = None
    hook_configuration = configuration
    async with native_host.contextlib.AsyncExitStack() as preparation_handoff:
        if configuration is not None:
            maximum = configuration.skill_context_maximum
            if maximum.get("_plugin_requirement_ceiling") or any(
                row.get("plugin_owned") for row in maximum.get("available_skills", ())
            ):
                local = getattr(
                    self._chat_controller._skills_service, "local_service", None
                )
                service = getattr(local, "plugin_service", None)
                if service is None:
                    raise PermissionError("plugin_hook_authority_unavailable")
                if self._native_plugin_hook_preparation_required:
                    preparation = self._native_plugin_hook_preparation
                    if preparation is None:
                        raise RuntimeError(
                            "Native plugin hook preparation unavailable."
                        )
                    (
                        native,
                        plugin_request,
                    ) = await preparation_handoff.enter_async_context(
                        preparation.capture(source, service, configuration, review)
                    )
                    if (
                        _standalone_compaction
                        and configuration.mcp_capture_pending
                        and configuration.mcp_definition_capture == "captured"
                        and any(handler.type == "mcp_tool" for handler in native.definitions)
                    ):
                        hook_configuration = await plugin_request.resolve_compaction_mcp()
                else:
                    native = await service.hook_configuration(
                        maximum,
                        scratch_manager=self._scratch_spaces,
                        scratch_snapshot=self._scratch_spaces.snapshot(session_id),
                    )
                self._require_hook_preparation_source(source)
        signature = (
            configured,
            targets,
            native.signature if native is not None else None,
        )
        engine = self.get_hooks_v2(session_id)
        previous = self._hooks_v2_configured.get(session_id)
        if (
            review.ready
            and not configured.v2_handlers
            and not configured.v2_invalid_admissions
            and (native is None or not native.definitions)
            and engine is None
            and session_id not in self._hooks_v2_engines
            and session_id not in self._hooks_v2_lifecycles
            and session_id not in self._hooks_v2_configured
        ):
            self._require_hook_preparation_source(source)
            return None
        context_key = await self._read_hook_preparation(
            native_host.functools.partial(self._hooks_v2_context_key, session_id),
            source,
        )
        self._require_hook_preparation_source(source)
        self._raise_if_disposed_or_session_fenced(session_id)
        owner = self._hooks_v2_lifecycles.get(session_id)
        context_changed = owner is not None and owner.context_key != context_key
        if context_changed and native is not None and configuration is None:
            raise PermissionError("plugin_hook_workspace_admission_required")
        if (
            test_only
            and engine is not None
            and (
                previous != signature
                or context_changed
                or owner is None
                or not owner.live
                or owner.turn_scope is not None
            )
        ):
            raise PermissionError("plugin_test_requires_compatible_idle_hook_session")
        if (previous is not None and previous != signature) or context_changed:
            # The controller owns a reversible validation slot at this point.
            owner = self._hooks_v2_lifecycles.get(session_id)
            if owner is not None and getattr(owner, "turn_scope", None) is not None:
                raise RuntimeError("hook replacement requires idle session")
            await self.close_hooks_v2(session_id)
            self._require_hook_preparation_source(source)
            if self._hooks_v2_lifecycles.get(session_id) is owner:
                self._hooks_v2_lifecycles.pop(session_id, None)
            if context_changed and previous is None and engine is not None:
                # Host-injected definitions retain their authority resolver.
                engine = self.ensure_hooks_v2(
                    session_id,
                    engine.definitions,
                    engine.authority_check,
                    _native_preparation=plugin_request,
                    enabled=engine.enabled,
                    invalid_admissions=engine.invalid_admissions,
                    process_owner=engine.processes.owner,
                    process_owner_for=engine._process_owner_for,
                    environment=engine._environment,
                    host_environment=engine._host_environment,
                    command_launch=engine._command_launch,
                    dependency_required=engine._dependency_required,
                    event_projector=engine._event_projector,
                    launch_guard=engine.processes._launch_guard,
                    effect_authority_check=engine._effect_authority_check,
                )
            else:
                engine = None
            reason = "configuration_changed"
        if plugin_request is not None:
            plugin_request.check(service)
        if engine is None:
            if not review.ready:
                raise RuntimeError("Review enabled hooks before execution.")
            if (
                not configured.v2_handlers
                and not configured.v2_invalid_admissions
                and (native is None or not native.definitions)
            ):
                return None
            captured = {target.spec.id: target for target in targets}

            def session_end_current(handler, event):
                owner = getattr(engine, "lifecycle_owner", None)
                return bool(
                    self._disposed
                    and handler.type == "command"
                    and handler.event == "SessionEnd"
                    and not handler.effects
                    and not handler.required
                    and owner is not None
                    and self.get_hooks_v2(session_id) is engine
                    and owner._session_end_current(event)
                )

            def authority(handler, _event, _stage):
                if native is not None and handler.id in native.owners:
                    return (
                        not self._disposed
                        and permissions.configuration_current(review)
                        and native.authority(handler, _event, _stage)
                        and permissions.configuration_current(review)
                    )
                target = captured.get(handler.id)
                return bool(
                    target is not None
                    and (
                        not self._disposed
                        and permissions.target_current(target)
                        or session_end_current(handler, _event)
                        and permissions._session_end_current(
                            review, target, lambda: session_end_current(handler, _event)
                        )
                    )
                )

            def effects_current(handler, _event, _stage):
                if native is not None and handler.id in native.owners:
                    return (
                        not self._disposed
                        and permissions.configuration_current(review)
                        and native.effects_current(handler, _event, _stage)
                    )
                target = captured.get(handler.id)
                return bool(
                    target is not None
                    and not self._disposed
                    and permissions.configuration_current(review)
                    and permissions.target_current(target, refresh=False)
                )

            @native_host.contextlib.contextmanager
            def launch_guard(handler, event):
                if native is not None and handler.id in native.owners:
                    from tldw_chatbook import config

                    with config.locked_hooks_config_snapshot() as current:
                        if (
                            current.section_stamp != review.config.section_stamp
                            or not effects_current(handler, event, "launch")
                        ):
                            raise PermissionError("plugin_hook_authority_changed")
                        yield
                else:
                    target = captured[handler.id]
                    guard = (
                        permissions._session_end_launch_guard(
                            review, target, lambda: session_end_current(handler, event)
                        )
                        if session_end_current(handler, event)
                        else permissions.launch_guard(target, tool_name=None)
                    )
                    with guard:
                        yield

            engine = self.ensure_hooks_v2(
                session_id,
                (configured.v2_handlers if configured.enabled else ())
                + (native.definitions if native is not None else ()),
                authority,
                _native_preparation=plugin_request,
                _native_plugins=native,
                _configured_signature=signature,
                process_owner=native,
                process_owner_for=native.process_owner if native is not None else None,
                host_environment=native.host_environment
                if native is not None
                else None,
                event_projector=native.project_event if native is not None else None,
                command_launch=native.command_launch if native is not None else None,
                dependency_required=native.dependency_required
                if native is not None
                else None,
                launch_guard=launch_guard,
                effect_authority_check=effects_current,
                enabled=configured.enabled or bool(native and native.definitions),
                invalid_admissions=(
                    configured.v2_invalid_admissions if configured.enabled else ()
                ),
            )
            if test_only:
                created = getattr(self, "_plugin_test_created_engines", None)
                if created is None:
                    created = self._plugin_test_created_engines = {}
                created[native_host.asyncio.current_task()] = engine
    lifecycle = self._hooks_v2_lifecycles.get(session_id)
    if lifecycle is None:

        def current():
            try:
                return (
                    not self._disposed
                    and self.get_hooks_v2(session_id) is engine
                    and self._hooks_v2_context_key(session_id) == context_key
                    and (
                        session_id not in self._hooks_v2_configured
                        or permissions.configuration_current(review)
                        and all(
                            permissions.target_current(target, refresh=False)
                            for target in targets
                        )
                    )
                )
            except Exception:  # noqa: BLE001 -- hook boundary
                return False

        lifecycle = HookSessionLifecycle(
            engine,
            session_id,
            current=current,
            render_context=native.render_context if native is not None else None,
        )
        lifecycle.context_key = context_key
        self._hooks_v2_lifecycles[session_id] = lifecycle
        engine.lifecycle_owner = lifecycle
    pending_scope = None
    if configuration is not None:
        pending_scope = lifecycle.open_scope()
        lifecycle.turn_scope = pending_scope
        self._chat_controller._hooks_v2_submissions[
            native_host.asyncio.current_task()
        ] = (
            lifecycle,
            pending_scope,
            session_id,
        )
        try:
            if engine.mcp_executor is not None:
                if hook_configuration is not configuration:
                    plugin_request.require_original_mcp()
                context = await self._chat_controller.compose_prospective_hook_context(
                    hook_configuration, lifecycle, pending_scope
                )
                self._require_hook_preparation_source(source)
                if hook_configuration is not configuration:
                    plugin_request.require_original_mcp()
                engine.mcp_executor.bind_context(context)
        except BaseException:
            lifecycle.close_scope(pending_scope)
            lifecycle.turn_scope = None
            raise
    if not lifecycle.live:
        start_event = lifecycle.event(
            "SessionStart", data={"reason": reason}, initiator=initiator
        )
        if test_only:
            start_event = start_event.model_copy(update={"origin": "host_plugin_test"})
        token = lifecycle.reserve(start_event)
        try:
            await lifecycle.initialize(token)
            self._require_hook_preparation_source(source)
            lifecycle.publish(token)
        except BaseException:
            lifecycle.cancel(token)
            if pending_scope is not None:
                lifecycle.close_scope(pending_scope)
                lifecycle.turn_scope = None
            # A failed provisional initialization has no live session effects.
            if self._hooks_v2_lifecycles.get(session_id) is lifecycle:
                self._hooks_v2_lifecycles.pop(session_id, None)
            raise
    if configuration is not None and engine.mcp_executor is not None:
        engine.mcp_executor.retain_runtime(session_id)
    return lifecycle
