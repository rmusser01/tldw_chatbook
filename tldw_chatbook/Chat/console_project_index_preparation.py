"""Exact run-captured project-index scope preparation for Console."""

from __future__ import annotations

import asyncio
import inspect
import sys
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType, ModuleType

from loguru import logger

from tldw_chatbook import config
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current,
)

from . import console_chat_controller as controller_module
from .console_chat_controller import ConsoleChatController
from .console_chat_store import ConsoleChatStore
from .console_preparation_reads import preparation_source_for, run_preparation_read
from .console_runtime import ConsoleRuntime


def _profile_selection():
    """Read the in-memory selectors for the worker's profile and Workspace DB."""
    cache = config._CONFIG_CACHE
    defaults = config.DEFAULT_CONFIG_FROM_TOML
    if cache is None:
        cache = {}
    if type(cache) is not dict or type(defaults) is not dict:
        return None

    sections = (
        cache.get("paths"),
        cache.get("Paths"),
        cache.get("general"),
        cache.get("database"),
        defaults.get("general"),
        defaults.get("database"),
    )
    if any(section is not None and type(section) is not dict for section in sections):
        return None
    paths, legacy_paths, general, database, default_general, default_database = (
        section if section is not None else {} for section in sections
    )
    data_dir = paths.get("data_dir")
    if data_dir is None:
        data_dir = legacy_paths.get("data_dir")
    values = (
        data_dir,
        general.get("users_name", default_general.get("users_name", "default_user")),
        database.get("workspaces_db_path"),
        default_database.get("workspaces_db_path"),
    )
    if any(value is not None and type(value) is not str for value in values):
        return None
    return (
        config.current_config_identity()[1],
        config._default_base_data_dir(),
        *values,
    )


class _ProjectIndexPreparation:
    """Parent-only sources for the complete finite preparation read."""

    def __init__(
        self, controller, session, runtime, callback, roots, issuing_context, *, profile
    ):
        self.controller = controller
        self.session_id = session.id
        self.session = session
        self.session_state = self._session_state()
        self.runtime = runtime
        self.app = controller.app
        self.domain = controller.domain_services
        self.sources = self.domain.sources if self.domain is not None else None
        if self.domain is not None:
            self.domain.require_current(self.sources)
        self.index_owner = (
            self.domain._project_index_owner if self.domain is not None else None
        )
        self.store = controller.store
        self.reads = controller._preparation_reads
        self.runtime_reads = runtime._preparation_reads
        self.registry = (
            self.sources.workspace_registry
            if self.domain is not None
            else inspect.getattr_static(self.app, "workspace_registry_service", None)
        )
        self.database = inspect.getattr_static(self.registry, "db", None)
        self.database_path = self._database_path()
        self.config = config
        self.profile = profile
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        self.callback = callback
        self.roots = roots
        self.turn_context, self.project_selection, self.project_guard = issuing_context
        self.service = (
            self.index_owner.service
            if self.index_owner is not None
            else None
            if self.domain is not None
            else inspect.getattr_static(self.app, "_project_index_service", None)
        )
        self.factory = inspect.getattr_static(self.service, "_factory", None)
        self.service_closed = (
            self.index_owner.closed
            if self.index_owner is not None
            else False
            if self.domain is not None
            else inspect.getattr_static(self.app, "_project_index_closed", False)
        )

    def _database_path(self):
        return (
            inspect.getattr_static(self.database, "db_path", None),
            inspect.getattr_static(self.database, "db_path_str", None),
        )

    def _session_state(self):
        return (
            self.session.incarnation_id,
            self.session.conversation_binding_revision,
            self.session.workspace_id,
            self.controller._lifecycle_revision_for(self.session_id),
        )

    def require_current(self):
        controller, runtime = self.controller, self.runtime
        if self.domain is not None:
            self.domain.require_current(self.sources)
            if (
                controller.domain_services is not self.domain
                or runtime.domain_services is not self.domain
                or self.domain._project_index_owner is not self.index_owner
                or self.sources.workspace_registry is not self.registry
                or (
                    self.index_owner is not None
                    and (
                        self.index_owner.registry is not self.registry
                        or self.index_owner.closed is not self.service_closed
                        or (
                            self.service is not None
                            and self.index_owner.service is not self.service
                        )
                    )
                )
            ):
                raise RecoveryRequired("console_snapshot_owner_changed")
            if self.index_owner is not None:
                self.index_owner.require_current()
        if (
            not _stock_callback(controller, self.callback)
            or controller.app is not self.app
            or controller.store is not self.store
            or self.store._sessions.get(self.session_id) is not self.session
            or self._session_state() != self.session_state
            or controller._disposed
            or runtime._disposed
            or controller._hooks_v2_runtime is not runtime
            or (
                self.domain is None
                and inspect.getattr_static(self.app, "console_runtime", None)
                is not runtime
            )
            or runtime._app is not self.app
            or runtime._chat_controller is not controller
            or runtime._chat_store is not self.store
            or self.session_id in runtime._admission_fenced_sessions
            or controller._preparation_reads is not self.reads
            or runtime._preparation_reads is not self.runtime_reads
            or (
                self.domain is None
                and inspect.getattr_static(self.app, "workspace_registry_service", None)
                is not self.registry
            )
            or inspect.getattr_static(self.registry, "db", None) is not self.database
            or self._database_path() != self.database_path
            or config is not self.config
            or not _profile_definitions_current()
            or _profile_selection() != self.profile
            or (
                self.domain is None
                and inspect.getattr_static(self.app, "_project_index_closed", False)
                is not self.service_closed
            )
            or (
                self.service is not None
                and (
                    (
                        self.domain is None
                        and inspect.getattr_static(
                            self.app, "_project_index_service", None
                        )
                        is not self.service
                    )
                    or inspect.getattr_static(self.service, "_factory", None)
                    is not self.factory
                )
            )
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    def require_adoption(self):
        if (
            asyncio.get_running_loop() is not self.loop
            or asyncio.current_task() is not self.task
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        self.require_current()


async def prepare_project_index_for_composition(
    controller, session_id, roots, turn_context, project_selection, project_guard
):
    """Keep the original worker and both of its physical read observers."""
    callback = controller._prepare_project_index_scopes
    runtime = (
        getattr(controller, "_hooks_v2_runtime", None)
        if type(controller) is ConsoleChatController
        and _stock_callback(controller, callback)
        else None
    )
    store = (
        getattr(controller, "store", None) if type(runtime) is ConsoleRuntime else None
    )
    if (
        type(runtime) is ConsoleRuntime
        and type(store) is ConsoleChatStore
        and runtime._chat_controller is controller
        and runtime._chat_store is store
        and runtime._app is controller.app
        and (
            controller.domain_services is not None
            or inspect.getattr_static(controller.app, "console_runtime", None)
            is runtime
        )
        and type(controller._preparation_reads) is set
        and type(runtime._preparation_reads) is set
        and _profile_definitions_current()
        and all(type(root) is RunAdmittedWorkspaceRoot for root in roots)
        and (session := store._sessions.get(session_id)) is not None
        and (profile := _profile_selection()) is not None
    ):
        source = _ProjectIndexPreparation(
            controller,
            session,
            runtime,
            callback,
            roots,
            (turn_context, project_selection, project_guard),
            profile=profile,
        )
        result = await run_preparation_read(
            partial(callback, roots),
            creator=controller,
            session_id=session_id,
            reads=source.reads,
            observers=(source.runtime_reads,),
            source=source,
            require_current=source.require_current,
        )
        source.require_adoption()
        return result
    return await controller_module.asyncio.to_thread(callback, roots)


def prepare_project_index_scopes(self: ConsoleChatController, admitted_roots):
    """Retire registry handles opened by scope preparation, including guards."""
    from tldw_chatbook.Workspaces.project_index.runtime import get_project_index_service

    domain = getattr(self, "domain_services", None)
    if domain is not None and domain.sources.workspace_registry is None:
        domain.require_current()
        return None, {}
    try:
        service = (
            domain.project_index_owner().get()
            if domain is not None
            else get_project_index_service(self.app)
        )
        source = preparation_source_for(self)
        if type(source) is _ProjectIndexPreparation and source.service is None:
            # The original worker owns cold construction and installs its result.
            source.service = service
            source.factory = inspect.getattr_static(service, "_factory", None)
        with service.registry_connection_scope():
            return _prepare_project_index_scopes_owned(self, admitted_roots)
    except Exception:
        if domain is not None:
            raise
        logger.debug("Project index preparation unavailable")
        return None, {}


def _prepare_project_index_scopes_owned(self: ConsoleChatController, admitted_roots):
    """Read compatible immutable scopes off-loop after exact run capture."""
    from types import MappingProxyType
    from tldw_chatbook.Tools.remote_root_types import is_remote, local_root_path
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service,
        project_index_key,
        project_index_status_for_console,
    )
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexRetirementUnproven,
    )

    scopes = {}
    service = None
    for authority in admitted_roots:
        if is_remote(authority.root):
            continue
        try:
            if not authority.guard(False):
                continue
            root = local_root_path(
                authority.root, site="project index scope preparation"
            )
            from tldw_chatbook.Workspaces.project_index.key_process import (
                derive_for_console,
            )

            domain = getattr(self, "domain_services", None)
            if domain is not None:
                from tldw_chatbook.Workspaces.project_index.runtime import (
                    _project_index_key_owned,
                )

                service = domain.project_index_owner().get()
                key = _project_index_key_owned(
                    service, authority.workspace_id, authority.binding_id, root
                )
            else:
                key = derive_for_console(
                    self,
                    authority.workspace_id,
                    authority.binding_id,
                    root,
                    ordinary=project_index_key,
                )
                service = get_project_index_service(self.app)
            status = project_index_status_for_console(service, key)
            if status is None:
                continue
            compatible = status.state in {"Ready", "Partial"}
            if status.state == "Building" and status.indexed > 0:
                from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

                paths = (
                    authority.exclusions_provider()
                    if authority.exclusions_provider
                    else ()
                )
                floor = tuple(
                    SensitiveExclusion(
                        "subtree", Path(path).relative_to(root).as_posix()
                    )
                    for path in paths
                )
                compatible = service.building_coverage(status.scope, floor)
            if compatible and authority.guard(False):
                scopes[authority.alias] = status.scope
        except ProjectIndexRetirementUnproven:
            raise
        except Exception:  # unavailable optional capability grants no authority
            logger.debug("Project index unavailable for this run binding")
    return service, MappingProxyType(scopes)


_CONTROLLER_DEFINITIONS = controller_module._CONSOLE_PROJECT_INDEX_PREPARATION_METHODS
_PREPARATION_DEFINITIONS = tuple(
    (function.__name__, function, function.__code__)
    for function in (prepare_project_index_scopes, _prepare_project_index_scopes_owned)
)
_PROFILE_DEFINITIONS = config._KEY_PROCESS_DEFINITIONS


def _profile_definitions_current():
    return (
        type(config) is ModuleType
        and sys.modules.get("tldw_chatbook.config") is config
        and vars(config) is config._KEY_PROCESS_GLOBALS
        and definitions_current(_PROFILE_DEFINITIONS)
    )


def _stock_callback(controller, callback):
    """Qualify just the original Console preparation and lifecycle readers."""
    return (
        type(callback) is MethodType
        and callback.__self__ is controller
        and callback.__func__ is _CONTROLLER_DEFINITIONS[0][1]
        and all(
            type(function) is FunctionType
            and code is not None
            and code.co_qualname == f"ConsoleChatController.{name}"
            and code.co_filename == controller_module.__file__
            and function.__globals__ is vars(controller_module)
            and inspect.getattr_static(controller, name, None) is function
            and function.__code__ is code
            and function.__defaults__
            is function.__kwdefaults__
            is function.__closure__
            is None
            for name, function, code in _CONTROLLER_DEFINITIONS
        )
        and all(
            globals().get(name) is function
            and function.__code__ is code
            and function.__defaults__
            is function.__kwdefaults__
            is function.__closure__
            is None
            for name, function, code in _PREPARATION_DEFINITIONS
        )
    )


_KEY_PREPARATION_READERS = capture_definitions(
    (
        (_ProjectIndexPreparation, "__init__", _ProjectIndexPreparation.__init__),
        (
            _ProjectIndexPreparation,
            "_database_path",
            _ProjectIndexPreparation._database_path,
        ),
        (
            _ProjectIndexPreparation,
            "_session_state",
            _ProjectIndexPreparation._session_state,
        ),
        (
            _ProjectIndexPreparation,
            "require_current",
            _ProjectIndexPreparation.require_current,
        ),
        (
            _ProjectIndexPreparation,
            "require_adoption",
            _ProjectIndexPreparation.require_adoption,
        ),
    )
)
