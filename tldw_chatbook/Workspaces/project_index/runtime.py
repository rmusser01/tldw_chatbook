"""Lazy application composition after admitted private data authority capture."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import asyncio
import errno
import hashlib
import json
import sqlite3
import stat
from pathlib import Path
import threading
import sys
from types import MethodType

from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current,
)
from tldw_chatbook.Backup_Recovery.activation import execution_scope
from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.DB.Workspace_Project_Index_DB import (
    WorkspaceProjectIndexDB,
    index_store_path,
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Utils import filesystem_identity, platform_files
from tldw_chatbook.Utils.private_paths import secure_private_directory, PrivatePathError
from ..registry_service import LocalWorkspaceRegistryService
from .lifetime import BindingLifetime
from .models import (
    MAX_CACHE_BYTES,
    OWNER_ID,
    ProjectIndexKey,
    ProjectIndexPolicy,
    ProjectIndexStatus,
    ProjectIndexRetirementUnproven,
    locator_fingerprint,
    root_identity_fingerprint,
)
from .scanner import ProjectScanner
from .service import ProjectIndexService, binding_bytes, _CONSOLE_STATUS_READERS


_composition_lock = threading.Lock()
_KEY_PROCESS_ENABLED = False


@contextmanager
def project_index_registry_connections(database):
    """Retire newly owned registry handles with positive native/core proof.

    This finite context grants no authority. Borrowed caller handles keep their
    own lifetime; a failed newly owned close remains counted by storage admission.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Backup_Recovery.participants import core_cached_connection

    if type(database) is not WorkspaceDB or database.is_memory_db:
        with operation_owned_connection(database):
            yield
        return
    previous = core_cached_connection(
        database, getattr(database._thread_local, "conn", None)
    )
    owned = None
    try:
        with operation_owned_connection(database):
            try:
                yield
            finally:
                current = getattr(database._thread_local, "conn", None)
                if current is not None and current is not previous:
                    owned = current
    finally:
        if owned is not None:
            retired = core_cached_connection(database, owned) is None
            try:
                sqlite3.Connection.in_transaction.__get__(owned)
            except sqlite3.ProgrammingError:
                native_closed = True
            except Exception:
                native_closed = False
            else:
                native_closed = False
            if not retired or not native_closed:
                raise ProjectIndexRetirementUnproven(
                    "index_registry_retirement_unproven"
                )


class _DataAuthorityPin:
    """A frozen identity backed by actual retained admission, not idle authority."""

    def __init__(self, factory, stack, identity, data_root, chain, lease):
        self.factory = factory
        self.stack = stack
        self.identity = identity
        self.data_root = data_root
        self.chain = chain
        self.closed = False
        self.lease = lease
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        with storage._lock:
            self.hold = storage._holds.get(lease._key)

    @contextmanager
    def bind(self):
        if self.closed:
            raise RuntimeError("data_authority_retired")
        previous = getattr(self.factory._local, "pin", None)
        self.factory._local.pin = self
        try:
            yield self
        finally:
            self.factory._local.pin = previous

    def close(self):
        if not self.closed:
            self.stack.close()
            from tldw_chatbook.Backup_Recovery import storage_admission as storage
            from tldw_chatbook.MCP.local_permission_operation import (
                _hold_retired_or_shared,
            )

            with storage._lock:
                if (
                    self.lease in storage._live_leases
                    or self.lease._key is not None
                    or not _hold_retired_or_shared(self.hold, storage)
                ):
                    raise ProjectIndexRetirementUnproven(
                        "index_pin_retirement_unproven"
                    )
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _AdmittedStoreFactory:
    def __init__(self):
        self._local = threading.local()
        self._unretired = {}
        self.identity = None
        self.data_root = None
        with self.retain() as pin:
            self._composition_pin = pin
            self.identity = pin.identity
            self.data_root = pin.data_root

    def retain(self, key=None):
        """Capture and verify outside fences; caller keeps this through retirement."""
        from tldw_chatbook.config import get_user_data_dir

        stack = ExitStack()
        try:
            data = Path(get_user_data_dir()).absolute()
            lease = stack.enter_context(acquire_storage(data))
            if not stack.enter_context(
                execution_scope((OWNER_ID,), data, retained=lease)
            ):
                raise RuntimeError("index_data_authority_unavailable")
            selector = str(effective_config_path().absolute())
            installation, generation = verified_data_identity()
            chain = capture_directory_chain(data)
            identity = hashlib.sha256(
                json.dumps(
                    (
                        selector,
                        str(data),
                        root_identity_fingerprint(chain),
                        installation,
                        generation,
                    ),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if self.identity is not None and (
                identity != self.identity or data != self.data_root
            ):
                raise ValueError("index_data_authority_changed")
            if key is not None and key.data_authority_id != identity:
                raise ValueError("index_data_authority_changed")
            pin = _DataAuthorityPin(self, stack, identity, data, chain, lease)
            recorder = sys.modules.get(
                "tldw_chatbook.Workspaces.project_index.key_process"
            )
            if recorder is not None:
                recorder._retain_pin(pin)
            return pin
        except BaseException:
            stack.close()
            raise

    def current_data_authority(self) -> str:
        """Bounded memory read only, valid exclusively inside a retained pin."""
        pin = getattr(self._local, "pin", None)
        return pin.identity if pin is not None and not pin.closed else ""

    @contextmanager
    def _migration_context(self, key, store):
        """Serialize only legacy migration and reserve the entire binding's disk."""
        group = store.db_path.parent.parent
        if key.binding_namespace in self._unretired:
            raise ProjectIndexRetirementUnproven("index_migration_retirement_unproven")
        lease = BindingLifetime.acquire(group / key.lifetime_name)
        retained = False
        try:
            total = binding_bytes(group)
            other = max(0, total - store._cache_bytes())
            with store.connection() as conn:
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                pages = (MAX_CACHE_BYTES - other - 1024**2) // (2 * page_size)
                if pages < 1 or conn.execute("PRAGMA page_count").fetchone()[0] > pages:
                    raise ValueError("index_cache_limit")
                conn.execute(f"PRAGMA max_page_count={pages}")
            yield
        except BaseException as cause:
            try:
                if isinstance(cause, ProjectIndexRetirementUnproven):
                    raise cause
                store.close_native()
            except BaseException as error:
                self._unretired[key.binding_namespace] = (store, lease)
                retained = True
                raise ProjectIndexRetirementUnproven(
                    "index_migration_retirement_unproven"
                ) from error
            raise
        finally:
            if not retained:
                lease.close()

    def assert_retired(self):
        if self._unretired:
            raise ProjectIndexRetirementUnproven("index_migration_retirement_unproven")

    def open_cleanup(self, key):
        """Open a validated management namespace without upgrading existing data."""
        return self(key, cleanup_only=True)

    def __call__(self, key, *, cleanup_only=False):
        if self.current_data_authority() != key.data_authority_id:
            raise ValueError("index_data_authority_unavailable")
        if key.binding_namespace in self._unretired:
            raise ProjectIndexRetirementUnproven("index_migration_retirement_unproven")
        path = index_store_path(self.data_root, key)
        group = path.parent.parent
        if not path.exists():
            result = secure_private_directory(
                group, create=True, application_owned=True
            )
            if not result.verified_private:
                raise PrivatePathError(result)
            lease = BindingLifetime.acquire(group / key.lifetime_name)
            try:
                # Namespace creation is serialized with all publishers, and its
                # schema/journal reserve is included before opening SQLite.
                if binding_bytes(group) + 1024**2 >= MAX_CACHE_BYTES:
                    raise ValueError("index_cache_limit")
                result = secure_private_directory(
                    path.parent, create=True, application_owned=True
                )
                if not result.verified_private:
                    raise PrivatePathError(result)
                return WorkspaceProjectIndexDB(
                    path,
                    migration_context=lambda store: self._migration_context(key, store),
                )
            finally:
                lease.close()
        return WorkspaceProjectIndexDB(
            path,
            cleanup_only=cleanup_only,
            migration_context=lambda store: self._migration_context(key, store),
        )


class ProjectIndexOwner:
    """One lazy stock composition and its physical close, shared by adapters."""

    def __init__(self, registry):
        if type(registry) is not LocalWorkspaceRegistryService:
            raise TypeError("Custom project index registries use legacy composition")
        self.registry = registry
        self._registry = registry
        self.service = None
        self.closed = False
        self._invalid = False
        self._app = None
        self._unpublished = []
        self._close_task = None
        self._console_controls = None

    def require_current(self):
        from tldw_chatbook.Utils.definition_records import definitions_current

        app = self._app
        if (
            self._invalid
            or not definitions_current(_KEY_PROCESS_DEFINITIONS)
            or self.registry is not self._registry
            or (
                app is not None
                and (
                    getattr(app, "_project_index_owner", None) is not self
                    or getattr(app, "workspace_registry_service", None)
                    is not self.registry
                    or getattr(app, "_project_index_service", None) is not self.service
                    or bool(getattr(app, "_project_index_closed", False)) != self.closed
                )
            )
        ):
            self._invalid = True
            raise RuntimeError("project_index_owner_changed")
        if self.closed:
            raise RuntimeError("project_index_closed")

    def get(self) -> ProjectIndexService:
        with _composition_lock:
            self.require_current()
            if self.service is None:
                from tldw_chatbook.Chat.console_control_index import (
                    index_composition,
                    bind_index_service,
                )

                with index_composition(self):
                    service = _compose_project_index_service(self.registry)
                    try:
                        service._unsubscribe = (
                            self.registry.subscribe_project_index_changes(
                                service.binding_changed
                            )
                        )
                    except BaseException:
                        self._unpublished.append(service)
                        self._invalid = True
                        raise
                    self.service = service
                    if self._console_controls is not None:
                        bind_index_service(self._console_controls, self, service)
                    if self._app is not None:
                        self._app._project_index_service = service
            return self.service

    async def aclose(self) -> None:
        def snapshot():
            with _composition_lock:
                self.closed = True
                if self._app is not None:
                    self._app._project_index_closed = True
                return tuple(self._unpublished) + (
                    (self.service,) if self.service is not None else ()
                )

        async def drain():
            services = await asyncio.to_thread(snapshot)
            for service in services:
                unsubscribe = getattr(service, "_unsubscribe", None)
                if unsubscribe is not None:
                    unsubscribe()
                    service._unsubscribe = None
                await service.aclose()
                service._factory.assert_retired()

        if self._close_task is None:
            self._close_task = asyncio.create_task(drain())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError


def get_project_index_owner(app) -> ProjectIndexOwner:
    """Attach the stock owner once; compatibility fields are checked mirrors."""
    with _composition_lock:
        owner = getattr(app, "_project_index_owner", None)
        if owner is not None:
            if type(owner) is not ProjectIndexOwner:
                raise RuntimeError("project_index_owner_changed")
            owner.require_current()
            return owner
        if getattr(app, "_project_index_closed", False):
            raise RuntimeError("project_index_closed")
        owner = ProjectIndexOwner(app.workspace_registry_service)
        existing = getattr(app, "_project_index_service", None)
        if existing is not None and type(existing) is not ProjectIndexService:
            raise TypeError("Custom project index services use legacy composition")
        if existing is not None:
            source = getattr(existing, "_console_preparation_source", ())
            if (
                len(source) != 5
                or source[3] is not owner.registry
                or source[0] is not existing._factory
                or source[1] is not existing._reader
                or source[2] is not existing._fence
            ):
                raise RuntimeError("project_index_owner_changed")
        owner.service = existing
        owner._app = app
        app._project_index_owner = owner
        return owner


def get_project_index_service(app) -> ProjectIndexService:
    """Build once off the UI thread after private data/activation admission."""
    existing = getattr(app, "_project_index_service", None)
    if getattr(app, "_project_index_owner", None) is not None or (
        type(getattr(app, "workspace_registry_service", None))
        is LocalWorkspaceRegistryService
        and (existing is None or type(existing) is ProjectIndexService)
    ):
        return get_project_index_owner(app).get()
    # Custom registries retain their original composition and lifecycle path.
    with _composition_lock:
        if getattr(app, "_project_index_closed", False):
            raise RuntimeError("project_index_closed")
        service = getattr(app, "_project_index_service", None)
        if service is not None:
            return service
        registry = app.workspace_registry_service
        service = _compose_project_index_service(registry)
        service._unsubscribe = registry.subscribe_project_index_changes(
            service.binding_changed
        )
        app._project_index_service = service
        return service


def _compose_project_index_service(registry):
    """Stock registry composition; app attachment and subscriptions stay outside."""
    factory = _AdmittedStoreFactory()
    reader, fence = registry.project_index_authority_adapters(
        factory.current_data_authority
    )
    service = ProjectIndexService(
        factory,
        ProjectScanner(),
        reader,
        fence,
        registry_connections=lambda: project_index_registry_connections(registry.db),
        detached_fence=registry.project_index_detached_fence,
    )
    if (
        type(registry) is LocalWorkspaceRegistryService
        and getattr(registry.project_index_authority_adapters, "__func__", None)
        is _CONSOLE_REGISTRY_ADAPTER[0]
        and _CONSOLE_REGISTRY_ADAPTER[0].__code__ is _CONSOLE_REGISTRY_ADAPTER[1]
    ):
        service._console_preparation_source = (
            factory,
            reader,
            fence,
            registry,
            tuple(
                (function, function.__code__)
                for function in (reader, fence, fence.__wrapped__)
            ),
        )
        service._key_preparation_source = tuple(
            (
                function,
                function.__code__,
                function.__defaults__,
                function.__kwdefaults__,
                function.__closure__,
                tuple(cell.cell_contents for cell in function.__closure__ or ()),
            )
            for function in (
                reader,
                fence,
                fence.__wrapped__,
                service._registry_connections,
            )
        )
    return service


def project_index_key(
    app, workspace_id: str, binding_id: str, root: Path
) -> ProjectIndexKey:
    """Derive one caller-selected binding key off-thread; never select a workspace."""
    service = get_project_index_service(app)
    return _project_index_key_owned(service, workspace_id, binding_id, root)


def _project_index_key_owned(
    service: ProjectIndexService, workspace_id: str, binding_id: str, root: Path
) -> ProjectIndexKey:
    """Derive a key using an already selected project-index service."""
    factory = service._factory
    with service.registry_connection_scope(), factory.retain() as pin, pin.bind():
        chain = capture_directory_chain(root)
        key = ProjectIndexKey(
            pin.identity,
            workspace_id,
            binding_id,
            locator_fingerprint(root),
            root_identity_fingerprint(chain),
        )
        if service._reader(key) is None:
            raise ValueError("index_binding_unavailable")
        return key


def _console_source_current(service, source):
    """Check this composition's small set of skipped/selected reader bindings."""
    if (
        type(service) is not ProjectIndexService
        or source is None
        or getattr(service, "_console_preparation_source", None) is not source
    ):
        return False
    factory, reader, fence, registry, callbacks = source
    if (
        getattr(service, "_factory", None) is not factory
        or type(factory) is not _AdmittedStoreFactory
        or getattr(service, "_reader", None) is not reader
        or getattr(service, "_fence", None) is not fence
        or type(registry) is not LocalWorkspaceRegistryService
        or getattr(
            getattr(registry, "project_index_authority_adapters", None),
            "__func__",
            None,
        )
        is not _CONSOLE_REGISTRY_ADAPTER[0]
        or _CONSOLE_REGISTRY_ADAPTER[0].__code__ is not _CONSOLE_REGISTRY_ADAPTER[1]
        or any(function.__code__ is not code for function, code in callbacks)
        or getattr(fence, "__wrapped__", None) is not callbacks[2][0]
        or capture_directory_chain is not _CONSOLE_CHAIN_READER[0]
        or filesystem_identity.capture_directory_chain is not capture_directory_chain
        or capture_directory_chain.__code__ is not _CONSOLE_CHAIN_READER[1]
        or platform_files.os is not _CONSOLE_OS
        or type(_CONSOLE_OS) is not getattr(platform_files, "WindowsOS", None)
    ):
        return False
    for owner, originals in (
        (service, _CONSOLE_STATUS_READERS),
        (factory, _CONSOLE_FACTORY_READERS),
    ):
        for name, method, body, records in originals:
            if (
                name in vars(owner)
                or getattr(type(owner), name, None) is not method
                or getattr(method, "__wrapped__", method) is not body
            ):
                return False
            if any(
                function.__code__ is not code
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not keywords
                or len(keywords or ()) != len(items)
                or any(
                    name not in keywords or keywords[name] is not value
                    for name, value in items
                )
                for function, code, defaults, keywords, items in records
            ):
                return False
    return all(
        getattr(_CONSOLE_OS, name, None) is method and method.__func__.__code__ is code
        for name, method, code in _CONSOLE_NATIVE_READERS
    )


def _console_missing_index_base(service, key, pin, source):
    """Return only negative advertising evidence; every native birth retires."""
    from tldw_chatbook.Utils.windows_files import _AdmissionMetadataCloseError

    def require_current():
        if (
            not _console_source_current(service, source)
            or pin.closed
            or pin.factory is not service._factory
            or service._factory.data_root != pin.data_root
            or service._factory.current_data_authority() != key.data_authority_id
        ):
            raise ValueError("index_preparation_source_changed")

    require_current()
    root = pin.data_root
    child = root / "project_index"
    paths = (*reversed(root.parents), root, child)
    try:
        # Native namespace/ACL observation owns no registry transaction. The
        # original reader takes its short binding fence only after a miss.
        try:
            observed = _CONSOLE_OS.stat_many_for_admission(paths)
        except PermissionError as error:
            if (
                type(error) is not PermissionError
                or error.errno != errno.EACCES
                or error.strerror != "windows_admission_descriptor_unavailable"
            ):
                raise
            # The existing scalar status supports descriptors that cannot be
            # copied by the optional batch; this grants no absence evidence.
            if capture_directory_chain(root) != pin.chain:
                raise ValueError("index_data_authority_changed") from error
            require_current()
            return False
        require_current()
        if observed[child] is not None:
            if capture_directory_chain(root) != pin.chain:
                raise ValueError("index_data_authority_changed")
            require_current()
            return False
        uid = _CONSOLE_OS.geteuid()
        for parent in paths[:-1]:
            value = observed[parent]
            if value is None:
                raise ValueError("index_data_authority_changed")
            info, _descriptor = value
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid not in (0, uid)
                or info.st_mode & 0o022
                and not info.st_mode & stat.S_ISVTX
            ):
                raise ValueError("index_data_ancestry_unsafe")
        info, _descriptor = observed[root]
        if info.st_uid != uid or info.st_mode & 0o077:
            raise ValueError("index_data_root_not_private")
        if service._reader(key) is None:
            raise ValueError("index_binding_unavailable")
        # Recheck after the reader's possibly waiting registry fence, using
        # retain's original reader, never comparing facade and CPython IDs.
        if capture_directory_chain(root) != pin.chain:
            raise ValueError("index_data_authority_changed")
        require_current()
        return True
    except _AdmissionMetadataCloseError as error:
        # Keep the actual exception and its exact failed_handles. _store_owned
        # keeps this operation's data pin; the outer registry scope marks it.
        service._operation_local.operation.retained.append(error)
        raise ProjectIndexRetirementUnproven(
            "index_metadata_retirement_unproven"
        ) from error


def project_index_status_for_console(
    service: ProjectIndexService, key: ProjectIndexKey
) -> ProjectIndexStatus | None:
    """Use checked missing-base evidence only for the stock Windows runtime."""
    source = getattr(service, "_console_preparation_source", None)
    if not _CONSOLE_NATIVE_READERS or not _console_source_current(service, source):
        return service.status(key)
    return service._console_status(
        key, lambda pin: _console_missing_index_base(service, key, pin, source)
    )


def project_index_management_snapshot_for_display(
    service: ProjectIndexService, key: ProjectIndexKey
) -> tuple[ProjectIndexStatus, ProjectIndexPolicy, bool]:
    """Show a proven unused index without provisioning it on a passive visit."""
    source = getattr(service, "_console_preparation_source", None)

    def current():
        return (
            _CONSOLE_NATIVE_READERS
            and _console_source_current(service, source)
            and definitions_current(_MANAGEMENT_INITIAL_DEFINITIONS)
        )

    if not current():
        return service.management_snapshot(key)

    def require_current():
        if not current():
            raise ValueError("index_preparation_source_changed")

    return service._console_management_snapshot(
        key,
        lambda pin: _console_missing_index_base(service, key, pin, source),
        require_current,
    )


# The absent display mirrors these exact original initial-state semantics.
_MANAGEMENT_INITIAL_DEFINITIONS = capture_definitions(
    (WorkspaceProjectIndexDB, name, getattr(WorkspaceProjectIndexDB, name))
    for name in ("_control", "load_status", "policy", "reconcile_authority")
)


_CONSOLE_CHAIN_READER = capture_directory_chain, capture_directory_chain.__code__
_CONSOLE_REGISTRY_ADAPTER = (
    LocalWorkspaceRegistryService.project_index_authority_adapters,
    LocalWorkspaceRegistryService.project_index_authority_adapters.__code__,
)
_CONSOLE_OS = platform_files.os
_CONSOLE_NATIVE_READERS = (
    tuple(
        (name, method, method.__func__.__code__)
        for name in ("stat_many_for_admission", "_stat_handle", "geteuid")
        for method in (getattr(_CONSOLE_OS, name, None),)
        if type(method) is MethodType
    )
    if _CONSOLE_OS.name == "nt"
    else ()
)
if len(_CONSOLE_NATIVE_READERS) != 3:
    _CONSOLE_NATIVE_READERS = ()
_CONSOLE_FACTORY_READERS = tuple(
    (
        name,
        method,
        body,
        tuple(
            (
                function,
                function.__code__,
                function.__defaults__,
                function.__kwdefaults__,
                tuple((function.__kwdefaults__ or {}).items()),
            )
            for function in ((method, body) if method is not body else (body,))
        ),
    )
    for name in ("retain", "__call__", "current_data_authority")
    for method in (getattr(_AdmittedStoreFactory, name),)
    for body in (getattr(method, "__wrapped__", method),)
)


async def close_project_index_service(app) -> None:
    """Drain actual owners before surfacing cancellation, including composition."""

    def snapshot():
        with _composition_lock:
            app._project_index_closed = True
            return getattr(app, "_project_index_service", None)

    async def drain():
        from .key_process import close_app

        owner = getattr(app, "_project_index_owner", None)
        if type(owner) is ProjectIndexOwner:

            def close_intake():
                with _composition_lock:
                    owner.closed = True
                    app._project_index_closed = True

            await asyncio.to_thread(close_intake)
            await close_app(app)
            await owner.aclose()
            return
        service = await asyncio.to_thread(snapshot)
        await close_app(app)
        if service is not None:
            unsubscribe = getattr(service, "_unsubscribe", None)
            if unsubscribe is not None:
                unsubscribe()
            await service.aclose()
            service._factory.assert_retired()

    task = asyncio.create_task(drain())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


# Definition-time anchors for the fixed key command; never a lazy-import baseline.
_KEY_PROCESS_DEFINITIONS = capture_definitions(
    (
        *(
            (ProjectIndexOwner, name, getattr(ProjectIndexOwner, name))
            for name in ("__init__", "require_current", "get", "aclose")
        ),
        (globals(), "get_project_index_owner", get_project_index_owner),
        (_AdmittedStoreFactory, "__init__", _AdmittedStoreFactory.__init__),
        (_AdmittedStoreFactory, "retain", _AdmittedStoreFactory.retain),
        (
            _AdmittedStoreFactory,
            "current_data_authority",
            _AdmittedStoreFactory.current_data_authority,
        ),
        (_DataAuthorityPin, "__init__", _DataAuthorityPin.__init__),
        (_DataAuthorityPin, "bind", _DataAuthorityPin.bind),
        (_DataAuthorityPin, "close", _DataAuthorityPin.close),
        (_DataAuthorityPin, "__enter__", _DataAuthorityPin.__enter__),
        (_DataAuthorityPin, "__exit__", _DataAuthorityPin.__exit__),
        (ProjectIndexService, "__init__", ProjectIndexService.__init__),
        (
            ProjectIndexService,
            "registry_connection_scope",
            ProjectIndexService.registry_connection_scope,
        ),
        (ProjectIndexKey, "__post_init__", ProjectIndexKey.__post_init__),
        (
            globals(),
            "get_project_index_service",
            globals().get("get_project_index_service"),
        ),
        (
            globals(),
            "_compose_project_index_service",
            globals().get("_compose_project_index_service"),
        ),
        (globals(), "project_index_key", globals().get("project_index_key")),
        (
            globals(),
            "_project_index_key_owned",
            globals().get("_project_index_key_owned"),
        ),
        (
            globals(),
            "project_index_registry_connections",
            globals().get("project_index_registry_connections"),
        ),
        (
            globals(),
            "capture_directory_chain",
            globals().get("capture_directory_chain"),
        ),
        (globals(), "verified_data_identity", globals().get("verified_data_identity")),
        (globals(), "effective_config_path", globals().get("effective_config_path")),
        (globals(), "acquire_storage", globals().get("acquire_storage")),
        (globals(), "execution_scope", globals().get("execution_scope")),
        (globals(), "locator_fingerprint", globals().get("locator_fingerprint")),
        (
            globals(),
            "root_identity_fingerprint",
            globals().get("root_identity_fingerprint"),
        ),
    )
)

_KEY_PROCESS_GLOBALS = globals()
_KEY_PROCESS_DEFINITIONS += (
    (globals(), "_AdmittedStoreFactory", _AdmittedStoreFactory, ()),
    (globals(), "_DataAuthorityPin", _DataAuthorityPin, ()),
    (globals(), "ProjectIndexService", ProjectIndexService, ()),
    (globals(), "ProjectIndexKey", ProjectIndexKey, ()),
)
