"""Lazy application composition after admitted private data authority capture."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
import threading

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
from tldw_chatbook.Utils.private_paths import secure_private_directory, PrivatePathError
from .lifetime import BindingLifetime
from .models import (
    MAX_CACHE_BYTES,
    OWNER_ID,
    ProjectIndexKey,
    ProjectIndexRetirementUnproven,
    locator_fingerprint,
    root_identity_fingerprint,
)
from .scanner import ProjectScanner
from .service import ProjectIndexService, binding_bytes


_composition_lock = threading.Lock()


@contextmanager
def project_index_registry_connections(database):
    """Retire newly owned registry handles with positive native/core proof.

    This finite context grants no authority. Borrowed caller handles keep their
    own lifetime; a failed newly owned close remains counted by storage admission.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Backup_Recovery.participants import _core_cached_connection

    if type(database) is not WorkspaceDB or database.is_memory_db:
        with operation_owned_connection(database):
            yield
        return
    previous = _core_cached_connection(
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
            retired = _core_cached_connection(database, owned) is None
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

    def __init__(self, factory, stack, identity, data_root):
        self.factory = factory
        self.stack = stack
        self.identity = identity
        self.data_root = data_root
        self.closed = False

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
            return _DataAuthorityPin(self, stack, identity, data)
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


def get_project_index_service(app) -> ProjectIndexService:
    """Build once off the UI thread after private data/activation admission."""
    with _composition_lock:
        if getattr(app, "_project_index_closed", False):
            raise RuntimeError("project_index_closed")
        service = getattr(app, "_project_index_service", None)
        if service is not None:
            # No idle pin: every later operation revalidates its own authority.
            return service
        factory = _AdmittedStoreFactory()
        registry = app.workspace_registry_service
        reader, fence = registry.project_index_authority_adapters(
            factory.current_data_authority
        )
        service = ProjectIndexService(
            factory,
            ProjectScanner(),
            reader,
            fence,
            registry_connections=lambda: project_index_registry_connections(
                registry.db
            ),
            detached_fence=registry.project_index_detached_fence,
        )
        service._unsubscribe = registry.subscribe_project_index_changes(
            service.binding_changed
        )
        app._project_index_service = service
        return service


def project_index_key(
    app, workspace_id: str, binding_id: str, root: Path
) -> ProjectIndexKey:
    """Derive one caller-selected binding key off-thread; never select a workspace."""
    service = get_project_index_service(app)
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


async def close_project_index_service(app) -> None:
    """Drain actual owners before surfacing cancellation, including composition."""

    def snapshot():
        with _composition_lock:
            app._project_index_closed = True
            return getattr(app, "_project_index_service", None)

    async def drain():
        service = await asyncio.to_thread(snapshot)
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
