"""Lazy application composition after admitted private data authority capture."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import threading

from tldw_chatbook.Backup_Recovery.activation import execution_scope
from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
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
    locator_fingerprint,
    root_identity_fingerprint,
)
from .scanner import ProjectScanner
from .service import ProjectIndexService, binding_bytes


_composition_lock = threading.Lock()


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
            self.closed = True
            self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _AdmittedStoreFactory:
    def __init__(self):
        self._local = threading.local()
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

    def __call__(self, key):
        if self.current_data_authority() != key.data_authority_id:
            raise ValueError("index_data_authority_unavailable")
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
                return WorkspaceProjectIndexDB(path)
            finally:
                lease.close()
        return WorkspaceProjectIndexDB(path)


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
        service = ProjectIndexService(factory, ProjectScanner(), reader, fence)
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
    with factory.retain() as pin, pin.bind():
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
    """Close an existing owner without constructing one or requiring a Console."""
    with _composition_lock:
        app._project_index_closed = True
        service = getattr(app, "_project_index_service", None)
    if service is not None:
        unsubscribe = getattr(service, "_unsubscribe", None)
        if unsubscribe is not None:
            unsubscribe()
        await service.aclose()
