"""One application-owned, explicitly started writer for folder-bound indexes."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext, ExitStack
from dataclasses import asdict, dataclass, field, replace
import json
import hashlib
import os
import re
import sqlite3
import stat
from pathlib import Path
import threading
import time
from uuid import uuid4

from tldw_chatbook.DB.Workspace_Project_Index_DB import (
    prepare_scan_unit,
    WorkspaceProjectIndexDB,
    _path_allowed,
)
from .allocation import allocation_gate, register_owned_lifetime, verify_settlement
from .lifetime import BindingLifetime, LifetimeBusy
from .models import (
    MAX_CACHE_BYTES,
    IndexAuthorityFence,
    IndexAuthorityReader,
    ProjectIndexJobClaim,
    ProjectIndexKey,
    ProjectIndexPolicy,
    ProjectIndexScope,
    ProjectIndexStatus,
    ProjectIndexSearchAuthority,
    ProjectIndexHit,
    ProjectSearchResult,
    ProjectIndexRetirementUnproven,
)


@dataclass
class _Run:
    claim: ProjectIndexJobClaim
    stop: threading.Event
    writer: threading.Thread | None = None
    supervisor: threading.Thread | None = None
    error: BaseException | None = None
    state: str = "Interrupted"
    db_closed: bool = False
    registry_closed: bool = False
    native_retired: bool = False
    pin: object | None = None
    lease: BindingLifetime | None = None
    read_admission: ExitStack | None = None


@dataclass
class _SemanticRun:
    job_id: str
    scope: ProjectIndexScope
    state: object
    authority: object
    stop: threading.Event
    thread: threading.Thread | None = None
    lease: BindingLifetime | None = None
    pin: object | None = None
    read_admission: ExitStack | None = None
    error: BaseException | None = None
    generation_id: str | None = None
    native_retired: bool = False
    close_pending: bool = False
    close_loop: asyncio.AbstractEventLoop | None = None
    close_signal: asyncio.Event | None = None
    store: object | None = None


@dataclass
class _SemanticQuery:
    key: ProjectIndexKey
    signal: threading.Event
    close_pending: bool = False


@dataclass(eq=False)
class _SearchRun:
    key: ProjectIndexKey
    done: threading.Event
    pin: object | None = None
    store: object | None = None
    retirement_unproven: bool = False


@dataclass(eq=False)
class _Operation:
    done: threading.Event
    retirement_unproven: bool = False
    retained: list[object] = field(default_factory=list)
    projection_close_pending: dict[object, tuple[ProjectIndexKey, object]] = field(
        default_factory=dict
    )
    retire_registry_for_model: object | None = None


def binding_bytes(directory: Path) -> int:
    """Count every owned namespace/sidecar, with bounded traversal and no aliases."""
    total = count = 0
    pending = [directory]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                if count > 4096:
                    return MAX_CACHE_BYTES
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError("cache_alias_refused")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    raise ValueError("cache_alias_refused")
    return total


_NO_STOP_RESULT = object()


class ProjectIndexService:
    """Synchronous management runs off the UI thread; only aclose is asynchronous.

    A service serializes its supplied scanner. Multiple applications coordinate
    with the stable binding capability and durable registry/store CAS.
    """

    def __init__(
        self,
        store_factory,
        scanner,
        authority_reader: IndexAuthorityReader,
        authority_fence: IndexAuthorityFence,
        *,
        registry_connections=nullcontext,
        detached_fence=None,
    ):
        self._factory = store_factory
        self._scanner = scanner
        self._reader = authority_reader
        self._fence = authority_fence
        self._registry_connections = registry_connections
        self._detached_fence = detached_fence
        self._cleanup_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._runs: dict[str, _Run] = {}
        self._semantic_runs: dict[str, _SemanticRun] = {}
        self._semantic_query: _SemanticQuery | None = None
        self._semantic_query_slot = threading.BoundedSemaphore(1)
        self._closed = False
        self._operations: set[_Operation] = set()
        self._operation_local = threading.local()
        self._searches: set[_SearchRun] = set()
        self._read_slots = threading.BoundedSemaphore(2)
        self._pending_reads: dict[str, int] = {}

    @contextmanager
    def registry_connection_scope(self):
        """Admit a complete finite operation through its last native registry close.

        Nested calls borrow the physical thread's admission and registry owner.
        This lifecycle record grants no data, source or tool authority.
        """
        if getattr(self._operation_local, "operation", None) is not None:
            yield
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("project_index_closed")
            if any(operation.retirement_unproven for operation in self._operations):
                raise ProjectIndexRetirementUnproven(
                    "index_operation_retirement_unproven"
                )
            operation = _Operation(threading.Event())
            self._operations.add(operation)
            self._operation_local.operation = operation
        try:
            with self._registry_connections() as retire_registry:
                operation.retire_registry_for_model = retire_registry
                yield
        except ProjectIndexRetirementUnproven as error:
            # A retained semantic model-close owner is known and separately
            # rooted. This management operation can still retire its own SQL.
            if str(error) != "project_embedding_close_pending":
                operation.retirement_unproven = True
            raise
        finally:
            self._operation_local.operation = None
            with self._lock:
                if not operation.retirement_unproven:
                    self._operations.discard(operation)
            operation.done.set()

    def _retire_registry_for_model(self) -> None:
        """Retire this operation's owned registry SQL before model effects."""
        operation = getattr(self._operation_local, "operation", None)
        if operation is None:
            raise ProjectIndexRetirementUnproven("index_operation_owner_missing")
        retire = operation.retire_registry_for_model
        if retire is not None:
            retire()

    def _retain_uncertain_model(self, *owners: object) -> None:
        """Keep actual owners when native registry retirement cannot be proved."""
        operation = getattr(self._operation_local, "operation", None)
        if operation is None:
            raise ProjectIndexRetirementUnproven("index_operation_owner_missing")
        operation.retirement_unproven = True
        operation.retained.extend(owner for owner in owners if owner is not None)

    def _require_projection_open(self, key: ProjectIndexKey, *, retiring=None) -> None:
        """Fence new binding work and retirement of its actual pending parent."""
        with self._lock:
            pending = (
                bool(retiring._parent.operation.projection_close_pending)
                if retiring is not None
                else any(
                    selected == key
                    for operation in self._operations
                    for selected, _ in operation.projection_close_pending.values()
                )
            )
        if pending:
            from tldw_chatbook.RAG_Search.projection_scope import (
                ProjectProjectionRequired,
            )

            raise ProjectProjectionRequired("project_embedding_close_pending")

    def _mark_projection_close_pending(
        self,
        operation: _Operation,
        key: ProjectIndexKey,
        token: object,
        wrapper: object,
    ) -> None:
        """Root the actual wrapper/factory under its accepted operation before close."""
        with self._lock:
            if operation not in self._operations or operation.done.is_set():
                raise RuntimeError("project_embedding_close_parent_unavailable")
            existing = operation.projection_close_pending.get(token)
            if existing is not None and existing != (key, wrapper):
                raise RuntimeError("project_embedding_close_parent_unavailable")
            operation.projection_close_pending[token] = (key, wrapper)

    def _settle_projection_close_pending(
        self, operation: _Operation, token: object, wrapper: object
    ) -> None:
        """Release only the matching actual owner after a positive factory close."""
        with self._lock:
            existing = operation.projection_close_pending.get(token)
            if existing is None or existing[1] is not wrapper:
                raise RuntimeError("project_embedding_close_parent_unavailable")
            del operation.projection_close_pending[token]

    def _projection_close_pending(self, operation: _Operation) -> bool:
        with self._lock:
            return bool(operation.projection_close_pending)

    def retry_semantic_close(self, key: ProjectIndexKey) -> bool:
        """Signal only an exact binding's retained model owner to retry close."""
        if type(key) is not ProjectIndexKey:
            return False
        signalled = False
        with self._lock:
            for run in self._semantic_runs.values():
                if (
                    run.scope.key == key and run.close_pending
                    and run.close_loop is not None and run.close_signal is not None
                ):
                    run.close_loop.call_soon_threadsafe(run.close_signal.set)
                    signalled = True
            query = self._semantic_query
            if query is not None and query.key == key and query.close_pending:
                query.signal.set()
                signalled = True
        return signalled

    def semantic_close_pending(self, key: ProjectIndexKey) -> bool:
        """Project exact-binding retained close state without opening a model/DB."""
        if type(key) is not ProjectIndexKey:
            return False
        with self._lock:
            return (
                (
                    self._semantic_query is not None
                    and self._semantic_query.key == key
                    and self._semantic_query.close_pending
                )
                or any(
                    run.scope.key == key and run.close_pending
                    for run in self._semantic_runs.values()
                )
                or any(
                    selected == key
                    for operation in self._operations
                    for selected, _ in operation.projection_close_pending.values()
                )
            )

    def _admission_closed(self):
        return (
            self._closed and getattr(self._operation_local, "operation", None) is None
        )

    @contextmanager
    def _store(self, key, *, cleanup_only=False):
        with self.registry_connection_scope(), self._store_owned(
            key, cleanup_only=cleanup_only
        ) as store:
            yield store

    @contextmanager
    def _store_owned(self, key, *, cleanup_only=False):
        pin = self._retain(key)
        retired = True
        try:
            with self._bind(pin):
                with self._opened_store(key, cleanup_only=cleanup_only) as store:
                    yield store
        except ProjectIndexRetirementUnproven:
            retired = False
            self._operation_local.operation.retained.append(pin)
            raise
        finally:
            if pin is not None and retired:
                self._close_capability(pin)

    def _retain(self, key):
        retain = getattr(self._factory, "retain", None)
        return retain(key) if retain is not None else None

    @staticmethod
    def _bind(pin):
        return pin.bind() if pin is not None else nullcontext()

    @contextmanager
    def _opened_store(self, key, *, cleanup_only=False):
        if self._admission_closed():
            raise RuntimeError("project_index_closed")
        opener = (
            getattr(self._factory, "open_cleanup", self._factory)
            if cleanup_only
            else self._factory
        )
        store = opener(key)
        try:
            store.status_cache_bytes = binding_bytes(self._binding_root(store))
            yield store
        finally:
            try:
                store.close_native()
            except BaseException as error:
                operation = self._operation_local.operation
                operation.retirement_unproven = True
                operation.retained.append(store)
                raise ProjectIndexRetirementUnproven(
                    "index_store_retirement_unproven"
                ) from error

    @staticmethod
    def _binding_root(store):
        return store.db_path.parent.parent

    def _lifetime(self, store, key, *, existing_only=False):
        lease = BindingLifetime.acquire(
            self._binding_root(store) / key.lifetime_name, existing_only=existing_only
        )
        store.bind_native_lifetime(lease)
        return lease

    @staticmethod
    def _claim_contract(claim, key):
        if claim is not None and not claim.owner_token.startswith(
            "native-v1:" + key.binding_namespace + ":"
        ):
            raise ValueError("legacy_retirement_unproven")

    @contextmanager
    def _admitted(self, store, key):
        # The allocation gate precedes the registry fence and retires with SQL.
        with store.sql_unit(write=True), self._fence(key) as authority:
            if authority is None:
                old = store.load_status(key)
                if old.state != "Off":
                    store.revoke(key, clear=False, expected_scope=old.scope)
                yield None
            else:
                store.reconcile_authority(key, authority.authority_revision)
                yield authority

    @contextmanager
    def _read_fenced(self, store, key):
        with store.sql_unit(), self._fence(key) as authority:
            yield authority

    @staticmethod
    def _fresh_status(key):
        return ProjectIndexStatus(
            ProjectIndexScope(key, 0, 0), "Off", 0, 0, 0, None, None, 0, 0
        )

    def _retry_read(self, key, operation):
        """Retry a transient gate miss after unwinding and signal a local reader."""
        deadline = time.monotonic() + 3.0
        group = key.binding_namespace
        pending = False
        try:
            while True:
                try:
                    return operation()
                except LifetimeBusy:
                    if not pending:
                        with self._lock:
                            if self._closed:
                                raise RuntimeError("project_index_closed")
                            self._pending_reads[group] = (
                                self._pending_reads.get(group, 0) + 1
                            )
                        pending = True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    time.sleep(min(0.005, remaining))
        finally:
            if pending:
                with self._lock:
                    count = self._pending_reads[group] - 1
                    if count:
                        self._pending_reads[group] = count
                    else:
                        del self._pending_reads[group]

    def status(self, key: ProjectIndexKey) -> ProjectIndexStatus:
        def read():
            try:
                with self._store(key, cleanup_only=True) as store:
                    return self._status_from_store(store, key)
            except FileNotFoundError:
                return self._fresh_status(key)

        return self._retry_read(key, read)

    def semantic_state(self, key: ProjectIndexKey):
        """Read separate consent/readiness without importing a model backend."""
        from .models import ProjectSemanticState

        def read():
            try:
                with self._store(key, cleanup_only=True) as store:
                    with self._read_fenced(store, key) as authority:
                        if authority is None:
                            raise ValueError("authority_unavailable")
                        status = store.load_status(key)
                        return store.load_semantic_state(status.scope)
            except FileNotFoundError:
                return ProjectSemanticState(self._fresh_status(key).scope, 0, False, None, "Off")

        return replace(
            self._retry_read(key, read),
            close_pending=self.semantic_close_pending(key),
        )

    def building_coverage(self, scope: ProjectIndexScope, exclusions=()) -> bool:
        """Check committed compatibility without granting a search authority."""
        try:
            with self._store(scope.key, cleanup_only=True) as store:
                with self._read_fenced(store, scope.key) as live:
                    if live is None:
                        return False
                    status = store.load_status(scope.key)
                    return (
                        status.scope == scope
                        and status.state in {"Building", "Partial", "Ready"}
                        and store.has_compatible_files(
                            scope, tuple(set(exclusions + live.exclusions))
                        )
                    )
        except (FileNotFoundError, LifetimeBusy):
            return False

    def management_snapshot(
        self, key: ProjectIndexKey
    ) -> tuple[ProjectIndexStatus, ProjectIndexPolicy, bool]:
        """Read exact management state without upgrading a validated legacy cache."""

        def read():
            try:
                with self._store(key, cleanup_only=True) as store:
                    status = self._status_from_store(store, key)
                    with self._read_fenced(store, key) as authority:
                        if authority is None:
                            raise ValueError("authority_unavailable")
                        policy = store.policy(status.scope)
                    return (
                        status,
                        policy,
                        store.schema_version < store._CURRENT_SCHEMA_VERSION,
                    )
            except FileNotFoundError:
                return self._fresh_status(key), ProjectIndexPolicy((), ()), False

        return self._retry_read(key, read)

    def _status_from_store(self, store, key):
        with self._read_fenced(store, key) as authority:
            result = store.load_status(key)
            claim = store.read_claim(result.scope)
        if authority is None:
            result = replace(result, state="Off", job_id=None)
        # Physical retirement makes a durable Building claim inert. Project the
        # status without writing control on an ordinary status query.
        run = self._runs.get(result.job_id)
        if (
            claim is not None
            and result.state == "Building"
            and (run is None or run.native_retired)
        ):
            try:
                lease = self._lifetime(store, key, existing_only=True)
            except LifetimeBusy:
                pass
            except FileNotFoundError:
                result = replace(result, state="Interrupted")
            else:
                try:
                    result = replace(result, state="Interrupted")
                finally:
                    self._close_capability(lease)
        return replace(result, cache_bytes=binding_bytes(self._binding_root(store)))

    def policy(self, scope: ProjectIndexScope) -> ProjectIndexPolicy:
        """Read saved processing choices only under the exact admitted scope."""

        def read():
            with self._store(scope.key, cleanup_only=True) as store:
                with self._read_fenced(store, scope.key) as authority:
                    if authority is None:
                        raise ValueError("authority_unavailable")
                    return store.policy(scope)

        return self._retry_read(scope.key, read)

    @contextmanager
    def _search_store(self, key):
        # Writer and Capture share two read slots; failed native retirement keeps its
        # operation admission and slot until the application is recovered.
        if not self._read_slots.acquire(blocking=False):
            raise RuntimeError("project_index_search_busy")
        with self._lock:
            if self._admission_closed() or any(
                run.key == key and run.retirement_unproven for run in self._searches
            ):
                self._read_slots.release()
                raise RuntimeError("project_index_search_unavailable")
            run = _SearchRun(key, threading.Event())
            self._searches.add(run)
        try:
            run.pin = self._retain(key)
            with self._bind(run.pin):
                opener = getattr(self._factory, "open_cleanup", self._factory)
                store = opener(key)
                run.store = store
                try:
                    store.status_cache_bytes = binding_bytes(self._binding_root(store))
                    yield store
                finally:
                    try:
                        store.close_native()
                    except BaseException as error:
                        raise ProjectIndexRetirementUnproven(
                            "index_store_retirement_unproven"
                        ) from error
        except ProjectIndexRetirementUnproven:
            run.retirement_unproven = True
            raise
        finally:
            try:
                if not run.retirement_unproven:
                    if run.pin is not None:
                        self._close_capability(run.pin)
                    with self._lock:
                        self._searches.discard(run)
                    self._read_slots.release()
            except ProjectIndexRetirementUnproven:
                run.retirement_unproven = True
                raise
            finally:
                run.done.set()

    def search(
        self, authority: ProjectIndexSearchAuthority, query: str, *, limit: int = 10
    ) -> ProjectSearchResult:
        """Return bounded evidence from live bytes under this run's authority.

        Args:
            authority: Code-issued run root, exclusion floor and pinned reader.
            query: Nonblank keyword query, at most 2048 characters.
            limit: Strict integer result limit from one through ten.

        Returns:
            Live file evidence, or an unavailable/partial result. Coverage is
            deliberately absent: management counters are not run-scoped.
        """
        try:
            if type(authority) is not ProjectIndexSearchAuthority:
                raise ValueError("invalid_search_authority")
            if not isinstance(query, str) or not query.strip() or len(query) > 2048:
                raise ValueError("invalid_query")
            if type(limit) is not int or not 1 <= limit <= 10:  # noqa: E721
                raise ValueError("invalid_limit")
            try:
                semantic_query = self._semantic_query_vector(authority, query)
            except ProjectIndexRetirementUnproven:
                raise
            except Exception:
                semantic_query = None
            with self.registry_connection_scope():
                return self._search_owned(
                    authority, query, limit=limit, semantic_query=semantic_query
                )
        except LifetimeBusy:
            return ProjectSearchResult("unavailable", (), None, False)

    def _semantic_query_vector(self, authority, query):
        """Project once on this finite tool worker, before acquiring a read slot."""
        if not authority.guard() or not self._semantic_query_slot.acquire(blocking=False):
            return None
        owned = _SemanticQuery(authority.scope.key, threading.Event())
        with self._lock:
            self._semantic_query = owned
        try:
            with self.registry_connection_scope():
                state = self.semantic_state(authority.scope.key)
                if (
                    state.scope != authority.scope or not state.enabled
                    or state.state != "Ready" or state.settings is None
                ):
                    return None
                from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import (
                    EmbeddingsServiceWrapper,
                )
                from . import semantic

                try:
                    carrier = semantic._issue_project_projection_scope(
                        self, authority.scope, state.settings, state.semantic_revision
                    )
                except ProjectIndexRetirementUnproven:
                    raise
                except Exception:
                    return None
                wrapper = None
                vector = None
                try:
                    with self._store(authority.scope.key, cleanup_only=True) as store:
                        with self._read_fenced(store, authority.scope.key) as live:
                            ready = (
                                live is not None
                                and live.root == authority.root
                                and live.chain == authority.chain
                                and store.semantic_ready_generation(
                                    authority.scope, state,
                                    carrier.artifact_fingerprint,
                                ) is not None
                            )
                except ProjectIndexRetirementUnproven:
                    self._retain_uncertain_model(carrier)
                    raise
                except BaseException:
                    try:
                        carrier.close()
                    except BaseException as error:
                        self._retain_uncertain_model(carrier)
                        raise ProjectIndexRetirementUnproven(
                            "index_projection_retirement_unproven"
                        ) from error
                    return None
                try:
                    self._retire_registry_for_model()
                except ProjectIndexRetirementUnproven:
                    self._retain_uncertain_model(carrier)
                    raise
                except BaseException:
                    try:
                        carrier.close()
                    except BaseException as error:
                        self._retain_uncertain_model(carrier)
                        raise ProjectIndexRetirementUnproven(
                            "index_projection_retirement_unproven"
                        ) from error
                    raise
                if not ready:
                    try:
                        carrier.close()
                    except BaseException as error:
                        self._retain_uncertain_model(carrier)
                        raise ProjectIndexRetirementUnproven(
                            "index_projection_retirement_unproven"
                        ) from error
                    return None
                try:
                    wrapper = EmbeddingsServiceWrapper(
                        model_name=carrier.model_name,
                        api_key=carrier.api_key,
                        base_url=carrier.base_url,
                        device=getattr(carrier, "_config", {}).get("device"),
                        execution_scope=carrier,
                        local_files_only=carrier.local_files_only,
                    )
                    try:
                        vector = wrapper.create_embedding(query)
                    except Exception:
                        pass
                finally:
                    try:
                        self._retire_registry_for_model()
                    except BaseException as error:
                        self._retain_uncertain_model(wrapper, carrier)
                        if isinstance(error, ProjectIndexRetirementUnproven):
                            raise
                        raise ProjectIndexRetirementUnproven(
                            "index_registry_retirement_unproven"
                        ) from error
                    if wrapper is not None:
                        try:
                            wrapper.close()
                        except BaseException:
                            # One immediate same-owner retry before explicit signals.
                            try:
                                wrapper.close()
                            except BaseException:
                                with self._lock:
                                    owned.close_pending = True
                                while True:
                                    owned.signal.wait()
                                    owned.signal.clear()
                                    try:
                                        wrapper.close()
                                    except BaseException:
                                        continue
                                    with self._lock:
                                        owned.close_pending = False
                                    break
                    try:
                        carrier.close()
                    except BaseException as error:
                        self._retain_uncertain_model(carrier)
                        raise ProjectIndexRetirementUnproven(
                            "index_projection_retirement_unproven"
                        ) from error
                if vector is not None:
                    return state, carrier.artifact_fingerprint, vector
                return None
        finally:
            with self._lock:
                if self._semantic_query is owned:
                    self._semantic_query = None
            self._semantic_query_slot.release()

    def _search_owned(self, authority, query, *, limit, semantic_query=None):
        if type(authority) is not ProjectIndexSearchAuthority:
            raise ValueError("invalid_search_authority")
        if not isinstance(query, str) or not query.strip() or len(query) > 2048:
            raise ValueError("invalid_query")
        if type(limit) is not int or not 1 <= limit <= 10:  # noqa: E721 - reject bool
            raise ValueError("invalid_limit")
        unavailable = ProjectSearchResult("unavailable", (), None, False)
        if not authority.guard():
            return unavailable
        with self._search_store(authority.scope.key) as store:
            with store.sql_unit(), self._fence(authority.scope.key) as live:
                if (
                    live is None
                    or live.root != authority.root
                    or live.chain != authority.chain
                ):
                    return unavailable
                status = store.load_status(authority.scope.key)
                if (
                    status.scope != authority.scope
                    or status.state
                    not in {
                        "Ready",
                        "Partial",
                        "Building",
                    }
                    or (status.state == "Building" and not status.indexed)
                ):
                    return unavailable
                admitted = replace(
                    authority,
                    exclusions=tuple(set(authority.exclusions + live.exclusions)),
                    authority_revision=live.authority_revision,
                )
                if status.state == "Building" and not store.has_compatible_files(
                    admitted.scope, admitted.exclusions
                ):
                    return unavailable
                candidates = store.lexical_candidates(admitted, query, limit=100)
            semantic_status = "unavailable"
            semantic_generation = None
            if semantic_query is not None:
                semantic_state, artifact, vector = semantic_query
                from . import semantic

                try:
                    spec, destination = semantic._require_current_catalog(
                        semantic_state.settings
                    )
                    if (
                        semantic._target(spec, destination).artifact_fingerprint
                        == artifact
                    ):
                        with store.sql_unit(), self._fence(
                            authority.scope.key
                        ) as current:
                            if current == live:
                                ranked = semantic.rank_semantic(
                                    store, admitted.scope, semantic_state, artifact,
                                    vector, admitted.exclusions,
                                )
                                if ranked is not None:
                                    semantic_generation = (
                                        store.semantic_ready_generation(
                                            admitted.scope, semantic_state, artifact
                                        )
                                    )
                                    semantic_status = "ready"
                                    candidates = semantic.fuse_candidates(candidates, ranked)
                except (ValueError, ImportError, OSError):
                    pass
            hits = []
            stale = False
            for candidate in candidates:
                if not authority.guard():
                    return unavailable
                # No worker read while holding a registry transaction. The
                # witness is checked again before accepting any disclosure.
                with store.sql_unit(), self._fence(authority.scope.key) as current:
                    current_status = store.load_status(authority.scope.key)
                    if (
                        current != live
                        or current_status.scope != status.scope
                        or current_status.state not in {"Ready", "Partial", "Building"}
                        or current_status.manifest_revision != status.manifest_revision
                        or (
                            semantic_generation is not None
                            and store.semantic_ready_generation(
                                admitted.scope, semantic_state, artifact
                            )
                            != semantic_generation
                        )
                        or not _path_allowed(
                            candidate.relative_path, admitted.exclusions
                        )
                    ):
                        return unavailable
                capture = authority.live_reader(candidate)
                if not authority.guard():
                    return unavailable
                if (
                    capture is None
                    or capture.relative_path != candidate.relative_path
                    or capture.identity != candidate.file_identity
                    or capture.content_sha256 != candidate.content_sha256
                ):
                    stale = True
                    continue
                lines = capture.text.splitlines()
                excerpt = "\n".join(
                    lines[candidate.line_start - 1 : candidate.line_end]
                )
                # A single long line may exceed the wire budget; retain its
                # honest line location while bounding the evidence text.
                excerpt = excerpt.encode("utf-8")[:12000].decode(
                    "utf-8", errors="ignore"
                )
                hit = ProjectIndexHit(
                    candidate.relative_path,
                    candidate.line_start,
                    min(candidate.line_end, candidate.line_start + excerpt.count("\n")),
                    excerpt,
                )
                proposed = ProjectSearchResult(
                    "partial" if stale or status.state != "Ready" else "ready",
                    tuple(hits + [hit]),
                    None,
                    stale,
                    semantic_status,
                )
                if (
                    len(
                        json.dumps(
                            asdict(proposed), ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    )
                    > 16384
                ):
                    break
                hits.append(hit)
                if len(hits) >= limit:
                    break
            if not authority.guard():
                return unavailable
            with store.sql_unit(), self._fence(authority.scope.key) as current:
                final = store.load_status(authority.scope.key)
                if (
                    current != live
                    or final.scope != status.scope
                    or final.manifest_revision != status.manifest_revision
                    or final.state not in {"Ready", "Partial", "Building"}
                    or (
                        semantic_generation is not None
                        and store.semantic_ready_generation(
                            admitted.scope, semantic_state, artifact
                        )
                        != semantic_generation
                    )
                ):
                    return unavailable
                result = ProjectSearchResult(
                    "partial" if stale or final.state != "Ready" else "ready",
                    tuple(hits),
                    None,
                    stale,
                    semantic_status,
                )
                # A later stale candidate enlarges the final wrapper. Trim
                # whole hits only, preserving valid JSON and honest evidence.
                while (
                    len(
                        json.dumps(
                            asdict(result), ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    )
                    > 16384
                ):
                    result = replace(result, hits=result.hits[:-1])
                return result

    def enable(
        self,
        key: ProjectIndexKey,
        policy: ProjectIndexPolicy,
        *,
        expected_scope: ProjectIndexScope,
    ) -> ProjectIndexStatus:
        with self._lock, self._store(key) as store:
            lease = self._lifetime(store, key)
            try:
                with self._admitted(store, key) as enrolled:
                    if enrolled is None:
                        raise ValueError("authority_unavailable")
                self._enroll_cleanup_owner(store, key, enrolled)
                with self._admitted(store, key) as authority:
                    if authority != enrolled:
                        raise ValueError("authority_changed")
                    store.assert_native_lifetime()
                    if authority is None:
                        raise ValueError("authority_unavailable")
                    self._claim_contract(store.read_claim(expected_scope), key)
                    store.configure(
                        key, policy, enabled=True, expected_scope=expected_scope
                    )
                    return store.load_status(key)
            finally:
                self._close_capability(lease)

    def enable_semantic(
        self, key, settings, *, expected_epoch: int, expected_semantic_revision: int
    ):
        """CAS separate semantic consent against current safe catalog identity."""
        from .models import ProjectSemanticSettings
        from .semantic import _settings_for_project_model

        if type(settings) is not ProjectSemanticSettings:
            raise ValueError("invalid_semantic_settings")
        candidate = _settings_for_project_model(settings.model_id)
        if replace(settings, approved_destination_fingerprint=None) != candidate:
            raise ValueError("project_embedding_configuration_changed")
        return self._configure_semantic(
            key, settings, enabled=True, expected_epoch=expected_epoch,
            expected_semantic_revision=expected_semantic_revision,
        )

    def disable_semantic(
        self, key, *, expected_epoch: int, expected_semantic_revision: int
    ):
        """Revoke semantic work without changing keyword consent or policy."""
        return self._configure_semantic(
            key, None, enabled=False, expected_epoch=expected_epoch,
            expected_semantic_revision=expected_semantic_revision,
        )

    def _configure_semantic(
        self, key, settings, *, enabled, expected_epoch, expected_semantic_revision
    ):
        if type(expected_epoch) is not int or expected_epoch < 0:  # noqa: E721
            raise ValueError("invalid_semantic_epoch")
        with self._store(key) as store:
            own_lease = None
            with self._lock:
                active = next(
                    (
                        run for run in self._semantic_runs.values()
                        if run.scope.key == key and not run.native_retired
                    ), None,
                )
                if active is None:
                    own_lease = self._lifetime(store, key)
                else:
                    store.bind_native_lifetime(active.lease)
                try:
                    with self._admitted(store, key) as authority:
                        if authority is None:
                            raise ValueError("authority_unavailable")
                        status = store.load_status(key)
                        if status.scope.epoch != expected_epoch:
                            raise ValueError("stale_scope")
                        result = store.configure_semantic(
                            status.scope, settings, enabled=enabled,
                            expected_semantic_revision=expected_semantic_revision,
                        )
                    affected = tuple(
                        run for run in self._semantic_runs.values()
                        if run.scope.key == key and not run.native_retired
                    )
                    for run in affected:
                        run.stop.set()
                finally:
                    if own_lease is not None:
                        self._close_capability(own_lease)
            self._stop_semantic_local(affected)
            return result

    def _stop_semantic_local(self, runs):
        for run in runs:
            run.stop.set()
        for run in runs:
            while run.thread is not None and run.thread.is_alive():
                if run.close_pending:
                    raise ProjectIndexRetirementUnproven(
                        "project_embedding_close_pending"
                    )
                run.thread.join(0.1)
            if not run.native_retired:
                raise RuntimeError("index_retirement_unproven") from run.error

    def refresh_semantic(
        self, key, *, expected_epoch: int, expected_semantic_revision: int
    ) -> str:
        """Start one finite off-UI async generation job for this binding."""
        with self.registry_connection_scope():
            if not self._read_slots.acquire(blocking=False):
                raise LifetimeBusy("index_reads_busy")
            with ExitStack() as read_admission:
                read_admission.callback(self._read_slots.release)
                pin = self._retain(key)
                store = lease = None
                dispatched = False
                try:
                    with self._bind(pin), self._lock:
                        if any(not item.native_retired for item in self._runs.values()):
                            raise LifetimeBusy("scanner_busy")
                        if any(
                            not item.native_retired
                            and item.scope.key.binding_namespace == key.binding_namespace
                            for item in self._semantic_runs.values()
                        ):
                            raise LifetimeBusy("semantic_writer_busy")
                        store = self._factory(key)
                        store.status_cache_bytes = binding_bytes(self._binding_root(store))
                        lease = self._lifetime(store, key)
                        with self._admitted(store, key) as authority:
                            if authority is None:
                                raise ValueError("authority_unavailable")
                            status = store.load_status(key)
                            state = store.load_semantic_state(status.scope)
                            if (
                                status.scope.epoch != expected_epoch
                                or status.state not in {"Ready", "Partial"}
                                or not state.enabled
                                or state.semantic_revision != expected_semantic_revision
                                or state.settings is None
                                or state.settings.approved_destination_fingerprint
                                != state.settings.destination_fingerprint
                            ):
                                raise ValueError("stale_semantic_configuration")
                        self._enroll_cleanup_owner(store, key, authority)
                        store.close_native()
                        job_id = uuid4().hex
                        run = _SemanticRun(
                            job_id, status.scope, state, authority,
                            threading.Event(), lease=lease, pin=pin,
                            read_admission=read_admission.pop_all(),
                        )
                        self._semantic_runs[job_id] = run
                        run.thread = threading.Thread(
                            target=self._semantic_worker,
                            args=(run, store), name="project-semantic-owner",
                        )
                        run.thread.start()
                        dispatched = True
                        return job_id
                finally:
                    if not dispatched:
                        if store is not None:
                            self._close_cleanup_store(store)
                        if lease is not None:
                            self._close_capability(lease)
                        if pin is not None:
                            self._close_capability(pin)

    async def _semantic_build(self, run: _SemanticRun, store) -> None:
        """Issue, embed and publish on one owned async task and worker loop."""
        from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import (
            EmbeddingsServiceWrapper,
        )
        from . import semantic
        from .semantic_vectors import encode_vector

        carrier = semantic._issue_project_projection_scope(
            self, run.scope, run.state.settings, run.state.semantic_revision
        )
        wrapper = None
        try:
            self._retire_registry_for_model()
        except ProjectIndexRetirementUnproven:
            self._retain_uncertain_model(carrier, store)
            raise
        except BaseException:
            try:
                carrier.close()
            except BaseException as error:
                self._retain_uncertain_model(carrier, store)
                raise ProjectIndexRetirementUnproven(
                    "index_projection_retirement_unproven"
                ) from error
            raise
        try:
            wrapper = EmbeddingsServiceWrapper(
                model_name=carrier.model_name,
                api_key=carrier.api_key,
                base_url=carrier.base_url,
                device=getattr(carrier, "_config", {}).get("device"),
                execution_scope=carrier,
                local_files_only=carrier.local_files_only,
            )
            def claim():
                with self._admitted(store, run.scope.key) as live:
                    if live != run.authority or run.stop.is_set():
                        raise ValueError("stale_semantic_claim")
                    manifest = store.load_status(run.scope.key).manifest_revision
                    return store.semantic_claim(
                        run.scope, run.state, manifest_revision=manifest,
                        artifact_fingerprint=carrier.artifact_fingerprint,
                        claim_token=run.job_id,
                    )

            run.generation_id = self._retry_read(run.scope.key, claim)
            after = 0
            source_limit = (
                128 if run.state.settings.provider_id == "huggingface" else 32
            )
            while True:
                def read_page():
                    with self._admitted(store, run.scope.key) as live:
                        if live != run.authority or run.stop.is_set():
                            raise ValueError("stale_semantic_page")
                        return store.semantic_source_page(
                            run.scope,
                            run.generation_id,
                            run.job_id,
                            after=after,
                            limit=source_limit,
                            exclusions=live.exclusions,
                        )

                page = self._retry_read(run.scope.key, read_page)
                if not page:
                    break
                vectors = await wrapper.create_embeddings_async(
                    [source.text for source in page]
                )
                if len(vectors) != len(page):
                    raise ValueError("invalid_semantic_vector_count")
                encoded = tuple(encode_vector(vector) for vector in vectors)
                if any(len(vector) != len(encoded[0]) for vector in encoded[1:]):
                    raise ValueError("invalid_semantic_vector_dimension")
                # Split vector commits if a high-dimensional model exceeds the
                # one-MiB page bound; source ownership remains one bounded page.
                first = 0
                while first < len(page):
                    end = first
                    total = 0
                    while (
                        end < len(page)
                        and end - first < 32
                        and total + len(encoded[end]) <= 1024 * 1024
                    ):
                        total += len(encoded[end])
                        end += 1
                    if end == first:
                        raise ValueError("semantic_vector_page_limit")
                    def append_page():
                        with self._admitted(store, run.scope.key) as live:
                            return (
                                live == run.authority
                                and not run.stop.is_set()
                                and store.semantic_append(
                                    run.scope,
                                    run.generation_id,
                                    run.job_id,
                                    page[first:end],
                                    encoded[first:end],
                                    exclusions=live.exclusions,
                                )
                            )
                    if not self._retry_read(run.scope.key, append_page):
                        raise ValueError("stale_semantic_checkpoint")
                    after = page[end - 1].chunk_id
                    first = end
        finally:
            try:
                self._retire_registry_for_model()
            except BaseException as error:
                self._retain_uncertain_model(wrapper, carrier, store)
                if isinstance(error, ProjectIndexRetirementUnproven):
                    raise
                raise ProjectIndexRetirementUnproven(
                    "index_registry_retirement_unproven"
                ) from error
            if wrapper is not None:
                try:
                    wrapper.close()
                except BaseException:
                    try:
                        wrapper.close()
                    except BaseException as error:
                        run.error = error
                        signal = asyncio.Event()
                        with self._lock:
                            run.close_pending = True
                            run.close_loop = asyncio.get_running_loop()
                            run.close_signal = signal
                        while True:
                            await signal.wait()
                            signal.clear()
                            try:
                                wrapper.close()
                            except BaseException as retry_error:
                                run.error = retry_error
                            else:
                                with self._lock:
                                    run.close_pending = False
                                    run.close_loop = None
                                    run.close_signal = None
                                break
            try:
                carrier.close()
            except BaseException as error:
                self._retain_uncertain_model(carrier, store)
                raise ProjectIndexRetirementUnproven(
                    "index_projection_retirement_unproven"
                ) from error
        # Only a positively closed wrapper/carrier may publish Ready. The
        # selected config and complete artifact identity are checked again
        # after model retirement, outside the final finite SQL/fence unit.
        spec, destination = semantic._require_current_catalog(run.state.settings)
        if (
            semantic._target(spec, destination).artifact_fingerprint
            != carrier.artifact_fingerprint
        ):
            raise ValueError("semantic_artifact_changed")

        def publish():
            with self._admitted(store, run.scope.key) as live:
                return (
                    live == run.authority
                    and not run.stop.is_set()
                    and store.semantic_publish(
                        run.scope,
                        run.generation_id,
                        run.job_id,
                        exclusions=live.exclusions,
                    )
                )

        if not self._retry_read(run.scope.key, publish):
            raise ValueError("stale_semantic_publication")

    def _semantic_worker(self, run: _SemanticRun, store) -> None:
        run.store = store
        try:
            with self._bind(run.pin), self.registry_connection_scope():
                try:
                    asyncio.run(self._semantic_build(run, store))
                except ProjectIndexRetirementUnproven:
                    raise
                except BaseException as error:
                    run.error = error
                    if run.generation_id is not None:
                        reason = (
                            "quota_refused"
                            if (
                                isinstance(error, ValueError)
                                and str(error) == "index_cache_limit"
                            )
                            or (
                                isinstance(error, sqlite3.OperationalError)
                                and getattr(error, "sqlite_errorcode", None)
                                == sqlite3.SQLITE_FULL
                            )
                            else "interrupted"
                            if run.stop.is_set()
                            else "source_changed"
                            if (
                                isinstance(error, ValueError)
                                and str(error)
                                in {
                                    "stale_semantic_page",
                                    "stale_semantic_checkpoint",
                                    "stale_semantic_publication",
                                }
                            )
                            else "provider_unavailable"
                        )
                        def fail():
                            with self._admitted(store, run.scope.key) as live:
                                if live == run.authority:
                                    store.semantic_fail(
                                        run.scope, run.generation_id, run.job_id,
                                        reason=reason,
                                    )
                        self._retry_read(run.scope.key, fail)
                finally:
                    try:
                        store.close_native()
                    except BaseException:
                        self._retain_uncertain_model(store)
                        raise
            run.lease.close()
            if run.pin is not None:
                run.pin.close()
            run.read_admission.close()
            run.native_retired = True
            run.store = None
            with self._lock:
                # A retired run is diagnostic history only. Its exception may
                # retain the page, encoded vectors and model worker frames.
                run.error = None
                retired = [
                    job for job, item in self._semantic_runs.items() if item.native_retired
                ]
                for job in retired[:-16]:
                    owner = self._semantic_runs[job].thread
                    if owner is None or not owner.is_alive():
                        self._semantic_runs.pop(job, None)
        except BaseException as error:
            run.error = error

    def _stop_local(self, runs: tuple[_Run, ...]) -> None:
        for run in runs:
            run.stop.set()
        for run in runs:
            if run.supervisor is not None:
                run.supervisor.join()
            if not run.native_retired:
                raise RuntimeError("index_retirement_unproven") from run.error

    def _revoke(self, key, expected_scope, *, clear):
        with self._store(key, cleanup_only=clear) as store:
            # Serialize capture with _start's claim/dispatch interval, but never
            # hold either lock while signalling or joining native borrowers.
            with self._lock:
                for run in self._runs.values():
                    if (
                        run.claim.scope.key == key
                        and run.lease is not None
                        and not run.lease.closed
                    ):
                        store.bind_native_lifetime(run.lease)
                        break
                for run in self._semantic_runs.values():
                    if (
                        run.scope.key == key
                        and run.lease is not None
                        and not run.lease.closed
                    ):
                        store.bind_native_lifetime(run.lease)
                        break
            with self._lock, store.sql_unit(write=True), self._fence(key) as authority:
                if expected_scope is not None and authority is None:
                    raise ValueError("authority_unavailable")
                if authority is not None:
                    store.reconcile_authority(key, authority.authority_revision)
                revoked = store.revoke(key, clear=False, expected_scope=expected_scope)
                affected = tuple(
                    run for run in self._runs.values() if run.claim.scope.key == key
                )
                semantic_affected = tuple(
                    run for run in self._semantic_runs.values()
                    if run.scope.key == key and not run.native_retired
                )
            self._stop_local(affected)
            self._stop_semantic_local(semantic_affected)
            if clear:
                # Revocation is already durable if external retirement or cleanup
                # fails. Retry Clear explicitly; there is no background reopen.
                self._clear_binding(store, key, revoked)
                store.status_cache_bytes = binding_bytes(self._binding_root(store))
            return store.load_status(key)

    def _close_capability(self, capability):
        operation = self._operation_local.operation
        if operation.retirement_unproven:
            operation.retained.append(capability)
            return
        try:
            capability.close()
        except BaseException as error:
            operation.retirement_unproven = True
            operation.retained.append(capability)
            raise ProjectIndexRetirementUnproven(
                "index_capability_retirement_unproven"
            ) from error

    def _close_cleanup_store(self, store):
        try:
            store.close_native()
        except BaseException as error:
            operation = self._operation_local.operation
            operation.retirement_unproven = True
            operation.retained.append(store)
            raise ProjectIndexRetirementUnproven(
                "index_store_retirement_unproven"
            ) from error

    @staticmethod
    def _binding_inventory(group, lifetime_name):
        # Full bounded preflight, including aliases/sidecars, precedes any deletion.
        binding_bytes(group)
        namespaces = []
        leases = {group / lifetime_name}
        for count, entry in enumerate(group.iterdir(), 1):
            if count > 4096:
                raise ValueError("index_cleanup_inventory_limit")
            if (
                entry.name in {"cleanup-owner-v1.lock", "allocation-v1.lock"}
                and entry.is_file()
            ):
                continue
            if entry.is_file() and re.fullmatch(
                r"[0-9a-f]{64}\.lifetime\.lock", entry.name
            ):
                leases.add(entry)
            elif entry.is_dir() and re.fullmatch(r"[0-9a-f]{64}", entry.name):
                path = entry / "workspace_project_index.db"
                allowed = {
                    path.name + suffix for suffix in ("", "-journal", "-wal", "-shm")
                }
                if not path.is_file() or any(
                    child.name not in allowed for child in entry.iterdir()
                ):
                    raise ValueError("index_cleanup_namespace_unknown")
                namespaces.append(path)
            else:
                raise ValueError("index_cleanup_namespace_unknown")
        return namespaces, leases

    def _preflight_cleanup(self, namespaces, binding_namespace):
        for path in namespaces:
            store = WorkspaceProjectIndexDB(path, cleanup_only=True)
            try:
                store.cleanup_eligible(path.parent.name, binding_namespace)
            finally:
                self._close_cleanup_store(store)

    def _reclaim_namespaces(
        self, group, namespaces, binding_namespace, current_store=None, revoked=None
    ):
        self._preflight_cleanup(namespaces, binding_namespace)
        for path in sorted(namespaces, key=lambda value: value.stat().st_size):
            total = binding_bytes(group)
            size = path.stat().st_size
            headroom = size + size // 64 + 65536
            if total + headroom > MAX_CACHE_BYTES:
                raise ValueError("index_cleanup_headroom_required")
            store = (
                current_store
                if current_store is not None and path == current_store.db_path
                else WorkspaceProjectIndexDB(path, cleanup_only=True)
            )
            try:
                with store.connection() as conn:
                    pages = conn.execute("PRAGMA page_count").fetchone()[0]
                    conn.execute(f"PRAGMA max_page_count={pages}")
                if store is current_store:
                    if not store.clear_revoked(revoked):
                        raise ValueError("stale_scope")
                else:
                    store.clear_owned_namespace(path.parent.name, binding_namespace)
                store.reclaim_free_pages()
            finally:
                if store is not current_store:
                    self._close_cleanup_store(store)

    def _clear_binding(self, current_store, key, revoked):
        group = self._binding_root(current_store)
        with ExitStack() as resources:
            resources.enter_context(allocation_gate(group))
            namespaces, leases = self._binding_inventory(group, key.lifetime_name)
            for path in sorted(leases):
                lease = BindingLifetime.acquire(path)
                resources.callback(self._close_capability, lease)
                register_owned_lifetime(group, lease)
            observed = self._binding_inventory(group, key.lifetime_name)
            if observed != (namespaces, leases):
                raise ValueError("index_cleanup_inventory_changed")
            with self._fence(key) as authority:
                if authority is None:
                    raise ValueError("authority_unavailable")
                current = current_store.load_status(key)
                if current.scope != revoked or current.state != "Off":
                    raise ValueError("stale_scope")
            self._reclaim_namespaces(
                group, namespaces, key.binding_namespace, current_store, revoked
            )

    @staticmethod
    def _read_cleanup_owner(lease, group):
        receipt = json.loads(lease.read_cleanup_receipt().decode("utf-8"))
        if not isinstance(receipt, dict) or set(receipt) != {
            "version",
            "workspace_id",
            "binding_id",
            "authority_revision",
        }:
            raise ValueError("invalid_cleanup_receipt")
        if (
            type(receipt["version"]) is not int  # noqa: E721 - reject bool
            or receipt["version"] != 1
            or type(receipt["authority_revision"]) is not int  # noqa: E721 - reject bool
            or receipt["authority_revision"] < 1
        ):
            raise ValueError("invalid_cleanup_receipt")
        ids = (receipt["workspace_id"], receipt["binding_id"])
        if any(
            not isinstance(value, str) or not 0 < len(value) <= 256 or "\x00" in value
            for value in ids
        ):
            raise ValueError("invalid_cleanup_receipt")
        namespace = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        if namespace != group.name:
            raise ValueError("cleanup_receipt_owner_mismatch")
        return receipt

    def _enroll_cleanup_owner(self, store, key, authority):
        """Caller holds the current generation lifetime; no registry lock for I/O."""
        from tldw_chatbook.Utils.private_paths import (
            secure_private_directory,
            PrivatePathError,
        )

        group = self._binding_root(store)
        private = secure_private_directory(group, create=False, application_owned=True)
        if not private.verified_private:
            raise PrivatePathError(private)
        path = group / "cleanup-owner-v1.lock"
        with ExitStack() as resources:
            resources.enter_context(allocation_gate(group))
            if store._owned_lifetime is not None and not store._owned_lifetime.closed:
                register_owned_lifetime(group, store._owned_lifetime)
            if not path.exists() and binding_bytes(group) + 4096 > MAX_CACHE_BYTES:
                raise ValueError("index_cache_limit")
            receipt = BindingLifetime.acquire(path)
            resources.callback(self._close_capability, receipt)
            try:
                existing = self._read_cleanup_owner(receipt, group)
            except (ValueError, UnicodeError):
                existing = None
            if existing is not None:
                return
            namespaces, leases = self._binding_inventory(group, key.lifetime_name)
            for lifetime in sorted(leases - {group / key.lifetime_name}):
                lease = BindingLifetime.acquire(lifetime)
                resources.callback(self._close_capability, lease)
                register_owned_lifetime(group, lease)
            if self._binding_inventory(group, key.lifetime_name) != (
                namespaces,
                leases,
            ):
                raise ValueError("index_cleanup_inventory_changed")
            self._preflight_cleanup(namespaces, key.binding_namespace)
            payload = json.dumps(
                {
                    "version": 1,
                    "workspace_id": key.workspace_id,
                    "binding_id": key.binding_id,
                    "authority_revision": authority.authority_revision,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            if len(payload) > 4096 or (
                binding_bytes(group) - path.stat().st_size + len(payload)
                > MAX_CACHE_BYTES
            ):
                raise ValueError("index_cache_limit")
            try:
                receipt.write_cleanup_receipt(payload)
            except BaseException as error:
                self._operation_local.operation.retirement_unproven = True
                raise ProjectIndexRetirementUnproven(
                    "index_cleanup_receipt_settlement_unproven"
                ) from error
            verify_settlement(store.db_path, group)
        # The caller must still compare the captured authority before any write.

    def retry_detached_cleanup(self):
        """Retry bounded cleanup-only groups without opening any source folder."""
        from tldw_chatbook.Utils.private_paths import (
            secure_private_directory,
            PrivatePathError,
        )

        result = {"completed": 0, "pending": 0, "refused": 0}
        with self.registry_connection_scope():
            pin = self._retain(None)
            try:
                with self._bind(pin):
                    if pin is None or self._detached_fence is None:
                        raise ValueError("index_cleanup_owner_unavailable")
                    root = pin.data_root / "project_index"
                    if not root.exists():
                        return result
                    private = secure_private_directory(
                        root, create=False, application_owned=True
                    )
                    if not private.verified_private:
                        raise PrivatePathError(private)
                    lifetime_name = (
                        hashlib.sha256(pin.identity.encode()).hexdigest()
                        + ".lifetime.lock"
                    )
                    with os.scandir(root) as groups:
                        for count, entry in enumerate(groups, 1):
                            if count > 4096:
                                result["pending"] += 1
                                break
                            group = Path(entry.path)
                            try:
                                if entry.is_symlink() or not re.fullmatch(
                                    r"[0-9a-f]{64}", entry.name
                                ):
                                    raise ValueError("index_cleanup_namespace_unknown")
                                private = secure_private_directory(
                                    group, create=False, application_owned=True
                                )
                                if not private.verified_private:
                                    raise PrivatePathError(private)
                                with ExitStack() as resources:
                                    resources.enter_context(allocation_gate(group))
                                    receipt = BindingLifetime.acquire(
                                        group / "cleanup-owner-v1.lock",
                                        existing_only=True,
                                    )
                                    resources.callback(self._close_capability, receipt)
                                    owner = self._read_cleanup_owner(receipt, group)
                                    with self._detached_fence(
                                        owner["workspace_id"],
                                        owner["binding_id"],
                                        owner["authority_revision"],
                                    ) as detached:
                                        if not detached:
                                            continue
                                    namespaces, leases = self._binding_inventory(
                                        group, lifetime_name
                                    )
                                    for path in sorted(leases):
                                        lease = BindingLifetime.acquire(path)
                                        resources.callback(
                                            self._close_capability, lease
                                        )
                                        register_owned_lifetime(group, lease)
                                    if self._binding_inventory(
                                        group, lifetime_name
                                    ) != (namespaces, leases):
                                        raise ValueError(
                                            "index_cleanup_inventory_changed"
                                        )
                                    with self._detached_fence(
                                        owner["workspace_id"],
                                        owner["binding_id"],
                                        owner["authority_revision"],
                                    ) as detached:
                                        if not detached:
                                            continue
                                    self._reclaim_namespaces(
                                        group, namespaces, group.name
                                    )
                                result["completed"] += 1
                            except ProjectIndexRetirementUnproven:
                                raise
                            except (ValueError, UnicodeError, FileNotFoundError):
                                result["refused"] += 1
                            except (OSError, RuntimeError):
                                result["pending"] += 1
            except ProjectIndexRetirementUnproven:
                self._operation_local.operation.retained.append(pin)
                pin = None
                raise
            finally:
                if pin is not None:
                    self._close_capability(pin)
        return result

    def disable(
        self, key: ProjectIndexKey, *, expected_scope: ProjectIndexScope
    ) -> ProjectIndexStatus:
        return self._revoke(key, expected_scope, clear=False)

    def clear(
        self, key: ProjectIndexKey, *, expected_scope: ProjectIndexScope
    ) -> ProjectIndexStatus:
        return self._revoke(key, expected_scope, clear=True)

    def revoke(self, key: ProjectIndexKey) -> None:
        self._revoke(key, None, clear=False)

    def pause(
        self,
        job_id: str,
        *,
        expected_scope: ProjectIndexScope,
        expected_job_revision: int,
    ) -> ProjectIndexStatus:
        key = expected_scope.key
        with self._store(key) as store:
            with self._lock:
                run = self._runs.get(job_id)
                if (
                    run is not None
                    and run.claim.scope.key == key
                    and run.lease is not None
                    and not run.lease.closed
                ):
                    store.bind_native_lifetime(run.lease)
            with self._lock, self._admitted(store, key):
                current = store.load_status(key)
                claim = store.read_claim(expected_scope)
                if current.scope != expected_scope:
                    raise ValueError("stale_scope")
                if (
                    claim is None
                    or claim.job_id != job_id
                    or claim.job_revision != expected_job_revision
                    or current.state != "Building"
                ):
                    raise ValueError("stale_job")
                store.suspend_job(claim, state="Paused")
                affected = tuple(
                    run for run in self._runs.values() if run.claim == claim
                )
            self._stop_local(affected)
            return store.load_status(key)

    def refresh(
        self,
        key: ProjectIndexKey,
        *,
        expected_scope: ProjectIndexScope,
        expected_job_revision: int,
    ) -> str:
        if expected_scope.key != key:
            raise ValueError("stale_scope")
        return self._start(expected_scope, expected_job_revision, None)

    def resume(
        self,
        job_id: str,
        *,
        expected_scope: ProjectIndexScope,
        expected_job_revision: int,
    ) -> str:
        return self._start(expected_scope, expected_job_revision, job_id)

    def _start(self, scope, revision, resume_id):
        with self.registry_connection_scope():
            return self._start_owned(scope, revision, resume_id)

    def _start_owned(self, scope, revision, resume_id):
        # Nonblocking admission precedes pins, registry fences and service locks.
        # A busy attempt owns no job and cannot later dispatch deferred work.
        if not self._read_slots.acquire(blocking=False):
            raise LifetimeBusy("index_reads_busy")
        with ExitStack() as read_admission:
            read_admission.callback(self._read_slots.release)
            pin = self._retain(scope.key)
            try:
                with self._bind(pin):
                    return self._start_pinned(
                        scope, revision, resume_id, pin, read_admission
                    )
            except BaseException:
                with self._lock:
                    owned = any(run.pin is pin for run in self._runs.values())
                if pin is not None and not owned:
                    self._close_capability(pin)
                raise

    def _start_pinned(self, scope, revision, resume_id, pin, read_admission):
        store = lease = run = None
        try:
            with self._lock:
                if self._admission_closed():
                    raise RuntimeError("project_index_closed")
                if any(
                    run.key == scope.key and run.retirement_unproven
                    for run in self._searches
                ):
                    raise ProjectIndexRetirementUnproven(
                        "index_reader_retirement_unproven"
                    )
                if any(not run.native_retired for run in self._runs.values()):
                    raise LifetimeBusy("scanner_busy")
                if any(
                    not run.native_retired and run.scope.key.binding_namespace == scope.key.binding_namespace
                    for run in self._semantic_runs.values()
                ):
                    raise LifetimeBusy("semantic_writer_busy")
                store = self._factory(scope.key)
                store.status_cache_bytes = binding_bytes(self._binding_root(store))
                lease = self._lifetime(store, scope.key)
                with self._admitted(store, scope.key) as enrolled:
                    if enrolled is None:
                        raise ValueError("authority_unavailable")
                self._enroll_cleanup_owner(store, scope.key, enrolled)
                with self._admitted(store, scope.key) as authority:
                    if authority != enrolled:
                        raise ValueError("authority_changed")
                    if authority is None:
                        raise ValueError("authority_unavailable")
                    current = store.load_status(scope.key)
                    if current.scope != scope:
                        raise ValueError("stale_scope")
                    if current.job_revision != revision:
                        raise ValueError("stale_job")
                    old = store.read_claim(scope)
                    store.assert_native_lifetime()
                    self._claim_contract(old, scope.key)
                    token = (
                        "native-v1:" + scope.key.binding_namespace + ":" + uuid4().hex
                    )
                    if resume_id is not None or old is not None:
                        if (
                            old is None
                            or (resume_id is not None and old.job_id != resume_id)
                            or current.state
                            not in {"Paused", "Interrupted", "Partial", "Error"}
                        ):
                            raise ValueError("stale_job")
                        # The held exclusive acquisition proves every prior
                        # inherited holder retired. This callback performs no I/O.
                        claim = store.replace_retired_job(
                            old,
                            owner_token=token,
                            authority_revision=authority.authority_revision,
                            retirement_verifier=lambda observed: observed == old
                            and not lease.closed,
                            restart=resume_id is None,
                        )
                    else:
                        if old is not None or current.state not in {
                            "Paused",
                            "Ready",
                            "Partial",
                            "Interrupted",
                            "Error",
                        }:
                            raise ValueError("stale_job")
                        claim = store.claim_job(
                            scope,
                            owner_token=token,
                            authority_revision=authority.authority_revision,
                            expected_job_revision=revision,
                        )
                    policy = store.policy(scope)
                # The management thread's native connection ends before dispatch.
                store.close_native()
                run = _Run(
                    claim,
                    threading.Event(),
                    pin=pin,
                    lease=lease,
                    read_admission=read_admission.pop_all(),
                )
                self._runs[claim.job_id] = run
                self._scanner.lifetime = lease
                run.writer = threading.Thread(
                    target=self._write,
                    args=(run, store, authority, policy, pin),
                    name="project-index-writer",
                )
                run.supervisor = threading.Thread(
                    target=self._retire,
                    args=(run, store, lease, pin),
                    name="project-index-owner",
                )
                run.writer.start()
                run.supervisor.start()
                return claim.job_id
        except BaseException:
            if store is None:
                raise
            if run is not None and run.writer.ident is not None:
                # A failed supervisor launch cannot release a live writer's
                # capability. The dispatch caller becomes its physical owner.
                # The dispatch lock has exited before this physical join.
                run.stop.set()
                run.supervisor = None
                self._retire(run, store, lease, pin)
            else:
                try:
                    self._close_cleanup_store(store)
                finally:
                    if lease is not None:
                        self._close_capability(lease)
                if run is not None:
                    run.supervisor = None
                    if pin is not None:
                        self._close_capability(pin)
                    run.read_admission.close()
                    run.native_retired = True
            raise

    def _retry_job_sql(self, run, operation, *, stop_result=_NO_STOP_RESULT):
        """Retry only a busy native allocation gate after the prior unit unwinds."""
        deadline = getattr(self._scanner, "_deadline", 0.0) or time.monotonic() + 30
        while True:
            if stop_result is not _NO_STOP_RESULT and run.stop.is_set():
                return stop_result
            try:
                result = operation()
            except LifetimeBusy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                delay = min(0.01, remaining)
                if stop_result is _NO_STOP_RESULT:
                    time.sleep(delay)
                else:
                    run.stop.wait(delay)
            else:
                group = run.claim.scope.key.binding_namespace
                with self._lock:
                    handoff = bool(self._pending_reads.get(group)) and not self._closed
                if handoff and not run.stop.is_set():
                    # The prior SQL/fence/gate unit has positively retired.
                    run.stop.wait(0.025)
                return result

    def _write(self, run, store, authority, policy, pin):
        try:
            with self._bind(pin), self._registry_connections():
                self._write_bound(run, store, authority, policy)
            run.registry_closed = True
        except BaseException as error:
            run.state = "Error"
            run.error = error

    def _write_bound(self, run, store, authority, policy):
        from tldw_chatbook.Backup_Recovery.activation import execution_scope
        from .models import OWNER_ID

        claim = run.claim

        def frontier():
            return self._retry_job_sql(
                run, lambda: store.eligible_directories(claim), stop_result=()
            )

        def admit(directory):
            def attempt():
                with self._admitted(store, claim.scope.key) as current:
                    if (
                        run.stop.is_set()
                        or current is None
                        or current.authority_revision != claim.authority_revision
                        or current.exclusions != authority.exclusions
                        or not store.admit_job(claim)
                    ):
                        return None
                    return current

            return self._retry_job_sql(run, attempt, stop_result=None)

        def emit(page):
            prepared = prepare_scan_unit(page)

            def attempt():
                with self._admitted(store, claim.scope.key) as live:
                    store.publication_cache_bytes = binding_bytes(
                        self._binding_root(store)
                    )
                    if (
                        run.stop.is_set()
                        or live is None
                        or live.authority_revision != claim.authority_revision
                        or live.exclusions != authority.exclusions
                    ):
                        return False
                    return store.commit_scan_unit(
                        claim.scope,
                        claim.job_id,
                        prepared,
                        expected_job_revision=claim.job_revision,
                        authority_revision=live.authority_revision,
                    )

            return self._retry_job_sql(run, attempt, stop_result=False)

        quota_exhausted = False
        try:
            with execution_scope((OWNER_ID,), store.db_path) as allowed:
                if not allowed:
                    raise RuntimeError("index_execution_not_admitted")
                result = self._scanner.scan(
                    authority,
                    claim.scope,
                    claim.job_id,
                    frontier,
                    emit,
                    run.stop,
                    admit_unit=admit,
                    policy=policy,
                )
                state = self._retry_job_sql(
                    run, lambda: store.load_status(claim.scope.key).state
                )
                run.state = (
                    "Partial"
                    if state == "Partial"
                    else {
                        "complete": "Ready",
                        "partial": "Partial",
                        "paused": "Paused",
                        "interrupted": "Interrupted",
                        "error": "Error",
                    }[result.status]
                )
                if run.state == "Ready" and self._retry_job_sql(
                    run, lambda: store.pending_directories(claim.scope)
                ):
                    run.state = "Partial"
                store.close_native()
                run.db_closed = True
        except BaseException as error:
            run.state = "Error"
            run.error = error
            quota_exhausted = (
                isinstance(error, sqlite3.OperationalError)
                and getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL
            )
        finally:
            try:
                store.close_native()
                run.db_closed = True
                if quota_exhausted:
                    run.state = "Partial"
            except BaseException as error:
                run.error = error

    def _retire(self, run, store, lease, pin):
        try:
            with self._bind(pin):
                with self._registry_connections():
                    self._retire_bound(run, store, lease, pin)
                # Every newly opened registry handle closes on its owning thread
                # before native lifetime, storage admission or read budget ends.
                if run.db_closed and run.registry_closed and self._scanner.retired:
                    self._scanner.lifetime = None
                    lease.close()
                    if pin is not None:
                        pin.close()
                    run.read_admission.close()
                    run.native_retired = True
                    # Keep a small diagnostic tail, never traceback-owned stores
                    # or an unbounded collection of completed Refresh/Resume runs.
                    with self._lock:
                        run.error = None
                        retired = [
                            job
                            for job, item in self._runs.items()
                            if item.native_retired
                        ]
                        for job in retired[:-16]:
                            supervisor = self._runs[job].supervisor
                            # Native retirement precedes the supervisor's exit.
                            # Keep its shutdown join record while it is alive.
                            if supervisor is None or not supervisor.is_alive():
                                self._runs.pop(job, None)
        except BaseException as error:
            run.error = error

    def _retire_bound(self, run, store, lease, pin):
        run.writer.join()  # actual thread exit, separately from coroutine return
        try:
            if (
                not run.db_closed
                or not run.registry_closed
                or not self._scanner.retired
            ):
                raise RuntimeError("index_retirement_unproven") from run.error

            def publish_final():
                with self._admitted(store, run.claim.scope.key) as authority:
                    current = store.read_claim(run.claim.scope)
                    if authority is not None and current == run.claim:
                        if run.state == "Ready":
                            store.retire_job(
                                run.claim.scope,
                                run.claim.job_id,
                                owner_token=run.claim.owner_token,
                                expected_job_revision=run.claim.job_revision,
                                state="Ready",
                            )
                        else:
                            store.suspend_job(run.claim, state=run.state, advance=False)

            self._retry_job_sql(run, publish_final)
        except BaseException as error:
            run.error = error
        finally:
            # A failed close propagates through the registry context and keeps all
            # remaining ownership. Publication errors alone do not leak handles.
            store.close_native()

    def binding_changed(self, binding_id: str) -> None:
        """Post-commit hint only: stop local work; next admission reads the witness."""
        for run in tuple(self._runs.values()):
            if run.claim.scope.key.binding_id == binding_id:
                run.stop.set()
        for run in tuple(self._semantic_runs.values()):
            if run.scope.key.binding_id == binding_id:
                run.stop.set()
        if self._detached_fence is None:
            return
        with self._lock:
            if self._closed or (
                self._cleanup_thread is not None and self._cleanup_thread.is_alive()
            ):
                return

            def attempt():
                try:
                    self.retry_detached_cleanup()
                except Exception:
                    # Durable receipt permits explicit retry; unknown native
                    # ownership is also retained by the operation record.
                    pass

            self._cleanup_thread = threading.Thread(
                target=attempt, name="project-index-cleanup"
            )
            self._cleanup_thread.start()

    async def aclose(self) -> None:
        """Stop admission and await actual owner threads despite waiter cancellation."""

        def stop_and_snapshot():
            with self._lock:
                self._closed = True
                runs = tuple(self._runs.values())
                semantic_runs = tuple(self._semantic_runs.values())
                searches = tuple(self._searches)
                operations = tuple(self._operations)
                for run in runs:
                    run.stop.set()
                for run in semantic_runs:
                    run.stop.set()
                return runs, semantic_runs, searches, operations

        async def drain():
            _, semantic_runs, _, operations = await asyncio.to_thread(stop_and_snapshot)
            if (
                (
                    self._semantic_query is not None
                    and self._semantic_query.close_pending
                )
                or any(run.close_pending for run in semantic_runs)
                or any(
                    self._projection_close_pending(operation)
                    for operation in operations
                )
            ):
                raise ProjectIndexRetirementUnproven("project_embedding_close_pending")
            if self._cleanup_thread is not None:
                await asyncio.to_thread(self._cleanup_thread.join)
            for operation in operations:
                while not operation.done.is_set():
                    if (
                        (
                            self._semantic_query is not None
                            and self._semantic_query.close_pending
                        )
                        or any(run.close_pending for run in semantic_runs)
                        or self._projection_close_pending(operation)
                    ):
                        raise ProjectIndexRetirementUnproven(
                            "project_embedding_close_pending"
                        )
                    await asyncio.to_thread(operation.done.wait, 0.1)
                if self._projection_close_pending(operation):
                    raise ProjectIndexRetirementUnproven(
                        "project_embedding_close_pending"
                    )
            # An already-admitted start may have dispatched while close waited.
            # Its operation retires before this final writer/search snapshot.
            runs, semantic_runs, searches, _ = await asyncio.to_thread(stop_and_snapshot)
            for search in searches:
                await asyncio.to_thread(search.done.wait)
            for run in runs:
                if run.supervisor is not None:
                    await asyncio.to_thread(run.supervisor.join)
                if not run.native_retired:
                    raise RuntimeError("index_retirement_unproven") from run.error
            for run in semantic_runs:
                if run.thread is not None:
                    await asyncio.to_thread(run.thread.join)
                if not run.native_retired:
                    raise RuntimeError("index_retirement_unproven") from run.error
            if any(search.retirement_unproven for search in searches):
                raise ProjectIndexRetirementUnproven("index_reader_retirement_unproven")
            if any(operation.retirement_unproven for operation in operations):
                raise ProjectIndexRetirementUnproven(
                    "index_operation_retirement_unproven"
                )

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
