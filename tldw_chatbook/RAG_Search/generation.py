"""Local generation readiness; queries never load models or construct source owners."""

import stat
import threading
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from uuid import uuid4

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    _private,
)
from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.Utils.platform_files import fcntl, os

from . import recovery

_SERVICES = weakref.WeakSet()
_REGISTRY_LOCK = threading.RLock()


def register_service(service):
    """Associate the actual store with its model/config owner without retaining it."""
    with _REGISTRY_LOCK:
        _SERVICES.add(service)
        service.vector_store._projection_service = weakref.ref(service)
        record_source_paths(service, ())


def _same_collection(first, second):
    a = getattr(first, "vector_store", None)
    b = getattr(second, "vector_store", None)
    return (
        type(a) is type(b)
        and hasattr(a, "persist_directory")
        and a.persist_directory.resolve() == b.persist_directory.resolve()
        and a.collection_name == b.collection_name
    )


def _dependency_paths(store, lease):
    """Read installed indexing dependencies; these paths never grant readiness."""
    root, _ = lease.execution_context(store.persist_directory)
    directory = root / "projection-dependencies"
    key = (str(store.persist_directory.resolve()), store.collection_name)
    try:
        with _private(root), _private(directory) as parent:
            record = bootstrap._read(parent, _record_name(store))
    except FileNotFoundError:
        if getattr(store, "_projection_dependencies_known", False):
            raise ValueError("projection_dependencies_missing") from None
        return set()
    if (
        set(record) != {"version", "collection", "paths"}
        or record["collection"] != list(key)
        or not bootstrap._paths(record["paths"])
    ):
        raise ValueError("projection_dependencies_invalid")
    paths = {Path(path) for path in record["paths"]}
    store._projection_dependencies_known = True
    return paths


def _persist_dependencies(store, lease, paths):
    """Union actual installed source paths before indexing under native admission."""
    from tldw_chatbook.Backup_Recovery.qualification import qualified_for

    root, _ = lease.execution_context(store.persist_directory)
    existing = root
    while not existing.exists():
        existing = existing.parent
    if not qualified_for("publish_file", existing)[0]:
        # Ordinary unqualified platforms retain only live dependency observations.
        # This branch never grants native recovery/readiness qualification.
        return paths
    from tldw_chatbook.Backup_Recovery.native_files import create_private_directory

    directory = root / "projection-dependencies"
    with _private(root):
        try:
            create_private_directory(directory)
        except FileExistsError:
            pass
        with _private(directory) as parent, ExitStack() as locks:
            lock = parent
            if os.name == "nt":
                # Windows byte locks require a regular file. Never unlink this
                # stable name: independent publishers must lock the same object.
                lock = os.open(
                    ".publication.lock",
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=parent,
                )
                locks.callback(os.close, lock)
                info = os.fstat(lock)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                    or info.st_nlink != 1
                ):
                    raise ValueError("projection_dependency_lock_unsafe")
            # Contention refuses before indexing; no admission acquisition occurs here.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.name == "nt":
                named = os.stat(
                    ".publication.lock", dir_fd=parent, follow_symlinks=False
                )
                if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("projection_dependency_lock_changed")
            return _publish_dependencies(store, lease, paths, directory, parent)


def _publish_dependencies(store, lease, paths, directory, parent):
    """Publish a bounded union while the caller holds the native publication lock."""
    import json

    from tldw_chatbook.Backup_Recovery.control_records import (
        _activation_record_identity,
        _publish_activation_record,
    )

    known = _dependency_paths(store, lease)
    paths = known | paths
    if paths == known:
        return paths
    record = {
        "version": 1,
        "collection": [str(store.persist_directory.resolve()), store.collection_name],
        "paths": sorted(str(path) for path in paths),
    }
    # Apply the existing bounded absolute-path parser before any publication.
    if (
        not bootstrap._paths(record["paths"])
        or len(json.dumps(record).encode()) > bootstrap.MAX_RECORD
    ):
        raise ValueError("projection_dependencies_invalid")
    name = _record_name(store)
    try:
        before = bootstrap._read(parent, name)
    except FileNotFoundError:
        before = None
    identity = _activation_record_identity(parent, name, before)
    _publish_activation_record(
        directory,
        parent,
        name,
        before,
        record,
        "dependencies-pending-" + uuid4().hex + ".json",
        identity,
    )
    store._projection_dependencies_known = True
    return paths


def record_source_paths(service, paths):
    """Retain actual ingestion dependencies across owner and process reopen."""
    with _REGISTRY_LOCK:
        observed = {
            Path(path).expanduser().absolute()
            for path in paths
            if isinstance(path, (str, Path)) and str(path) != ":memory:"
        }
        peers = [
            candidate
            for candidate in tuple(_SERVICES)
            if candidate is service or _same_collection(candidate, service)
        ]
        for peer in [service, *peers]:
            observed.update(vars(peer).get("_projection_indexed_source_paths", ()))
        store = service.vector_store
        if observed and hasattr(store, "persist_directory"):
            with acquire_storage(store.persist_directory) as lease:
                observed = _persist_dependencies(store, lease, observed)
        for peer in [service, *peers]:
            peer._projection_indexed_source_paths = frozenset(observed)


@contextmanager
def _scope(store, sources, service=None):
    """Hold actual native/source namespaces through owner verification and query."""
    with ExitStack() as stack:
        path = getattr(store, "persist_directory", None)
        lease = (
            participant.retained_lease(store)
            if getattr(store, "_client", None) is not None
            else stack.enter_context(acquire_storage(path))
        )
        witnesses = _witnesses(path, lease)
        paths = [
            source.db_path
            for source in (sources or {}).values()
            if hasattr(source, "db_path") and not getattr(source, "is_memory_db", False)
        ]
        if path is not None:
            paths.extend(_dependency_paths(store, lease))
        if service is not None:
            from tldw_chatbook.config import (
                get_chachanotes_db_path,
                get_media_db_path,
                get_prompts_db_path,
            )
            from tldw_chatbook.Utils.path_validation import validate_path_simple
            from tldw_chatbook.Utils.private_paths import lexical_path

            for configured, default in (
                (service.config.search.media_db_path, get_media_db_path),
                (service.config.search.chachanotes_db_path, get_chachanotes_db_path),
                (service.config.search.prompts_db_path, get_prompts_db_path),
            ):
                # Use the installed keyword owners' lexical validation before
                # their existence filtering: missing sources still select scope.
                selected = lexical_path(
                    validate_path_simple(
                        Path(str(configured or default())).expanduser(),
                        require_exists=False,
                        probe_existing=False,
                    )
                )
                paths.append(selected)
        if service is not None:
            paths.extend(getattr(service, "_projection_indexed_source_paths", ()))
        for source_path in set(paths):
            source_lease = stack.enter_context(acquire_storage(source_path))
            for witness in _witnesses(source_path, source_lease):
                if witness not in witnesses:
                    witnesses.append(witness)
        yield sorted(witnesses, key=lambda row: (row["store_root"], row["generation"]))


def _record_name(store):
    return (
        "projection-"
        + recovery._digest((str(store.persist_directory), store.collection_name))
        + ".json"
    )


def _read_record(store, witnesses):
    records = []
    for witness in witnesses:
        activation = ActivationStore(Path(witness["store_root"]))
        with (
            _private(activation.root),
            _private(activation._generation(witness["generation"])) as parent,
        ):
            records.append(bootstrap._read(parent, _record_name(store)))
    if not records or any(record != records[0] for record in records):
        raise ValueError("projection_readiness_missing")
    record = records[0]
    if (
        set(record) != {"version", "witnesses", "build"}
        or record["version"] != 1
        or record["witnesses"] != witnesses
    ):
        raise ValueError("projection_readiness_invalid")
    return record


def _durable_build(build):
    value = asdict(build)
    # Process-object identity is never used as a serialized reopen credential.
    value["model"] = build.model[0]
    import json

    return json.loads(json.dumps(value))


def _clear_caches():
    # Every extant service is independently invalidated, including pre-restore caches.
    with _REGISTRY_LOCK:
        services = tuple(_SERVICES)
    for service in services:
        service.cache.clear()


def load_provenance(service, sources):
    """Explicit reconciliation may recheck durable proof against supplied owners."""
    with _scope(service.vector_store, sources) as witnesses:
        if not witnesses:
            return
        record = _read_record(service.vector_store, witnesses)
        value = record["build"]
        from .simplified.embeddings_wrapper import _DeterministicEmbeddingFactory

        # Other loaded models retain same-process proof only. A revision string
        # alone cannot prove fresh-process weights; never acquire a model here.
        if type(service.embeddings.factory) is not _DeterministicEmbeddingFactory:
            raise ValueError("projection_model_reopen_unavailable")
        model = recovery._model(service)
        if value["model"] != model[0]:
            raise ValueError("projection_model_identity_unavailable")
        build = recovery._Build(
            **{
                **value,
                "sources": tuple(tuple(s) for s in value["sources"]),
                "root": tuple(value["root"]),
                "dimensions": tuple(value["dimensions"]),
                "model": model,
            }
        )
        service._recovery_projection_build = build


def persist_readiness(service, observation, **sources):
    """Mint private local readiness only after explicit full owner reconciliation."""
    service._projection_sources = sources
    record_source_paths(
        service,
        (source.db_path for source in sources.values() if hasattr(source, "db_path")),
    )
    with _scope(service.vector_store, sources) as witnesses:
        if not witnesses:
            return
        current = recovery._reconcile_projection(service, **sources)
        if not current.ready or current != observation:
            raise ValueError("projection_final_reconciliation_failed")
        record = {
            "version": 1,
            "witnesses": witnesses,
            "build": _durable_build(service._recovery_projection_build),
        }
        from tldw_chatbook.Backup_Recovery.control_records import (
            _activation_record_identity,
            _publish_activation_record,
        )

        _clear_caches()
        for witness in witnesses:
            activation = ActivationStore(Path(witness["store_root"]))
            directory = activation._generation(witness["generation"])
            with _private(activation.root), _private(directory) as parent:
                name = _record_name(service.vector_store)
                try:
                    before = bootstrap._read(parent, name)
                except FileNotFoundError:
                    before = None
                identity = _activation_record_identity(parent, name, before)
                _publish_activation_record(
                    directory,
                    parent,
                    name,
                    before,
                    record,
                    "projection-pending-" + uuid4().hex + ".json",
                    identity,
                )


@contextmanager
def query_scope(target):
    """Refuse before cache/provider/native query effects without rebuilding."""
    service = target if hasattr(target, "vector_store") else None
    store = service.vector_store if service is not None else target
    if service is None:
        reference = getattr(store, "_projection_service", None)
        service = reference() if reference is not None else None
    sources = getattr(service, "_projection_sources", None)
    with ExitStack() as stack:
        try:
            witnesses = stack.enter_context(_scope(store, sources, service))
            if witnesses:
                if service is None or sources is None:
                    raise ValueError("projection_source_owner_required")
                record = _read_record(store, witnesses)
                build = getattr(service, "_recovery_projection_build", None)
                if (
                    type(build) is not recovery._Build
                    or _durable_build(build) != record["build"]
                ):
                    raise ValueError("projection_reconciliation_required")
                observation = recovery._reconcile_projection(service, **sources)
                if not observation.ready:
                    raise ValueError(
                        "projection_reconciliation_required:"
                        + ",".join(observation.issues)
                    )
                marker = recovery._digest(record)
                if getattr(service, "_projection_cache_generation", None) != marker:
                    _clear_caches()
                    service._projection_cache_generation = marker
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            AttributeError,
        ) as error:
            _clear_caches()
            if str(error).startswith("projection_"):
                raise
            raise ValueError("projection_readiness_unavailable") from None
        yield


def service_query(function):
    @wraps(function)
    async def invoke(self, *args, **kwargs):
        with query_scope(self):
            return await function(self, *args, **kwargs)

    return invoke


def store_query(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with query_scope(self):
            return function(self, *args, **kwargs)

    return invoke
