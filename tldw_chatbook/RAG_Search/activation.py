"""Local recovery review at installed RAG effects, separate from index readiness."""

import asyncio
import hashlib
import inspect
import json
import os
import sys
import threading
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit

from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths
from tldw_chatbook.Backup_Recovery.activation import (
    activation_permission,
    execution_scope,
)
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.TTS._async_lifecycle import join_retained_task

_active = ContextVar("rag_execution", default=None)


class RAGActivationRequired(PermissionError):
    """Restored RAG execution requires its owner's local review."""

    def __init__(self, reason="rag_activation_required"):
        super().__init__(reason)


def _mock_model(model):
    name = str(model).lower()
    return name in {
        "mock",
        "mock-embedding-model",
        "mock_embedding_model",
    } or name.startswith("mock/")


def source_paths(service=None, *, config=None, sources=()):
    """Observe installed selectors and supplied owners without constructing them."""
    selector = bootstrap.effective_config_path()
    loaded = sys.modules.get("tldw_chatbook.config")
    cached = getattr(loaded, "_CONFIG_CACHE", None)
    if getattr(loaded, "_CONFIG_CACHE_SOURCE", None) != selector:
        cached = None
    cached = cached or {}
    data = profile_paths.user_data_dir(cached)
    config = config or getattr(service, "_rag_activation_config", None)
    config = config or getattr(service, "config", None)
    config = getattr(config, "rag_config", config)
    paths = [("rag.definitions", selector)]
    paths.extend(getattr(service, "_rag_activation_sources", ()))
    profiles = sys.modules.get("tldw_chatbook.RAG_Search.config_profiles")
    manager = getattr(service, "profile_manager", None)
    if service is None:
        manager = getattr(profiles, "_GLOBAL_PROFILE_MANAGER", None)
    if manager is not None and (
        config is not None or not hasattr(service, "_cache_dir")
    ):
        paths.append(("rag.definitions", manager.profiles_dir))
    if hasattr(service, "model_name"):
        model_name = str(service.model_name)
        model = model_name.lower()
        # Only the installed local HF wrapper has model receipt/closure checks.
        paths[0] = ("config", selector)
        wrapper = sys.modules.get(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper"
        )
        if type(service) is getattr(wrapper, "EmbeddingsServiceWrapper", None):
            if _mock_model(model):
                paths[0] = ("rag.definitions", selector)
            elif not model.startswith("openai/"):
                paths[0] = ("models.artifacts", selector)
        if not _mock_model(model) and not model.startswith("openai/"):
            local = Path(model_name).expanduser()
            if (
                local.is_absolute()
                or local.exists()
                or model_name.startswith(("./", "../", "~"))
                or model_name[1:3] in (":\\", ":/")
            ):
                paths.append(("models.artifacts", local))
            else:
                cache = getattr(service, "_cache_dir", None)
                if cache is None:
                    from huggingface_hub.constants import HF_HUB_CACHE

                    cache = HF_HUB_CACHE
                paths.append(("models.artifacts", cache))
    if service is None or config is not None or not hasattr(service, "model_name"):
        paths.extend(
            (
                ("rag.definitions", selector),
                ("db.rag_indexing", selector),
            )
        )
        if service is None:
            paths.append(("rag.definitions", data / "rag_profiles"))
        vector = getattr(config, "vector_store", None)
        root = getattr(
            getattr(service, "vector_store", None), "persist_directory", None
        )
        root = (
            root
            or getattr(vector, "persist_directory", None)
            or getattr(service, "persist_directory", None)
        )
        if root is not None:
            paths.append(("rag.projections", root))
        search = getattr(config, "search", None)
        for key, database_key in (
            ("media_db_path", "media_db_path"),
            ("chachanotes_db_path", "chachanotes_db_path"),
            ("prompts_db_path", "prompts_db_path"),
        ):
            path = getattr(search, key, None)
            if path is None and search is not None:
                path = profile_paths.database_path(cached, database_key)
            # Source content is passive. Its admitted restored generation must
            # approve RAG use, not execution of the source database itself.
            paths.append(("rag.projections", path))
        paths.extend(
            ("rag.projections", path)
            for path in getattr(service, "_projection_indexed_source_paths", ())
        )
    paths.extend(sources)
    return tuple(
        dict.fromkeys(
            (owner, profile_paths.lexical_path(path))
            for owner, path in paths
            if path is not None
        )
    )


def _identity():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


@dataclass
class _Execution:
    identity: tuple
    leases: dict
    sources: tuple
    parent: object = None
    live: bool = True

    def check(self):
        if not self.live or self.identity[0] != os.getpid():
            raise RAGActivationRequired()
        if self.parent is not None:
            self.parent.check()


@contextmanager
def execution(service=None, *, config=None, sources=(), _transfer=None):
    """Reuse accepted exact leases; a transferred worker cannot outlive them."""
    active = _transfer or _active.get()
    if active is not None:
        active.check()
        if _transfer is None and active.identity != _identity():
            raise RAGActivationRequired()
    leases = dict(active.leases) if active else {}
    observed = tuple(
        dict.fromkeys(
            (active.sources if active else ())
            + source_paths(service, config=config, sources=sources)
        )
    )
    with ExitStack() as stack:
        try:
            for owner, path in observed:
                if path not in leases:
                    leases[path] = stack.enter_context(acquire_storage(path))
                if not stack.enter_context(
                    execution_scope((owner,), path, retained=leases[path])
                ):
                    raise RAGActivationRequired()
        except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
            raise RAGActivationRequired() from None
        scope = _Execution(_identity(), leases, observed, active)
        token = _active.set(scope)
        try:
            yield
        finally:
            scope.live = False
            _active.reset(token)


def require_local_model_construction(config):
    """Approval never means consent to acquire HF weights during construction."""
    if _mock_model(config.embedding.model):
        return
    if not ordinary_configuration(config):
        from .model_recovery import config_spec, require_local_embedding

        require_local_embedding(config_spec(config))


def ordinary_configuration(config):
    """Automatic legacy adoption is confined to configurations without recovery."""
    for _, path in source_paths(config=config):
        with acquire_storage(path) as lease:
            root, names = lease.execution_context(path)
            if not activation_permission(
                "config", bootstrap_root=root, namespaces=names, ordinary_only=True
            ):
                return False
    return True


def _call_scope(arguments, *, transfer=None):
    consumer = arguments.get(
        "service", arguments.get("rag_service", arguments.get("self"))
    )
    service = getattr(consumer, "_service", consumer)
    attached = getattr(service, "_projection_service", None)
    if attached is not None:
        service = attached() or service
    sources = [("rag.projections", arguments.get("persist_directory"))]
    manager = arguments.get("profile_manager")
    if manager is not None:
        sources.append(("rag.definitions", manager.profiles_dir))
    for key, owner in (
        ("media_db", "rag.projections"),
        ("chachanotes_db", "rag.projections"),
        ("indexing_db", "db.rag_indexing"),
    ):
        database = arguments.get(key, getattr(consumer, "_" + key, None))
        if database is not None and not getattr(database, "is_memory_db", False):
            sources.append((owner, getattr(database, "db_path", None)))
    sources.append(("db.rag_indexing", getattr(consumer, "_indexing_db_path", None)))
    entries = arguments.get(
        "entries", arguments.get("removals", arguments.get("batch", ()))
    )
    entry = arguments.get("entry")
    if entry is not None:
        entries = (entry,)
    sources.extend(
        ("rag.projections", getattr(entry, "source_path", None)) for entry in entries
    )
    return execution(
        service, config=arguments.get("config"), sources=sources, _transfer=transfer
    )


def guarded(function):
    """Gate named installed synchronous boundaries before their first effect."""
    signature = inspect.signature(function)

    @wraps(function)
    def call(*args, **kwargs):
        with _call_scope(signature.bind_partial(*args, **kwargs).arguments):
            result = function(*args, **kwargs)
            return result

    return call


def async_guarded(function):
    """Join outer native work on cancellation; reuse scopes within its task."""
    signature = inspect.signature(function)

    @wraps(function)
    async def call(*args, **kwargs):
        arguments = signature.bind_partial(*args, **kwargs).arguments
        active = _active.get()
        if active is not None and active.identity == _identity():
            with _call_scope(arguments):
                return await function(*args, **kwargs)

        async def retained():
            with _call_scope(arguments, transfer=active):
                return await function(*args, **kwargs)

        completion = asyncio.create_task(retained())
        await join_retained_task(completion)
        return completion.result()

    return call


def native_worker(service, function):
    """Transfer an accepted scope to the installed awaited vector worker."""
    active = _active.get()
    if active is not None:
        active.check()
        if active.identity != _identity():
            raise RAGActivationRequired()

    @wraps(function)
    def call(*args, **kwargs):
        with execution(service, _transfer=active):
            return function(*args, **kwargs)

    return call


_REVIEW_OWNERS = ("rag.definitions", "rag.projections", "db.rag_indexing")


@dataclass(frozen=True)
class RAGRecoveryReview:
    """Local, current-generation review; contains no credentials or raw endpoint."""

    fingerprint: str
    model: str
    provider_host: str
    sources: tuple[str, ...]
    owners: tuple[str, ...]
    prerequisites: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def _settings_identity(owner, path):
    """Bind approval to exact finite configuration/definition bytes, never display them."""
    if owner not in ("config", "rag.definitions"):
        return None
    candidates = sorted(path.glob("*.json")) if path.is_dir() else (path,)
    if len(candidates) > 4096:
        raise RAGActivationRequired("rag_recovery_review_unavailable")
    identities = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("rb") as source:
            content = source.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise RAGActivationRequired("rag_recovery_review_unavailable")
        identities.append((str(candidate), hashlib.sha256(content).hexdigest()))
    return identities


@contextmanager
def _review_scope(config, sources):
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.config import get_rag_indexing_db_path

    from .generation import _witnesses

    with ExitStack() as stack:
        # Existing Settings/CLI Backfill opens this configured tracking DB
        # lazily. Review its actual owner without constructing the database.
        paths = source_paths(
            config=config,
            sources=tuple(sources) + (("db.rag_indexing", get_rag_indexing_db_path()),),
        )
        witnesses, identities = [], []
        prerequisites = set()
        for owner, path in paths:
            lease = stack.enter_context(acquire_storage(path))
            for witness in _witnesses(path, lease):
                if witness not in witnesses:
                    witnesses.append(witness)
            try:
                info = path.stat()
                identity = (info.st_dev, info.st_ino)
            except FileNotFoundError:
                identity = None
            identities.append(
                (owner, str(path), identity, _settings_identity(owner, path))
            )
        if not _mock_model(config.embedding.model) and witnesses:
            from .model_recovery import config_spec, require_local_embedding

            try:
                require_local_embedding(config_spec(config))
            except (OSError, ValueError, RuntimeError):
                prerequisites.add("local_model_setup_required")
        targets = sorted(
            {
                (w["store_root"], w["generation"], owner)
                for w in witnesses
                for owner in _REVIEW_OWNERS
                if owner in w["owners"]
            }
        )
        # Hash full exact settings; only the digest and sanitized display leave
        # this owner. Stored credentials/query parameters never enter UI text.
        payload = (config.to_dict(), identities, witnesses)
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        endpoint = urlsplit(config.embedding.base_url or "")
        host = endpoint.hostname or "local/configured provider"
        model = str(config.embedding.model)
        if "://" in model:
            model = "configured model location"
        review = RAGRecoveryReview(
            fingerprint,
            model,
            host,
            tuple(sorted({str(p) for _, p in paths})),
            tuple(sorted({t[2] for t in targets})),
            tuple(sorted(prerequisites)),
        )
        yield (
            review,
            tuple(
                (ActivationStore(Path(root)), generation, owner)
                for root, generation, owner in targets
            ),
        )


def preview_recovery_review(config, *, sources=()):
    """Inspect current settings and paired local authority without index/model use."""
    with _review_scope(config, sources) as (review, _):
        return review


def approve_recovery_review(config, expected_fingerprint, *, sources=()):
    """Approve only reviewed RAG owners; never execute, rebuild or approve models."""
    try:
        with _review_scope(config, sources) as (review, targets):
            if review.fingerprint != expected_fingerprint:
                raise RAGActivationRequired("rag_recovery_review_changed")
            for store, generation, owner in targets:
                store.approve(generation, owner)
            return review
    except (OSError, ValueError, RuntimeError):
        raise RAGActivationRequired("rag_recovery_review_changed") from None
