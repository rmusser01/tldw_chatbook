"""Deliberate recovery of existing local HF embeddings and their native lifetime.

Receipts live in the existing generation's private activation directory. They
never provision weights or authorize configuration, providers, or RAG indexing.
"""

import asyncio
import hashlib
import json
import os
import stat
import threading
import time
import weakref
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    _flush_existing,
    _private,
    _read,
    _write,
    execution_scope,
)
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
from tldw_chatbook.Backup_Recovery.profile_paths import lexical_path
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

from .activation import RAGActivationRequired, _identity


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _closure(path):
    """Hash the exact bounded no-follow regular-file closure, refusing aliases."""
    entries = []
    manifests = {}

    def visit(fd, relative):
        before = os.fstat(fd)
        if len(entries) >= 4096:
            raise ValueError("local_model_too_many_files")
        if stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1:
                raise ValueError("local_model_linked_file")
            digest = hashlib.sha256()
            manifest = (
                bytearray()
                if relative
                in ("/config.json", "/tokenizer_config.json", "/adapter_config.json")
                or relative.endswith(".index.json")
                else None
            )
            while chunk := os.read(fd, 1024 * 1024):
                digest.update(chunk)
                if manifest is not None:
                    manifest.extend(chunk)
                    if len(manifest) > 8 * 1024 * 1024:
                        raise ValueError("local_model_manifest_too_large")
            entries.append((relative, digest.hexdigest()))
            if manifest is not None:
                manifests[relative] = json.loads(manifest)
        elif stat.S_ISDIR(before.st_mode):
            entries.append((relative, "directory"))
            for name in sorted(os.listdir(fd)):
                child = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                try:
                    visit(child, relative + "/" + name)
                finally:
                    os.close(child)
        else:
            raise ValueError("local_model_nonregular_file")
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("local_model_changed")

    with pinned_directory(path) as fd:
        visit(fd, "")
    files = {relative for relative, digest in entries if digest != "directory"}

    def require_file(value):
        if value is None:
            return
        if not isinstance(value, str) or ".." in Path(value).parts:
            raise ValueError("local_model_reference_outside_closure")
        target = Path(value)
        if target.is_absolute():
            try:
                target = target.relative_to(path)
            except ValueError:
                raise ValueError("local_model_reference_outside_closure") from None
        if "/" + target.as_posix() not in files:
            raise ValueError("local_model_reference_outside_closure")

    def references(value):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key == "weight_map":
                for filename in item.values():
                    require_file(filename)
            elif key.endswith("_file"):
                require_file(item)
            elif key.endswith("_files") and isinstance(item, list):
                for filename in item:
                    require_file(filename)
            elif isinstance(item, dict):
                references(item)

    # An adapter may redirect AutoModel to a different base model. This owner
    # controls an existing complete local model, not adapter provisioning.
    if "/adapter_config.json" in manifests:
        raise ValueError("local_model_adapter_setup_unsupported")
    for manifest in manifests.values():
        references(manifest)
    return entries


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: int = 1
    generation: str
    fingerprint: str
    model_path: str
    settings_digest: str
    closure_digest: str


@dataclass(frozen=True)
class LocalEmbeddingReview:
    """Sanitized review evidence; no configuration content or credentials."""

    fingerprint: str
    model_path: str
    file_count: int
    generations: tuple[str, ...]


def _settings(cfg):
    values = cfg.model_dump()
    # cache_dir is not consumed by a local-only loader; bind every actual loader
    # and encoding option, including the selected local path and revision.
    values.pop("cache_dir", None)
    values.pop("provider", None)
    values["device"] = values.get("device") or "auto"
    return values


def config_spec(config):
    """Mirror the installed wrapper's actual HF loader options, without loading."""
    from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg

    return HFModelCfg(
        model_name_or_path=str(config.embedding.model),
        device=config.embedding.device,
        local_files_only=True,
    )


@contextmanager
def _review(cfg, *, retained=None):
    from .generation import _witnesses

    selector = bootstrap.effective_config_path()
    configured = Path(cfg.model_name_or_path).expanduser()
    if (
        configured.is_absolute()
        or configured.exists()
        or str(configured).startswith(("./", "../"))
    ):
        model = lexical_path(configured)
    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        model = lexical_path(cfg.cache_dir or HF_HUB_CACHE)
    paths = (selector, model)
    with ExitStack() as stack:
        leases, witnesses = {}, []
        for path in paths:
            lease = retained.get(path) if retained else None
            if lease is None:
                lease = stack.enter_context(acquire_storage(path))
            leases[path] = lease
            for witness in _witnesses(path, lease):
                if witness not in witnesses:
                    witnesses.append(witness)
        if not witnesses:
            yield None, (), leases
            return
        selected = Path(cfg.model_name_or_path).expanduser()
        if (
            not selected.is_absolute()
            or not selected.is_dir()
            or not cfg.local_files_only
            or cfg.trust_remote_code
        ):
            raise RAGActivationRequired("rag_model_setup_required")
        closure = _closure(model)
        # Selectors are private regular files, read through the existing strict
        # bootstrap parser's no-follow envelope where records are used below.
        with pinned_directory(selector.parent) as parent:
            fd = os.open(
                selector.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_size > 8 * 1024 * 1024
                ):
                    raise ValueError("local_model_selector_unavailable")
                with os.fdopen(fd, "rb", closefd=False) as source:
                    content = source.read(8 * 1024 * 1024 + 1)
                if len(content) != info.st_size:
                    raise ValueError("local_model_selector_changed")
                after = os.fstat(fd)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) != (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ):
                    raise ValueError("local_model_selector_changed")
            finally:
                os.close(fd)
        settings = _digest(
            (_settings(cfg), str(selector), hashlib.sha256(content).hexdigest())
        )
        closure_digest = _digest(closure)
        fingerprint = _digest((str(model), settings, closure_digest, witnesses))
        records = tuple(
            (
                ActivationStore(Path(w["store_root"])),
                _Receipt(
                    generation=w["generation"],
                    fingerprint=fingerprint,
                    model_path=str(model),
                    settings_digest=settings,
                    closure_digest=closure_digest,
                ),
            )
            for w in witnesses
        )
        review = LocalEmbeddingReview(
            fingerprint,
            str(model),
            sum(digest != "directory" for _, digest in closure),
            tuple(sorted({w["generation"] for w in witnesses})),
        )
        yield review, records, leases


def _require(records):
    for store, record in records:
        with (
            _private(store.root),
            _private(store._generation(record.generation)) as parent,
        ):
            name = "local-embedding-" + record.fingerprint + ".json"
            try:
                actual = _Receipt.model_validate(_read(parent, name))
            except (OSError, ValueError):
                raise RAGActivationRequired("rag_model_setup_required") from None
            if actual != record or not store.allowed(
                record.generation, "models.artifacts"
            ):
                raise RAGActivationRequired("rag_model_setup_required")


def preview_local_embedding(config):
    """Inspect exact configured local files without constructing an embedder."""
    with _review(config_spec(config)) as (review, _, _):
        if review is None:
            raise ValueError("local_model_recovery_not_required")
        return review


def approve_local_embedding(config, expected_fingerprint):
    """Confirm current evidence and approve only its model owner, without loading."""
    with _review(config_spec(config)) as (review, records, _):
        if review is None or review.fingerprint != expected_fingerprint:
            raise ValueError("local_model_review_changed")
        for store, record in records:
            with (
                _private(store.root),
                _private(store._generation(record.generation)) as parent,
            ):
                if (
                    "models.artifacts"
                    not in store._required(parent, record.generation).owners
                ):
                    raise ValueError("local_model_owner_not_required")
                name = "local-embedding-" + record.fingerprint + ".json"
                try:
                    _write(parent, name, record)
                except FileExistsError:
                    _flush_existing(parent, name, record)
            store.approve(record.generation, "models.artifacts")
        return review


def require_local_embedding(cfg):
    """Return exact recovered binding; ordinary HF behavior remains unchanged."""
    with _review(cfg) as (review, records, _):
        _require(records)
        return review.fingerprint if review is not None else None


class LocalEmbeddingLifetime:
    """Only installed HF handles/wrappers; bare handles require explicit close."""

    def __init__(self):
        self._lock = threading.RLock()
        self._borrowers = {}
        self._wrappers = weakref.WeakSet()
        self._tokens = {}
        self._accepted = ContextVar("local_embedding_accepted", default=None)
        self._closed = False
        self._retirement = None
        self._retirement_error = None

    @contextmanager
    def operation(self, *, _parent=None):
        with self._lock:
            previous = _parent or self._accepted.get()
            if previous is not None and (
                previous not in self._tokens
                or (_parent is None and self._tokens[previous] != _identity())
            ):
                raise RAGActivationRequired("local_model_operation_context_changed")
            if self._closed and previous is None:
                raise RAGActivationRequired("local_model_operations_paused")
            token = object()
            self._tokens[token] = _identity()
        context = self._accepted.set(token)
        try:
            yield
        finally:
            self._accepted.reset(context)
            with self._lock:
                self._tokens.pop(token)

    def worker(self, function):
        """Transfer only the installed wrapper's explicitly retained native call."""
        parent = self._accepted.get()
        with self._lock:
            if parent not in self._tokens or self._tokens[parent] != _identity():
                raise RAGActivationRequired("local_model_operation_missing")

        @wraps(function)
        def invoke(*args, **kwargs):
            with self.operation(_parent=parent):
                return function(*args, **kwargs)

        return invoke

    def register_wrapper(self, wrapper):
        with self._lock:
            self._wrappers.add(wrapper)

    def _maintenance_close_admission(self):
        with self._lock:
            self._closed = True

    def _close_wrappers(self):
        try:
            with self._lock:
                wrappers = tuple(self._wrappers)
            for wrapper in wrappers:
                if wrapper.factory is not None:
                    # The real factory lock joins native cached work. Only a
                    # successful native close may release its source holds.
                    wrapper.factory.close()
                    wrapper.factory = None
                    wrapper._config_dict = None
                    wrapper._embedding_dimension = None
        except BaseException as error:  # noqa: BLE001 - return native failure to the drain waiter, retaining holds
            self._retirement_error = error

    async def _maintenance_drain(self, deadline):
        while True:
            with self._lock:
                if not self._closed:
                    raise RAGActivationRequired("local_model_intake_open")
                if not self._tokens:
                    if not self._borrowers and self._retirement is None:
                        return True
                    if self._retirement is None:
                        self._retirement_error = None
                        self._retirement = threading.Thread(
                            target=self._close_wrappers,
                            name="local-embedding-retirement",
                            daemon=True,
                        )
                        self._retirement.start()
                    elif not self._retirement.is_alive():
                        if self._retirement_error is not None:
                            error = self._retirement_error
                            self._retirement = None
                            raise error
                        return not self._borrowers
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.01))

    async def _maintenance_resume(self):
        while self._retirement is not None and self._retirement.is_alive():
            await asyncio.sleep(0.01)
        with self._lock:
            self._retirement = None
            self._closed = False


participant = LocalEmbeddingLifetime()


@contextmanager
def reviewed_local_identity(backend):
    """Yield a reviewed digest and native borrower presence while holding scopes."""
    item = participant._borrowers.get(backend)
    if item is None:
        yield None, False
        return
    with participant.operation(), item[4]:
        if participant._borrowers.get(backend) is not item:
            raise RAGActivationRequired("local_model_closed")
        _, leases, cfg, expected, _ = item
        with _review(cfg, retained=leases) as (review, records, _):
            if review != expected or bool(records) != (review is not None):
                raise RAGActivationRequired("local_model_review_changed")
            _require(records)
            with ExitStack() as stack:
                for path, lease in leases.items():
                    if not stack.enter_context(
                        execution_scope(("models.artifacts",), path, retained=lease)
                    ):
                        raise RAGActivationRequired()
                identity = (
                    _digest(
                        (
                            "reviewed-local-hf-v1",
                            records[0][1].closure_digest,
                            _settings(cfg),
                        )
                    )
                    if review is not None
                    else None
                )
                yield identity, True


def recovered_load(function):
    """Retain actual source holds through the loaded native object's lifetime."""

    @wraps(function)
    def invoke(self, cfg):
        with _review(cfg) as (review, records, _):
            _require(records)
            if (
                review is None
                and not Path(cfg.model_name_or_path).expanduser().is_dir()
            ):
                return function(self, cfg)
            return _load_local(self, cfg, review)

    def _load_local(self, cfg, review):
        with participant.operation():
            holds = ExitStack()
            try:
                leases = {
                    path: holds.enter_context(acquire_storage(path))
                    for path in (
                        bootstrap.effective_config_path(),
                        lexical_path(cfg.model_name_or_path),
                    )
                }
                with _review(cfg, retained=leases) as (current, current_records, _):
                    if current != review:
                        raise RAGActivationRequired("local_model_review_changed")
                    _require(current_records)
                    for path, lease in leases.items():
                        if not holds.enter_context(
                            execution_scope(
                                ("models.artifacts",), path, retained=lease
                            )
                        ):
                            raise RAGActivationRequired()
                with participant._lock:
                    participant._borrowers[self] = (
                        holds,
                        leases,
                        cfg,
                        review,
                        threading.RLock(),
                    )
                # A partial native load is not proof of retirement; any
                # exception retains this handle's native source exclusion.
                result = function(self, cfg)
                with _review(cfg, retained=leases) as (current, _, _):
                    if current != review:
                        raise RAGActivationRequired("local_model_review_changed")
                return result
            except BaseException:
                if self not in participant._borrowers:
                    holds.close()
                raise

    return invoke


def recovered_encode(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        item = participant._borrowers.get(self)
        if item is None:
            return function(self, *args, **kwargs)
        with participant.operation(), item[4]:
            if participant._borrowers.get(self) is not item:
                raise RAGActivationRequired("local_model_closed")
            _, leases, cfg, expected, _ = item
            with _review(cfg, retained=leases) as (review, records, _):
                if review != expected:
                    raise RAGActivationRequired("local_model_review_changed")
                _require(records)
                with ExitStack() as stack:
                    for path, lease in leases.items():
                        if not stack.enter_context(
                            execution_scope(
                                ("models.artifacts",), path, retained=lease
                            )
                        ):
                            raise RAGActivationRequired()
                    return function(self, *args, **kwargs)

    return invoke


def recovered_close(function):
    @wraps(function)
    def invoke(self):
        item = participant._borrowers.get(self)
        if item is None:
            return function(self)
        with item[4]:
            result = function(self)
            with participant._lock:
                participant._borrowers.pop(self)
            item[0].close()
            return result

    return invoke


def model_operation(function):
    """Count a complete installed factory call, including load then encode."""

    @wraps(function)
    def invoke(self, *args, **kwargs):
        local = any(
            getattr(spec, "provider", None) == "huggingface"
            and Path(spec.model_name_or_path).expanduser().is_dir()
            for spec in self._cfg.models.values()
        )
        if not local:
            return function(self, *args, **kwargs)
        with participant.operation():
            return function(self, *args, **kwargs)

    return invoke
