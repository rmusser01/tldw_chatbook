"""Finite recovered provider guards and deliberate OpenAI connection review."""

import hashlib
import inspect
import json
import stat
import threading
import tomllib
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    _flush_existing,
    _private,
    _read,
    _write,
)
from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.Utils.platform_files import os


class ProviderReconnectRequired(RuntimeError):
    """The selected restored connection has not been deliberately reviewed."""

    def __init__(self):
        super().__init__("provider_reconnect_required")


def _ordinary_operation():
    active = _current.get()
    if active is not None and active[1] == _identity():
        if active[0].history:
            raise ProviderReconnectRequired()
        active[0].check()
        return active[0], False
    return _Operation(bootstrap.effective_config_path(), ordinary_only=True), True


def unqualified(function):
    """Refuse fixed unsupported recovered effects and retain accepted streams."""
    if inspect.isasyncgenfunction(function):

        @wraps(function)
        async def stream(*args, **kwargs):
            operation, owned = _ordinary_operation()
            iterator = function(*args, **kwargs)
            try:
                with _using(operation):
                    async for item in iterator:
                        yield item
            finally:
                # A failed actual cleanup deliberately leaves admission held.
                await iterator.aclose()
                if owned:
                    operation.close()

        return stream
    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def asynchronous(*args, **kwargs):
            operation, owned = _ordinary_operation()
            try:
                with _using(operation):
                    return await function(*args, **kwargs)
            finally:
                if owned:
                    operation.close()

        return asynchronous

    @wraps(function)
    def synchronous(*args, **kwargs):
        operation, owned = _ordinary_operation()
        transferred = False
        try:
            with _using(operation):
                result = function(*args, **kwargs)
            if isinstance(result, Iterator) and owned:
                operation.own(result)
                stream = _OpenAIStream(result, operation.path, operation)
                transferred = True
                return stream
            return result
        finally:
            if owned and not transferred:
                operation.close()

    return synchronous


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _config(path):
    with pinned_directory(path.parent) as parent:
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or before.st_mode & 0o077
                or before.st_size > 8 * 1024 * 1024
            ):
                raise ProviderReconnectRequired()
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(8 * 1024 * 1024 + 1)
            after = os.fstat(fd)
            identity = lambda item: (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            if len(data) != before.st_size or identity(before) != identity(after):
                raise ProviderReconnectRequired()
            return tomllib.loads(data.decode())
        finally:
            os.close(fd)


def _selection(path, auth_source=None, *, resolve=False):
    raw = _config(path)
    modern = raw.get("api_settings", {}).get("openai", {})
    legacy = raw.get("API", {})
    endpoint = modern.get("api_base_url") or "https://api.openai.com/v1"
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ProviderReconnectRequired()
    endpoint = endpoint.rstrip("/")
    # Recovered inputs are deliberately finite. Other supported ordinary loader
    # fallbacks are not silently promoted to reviewed credential sources.
    sources = {
        "config:api_settings.openai.api_key": modern.get("api_key"),
        "config:API.openai_api_key": legacy.get("openai_api_key"),
    }
    env_name = modern.get("api_key_env_var") or "OPENAI_API_KEY"
    if not isinstance(env_name, str) or not env_name.isidentifier():
        raise ProviderReconnectRequired()
    env_source = "env:" + env_name
    if auth_source is None:
        auth_source = next((key for key, value in sources.items() if value), env_source)
    if auth_source not in (*sources, env_source):
        raise ProviderReconnectRequired()
    # Include all actual competing OpenAI auth selectors, not unrelated settings.
    selection = _digest(
        (
            endpoint,
            modern.get("api_key"),
            modern.get("api_key_env_var"),
            modern.get("credential_source"),
            legacy.get("openai_api_key"),
            auth_source,
        )
    )
    secret = None
    if resolve:
        value = (
            os.environ.get(env_name)
            if auth_source == env_source
            else sources[auth_source]
        )
        if (
            not isinstance(value, str)
            or not value.strip()
            or value.strip()
            in {"<API_KEY_HERE>", "YOUR_KEY", "your_key", "your-api-key"}
            or value.startswith(("ENC:", "encrypted:", "recovery:"))
        ):
            raise ProviderReconnectRequired()
        secret = value.strip()
    return endpoint, auth_source, selection, secret


def _history(path, lease):
    try:
        return tuple(_witnesses(path, lease))
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, AttributeError):
        raise ProviderReconnectRequired() from None


@dataclass(frozen=True)
class OpenAIReconnectReview:
    fingerprint: str
    endpoint: str
    auth_source: str
    generations: tuple[str, ...]
    config_selector: str


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    generation: str
    config_selector: str
    endpoint: str
    auth_source: str
    selection: str = Field(pattern=r"^[a-f0-9]{64}$")
    credential_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


def _receipt_prefix(path, selection):
    return "openai-connection-" + _digest((path, selection)) + "-"


def _receipt_name(record):
    return (
        _receipt_prefix(record.config_selector, record.selection)
        + _digest(record.model_dump())
        + ".json"
    )


def prepare_openai_reconnect(*, auth_source=None):
    """Inspect one actual OpenAI selector without resolving secrets or clients."""
    path = bootstrap.effective_config_path()
    with acquire_storage(path) as lease:
        history = _history(path, lease)
        if not history:
            raise ProviderReconnectRequired()
        endpoint, source, selection, _ = _selection(path, auth_source)
        return OpenAIReconnectReview(
            _digest((str(path), selection, history)),
            endpoint,
            source,
            tuple(sorted({row["generation"] for row in history})),
            str(path),
        )


def confirm_openai_reconnect(review):
    """Record only this generation's explicit OpenAI endpoint/auth selection."""
    if type(review) is not OpenAIReconnectReview:
        raise ProviderReconnectRequired()
    path = bootstrap.effective_config_path()
    with acquire_storage(path) as lease:
        history = _history(path, lease)
        endpoint, source, selection, secret = _selection(
            path, review.auth_source, resolve=True
        )
        if (
            str(path) != review.config_selector
            or endpoint != review.endpoint
            or _digest((str(path), selection, history)) != review.fingerprint
        ):
            raise ProviderReconnectRequired()
        for witness in history:
            store = ActivationStore(Path(witness["store_root"]))
            record = _Receipt(
                generation=witness["generation"],
                config_selector=str(path),
                endpoint=endpoint,
                auth_source=source,
                selection=selection,
                credential_digest=_digest(secret),
            )
            name = _receipt_name(record)
            with (
                _private(store.root),
                _private(store._generation(record.generation)) as parent,
            ):
                try:
                    _write(parent, name, record)
                except FileExistsError:
                    _flush_existing(parent, name, record)
        # No network, config mutation, auto-refresh consent or other owner approval.


class _Operation:
    def __init__(self, path, *, ordinary_only=False):
        self.path = path
        self.lease = acquire_storage(path)
        self.resources = []
        self.closed = False
        self.async_contexts = []
        try:
            self.history = _history(path, self.lease)
            if ordinary_only and self.history:
                raise ProviderReconnectRequired()
            self.records = self._records() if self.history else ()
        except BaseException:
            self.lease.close()
            raise

    def _records(self):
        # Filename selection is nonsecret. Never read unrelated receipts or
        # resolve competing credentials to discover which source was reviewed.
        raw = _config(self.path)
        env = (
            raw.get("api_settings", {}).get("openai", {}).get("api_key_env_var")
            or "OPENAI_API_KEY"
        )
        selections = {}
        for source in (
            "config:api_settings.openai.api_key",
            "config:API.openai_api_key",
            "env:" + env,
        ):
            endpoint, source, selection, _ = _selection(self.path, source)
            selections[_receipt_prefix(str(self.path), selection)] = (
                endpoint,
                source,
                selection,
            )
        by_generation = []
        sources = set()
        for witness in self.history:
            store = ActivationStore(Path(witness["store_root"]))
            candidates = []
            with (
                _private(store.root),
                _private(store._generation(witness["generation"])) as parent,
            ):
                names = []
                with os.scandir(parent) as entries:
                    for index, entry in enumerate(entries):
                        if index >= 4096:
                            raise ProviderReconnectRequired()
                        if any(entry.name.startswith(prefix) for prefix in selections):
                            names.append(entry.name)
                names.sort()
                if not names or len(names) > 256:
                    raise ProviderReconnectRequired()
                for name in names:
                    try:
                        record = _Receipt.model_validate(_read(parent, name))
                    except (OSError, ValueError, TypeError, RuntimeError):
                        raise ProviderReconnectRequired() from None
                    expected = selections.get(
                        _receipt_prefix(record.config_selector, record.selection)
                    )
                    if (
                        name != _receipt_name(record)
                        or expected is None
                        or record.generation != witness["generation"]
                        or (record.endpoint, record.auth_source, record.selection)
                        != expected
                    ):
                        raise ProviderReconnectRequired()
                    candidates.append(record)
                    sources.add(record.auth_source)
            by_generation.append(candidates)
        if len(sources) != 1:
            raise ProviderReconnectRequired()
        # Exactly one reviewed source across the paired histories; no fallback.
        _, source, _, secret = _selection(self.path, sources.pop(), resolve=True)
        fingerprint = _digest(secret)
        records = []
        for candidates in by_generation:
            matches = [
                record
                for record in candidates
                if record.credential_digest == fingerprint
            ]
            if len(matches) != 1:
                raise ProviderReconnectRequired()
            records.append(matches[0])
        return tuple(records)

    def check(self):
        if self.closed or self.path != bootstrap.effective_config_path():
            raise ProviderReconnectRequired()
        if _history(self.path, self.lease) != self.history:
            raise ProviderReconnectRequired()

    def check_request(self, url, key):
        self.check()
        if self.history:
            records = self._records()
            if records != self.records or any(
                url
                not in {
                    record.endpoint + "/chat/completions",
                    record.endpoint + "/responses",
                    record.endpoint + "/models",
                }
                or _digest(key) != record.credential_digest
                for record in records
            ):
                raise ProviderReconnectRequired()

    def own(self, resource):
        if not any(item is resource for item in self.resources):
            self.resources.append(resource)

    def close(self):
        if self.closed:
            return
        if self.async_contexts:
            raise ProviderReconnectRequired()  # Native async cleanup did not settle.
        error = None
        for resource in reversed(self.resources):
            try:
                resource.close()
            except BaseException as failure:  # noqa: BLE001 - settle every native resource before propagating failure.
                error = error or failure
        if error is not None:
            raise error  # Native source remains held when actual cleanup fails.
        self.closed = True
        self.lease.close()


def _identity():
    import asyncio

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


_current = ContextVar("openai_connection", default=None)


@contextmanager
def _using(operation):
    operation.check()
    token = _current.set((operation, _identity()))
    try:
        yield operation
    finally:
        _current.reset(token)


class _OpenAIStream(Iterator):
    def __init__(self, iterator, path, operation=None):
        self.iterator = iterator
        self.path = path
        self.operation = operation
        self.lock = threading.RLock()
        self.close_requested = threading.Event()
        self.closed = False

    @property
    def terminal_turn(self):
        return self.iterator.terminal_turn

    @property
    def provider_continuation(self):
        return self.iterator.provider_continuation

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            if self.closed or self.close_requested.is_set():
                self.close()
                raise StopIteration
            if self.operation is None:
                self.operation = _Operation(self.path)
            try:
                with _using(self.operation):
                    value = next(self.iterator)
                if self.close_requested.is_set():
                    self.close()
                    raise StopIteration
                return value
            except BaseException:
                self.close()
                raise

    def close(self):
        self.close_requested.set()
        # Console cancellation may race a native next() in another thread.
        # That worker performs the actual close as soon as its I/O settles.
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.closed:
                return
            try:
                self.iterator.close()
            finally:
                if self.operation is not None:
                    self.operation.close()
            self.closed = True
        finally:
            self.lock.release()


def openai_call(function):
    @wraps(function)
    def call(*args, **kwargs):
        path = bootstrap.effective_config_path()
        operation = _Operation(path)
        try:
            with _using(operation):
                result = function(*args, **kwargs)
            if inspect.isgenerator(result):
                return _OpenAIStream(result, path)
            return result
        finally:
            operation.close()

    return call


def openai_post(session, url, **kwargs):
    """Validate the actual resolved target and key immediately before native I/O."""
    active = _current.get()
    if active is None or active[1] != _identity():
        raise ProviderReconnectRequired()
    operation = active[0]
    key = kwargs.get("headers", {}).get("Authorization", "").removeprefix("Bearer ")
    operation.check_request(url, key)
    operation.own(session)
    if operation.history:
        # Requests otherwise resolves ambient .netrc auth after header review.
        session.trust_env = False
        kwargs["allow_redirects"] = False
    response = session.post(url, **kwargs)
    operation.own(response)
    if operation.history and 300 <= response.status_code < 400:
        raise ProviderReconnectRequired()
    return response


def recovered_settings():
    """Read only actual selected settings; never run the global auth bridge."""
    active = _current.get()
    if active is None or active[1] != _identity() or not active[0].history:
        return None
    operation = active[0]
    operation.check()
    raw = _config(operation.path)
    record = operation.records[0]
    endpoint, _, _, key = _selection(operation.path, record.auth_source, resolve=True)
    operation.check_request(endpoint + "/models", key)
    legacy = raw.get("API", {})
    parameters = {
        name: legacy.get("openai_" + suffix, default)
        for name, suffix, default in (
            ("model", "model", "gpt-5.6-terra"),
            ("streaming", "streaming", False),
            ("temperature", "temperature", 0.7),
            ("top_p", "top_p", 0.95),
            ("max_tokens", "max_tokens", 4096),
            ("api_timeout", "api_timeout", 90),
            ("api_retries", "api_retry", 3),
            ("api_retry_delay", "api_retry_delay", 5),
        )
    }
    connection = {"api_key": key, "api_base_url": endpoint}
    return {
        **raw,
        "openai_api": {**parameters, **connection},
        "api_settings": {"openai": connection},
    }


def catalog_call(function):
    """Bind the installed manual catalog service before credential resolution."""

    @wraps(function)
    async def call(self, *, provider, staged_settings=None, **kwargs):
        from tldw_chatbook.config import get_cli_providers_and_models, load_settings
        from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
            discover_openai_compatible_models,
        )

        unsupported = (
            provider.strip().lower() != "openai"
            or staged_settings is not None
            or self.settings_loader is not load_settings
            or self.provider_catalog_loader is not get_cli_providers_and_models
            or self.discovery_client is not discover_openai_compatible_models
            or self.environ is not os.environ
        )
        operation = _Operation(
            bootstrap.effective_config_path(), ordinary_only=unsupported
        )
        try:
            with _using(operation):
                return await function(
                    self, provider=provider, staged_settings=staged_settings, **kwargs
                )
        finally:
            operation.close()

    return call


def discovery_call(function):
    """Hold the actual async discovery call, including native context cleanup."""

    @wraps(function)
    async def call(*, provider, endpoint, api_key, client=None, **kwargs):
        active = _current.get()
        borrowed = active is not None and active[1] == _identity()
        operation = (
            active[0]
            if borrowed
            else _Operation(
                bootstrap.effective_config_path(),
                ordinary_only=provider.strip().lower() != "openai"
                or client is not None,
            )
        )
        try:
            with _using(operation):
                if operation.history and (
                    provider.strip().lower() != "openai" or client is not None
                ):
                    raise ProviderReconnectRequired()
                if operation.history:
                    operation.check_request(endpoint.rstrip("/") + "/models", api_key)
                return await function(
                    provider=provider,
                    endpoint=endpoint,
                    api_key=api_key,
                    client=client,
                    **kwargs,
                )
        finally:
            if not borrowed:
                operation.close()

    return call


@asynccontextmanager
async def discovery_native_context(context, *, response=False):
    """Retain admission if the actual httpx response/client fails to close."""
    import sys

    active = _current.get()
    if active is None or active[1] != _identity():
        raise ProviderReconnectRequired()
    operation = active[0]
    # httpx's response context owns a response only after send succeeds. Its
    # containing owned client already covers the in-flight send and cleanup.
    if not response:
        operation.async_contexts.append(context)
    resource = await context.__aenter__()
    if response:
        operation.async_contexts.append(context)
    try:
        yield resource
    except BaseException:
        suppressed = await context.__aexit__(*sys.exc_info())
        operation.async_contexts.remove(context)
        if not suppressed:
            raise
    else:
        await context.__aexit__(None, None, None)
        operation.async_contexts.remove(context)


def discovery_client_options():
    """Exclude ambient proxy/auth configuration from the reviewed transport."""
    active = _current.get()
    if active is None or active[1] != _identity():
        raise ProviderReconnectRequired()
    return {"trust_env": False} if active[0].history else {}
