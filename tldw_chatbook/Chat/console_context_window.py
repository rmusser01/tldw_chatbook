"""Shared context limits and bounded, ephemeral serving-metadata discovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from time import monotonic

import httpx

from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    check_url_or_raise_async,
    origin_set,
)
from tldw_chatbook.Utils.token_counter import (
    ContextWindowResolution,
    positive_window,
    resolve_context_window,
)

from .provider_endpoint_contract import resolve_provider_endpoint

METADATA_MAX_BYTES = 256 * 1024
METADATA_CHUNK_BYTES = 64 * 1024
METADATA_SUCCESS_TTL = 60
METADATA_FAILURE_TTL = 5


@dataclass(frozen=True, slots=True)
class ContextWindowTarget:
    """An exact selected server/model, with credentials excluded from repr."""

    owner: str
    family: str
    endpoint: str
    model: str
    api_key: str | None = field(default=None, repr=False)

    @property
    def key(self) -> tuple[str, ...]:
        credential = hashlib.sha256((self.api_key or "").encode()).hexdigest()
        endpoint = resolve_provider_endpoint(
            "custom" if self.family == "openrouter" else self.family, self.endpoint
        )
        return (
            self.owner,
            self.family,
            endpoint.models_url or self.endpoint,
            self.model,
            credential,
        )


class ContextWindowCache:
    """Bounded cross-loop cache; callers own all in-flight async work."""

    def __init__(self, *, timeout: float = 1.0, max_entries: int = 64) -> None:
        self.timeout = timeout
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, ...], tuple[float, int | None]] = (
            OrderedDict()
        )
        self._pending: dict[tuple[str, ...], Future] = {}

    def cached(self, target: ContextWindowTarget) -> ContextWindowResolution:
        """Read current cached capacity without starting network work.

        Args:
            target: Exact provider, endpoint, model, and credential identity.

        Returns:
            The unexpired server capacity, or the model/API/system fallback
            when no current serving metadata is available.
        """
        with self._lock:
            record = self._cache.get(target.key)
            value = record[1] if record and record[0] > monotonic() else None
        return resolve_context_window(target.family, target.model, server_tokens=value)

    async def resolve(
        self, target: ContextWindowTarget, client: httpx.AsyncClient
    ) -> ContextWindowResolution:
        """Resolve capacity with bounded, shared asynchronous metadata work.

        Args:
            target: Exact provider, endpoint, model, and credential identity.
            client: Caller-owned HTTP client; this method does not close it.

        Returns:
            Cached or discovered server capacity, falling back to the model,
            API, or system default when metadata is unavailable or denied.

        Raises:
            asyncio.CancelledError: If the caller cancels; shared waiters
                still settle without cancelling one another.
        """
        key = target.key
        with self._lock:
            record = self._cache.get(key)
            if record and record[0] > monotonic():
                self._cache.move_to_end(key)
                return resolve_context_window(
                    target.family, target.model, server_tokens=record[1]
                )
            pending = self._pending.get(key)
            leader = pending is None
            if leader:
                # Bound concurrent identities as well as completed entries.
                if len(self._pending) >= self.max_entries:
                    return resolve_context_window(target.family, target.model)
                pending = self._pending[key] = Future()
        if not leader:
            value = await asyncio.shield(asyncio.wrap_future(pending))
            return resolve_context_window(
                target.family, target.model, server_tokens=value
            )
        value = None
        try:
            async with asyncio.timeout(self.timeout):
                value = await self._probe(target, client)
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError):
            pass
        finally:
            with self._lock:
                ttl = METADATA_SUCCESS_TTL if value else METADATA_FAILURE_TTL
                self._cache[key] = (monotonic() + ttl, value)
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
                self._pending.pop(key, None)
                pending.set_result(value)
        return resolve_context_window(target.family, target.model, server_tokens=value)

    async def _probe(
        self, target: ContextWindowTarget, client: httpx.AsyncClient
    ) -> int | None:
        if not target.model or not target.endpoint:
            return None
        family = target.family
        supported = {
            "llama_cpp",
            "local_llamacpp",
            "llamafile",
            "local_llamafile",
            "vllm",
            "local_vllm",
            "ollama",
            "local_ollama",
            "custom",
            "custom_2",
            "tabbyapi",
            "aphrodite",
            "koboldcpp",
            "oobabooga",
            "openrouter",
        }
        if family not in supported:
            return None
        endpoint = resolve_provider_endpoint(
            "custom" if family == "openrouter" else family, target.endpoint
        )
        if endpoint.errors or not endpoint.models_url:
            return None
        root = endpoint.models_url.removesuffix("/v1/models")
        if family in {"llama_cpp", "local_llamacpp", "llamafile", "local_llamafile"}:
            url = root + "/props"
        elif family in {"ollama", "local_ollama"}:
            url = root + "/api/ps"
        else:
            url = endpoint.models_url
        try:
            await check_url_or_raise_async(
                url, trusted_origins=origin_set(target.endpoint)
            )
        except EgressBlockedError:
            return None
        headers = (
            {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
        )
        # llama.cpp router reads the selected model from this query; metadata
        # inspection must not load a model as a side effect.
        params = (
            {"model": target.model, "autoload": "false"}
            if url.endswith("/props")
            else None
        )
        async with client.stream(
            "GET",
            url,
            headers=headers,
            params=params,
            timeout=self.timeout,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                return None
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=METADATA_CHUNK_BYTES):
                if len(body) + len(chunk) > METADATA_MAX_BYTES:
                    return None
                body.extend(chunk)
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return None
        if url.endswith("/props"):
            settings = payload.get("default_generation_settings")
            return (
                positive_window(settings.get("n_ctx"))
                if isinstance(settings, dict)
                else None
            )
        records = payload.get("models" if url.endswith("/api/ps") else "data", [])
        if not isinstance(records, list):
            return None
        for record in records:
            if not isinstance(record, dict) or target.model not in (
                record.get("id"),
                record.get("name"),
                record.get("model"),
            ):
                continue
            for name in (
                "max_model_len",
                "context_length",
                "context_window",
                "max_context_length",
            ):
                value = positive_window(record.get(name))
                if value is not None:
                    return value
        return None
