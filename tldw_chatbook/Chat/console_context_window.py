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
# TASK-32923: a 5 s failure TTL re-probed an unreachable or slow endpoint on
# nearly every send, each costing up to the 1 s timeout before the message left.
METADATA_FAILURE_TTL = METADATA_SUCCESS_TTL


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

    def __init__(
        self,
        *,
        timeout: float = 1.0,
        max_entries: int = 64,
        success_ttl: float = METADATA_SUCCESS_TTL,
        failure_ttl: float = METADATA_FAILURE_TTL,
    ) -> None:
        self.timeout = timeout
        self.max_entries = max_entries
        self.success_ttl = success_ttl
        self.failure_ttl = failure_ttl
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, ...], tuple[float, int | None]] = (
            OrderedDict()
        )
        self._pending: dict[tuple[str, ...], Future] = {}

    def needs_refresh(self, target: ContextWindowTarget) -> bool:
        """Report whether this target has no unexpired cached record.

        task-33081: the send path checks this before scheduling a background
        refresh so steady-state sends spawn no probe work at all.

        Args:
            target: Exact provider, endpoint, model, and credential identity.

        Returns:
            True when no unexpired record exists for ``target`` and a
            background refresh would perform real work; False when the
            cached record is still fresh.
        """
        with self._lock:
            record = self._cache.get(target.key)
            return not (record and record[0] > monotonic())

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
        # task-33081: a probe that ran to completion -- whatever it answered
        # -- takes the success TTL: re-asking soon cannot change the answer.
        # Only transport-level failures (exception paths below) deserve the
        # shorter failure TTL. The shipped TTLs are currently equal
        # (TASK-32923); the distinction still governs injected test TTLs and
        # any future divergence.
        stable = True
        try:
            async with asyncio.timeout(self.timeout):
                value = await self._probe(target, client)
        except asyncio.CancelledError:
            # Qodo round: cancellation is not a completed answer. Settle
            # shared waiters with the fallback (the finally block below)
            # but under the SHORT failure TTL -- a cancelled leader must
            # not pin an unverified fallback for a full success TTL --
            # then propagate the cancellation.
            stable = False
            raise
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError):
            stable = False
        finally:
            with self._lock:
                ttl = self.success_ttl if stable else self.failure_ttl
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
            # Not "openrouter": its model list (~750 KB) always exceeds
            # METADATA_MAX_BYTES, so the probe could only fail; the model
            # catalog resolves OpenRouter windows instead (TASK-32923).
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
            content_length = response.headers.get("Content-Length", "")
            if content_length.isdigit() and int(content_length) > METADATA_MAX_BYTES:
                # task-33081: declared-oversized payload -- abort before
                # downloading a body that can only be discarded.
                return None
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=METADATA_CHUNK_BYTES):
                if len(body) + len(chunk) > METADATA_MAX_BYTES:
                    return None
                body.extend(chunk)
        try:
            payload = json.loads(body)
        except ValueError:
            # task-33081: a completed response with an unparseable body is an
            # answer, not a transport failure -- retrying it on the failure
            # TTL would re-download the same garbage.
            return None
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


_CONTEXT_CAPACITY_CACHE_ORIGINAL = (
    globals(), ContextWindowCache, ContextWindowCache.cached,
    ContextWindowCache.cached.__code__,
)
