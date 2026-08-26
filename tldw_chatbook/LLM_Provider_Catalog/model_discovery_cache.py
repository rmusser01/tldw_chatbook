"""App-lifetime runtime cache for manually discovered models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from threading import RLock

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel

_MODEL_ID_MAX_CHARS = 120


class ModelDiscoveryCache:
    """Store discovered model snapshots by provider key and endpoint fingerprint."""

    def __init__(self, *, max_snapshots: int = 128, max_models: int = 4096) -> None:
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= 4096:
            raise ValueError("max_snapshots is invalid")
        if type(max_models) is not int or not 1 <= max_models <= 100_000:
            raise ValueError("max_models is invalid")
        self._max_snapshots = max_snapshots
        self._max_models = max_models
        self._models_by_provider_endpoint: OrderedDict[
            tuple[str, str], tuple[DiscoveredModel, ...]
        ] = OrderedDict()
        self._model_count = 0
        self._lock = RLock()

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._models_by_provider_endpoint)

    @property
    def model_count(self) -> int:
        with self._lock:
            return self._model_count

    def replace(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        models: Iterable[DiscoveredModel],
    ) -> None:
        """Replace one provider/endpoint snapshot with immutable model results."""
        key = self._snapshot_key(provider_list_key, endpoint_fingerprint)
        snapshot_items: list[DiscoveredModel] = []
        for item in models:
            if len(snapshot_items) >= self._max_models:
                raise ValueError("model snapshot exceeds cache bounds")
            self._validate_model(item)
            snapshot_items.append(item)
        snapshot = tuple(snapshot_items)
        with self._lock:
            previous = self._models_by_provider_endpoint.pop(key, ())
            self._model_count -= len(previous)
            self._models_by_provider_endpoint[key] = snapshot
            self._model_count += len(snapshot)
            while (
                len(self._models_by_provider_endpoint) > self._max_snapshots
                or self._model_count > self._max_models
            ):
                _, evicted = self._models_by_provider_endpoint.popitem(last=False)
                self._model_count -= len(evicted)

    def list(
        self,
        provider_list_key: str | None = None,
        endpoint_fingerprint: str | None = None,
    ) -> tuple[DiscoveredModel, ...]:
        """Return cached models filtered by provider key and/or endpoint fingerprint."""
        provider_filter = self._optional_identity(provider_list_key, 128)
        endpoint_filter = self._optional_identity(endpoint_fingerprint, 512)
        with self._lock:
            models: list[DiscoveredModel] = []
            touched: list[tuple[str, str]] = []
            for (
                cached_provider,
                cached_endpoint,
            ), cached_models in self._models_by_provider_endpoint.items():
                if provider_filter is not None and cached_provider != provider_filter:
                    continue
                if endpoint_filter is not None and cached_endpoint != endpoint_filter:
                    continue
                models.extend(cached_models)
                touched.append((cached_provider, cached_endpoint))
            for key in touched:
                self._models_by_provider_endpoint.move_to_end(key)
            return tuple(models)

    def has_snapshot(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
    ) -> bool:
        """Return whether a snapshot exists, including an empty snapshot."""
        key = self._snapshot_key(provider_list_key, endpoint_fingerprint)
        with self._lock:
            if key not in self._models_by_provider_endpoint:
                return False
            self._models_by_provider_endpoint.move_to_end(key)
            return True

    def clear(self, provider_list_key: str | None = None) -> None:
        """Clear all cached models, or only one exact provider key."""
        with self._lock:
            if provider_list_key is None:
                self._models_by_provider_endpoint.clear()
                self._model_count = 0
                return
            provider_filter = self._required_identity(provider_list_key, 128)
            for key in tuple(self._models_by_provider_endpoint):
                if key[0] == provider_filter:
                    self._model_count -= len(self._models_by_provider_endpoint.pop(key))

    @staticmethod
    def _required_identity(value: object, maximum: int) -> str:
        if type(value) is not str:
            raise TypeError("cache identity is invalid")
        normalized = value.strip()
        if not normalized or len(normalized) > maximum or not normalized.isprintable():
            raise ValueError("cache identity is invalid")
        return normalized

    @classmethod
    def _optional_identity(cls, value: object | None, maximum: int) -> str | None:
        return None if value is None else cls._required_identity(value, maximum)

    @classmethod
    def _snapshot_key(cls, provider: object, endpoint: object) -> tuple[str, str]:
        return (
            cls._required_identity(provider, 128),
            cls._required_identity(endpoint, 512),
        )

    @staticmethod
    def _validate_model(item: object) -> None:
        if type(item) is not DiscoveredModel:
            raise TypeError("model snapshot is invalid")
        model_id = item.model_id
        if (
            type(model_id) is not str
            or not model_id
            or model_id != model_id.strip()
            or len(model_id) > _MODEL_ID_MAX_CHARS
            or not model_id.isprintable()
        ):
            raise ValueError("model snapshot is invalid")
