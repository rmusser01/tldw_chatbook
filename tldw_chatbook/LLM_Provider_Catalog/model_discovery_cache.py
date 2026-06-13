"""App-lifetime runtime cache for manually discovered models."""

from __future__ import annotations

from collections.abc import Iterable

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel


class ModelDiscoveryCache:
    """Store discovered model snapshots by provider key and endpoint fingerprint."""

    def __init__(self) -> None:
        self._models_by_provider_endpoint: dict[tuple[str, str], tuple[DiscoveredModel, ...]] = {}

    def replace(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        models: Iterable[DiscoveredModel],
    ) -> None:
        """Replace one provider/endpoint snapshot with immutable model results."""
        key = (str(provider_list_key), str(endpoint_fingerprint))
        self._models_by_provider_endpoint[key] = tuple(models)

    def list(
        self,
        provider_list_key: str | None = None,
        endpoint_fingerprint: str | None = None,
    ) -> tuple[DiscoveredModel, ...]:
        """Return cached models filtered by provider key and/or endpoint fingerprint."""
        provider_filter = None if provider_list_key is None else str(provider_list_key)
        endpoint_filter = None if endpoint_fingerprint is None else str(endpoint_fingerprint)
        models: list[DiscoveredModel] = []
        for (cached_provider, cached_endpoint), cached_models in (
            self._models_by_provider_endpoint.items()
        ):
            if provider_filter is not None and cached_provider != provider_filter:
                continue
            if endpoint_filter is not None and cached_endpoint != endpoint_filter:
                continue
            models.extend(cached_models)
        return tuple(models)

    def clear(self, provider_list_key: str | None = None) -> None:
        """Clear all cached models, or only one exact provider key."""
        if provider_list_key is None:
            self._models_by_provider_endpoint.clear()
            return
        provider_filter = str(provider_list_key)
        for key in tuple(self._models_by_provider_endpoint):
            if key[0] == provider_filter:
                del self._models_by_provider_endpoint[key]
