"""Detached pricing for one checked Console display; never live action data."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from inspect import getattr_static
from types import MappingProxyType
from typing import Mapping

from ...LLM_Calls import pricing_catalog as pricing
from ...LLM_Provider_Catalog import models_dev_catalog as upstream

_PRICING_SOURCE = pricing._DISPLAY_PRICING_SOURCE
_CATALOG_SOURCE = upstream._DISPLAY_CATALOG_SOURCE
_PRICING_CLASS = pricing.PricingCatalog
_PRICE_CLASS = pricing.ModelPricing
_CACHE_CLASS = upstream.ModelsDevCache
_ENTRY_CLASS = upstream.ModelsDevEntry
_MODULE = sys.modules[__name__]


def _bindings_current(module, rows) -> bool:
    for owner, name, original, code, defaults, kwdefaults in rows:
        current = getattr_static(owner or module, name, None)
        if isinstance(current, (staticmethod, classmethod)):
            current = current.__func__
        if (
            current is not original
            or original.__code__ is not code
            or original.__defaults__ is not defaults
            or original.__kwdefaults__ is not kwdefaults
        ):
            return False
    return True


def _stock_catalog_current(catalog) -> bool:
    return (
        sys.modules.get(pricing.__name__) is pricing
        and sys.modules.get(upstream.__name__) is upstream
        and pricing.PricingCatalog is _PRICING_CLASS
        and pricing.ModelPricing is _PRICE_CLASS
        and upstream.ModelsDevCache is _CACHE_CLASS
        and upstream.ModelsDevEntry is _ENTRY_CLASS
        and type(catalog) is pricing.PricingCatalog
        and catalog is pricing._global_catalog is pricing._global_catalog_owner
        and pricing._DISPLAY_PRICING_SOURCE is _PRICING_SOURCE
        and upstream._DISPLAY_CATALOG_SOURCE is _CATALOG_SOURCE
        and _bindings_current(pricing, _PRICING_SOURCE)
        and _bindings_current(upstream, _CATALOG_SOURCE)
        and not any(
            name in vars(catalog)
            for name in (
                "get_pricing",
                "_configured_pricing",
                "_to_model_pricing",
                "cost_for_usage",
            )
        )
    )


@dataclass(frozen=True)
class DisplayPricingCatalog(pricing.PricingCatalog):
    """An explicit price-only view; inherited usage arithmetic stays unchanged."""

    source_catalog: pricing.PricingCatalog = field(repr=False)
    upstream_prices: Mapping[tuple[str, str], pricing.ModelPricing] = field(repr=False)

    def get_pricing(self, provider: str, model: str) -> pricing.ModelPricing | None:
        provider_key = pricing.provider_config_key(provider)
        model_key = (model or "").strip().lower()
        configured = self.source_catalog._configured_pricing(provider_key, model_key)
        if configured is not pricing._NO_CONFIGURED_PRICE:
            return configured
        return self.upstream_prices.get((provider_key, model_key))


_DISPLAY_CLASS = DisplayPricingCatalog


def prepare_pricing_display() -> DisplayPricingCatalog | None:
    """Read optional upstream metadata inside the caller's finite config worker."""
    if not _bindings_current(pricing, _PRICING_SOURCE):
        return None
    catalog = pricing.get_pricing_catalog()
    if not _stock_catalog_current(catalog):
        return None
    cache = upstream._memory_cache() if upstream._enabled() else None
    # A native read above may have yielded to a replacement. Inspect captured
    # functions/classes without dispatching any newly installed helper first.
    if (
        sys.modules.get(__name__) is not _MODULE
        or DisplayPricingCatalog is not _DISPLAY_CLASS
        or any(
            getattr_static(owner or _MODULE, name, None) is not original
            or original.__code__ is not code
            or original.__defaults__ is not defaults
            or original.__kwdefaults__ is not kwdefaults
            for owner, name, original, code, defaults, kwdefaults in _SOURCE_CAPSULE
        )
    ):
        return None
    if not _stock_catalog_current(catalog):
        return None
    prices = {}
    if cache is not None:
        if (
            type(cache) is not upstream.ModelsDevCache
            or type(cache.catalog) is not dict  # noqa: E721 -- refuse injected mappings.
        ):
            return None
        for key, entry in cache.catalog.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or any(type(item) is not str for item in key)  # noqa: E721 -- detached stock keys only.
                or type(entry) is not upstream.ModelsDevEntry
            ):
                return None
            converted = pricing._pricing_from_models_dev_entry(entry)
            if converted is not None:
                prices[key] = converted
    return DisplayPricingCatalog(catalog, MappingProxyType(prices))


def pricing_display_current(value) -> bool:
    """Check stock source and exact catalog ownership without invoking readers."""
    return type(value) is DisplayPricingCatalog and _stock_catalog_current(
        value.source_catalog
    )


_SOURCE_CAPSULE = tuple(
    (
        owner,
        name,
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
    )
    for owner, name, function in (
        (None, "_bindings_current", _bindings_current),
        (None, "_stock_catalog_current", _stock_catalog_current),
        (None, "prepare_pricing_display", prepare_pricing_display),
        (None, "pricing_display_current", pricing_display_current),
        (DisplayPricingCatalog, "get_pricing", DisplayPricingCatalog.get_pricing),
        (DisplayPricingCatalog, "__init__", DisplayPricingCatalog.__init__),
        (DisplayPricingCatalog, "__eq__", DisplayPricingCatalog.__eq__),
    )
)
