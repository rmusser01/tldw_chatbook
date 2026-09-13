"""TASK-26023: models.dev as a lower-priority merge layer.

Pure parse + conditional-fetch + gap-fill lookup, all with the network
mocked. The upstream is only ever a FALLBACK beneath hand-maintained
entries, fetched explicitly/in-background (never on a lookup), and honest
about unknowns.
"""

from __future__ import annotations

import json
from pathlib import Path

from tldw_chatbook.LLM_Provider_Catalog.models_dev_catalog import (
    ModelsDevCache,
    fetch_models_dev,
    parse_models_dev,
)


_SAMPLE = {
    "anthropic": {
        "models": {
            "claude-opus-5": {
                "limit": {"context": 200000, "output": 64000},
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "cost": {"input": 5.0, "output": 25.0},
            },
            "claude-haiku-4-5": {
                "limit": {"context": 200000},
                "modalities": {"input": ["text"]},
                "cost": {"input": 0.8, "output": 4.0},
            },
        }
    },
    "fictprov": {
        "models": {
            "fict-model-xyz9": {
                "limit": {"context": 128000},
                "modalities": {"input": ["text", "image"]},
                "cost": {"input": 0.3, "output": 1.2},
            }
        }
    },
}


def test_parse_maps_context_vision_and_price():
    catalog = parse_models_dev(_SAMPLE)
    opus = catalog[("anthropic", "claude-opus-5")]
    assert opus.context_window == 200000
    assert opus.supports_vision is True
    assert opus.input_price_per_mtok == 5.0
    assert opus.output_price_per_mtok == 25.0
    assert opus.source == "models.dev"

    haiku = catalog[("anthropic", "claude-haiku-4-5")]
    assert haiku.supports_vision is False
    assert haiku.context_window == 200000


def test_parse_is_defensive_about_junk():
    assert parse_models_dev({}) == {}
    assert parse_models_dev({"p": {"models": "not-a-dict"}}) == {}
    assert parse_models_dev(None) == {}
    partial = parse_models_dev(
        {"p": {"models": {"m": {"cost": {"input": 1.0}}}}}
    )
    entry = partial[("p", "m")]
    assert entry.input_price_per_mtok == 1.0
    assert entry.context_window is None
    assert entry.output_price_per_mtok is None


def test_conditional_get_stores_etag_and_honors_304(tmp_path: Path):
    calls = []

    def fake_get(url, headers):
        calls.append(dict(headers))
        if "If-None-Match" in headers:
            return 304, {}, b""
        return 200, {"ETag": '"v1"'}, json.dumps(_SAMPLE).encode()

    path = tmp_path / "models_dev.json"
    fetch_models_dev(disk_path=path, http_get=fake_get)
    assert path.exists()
    cache = ModelsDevCache.load(path)
    assert cache.etag == '"v1"'
    assert cache.lookup("anthropic", "claude-opus-5") is not None

    # second fetch sends the stored ETag; a 304 keeps the cached body
    fetch_models_dev(disk_path=path, http_get=fake_get)
    assert calls[1].get("If-None-Match") == '"v1"'
    cache2 = ModelsDevCache.load(path)
    assert cache2.lookup("anthropic", "claude-opus-5") is not None


def test_fetch_never_raises_on_network_error(tmp_path: Path):
    def boom(url, headers):
        raise ConnectionError("down")

    path = tmp_path / "models_dev.json"
    # no exception, no file written
    fetch_models_dev(disk_path=path, http_get=boom)
    assert not path.exists()


def test_offline_cold_start_uses_cache_then_nothing(tmp_path: Path):
    path = tmp_path / "models_dev.json"
    # nothing on disk, no network: lookup is simply empty (AC#3/#7)
    cache = ModelsDevCache.load(path)
    assert cache.lookup("anthropic", "claude-opus-5") is None


def _seed_cache(tmp_path, monkeypatch, *, enabled=True):
    """Point the gap-fill layer at a temp cache with the sample data."""
    import tldw_chatbook.LLM_Provider_Catalog.models_dev_catalog as mdc

    path = tmp_path / "models_dev.json"

    def fake_get(url, headers):
        return 200, {"ETag": '"v1"'}, json.dumps(_SAMPLE).encode()

    fetch_models_dev(disk_path=path, http_get=fake_get)
    monkeypatch.setattr(mdc, "default_cache_path", lambda: path)
    monkeypatch.setattr(mdc, "_enabled", lambda: enabled)
    mdc.reset_memory_cache()
    return mdc


def test_pricing_gap_fill_uses_models_dev_when_enabled(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch, enabled=True)
    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()
    # a model with no hand-maintained entry falls back to models.dev
    pricing = catalog.get_pricing("fictprov", "fict-model-xyz9")
    assert pricing is not None
    assert pricing.input_per_mtok == 0.3
    assert pricing.as_of == "models.dev", "origin must be inspectable (AC#5)"


def test_pricing_gap_fill_is_off_by_default(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch, enabled=False)
    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()
    # disabled => the same honest None as today (AC#6/#7)
    assert catalog.get_pricing("fictprov", "fict-model-xyz9") is None


def test_hand_maintained_pricing_always_wins(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch, enabled=True)
    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()
    # claude-haiku-4-5 is in BOTH the models.dev fixture (input 0.8) AND
    # the hand-maintained Anthropic patterns -- the hand-maintained rate
    # must win with its own as_of, never the models.dev 0.8/"models.dev".
    seeded = catalog.get_pricing("anthropic", "claude-haiku-4-5")
    assert seeded is not None, "the colliding model must resolve"
    assert seeded.as_of != "models.dev", "local override must win (AC#2)"
    assert seeded.input_per_mtok != 0.8, "must not use the models.dev rate"


def test_capability_gap_fill_context_window(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch, enabled=True)
    from tldw_chatbook.model_capabilities import ModelCapabilities

    caps = ModelCapabilities()
    window = caps.get_context_window("fictprov", "fict-model-xyz9")
    assert window == 128000
    info = caps.get_model_capabilities("fictprov", "fict-model-xyz9")
    assert info["vision"] is True
    assert info["source"] == "models.dev"


def test_capability_gap_fill_off_by_default(tmp_path, monkeypatch):
    _seed_cache(tmp_path, monkeypatch, enabled=False)
    from tldw_chatbook.model_capabilities import ModelCapabilities

    caps = ModelCapabilities()
    assert caps.get_context_window("fictprov", "fict-model-xyz9") is None


def test_gap_fill_price_requires_both_input_and_output(tmp_path, monkeypatch):
    """Review minor 3: a models.dev entry missing output price must not
    produce a half-price ($0 output); it stays honestly unpriced."""
    import tldw_chatbook.LLM_Provider_Catalog.models_dev_catalog as mdc

    sample = {
        "partialprov": {
            "models": {
                "input-only-model": {
                    "limit": {"context": 100000},
                    "cost": {"input": 2.0},
                }
            }
        }
    }
    path = tmp_path / "models_dev.json"
    fetch_models_dev(
        disk_path=path,
        http_get=lambda url, headers: (200, {"ETag": '"v"'}, json.dumps(sample).encode()),
    )
    monkeypatch.setattr(mdc, "default_cache_path", lambda: path)
    monkeypatch.setattr(mdc, "_enabled", lambda: True)
    mdc.reset_memory_cache()

    from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

    # input-only => no gap-fill price (honest, not a $0-output half price)
    assert PricingCatalog().get_pricing("partialprov", "input-only-model") is None
