import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.LLM_Provider_Catalog import model_discovery_disk_cache as disk_module
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)

_EXPECTED_MAX_BYTES = 2 * 1024 * 1024
_EXPECTED_MAX_ENTRIES = 128
_EXPECTED_MAX_MODELS_PER_ENTRY = 100


def _store(tmp_path):
    return ModelCatalogDiskStore(tmp_path / "model_catalog_cache.json")


def test_round_trip_into_memory_cache(tmp_path):
    store = _store(tmp_path)
    store.record(
        "OpenAI",
        "https://api.openai.com/v1",
        ["gpt-a", "gpt-b"],
        fetched_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
    )
    store.save()

    cache = ModelDiscoveryCache()
    reloaded = _store(tmp_path)
    reloaded.load_into(cache)
    models = cache.list("OpenAI", "https://api.openai.com/v1")
    assert [m.model_id for m in models] == ["gpt-a", "gpt-b"]
    assert all(m.source == "runtime_discovered" for m in models)
    assert reloaded.fetched_at("OpenAI", "https://api.openai.com/v1") == datetime(
        2026, 7, 17, 12, 0, tzinfo=UTC
    )


def test_missing_and_corrupt_files_load_empty(tmp_path):
    cache = ModelDiscoveryCache()
    _store(tmp_path).load_into(cache)  # missing
    (tmp_path / "model_catalog_cache.json").write_text("{not json", encoding="utf-8")
    _store(tmp_path).load_into(cache)  # corrupt
    assert cache.list() == ()


def test_staleness_boundaries(tmp_path):
    store = _store(tmp_path)
    now = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    store.record(
        "OpenAI", "fp", ["gpt-a"], fetched_at=now - timedelta(hours=23, minutes=59)
    )
    assert store.is_stale("OpenAI", "fp", stale_after_hours=24, now=now) is False
    store.record(
        "OpenAI", "fp", ["gpt-a"], fetched_at=now - timedelta(hours=24, minutes=1)
    )
    assert store.is_stale("OpenAI", "fp", stale_after_hours=24, now=now) is True
    assert store.is_stale("OpenAI", "fp", stale_after_hours=0, now=now) is True
    assert store.is_stale("Nobody", "fp", stale_after_hours=24, now=now) is True


def test_prune_drops_unconfigured_providers(tmp_path):
    store = _store(tmp_path)
    store.record("OpenAI", "fp1", ["gpt-a"])
    store.record("Ghost", "fp2", ["ghost-model"])
    store.prune({"OpenAI"})
    assert store.fetched_at("Ghost", "fp2") is None
    assert store.fetched_at("OpenAI", "fp1") is not None


def test_disk_store_holds_no_credentials(tmp_path):
    store = _store(tmp_path)
    store.record("OpenAI", "fp", ["gpt-a"])
    store.save()
    raw = (tmp_path / "model_catalog_cache.json").read_text(encoding="utf-8")
    assert (
        "api_key" not in raw and "Authorization" not in raw and "x-api-key" not in raw
    )


def test_save_uses_pid_scoped_temp_name(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.record("OpenAI", "fp", ["gpt-a"])
    captured = {}
    real_replace = os.replace

    def fake_replace(src, dst):
        captured["src"] = Path(src)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fake_replace)
    store.save()
    assert captured["src"].name == f"model_catalog_cache.json.{os.getpid()}.tmp"
    assert (tmp_path / "model_catalog_cache.json").exists()


def test_future_dated_fetched_at_is_stale(tmp_path):
    store = _store(tmp_path)
    now = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    store.record("OpenAI", "fp", ["gpt-a"], fetched_at=now + timedelta(days=7))
    assert store.is_stale("OpenAI", "fp", stale_after_hours=24, now=now) is True


def test_empty_model_list_round_trips_with_timestamp(tmp_path):
    store = _store(tmp_path)
    stamp = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    store.record("OpenAI", "fp", [], fetched_at=stamp)
    store.save()

    cache = ModelDiscoveryCache()
    reloaded = _store(tmp_path)
    reloaded.load_into(cache)
    assert reloaded.fetched_at("OpenAI", "fp") == stamp
    assert cache.list() == ()


def test_save_creates_missing_parent_directories(tmp_path):
    store = ModelCatalogDiskStore(
        tmp_path / "nested" / "deeper" / "model_catalog_cache.json"
    )
    store.record("OpenAI", "fp", ["gpt-a"])
    store.save()
    assert (tmp_path / "nested" / "deeper" / "model_catalog_cache.json").exists()


def _disk_entry(provider: str, endpoint: str, models: list[object]) -> dict:
    return {
        "provider_list_key": provider,
        "endpoint_fingerprint": endpoint,
        "fetched_at": "2026-07-17T12:00:00Z",
        "models": models,
    }


def test_oversized_file_is_rejected_before_json_parsing(tmp_path, monkeypatch):
    path = tmp_path / "model_catalog_cache.json"
    path.write_bytes(b"{" + b"x" * _EXPECTED_MAX_BYTES)
    called = False

    def forbidden_loads(_value):
        nonlocal called
        called = True
        raise AssertionError("JSON parser must not receive an oversized file")

    monkeypatch.setattr(disk_module.json, "loads", forbidden_loads)
    cache = ModelDiscoveryCache()
    _store(tmp_path).load_into(cache)

    assert called is False
    assert cache.list() == ()


def test_load_caps_entry_count_deterministically(tmp_path):
    entries = {
        f"entry-{index}": _disk_entry("Custom", f"fp-{index}", [f"model-{index}"])
        for index in range(_EXPECTED_MAX_ENTRIES + 1)
    }
    (tmp_path / "model_catalog_cache.json").write_text(
        json.dumps({"version": 1, "entries": entries}), encoding="utf-8"
    )
    cache = ModelDiscoveryCache(max_snapshots=_EXPECTED_MAX_ENTRIES + 1)

    _store(tmp_path).load_into(cache)

    assert cache.snapshot_count == _EXPECTED_MAX_ENTRIES
    assert cache.has_snapshot("Custom", "fp-0")
    assert not cache.has_snapshot("Custom", f"fp-{_EXPECTED_MAX_ENTRIES}")


def test_invalid_duplicate_entry_preserves_existing_and_later_valid_loads(tmp_path):
    cache = ModelDiscoveryCache()
    cache.replace(
        "Custom",
        "same-key",
        (
            disk_module.DiscoveredModel(
                provider="Custom",
                provider_list_key="Custom",
                model_id="existing",
                display_name="existing",
                source="runtime_discovered",
                endpoint_fingerprint="same-key",
                discovered_at="2026-07-17T12:00:00Z",
            ),
        ),
    )
    payload = {
        "version": 1,
        "entries": {
            "invalid-same-key": _disk_entry(
                "Custom", "same-key", ["candidate", "unsafe\nmodel"]
            ),
            "later-valid": _disk_entry("Custom", "later-key", ["later-model"]),
        },
    }
    (tmp_path / "model_catalog_cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    store = _store(tmp_path)

    store.load_into(cache)

    assert [item.model_id for item in cache.list("Custom", "same-key")] == ["existing"]
    assert [item.model_id for item in cache.list("Custom", "later-key")] == [
        "later-model"
    ]
    assert store.fetched_at("Custom", "same-key") is None
    assert store.fetched_at("Custom", "later-key") is not None


@pytest.mark.parametrize("unsafe_model_id", ("", "x" * 121, "unsafe\nmodel", 42))
def test_invalid_model_id_rejects_only_its_entry_and_preserves_same_key(
    tmp_path, unsafe_model_id
):
    cache = ModelDiscoveryCache()
    cache.replace(
        "Custom",
        "same-key",
        (
            disk_module.DiscoveredModel(
                provider="Custom",
                provider_list_key="Custom",
                model_id="existing",
                display_name="existing",
                source="runtime_discovered",
                endpoint_fingerprint="same-key",
                discovered_at="2026-07-17T12:00:00Z",
            ),
        ),
    )
    payload = {
        "version": 1,
        "entries": {
            "invalid": _disk_entry("Custom", "same-key", [unsafe_model_id]),
            "later": _disk_entry("Custom", "later-key", ["later-model"]),
        },
    }
    (tmp_path / "model_catalog_cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    _store(tmp_path).load_into(cache)

    assert [item.model_id for item in cache.list("Custom", "same-key")] == ["existing"]
    assert [item.model_id for item in cache.list("Custom", "later-key")] == [
        "later-model"
    ]


def test_oversized_entry_does_not_abort_later_entry_or_replace_existing(tmp_path):
    cache = ModelDiscoveryCache(max_models=_EXPECTED_MAX_MODELS_PER_ENTRY)
    cache.replace("Custom", "same-key", ())
    payload = {
        "version": 1,
        "entries": {
            "oversized": _disk_entry(
                "Custom",
                "same-key",
                [
                    f"model-{index}"
                    for index in range(_EXPECTED_MAX_MODELS_PER_ENTRY + 1)
                ],
            ),
            "later": _disk_entry("Custom", "later-key", ["later-model"]),
        },
    }
    (tmp_path / "model_catalog_cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    _store(tmp_path).load_into(cache)

    assert cache.has_snapshot("Custom", "same-key")
    assert cache.list("Custom", "same-key") == ()
    assert [item.model_id for item in cache.list("Custom", "later-key")] == [
        "later-model"
    ]


def test_cache_replace_failure_isolated_from_later_disk_entry(tmp_path):
    payload = {
        "version": 1,
        "entries": {
            "too-large-for-runtime": _disk_entry("Custom", "first", ["one", "two"]),
            "later": _disk_entry("Custom", "later", ["three"]),
        },
    }
    (tmp_path / "model_catalog_cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    cache = ModelDiscoveryCache(max_models=1)

    _store(tmp_path).load_into(cache)

    assert cache.list("Custom", "first") == ()
    assert [item.model_id for item in cache.list("Custom", "later")] == ["three"]


def test_disk_cache_diagnostics_are_bounded_and_secret_free(tmp_path):
    secret = "disk-cache-secret-canary"
    payload = {
        "version": 1,
        "entries": {
            "hostile": _disk_entry(
                "Custom",
                f"https://user:{secret}@example.test/v1?token={secret}",
                ["model"],
            ),
            "later": _disk_entry("Custom", "later", ["safe-model"]),
        },
    }
    (tmp_path / "model_catalog_cache.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(str(message)), level="WARNING")
    try:
        cache = ModelDiscoveryCache()
        _store(tmp_path).load_into(cache)
    finally:
        logger.remove(sink)

    text = "".join(logged)
    assert secret not in text
    assert "user:" not in text
    assert len(text) < 1000
    assert [item.model_id for item in cache.list("Custom", "later")] == ["safe-model"]


def test_record_stops_infinite_duplicate_iterable_at_max_plus_one(tmp_path):
    store = _store(tmp_path)
    store.record("Custom", "endpoint", ["existing-model"])
    consumed = 0

    def duplicate_ids():
        nonlocal consumed
        while True:
            consumed += 1
            yield "same-model"

    with pytest.raises(ValueError, match="bounds"):
        store.record("Custom", "endpoint", duplicate_ids())

    assert consumed == _EXPECTED_MAX_MODELS_PER_ENTRY + 1
    store.save()
    cache = ModelDiscoveryCache()
    _store(tmp_path).load_into(cache)
    assert [item.model_id for item in cache.list("Custom", "endpoint")] == [
        "existing-model"
    ]
