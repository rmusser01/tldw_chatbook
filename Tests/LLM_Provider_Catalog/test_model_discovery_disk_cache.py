import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)


def _store(tmp_path):
    return ModelCatalogDiskStore(tmp_path / "model_catalog_cache.json")


def test_round_trip_into_memory_cache(tmp_path):
    store = _store(tmp_path)
    store.record("OpenAI", "https://api.openai.com/v1", ["gpt-a", "gpt-b"],
                 fetched_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC))
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
    store.record("OpenAI", "fp", ["gpt-a"], fetched_at=now - timedelta(hours=23, minutes=59))
    assert store.is_stale("OpenAI", "fp", stale_after_hours=24, now=now) is False
    store.record("OpenAI", "fp", ["gpt-a"], fetched_at=now - timedelta(hours=24, minutes=1))
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
    assert "api_key" not in raw and "Authorization" not in raw and "x-api-key" not in raw


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
    store = ModelCatalogDiskStore(tmp_path / "nested" / "deeper" / "model_catalog_cache.json")
    store.record("OpenAI", "fp", ["gpt-a"])
    store.save()
    assert (tmp_path / "nested" / "deeper" / "model_catalog_cache.json").exists()
