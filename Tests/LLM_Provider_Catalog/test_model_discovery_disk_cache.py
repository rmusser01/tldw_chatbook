from datetime import UTC, datetime, timedelta

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
