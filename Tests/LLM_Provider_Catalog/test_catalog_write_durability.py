"""Provider-catalog disk writes must reach the platter before the rename.

TASK-32901 (tier-2 S15 P2): both catalog writers did ``os.replace`` with no
flush and no ``fsync`` -- ``grep -rn fsync tldw_chatbook/LLM_Provider_Catalog/``
returned nothing, while eight sibling writers in the same slices fsync. A
power loss between the write and writeback publishes a zero-length or partial
file under the real name, and ``ModelDiscoveryDiskCache.load`` treats a corrupt
entry as merely "rejected" -- so the user silently loses their discovered-model
catalog instead of seeing a failure. ``Utils/atomic_file_ops`` already owns
this discipline.
"""

from __future__ import annotations

import os

import pytest

from tldw_chatbook.LLM_Provider_Catalog import models_dev_catalog
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)


@pytest.fixture()
def fsync_spy(monkeypatch):
    calls: list[int] = []
    real = os.fsync

    def _spy(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    return calls


def test_models_dev_cache_write_is_durable(tmp_path, fsync_spy):
    target = tmp_path / "models-dev.json"

    models_dev_catalog._write_cache_file(target, {"providers": {}}, "etag-1")

    assert target.exists()
    assert fsync_spy, "cache file published without an fsync"


def test_discovery_disk_cache_save_is_durable(tmp_path, fsync_spy):
    cache = ModelCatalogDiskStore(tmp_path / "discovery.json")
    cache.record("openai", "endpoint-fingerprint", ("gpt-4o",))

    cache.save()

    assert (tmp_path / "discovery.json").exists()
    assert fsync_spy, "discovery cache published without an fsync"
