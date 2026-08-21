import json

import pytest


def test_json_single_metadata_reference_list_mode():

    from tldw_chatbook.Chunking.engine.strategies.json_xml import (
        JSONChunkingStrategy,
    )

    strategy = JSONChunkingStrategy()

    payload = {
        "title": "Report",
        "source": "unit-test",
        "data": [{"id": 1}, {"id": 2}, {"id": 3}],
    }

    chunks = strategy.chunk(
        json.dumps(payload),
        max_size=2,
        overlap=0,
        preserve_metadata=True,
        single_metadata_reference=True,
        metadata_reference_key="__meta_ref__",
        output_format="json",
    )

    # Expect 1 metadata-only chunk + 2 data chunks (2 items, then 1 item)
    assert isinstance(chunks, list)
    assert len(chunks) == 3

    first = json.loads(chunks[0])
    assert set(first.keys()) == {"__meta_ref__", "metadata"}
    assert first["metadata"] == {"title": "Report", "source": "unit-test"}
    ref_id = first["__meta_ref__"]
    assert isinstance(ref_id, str) and len(ref_id) > 0

    # Subsequent chunks should reference metadata and not duplicate it
    c1 = json.loads(chunks[1])
    c2 = json.loads(chunks[2])

    for c in (c1, c2):
        assert set(c.keys()) == {"data", "__meta_ref__"}
        assert c["__meta_ref__"] == ref_id
        # Ensure metadata keys are not duplicated in data chunks
        assert "title" not in c and "source" not in c


def test_json_single_metadata_reference_dict_mode():

    from tldw_chatbook.Chunking.engine.strategies.json_xml import (
        JSONChunkingStrategy,
    )

    strategy = JSONChunkingStrategy()

    payload = {
        "dataset": "X",
        "meta": 42,
        "data": {"a": 1, "b": 2, "c": 3},
    }

    chunks = strategy.chunk(
        json.dumps(payload),
        max_size=2,
        overlap=0,
        preserve_metadata=True,
        single_metadata_reference=True,
        metadata_reference_key="__meta_ref__",
        output_format="json",
    )

    # Expect 1 metadata-only chunk + 2 dict chunks (2 keys, then 1 key)
    assert isinstance(chunks, list)
    assert len(chunks) == 3

    first = json.loads(chunks[0])
    assert set(first.keys()) == {"__meta_ref__", "metadata"}
    assert first["metadata"] == {"dataset": "X", "meta": 42}
    ref_id = first["__meta_ref__"]

    c1 = json.loads(chunks[1])
    c2 = json.loads(chunks[2])
    for c in (c1, c2):
        assert set(c.keys()) == {"data", "__meta_ref__"}
        assert c["__meta_ref__"] == ref_id
        assert "dataset" not in c and "meta" not in c


def test_json_single_metadata_reference_via_config(monkeypatch):

    import configparser
    from tldw_chatbook.Chunking.engine.strategies.json_xml import (
        JSONChunkingStrategy,
    )

    # Ensure env does not override config
    monkeypatch.delenv("JSON_SINGLE_METADATA_REFERENCE", raising=False)
    monkeypatch.delenv("JSON_METADATA_REFERENCE_KEY", raising=False)

    # Provide a temporary ConfigParser with [Chunking] settings
    cp = configparser.ConfigParser()
    cp.add_section("Chunking")
    cp.set("Chunking", "json_single_metadata_reference", "true")
    cp.set("Chunking", "json_metadata_reference_key", "ref_id")

    # Patch the loader used by the strategy helpers
    monkeypatch.setattr("tldw_chatbook.Chunking._shims.config.load_comprehensive_config", lambda: cp)

    strategy = JSONChunkingStrategy()
    payload = {
        "metaA": "M",
        "metaB": 99,
        "data": [{"x": 1}, {"x": 2}, {"x": 3}],
    }

    chunks = strategy.chunk(
        json.dumps(payload),
        max_size=2,
        overlap=0,
        preserve_metadata=True,
        # Intentionally omit single_metadata_reference and metadata_reference_key
        output_format="json",
    )

    assert len(chunks) == 3
    first = json.loads(chunks[0])
    assert set(first.keys()) == {"ref_id", "metadata"}
    assert first["metadata"] == {"metaA": "M", "metaB": 99}
    rid = first["ref_id"]

    c1 = json.loads(chunks[1])
    c2 = json.loads(chunks[2])
    for c in (c1, c2):
        assert set(c.keys()) == {"data", "ref_id"}
        assert c["ref_id"] == rid
        assert "metaA" not in c and "metaB" not in c
