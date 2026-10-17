"""Bounded capability evidence without pretending simulation is native support."""

import json
from types import SimpleNamespace

from Tests.Plugins import platform_support


def test_actual_collection_artifact_records_capability_and_keeps_pure_node(
    tmp_path, monkeypatch
):
    import pytest

    class Item:
        def __init__(self, name, marks=(), fixtures=()):
            self.nodeid, self.marks, self.fixturenames = name, marks, fixtures

        def iter_markers(self, name):
            return (mark.mark for mark in self.marks if mark.mark.name == name)

    target = tmp_path / "selection.json"
    monkeypatch.setenv("PLUGIN_COLLECTION_ARTIFACT", str(target))
    items = [
        Item("pure-parser"),
        Item(
            "command", (pytest.mark.skipif(True, reason="explicit capability refusal"),)
        ),
    ]
    platform_support.pytest_collection_finish(SimpleNamespace(items=items))
    result = json.loads(target.read_text())
    assert result["nodes"][0]["capability_skip_reasons"] == []
    assert result["nodes"][1]["capability_skip_reasons"] == [
        "explicit capability refusal"
    ]
    assert result["nodes"][1]["skipif_declarations"] == [
        {"condition": True, "reason": "explicit capability refusal"}
    ]
    assert "Collection" in result["qualification"]


def test_fixture_refusal_is_explicit_and_precedes_runtime_setup(monkeypatch):
    import pytest

    monkeypatch.setattr(platform_support, "MANAGED_RUNTIME", False)
    with pytest.raises(pytest.skip.Exception, match="64-bit Darwin"):
        platform_support.require_managed_runtime()
