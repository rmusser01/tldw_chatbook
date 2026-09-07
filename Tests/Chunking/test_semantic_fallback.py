import types
import sys

import pytest

from tldw_chatbook.Chunking.engine.strategies.semantic import SemanticChunkingStrategy


def test_token_unit_count_fallback_when_transformers_offline(monkeypatch):
    """Simulate transformers present but tokenizer unavailable; expect fallback.

    We inject a dummy 'transformers' module where AutoTokenizer.from_pretrained
    raises, to mimic an offline or uncached environment.
    """

    class _DummyAT:
        @staticmethod
        def from_pretrained(model_name: str):
            raise RuntimeError("offline or uncached")

    dummy_module = types.ModuleType("transformers")
    dummy_module.AutoTokenizer = _DummyAT  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", dummy_module)

    strat = SemanticChunkingStrategy()
    text = "one two three four five"
    words = len(text.split())

    count = strat._count_units(text, unit="tokens")

    # Fallback should not raise and should return a reasonable approximation
    assert isinstance(count, int)
    assert count >= words
    assert count < words * 3
