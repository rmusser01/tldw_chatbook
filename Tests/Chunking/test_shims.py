# Tests/Chunking/test_shims.py
"""Shim contract tests (spec §5.3): the three phase-1 shims exist and behave."""
import pytest


def test_testing_shim():
    from tldw_chatbook.Chunking._shims import testing
    assert testing.is_truthy("true") is True
    assert testing.is_truthy("no") is False
    assert testing.is_truthy(True) is True
    assert testing.is_truthy(0) is False
    assert isinstance(testing.is_test_mode(), bool)


def test_config_shim():
    from tldw_chatbook.Chunking._shims import config
    cfg = config.load_comprehensive_config()
    # Server code calls .has_section('Chunking') / .get(section, key) — a
    # config-parser-like object must come back.
    assert hasattr(cfg, "has_section")
    assert hasattr(cfg, "get")


def test_prompt_loader_shim_maps_rolling_summarize():
    from tldw_chatbook.Chunking._shims.Utils import prompt_loader
    prompt = prompt_loader.load_prompt("chunking", "Rolling Summarization")
    assert isinstance(prompt, str)
    # Chatbook's canonical prompt for this pairing is 37 chars by default
    # ("Rewrite this text in summarized form."); the brief's >50 threshold
    # measured upstream's 68-char YAML text. Non-empty is the real contract.
    assert len(prompt) > 0  # a real prompt, not an empty string
    # The mapping must hit the documented resolver key.
    from tldw_chatbook.Internal_Prompts.resolver import get_internal_prompt
    assert prompt == get_internal_prompt("summarization.rolling_summarize_system")


def test_prompt_loader_unknown_pairing_raises():
    # Unknown pairings must fail loudly, never silently return "".
    from tldw_chatbook.Chunking._shims.Utils import prompt_loader
    with pytest.raises(KeyError):
        prompt_loader.load_prompt("chunking", "Not A Real Prompt")


def test_prompt_loader_flat_alias():
    # The plan-documented flat path re-exports the Utils implementation that
    # the vendored engine actually imports (rolling_summarize.py:13).
    from tldw_chatbook.Chunking._shims import prompt_loader as flat
    from tldw_chatbook.Chunking._shims.Utils import prompt_loader as nested
    assert flat.load_prompt is nested.load_prompt


def test_engine_imports_with_shims():
    # The engine's module graph must resolve entirely through the shims.
    import tldw_chatbook.Chunking.engine  # noqa: F401
    from tldw_chatbook.Chunking.engine import Chunker, ChunkerConfig
    c = Chunker(ChunkerConfig())
    assert c.config.default_max_size == 400
