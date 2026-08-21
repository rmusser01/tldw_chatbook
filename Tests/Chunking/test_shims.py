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
    # No chunking table ships by default, so every engine has_section guard
    # is False and engine defaults apply (review M-1).
    assert cfg.has_section("Chunking") is False
    # configparser get() semantics: fallback is returned for missing keys,
    # including when the section itself is absent (review M-1).
    assert cfg.get("Chunking", "missing", fallback="d") == "d"


def test_config_shim_percent_values_do_not_crash(monkeypatch):
    # Regression (review I-1): with ConfigParser's default BasicInterpolation
    # any '%' in a [Chunking]/[chunking] TOML value raised
    # ValueError("invalid interpolation syntax in '50%'") at construction
    # time — and the engine catches only ImportError there, so EVERY Chunker
    # construction died, even for keys the engine never reads. Interpolation
    # is now disabled; raw values must round-trip verbatim, including
    # '%(name)s' (which previously raised InterpolationMissingOptionError at
    # .get() time, in no engine exception tuple) and non-string TOML keys.
    from tldw_chatbook.Chunking._shims import config as shim

    monkeypatch.setattr(
        shim,
        "load_cli_config_and_ensure_existence",
        lambda: {"Chunking": {
            "max_size": "50%",
            "note": "100% sure",
            "tmpl": "pre-%(name)s-post",
            "count": 3,
            7: "int-key",
        }},
    )
    cfg = shim.load_comprehensive_config()
    assert cfg.get("Chunking", "max_size") == "50%"
    assert cfg.get("Chunking", "note") == "100% sure"
    assert cfg.get("Chunking", "tmpl") == "pre-%(name)s-post"
    assert cfg.get("Chunking", "count") == "3"
    assert cfg.get("Chunking", "7") == "int-key"


def test_config_shim_merge_order_capitalized_wins(monkeypatch):
    # Review M-1/M-4: both [chunking] and [Chunking] tables are accepted and
    # merged; on key conflict the capitalized table wins (applied last in
    # _chunking_section()).
    from tldw_chatbook.Chunking._shims import config as shim

    monkeypatch.setattr(
        shim,
        "load_cli_config_and_ensure_existence",
        lambda: {
            "chunking": {"max_size": "100", "lower_only": "yes"},
            "Chunking": {"max_size": "200"},
        },
    )
    cfg = shim.load_comprehensive_config()
    assert cfg.get("Chunking", "max_size") == "200"
    assert cfg.get("Chunking", "lower_only") == "yes"


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
