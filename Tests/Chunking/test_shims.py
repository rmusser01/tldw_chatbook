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
    # (task 11, spec §9.1/AC 40) The shipped config template now carries a
    # lowercase [chunking] table (default_template — the INGEST resolution
    # tier), which the shim merges into the engine's view, so the section
    # exists on a fresh profile. Review M-1's intent still holds through a
    # present-but-foreign section: the engine reads only its OWN keys with
    # fallbacks, and none of them ship, so every engine default applies.
    if cfg.has_section("Chunking"):
        assert not any(
            cfg.has_option("Chunking", key)
            for key in (
                "regex_timeout_seconds",
                "cache_copy_on_access",
                "verbose_logging",
                "max_streaming_flush_threshold_chars",
                "regex_simple_only",
                "regex_disable_multiprocessing",
            )
        )
    else:
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


def test_prompt_loader_shim_maps_proposition_profiles():
    """The vendored propositions strategy calls load_prompt for three
    OPTIONAL profile overrides (propositions.py:321/334/347 at the pin).
    Chatbook does not carry the server's Prompts runtime, so a known
    pairing resolves to "" and the strategy's in-code defaults are
    chatbook's effective instructions. Upstream at the pin DOES ship
    chunking.prompts.yaml entries for all three pairs; for
    claimify/gemma_aps the YAML wording differs from the in-code
    defaults — a recorded divergence, not byte-faithful parity. With
    the _KNOWN value "" the resolver is never consulted, so overrides
    cannot ride the Internal_Prompts catalog; a future override
    mechanism changes the map VALUES, not the keys."""
    from tldw_chatbook.Chunking._shims.Utils import prompt_loader
    for name in ("proposition_claimify", "proposition_gemma_aps",
                 "proposition_generic"):
        assert prompt_loader.load_prompt("chunking", name) == ""


def test_prompt_loader_known_covers_vendored_propositions_calls():
    """Source-scan pin: every literal load_prompt(category, name) pair the
    vendored file actually passes must be a known pairing, so the
    raise-loudly contract can never fire from inside the vendored engine
    (a KeyError there escapes chunk() — the per-window try does not cover
    _build_llm_prompt)."""
    import re
    from pathlib import Path
    from tldw_chatbook.Chunking._shims.Utils import prompt_loader
    src = (Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Chunking"
           / "engine" / "strategies" / "propositions.py").read_text()
    pairs = set(re.findall(
        r'load_prompt\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', src))
    assert pairs, "scan found no load_prompt call sites — anchor drifted"
    assert pairs <= set(prompt_loader._KNOWN), \
        f"unmapped load_prompt pairs in the vendored strategy: {pairs - set(prompt_loader._KNOWN)}"


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
