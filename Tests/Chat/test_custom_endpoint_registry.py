"""Pure custom endpoint registry tests."""
import logging

import pytest

from tldw_chatbook.Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,  # noqa: F401  (import-surface check)
    CustomEndpointEntry,
    CustomEndpointSlugError,
    build_entry_mutation,
    derive_slug,
    entry_for,
    family_execution_key,  # noqa: F401  (import-surface check)
    load_custom_endpoints,
    split_custom_endpoint_id,
    validate_entry,
)


def _config_with(slug: str, **overrides) -> dict:
    entry = {"display_name": "GPU box", "family": "llama_cpp",
             "base_url": "http://192.168.1.5:8080"}
    entry.update(overrides)
    return {"custom_endpoints": {slug: entry}}

def test_split_custom_endpoint_id():
    assert split_custom_endpoint_id("custom-ep:llama-gpu") == "llama-gpu"
    assert split_custom_endpoint_id("llama_cpp") is None
    assert split_custom_endpoint_id(None) is None
    assert split_custom_endpoint_id("custom-ep:") is None

def test_load_returns_valid_entries_and_drops_invalid_with_warning(caplog):
    cfg = _config_with("ok")
    cfg["custom_endpoints"]["bad-family"] = {
        "display_name": "X", "family": "groq", "base_url": "http://h:1"}
    with caplog.at_level(logging.WARNING):
        entries = load_custom_endpoints(cfg)
    assert set(entries) == {"ok"}
    assert entries["ok"].family == "llama_cpp"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bad-family" in warnings[0].getMessage()

def test_llama_family_normalizes_v1_suffix():
    entries = load_custom_endpoints(_config_with(
        "s", base_url="http://127.0.0.1:8080/v1"))
    assert entries["s"].base_url == "http://127.0.0.1:8080"

def test_derive_slug_collapses_and_uniquifies():
    assert derive_slug("GPU box llama.cpp!", {"gpu-box-llama-cpp"}) == "gpu-box-llama-cpp-2"

def test_derive_slug_extends_suffix_search_into_four_digits():
    # -2 .. -99 taken: the search must reach at least 4-digit suffixes
    # instead of falling back to the colliding bare base.
    taken = {"gpu-box"} | {f"gpu-box-{n}" for n in range(2, 100)}
    assert derive_slug("GPU box", taken) == "gpu-box-100"

def test_derive_slug_exhaustion_raises_instead_of_returning_colliding_slug():
    # Every candidate (bare base plus -2 .. -9999) is taken: returning the
    # bare base would make the creation path overwrite an existing entry's
    # config section, so exhaustion must raise user-facing copy instead.
    taken = {"gpu-box"} | {f"gpu-box-{n}" for n in range(2, 10000)}
    with pytest.raises(CustomEndpointSlugError, match="already in use"):
        derive_slug("GPU box", taken)

def test_derive_slug_truncates_long_base_to_reserve_suffix_space():
    # Clamping the assembled candidate shears the suffix off a long base
    # (a 63/64-char base re-derives the already-taken base for every
    # suffix), so the stem must be truncated first to keep each suffixed
    # candidate distinct and creatable.
    base63 = "g" * 63
    assert derive_slug(base63, {base63}) == "g" * 62 + "-2"
    base64 = "g" * 64
    assert derive_slug(base64, {base64}) == "g" * 62 + "-2"

def test_derive_slug_keeps_multi_digit_suffixes_whole_for_long_bases():
    # A 62-char base plus "-10" overflows the 64-char clamp: the stem is
    # truncated instead of shearing the suffix down to "-1".
    base = "g" * 62
    taken = {base} | {f"{base}-{n}" for n in range(2, 10)}
    assert derive_slug(base, taken) == "g" * 61 + "-10"

def test_slug_collision_error_is_value_error_with_user_facing_copy():
    # The F9 convert worker surfaces ValueError messages in its status line;
    # the slug error rides that contract with copy safe to show as-is.
    error = CustomEndpointSlugError()
    assert isinstance(error, ValueError)
    assert str(error) == "That name is already in use; choose another."

def test_entry_for_resolves_provider_id():
    assert entry_for(_config_with("s"), "custom-ep:s").display_name == "GPU box"
    assert entry_for({}, "custom-ep:missing") is None

def test_validate_entry_rejects_bad_family_and_url():
    assert validate_entry("N", "groq", "http://h:1") == ["Unknown endpoint family: groq."]
    assert any("http(s)" in e for e in validate_entry("N", "llama_cpp", "ftp://h"))

def test_mutation_round_trips_and_omits_none():
    entry = CustomEndpointEntry(slug="s", display_name="D", family="ollama",
                                base_url="http://127.0.0.1:11434")
    mutation = build_entry_mutation(entry)
    assert mutation == {"custom_endpoints.s": {
        "display_name": "D", "family": "ollama",
        "base_url": "http://127.0.0.1:11434", "models": []}}

def test_entry_repr_hides_api_key():
    entry = CustomEndpointEntry(slug="s", display_name="D", family="ollama",
                                base_url="http://127.0.0.1:11434",
                                api_key="sekrit")
    # "api_key=" (not the substring "api_key"): api_key_env legitimately
    # renders and contains "api_key" as a prefix.
    assert "api_key=" not in repr(entry)
    assert "sekrit" not in repr(entry)
    assert "slug='s'" in repr(entry)
    assert "display_name='D'" in repr(entry)
    assert "family='ollama'" in repr(entry)
    assert "base_url='http://127.0.0.1:11434'" in repr(entry)
    assert "api_key_env=None" in repr(entry)

def test_entry_loads_params_table():
    config = {"custom_endpoints": {"qwen-local": {
        "display_name": "Qwen Local", "family": "llama_cpp",
        "base_url": "http://127.0.0.1:8080",
        "params": {"temperature": 0.2, "top_k": 40},
    }}}
    entries = load_custom_endpoints(config)
    assert entries["qwen-local"].params == (("temperature", 0.2), ("top_k", 40))

def test_entry_without_params_unchanged():
    config = {"custom_endpoints": {"plain": {
        "display_name": "Plain", "family": "ollama",
        "base_url": "http://127.0.0.1:11434",
    }}}
    assert load_custom_endpoints(config)["plain"].params == ()

def test_invalid_param_key_drops_entry_with_warning(caplog):
    config = {"custom_endpoints": {"bad": {
        "display_name": "Bad", "family": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "params": {"temprature": 0.2},
    }}}
    with caplog.at_level(logging.WARNING):
        entries = load_custom_endpoints(config)
    assert "bad" not in entries
    assert any("bad" in record.getMessage() for record in caplog.records)

def test_validate_entry_reports_param_errors():
    errors = validate_entry(
        "Name", "ollama", "http://127.0.0.1:11434",
        params={"seed": "not-an-int"},
    )
    assert any("seed" in error for error in errors)

def test_entry_mutation_round_trips_params():
    entry = CustomEndpointEntry(
        slug="qwen-local", display_name="Qwen Local", family="llama_cpp",
        base_url="http://127.0.0.1:8080",
        params=(("temperature", 0.2),),
    )
    mutation = build_entry_mutation(entry)
    assert mutation["custom_endpoints.qwen-local"]["params"] == {
        "temperature": 0.2,
    }
    reloaded = load_custom_endpoints(
        {"custom_endpoints": {
            "qwen-local": mutation["custom_endpoints.qwen-local"],
        }}
    )
    assert reloaded["qwen-local"].params == (("temperature", 0.2),)
