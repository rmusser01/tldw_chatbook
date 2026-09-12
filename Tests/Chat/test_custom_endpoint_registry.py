"""Pure custom endpoint registry tests."""
import logging

from tldw_chatbook.Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX,  # noqa: F401  (import-surface check)
    CustomEndpointEntry,
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
