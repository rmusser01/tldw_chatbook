"""TASK-33002.5: one provider display-name catalog serves every surface."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat import console_provider_support
from tldw_chatbook.Chat.console_provider_support import (
    supported_console_provider_catalog,
)
from tldw_chatbook.Chat.provider_catalog import (
    PROVIDER_DISPLAY_NAMES,
    provider_display_name,
)
from tldw_chatbook.config import (
    DEFAULT_CONFIG_FROM_TOML,
    normalize_provider_config_key,
)

_SHIPPED_PROVIDER_KEYS = tuple(DEFAULT_CONFIG_FROM_TOML["providers"])
_GPU_BOX_CONFIG = {
    "custom_endpoints": {
        "gpu-box": {
            "display_name": "GPU Box",
            "family": "openai_compatible",
            "base_url": "http://127.0.0.1:9000/v1",
        }
    }
}


def test_shipped_table_spells_the_keys_this_test_exists_for() -> None:
    # A vacuity guard: the walk below is only meaningful while the shipped
    # table still carries the spellings exact lookup used to miss.
    assert {"Llama_cpp", "local-llm", "local_onnx", "local_transformers"} <= set(
        _SHIPPED_PROVIDER_KEYS
    )


@pytest.mark.parametrize("key", _SHIPPED_PROVIDER_KEYS)
def test_every_shipped_provider_key_renders_a_human_name(key: str) -> None:
    # "output != input" cannot catch a raw fallback: OpenAI's name IS its key.
    normalized = normalize_provider_config_key(key)
    assert normalized in PROVIDER_DISPLAY_NAMES, f"no display name for {key!r}"
    assert provider_display_name(key) == PROVIDER_DISPLAY_NAMES[normalized]


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("Llama_cpp", "llama.cpp"),
        ("local-llm", "Local LLM (legacy generic)"),
        ("local_llamacpp", "llama.cpp (legacy alias)"),
        ("Google", "Google Gemini"),
        ("OpenAI", "OpenAI"),
    ],
)
def test_saved_spellings_resolve_to_the_catalog_name(key: str, name: str) -> None:
    assert provider_display_name(key) == name


def test_unmapped_ids_come_back_unchanged() -> None:
    # A custom endpoint id must never be mangled into custom_ep:gpu_box.
    assert provider_display_name("custom-ep:gpu-box") == "custom-ep:gpu-box"
    assert provider_display_name("My-Gateway") == "My-Gateway"
    assert provider_display_name("") == ""


def test_custom_endpoint_id_names_its_registry_entry() -> None:
    assert provider_display_name("custom-ep:gpu-box", _GPU_BOX_CONFIG) == "GPU Box"
    # An id whose entry is gone stays identifiable rather than blank.
    assert provider_display_name("custom-ep:gone", _GPU_BOX_CONFIG) == "custom-ep:gone"
    assert provider_display_name("openai", _GPU_BOX_CONFIG) == "OpenAI"


def test_console_catalog_labels_come_from_the_shared_catalog() -> None:
    assert not hasattr(console_provider_support, "_PROVIDER_DISPLAY_NAMES")
    assert not hasattr(console_provider_support, "_provider_display_name")
    catalog = supported_console_provider_catalog()
    assert catalog
    for entry in catalog:
        assert entry.readiness_key in PROVIDER_DISPLAY_NAMES, entry.readiness_key
        assert entry.display_name == PROVIDER_DISPLAY_NAMES[entry.readiness_key]


@pytest.mark.parametrize("key", ["local-llm", "Llama_cpp", "Google", "openai"])
def test_personas_preview_names_providers_from_the_shared_catalog(key: str) -> None:
    # Final review I3: the Personas preview kept a private lookup with a
    # title-case fallback, so a saved local-llm read "Local Llm" there and
    # "Local LLM (legacy generic)" on the Console chip.
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        PersonasPreviewController,
    )

    assert PersonasPreviewController._provider_label(key) == provider_display_name(key)
