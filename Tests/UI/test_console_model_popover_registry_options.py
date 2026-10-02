"""Switch model's provider parity with the registry (ADR-146, TASK-194).

Rewritten on purpose for TASK-33004.4: the provider Select this pinned is
gone. The pair list now takes its providers from the same registry-aware
builder, and names every provider through the shared display-name catalog.
"""

from __future__ import annotations

from Tests.UI.test_console_model_switcher import Recorder, build_switcher


def test_model_popover_provider_options_include_registry_entries() -> None:
    """ADR-146: registry entries are listed and keep their display_name."""
    app_config = {
        "custom_endpoints": {
            "vale": {
                "display_name": "Vale endpoint",
                "base_url": "http://127.0.0.1:9999/v1",
                "family": "openai_compatible",
                "models": ["vale-model"],
            }
        }
    }
    switcher = build_switcher(
        Recorder(), providers_models={"openai": ["gpt-test"]}, app_config=app_config
    )

    assert "custom-ep:vale" in switcher._provider_order
    assert switcher._display("custom-ep:vale") == "Vale endpoint"
    assert switcher._models_for("custom-ep:vale") == ("vale-model",)
    # TASK-194 AC#1: display names from Chat/provider_catalog, never raw keys.
    assert switcher._display("openai") == "OpenAI"
    assert switcher._display("llama_cpp") == "llama.cpp"
    assert {"custom", "custom_2"} <= set(switcher._provider_order)
