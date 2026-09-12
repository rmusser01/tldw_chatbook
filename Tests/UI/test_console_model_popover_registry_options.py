"""Quick-popover provider-option parity with the registry (ADR-146)."""

from __future__ import annotations

from tldw_chatbook.Widgets.Console.console_model_popover import (
    ConsoleModelPopover,
)


def test_model_popover_provider_options_include_registry_entries() -> None:
    """ADR-146: the quick popover lists registry entries with their labels.

    The popover builds its options through the same registry-aware builder
    as the full modal (app_config passed, registry labels preserved), so a
    ``custom-ep:<slug>`` entry is selectable with its display_name instead
    of being dropped or relabeled through the shared catalog.
    """
    from tldw_chatbook.Widgets.Console.console_model_popover import (
        ConsoleModelPopover,
    )

    app_config = {
        "custom_endpoints": {
            "vale": {
                "display_name": "Vale endpoint",
                "base_url": "http://127.0.0.1:9999/v1",
                "family": "openai_compatible",
            }
        }
    }
    popover = ConsoleModelPopover.__new__(ConsoleModelPopover)
    popover._providers_models = {"openai": ["gpt-test"]}
    popover._app_config = app_config

    options = popover._provider_select_options()

    labels_by_value = {value: label for label, value in options}
    assert labels_by_value["custom-ep:vale"] == "Vale endpoint"
    assert labels_by_value["openai"] == "OpenAI"
