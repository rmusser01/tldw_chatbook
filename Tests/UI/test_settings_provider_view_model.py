from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.console_provider_support import ConsoleProviderCatalogEntry
from tldw_chatbook.config import normalize_provider_config_key


def _entry(
    provider_id: str,
    display_name: str,
    *,
    requires_api_key: bool,
) -> ConsoleProviderCatalogEntry:
    return ConsoleProviderCatalogEntry(
        readiness_key=provider_id,
        execution_key=provider_id,
        display_name=display_name,
        requires_api_key=requires_api_key,
    )


def _catalog() -> tuple[ConsoleProviderCatalogEntry, ...]:
    return (
        _entry("local_ollama", "Ollama (legacy alias)", requires_api_key=False),
        _entry("custom", "Custom OpenAI-compatible", requires_api_key=False),
        _entry("openai", "OpenAI", requires_api_key=True),
        _entry("ollama", "Ollama", requires_api_key=False),
        _entry("anthropic", "Anthropic", requires_api_key=True),
    )


def _all_options(groups):
    return tuple(option for group in groups for option in group.options)


def test_overview_primary_rows_follow_user_task_order_without_handoff_jargon():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_settings_overview,
    )

    presentation = build_settings_overview(
        {
            "configuration": "OpenAI / gpt-4.1",
            "last_connection_test": "Passed",
            "storage_privacy": "Local config; secrets redacted",
            "sync": "Ready",
            "runtime_ownership": "Local",
            "server_binding": "Not connected",
            "handoff": "No pending update",
        }
    )

    assert [row.key for row in presentation.primary_rows] == [
        "configuration",
        "last_connection_test",
        "storage_privacy",
        "sync",
    ]
    primary_labels = " ".join(row.label.lower() for row in presentation.primary_rows)
    assert "handoff" not in primary_labels
    assert {row.key for row in presentation.advanced_rows} == {
        "runtime_ownership",
        "server_binding",
        "handoff",
    }
    assert "handoff" not in " ".join(
        row.label.lower() for row in presentation.advanced_rows
    )


def test_provider_picker_records_are_immutable_and_slotted():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        ProviderPickerGroup,
        ProviderPickerOption,
    )

    option = ProviderPickerOption("openai", "OpenAI", "openai")
    group = ProviderPickerGroup("cloud", "Cloud", (option,))

    with pytest.raises(FrozenInstanceError):
        option.label = "Changed"
    with pytest.raises(FrozenInstanceError):
        group.label = "Changed"
    assert not hasattr(option, "__dict__")
    assert not hasattr(group, "__dict__")


def test_provider_picker_preserves_saved_unknown_and_manual_entry():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    groups = build_provider_picker_groups(_catalog(), "my_proxy", "proxy")
    options = _all_options(groups)

    unknown = next(option for option in options if option.saved_unknown)
    assert unknown.provider_id == "my_proxy"
    assert "my_proxy" in unknown.label
    assert any(option.action == "enter_provider_id" for option in options)


def test_provider_picker_searches_display_name_and_provider_id():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    by_name = _all_options(build_provider_picker_groups(_catalog(), "", "anthro"))
    by_id = _all_options(build_provider_picker_groups(_catalog(), "", "local_ollama"))

    assert [option.provider_id for option in by_name if option.provider_id] == [
        "anthropic"
    ]
    assert [option.provider_id for option in by_id if option.provider_id] == [
        "local_ollama"
    ]


def test_provider_picker_grouping_is_stable_and_empty_search_lists_catalog():
    """TASK-33007.2 AC#2, rewritten on purpose: configured providers lead.

    The old order (Cloud, Local, Custom & legacy aliases) mixed the few
    providers a user set up in among the whole catalog. A configured custom
    slot leads too; a configured legacy alias stays last (ADR-066).
    """
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    groups = build_provider_picker_groups(
        _catalog(),
        "openai",
        "",
        configured={"Anthropic", "custom", "LOCAL-OLLAMA"},
    )

    assert [group.group_id for group in groups] == [
        "configured",
        "cloud",
        "local",
        "custom",
        "actions",
    ]
    assert groups[0].label == "Configured"
    assert [option.provider_id for option in groups[0].options] == [
        "anthropic",
        "custom",
    ]
    assert [option.provider_id for option in groups[1].options] == ["openai"]
    assert [option.provider_id for option in groups[2].options] == ["ollama"]
    assert [option.provider_id for option in groups[3].options] == ["local_ollama"]
    assert groups[3].options[0].label == "Ollama (legacy alias)"


def test_provider_picker_without_configured_providers_keeps_catalog_groups():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    groups = build_provider_picker_groups(_catalog(), "openai", "")

    assert [group.group_id for group in groups] == [
        "cloud",
        "local",
        "custom",
        "actions",
    ]
    assert [option.provider_id for option in groups[0].options] == [
        "anthropic",
        "openai",
    ]


def test_provider_picker_summary_names_configured_providers_and_the_rest():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
        provider_picker_summary,
    )

    some = build_provider_picker_groups(
        _catalog(), "openai", "", configured={"anthropic", "ollama"}
    )
    many = build_provider_picker_groups(
        _catalog(),
        "openai",
        "",
        configured={"anthropic", "ollama", "openai", "custom"},
    )
    none = build_provider_picker_groups(_catalog(), "openai", "")

    assert provider_picker_summary(some) == (
        "configured: Anthropic, Ollama · 3 more"
    )
    assert provider_picker_summary(many) == (
        "configured: Anthropic, Custom OpenAI-compatible +2 · 1 more"
    )
    assert provider_picker_summary(none) == "none configured yet · 5 providers"


def test_configured_provider_keys_reads_credentials_and_own_endpoints():
    """A key in config or the shell, or an endpoint the user changed, counts;
    the shipped template's default localhost endpoints do not."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        configured_provider_keys,
    )

    template = DEFAULT_CONFIG_FROM_TOML["api_settings"]
    app_config = {
        "api_settings": {
            "openai": {"api_key": "sk-proj-abcdefghijklmnop1234"},
            "ollama": dict(template["ollama"]),
            "llama_cpp": {**template["llama_cpp"], "api_url": "http://10.0.0.9:9099"},
            # Final review finding 2: readiness drops these keys' source until
            # a base URL is set; the key still counts.
            "azure": {"api_key": "sk-proj-abcdefghijklmnop1234"},
            "databricks": {},
        }
    }
    catalog = (
        _entry("openai", "OpenAI", requires_api_key=True),
        _entry("anthropic", "Anthropic", requires_api_key=True),
        _entry("groq", "Groq", requires_api_key=True),
        _entry("ollama", "Ollama", requires_api_key=False),
        _entry("llama_cpp", "llama.cpp", requires_api_key=False),
        _entry("azure", "Azure OpenAI", requires_api_key=True),
        _entry("databricks", "Databricks", requires_api_key=True),
        _entry("cloudflare", "Cloudflare", requires_api_key=True),
    )

    configured = configured_provider_keys(
        catalog,
        app_config,
        ("api_base_url", "api_url"),
        environ={
            "ANTHROPIC_API_KEY": "sk-ant-abcdefghijklmnop1234",
            "DATABRICKS_TOKEN": "dapi-abcdefghijklmnop1234",
        },
    )

    assert configured == frozenset(
        {"openai", "anthropic", "llama_cpp", "azure", "databricks"}
    )


def test_provider_picker_no_match_keeps_only_honest_manual_action():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    groups = build_provider_picker_groups(_catalog(), "openai", "does-not-exist")

    assert [group.group_id for group in groups] == ["actions"]
    assert groups[0].options[0].action == "enter_provider_id"


@pytest.mark.parametrize(
    ("catalog_id", "saved_provider"),
    [
        ("alpha__beta", "  ALPHA  BETA  "),
        ("unicode_ß", "UNICODE-ẞ"),
        ("local_ollama", "LOCAL-OLLAMA"),
    ],
)
def test_provider_picker_known_identity_matches_runtime_normalization(
    catalog_id,
    saved_provider,
):
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    assert normalize_provider_config_key(saved_provider) == catalog_id
    groups = build_provider_picker_groups(
        (_entry(catalog_id, "Matched provider", requires_api_key=False),),
        saved_provider,
        "",
    )

    assert not any(option.saved_unknown for option in _all_options(groups))


def test_provider_picker_unknown_identity_preserves_exact_saved_text():
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    saved_provider = "  Exact_Custom-ID  "
    groups = build_provider_picker_groups(_catalog(), saved_provider, "")
    saved = next(option for option in _all_options(groups) if option.saved_unknown)

    assert saved.provider_id == saved_provider
    assert saved_provider in saved.label
