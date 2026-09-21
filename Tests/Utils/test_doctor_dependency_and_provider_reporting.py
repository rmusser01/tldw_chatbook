"""The doctor must report what is actually installed and configured.

TASK-32811.3.

Under the shipped lazy dependency mode, `DEPENDENCIES_AVAILABLE` starts
all-False and is populated only by `initialize_dependency_checks`, which
nothing on the doctor path called -- so `check_optional_dependencies`
reported every optional group as "not installed", installed ones included.

`get_detected_api_providers` iterated `config.items()` for flat dotted keys
`"api_settings.<provider>"` that a nested TOML load never produces, so it
always returned `[]` and the doctor's providers check always warned "no API
providers are configured".
"""

from __future__ import annotations

from unittest.mock import patch

import tldw_chatbook.config as config
from tldw_chatbook.Utils.doctor import check_optional_dependencies


def test_optional_dependency_check_reflects_installed_groups():
    # An injected map is used verbatim; the real path (available=None) probes
    # first, which the test below covers structurally.
    result = check_optional_dependencies(
        available={"pdf": True, "ebook": True, "websearch": False}
    )
    assert result.status == "warn"
    assert "websearch" in result.detail
    assert "pdf" not in result.detail.split(":", 1)[-1]  # pdf is installed


def test_optional_dependency_check_probes_before_reading_the_registry():
    """The default path must populate the registry, not read it cold."""
    import ast
    import inspect

    source = inspect.getsource(check_optional_dependencies)
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "initialize_dependency_checks" in names, (
        "the default path reads DEPENDENCIES_AVAILABLE without probing it first"
    )


def test_detected_providers_reads_the_nested_api_settings_table():
    nested = {
        "api_settings": {
            "openai": {"api_key": "sk-real-openai-key"},
            "anthropic": {"api_key": "<API_KEY_HERE>"},  # placeholder
            "groq": {"api_key": "  sk-padded-groq  "},   # padded but real
            "cohere": {"api_key": "   "},                # blank
        }
    }
    with patch.object(config, "load_cli_config_and_ensure_existence", lambda: nested):
        detected = config.get_detected_api_providers()
    assert sorted(detected) == ["groq", "openai"]


def test_detected_providers_is_empty_when_nothing_is_configured():
    empty = {"api_settings": {"openai": {"api_key": "<API_KEY_HERE>"}}}
    with patch.object(config, "load_cli_config_and_ensure_existence", lambda: empty):
        assert config.get_detected_api_providers() == []
