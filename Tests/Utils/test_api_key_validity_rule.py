"""The shared accessor must not hand out a key the shared rule rejects.

TASK-32806.1. `config.get_api_key()` screened exactly one literal
(`<API_KEY_HERE>`) on one of its four branches and returned every value
raw, while `resolve_provider_api_key` -- the rule the rest of the codebase
uses -- rejects a whole placeholder set and strips surrounding whitespace.
Measured before the fix:

    'YOUR_KEY'         get_api_key -> 'YOUR_KEY'         shared rule -> None
    '  sk-real-key  '  get_api_key -> '  sk-real-key  '  shared rule -> 'sk-real-key'
    '   '              get_api_key -> '   '              shared rule -> None

Five live spend and readiness paths read that accessor. The realtime
pre-connect gate compounded it by testing truthiness, so a placeholder
passed the gate and was then put on the wire as the session credential --
the outcome the gate's own comment exists to prevent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import tldw_chatbook.config as config
from tldw_chatbook.config import (
    PROVIDER_API_KEY_PLACEHOLDERS,
    get_api_key,
    resolve_provider_api_key,
)


def _with_stored_key(key):
    """Patch only the settings read, so the real accessor logic runs."""
    return patch.object(
        config, "load_settings", lambda: {"api_settings": {"openai": {"api_key": key}}}
    )


@pytest.mark.parametrize("placeholder", sorted(PROVIDER_API_KEY_PLACEHOLDERS))
def test_no_placeholder_is_ever_returned(placeholder):
    with _with_stored_key(placeholder):
        assert get_api_key("openai") is None


@pytest.mark.parametrize(
    "padded", ["  sk-real-key  ", "\tsk-real-key", "sk-real-key\n", " sk-real-key "]
)
def test_a_padded_key_is_stripped_rather_than_sent_with_its_spaces(padded):
    with _with_stored_key(padded):
        assert get_api_key("openai") == "sk-real-key"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_key_is_not_a_key(blank):
    with _with_stored_key(blank):
        assert get_api_key("openai") is None


def test_an_ordinary_key_still_resolves():
    with _with_stored_key("sk-genuine-key"):
        assert get_api_key("openai") == "sk-genuine-key"


@pytest.mark.parametrize(
    "value", [*sorted(PROVIDER_API_KEY_PLACEHOLDERS), "  ", "  sk-padded  ", "sk-ok"]
)
def test_the_accessor_agrees_with_the_shared_rule(value):
    """The invariant, stated once: these two must not disagree again."""
    with _with_stored_key(value):
        assert get_api_key("openai") == resolve_provider_api_key(value)


@pytest.mark.parametrize("env_value", ["YOUR_KEY", "   ", "<API_KEY_HERE>"])
def test_the_named_env_var_is_screened_too(env_value, monkeypatch):
    """The env fallback was returned raw as well."""
    monkeypatch.setenv("TLDW_TEST_OPENAI_REALTIME_KEY", env_value)
    settings = {
        "api_settings": {
            "openai": {
                "api_key": "<API_KEY_HERE>",
                "api_key_env_var": "TLDW_TEST_OPENAI_REALTIME_KEY",
            }
        }
    }
    with patch.object(config, "load_settings", lambda: settings):
        assert get_api_key("openai") is None
