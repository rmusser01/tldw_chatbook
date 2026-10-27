"""TASK-34100.2 AC#17: ship only OpenAI and Anthropic ids that answer today.

``Helper_Scripts/smoke_shipped_model_ids.py`` found twelve shipped catalog ids
retired upstream on 2026-10-03 and again on 2026-10-04 (each a 404 on a real
1-output-token call): OpenAI's ``chatgpt-4o-latest`` (retired 2026-02-17) and
eleven Anthropic ids, from ``claude-opus-4-20250514`` back to ``claude-2.0``.
A user who picked one from the switcher got 'model not found' on the first
send. The replacements are ids the providers' own model lists returned on
2026-10-04 (``chat-latest`` is OpenAI's rolling alias for the ChatGPT chat
model).

These tests pin the offline half; the smoke script is the live half. A
value the user set in their own config is never rewritten: only the shipped
template and the code catalog change.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.config import API_MODELS_BY_PROVIDER, DEFAULT_CONFIG_FROM_TOML

#: Measured 2026-10-04 with real keys: HTTP 404 on a 1-output-token call.
RETIRED_IDS = {
    "OpenAI": frozenset({"chatgpt-4o-latest"}),
    "Anthropic": frozenset(
        {
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-2.1",
            "claude-2.0",
        }
    ),
}


@pytest.mark.parametrize("provider", sorted(RETIRED_IDS))
def test_the_shipped_catalog_lists_no_retired_id(provider: str) -> None:
    template_ids = list(DEFAULT_CONFIG_FROM_TOML["providers"][provider])
    code_ids = list(API_MODELS_BY_PROVIDER[provider])

    assert code_ids == template_ids
    assert sorted(RETIRED_IDS[provider].intersection(template_ids)) == []


@pytest.mark.parametrize(
    ("section", "provider"), [("openai", "OpenAI"), ("anthropic", "Anthropic")]
)
def test_the_shipped_default_model_is_in_the_catalog_and_not_retired(
    section: str, provider: str
) -> None:
    default = DEFAULT_CONFIG_FROM_TOML["api_settings"][section]["model"]

    assert default not in RETIRED_IDS[provider]
    assert default in DEFAULT_CONFIG_FROM_TOML["providers"][provider]
