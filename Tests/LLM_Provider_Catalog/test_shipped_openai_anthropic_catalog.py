"""TASK-34100.2 AC#17: ship only OpenAI and Anthropic ids that answer today.

``Helper_Scripts/smoke_shipped_model_ids.py`` found twelve shipped catalog ids
retired upstream on 2026-10-03 and again on 2026-10-04 (each a 404 on a real
1-output-token call): OpenAI's ``chatgpt-4o-latest`` (retired 2026-02-17) and
eleven Anthropic ids, from ``claude-opus-4-20250514`` back to ``claude-2.0``.
A user who picked one from the switcher got 'model not found' on the first
send. The replacements are ids the providers' own model lists returned on
2026-10-04 (``chat-latest`` is OpenAI's rolling alias for the ChatGPT chat
model).

On 2026-11-05 the same smoke found four more OpenAI ids retired (HTTP 404
'has been deprecated'): ``o4-mini-2025-04-16``, ``o3-mini-2025-01-31``,
``o1-2024-12-17`` and ``gpt-4.1-nano-2025-04-14`` (review IV-1). They are
replaced by dated snapshots OpenAI's own model list returned that day, each
answering the smoke's 1-output-token call, and their capability rows go with
them.

These tests pin the offline half; the smoke script is the live half. A
value the user set in their own config is never rewritten: only the shipped
template and the code catalog change.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.config import API_MODELS_BY_PROVIDER, DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.model_capabilities import DEFAULT_MODEL_CAPABILITIES

#: Measured with real keys: HTTP 404 on a 1-output-token call (2026-10-04,
#: and the four dated OpenAI ids on 2026-11-05).
RETIRED_IDS = {
    "OpenAI": frozenset(
        {
            "chatgpt-4o-latest",
            "o4-mini-2025-04-16",
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "gpt-4.1-nano-2025-04-14",
        }
    ),
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


def test_no_openai_capability_row_names_a_retired_id() -> None:
    """IV-1: OpenAI's capability rows mirror its shipped catalog, so a retired
    id's row goes with it (a user's own config may still name it; its send
    then reads 'model not found', which no capability row changes). The
    claude-3 rows predate this task and are not the catalog's mirror."""
    assert sorted(RETIRED_IDS["OpenAI"].intersection(DEFAULT_MODEL_CAPABILITIES)) == []
