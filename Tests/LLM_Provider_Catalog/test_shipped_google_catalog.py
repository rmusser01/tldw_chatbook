"""TASK-34100.2 AC#6 (gap-03): ship only Gemini ids that answer today.

The 2026-10-02 review found every shipped Google id retired: the
``[providers] Google`` catalog, ``[api_settings.google] model`` and the code
fallbacks all named ``gemini-2.5-flash`` and older, and each returned 404 on
generateContent ("no longer available to new users"). A fresh Gemini user's
first chat failed even though setup said it was ready.

These tests pin the offline half: the shipped catalog and every fallback
exclude the ids Google has retired, and the default is Google's rolling
alias so it cannot rot the same way. The live half -- does each shipped id
answer a 1-output-token generate call today -- is
``Helper_Scripts/smoke_shipped_model_ids.py``.
"""

from __future__ import annotations

from tldw_chatbook import config
from tldw_chatbook.config import (
    API_MODELS_BY_PROVIDER,
    DEFAULT_CONFIG_FROM_TOML,
    DEFAULT_GOOGLE_CHAT_MODEL,
)

#: Measured 2026-10-03 with a real key: 404 on generateContent
#: ("no longer available to new users" / "not found for API version").
RETIRED_GEMINI_IDS = frozenset(
    {
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
    }
)


def test_the_default_gemini_model_is_googles_rolling_alias() -> None:
    assert DEFAULT_GOOGLE_CHAT_MODEL == "gemini-flash-latest"
    assert (
        DEFAULT_CONFIG_FROM_TOML["api_settings"]["google"]["model"]
        == DEFAULT_GOOGLE_CHAT_MODEL
    )


def test_the_shipped_google_catalog_lists_no_retired_id() -> None:
    template_ids = list(DEFAULT_CONFIG_FROM_TOML["providers"]["Google"])
    code_ids = list(API_MODELS_BY_PROVIDER["Google"])

    assert template_ids[0] == DEFAULT_GOOGLE_CHAT_MODEL
    assert code_ids == template_ids
    assert RETIRED_GEMINI_IDS.isdisjoint(template_ids)


def test_no_code_fallback_names_a_retired_gemini_id() -> None:
    from pathlib import Path

    from tldw_chatbook.LLM_Calls import LLM_API_Calls, Summarization_General_Lib

    def source(module: object) -> str:
        return Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]

    # chat_with_google's and the summarizer's model fallbacks: any quoted
    # retired id in these modules is a fallback that would be sent.
    named = sorted(
        f"{name}: {model_id}"
        for name, text in (
            ("LLM_API_Calls.py", source(LLM_API_Calls)),
            ("Summarization_General_Lib.py", source(Summarization_General_Lib)),
        )
        for model_id in RETIRED_GEMINI_IDS
        if f'"{model_id}"' in text
    )
    # config.py also carries a capability table that merely DESCRIBES old
    # models; only its legacy [API] google_model fallback is ever sent.
    named += sorted(
        f"config.py: {model_id}"
        for model_id in RETIRED_GEMINI_IDS
        if f'"google_model", "{model_id}"' in source(config)
    )

    assert named == []
