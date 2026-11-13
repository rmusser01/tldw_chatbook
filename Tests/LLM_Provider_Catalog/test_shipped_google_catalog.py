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

#: Listed by Google but not answering a generate call (TASK-34100.2 review
#: v5r2, G2C-R3-1). Measured 2026-11-12 with the repo's real key, five
#: 1-output-token calls each over 13 minutes: gemini-flash-latest 0/5 (four
#: 503 'high demand', one 60 s timeout) and gemini-3.8-flash, the model it
#: resolves to, 0/5; earlier smoke runs since 2026-10-06 recorded 0/8 and 1/8.
#: The key was not the cause: gemini-flash-lite-latest and
#: gemini-3.5-flash-lite answered 5/5 in about a second, with no 429.
#: Evidence: .worktrees/setup-wizard-ux-qa/evidence/g2cv5-r3-catalog/.
NOT_ANSWERING_GEMINI_IDS = frozenset({"gemini-flash-latest", "gemini-3.8-flash"})


def test_the_default_gemini_model_is_googles_rolling_alias() -> None:
    # Review v5r2 G2C-R3-1: the Flash-Lite alias, the rolling alias that
    # answered every run (gemini-flash-latest stopped answering).
    assert DEFAULT_GOOGLE_CHAT_MODEL == "gemini-flash-lite-latest"
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
    assert NOT_ANSWERING_GEMINI_IDS.isdisjoint(template_ids)
    assert DEFAULT_GOOGLE_CHAT_MODEL not in NOT_ANSWERING_GEMINI_IDS


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
