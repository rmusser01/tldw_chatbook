# tldw_chatbook/LLM_Calls/pricing_catalog.py
# Description: Per-model pricing (dollars per million tokens) for the Console cost ticker.
#
"""Per-model pricing (dollars per million tokens) for the cost ticker.

Resolution order, per lookup: a ``[pricing].models`` config entry beats the
seeded direct map for the same "provider:model" key -> otherwise the seeded
direct map -> then that provider's pattern list, where a ``[pricing].
patterns`` entry for a provider REPLACES that provider's whole seeded list
(providers absent from config keep their seeded patterns) -> then the
local-provider zero-rate -> then None. None means "no pricing data" and the
UI must show token counts instead of fabricating a dollar figure.

Provider names are normalized through ``provider_config_key`` (the same
mapping the rest of the app uses: lowercase, spaces and dashes to
underscores), so an execution-key spelling ("local-llm") and a readiness-key
spelling ("local_llm") resolve identically.

Rates were verified against each provider's official pricing page on
_SEED_AS_OF (see task-2-report.md for the full source list); a few entries
are marked UNVERIFIED below where the official page did not publish a
per-token rate for that model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

from loguru import logger

from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_usage import ProviderUsage

#
#######################################################################################################################
#
# Seed data
#
# Date the seed table below was last checked against official provider pricing pages.
_SEED_AS_OF = "2026-08-01"
# Anthropic and OpenAI were re-verified against their live pricing pages on
# this date (the final-review F5 wave); every other provider's rates still
# carry their original _SEED_AS_OF, because they were NOT re-checked then --
# `as_of` is the staleness defence surfaced in the PR3 tooltip, so it must
# never claim a verification that did not happen.
_RECHECKED_AS_OF = "2026-08-02"

# Providers that run locally: always $0.00.
#
# Written in POST-NORMALIZATION form: every lookup runs the caller's provider
# through `provider_config_key` first (lowercase, spaces/dashes -> "_"), which
# is the same key the rest of the app stores on a resolution
# (`ConsoleProviderResolution.provider` is the READINESS key -- "local_llm",
# "local_mlx_lm" -- not the execution key "local-llm"/"mlx_lm" that
# `Chat_Functions.API_CALL_HANDLERS` dispatches on). Seeding the execution
# spellings made a local send resolve to None (no pricing) instead of $0.00.
# Cross-checked against `provider_readiness.KEYLESS_PROVIDER_KEYS` plus the
# local-runtime sections in config.py's [api_settings].
LOCAL_PROVIDERS = frozenset(
    {
        "aphrodite",
        "koboldcpp",
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local_llm",
        "local_mlx_lm",
        "local_ollama",
        "local_onnx",
        "local_transformers",
        "local_vllm",
        "ollama",
        "oobabooga",
        "tabbyapi",
        "vllm",
    }
)

_ZERO = {
    "input_per_mtok": 0.0,
    "output_per_mtok": 0.0,
    "cache_read_per_mtok": 0.0,
    "cache_write_per_mtok": 0.0,
    "as_of": _SEED_AS_OF,
}


def _entry(
    inp: float, out: float, cr: Optional[float] = None, cw: Optional[float] = None,
    as_of: str = _SEED_AS_OF,
    audio_in: Optional[float] = None, audio_out: Optional[float] = None,
    cached_audio_in: Optional[float] = None,
    transcription_per_minute: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "input_per_mtok": inp,
        "output_per_mtok": out,
        "cache_read_per_mtok": cr,
        "cache_write_per_mtok": cw,
        "as_of": as_of,
        # task-2390 (realtime): all four default to None/absent for every
        # non-realtime entry above, so an ordinary text model's ModelPricing
        # carries no audio/transcription rate at all.
        "audio_in_per_mtok": audio_in,
        "audio_out_per_mtok": audio_out,
        "cached_audio_in_per_mtok": cached_audio_in,
        "transcription_per_minute": transcription_per_minute,
    }


def _stamped(as_of: str, entries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Restamp a provider block's entries with the date IT was verified."""
    return {key: {**value, "as_of": as_of} for key, value in entries.items()}


def _stamped_patterns(
    as_of: str, patterns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Restamp a provider's pattern list with the date IT was verified."""
    return [{**pattern, "as_of": as_of} for pattern in patterns]


def _lower_keys(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase every "provider:model" key so lookups are case-insensitive.

    get_pricing() always queries with a lowercased "provider:model" key; both
    the seed table and any config-supplied `[pricing].models` overrides must
    go through this before landing in `direct_mappings`, or a naturally-cased
    override key would silently never match.

    Only the PROVIDER half is run through ``provider_config_key`` (which also
    folds dashes to underscores): model ids are full of meaningful dashes
    ("claude-sonnet-5", "gpt-4o-mini"), so folding the whole key would
    scramble every one of them.
    """
    return {_normalize_pricing_key(key): value for key, value in mapping.items()}


def _normalize_pricing_key(key: Any) -> str:
    """Return a "<normalized provider>:<lowercased model>" lookup key."""
    text = str(key).strip()
    provider, separator, model = text.partition(":")
    if not separator:
        return text.lower()
    return f"{provider_config_key(provider)}:{model.strip().lower()}"


# Anthropic - re-verified 2026-08-02 against the current API pricing docs.
# House rule for the whole family: cache read = 0.1x input, cache write =
# 1.25x input (5-min TTL); the retired families below carry cache rates
# derived from that published multiplier, since Anthropic's archive pages
# publish only the input/output pair.
#
# Rates are per GENERATION, never per family name: Opus 4.1 bills $15/$75
# while Opus 4.6-4.8 and Opus 5 bill $5/$25, so the seeded entries and the
# pattern fallbacks below are both generation-scoped. A model whose
# generation is not recognized resolves to None (UI shows token counts)
# rather than silently inheriting an older generation's -- 3x wrong -- rate.
_ANTHROPIC_MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    # Current lineup (the models config.py actually ships/defaults to).
    "anthropic:claude-sonnet-5": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-opus-5": _entry(5.00, 25.00, 0.50, 6.25),
    "anthropic:claude-fable-5": _entry(10.00, 50.00, 1.00, 12.50),
    "anthropic:claude-opus-4-8": _entry(5.00, 25.00, 0.50, 6.25),
    "anthropic:claude-opus-4-7": _entry(5.00, 25.00, 0.50, 6.25),
    "anthropic:claude-opus-4-6": _entry(5.00, 25.00, 0.50, 6.25),
    "anthropic:claude-opus-4-1": _entry(15.00, 75.00, 1.50, 18.75),
    "anthropic:claude-sonnet-4-6": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-sonnet-4-5": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-haiku-4-5": _entry(1.00, 5.00, 0.10, 1.25),
    # Retired families kept for history rows people still have on disk.
    "anthropic:claude-3-7-sonnet": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-3-5-sonnet": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-3-opus": _entry(15.00, 75.00, 1.50, 18.75),
    "anthropic:claude-3-5-haiku": _entry(0.80, 4.00, 0.08, 1.00),
    "anthropic:claude-3-haiku": _entry(0.25, 1.25, 0.025, 0.3125),
}

# OpenAI - re-verified 2026-08-02 via https://developers.openai.com/api/docs/pricing
# (platform.openai.com/docs/pricing 301s there; openai.com/api/pricing/ 403s).
# Standard tier. "Cached input" maps to cache_read_per_mtok.
#
# The gpt-5.6 family is the only OpenAI line that publishes a "Cache writes"
# rate (every other model's cache_write is None), and it is also priced in
# two context tiers -- the short-context (<=272K input tokens) rate is
# seeded, since this per-model schema cannot express a context-dependent
# tier (same limitation noted for Gemini below).
_OPENAI_MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "openai:gpt-5.6-terra": _entry(2.00, 12.00, 0.20, 2.50),
    "openai:gpt-5.6-sol": _entry(5.00, 30.00, 0.50, 6.25),
    "openai:gpt-5.6-luna": _entry(0.20, 1.20, 0.02, 0.25),
    "openai:gpt-4o": _entry(2.50, 10.00, 1.25, None),
    "openai:gpt-4o-mini": _entry(0.15, 0.60, 0.075, None),
    "openai:gpt-4.1": _entry(2.00, 8.00, 0.50, None),
    "openai:gpt-4.1-mini": _entry(0.40, 1.60, 0.10, None),
    "openai:gpt-4.1-nano": _entry(0.10, 0.40, 0.025, None),
    "openai:gpt-5": _entry(1.25, 10.00, 0.125, None),
    "openai:gpt-5-mini": _entry(0.25, 2.00, 0.025, None),
    "openai:gpt-5-nano": _entry(0.05, 0.40, 0.005, None),
    "openai:gpt-5.1": _entry(1.25, 10.00, 0.125, None),
    "openai:gpt-5.2": _entry(1.75, 14.00, 0.175, None),
    "openai:o1": _entry(15.00, 60.00, 7.50, None),
    # o1-pro's cached-input cell is published as null, not as a number.
    "openai:o1-pro": _entry(150.00, 600.00, None, None),
    "openai:o3": _entry(2.00, 8.00, 0.50, None),
    "openai:o3-mini": _entry(1.10, 4.40, 0.55, None),
    "openai:o4-mini": _entry(1.10, 4.40, 0.275, None),
    # Deliberately NOT seeded: `chatgpt-4o-latest` (retired 2026-02-17) and
    # `o1-mini` (retired 2025-10-27) appear in config.py's model browse list
    # but carry no published per-token rate on any official page. They
    # resolve to None so the UI shows token counts rather than a fabricated
    # dollar figure -- a `[pricing].models` override is the escape hatch.
}

# OpenAI Realtime (voice) - verified 2026-08-06 via
# https://developers.openai.com/api/docs/pricing (task-2390's "## Research"
# section carries the full source table). Realtime bills per 1M TOKENS like
# every chat model above, NOT per audio minute -- the task's own Description
# premise was wrong on that point; only *transcription* (below) is
# per-minute. "openai:gpt-realtime" is the app's default (see
# Chat/console_voice_input.py's DEFAULT_REALTIME_MODEL).
#
# Cached AUDIO input is a SEPARATE rate from cached TEXT input
# (cache_read_per_mtok) -- they coincide at $0.40 for gpt-realtime but
# diverge for -mini ($0.30 audio vs $0.06 text). `cached_audio_in_per_mtok`
# is seeded here for the record (AC2: catalog entries exist for every
# published rate) but PricingCatalog.cost_for_usage() does not read it: see
# that method's own docstring for why (ProviderUsage cannot attribute a
# cache-read token to audio vs text -- ground truth in
# LLM_Calls/realtime/openai_session.py's header shows the wire payload
# splits `cached_tokens_details` this finely, but `ProviderUsage.
# from_provider_payload` does not parse that sub-object).
#
# No published cache-WRITE rate for realtime (prompt caching is automatic,
# server-managed) -- cache_write_per_mtok stays None like every OpenAI chat
# model above.
_REALTIME_AS_OF = "2026-08-06"
# Whisper transcribes realtime's input audio (see openai_session.py's
# `_TRANSCRIPTION_MODEL = "whisper-1"`) at a flat per-minute rate,
# independent of which gpt-realtime variant is running the session --
# duplicated across every entry below rather than looked up separately,
# since ProviderUsage never records the transcription sub-model.
_WHISPER_TRANSCRIPTION_PER_MINUTE = 0.006

_OPENAI_REALTIME_MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "openai:gpt-realtime": _entry(
        4.00, 16.00, 0.40, None, as_of=_REALTIME_AS_OF,
        audio_in=32.00, audio_out=64.00, cached_audio_in=0.40,
        transcription_per_minute=_WHISPER_TRANSCRIPTION_PER_MINUTE,
    ),
    "openai:gpt-realtime-mini": _entry(
        0.60, 2.40, 0.06, None, as_of=_REALTIME_AS_OF,
        audio_in=10.00, audio_out=20.00, cached_audio_in=0.30,
        transcription_per_minute=_WHISPER_TRANSCRIPTION_PER_MINUTE,
    ),
    "openai:gpt-realtime-2.1": _entry(
        4.00, 24.00, 0.40, None, as_of=_REALTIME_AS_OF,
        audio_in=32.00, audio_out=64.00, cached_audio_in=0.40,
        transcription_per_minute=_WHISPER_TRANSCRIPTION_PER_MINUTE,
    ),
    "openai:gpt-realtime-2": _entry(
        4.00, 24.00, 0.40, None, as_of=_REALTIME_AS_OF,
        audio_in=32.00, audio_out=64.00, cached_audio_in=0.40,
        transcription_per_minute=_WHISPER_TRANSCRIPTION_PER_MINUTE,
    ),
    "openai:gpt-realtime-1.5": _entry(
        4.00, 16.00, 0.40, None, as_of=_REALTIME_AS_OF,
        audio_in=32.00, audio_out=64.00, cached_audio_in=0.40,
        transcription_per_minute=_WHISPER_TRANSCRIPTION_PER_MINUTE,
    ),
}

DEFAULT_MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    **_stamped(_RECHECKED_AS_OF, _ANTHROPIC_MODEL_PRICING),
    **_stamped(_RECHECKED_AS_OF, _OPENAI_MODEL_PRICING),
    **_OPENAI_REALTIME_MODEL_PRICING,

    # Google Gemini - verified 2026-08-01 via https://ai.google.dev/gemini-api/docs/pricing
    # (standard, <=200k-token tier for models with a long-context surcharge).
    # Gemini has a context-caching *storage* fee ($/1M tokens/hour) that this
    # per-request schema cannot represent, so cache_write_per_mtok is None.
    "google:gemini-2.5-pro": _entry(1.25, 10.00, 0.125, None),
    "google:gemini-2.5-flash": _entry(0.30, 2.50, 0.03, None),
    "google:gemini-2.5-flash-lite": _entry(0.10, 0.40, 0.01, None),
    "google:gemini-2.0-flash": _entry(0.10, 0.40, None, None),
    "google:gemini-2.0-flash-lite": _entry(0.075, 0.30, None, None),

    # Mistral - verified 2026-08-01 via https://mistral.ai/pricing/api/. Seeded under
    # both "mistral:" and "mistralai:" since Chat_Functions.API_CALL_HANDLERS dispatches
    # both provider strings to chat_with_mistral. No cache pricing published.
    "mistral:mistral-large-latest": _entry(0.50, 1.50, None, None),
    "mistral:mistral-medium-latest": _entry(1.50, 7.50, None, None),
    "mistral:mistral-small-latest": _entry(0.15, 0.60, None, None),
    "mistral:codestral-latest": _entry(0.30, 0.90, None, None),
    "mistralai:mistral-large-latest": _entry(0.50, 1.50, None, None),
    "mistralai:mistral-medium-latest": _entry(1.50, 7.50, None, None),
    "mistralai:mistral-small-latest": _entry(0.15, 0.60, None, None),
    "mistralai:codestral-latest": _entry(0.30, 0.90, None, None),

    # Cohere - verified 2026-08-01 via https://cohere.com/pricing (per-token rates are
    # published only for "legacy" models on that page; current Command A is billed via
    # Model Vault instance pricing, so command-a below is UNVERIFIED - see task-2-report.md).
    "cohere:command-r-plus": _entry(2.50, 10.00, None, None),
    "cohere:command-r": _entry(0.50, 1.50, None, None),
    "cohere:command": _entry(1.00, 2.00, None, None),
    "cohere:command-light": _entry(0.30, 0.60, None, None),
    "cohere:command-a": _entry(2.50, 10.00, None, None),  # UNVERIFIED: no official per-token rate found

    # Groq - verified 2026-08-01 via https://groq.com/pricing/. No cache pricing published.
    "groq:llama-3.3-70b-versatile": _entry(0.59, 0.79, None, None),
    "groq:llama-3.1-8b-instant": _entry(0.05, 0.08, None, None),
    "groq:openai/gpt-oss-20b": _entry(0.075, 0.30, None, None),
    "groq:openai/gpt-oss-120b": _entry(0.15, 0.60, None, None),

    # DeepSeek - verified 2026-08-01 via https://api-docs.deepseek.com/quick_start/pricing/.
    # deepseek-chat/deepseek-reasoner were retired 2026-07-24; deepseek-v4-flash and
    # deepseek-v4-pro are the current lineup. "Cache hit" -> cache_read_per_mtok,
    # "cache miss" -> input_per_mtok (no separate cache-write rate is published).
    "deepseek:deepseek-v4-flash": _entry(0.14, 0.28, 0.0028, None),
    "deepseek:deepseek-v4-pro": _entry(0.435, 0.87, 0.003625, None),
}

# Pattern fallbacks exist to price DATED SNAPSHOTS of a seeded model
# ("claude-sonnet-4-5-20250929", "gpt-4o-2024-11-20"), never to extend a
# family name across generations. Every pattern is therefore anchored to the
# generation it was verified for; an unrecognized generation falls through to
# None, which the UI renders as token counts. The alternative -- a loose
# `^claude-opus` -- charged Opus 4.1's $15/$75 for every Opus 5 turn.
DEFAULT_PRICING_PATTERNS: Dict[str, List[Dict[str, Any]]] = {
    "anthropic": _stamped_patterns(_RECHECKED_AS_OF, [
        # Current generations.
        {"pattern": r"^claude-opus-4-1", **_entry(15.00, 75.00, 1.50, 18.75)},
        {"pattern": r"^claude-opus-(?:4-[678]|5)", **_entry(5.00, 25.00, 0.50, 6.25)},
        {"pattern": r"^claude-sonnet-(?:4-[56]|5)", **_entry(3.00, 15.00, 0.30, 3.75)},
        {"pattern": r"^claude-haiku-4-5", **_entry(1.00, 5.00, 0.10, 1.25)},
        {"pattern": r"^claude-fable-5", **_entry(10.00, 50.00, 1.00, 12.50)},
        # Retired generations (claude-3 family named its tier last).
        {"pattern": r"^claude-3-opus", **_entry(15.00, 75.00, 1.50, 18.75)},
        {"pattern": r"^claude-3-7-sonnet", **_entry(3.00, 15.00, 0.30, 3.75)},
        {"pattern": r"^claude-3-5-sonnet", **_entry(3.00, 15.00, 0.30, 3.75)},
        {"pattern": r"^claude-3-5-haiku", **_entry(0.80, 4.00, 0.08, 1.00)},
        {"pattern": r"^claude-3-haiku", **_entry(0.25, 1.25, 0.025, 0.3125)},
    ]),
    "openai": _stamped_patterns(_RECHECKED_AS_OF, [
        # Longer/more specific prefixes must precede their shorter siblings.
        {"pattern": r"^gpt-5\.6-terra", **_entry(2.00, 12.00, 0.20, 2.50)},
        {"pattern": r"^gpt-5\.6-sol", **_entry(5.00, 30.00, 0.50, 6.25)},
        {"pattern": r"^gpt-5\.6-luna", **_entry(0.20, 1.20, 0.02, 0.25)},
        {"pattern": r"^gpt-5\.2", **_entry(1.75, 14.00, 0.175, None)},
        {"pattern": r"^gpt-5\.1", **_entry(1.25, 10.00, 0.125, None)},
        {"pattern": r"^gpt-5-nano", **_entry(0.05, 0.40, 0.005, None)},
        {"pattern": r"^gpt-5-mini", **_entry(0.25, 2.00, 0.025, None)},
        # `(?:-\d|$)` keeps the BASE gpt-5 rate off future point releases:
        # a bare `^gpt-5` also matches "gpt-5.6-terra", which bills 2.00/12.00
        # rather than 1.25/10.00. Dated snapshots ("gpt-5-2025-08-07") still
        # resolve.
        {"pattern": r"^gpt-5(?:-\d|$)", **_entry(1.25, 10.00, 0.125, None)},
        {"pattern": r"^gpt-4\.1-nano", **_entry(0.10, 0.40, 0.025, None)},
        {"pattern": r"^gpt-4\.1-mini", **_entry(0.40, 1.60, 0.10, None)},
        {"pattern": r"^gpt-4\.1", **_entry(2.00, 8.00, 0.50, None)},
        {"pattern": r"^gpt-4o-mini", **_entry(0.15, 0.60, 0.075, None)},
        {"pattern": r"^gpt-4o", **_entry(2.50, 10.00, 1.25, None)},
        {"pattern": r"^o1-pro", **_entry(150.00, 600.00, None, None)},
        # Bare/dated o1 only -- `o1-mini` is retired with no published rate.
        {"pattern": r"^o1(?:-\d|$)", **_entry(15.00, 60.00, 7.50, None)},
        {"pattern": r"^o3-mini", **_entry(1.10, 4.40, 0.55, None)},
        {"pattern": r"^o3(?:-\d|$)", **_entry(2.00, 8.00, 0.50, None)},
        {"pattern": r"^o4-mini", **_entry(1.10, 4.40, 0.275, None)},
    ]),
    # Cohere publishes per-token rates only for the "legacy" models; Command A
    # is billed through Model Vault instance pricing, so its per-token entry
    # stays UNVERIFIED (see the direct mapping above). Patterned so the app's
    # own default model id -- "command-a-03-2025" -- resolves at all.
    # `command-r7b` is deliberately unmatched: it is a distinct, cheaper model
    # with no rate in this table, and inheriting command-r's would overbill.
    "cohere": [
        {"pattern": r"^command-a", **_entry(2.50, 10.00, None, None)},  # UNVERIFIED
        {"pattern": r"^command-r-plus", **_entry(2.50, 10.00, None, None)},
        {"pattern": r"^command-r(?:-\d|$)", **_entry(0.50, 1.50, None, None)},
        {"pattern": r"^command-light", **_entry(0.30, 0.60, None, None)},
        {"pattern": r"^command(?:-\d|$)", **_entry(1.00, 2.00, None, None)},
    ],
    "google": [
        {"pattern": r"^gemini-2\.5-flash-lite", **_entry(0.10, 0.40, 0.01, None)},
        {"pattern": r"^gemini-2\.5-flash", **_entry(0.30, 2.50, 0.03, None)},
        {"pattern": r"^gemini-2\.5-pro", **_entry(1.25, 10.00, 0.125, None)},
        {"pattern": r"^gemini-2\.0-flash-lite", **_entry(0.075, 0.30, None, None)},
        {"pattern": r"^gemini-2\.0-flash", **_entry(0.10, 0.40, None, None)},
    ],
    "mistral": [
        {"pattern": r"^mistral-large", **_entry(0.50, 1.50, None, None)},
        {"pattern": r"^mistral-medium", **_entry(1.50, 7.50, None, None)},
        {"pattern": r"^mistral-small", **_entry(0.15, 0.60, None, None)},
        {"pattern": r"^codestral", **_entry(0.30, 0.90, None, None)},
    ],
    "mistralai": [
        {"pattern": r"^mistral-large", **_entry(0.50, 1.50, None, None)},
        {"pattern": r"^mistral-medium", **_entry(1.50, 7.50, None, None)},
        {"pattern": r"^mistral-small", **_entry(0.15, 0.60, None, None)},
        {"pattern": r"^codestral", **_entry(0.30, 0.90, None, None)},
    ],
    "deepseek": [
        {"pattern": r"^deepseek-v4-flash", **_entry(0.14, 0.28, 0.0028, None)},
        {"pattern": r"^deepseek-v4-pro", **_entry(0.435, 0.87, 0.003625, None)},
    ],
}


#
#######################################################################################################################
#
# Dataclasses
#
def _models_dev_pricing(provider: str, model: str) -> "ModelPricing | None":
    """TASK-26023: a models.dev gap-fill price, or None.

    ``as_of="models.dev"`` records the origin so a displayed price is
    traceable to its source (AC#5). Returns None when the model is unknown
    upstream too, preserving the honest no-fabricated-price behavior (AC#6).
    """
    try:
        from tldw_chatbook.LLM_Provider_Catalog.models_dev_catalog import (
            models_dev_entry,
        )

        entry = models_dev_entry(provider, model)
    except Exception:  # noqa: BLE001 -- the gap-fill never breaks a lookup
        return None
    if entry is None or entry.input_price_per_mtok is None:
        return None
    return ModelPricing(
        input_per_mtok=entry.input_price_per_mtok,
        output_per_mtok=entry.output_price_per_mtok or 0.0,
        cache_read_per_mtok=None,
        cache_write_per_mtok=None,
        as_of="models.dev",
    )


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token USD rates for one resolved provider/model pair.

    Returned by :meth:`PricingCatalog.get_pricing`, which resolves a
    provider/model pair through direct mappings, then per-provider regex
    patterns, then a local-provider zero-rate fallback (see that method's
    own docstring for the full order). ``cache_read_per_mtok``/
    ``cache_write_per_mtok`` are ``None`` rather than ``0.0`` when a
    provider does not publish a cache rate at all, so
    :meth:`PricingCatalog.cost_for_usage` can distinguish "no cache
    discount published" (bill at $0 for that bucket) from "explicitly
    free" -- both currently price the bucket at zero, but keeping the
    distinction lets a future catalog update tell the two cases apart.

    Attributes:
        input_per_mtok: USD cost per 1,000,000 uncached input tokens.
        output_per_mtok: USD cost per 1,000,000 output tokens.
        cache_read_per_mtok: USD cost per 1,000,000 cache-read input
            tokens, or ``None`` when this provider/model has no published
            cache-read rate.
        cache_write_per_mtok: USD cost per 1,000,000 cache-write input
            tokens, or ``None`` when this provider/model has no published
            cache-write rate.
        as_of: Human-readable date/label for when this rate was last
            verified against the provider's published pricing, surfaced to
            the user so a stale rate is visibly stale rather than silently
            trusted.
        audio_in_per_mtok: USD cost per 1,000,000 UNCACHED audio input
            tokens (task-2390, realtime only), or ``None`` when this
            provider/model has no published audio rate -- i.e. every
            non-realtime model. Realtime audio bills per TOKEN like text,
            not per audio minute; see :meth:`PricingCatalog.cost_for_usage`
            for why this is the only audio-input rate that method reads.
        audio_out_per_mtok: USD cost per 1,000,000 audio output tokens
            (task-2390, realtime only), or ``None``. Output is never
            served from a cache, so this rate is unambiguous to apply.
        cached_audio_in_per_mtok: USD cost per 1,000,000 CACHED audio
            input tokens (task-2390), or ``None``. Seeded for the record
            (a genuinely published, separate rate -- it diverges from
            ``cache_read_per_mtok`` for at least one shipped model) but
            :meth:`PricingCatalog.cost_for_usage` does NOT read it: see
            that method's docstring for the attribution gap that makes it
            currently unusable.
        transcription_per_minute: USD cost per minute of input-audio
            transcription (task-2390, realtime only -- OpenAI's Whisper
            sub-model), or ``None``. The one PER-MINUTE (not per-token)
            rate in this dataclass, feeding ``ProviderUsage.
            transcription_seconds`` rather than any token bucket.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: Optional[float]
    cache_write_per_mtok: Optional[float]
    as_of: str
    audio_in_per_mtok: Optional[float] = None
    audio_out_per_mtok: Optional[float] = None
    cached_audio_in_per_mtok: Optional[float] = None
    transcription_per_minute: Optional[float] = None


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Dollar cost of one :class:`ProviderUsage` record, bucket by bucket.

    Returned by :meth:`PricingCatalog.cost_for_usage`, which multiplies each
    of a ``ProviderUsage``'s disjoint token buckets by the matching
    :class:`ModelPricing` rate (dividing by 1,000,000 since rates are
    per-million-token) and rounds every field -- including ``total`` -- to
    6 decimal places independently, so a UI reading only ``total`` never
    has to re-derive it from the per-bucket costs to get a consistent
    figure.

    Attributes:
        input_cost: USD cost of the uncached (TEXT-only, once audio is
            accounted for separately -- see ``audio_input_cost``) input
            tokens.
        cache_read_cost: USD cost of the cache-read (TEXT-only) input
            tokens.
        cache_write_cost: USD cost of the cache-write input tokens.
        output_cost: USD cost of the (TEXT-only) output tokens.
        total: Sum of every cost field on this dataclass.
        as_of: The ``ModelPricing.as_of`` label the rates were resolved
            from, carried through so a displayed cost can show how current
            its pricing basis is.
        audio_input_cost: USD cost of ``ProviderUsage.audio_input``
            (task-2390, realtime only), ``0.0`` when the resolved
            ``ModelPricing`` has no audio rate or the usage has no audio
            tokens. Kept as its own field -- rather than folded into
            ``input_cost`` -- so the breakdown modal can show it as a
            distinct line instead of an undecomposable total.
        audio_output_cost: USD cost of ``ProviderUsage.audio_output``
            (task-2390), same "own field" rationale as ``audio_input_cost``.
        transcription_cost: USD cost of ``ProviderUsage.
            transcription_seconds`` at ``ModelPricing.
            transcription_per_minute`` (task-2390), ``0.0`` when either is
            unset.
    """

    input_cost: float
    cache_read_cost: float
    cache_write_cost: float
    output_cost: float
    total: float
    as_of: str
    audio_input_cost: float = 0.0
    audio_output_cost: float = 0.0
    transcription_cost: float = 0.0


#
#######################################################################################################################
#
# PricingCatalog class
#
class PricingCatalog:
    """
    Resolves per-model pricing and computes dollar costs from ProviderUsage.

    Mirrors the structure of tldw_chatbook.model_capabilities.ModelCapabilities:
    plain-dict seeds, config-overridable direct mappings and per-provider regex
    patterns, and a case-insensitive provider index.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration.

        Args:
            config: `[pricing]` configuration dict. If None, loads from config file.
        """
        if config is None:
            # Load from config file
            # Get pricing from config - it's a top-level section
            from tldw_chatbook.config import load_cli_config_and_ensure_existence

            full_config = load_cli_config_and_ensure_existence()
            config = full_config.get("pricing", {})

        # Direct "provider:model" mappings (highest priority). Seed table merged
        # with config overrides - overrides win on a per-key basis. get_pricing()
        # always looks up a lowercased "provider:model" key, so both the seed and
        # any config-supplied keys must be normalized to lowercase here - otherwise
        # a naturally-capitalized TOML key (e.g. "Anthropic:Claude-Sonnet-4-6")
        # would sit under its original casing and silently never match a lookup.
        self.direct_mappings: Dict[str, Dict[str, Any]] = {
            **_lower_keys(DEFAULT_MODEL_PRICING),
            **_lower_keys(config.get("models", {})),
        }

        # Pattern configurations by provider, merged the same way (a provider
        # key present in config REPLACES that provider's whole pattern list;
        # providers absent from config keep their seeded patterns). Both sides
        # are keyed by the normalized provider so a config-supplied
        # "Anthropic" replaces the seeded "anthropic" list rather than sitting
        # beside it as a second, unreachable entry.
        self.pattern_configs: Dict[str, List[Dict[str, Any]]] = {
            **{
                provider_config_key(provider): patterns
                for provider, patterns in DEFAULT_PRICING_PATTERNS.items()
            },
            **{
                provider_config_key(provider): patterns
                for provider, patterns in config.get("patterns", {}).items()
            },
        }

        # Compile patterns for efficiency
        self._compiled_patterns = self._compile_patterns()
        # Normalized provider index: callers pass whatever spelling the app
        # stored ("Anthropic", "local-llm", "local_llm") while pattern keys
        # are whatever case/dash style the seed or config used. Both sides go
        # through `provider_config_key`, the app's own provider normalization.
        self._provider_key_by_normalized = {
            provider_config_key(provider): provider
            for provider in self._compiled_patterns
        }

        self._zero_pricing = self._to_model_pricing(_ZERO)

        logger.debug(
            f"PricingCatalog initialized with {len(self.direct_mappings)} direct mappings and "
            f"patterns for {len(self.pattern_configs)} providers"
        )

    def _compile_patterns(self) -> Dict[str, List[Tuple[Pattern, Dict[str, Any]]]]:
        """Compile regex patterns for each provider."""
        compiled: Dict[str, List[Tuple[Pattern, Dict[str, Any]]]] = {}

        for provider, patterns in self.pattern_configs.items():
            compiled_list = []
            for pattern_config in patterns:
                if isinstance(pattern_config, dict) and "pattern" in pattern_config:
                    try:
                        pattern = re.compile(pattern_config["pattern"], re.IGNORECASE)
                        # Extract the pricing entry from pattern config
                        entry = {k: v for k, v in pattern_config.items() if k != "pattern"}
                        compiled_list.append((pattern, entry))
                    except re.error as e:
                        logger.error(
                            f"Invalid regex pattern for {provider}: {pattern_config['pattern']} - {e}"
                        )

            if compiled_list:
                compiled[provider] = compiled_list
                logger.debug(f"Compiled {len(compiled_list)} pricing patterns for provider {provider}")

        return compiled

    @staticmethod
    def _to_model_pricing(entry: Dict[str, Any]) -> ModelPricing:
        return ModelPricing(
            input_per_mtok=float(entry["input_per_mtok"]),
            output_per_mtok=float(entry["output_per_mtok"]),
            cache_read_per_mtok=entry.get("cache_read_per_mtok"),
            cache_write_per_mtok=entry.get("cache_write_per_mtok"),
            as_of=entry.get("as_of", _SEED_AS_OF),
            # task-2390: absent on every non-realtime entry, so `.get()`
            # leaves these None -- additive, never repurposes an existing
            # key.
            audio_in_per_mtok=entry.get("audio_in_per_mtok"),
            audio_out_per_mtok=entry.get("audio_out_per_mtok"),
            cached_audio_in_per_mtok=entry.get("cached_audio_in_per_mtok"),
            transcription_per_minute=entry.get("transcription_per_minute"),
        )

    def get_pricing(self, provider: str, model: str) -> Optional[ModelPricing]:
        """
        Resolve pricing for a provider/model pair.

        Resolution order: direct "provider:model" mapping -> provider pattern
        fallback (first match wins) -> local-provider zero-rate -> None.

        The provider is normalized through ``provider_config_key`` -- the same
        mapping the rest of the app uses -- so the readiness spelling stored on
        a message ("local_llm") and the execution spelling used by
        ``Chat_Functions.API_CALL_HANDLERS`` ("local-llm") resolve to the same
        rates instead of one of them silently returning None.

        Args:
            provider: The provider name (case- and dash-insensitive).
            model: The model identifier (case-insensitive).

        Returns:
            A ModelPricing instance, or None if no pricing data is available.
        """
        provider_key_normalized = provider_config_key(provider)
        model_l = (model or "").strip().lower()

        # 1. Direct mapping (highest priority)
        entry = self.direct_mappings.get(f"{provider_key_normalized}:{model_l}")
        if entry is not None:
            return self._to_model_pricing(entry)

        # 2. Provider-specific patterns (normalized provider match)
        provider_key = self._provider_key_by_normalized.get(provider_key_normalized)
        if provider_key is not None:
            for pattern, pattern_entry in self._compiled_patterns[provider_key]:
                if pattern.match(model_l):
                    return self._to_model_pricing(pattern_entry)

        # 3. Local providers always cost $0.00
        if provider_key_normalized in LOCAL_PROVIDERS:
            return self._zero_pricing

        # 4. TASK-26023: upstream models.dev as a LOWER-priority gap-fill,
        # beneath the hand-maintained direct/pattern entries above (AC#2).
        # Disabled by default and network-free -- see models_dev_catalog.
        upstream = _models_dev_pricing(provider_key_normalized, model_l)
        if upstream is not None:
            return upstream

        # 5. Unknown model: no fabricated price, let the UI fall back to token counts.
        return None

    def cost_for_usage(self, usage: ProviderUsage) -> Optional[CostBreakdown]:
        """
        Compute a dollar cost breakdown for a ProviderUsage's disjoint token buckets.

        task-2390 (realtime audio/transcription billing): ``ProviderUsage.
        audio_input``/``audio_output`` are documented SUBSETS of
        ``uncached_input``+``cache_read`` and ``output`` respectively (see
        that dataclass's own docstring) -- never additional tokens, so
        pricing them on top of the plain buckets without adjustment would
        double count. Two attribution gaps this method resolves
        conservatively rather than guessing:

        1. Cache attribution (input side only): ``ProviderUsage`` cannot
           say how many of ``audio_input``'s tokens came from
           ``uncached_input`` vs ``cache_read`` -- the wire payload splits
           ``cached_tokens_details`` this finely (see
           ``LLM_Calls/realtime/openai_session.py``'s ground-truth header)
           but ``from_provider_payload`` does not parse that sub-object,
           and this method does not invent a split. Every audio-input
           token is therefore priced at the (higher) UNCACHED audio rate
           -- ``ModelPricing.cached_audio_in_per_mtok`` is never read here
           -- regardless of which bucket it is numerically attributed to
           below; that attribution only decides how many TEXT tokens are
           left over in each bucket, never audio's own rate.

           To avoid double-counting those same audio tokens under
           ``input_cost``/``cache_read_cost``, they must be subtracted
           from ``uncached_input``/``cache_read`` before those buckets are
           priced at their (much cheaper) TEXT rates. Which bucket a given
           audio token is subtracted from therefore changes how many TEXT
           tokens remain in the expensive ``uncached_input`` bucket vs the
           cheap ``cache_read`` bucket -- and since ``input_per_mtok`` is
           always higher than ``cache_read_per_mtok``, the conservative
           (cost-MAXIMIZING, never-underbilling) choice is to remove audio
           tokens from ``cache_read`` FIRST, leaving as many TEXT tokens
           as possible stranded in the expensive ``uncached_input``
           bucket -- and only "spill" the removal into ``uncached_input``
           once ``cache_read`` itself is exhausted. (Removing from the
           EXPENSIVE bucket first -- the intuitive-looking order -- is
           backwards: it evacuates text OUT of the bucket that costs the
           most, which minimizes the bill instead of bounding it from
           above. Don't "fix" this back without re-deriving the sign.)
        2. Output side: unambiguous. Output is never served from a cache
           (only input can be), so ``audio_output`` is a clean subset of
           ``output`` alone with no cache-attribution question at all.

        Args:
            usage: Normalized per-message token usage.

        Returns:
            A CostBreakdown, or None if pricing for usage.provider/usage.model is unknown.
        """
        pricing = self.get_pricing(usage.provider, usage.model)
        if pricing is None:
            return None

        # Defensive clamp: a corrupted stored record (e.g. hand-edited JSON)
        # could carry an audio count larger than its own parent bucket(s);
        # without this, the subtractions below would go negative.
        audio_input = min(usage.audio_input, usage.uncached_input + usage.cache_read)
        audio_output = min(usage.audio_output, usage.output)

        if pricing.audio_in_per_mtok is not None and audio_input:
            # Drain the CHEAP bucket (cache_read) first -- see this
            # method's docstring, point 1, for why that (not uncached_input
            # first) is the conservative direction.
            audio_from_cache = min(audio_input, usage.cache_read)
            audio_from_uncached = audio_input - audio_from_cache
            text_uncached = usage.uncached_input - audio_from_uncached
            text_cache_read = usage.cache_read - audio_from_cache
            audio_input_cost = round(audio_input * pricing.audio_in_per_mtok / 1_000_000, 6)
        else:
            text_uncached = usage.uncached_input
            text_cache_read = usage.cache_read
            audio_input_cost = 0.0

        if pricing.audio_out_per_mtok is not None and audio_output:
            text_output = usage.output - audio_output
            audio_output_cost = round(audio_output * pricing.audio_out_per_mtok / 1_000_000, 6)
        else:
            text_output = usage.output
            audio_output_cost = 0.0

        input_cost = round(text_uncached * pricing.input_per_mtok / 1_000_000, 6)
        cache_read_cost = round(
            text_cache_read * (pricing.cache_read_per_mtok or 0.0) / 1_000_000, 6
        )
        cache_write_cost = round(
            usage.cache_write * (pricing.cache_write_per_mtok or 0.0) / 1_000_000, 6
        )
        output_cost = round(text_output * pricing.output_per_mtok / 1_000_000, 6)
        transcription_cost = round(
            usage.transcription_seconds / 60.0 * (pricing.transcription_per_minute or 0.0), 6
        )
        total = round(
            input_cost + cache_read_cost + cache_write_cost + output_cost
            + audio_input_cost + audio_output_cost + transcription_cost,
            6,
        )

        return CostBreakdown(
            input_cost=input_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            output_cost=output_cost,
            total=total,
            as_of=pricing.as_of,
            audio_input_cost=audio_input_cost,
            audio_output_cost=audio_output_cost,
            transcription_cost=transcription_cost,
        )


#
#######################################################################################################################
#
# Module-level convenience functions
#

# Global instance (lazy-loaded)
_global_catalog: Optional[PricingCatalog] = None


def get_pricing_catalog() -> PricingCatalog:
    """
    Get the global PricingCatalog instance.

    Returns:
        PricingCatalog instance configured from user settings.
    """
    global _global_catalog
    if _global_catalog is None:
        _global_catalog = PricingCatalog()
    return _global_catalog


def reload_pricing_catalog() -> None:
    """Reload the pricing catalog from configuration."""
    global _global_catalog
    _global_catalog = None
    logger.info("Pricing catalog reloaded from configuration")


#
# End of pricing_catalog.py
#######################################################################################################################
