# tldw_chatbook/model_capabilities.py
# Description: Configuration-based model capability detection system
#
# This module provides a flexible way to detect model capabilities (like vision support)
# based on user configuration, eliminating the need for code updates when new models are released.
#
# Imports
#
# Standard Library
import re
from typing import Dict, List, Any, Optional, Pattern, Tuple
from functools import lru_cache
import logging

# Local Imports

# Configure logger
logger = logging.getLogger(__name__)

#
#######################################################################################################################
#
# Default Model Patterns
#
# These defaults are used if the user hasn't configured model_capabilities in their config.
# Users can override or extend these in their config.toml file.
DEFAULT_MODEL_PATTERNS = {
    "OpenAI": [
        {"pattern": r"^gpt-4.*vision", "vision": True, "context_window": 128000},
        {
            "pattern": r"^gpt-4[o0](?:-mini)?",
            "vision": True,
            "context_window": 128000,
        },  # gpt-4o, gpt-40, gpt-4o-mini
        {"pattern": r"^gpt-4.*turbo", "vision": True, "context_window": 128000},
        {"pattern": r"^gpt-4\.1", "vision": True, "context_window": 1047576},  # gpt-4.1 series
        {
            "pattern": r"^o[34](?:-mini)?",
            "vision": True,
            "context_window": 200000,
        },  # o3, o4, o3-mini, o4-mini series
        {"pattern": r"^dall-e", "vision": True, "image_generation": True},
    ],
    "Anthropic": [
        {"pattern": r"^claude-3", "vision": True, "context_window": 200000},  # All Claude 3 models have vision
        {"pattern": r"^claude.*opus-4", "vision": True, "context_window": 200000},  # Claude Opus 4 series
        {"pattern": r"^claude.*sonnet-4", "vision": True, "context_window": 200000},  # Claude Sonnet 4 series
    ],
    "Google": [
        {"pattern": r"gemini.*vision", "vision": True},
        {
            "pattern": r"gemini-[0-9.]+-(pro|flash)",
            "vision": True,
        },  # Modern Gemini models
        {"pattern": r"gemini-2\.", "vision": True},  # Gemini 2.x series
    ],
    "OpenRouter": [
        # OpenRouter uses provider/model format
        {"pattern": r"openai/gpt-4.*vision", "vision": True},
        {"pattern": r"openai/gpt-4[o0]", "vision": True},
        {"pattern": r"openai/gpt-4\.1", "vision": True},
        {"pattern": r"openai/o[34](?:-mini)?", "vision": True},
        {"pattern": r"anthropic/claude-3", "vision": True},
        {"pattern": r"google/gemini.*vision", "vision": True},
        {"pattern": r"google/gemini-[0-9.]+-(pro|flash)", "vision": True},
    ],
    "Moonshot": [
        # Moonshot vision models
        {
            "pattern": r"moonshot-v1-.*-vision-preview",
            "vision": True,
        },  # Matches all vision preview models
        {"pattern": r"moonshot-v1-8k-vision-preview", "vision": True},
        {"pattern": r"moonshot-v1-32k-vision-preview", "vision": True},
        {"pattern": r"moonshot-v1-128k-vision-preview", "vision": True},
    ],
    "ZAI": [
        # Z.AI models - currently no vision support
        {
            "pattern": r"^glm-",
            "vision": False,
        }  # All GLM models currently don't support vision
    ],
}

# Known models with direct capabilities (for common models)
DEFAULT_MODEL_CAPABILITIES = {
    # OpenAI
    "gpt-4-vision-preview": {"vision": True, "max_images": 1, "context_window": 128000},
    "gpt-4-turbo": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4-turbo-2024-04-09": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4o": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4o-mini": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-5.6-terra": {"vision": True, "max_images": 10},
    "gpt-4.1-2025-04-14": {"vision": True, "max_images": 10, "context_window": 1047576},
    "o4-mini-2025-04-16": {"vision": True, "max_images": 10, "context_window": 200000},
    "o3-2025-04-16": {"vision": True, "max_images": 10, "context_window": 200000},
    "o3-mini-2025-01-31": {"vision": True, "max_images": 10, "context_window": 200000},
    "gpt-4.1-mini-2025-04-14": {"vision": True, "max_images": 10, "context_window": 1047576},
    "gpt-4.1-nano-2025-04-14": {"vision": True, "max_images": 10, "context_window": 1047576},
    # Anthropic
    "claude-3-opus-20240229": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-sonnet-20240229": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-haiku-20240307": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-5-sonnet-20240620": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-5-sonnet-20241022": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-sonnet-5": {"vision": True, "max_images": 5},
    # Google
    "gemini-pro-vision": {"vision": True, "max_images": 1, "context_window": 12288},
    "gemini-1.5-pro": {"vision": True, "max_images": 10, "context_window": 2097152},
    "gemini-1.5-flash": {"vision": True, "max_images": 10, "context_window": 1048576},
    "gemini-2.0-flash": {"vision": True, "max_images": 10, "context_window": 1048576},
    # Moonshot
    "moonshot-v1-8k-vision-preview": {"vision": True, "max_images": 1},
    "moonshot-v1-32k-vision-preview": {"vision": True, "max_images": 1},
    "moonshot-v1-128k-vision-preview": {"vision": True, "max_images": 1},
    # Z.AI Models
    "glm-4.5": {"vision": False, "max_tokens": 8192},
    "glm-4.5-air": {"vision": False, "max_tokens": 8192},
    "glm-4.5-x": {"vision": False, "max_tokens": 8192},
    "glm-4.5-airx": {"vision": False, "max_tokens": 8192},
    "glm-4.5-flash": {"vision": False, "max_tokens": 16384},
    "glm-4-32b-0414-128k": {"vision": False, "max_tokens": 128000},
}


#
#######################################################################################################################
#
# Anthropic per-model request capabilities
#
# These answer two questions about what the *provider API* accepts, not what the
# user prefers:
#
#   1. Does this model reject the sampling parameters (temperature/top_p/top_k)?
#   2. Does this model reject a fixed thinking budget (thinking.budget_tokens)?
#
# Both were previously encoded as ad-hoc name checks inside the Anthropic request
# builder, which knew only about Claude Sonnet 5 -- so every newer model release
# silently broke the provider with an HTTP 400 (TASK-18414).
#
# Deliberately NOT part of the config-driven tables above. Those are wholly
# replaceable from `config.toml` (`config.get("models", ...)` /
# `config.get("patterns", ...)`), and a direct mapping shadows every pattern --
# `claude-sonnet-5` already has one. A user edit to a request-*validity* fact
# could therefore only reintroduce the 400 it exists to prevent.
#
# Families are matched by (tier, major, minor) so that bare ids, dotted variants,
# dated/suffixed snapshots and provider-prefixed forms all resolve, without
# over-matching the older families that still accept both parameters.
_ANTHROPIC_FAMILY_RE = re.compile(
    r"claude[-_.]?(?P<tier>opus|sonnet|haiku|fable|mythos)[-_.]?(?P<major>\d+)"
    r"(?:[-_.](?P<minor>\d+))?",
    re.IGNORECASE,
)

# A ``None`` minor means "every minor version in this major line".
# Verified live against api.anthropic.com on 2026-08-18: each family below
# returns 400 for `temperature`/`top_p`/`top_k` and for
# `thinking={"type": "enabled", "budget_tokens": N}`, while Opus 4.6, Sonnet 4.5
# and Haiku 4.5 return 200 for the same payloads.
_ANTHROPIC_MODERN_REQUEST_FAMILIES = frozenset(
    {
        ("fable", 5, None),
        ("mythos", 5, None),
        ("opus", 5, None),
        ("opus", 4, 8),
        ("opus", 4, 7),
        ("sonnet", 5, None),
    }
)


def _anthropic_model_family(model: object) -> Optional[Tuple[str, int, Optional[int]]]:
    """Parse an Anthropic model id into ``(tier, major, minor)``.

    Args:
        model: A model identifier in any form the codebase passes through --
            bare (``claude-opus-5``), dotted (``claude-opus-4.8``), dated
            (``claude-opus-4-5-20251101``), suffixed (``claude-opus-4-8-fast``)
            or provider-prefixed (``anthropic/claude-opus-5``,
            ``us.anthropic.claude-opus-4-8``).

    Returns:
        The parsed family tuple, or ``None`` when the id is not a recognisable
        modern Anthropic model name (including the ``claude-3-5-sonnet-*``
        generation, whose tier follows the version rather than preceding it).
    """
    if not isinstance(model, str):
        return None
    match = _ANTHROPIC_FAMILY_RE.search(model.strip())
    if match is None:
        return None
    minor = match.group("minor")
    return (
        match.group("tier").lower(),
        int(match.group("major")),
        int(minor) if minor is not None else None,
    )


def _anthropic_family_matches(
    model: object, families: frozenset  # frozenset[Tuple[str, int, Optional[int]]]
) -> bool:
    """Return whether ``model`` parses into one of ``families``.

    A ``(tier, major, None)`` row matches every minor version in that major
    line; a ``(tier, major, minor)`` row matches that exact minor only.
    """
    family = _anthropic_model_family(model)
    if family is None:
        return False
    tier, major, minor = family
    return (tier, major, None) in families or (tier, major, minor) in families


def _anthropic_is_modern_request_family(model: object) -> bool:
    """Return whether ``model`` is in the modern Anthropic request family."""
    return _anthropic_family_matches(model, _ANTHROPIC_MODERN_REQUEST_FAMILIES)


def anthropic_model_rejects_sampling_params(model: object) -> bool:
    """Return whether ``model`` rejects ``temperature``/``top_p``/``top_k``.

    Args:
        model: An Anthropic model identifier (any prefixed or suffixed form).

    Returns:
        True when sending any sampling parameter would be answered with
        ``400 invalid_request_error: `temperature` is deprecated for this
        model.`` -- the Fable 5, Mythos 5, Opus 5, Opus 4.8, Opus 4.7 and
        Sonnet 5 families. False for Opus 4.6 and earlier, Sonnet 4.6/4.5 and
        Haiku, which still accept them.
    """
    return _anthropic_is_modern_request_family(model)


def anthropic_model_rejects_fixed_thinking_budget(model: object) -> bool:
    """Return whether ``model`` rejects ``thinking.budget_tokens``.

    Args:
        model: An Anthropic model identifier (any prefixed or suffixed form).

    Returns:
        True when ``thinking={"type": "enabled", "budget_tokens": N}`` would be
        answered with ``400 invalid_request_error: "thinking.type.enabled" is
        not supported for this model``; such a model must use adaptive thinking
        plus ``output_config.effort`` instead.

    Note:
        This currently covers exactly the same families as
        :func:`anthropic_model_rejects_sampling_params` -- Anthropic removed both
        parameters in the same generation -- but they are separate questions
        about the request surface and are kept as separate predicates so a future
        model can answer them differently.
    """
    return _anthropic_is_modern_request_family(model)


def anthropic_model_rejects_temperature_top_p_combination(model: object) -> bool:
    """Return whether ``model`` rejects ``temperature`` and ``top_p`` together.

    Distinct from :func:`anthropic_model_rejects_sampling_params`: the families
    that still *accept* sampling parameters individually reject the pair --
    ``400 invalid_request_error: `temperature` and `top_p` cannot both be
    specified for this model. Please use only one.``

    Probe-verified against api.anthropic.com with the exact trio the
    summarization path used to build (``temperature=0.1, top_k=0, top_p=1.0``):

    * 2026-08-20 (TASK-18802 discovery probes): ``claude-haiku-4-5``
      (req_011CeEDXPHNyF7apkaZepbTN) and ``claude-sonnet-4-5``
      (req_011CeEDXa9V99yBoHN5vcjDG) -> 400; ``temperature`` alone and
      ``temperature``+``top_k`` -> 200 (req_011CeEDXQwqi7yXoozbdrXFX,
      req_011CeEDXVk4nXXCoBGdf9mFm).
    * 2026-08-20 (TASK-19020 boundary probes): ``claude-opus-4-6``
      (req_011CeEFGsbHd7VCjcjz4etar), ``claude-sonnet-4-6``
      (req_011CeEFGuRfeCzC6PiLyDtFb) and ``claude-opus-4-5``
      (req_011CeEFGvySC6z61NDRH5uN5) -> the identical 400;
      ``claude-opus-4-6`` + ``temperature``+``top_k`` without ``top_p`` -> 200
      (msg_011CeEFGzjeXQ6ftPf9KH45n).

    Together with Anthropic's published migration guidance ("passing both will
    error on every Claude 4+ model"), the rule covers every tier-first-named
    family -- a naming scheme that began with the Claude 4 generation -- so the
    predicate is true for any id the family parser recognises with major >= 4.
    The Claude 3.x generation, which accepted the pair, is number-first-named
    (``claude-3-haiku-20240307``), never parses into a family, and is entirely
    retired (that id itself now 404s: req_011CeEDXZ8iS29MZCgyySwQa); unparsed
    ids keep their historical payload unchanged.

    Args:
        model: An Anthropic model identifier (any prefixed or suffixed form).

    Returns:
        True when sending ``temperature`` and ``top_p`` in the same request
        would be answered with the 400 above. A caller holding both must send
        temperature and drop top_p (``top_k`` remains compatible alongside
        temperature). False for unrecognisable ids.
    """
    family = _anthropic_model_family(model)
    if family is None:
        return False
    _tier, major, _minor = family
    return major >= 4


# How "thinking off" must be expressed, per family (TASK-18800). Two separate
# facts about the request surface, composed by the request builder into a
# three-way behaviour:
#
#   thinks_by_default=False                          -> omission already means
#       no thinking (Opus 4.8/4.7/4.6 and earlier, Sonnet 4.6/4.5, Haiku)
#   thinks_by_default=True, rejects_disabled=False   -> OFF needs an explicit
#       thinking={"type": "disabled"} (Sonnet 5, Opus 5)
#   thinks_by_default=True, rejects_disabled=True    -> OFF cannot be expressed
#       at all; omission is the only valid move and adaptive thinking still
#       runs (Fable 5, Mythos 5)
#
# Kept as two boolean predicates rather than one three-way enum because they
# are independent questions a future model could answer in a new combination,
# and because boolean family predicates are this module's established shape
# (TASK-18414 / TASK-19020).
#
# Probe-verified against api.anthropic.com on 2026-08-20 (TASK-18800 report):
#
#   * claude-opus-5 + thinking={"type": "disabled"}, no effort -> 200
#     (msg_011CeFGfHpYVXE7X7LnRmYCF, thinking_tokens 0). The effort cap on
#     disabled thinking binds only at xhigh/max (req_011CeFGfT1wJxmsd2rRUszbc);
#     the builder's OFF branch never pairs disabled with an effort.
#   * claude-opus-5, thinking omitted -> 200 WITH a thinking block and 13
#     billed thinking tokens (msg_011CeFGfkyS1LDXT46nVU5Gb) -- omission runs
#     thinking on this family.
#   * claude-fable-5 + thinking={"type": "disabled"} -> 400
#     '"thinking.type.disabled" is not supported for this model.'
#     (req_011CeFGfU3CpiKFwRigU2jRa); omitted -> 200 with a thinking block
#     even for "Say OK." (msg_011CeFGfVjQMY6gm6SzUDpHq, thinking_tokens 7).
#   * claude-sonnet-5 + disabled -> 200 (msg_011CeFGfvkJLz54KaEvyYXTY).
#   * claude-sonnet-4-6 / claude-haiku-4-5, thinking omitted -> 200 with no
#     thinking block (msg_011CeFGfzwKC4Uqtx2G7oYJW, msg_011CeFGg5DnVz6iXa22WhgRj).
#   * claude-mythos-5 is Project Glasswing-only (404 on this key,
#     req_011CeFGg7HZZenFq2CaQxi5A) and is included on the documented grounds
#     that it shares Fable 5's request surface exactly -- same standing as in
#     the TASK-18414 capability set.
_ANTHROPIC_DEFAULT_THINKING_FAMILIES = frozenset(
    {
        ("sonnet", 5, None),
        ("opus", 5, None),
        ("fable", 5, None),
        ("mythos", 5, None),
    }
)

_ANTHROPIC_ALWAYS_ON_THINKING_FAMILIES = frozenset(
    {
        ("fable", 5, None),
        ("mythos", 5, None),
    }
)


def anthropic_model_thinks_by_default(model: object) -> bool:
    """Return whether omitting ``thinking`` leaves thinking RUNNING on ``model``.

    Args:
        model: An Anthropic model identifier (any prefixed or suffixed form).

    Returns:
        True when a request with no ``thinking`` key runs (and bills) adaptive
        thinking -- the Sonnet 5, Opus 5, Fable 5 and Mythos 5 families -- so
        that turning thinking off requires more than omission. False for
        Opus 4.8 and earlier, Sonnet 4.6 and earlier, and Haiku, where
        omission already means no thinking, and for unrecognisable ids.
    """
    return _anthropic_family_matches(model, _ANTHROPIC_DEFAULT_THINKING_FAMILIES)


def anthropic_model_rejects_disabled_thinking(model: object) -> bool:
    """Return whether ``thinking={"type": "disabled"}`` is a 400 on ``model``.

    Args:
        model: An Anthropic model identifier (any prefixed or suffixed form).

    Returns:
        True when an explicit disabled config would be answered with
        ``400 invalid_request_error: "thinking.type.disabled" is not supported
        for this model.`` -- the always-on-thinking Fable 5 and Mythos 5
        families, where omission is the only valid move and thinking runs
        regardless. False everywhere else, including Opus 5 (which accepts
        ``disabled`` alongside effort ``high`` or lower -- and the builder's
        OFF branch sends no effort at all).
    """
    return _anthropic_family_matches(model, _ANTHROPIC_ALWAYS_ON_THINKING_FAMILIES)


#
#######################################################################################################################
#
# OpenAI per-model request capabilities
#
# Same design as the Anthropic predicates above (TASK-18414), for the same two
# reasons: these are facts about what api.openai.com *accepts*, not user
# preferences, so they live outside the config-driven tables; and a direct
# mapping in those tables shadows every pattern, so a pattern row could never
# be trusted to fire.
#
# Probe-verified against api.openai.com on 2026-08-20 (TASK-18802) with the
# exact payload shape the summarization path builds:
#
#   * gpt-5, gpt-5.6, o3, o4-mini + ``max_tokens`` -> 400
#     ``unsupported_parameter: 'max_tokens' is not supported with this model.
#     Use 'max_completion_tokens' instead.``
#   * gpt-5, gpt-5.6 + ``temperature: 0.7`` -> 400
#     ``unsupported_value: 'temperature' does not support 0.7 with this model.
#     Only the default (1) value is supported.``
#   * gpt-5, gpt-5.6, o4-mini + ``max_completion_tokens`` and no sampling
#     params -> 200
#   * Controls: gpt-4o and gpt-4.1 return 200 with ``temperature: 0.7`` +
#     ``max_tokens`` unchanged.
#
# The o1 family was not probed (no access on the project key) and is included
# on the documented grounds already encoded in the chat path's reasoning-model
# marker list (task-404): it shares the o-series request surface.
#
# Families are matched as (series, major) so dated snapshots
# (``gpt-5-2025-08-07``, ``o3-2025-04-16``), dotted minors (``gpt-5.6``,
# ``gpt-5.1``), suffixed variants (``gpt-5.6-terra``, ``o4-mini``) and
# provider-prefixed forms (``openai/gpt-5``) all resolve, without matching the
# legacy families that still accept both parameters (``gpt-4o``, ``gpt-4.1``,
# ``gpt-4-turbo``, ``gpt-3.5-turbo``) or non-OpenAI lookalikes
# (``o365-copilot``, ``olmo-7b``, ``gpt-oss-120b``).
_OPENAI_O_SERIES_RE = re.compile(r"^o(?P<major>\d)(?=$|[-_.@\[])")
_OPENAI_GPT_SERIES_RE = re.compile(r"^gpt[-_.](?P<major>\d+)(?=$|[-_.@\[])")

# (series, major) pairs whose chat-completions surface rejects the classic
# ``max_tokens`` cap and non-default sampling parameters.
_OPENAI_MODERN_REQUEST_FAMILIES = frozenset(
    {
        ("gpt", 5),
        ("o", 1),
        ("o", 3),
        ("o", 4),
    }
)


def _openai_model_family(model: object) -> Optional[Tuple[str, int]]:
    """Parse an OpenAI model id into ``(series, major)``.

    Args:
        model: A model identifier in any form the codebase passes through --
            bare (``gpt-5``, ``o3``), dotted (``gpt-5.6``), dated
            (``gpt-5-2025-08-07``, ``o3-2025-04-16``), suffixed
            (``gpt-5.6-terra``, ``o4-mini``) or provider-prefixed
            (``openai/gpt-5``).

    Returns:
        ``("gpt", major)`` or ``("o", major)``, or ``None`` when the id is not
        a recognisable OpenAI series name. The o-series major is a single
        digit and the gpt major must sit at a token boundary, so
        ``o365-copilot``, ``olmo-7b`` and ``gpt-4o`` never parse into a
        family.
    """
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    for pattern, series in (
        (_OPENAI_O_SERIES_RE, "o"),
        (_OPENAI_GPT_SERIES_RE, "gpt"),
    ):
        match = pattern.match(normalized)
        if match is not None:
            return (series, int(match.group("major")))
    return None


def _openai_is_modern_request_family(model: object) -> bool:
    """Return whether ``model`` is in the modern OpenAI request family."""
    family = _openai_model_family(model)
    if family is None:
        return False
    return family in _OPENAI_MODERN_REQUEST_FAMILIES


def openai_model_rejects_sampling_params(model: object) -> bool:
    """Return whether ``model`` rejects non-default ``temperature``/``top_p``.

    Args:
        model: An OpenAI model identifier (any prefixed or suffixed form).

    Returns:
        True when sending a non-default sampling value would be answered with
        ``400 unsupported_value: 'temperature' does not support 0.7 with this
        model. Only the default (1) value is supported.`` -- the o-series and
        gpt-5 reasoning families. False for gpt-4o, gpt-4.1 and earlier, which
        still accept them.
    """
    return _openai_is_modern_request_family(model)


def openai_model_requires_max_completion_tokens(model: object) -> bool:
    """Return whether ``model`` requires ``max_completion_tokens``.

    Args:
        model: An OpenAI model identifier (any prefixed or suffixed form).

    Returns:
        True when sending the classic ``max_tokens`` cap would be answered
        with ``400 unsupported_parameter: 'max_tokens' is not supported with
        this model. Use 'max_completion_tokens' instead.``

    Note:
        This currently covers the same families as
        :func:`openai_model_rejects_sampling_params` -- OpenAI changed both
        rules with the reasoning generation -- but they are separate questions
        about the request surface and are kept as separate predicates so a
        future model can answer them differently.
    """
    return _openai_is_modern_request_family(model)


#
#######################################################################################################################
#
# Moonshot (Kimi) per-model request capabilities
#
# Same design as the Anthropic (TASK-18414) and OpenAI (TASK-18802) predicates
# above: facts about what api.moonshot.ai *accepts*, kept as immutable
# module-level functions outside the config-driven tables, replacing the
# hand-maintained name checks the chat request builder used to carry
# (TASK-18803).
#
# Probe-verified against api.moonshot.ai on 2026-08-20 (TASK-18803) with the
# real project key. ``GET /v1/models`` served kimi-k2.5, kimi-k2.6,
# kimi-k2.7-code, kimi-k2.7-code-highspeed, kimi-k3, kimi-latest and the
# moonshot-v1 family. Acceptance:
#
#   * Versioned kimi (kimi-k3, kimi-k2.6, kimi-k2.5) + non-default sampling
#     -> 400 value-level rejections: ``invalid temperature: only 1 is allowed
#     for this model`` / ``invalid top_p: only 0.95 is allowed`` /
#     ``invalid presence_penalty: only 0 is allowed``; ``temperature: 1``
#     -> 200 (chatcmpl-6a872afa62f375d4129446c7).
#   * kimi-latest + the full five-parameter sampling set -> 200
#     (chatcmpl-6a872b9816ceb0c0ae780b1e; serves as kimi-latest-8k), and
#     moonshot-v1-8k likewise (chatcmpl-6a872ac1fe949ba3ecc8b094) -- neither
#     rejects sampling.
#   * ``reasoning_effort`` -> 200 on kimi-k3
#     (chatcmpl-6a872abcc8d3fc4c055ea030), kimi-k2.6
#     (chatcmpl-6a872abe6dd71293f91e1d59), kimi-k2.7-code
#     (chatcmpl-6a872b01a06896e50a1ab394) and kimi-latest
#     (chatcmpl-6a872ac016ceb0c0ae780b0c) -- the whole kimi series, not the
#     single literal ``kimi-k3`` the builder used to allow.
#
# Boundary-safe: ``kimi`` / ``kimi-k<major>`` must sit at a token boundary,
# so ``kimiko-7b`` never matches, and the legacy accepting family
# (``moonshot-v1-*``) is never parsed into the kimi series.
_MOONSHOT_KIMI_SERIES_RE = re.compile(r"^kimi(?=$|[-_.@\[])")
_MOONSHOT_KIMI_VERSIONED_RE = re.compile(r"^kimi[-_.]k(?P<major>\d+)(?=$|[-_.@\[])")
_MOONSHOT_LEGACY_V1_RE = re.compile(r"^moonshot[-_.]v1(?=$|[-_.@\[])")


def _moonshot_normalized_model(model: object) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized or None


def moonshot_model_supports_reasoning_effort(model: object) -> bool:
    """Return whether ``model`` accepts the ``reasoning_effort`` parameter.

    Args:
        model: A Moonshot model identifier (any prefixed or suffixed form).

    Returns:
        True for the whole kimi series -- versioned ids (``kimi-k3``,
        ``kimi-k2.6``, ``kimi-k3-turbo``) and unversioned aliases
        (``kimi-latest``) alike, all probe-verified to answer 200. False for
        the legacy ``moonshot-v1`` family and unrecognisable ids, which keep
        the historical client-side rejection.
    """
    normalized = _moonshot_normalized_model(model)
    if normalized is None:
        return False
    return _MOONSHOT_KIMI_SERIES_RE.match(normalized) is not None


def moonshot_model_rejects_sampling_params(model: object) -> bool:
    """Return whether ``model`` rejects non-default sampling parameters.

    Args:
        model: A Moonshot model identifier (any prefixed or suffixed form).

    Returns:
        True when sending a non-default ``temperature``/``top_p``/``n``/
        ``presence_penalty``/``frequency_penalty`` would be answered with a
        value-level 400 (``invalid temperature: only 1 is allowed for this
        model``) -- the versioned kimi reasoning family (``kimi-k<major>``,
        any suffix). False for ``kimi-latest`` and the ``moonshot-v1``
        family, which accept them (probe-verified), and for unrecognisable
        ids, which pass the caller's values through for the server to
        adjudicate.
    """
    normalized = _moonshot_normalized_model(model)
    if normalized is None:
        return False
    return _MOONSHOT_KIMI_VERSIONED_RE.match(normalized) is not None


def moonshot_model_requires_min_temperature_for_multiple_choices(
    model: object,
) -> bool:
    """Return whether ``model`` documents the n>1 minimum-temperature rule.

    The legacy ``moonshot-v1`` family documents that requesting multiple
    choices (``n > 1``) requires ``temperature >= 0.3``. This is a
    value-interplay constraint of that family, not a capability gate, but it
    is still a per-model fact and lives here so the request builder carries
    no model-name checks at all (TASK-18803).

    Args:
        model: A Moonshot model identifier (any prefixed or suffixed form).

    Returns:
        True for the ``moonshot-v1`` family only.
    """
    normalized = _moonshot_normalized_model(model)
    if normalized is None:
        return False
    return _MOONSHOT_LEGACY_V1_RE.match(normalized) is not None


def moonshot_model_returns_reasoning_content(model: object) -> bool:
    """Return whether ``model`` returns ``reasoning_content`` in responses.

    This is a RESPONSE-side fact -- which models emit private reasoning that
    the preserved-thinking checkpoint machinery should capture and replay --
    distinct from the request-side question of which models *accept* the
    ``reasoning_effort`` parameter (the whole kimi series, including
    ``kimi-latest``, per :func:`moonshot_model_supports_reasoning_effort`).

    Probe-verified against api.moonshot.ai on 2026-08-20 (TASK-19170) with
    the real project key:

    * Every versioned kimi id probed returns ``reasoning_content`` on every
      turn, with AND without ``reasoning_effort``: kimi-k2.5
      (chatcmpl-6a8768d3666d8454604d8b5f), kimi-k2.6
      (chatcmpl-6a8768a3b5c429b466fbc42d with effort,
      chatcmpl-6a8768a9b5c429b466fbc42f without), kimi-k2.7-code
      (chatcmpl-6a8768d705f910ba798aeca0), kimi-k3
      (chatcmpl-6a8768a7659da119063ca38f).
    * ``kimi-latest`` (served as ``kimi-latest-8k``) returns none
      (chatcmpl-6a8768a616ceb0c0ae780f2c) -- hence versioned-family, not
      whole-series.
    * Replaying the prior turn's ``reasoning_content`` is accepted and never
      required: multi-turn and tool-loop follow-ups answered 200 both with
      and without it (chatcmpl-6a8768cb.../6a8768cc... plain,
      chatcmpl-6a876916.../6a876918... tool loop), so widening k3-style
      preserved-thinking replay to the family cannot 400.

    Args:
        model: A Moonshot model identifier (any prefixed or suffixed form).

    Returns:
        True for the versioned kimi reasoning family (``kimi-k<major>``, any
        suffix). False for ``kimi-latest``, the legacy ``moonshot-v1``
        family and unrecognisable ids.
    """
    normalized = _moonshot_normalized_model(model)
    if normalized is None:
        return False
    return _MOONSHOT_KIMI_VERSIONED_RE.match(normalized) is not None


#
#######################################################################################################################
#
# Z.ai (GLM) per-model request capabilities
#
# Same design as above. No Z.ai key is available to this repo, so unlike the
# Anthropic/OpenAI/Moonshot predicates this one is NOT wire-verified
# (recorded in TASK-18803): it conservatively liberalises the builder's
# exact-id pin (``reasoning_effort`` only on the literal ``glm-5.2``), which
# client-side-rejected every other GLM release before a request was ever
# made. The floor is the version the pin already proved supported; newer
# releases in the family (``glm-5.3``, ``glm-6``, ``glm-5.2-air``) are no
# longer rejected on release day, and anything older or unrecognisable keeps
# the historical rejection.
_ZAI_GLM_FAMILY_RE = re.compile(
    r"^glm[-_.](?P<major>\d+)(?:\.(?P<minor>\d+))?(?=$|[-_.@\[])"
)

# The oldest (major, minor) known to accept ``reasoning_effort``.
_ZAI_REASONING_EFFORT_VERSION_FLOOR = (5, 2)


def _zai_glm_family(model: object) -> Optional[Tuple[int, int]]:
    """Parse a Z.ai model id into ``(major, minor)``.

    Args:
        model: A model identifier in any form the codebase passes through --
            bare (``glm-5.2``, ``glm-6``), suffixed (``glm-5.2-air``) or
            provider-prefixed (``zai/glm-5.2``).

    Returns:
        ``(major, minor)`` with a missing minor as 0, or ``None`` when the id
        is not a recognisable GLM family name. The version must sit at a
        token boundary, so e.g. ``glm-5x`` never parses.
    """
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    match = _ZAI_GLM_FAMILY_RE.match(normalized)
    if match is None:
        return None
    minor = match.group("minor")
    return (int(match.group("major")), int(minor) if minor is not None else 0)


def zai_model_supports_reasoning_effort(model: object) -> bool:
    """Return whether ``model`` accepts the ``reasoning_effort`` parameter.

    Args:
        model: A Z.ai model identifier (any prefixed or suffixed form).

    Returns:
        True for the GLM family at or above the 5.2 version floor
        (``glm-5.2``, ``glm-5.2-air``, ``glm-5.3``, ``glm-6``). False for
        older GLM releases and unrecognisable ids, which keep the historical
        client-side rejection.
    """
    family = _zai_glm_family(model)
    if family is None:
        return False
    return family >= _ZAI_REASONING_EFFORT_VERSION_FLOOR


#
#######################################################################################################################
#
# DeepSeek per-model completion-budget capability
#
# Same design as the Moonshot/Z.ai predicates above: a per-model fact about
# the provider's wire behavior, deliberately outside the config-driven
# tables. The DeepSeek handler (LLM_API_Calls.chat_with_deepseek) has no
# reasoning-effort parameter and passes ``max_tokens`` straight through as
# the whole completion budget, reasoning-INCLUSIVE -- so a reasoning-typed
# model handed a plain budget can spend all of it thinking and answer
# ``finish_reason=length`` with an EMPTY completion (TASK-21515: the
# config-default ``deepseek-v4-flash`` exhausted a 2000-token briefing
# budget exactly this way, while ``deepseek-chat`` completed the same
# prompt). Callers use this to widen the budget, not to change the prompt.
#
# Boundary-safe like the kimi regexes: ``deepseek-v4`` must sit at a token
# boundary, so ``deepseek-v40`` never matches, and the prefixed OpenRouter
# form (``deepseek/deepseek-v4-flash``) normalizes to its bare id -- the
# same idiom ``_moonshot_normalized_model``/``_zai_glm_family`` use.
_DEEPSEEK_REASONING_FAMILY_RE = re.compile(
    r"^deepseek[-_.](?:reasoner|v4)(?=$|[-_.@\[])"
)


def _deepseek_normalized_model(model: object) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized or None


def deepseek_model_thinks_by_default(model: object) -> bool:
    """Whether a DeepSeek model spends completion tokens on reasoning by default.

    The DeepSeek handler has no reasoning-effort parameter and its
    ``max_tokens`` budget is reasoning-inclusive, so these models need a
    larger completion budget for the same output length (TASK-21515:
    deepseek-v4-flash exhausted ``BRIEFING_MAX_TOKENS`` on reasoning and
    returned an empty completion).

    Args:
        model: A DeepSeek model identifier (any prefixed or suffixed form).

    Returns:
        True for the reasoning-typed families -- ``deepseek-reasoner`` and
        the ``deepseek-v4`` generation (``deepseek-v4-flash``,
        ``deepseek-v4-pro``, this catalog's DeepSeek defaults). False for
        ``deepseek-chat``, which completes within a plain budget
        (live-verified during the TASK-21515 incident), and for
        unrecognisable ids.
    """
    normalized = _deepseek_normalized_model(model)
    if normalized is None:
        return False
    return _DEEPSEEK_REASONING_FAMILY_RE.match(normalized) is not None


#: The config-default chain's terminal fallback -- identical to the literal
#: ``chat_with_deepseek`` resolves when neither the caller nor
#: ``[api_settings.deepseek]`` names a model, so the resolver below and the
#: handler can never disagree about which model a ``model=None`` call runs.
_DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


def resolve_deepseek_effective_model(model: Optional[str]) -> Optional[str]:
    """Resolve the model a DeepSeek call will actually run when none is given.

    Qodo #7/#8 (TASK-21515 follow-up): the budget gates call this BEFORE
    consulting :func:`deepseek_model_thinks_by_default`, because a caller
    that resolved no model does not get "no model" -- ``chat_with_deepseek``
    picks its own default, and that default (``deepseek-v4-flash`` unless
    configured otherwise) is reasoning-typed. A gate that tested the literal
    ``None`` kept the plain budget for exactly the model that needed the
    widened one.

    Mirrors the handler's own lookup (``LLM_API_Calls.chat_with_deepseek``):
    ``[api_settings.deepseek].model`` from the runtime config snapshot,
    falling back to the same terminal default. The config import is lazy
    (function-level) for the same circular-import reason
    ``ModelCapabilities.__init__`` imports its config loader lazily --
    ``briefing_service.default_briefing_provider`` is the established
    read-through-at-call-time idiom this follows, which also lets tests
    monkeypatch the seam and be observed on the very next call.

    Args:
        model: The caller-supplied model id, or ``None``/empty for "let the
            DeepSeek handler pick its own default".

    Returns:
        ``model`` unchanged whenever it is a non-empty string; otherwise the
        configured DeepSeek default model, or ``None`` when the configured
        value is itself blank (the predicate then correctly answers False --
        a blank model is never a reasoning family).
    """
    if isinstance(model, str) and model.strip():
        return model
    # Late import: `tldw_chatbook.config` pulls in most of the app, and this
    # module is imported far earlier in the boot sequence than it.
    from tldw_chatbook.config import get_runtime_config_snapshot

    try:
        api_settings = get_runtime_config_snapshot().values.get(
            "api_settings", {}
        )
        deepseek_config = api_settings.get("deepseek", {})
        resolved = deepseek_config.get("model", _DEEPSEEK_DEFAULT_MODEL)
    except Exception:  # noqa: BLE001 - a budget gate must never crash a call
        logger.warning(
            "DeepSeek default model resolution failed; assuming the "
            "documented default."
        )
        return _DEEPSEEK_DEFAULT_MODEL
    if isinstance(resolved, str) and resolved.strip():
        return resolved
    return None


#
#######################################################################################################################
#
# ModelCapabilities Class
#
def _models_dev_capabilities(provider: str, model: str) -> Dict[str, Any]:
    """TASK-26023: gap-fill capabilities from models.dev, or an empty dict.

    Origin-inspectable via the ``source`` key (AC#5). Returns {} when the
    model is unknown upstream too, so the caller keeps its honest-default
    behavior (AC#6).
    """
    try:
        from tldw_chatbook.LLM_Provider_Catalog.models_dev_catalog import (
            models_dev_entry,
        )

        entry = models_dev_entry(provider, model)
    except Exception:  # noqa: BLE001 -- the gap-fill never breaks a lookup
        return {}
    if entry is None:
        return {}
    caps: Dict[str, Any] = {"vision": entry.supports_vision, "source": "models.dev"}
    if entry.context_window is not None:
        caps["context_window"] = entry.context_window
    return caps


class ModelCapabilities:
    """
    Manages model capability detection based on configuration.

    Supports:
    - Direct model name to capability mapping
    - Pattern-based matching for model families
    - Provider-specific patterns
    - Default fallbacks
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration.

        Args:
            config: Model capabilities configuration dict. If None, loads from config file.
        """
        if config is None:
            # Load from config file
            # Get model_capabilities from config - it's a top-level section
            from tldw_chatbook.config import load_cli_config_and_ensure_existence

            full_config = load_cli_config_and_ensure_existence()
            config = full_config.get("model_capabilities", {})

        # Direct model mappings (highest priority)
        self.direct_mappings = config.get("models", DEFAULT_MODEL_CAPABILITIES.copy())

        # Pattern configurations by provider
        self.pattern_configs = config.get("patterns", DEFAULT_MODEL_PATTERNS.copy())

        # Default settings
        self.defaults = config.get(
            "defaults", {"unknown_models_vision": False, "log_unknown_models": True}
        )

        # Compile patterns for efficiency
        self._compiled_patterns = self._compile_patterns()
        # Case-insensitive provider index: callers pass mixed/lowercase provider
        # names ("openai") while pattern keys are title-case ("OpenAI").
        self._provider_key_by_lower = {
            provider.lower(): provider for provider in self._compiled_patterns
        }

        # Cache for resolved capabilities
        self._capability_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        logger.debug(
            f"ModelCapabilities initialized with {len(self.direct_mappings)} direct mappings and patterns for {len(self.pattern_configs)} providers"
        )

    def _compile_patterns(self) -> Dict[str, List[Tuple[Pattern, Dict[str, Any]]]]:
        """Compile regex patterns for each provider."""
        compiled = {}

        for provider, patterns in self.pattern_configs.items():
            compiled_list = []
            for pattern_config in patterns:
                if isinstance(pattern_config, dict) and "pattern" in pattern_config:
                    try:
                        pattern = re.compile(pattern_config["pattern"], re.IGNORECASE)
                        # Extract capabilities from pattern config
                        capabilities = {
                            k: v for k, v in pattern_config.items() if k != "pattern"
                        }
                        compiled_list.append((pattern, capabilities))
                    except re.error as e:
                        logger.error(
                            f"Invalid regex pattern for {provider}: {pattern_config['pattern']} - {e}"
                        )

            if compiled_list:
                compiled[provider] = compiled_list
                logger.debug(
                    f"Compiled {len(compiled_list)} patterns for provider {provider}"
                )

        return compiled

    @lru_cache(maxsize=128)
    def is_vision_capable(self, provider: str, model: str) -> bool:
        """
        Check if a model supports vision/image input.

        Args:
            provider: The provider name (e.g., "OpenAI", "Anthropic")
            model: The model identifier

        Returns:
            True if the model supports vision input, False otherwise
        """
        capabilities = self.get_model_capabilities(provider, model)
        return capabilities.get("vision", False)

    def get_context_window(self, provider: str, model: str) -> Optional[int]:
        """Return the model's input context window.

        Args:
            provider: The provider name (case-insensitive).
            model: The model identifier.

        Returns:
            The input context window in tokens, or ``None`` if unknown.
        """
        return self.get_model_capabilities(provider, model).get("context_window")

    def get_model_capabilities(self, provider: str, model: str) -> Dict[str, Any]:
        """
        Get all capabilities for a model.

        Args:
            provider: The provider name
            model: The model identifier

        Returns:
            Dictionary of capabilities (e.g., {"vision": True, "max_images": 10})
        """
        cache_key = (provider, model)

        # Check cache first
        if cache_key in self._capability_cache:
            return self._capability_cache[cache_key]

        capabilities = {}

        # 1. Check direct mapping (highest priority)
        if model in self.direct_mappings:
            capabilities = self.direct_mappings[model].copy()
            logger.debug(f"Found direct mapping for {model}: {capabilities}")

        # 2. Check provider-specific patterns (case-insensitive provider match)
        else:
            provider_key = self._provider_key_by_lower.get((provider or "").lower())
            if provider_key is not None:
                for pattern, pattern_capabilities in self._compiled_patterns[provider_key]:
                    if pattern.match(model):
                        capabilities = pattern_capabilities.copy()
                        logger.debug(
                            f"Pattern matched for {provider}/{model}: {capabilities}"
                        )
                        break

        # 3. TASK-26023: upstream models.dev gap-fill, BENEATH the hand-
        # maintained direct/pattern entries above (AC#2). Disabled by
        # default and network-free.
        if not capabilities:
            upstream = _models_dev_capabilities(provider, model)
            if upstream:
                capabilities = upstream

        # 4. If still no match found, use defaults
        if not capabilities:
            if self.defaults.get("log_unknown_models", True):
                logger.info(
                    f"No capability information found for {provider}/{model}, using defaults"
                )
            capabilities = {"vision": self.defaults.get("unknown_models_vision", False)}

        # Cache the result
        self._capability_cache[cache_key] = capabilities

        return capabilities

    def add_model_capability(self, model: str, capabilities: Dict[str, Any]):
        """
        Add or update capabilities for a specific model.

        Args:
            model: The model identifier
            capabilities: Dictionary of capabilities
        """
        self.direct_mappings[model] = capabilities
        # Clear cache entry if it exists
        for key in list(self._capability_cache.keys()):
            if key[1] == model:
                del self._capability_cache[key]

    def list_vision_models(self, provider: Optional[str] = None) -> List[str]:
        """
        List all known vision-capable models.

        Args:
            provider: Optional provider filter

        Returns:
            List of model names that support vision
        """
        vision_models = []

        # Add direct mappings
        for model, caps in self.direct_mappings.items():
            if caps.get("vision", False):
                vision_models.append(model)

        # Note: Pattern-based models can't be listed without knowing all possible model names

        return sorted(vision_models)

    def clear_cache(self):
        """Clear the capability cache."""
        self._capability_cache.clear()
        self.is_vision_capable.cache_clear()


#
#######################################################################################################################
#
# Module-level convenience functions
#

# Global instance (lazy-loaded)
_global_capabilities: Optional[ModelCapabilities] = None


def get_model_capabilities() -> ModelCapabilities:
    """
    Get the global ModelCapabilities instance.

    Returns:
        ModelCapabilities instance configured from user settings
    """
    global _global_capabilities
    if _global_capabilities is None:
        _global_capabilities = ModelCapabilities()
    return _global_capabilities


def is_vision_capable(provider: str, model: str) -> bool:
    """
    Convenience function to check if a model supports vision.

    Args:
        provider: The provider name
        model: The model identifier

    Returns:
        True if the model supports vision input
    """
    return get_model_capabilities().is_vision_capable(provider, model)


def get_context_window(provider: str, model: str) -> Optional[int]:
    """Resolve a model's input context window from the global capabilities.

    Args:
        provider: The provider name (case-insensitive).
        model: The model identifier.

    Returns:
        The input context window in tokens, or ``None`` if unknown.
    """
    return get_model_capabilities().get_context_window(provider, model)


def reload_capabilities():
    """Reload model capabilities from configuration."""
    global _global_capabilities
    _global_capabilities = None
    logger.info("Model capabilities reloaded from configuration")


#
# End of model_capabilities.py
#######################################################################################################################
