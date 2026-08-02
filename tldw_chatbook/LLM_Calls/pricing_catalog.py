# tldw_chatbook/LLM_Calls/pricing_catalog.py
# Description: Per-model pricing (dollars per million tokens) for the Console cost ticker.
#
"""Per-model pricing (dollars per million tokens) for the cost ticker.

Resolution order: [pricing].models config override -> seeded direct map ->
[pricing].patterns config override -> seeded pattern fallback -> local
provider zero-rate -> None. None means "no pricing data" and the UI must
show token counts instead of fabricating a dollar figure.

Rates were verified against each provider's official pricing page on
_SEED_AS_OF (see task-2-report.md for the full source list); a few entries
are marked UNVERIFIED below where the official page did not publish a
per-token rate for that model.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

from tldw_chatbook.Chat.provider_usage import ProviderUsage

logger = logging.getLogger(__name__)

#
#######################################################################################################################
#
# Seed data
#
# Date the seed table below was last checked against official provider pricing pages.
_SEED_AS_OF = "2026-08-01"

# Providers that run locally: always $0.00. Cross-checked against the exact
# provider strings dispatched in tldw_chatbook/Chat/Chat_Functions.py
# (API_CALL_HANDLERS keys) plus the local-runtime sections that exist in
# config.py's [api_settings] but have no handler wired up yet.
LOCAL_PROVIDERS = frozenset(
    {
        "llama_cpp",
        "koboldcpp",
        "oobabooga",
        "tabbyapi",
        "vllm",
        "local-llm",
        "ollama",
        "aphrodite",
        "mlx_lm",
        "local_llamacpp",
        "local_llamafile",
        "local_ollama",
        "local_vllm",
        "local_mlx_lm",
        "local_onnx",
        "local_transformers",
    }
)

_ZERO = {
    "input_per_mtok": 0.0,
    "output_per_mtok": 0.0,
    "cache_read_per_mtok": 0.0,
    "cache_write_per_mtok": 0.0,
    "as_of": _SEED_AS_OF,
}


def _entry(inp: float, out: float, cr: Optional[float] = None, cw: Optional[float] = None,
           as_of: str = _SEED_AS_OF) -> Dict[str, Any]:
    return {
        "input_per_mtok": inp,
        "output_per_mtok": out,
        "cache_read_per_mtok": cr,
        "cache_write_per_mtok": cw,
        "as_of": as_of,
    }


def _lower_keys(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase every "provider:model" key so lookups are case-insensitive.

    get_pricing() always queries with a lowercased "provider:model" key; both
    the seed table and any config-supplied `[pricing].models` overrides must
    go through this before landing in `direct_mappings`, or a naturally-cased
    override key would silently never match.
    """
    return {str(key).lower(): value for key, value in mapping.items()}


# Anthropic: cache read = 0.1x input, cache write = 1.25x input (5-min TTL).
# Verified 2026-08-01 (pre-verified rates supplied with the task brief).
DEFAULT_MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "anthropic:claude-opus-4-1": _entry(15.00, 75.00, 1.50, 18.75),
    "anthropic:claude-sonnet-4-6": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-sonnet-4-5": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-haiku-4-5": _entry(1.00, 5.00, 0.10, 1.25),

    # OpenAI - verified 2026-08-01 via https://developers.openai.com/api/docs/pricing
    # (platform.openai.com/docs/pricing now redirects there). "Cached input" column
    # maps to cache_read_per_mtok; OpenAI has no cache-write concept (cache_write=None).
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
    "openai:o3": _entry(2.00, 8.00, 0.50, None),
    "openai:o3-mini": _entry(1.10, 4.40, 0.55, None),
    "openai:o4-mini": _entry(1.10, 4.40, 0.275, None),

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

DEFAULT_PRICING_PATTERNS: Dict[str, List[Dict[str, Any]]] = {
    "anthropic": [
        {"pattern": r"^claude-opus", **_entry(15.00, 75.00, 1.50, 18.75)},
        {"pattern": r"^claude-sonnet", **_entry(3.00, 15.00, 0.30, 3.75)},
        {"pattern": r"^claude-haiku", **_entry(1.00, 5.00, 0.10, 1.25)},
    ],
    "openai": [
        # Longer/more specific prefixes must precede their shorter siblings.
        {"pattern": r"^gpt-5\.2", **_entry(1.75, 14.00, 0.175, None)},
        {"pattern": r"^gpt-5\.1", **_entry(1.25, 10.00, 0.125, None)},
        {"pattern": r"^gpt-5-nano", **_entry(0.05, 0.40, 0.005, None)},
        {"pattern": r"^gpt-5-mini", **_entry(0.25, 2.00, 0.025, None)},
        {"pattern": r"^gpt-5", **_entry(1.25, 10.00, 0.125, None)},
        {"pattern": r"^gpt-4\.1-nano", **_entry(0.10, 0.40, 0.025, None)},
        {"pattern": r"^gpt-4\.1-mini", **_entry(0.40, 1.60, 0.10, None)},
        {"pattern": r"^gpt-4\.1", **_entry(2.00, 8.00, 0.50, None)},
        {"pattern": r"^gpt-4o-mini", **_entry(0.15, 0.60, 0.075, None)},
        {"pattern": r"^gpt-4o", **_entry(2.50, 10.00, 1.25, None)},
        {"pattern": r"^o3-mini", **_entry(1.10, 4.40, 0.55, None)},
        {"pattern": r"^o3", **_entry(2.00, 8.00, 0.50, None)},
        {"pattern": r"^o4-mini", **_entry(1.10, 4.40, 0.275, None)},
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
@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: Optional[float]
    cache_write_per_mtok: Optional[float]
    as_of: str


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    input_cost: float
    cache_read_cost: float
    cache_write_cost: float
    output_cost: float
    total: float
    as_of: str


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
        # key present in config replaces that provider's whole pattern list;
        # providers absent from config keep their seeded patterns).
        self.pattern_configs: Dict[str, List[Dict[str, Any]]] = {
            **DEFAULT_PRICING_PATTERNS,
            **config.get("patterns", {}),
        }

        # Compile patterns for efficiency
        self._compiled_patterns = self._compile_patterns()
        # Case-insensitive provider index: callers pass mixed/lowercase provider
        # names while pattern keys are whatever case the seed/config used.
        self._provider_key_by_lower = {
            provider.lower(): provider for provider in self._compiled_patterns
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
        )

    def get_pricing(self, provider: str, model: str) -> Optional[ModelPricing]:
        """
        Resolve pricing for a provider/model pair.

        Resolution order: direct "provider:model" mapping -> provider pattern
        fallback (first match wins) -> local-provider zero-rate -> None.

        Args:
            provider: The provider name (case-insensitive).
            model: The model identifier (case-insensitive).

        Returns:
            A ModelPricing instance, or None if no pricing data is available.
        """
        provider_l = (provider or "").lower()
        model_l = (model or "").lower()

        # 1. Direct mapping (highest priority)
        entry = self.direct_mappings.get(f"{provider_l}:{model_l}")
        if entry is not None:
            return self._to_model_pricing(entry)

        # 2. Provider-specific patterns (case-insensitive provider match)
        provider_key = self._provider_key_by_lower.get(provider_l)
        if provider_key is not None:
            for pattern, pattern_entry in self._compiled_patterns[provider_key]:
                if pattern.match(model_l):
                    return self._to_model_pricing(pattern_entry)

        # 3. Local providers always cost $0.00
        if provider_l in LOCAL_PROVIDERS:
            return self._zero_pricing

        # 4. Unknown model: no fabricated price, let the UI fall back to token counts.
        return None

    def cost_for_usage(self, usage: ProviderUsage) -> Optional[CostBreakdown]:
        """
        Compute a dollar cost breakdown for a ProviderUsage's disjoint token buckets.

        Args:
            usage: Normalized per-message token usage.

        Returns:
            A CostBreakdown, or None if pricing for usage.provider/usage.model is unknown.
        """
        pricing = self.get_pricing(usage.provider, usage.model)
        if pricing is None:
            return None

        input_cost = round(usage.uncached_input * pricing.input_per_mtok / 1_000_000, 6)
        cache_read_cost = round(
            usage.cache_read * (pricing.cache_read_per_mtok or 0.0) / 1_000_000, 6
        )
        cache_write_cost = round(
            usage.cache_write * (pricing.cache_write_per_mtok or 0.0) / 1_000_000, 6
        )
        output_cost = round(usage.output * pricing.output_per_mtok / 1_000_000, 6)
        total = round(input_cost + cache_read_cost + cache_write_cost + output_cost, 6)

        return CostBreakdown(
            input_cost=input_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            output_cost=output_cost,
            total=total,
            as_of=pricing.as_of,
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
