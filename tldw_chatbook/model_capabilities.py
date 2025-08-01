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
from tldw_chatbook.config import get_cli_setting

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
        {"pattern": r"^gpt-4.*vision", "vision": True},
        {"pattern": r"^gpt-4[o0](?:-mini)?", "vision": True},  # gpt-4o, gpt-40, gpt-4o-mini
        {"pattern": r"^gpt-4.*turbo", "vision": True},
        {"pattern": r"^gpt-4\.1", "vision": True},  # gpt-4.1 series
        {"pattern": r"^o[34](?:-mini)?", "vision": True},  # o3, o4, o3-mini, o4-mini series
        {"pattern": r"^dall-e", "vision": True, "image_generation": True}
    ],
    "Anthropic": [
        {"pattern": r"^claude-3", "vision": True},  # All Claude 3 models have vision
        {"pattern": r"^claude.*opus-4", "vision": True},  # Claude Opus 4 series
        {"pattern": r"^claude.*sonnet-4", "vision": True}  # Claude Sonnet 4 series
    ],
    "Google": [
        {"pattern": r"gemini.*vision", "vision": True},
        {"pattern": r"gemini-[0-9.]+-(pro|flash)", "vision": True},  # Modern Gemini models
        {"pattern": r"gemini-2\.", "vision": True}  # Gemini 2.x series
    ],
    "OpenRouter": [
        # OpenRouter uses provider/model format
        {"pattern": r"openai/gpt-4.*vision", "vision": True},
        {"pattern": r"openai/gpt-4[o0]", "vision": True},
        {"pattern": r"openai/gpt-4\.1", "vision": True},
        {"pattern": r"openai/o[34](?:-mini)?", "vision": True},
        {"pattern": r"anthropic/claude-3", "vision": True},
        {"pattern": r"google/gemini.*vision", "vision": True},
        {"pattern": r"google/gemini-[0-9.]+-(pro|flash)", "vision": True}
    ],
    "Moonshot": [
        # Moonshot vision models
        {"pattern": r"moonshot-v1-.*-vision-preview", "vision": True},  # Matches all vision preview models
        {"pattern": r"moonshot-v1-8k-vision-preview", "vision": True},
        {"pattern": r"moonshot-v1-32k-vision-preview", "vision": True},
        {"pattern": r"moonshot-v1-128k-vision-preview", "vision": True}
    ],
    "ZAI": [
        # Z.AI models - currently no vision support
        {"pattern": r"^glm-", "vision": False}  # All GLM models currently don't support vision
    ]
}

# Known models with direct capabilities (for common models)
DEFAULT_MODEL_CAPABILITIES = {
    # OpenAI
    "gpt-4-vision-preview": {"vision": True, "max_images": 1},
    "gpt-4-turbo": {"vision": True, "max_images": 10},
    "gpt-4-turbo-2024-04-09": {"vision": True, "max_images": 10},
    "gpt-4o": {"vision": True, "max_images": 10},
    "gpt-4o-mini": {"vision": True, "max_images": 10},
    "gpt-4.1-2025-04-14": {"vision": True, "max_images": 10},
    "o4-mini-2025-04-16": {"vision": True, "max_images": 10},
    "o3-2025-04-16": {"vision": True, "max_images": 10},
    "o3-mini-2025-01-31": {"vision": True, "max_images": 10},
    "gpt-4.1-mini-2025-04-14": {"vision": True, "max_images": 10},
    "gpt-4.1-nano-2025-04-14": {"vision": True, "max_images": 10},

    # Anthropic
    "claude-3-opus-20240229": {"vision": True, "max_images": 5},
    "claude-3-sonnet-20240229": {"vision": True, "max_images": 5},
    "claude-3-haiku-20240307": {"vision": True, "max_images": 5},
    "claude-3-5-sonnet-20240620": {"vision": True, "max_images": 5},
    "claude-3-5-sonnet-20241022": {"vision": True, "max_images": 5},

    # Google
    "gemini-pro-vision": {"vision": True, "max_images": 1},
    "gemini-1.5-pro": {"vision": True, "max_images": 10},
    "gemini-1.5-flash": {"vision": True, "max_images": 10},
    "gemini-2.0-flash": {"vision": True, "max_images": 10},
    
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
    "glm-4-32b-0414-128k": {"vision": False, "max_tokens": 128000}
}


#
#######################################################################################################################
#
# ModelCapabilities Class
#
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
        self.defaults = config.get("defaults", {
            "unknown_models_vision": False,
            "log_unknown_models": True
        })

        # Compile patterns for efficiency
        self._compiled_patterns = self._compile_patterns()

        # Cache for resolved capabilities
        self._capability_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        logger.debug(
            f"ModelCapabilities initialized with {len(self.direct_mappings)} direct mappings and patterns for {len(self.pattern_configs)} providers")

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
                        capabilities = {k: v for k, v in pattern_config.items() if k != "pattern"}
                        compiled_list.append((pattern, capabilities))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern for {provider}: {pattern_config['pattern']} - {e}")

            if compiled_list:
                compiled[provider] = compiled_list
                logger.debug(f"Compiled {len(compiled_list)} patterns for provider {provider}")

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

        # 2. Check provider-specific patterns
        elif provider in self._compiled_patterns:
            for pattern, pattern_capabilities in self._compiled_patterns[provider]:
                if pattern.match(model):
                    capabilities = pattern_capabilities.copy()
                    logger.debug(f"Pattern matched for {provider}/{model}: {capabilities}")
                    break

        # 3. If no match found, use defaults
        if not capabilities:
            if self.defaults.get("log_unknown_models", True):
                logger.info(f"No capability information found for {provider}/{model}, using defaults")
            capabilities = {
                "vision": self.defaults.get("unknown_models_vision", False)
            }

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


def reload_capabilities():
    """Reload model capabilities from configuration."""
    global _global_capabilities
    _global_capabilities = None
    logger.info("Model capabilities reloaded from configuration")

#
# End of model_capabilities.py
#######################################################################################################################
